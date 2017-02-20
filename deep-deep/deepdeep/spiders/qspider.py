# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, List, Iterator, Set
import abc
import time
import gzip
import logging

import psutil
import tqdm
import joblib
import numpy as np
import scipy.sparse as sp
import networkx as nx
import scrapy
from scrapy.http import TextResponse, Response
from scrapy.statscollectors import StatsCollector
from scrapy_cdr.utils import text_cdr_item
import tensorboard_logger

from deepdeep.queues import (
    BalancedPriorityQueue,
    RequestsPriorityQueue,
    score_to_priority,
    priority_to_score, FLOAT_PRIORITY_MULTIPLIER)
from deepdeep.scheduler import Scheduler
from deepdeep.spiders._base import BaseSpider
from deepdeep.qlearning import QLearner
from deepdeep.utils import set_request_domain, get_domain, log_time, chunks
from deepdeep.vectorizers import LinkVectorizer, PageVectorizer
from deepdeep.goals import BaseGoal
from deepdeep.metrics import ndcg_score


class QSpider(BaseSpider, metaclass=abc.ABCMeta):
    """
    This spider learns how to crawl using Q-Learning.

    Subclasses must override :meth:`get_goal` method to define the reward.

    It starts from a list of seed URLs. When a page is received, spider

    1. updates Q function based on observed reward;
    2. extracts links and creates requests for them, using Q function
       to set priorities

    """
    _ARGS = {
        'double', 'use_urls', 'use_full_urls', 'use_same_domain',
        'use_link_text', 'use_page_urls', 'use_full_page_urls',
        'use_pages', 'page_vectorizer_path',
        'eps', 'balancing_temperature', 'gamma',
        'clf_alpha', 'clf_penalty',
        'replay_sample_size', 'replay_maxsize', 'replay_maxlinks',
        'domain_queue_maxsize', 'steps_before_switch',
        'checkpoint_path', 'checkpoint_interval', 'checkpoint_latest',
        'baseline', 'export_cdr',
    }
    ALLOWED_ARGUMENTS = _ARGS | BaseSpider.ALLOWED_ARGUMENTS
    custom_settings = {
        # 'DEPTH_LIMIT': 100,
        'DEPTH_PRIORITY': 1,
    }
    initial_priority = score_to_priority(5)

    # whether to export page data as CDR items
    export_cdr = 1

    # whether to use URL path/query or a full URL as a feature
    use_urls = 0
    use_full_urls = 0

    # whether to use link text feature
    use_link_text = 1

    # whether to use page URL path/query or a full page URL as a feature
    use_page_urls = 0
    use_full_page_urls = 0

    # whether to use a 'link is to the same domain' feature
    use_same_domain = 1

    # whether to use page content as a feature
    use_pages = 0

    # Link classifier hyper-parameters
    clf_penalty = 'l2'
    clf_alpha = 1e-6

    # path to a saved page vectorizer model
    page_vectorizer_path = None  # type: str

    # use Double Learning
    double = 1

    # probability of selecting a random request
    eps = 0.2

    # 0 <= gamma < 1; lower values make spider focus on immediate reward.
    gamma = 0.4

    # softmax temperature for domain balancer;
    # higher values => more randomeness in domain selection.
    balancing_temperature = 1.0

    # parameters of online Q function are copied to target Q function
    # every `steps_before_switch` steps
    steps_before_switch = 100

    # how many examples to fetch from experience replay on each iteration
    replay_sample_size = 300

    # Max size of experience replay memory.
    # When all features are enabled (use_pages, use_full_urls)
    # a single observation uses about 1MB memory on average, so
    # replay_maxsize=10000 *roughly* means 10GB experience replay memory limit.
    #
    # With use_full_url=1 and use_pages=0 a single observation uses
    # about 10Kb on average.
    replay_maxsize = 100000

    # Maximum number of links: useful to limit when running separate spiders
    # for each domain. No limit by default.
    replay_maxlinks = 0

    domain_queue_maxsize = 0  # no limit by default

    # current model is saved every checkpoint_interval timesteps
    checkpoint_interval = 1000

    # Where to store checkpoints. By default they are not stored.
    checkpoint_path = None  # type: Optional[str]

    # Store only latest checkpoint to save disk space.
    checkpoint_latest = 0

    # Is spider allowed to follow out-of-domain links?
    # XXX: it is not enough to set this to False; a middleware should be also
    # turned off.
    stay_in_domain = True

    # use baseline algorithm (BFS) instead of Q-Learning
    baseline = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.export_cdr = int(self.export_cdr)
        self.eps = float(self.eps)
        self.balancing_temperature = float(self.balancing_temperature)
        self.gamma = float(self.gamma)
        self.use_urls = bool(int(self.use_urls))
        self.use_full_urls = bool(int(self.use_full_urls))
        self.use_same_domain = int(self.use_same_domain)
        self.use_link_text = bool(int(self.use_link_text))
        self.use_page_urls = bool(int(self.use_page_urls))
        self.use_full_page_urls = bool(int(self.use_full_page_urls))
        self.double = int(self.double)
        self.stay_in_domain = bool(int(self.stay_in_domain))
        self.steps_before_switch = int(self.steps_before_switch)
        self.replay_sample_size = int(self.replay_sample_size)
        self.replay_maxsize = int(self.replay_maxsize)
        self.replay_maxlinks = int(self.replay_maxlinks)
        self.clf_penalty = str(self.clf_penalty)
        self.clf_alpha = float(self.clf_alpha)
        self.domain_queue_maxsize = int(self.domain_queue_maxsize)
        self.baseline = bool(int(self.baseline))
        self.Q = QLearner(
            steps_before_switch=self.steps_before_switch,
            replay_sample_size=self.replay_sample_size,
            gamma=self.gamma,
            double_learning=bool(self.double),
            on_model_changed=self.on_model_changed,
            pickle_memory=False,
            dummy=self.baseline,
            er_maxsize=self.replay_maxsize,
            er_maxlinks=self.replay_maxlinks,
            clf_alpha=self.clf_alpha,
            clf_penalty=self.clf_penalty,
        )
        self.link_vectorizer = LinkVectorizer(
            use_url=bool(self.use_urls),
            use_full_url=bool(self.use_full_urls),
            use_same_domain=bool(self.use_same_domain),
            use_link_text=bool(self.use_link_text),
            use_page_url=bool(self.use_page_urls),
            use_full_page_url=bool(self.use_full_page_urls),
        )
        if self.page_vectorizer_path:
            self.use_pages = True
            self.page_vectorizer = joblib.load(self.page_vectorizer_path)
            self.page_vectorizer.steps[-1][1].verbose = False
        else:
            self.use_pages = int(self.use_pages)
            self.page_vectorizer = PageVectorizer() if self.use_pages else None

        self.total_reward = 0
        self.rewards = []
        self.steps_before_reschedule = 0
        self.goal = self.get_goal()

        self.crawled_domains = set()  # type: Set[str]
        self.relevant_domains = set()  # type: Set[str]

        self.checkpoint_interval = int(self.checkpoint_interval)
        self.checkpoint_latest = bool(int(self.checkpoint_latest))
        self._save_params_json()
        self._setup_tensorboard_logger()

    def _save_params_json(self):
        if self.checkpoint_path:
            params = json.dumps(self.get_params(), indent=4)
            logging.info(params)
            (Path(self.checkpoint_path)/"params.json").write_text(params)

    def _setup_tensorboard_logger(self):
        if self.checkpoint_path:
            self._tensortboard_logger = tensorboard_logger.Logger(
                self.checkpoint_path, flush_secs=5)
        else:
            self._tensortboard_logger = None

    def log_value(self, name, value):
        if self._tensortboard_logger and self.Q.t_ % 20 == 0:
            self._tensortboard_logger.log_value(name, value, step=self.Q.t_)

    @abc.abstractmethod
    def get_goal(self) -> BaseGoal:
        """ This method should return a crawl goal object """
        pass

    def is_seed(self, r: Union[scrapy.Request, Response]) -> bool:
        return 'link_vector' not in r.meta

    def update_node(self, response: Response, data: Dict) -> None:
        """ Store extra information in crawl graph node """
        if not hasattr(self, 'G'):
            return
        node = self.G.node[response.meta['node_id']]
        node['t'] = self.Q.t_
        node.update(data)

    def parse(self, response: Response):
        self.increase_response_count()
        self.close_finished_queues()
        if not self.is_seed(response):
            self.steps_before_reschedule -= 1
        self._debug_expected_vs_got(response)
        output, reward = self._parse(response)
        self.log_stats()

        if not self.is_seed(response):
            # timestep is not increased for seed urls, so
            # making checkpoint for them can lead to duplicate work
            self.maybe_checkpoint()

        stats = self.get_stats_item()
        stats['ts'] = time.time()
        stats['is_seed'] = self.is_seed(response)
        stats['rss'] = psutil.Process().memory_info().rss
        stats['reward'] = reward
        stats['url'] = response.url
        stats['Q'] = priority_to_score(response.request.priority)
        stats['eps-policy'] = response.request.meta.get('from_random_policy', None)

        if self.export_cdr:
            cdr_item = text_cdr_item(
                response,
                crawler_name='deep-deep',
                team_name='HG',
                metadata={
                    'depth': response.meta.get('depth'),
                    'stats': stats
                }
            )
            yield cdr_item
        else:
            yield stats

        yield from output

    @log_time
    def _parse(self, response):
        if self.is_seed(response) and not hasattr(response, 'text'):
            # bad seed
            return [], 0

        as_t = response.meta.get('link_vector')

        if not hasattr(response, 'text'):
            # learn to avoid non-html responses
            self.Q.add_experience(
                as_t=as_t,
                AS_t1=None,
                r_t1=0
            )
            self.update_node(response, {'reward': 0})
            return [], 0

        page_vector = self._page_vector(response) if self.use_pages else None
        links = self._extract_links(response)
        links_matrix = self.link_vectorizer.transform(links) if links else None
        links_matrix = self.Q.join_As(links_matrix, page_vector)
        if links_matrix is not None:
            links_matrix = links_matrix.astype(np.float32)  # saving memory

        reward = 0
        if not self.is_seed(response):
            reward = self.goal.get_reward(response)
            self.update_node(response, {'reward': reward})
            self.total_reward += reward
            self.rewards.append(reward)
            self.Q.add_experience(
                as_t=as_t,
                AS_t1=links_matrix,
                r_t1=reward
            )
            self.goal.response_observed(response)
        domain = get_domain(response.url)
        self.crawled_domains.add(domain)
        if reward > 0.5:
            self.relevant_domains.add(domain)

        return (list(self._links_to_requests(response, links, links_matrix)),
                reward)

    def _extract_links(self, response: TextResponse) -> List[Dict]:
        """ Return a list of all unique links on a page """
        return list(self.le.iter_link_dicts(
            response=response,
            limit_by_domain=self.stay_in_domain,
            deduplicate=False,
            deduplicate_local=True,
        ))

    def _links_to_requests(self,
                           response: TextResponse,
                           links: List[Dict],
                           links_matrix: sp.csr_matrix,
                           ) -> Iterator[scrapy.Request]:
        indices_and_links = list(self.le.deduplicate_links_enumerated(links))
        if not indices_and_links:
            return
        indices, links_to_follow = zip(*indices_and_links)
        AS = links_matrix[list(indices)]
        scores = self.Q.predict(AS)

        for link, v, score in zip(links_to_follow, AS, scores):
            url = link['url']
            next_domain = get_domain(url)
            meta = {
                'link_vector': v,
                # 'link': link,  # turn it on for debugging
                'scheduler_slot': next_domain,
            }
            priority = score_to_priority(score)
            req = scrapy.Request(url, priority=priority, meta=meta)
            set_request_domain(req, next_domain)
            yield req

    def _page_vector(self, response: TextResponse) -> np.ndarray:
        """ Convert response content to a feature vector """
        if hasattr(response, '_cached_page_vector'):
            return response._cached_page_vector
        vec = self.page_vectorizer.transform([response.text])[0]
        response._cached_page_vector = vec
        return vec

    def get_scheduler_queue(self):
        """
        This method is called by deepdeep.scheduler.Scheduler
        to create a new queue.
        """
        def new_queue(domain):
            return RequestsPriorityQueue(fifo=True,
                                         maxsize=self.domain_queue_maxsize)
        return BalancedPriorityQueue(
            queue_factory=new_queue,
            eps=self.eps,
            balancing_temperature=self.balancing_temperature,
        )

    @property
    def scheduler(self) -> Scheduler:
        return self.crawler.engine.slot.scheduler

    def on_model_changed(self):
        # TODO: this should pause engine first, in order
        # for download timeouts to work correctly
        if self.steps_before_reschedule <= 0:
            num_updated = self.recalculate_request_priorities()
            self.steps_before_reschedule = self._steps_before_rescheduling(num_updated)
        logging.info("{} steps left before next re-scheduling"
                     .format(self.steps_before_reschedule))

    def close_finished_queues(self):
        for slot in self.scheduler.queue.get_active_slots():
            if self.goal.is_acheived_for(domain=slot):
                self.scheduler.close_slot(slot)

    @log_time
    def recalculate_request_priorities(self) -> int:
        if self.baseline:
            return 0

        scores_new = []
        scores_old = []

        def request_priorities(requests: List[scrapy.Request]) -> List[int]:
            priorities = np.ndarray(len(requests), dtype=int)
            old_priorities = np.zeros_like(priorities)
            vectors, indices = [], []
            for idx, request in enumerate(requests):
                old_priorities[idx] = request.priority
                if self.is_seed(request):
                    priorities[idx] = request.priority
                    continue
                vectors.append(request.meta['link_vector'])
                indices.append(idx)
            if vectors:
                scores = np.concatenate([self.Q.predict(sp.vstack(batch))
                                         for batch in chunks(vectors, 4096)])
                priorities[indices] = scores * FLOAT_PRIORITY_MULTIPLIER

            # keep scores in order to compute metrics later
            scores_new.append(priorities / FLOAT_PRIORITY_MULTIPLIER)
            scores_old.append(old_priorities / FLOAT_PRIORITY_MULTIPLIER)

            # convert priorities to Python ints because scrapy.Request
            # doesn't support numpy int types
            priorities = [p.item() for p in priorities]

            # TODO: use _log_promising_link or remove it
            return priorities

        for slot in tqdm.tqdm(self.scheduler.queue.get_active_slots()):
            queue = self.scheduler.queue.get_queue(slot)
            queue.update_all_priorities(request_priorities)

        # Compute & print metrics.
        # The idea is to check how stable are results:
        #
        # 1. how different is domain ranking after model update?
        # 2. how different is request ranking after model update?
        #
        # For requests we're only interested in top N requests
        # (for each domain?); low-priority requests don't matter.
        #
        # For domains we're also interested mostly in top domains.
        #
        domain_scores_old = np.array([p.max() if p.size else 0 for p in scores_old])
        domain_scores_new = np.array([p.max() if p.size else 0 for p in scores_new])
        if len(scores_new) == 0:
            return 0
        scores_old_all = np.hstack(scores_old)
        scores_new_all = np.hstack(scores_new)

        logging.info("Top-100 domain ranking: NDCG={:0.4f}".format(
            ndcg_score(domain_scores_new, domain_scores_old, k=100)
        ))

        logging.info("Top-100 request ranking: NDCG={:0.4f}".format(
            ndcg_score(scores_new_all, scores_old_all, k=100)
        ))

        # FIXME: something is wrong with this micro-averaging,
        # sometimes it returns values > 1
        # domain_ndcg = np.array([
        #     ndcg_score(new, old, k=10)
        #     for new, old in zip(scores_new, scores_old)
        # ])
        # mean_domain_ndcg = domain_ndcg[~np.isnan(domain_ndcg)].mean()
        # logging.info("Top-10 micro-averaged in-domain request ranking: NDCG={:0.4f}".format(
        #     mean_domain_ndcg
        # ))

        diff = scores_new_all - scores_old_all
        rmse = np.sqrt((diff ** 2).sum() / diff.size)
        mean_abs_error = np.abs(diff).mean()
        logging.info("Request score changes: RMSE={:0.4f}, MAE={:0.4}".format(
            rmse, mean_abs_error
        ))

        for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            changed = np.abs(diff) > threshold
            logging.info("    Changed by more than {:0.2f}: {:d} ({:0.1%})".format(
                threshold, changed.sum(), changed.mean(),
            ))

        # TODO: ranking metric other than NDSG
        # It shouldn't matter that much if a request is 1st or 10th in a queue
        return scores_new_all.size  # num updated requests

    def _log_promising_link(self, link, score):
        self.logger.debug("PROMISING LINK {:0.4f}: {}\n        {}".format(
            score, link['url'], link['inside_text']
        ))

    def _examples(self):
        return None, None

    def log_stats(self):
        if self.checkpoint_path:
            logging.debug(self.checkpoint_path)
        examples, AS = self._examples()
        if examples:
            scores_target = self.Q.predict(AS)
            scores_online = self.Q.predict(AS, online=True)
            for ex, score1, score2 in zip(examples, scores_target, scores_online):
                logging.debug(" {:0.4f} {:0.4f} {}".format(score1, score2, ex))

        average_reward = self.total_reward / self.Q.t_ if self.Q.t_ else 0
        run_average_reward = np.mean(self.rewards[-100:]) if self.rewards else 0
        coef_norm_online = self.Q.coef_norm(online=True)
        coef_norm_target = self.Q.coef_norm(online=False)
        logging.debug(
            "t={}, return={:0.4f}, avg reward={:0.4f}, L2 norm: {:0.4f} {:0.4f}"
            .format(
                self.Q.t_,
                self.total_reward,
                average_reward,
                coef_norm_online,
                coef_norm_target,
            ))
        self.goal.debug_print()
        self.log_value('Reward/total', self.total_reward)
        self.log_value('Reward/average', average_reward)
        self.log_value('Reward/run-average', run_average_reward)
        self.log_value('Coef/norm_online', coef_norm_online)
        self.log_value('Coef/norm_target', coef_norm_target)

        stats = self.get_stats_item()
        logging.debug(
            "Domains: {domains_open} open, {domains_closed} closed; "
            "{todo} requests in queue, {processed} processed, "
            "{dropped} dropped, {crawled_domains} crawled, "
            "{relevant_domains} relevant."
            .format(**stats))
        self.log_value('Domains/crawled', stats['crawled_domains'])
        self.log_value('Domains/relevant', stats['relevant_domains'])
        self.log_value('Domains/open', stats['domains_open'])
        self.log_value('Domains/closed', stats['domains_closed'])
        self.log_value('Queue/todo', stats['todo'])
        self.log_value('Queue/processed', stats['processed'])
        self.log_value('Queue/dropped', stats['dropped'])

    def get_stats_item(self):
        domains_open, domains_closed = self._domain_stats()
        stats = self.crawler.stats  # type: StatsCollector
        enqueued = stats.get_value('custom-scheduler/enqueued/', 0)
        dequeued = stats.get_value('custom-scheduler/dequeued/', 0)
        dropped = stats.get_value('custom-scheduler/dropped/', 0)
        todo = enqueued - dequeued - dropped
        crawled_domains = len(self.crawled_domains)
        relevant_domains = len(self.relevant_domains)

        return {
            '_type': 'stats',
            't': self.Q.t_,
            'return': self.total_reward,
            'domains_open': domains_open,
            'domains_closed': domains_closed,
            'enqueued': enqueued,
            'processed': dequeued,
            'dropped': dropped,
            'todo': todo,
            'crawled_domains': crawled_domains,
            'relevant_domains': relevant_domains,
        }

    def _debug_expected_vs_got(self, response: Response):
        if 'link' not in response.meta:
            return
        reward = self.goal.get_reward(response)
        self.logger.debug("\nGOT {:0.4f} (expected return was {:0.4f}) {}\n{}".format(
            reward,
            priority_to_score(response.request.priority),
            response.url,
            response.meta['link'].get('inside_text'),
        ))

    def _domain_stats(self) -> Tuple[int, int]:
        domains_open = len(self.scheduler.queue.get_active_slots())
        domains_closed = len(self.scheduler.queue.closed_slots)
        return domains_open, domains_closed

    def get_params(self) -> Dict:
        keys = self._ARGS - {'checkpoint_path', 'checkpoint_interval'}
        params = {key: getattr(self, key) for key in keys}
        if getattr(self, 'crawler', None):
            params['DEPTH_PRIORITY'] = self.crawler.settings.get('DEPTH_PRIORITY')
        return params

    def maybe_checkpoint(self) -> None:
        if (self.Q.t_ % self.checkpoint_interval) != 0 or self.Q.t_ == 0:
            return
        self.do_checkpoint()

    def do_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        path = Path(self.checkpoint_path)
        id_ = 'latest' if self.checkpoint_latest else self.Q.t_
        self.dump_policy(path/("Q-%s.joblib" % id_), False)
        self.dump_crawl_graph(path/"graph.pickle")
        self.dump_queue(path/("queue-%s.csv.gz" % id_))
        # Logging queue memory stats only on checkpoints because we need
        # to do a linear scan over all queues, which can be slow.
        queue = self.scheduler.queue
        self.logger.info(
            'Queue entries {:,}, vectors bytes {:,}; '
            'Replay entries {:,}, vectors bytes {:,}'
            .format(len(queue), queue.nbytes(),
                    len(self.Q.memory), self.Q.memory.nbytes()))

    @log_time
    def dump_crawl_graph(self, path) -> None:
        if hasattr(self, 'G'):
            nx.write_gpickle(self.G, str(path))

    @log_time
    def dump_policy(self, path: Path, save_experience_replay: bool) -> None:
        """ Save the current policy """
        data = {
            'Q': self.Q,
            'link_vectorizer': self.link_vectorizer,
            'page_vectorizer': self.page_vectorizer,
            '_params': self.get_params(),
        }
        self.Q.pickle_memory = save_experience_replay
        try:
            joblib.dump(data, str(path), compress=3)
        finally:
            self.Q.pickle_memory = False
        self._save_params_json()

    @log_time
    def dump_queue(self, path: Path) -> None:
        with gzip.open(str(path), 'wt', encoding='utf8') as f:
            self.scheduler.queue.debug_dump(f)

    @classmethod
    def _steps_before_rescheduling(cls, n_requests: int,
                                   scheduling_rps: float=30000,
                                   budget: float=0.33,
                                   page_process_time_s: float=0.1) -> int:
        """
        How many steps to wait before re-scheduling if there are ``n_requests``
        in a queue, priorities can be updated at ``scheduling_rps`` speed,
        page processing time is ``page_processing_time``, and spider should
        spend about ``budget*100`` percent of time updating request priorities?
        """
        ratio = budget / (1-budget)
        return int(n_requests / scheduling_rps / ratio / page_process_time_s)
