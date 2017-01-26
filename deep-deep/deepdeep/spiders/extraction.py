import importlib
from weakref import WeakKeyDictionary
import traceback
from typing import Any, Callable, Iterable, Dict, List, Tuple

import autopager
from scrapy import Request
from scrapy.dupefilters import RFPDupeFilter
from scrapy.http.response.text import TextResponse

from .qspider import QSpider
from deepdeep.goals import BaseGoal


class ExtractionGoal(BaseGoal):
    def __init__(self,
                 extractor: Callable[[TextResponse], Iterable[Tuple[Any, Any]]],
                 request_penalty: float=1.0,
                 ) -> None:
        """ A goal is to find maximum number of unique items by doing
        minimum number of requests.
        extractor should be a function that extracts key-value pairs
        for each item found in response.
        """
        self.extractor = extractor
        self.extracted_items = set()
        self.request_reward = -request_penalty
        self.item_reward = 1.0
        self._cache = WeakKeyDictionary()  # type: WeakKeyDictionary

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            score = self.request_reward
            run_id = response.meta['run_id']
            try:
                items = list(self.extractor(response))
            except Exception:
                traceback.print_exc()
            else:
                for _key, _ in items:
                    key = (run_id, _key)
                    if key not in self.extracted_items:
                        self.extracted_items.add(key)
                        score += self.item_reward
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        pass


class ExtractionSpider(QSpider):
    """
    This spider learns how to extract data from a single domain.
    """
    name = 'extraction'
    use_urls = True
    use_same_domain = 0  # not supported by eli5 yet, and we don't need it
    balancing_temperature = 5.0  # high to make all simultaneous runs equal
    # copied from relevancy spider
    replay_sample_size = 50
    replay_maxsize = 100000  # decrease it to ~10K if use_pages is 1
    # number of simultaneous runs
    n_copies = 10

    _ARGS = {'extractor', 'n_copies'} | QSpider._ARGS
    ALLOWED_ARGUMENTS = _ARGS | QSpider.ALLOWED_ARGUMENTS

    custom_settings = dict(
        DUPEFILTER_CLASS='deepdeep.spiders.extraction.RunAwareDupeFilter',
        **QSpider.custom_settings)

    def __init__(self, *args, **kwargs):
        """ extractor argument has a "module:function" format
        and specifies where to load the extractor from.
        """
        super().__init__(*args, **kwargs)
        self.n_copies = int(self.n_copies)
        self.extractor = str(self.extractor)

    def get_goal(self):
        try:
            ex_module, ex_function = self.extractor.split(':')
        except (KeyError, ValueError):
            raise ValueError(
                'Please give extractor argument in "module:function" format')
        ex_module = importlib.import_module(ex_module)
        extractor_fn = getattr(ex_module, ex_function)
        return ExtractionGoal(extractor_fn)

    # _parse_seeds and _links_to_requests are override to allow
    # running several simultaneous independent spiders on the same domain
    # which still share the model, so it is more general.

    def _parse_seeds(self, response):
        for orig_req in super()._parse_seeds(response):
            for idx in range(self.n_copies):
                req = orig_req.copy()
                set_run_id(req, 'run-{}'.format(idx))
                yield req

    def _links_to_requests(self, response, *args, **kwargs):
        run_id = response.request.meta['run_id']
        for req in super()._links_to_requests(response, *args, **kwargs):
            set_run_id(req, run_id)
            yield req


class AutopagerBaseline(ExtractionSpider):
    """ A BFS + autopager baseline.
    """
    name = 'autopager_extraction'
    baseline = True
    eps = 0.0  # do not select requests at random
    # disable depth middleware to avoid increasing depth for pagination urls
    custom_settings = dict(ExtractionSpider.custom_settings)
    custom_settings['SPIDER_MIDDLEWARES'] = dict(
        custom_settings.get('SPIDER_MIDDLEWARES', {}))
    custom_settings['SPIDER_MIDDLEWARES'][
        'scrapy.spidermiddlewares.depth.DepthMiddleware'] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autopager = autopager.AutoPager()

    def _links_to_requests(self, response, *args, **kwargs):
        pagination_urls = set(self.autopager.urls(response))
        depth = response.meta.get('depth', 1)
        real_depth = response.meta.get('real_depth', 1)
        # print(depth, real_depth, response.meta.get('is_pagination'),
        #      response.request.priority, response.url)
        for req in super()._links_to_requests(response, *args, **kwargs):
            is_pagination = req.url in pagination_urls
            req.meta['depth'] = depth + (1 - is_pagination)
            req.meta['real_depth'] = real_depth + 1
            req.meta['is_pagination'] = is_pagination
            req.priority = -100 * req.meta['depth']
            yield req


def set_run_id(request: Request, run_id: str):
    for key in ['run_id', 'cookiejar', 'scheduler_slot']:
        request.meta[key] = run_id


class RunAwareDupeFilter(RFPDupeFilter):
    def request_fingerprint(self, request):
        fp = super().request_fingerprint(request)
        return '{}-{}'.format(request.meta.get('run_id'), fp)
