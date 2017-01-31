import importlib
import traceback
from typing import Any, Callable, Iterable, Tuple
from weakref import WeakKeyDictionary

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
                 item_callback=None,
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
        self.item_callback = item_callback
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
                for key, item in items:
                    full_key = (run_id, key)
                    if full_key not in self.extracted_items:
                        self.extracted_items.add(full_key)
                        score += self.item_reward
                    if self.item_callback:
                        self.item_callback(response.url, key, item)
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
    use_link_text = 1
    use_page_urls = 1
    use_same_domain = 0  # not supported by eli5 yet, and we don't need it
    balancing_temperature = 5.0  # high to make all simultaneous runs equal
    export_items = 1
    export_cdr = 0
    seed_url = None
    replay_sample_size = 50
    replay_maxsize = 5000  # single site needs lower replay
    replay_maxlinks = 500000  # some sites can have lots of links per page
    # number of simultaneous runs
    n_copies = 1

    _ARGS = {'extractor', 'n_copies', 'seed_url', 'export_items'} | QSpider._ARGS
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
        self.seed_url = self.seed_url
        self.export_items = bool(int(self.export_items))
        self.exported_keys = set()
        self.export_buffer = []

    def get_goal(self):
        try:
            ex_module, ex_function = self.extractor.split(':')
        except (AttributeError, ValueError):
            raise ValueError(
                'Please give extractor argument in "module:function" format')
        ex_module = importlib.import_module(ex_module)
        extractor_fn = getattr(ex_module, ex_function)
        return ExtractionGoal(extractor_fn, item_callback=self.item_callback)

    def item_callback(self, url, key, item):
        if self.export_items and key not in self.exported_keys:
            self.export_buffer.append({'url': url, 'key': key, 'item': item})
            self.exported_keys.add(key)

    def parse(self, response):
        parse_result = super().parse(response)
        if self.export_items:
            yield from self.export_buffer
            self.export_buffer = []
            for item_or_link in parse_result:
                if isinstance(item_or_link, Request):
                    yield item_or_link
        else:
            yield from parse_result

    def start_requests(self):
        if self.seeds_url is None:
            if self.seed_url is None:
                raise ValueError('Pass seeds_url or seed_url')
            yield from self._start_requests([self.seed_url])
        else:
            yield from super().start_requests()

    # Allow running several simultaneous independent spiders on the same domain
    # which still share the model, so it is more general.

    def _start_requests(self, urls):
        for orig_req in super()._start_requests(urls):
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
