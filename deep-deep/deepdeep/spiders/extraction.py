from weakref import WeakKeyDictionary
from typing import Any, Callable, Iterable, Tuple

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
        self.extractor = extractor
        self.extracted_items = set()
        self.request_reward = -request_penalty
        self.item_reward = 1.0
        self._cache = WeakKeyDictionary()  # type: WeakKeyDictionary

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            score = self.request_reward
            run_id = response.meta['run_id']
            for _key, _ in self.extractor(response):
                key = (run_id, _key)
                if key not in self.extracted_items:
                    self.extracted_items.add(key)
                    score += self.item_reward
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        pass


def example_forum_extractor(response: TextResponse) -> Iterable[Any]:
    if not hasattr(response, 'xpath'):
        return
    thread_links = response.xpath('//a[starts-with(@id, "tid-link-")]')
    for link in thread_links:
        # FIXME - there must be a better way to do it
        thread_id = link.xpath('@id')[0].extract()
        thread_name = link.xpath('text()')[0].extract()
        yield ('thread', thread_id), thread_name
    posts = response.xpath('//td[starts-with(@id, "post-main-")]')
    for post in posts:
        post_id = post.xpath('@id')[0].extract()
        yield ('post', post_id), None


class ExtractionSpider(QSpider):
    """
    This spider learns how to extract data from a single domain.
    """
    name = 'extraction'
    use_urls = 1
    use_same_domain = 0  # not supported by eli5 yet, and we don't need it
    balancing_temperature = 5.0  # high to make all simultaneous runs equal
    # copied from relevancy spider
    replay_sample_size = 50
    replay_maxsize = 100000  # decrease it to ~10K if use_pages is 1

    n_copies = 10

    _ARGS = {'n_copies'} | QSpider._ARGS
    ALLOWED_ARGUMENTS = _ARGS | QSpider.ALLOWED_ARGUMENTS

    def get_goal(self):
        return ExtractionGoal(example_forum_extractor)

    # _parse_seeds and _links_to_requests are override to allow
    # running several simultaneous independent spiders on the same domain
    # which still share the model, so it is more general.

    custom_settings = dict(
        DUPEFILTER_CLASS='deepdeep.spiders.extraction.RunAwareDupeFilter',
        **QSpider.custom_settings)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_copies = int(self.n_copies)

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


def set_run_id(request: Request, run_id: str):
    for key in ['run_id', 'cookiejar', 'scheduler_slot']:
        request.meta[key] = run_id


class RunAwareDupeFilter(RFPDupeFilter):
    def request_fingerprint(self, request):
        fp = super().request_fingerprint(request)
        return '{}-{}'.format(request.meta.get('run_id'), fp)
