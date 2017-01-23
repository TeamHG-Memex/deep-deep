from weakref import WeakKeyDictionary
from typing import Any, Callable, Iterable, Tuple

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
            for key, _ in self.extractor(response):
                if key not in self.extracted_items:
                    self.extracted_items.add(key)
                    score += self.item_reward
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse):
        pass


def example_forum_extractor(response: TextResponse) -> Iterable[Any]:
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

    def get_goal(self):
        return ExtractionGoal(example_forum_extractor)
