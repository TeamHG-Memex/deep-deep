from typing import Any, Iterable, Tuple

from scrapy.http.response.text import TextResponse


def forum_ipb_extractor(response: TextResponse) -> Iterable[Any]:
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
