import scrapy
from scrapy.http.response import Response

from deepdeep.predictor import LinkClassifier


class Spider(scrapy.Spider):
    """ Example standalone spider. Run it with:

    scrapy runspider ../examples/standalone.py \
        -a url=http://example.com \
        -a q_model=Q.joblib

    """
    name = 'standalone'

    def __init__(self, url, q_model):
        super().__init__()
        self.start_urls = [url]
        self.link_clf = LinkClassifier.load(q_model)

    def parse(self, response: Response):
        yield {'url': response.url,
               'priority': response.request.priority}
        for score, url in self.link_clf.extract_urls_from_response(response):
            # To use default scrapy queue, we need to make priority
            # a low-cardinality integer. Typical score range is 0 .. 1.2,
            # so we'll have about 20 different "levels" of requests.
            priority = int(score * 20)
            yield scrapy.Request(url, priority=priority)
