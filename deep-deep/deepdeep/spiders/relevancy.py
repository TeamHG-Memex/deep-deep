# -*- coding: utf-8 -*-
from __future__ import absolute_import
import abc
from pathlib import Path
import pickle
from typing import List, Optional

import joblib
from scrapy.http import Response, TextResponse

from .qspider import QSpider
from deepdeep.goals import RelevancyGoal
from deepdeep.utils import html2text


class _RelevancySpider(QSpider, metaclass=abc.ABCMeta):
    """
    This spider learns how to crawl relevant pages.
    """
    ALLOWED_ARGUMENTS = QSpider.ALLOWED_ARGUMENTS | {
        'max_requests_per_domain',
        'max_relevant_pages_per_domain'
    }
    _ARGS = QSpider._ARGS | {
        'max_requests_per_domain',
        'max_relevant_pages_per_domain'
    }

    # overrides for default option values used in QSpider
    stay_in_domain = False
    balancing_temperature = 0.1
    replay_sample_size = 50
    replay_maxsize = 100000  # decrease it to ~10K if use_pages is 1
    use_full_urls = 1

    # Options to limit a number of requests per domains.
    # Set a limit if the goal is to find many relevant domains
    # and/or train Q function to use in a large crawl.
    max_requests_per_domain = None  # type: Optional[int]
    max_relevant_pages_per_domain = None  # type: Optional[int]

    custom_settings = {
        # copied from QSpider
        # 'DEPTH_LIMIT': 100,
        'DEPTH_PRIORITY': 1,

        # disable OffsiteDownloaderMiddleware
        'DOWNLOADER_MIDDLEWARES': {
           'deepdeep.downloadermiddlewares.OffsiteDownloaderMiddleware': None,
        },
    }

    @abc.abstractmethod
    def relevancy(self, response: Response) -> float:
        pass

    def get_goal(self):
        if self.max_requests_per_domain is not None:
            self.max_requests_per_domain = int(self.max_requests_per_domain)
        if self.max_relevant_pages_per_domain is not None:
            self.max_relevant_pages_per_domain = int(self.max_relevant_pages_per_domain)
        return RelevancyGoal(
            relevancy=self.relevancy,
            max_requests_per_domain=self.max_requests_per_domain,
            max_relevant_pages_per_domain=self.max_relevant_pages_per_domain,
        )


class KeywordRelevancySpider(_RelevancySpider):
    """
    This spider learns how to crawl relevant pages.
    What is relevant is defined by a keywords.txt file: it is
    a file with keywords, each keyword (probably multi-word) on a single line.
    Start line with - if keyword should be considered negative,
    i.e. page is less relevant if keyword is present.

    Pass a path to keywords file using keywords_file argument::

        scrapy crawl relevant-keywords -a keywords_file=/path/to/keywords.txt

    """
    name = 'relevant-keywords'
    ALLOWED_ARGUMENTS = _RelevancySpider.ALLOWED_ARGUMENTS | {'keywords_file'}
    _ARGS = _RelevancySpider._ARGS | {'pos_keywords', 'neg_keywords'}

    # a file with keywords
    keywords_file = None   # type: str

    # these are not spider arguments!
    pos_keywords = []      # type: List[str]
    neg_keywords = []      # type: List[str]

    def __init__(self, *args, **kwargs):
        from deepdeep.score_pages import max_ngram_length

        super().__init__(*args, **kwargs)
        keywords = Path(self.keywords_file).read_text().splitlines()
        self.pos_keywords = [k for k in keywords if not k.startswith('-')]
        self.neg_keywords = [k[1:] for k in keywords if k.startswith('-')]
        self.max_ngram = max_ngram_length(self.pos_keywords)
        self._save_params_json()

    def relevancy(self, response: Response) -> float:
        from deepdeep.score_pages import keywords_response_relevancy
        return keywords_response_relevancy(response,
                                           pos_keywords=self.pos_keywords,
                                           neg_keywords=self.neg_keywords,
                                           max_ngram=self.max_ngram)


class ClassifierRelevancySpider(_RelevancySpider):
    name = 'relevant'
    ALLOWED_ARGUMENTS = _RelevancySpider.ALLOWED_ARGUMENTS | {
        'classifier_path',
        'classifier_input',
    }
    _ARGS = _RelevancySpider._ARGS | {
        'classifier_path',
        'classifier_input',
    }
    CLASSIFIER_INPUT_ALLOWED_VALUES = ['text', 'text_url', 'html', 'vector']

    # a file with saved page relevancy classifier
    classifier_path = None  # type: str

    # Relevancy classifier input. Allowed values:
    # * 'text' - use text content of the response (default);
    # * 'text_url' - use text content and url of the response;
    # * 'html' - use raw HTML of the response;
    # * 'vector' - reuse page vector computed for Q learning.
    classifier_input = 'text'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.classifier_path:
            raise ValueError("classifier_path is required")

        if self.classifier_path.endswith('.pkl'):
            with open(self.classifier_path, 'wb') as f:
                self.relevancy_clf = pickle.load(f)
        else:
            self.relevancy_clf = joblib.load(self.classifier_path)
        if self.classifier_input not in self.CLASSIFIER_INPUT_ALLOWED_VALUES:
            raise ValueError("classifier_input must be one of %r" %
                             self.CLASSIFIER_INPUT_ALLOWED_VALUES)

    def relevancy(self, response: Response) -> float:
        if not isinstance(response, TextResponse):
            # XXX: only text responses are supported
            return 0.0

        if self.classifier_input == 'vector':
            x = self._page_vector(response)
        elif self.classifier_input == 'text':
            x = html2text(response.text)
        elif self.classifier_input == 'text_url':
            x = {'text': html2text(response.text), 'url': response.url}
        elif self.classifier_input == 'html':
            x = response.text
        else:
            raise ValueError("self.classifier_input is invalid")

        return float(self.relevancy_clf.predict_proba([x])[0, 1])
