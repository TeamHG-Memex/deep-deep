# -*- coding: utf-8 -*-
import math
from typing import List

import formasaurus  # type: ignore
from formasaurus.text import tokenize, token_ngrams  # type: ignore
from scrapy.http import Response  # type: ignore

from deepdeep.utils import dict_aggregate_max, html2text


# ========== form-based relevancy functions

def forms_info(response):
    """ Return a list of form classification results """
    res = formasaurus.extract_forms(response.text, proba=True,
                                    threshold=0, fields=True)
    return [info for form, info in res]


def max_scores(page_forms_info):
    """ Return aggregate form scores for a page """
    return dict_aggregate_max(*[f['form'] for f in page_forms_info])


def response_max_scores(response):
    """ Return aggregate form scores for a page """
    return max_scores(forms_info(response))


# ========== keyword-based relevancy functions

def keywords_response_relevancy(response: Response,
                                pos_keywords: List[str],
                                neg_keywords: List[str],
                                max_ngram=1):
    """
    Relevancy score based on how many keywords from a list are
    in response text.

    Score is transformed using a weird log scale (fixme)
    to *roughly* fit [0,1] interval and to not require all keywords to be
    present for a page to be relevant.
    """
    if not hasattr(response, 'text'):
        return 0.0
    return keyword_relevancy(response.text, pos_keywords, neg_keywords, max_ngram)


def keyword_text_relevancy(text: str,
                           pos_keywords: List[str],
                           neg_keywords: List[str],
                           max_ngram=1):
    tokens = tokenize(text)
    tokens = set(token_ngrams(tokens, 1, max_ngram))

    def _score(keywords: List[str]) -> float:
        s = sum(int(k in tokens) for k in keywords)
        return _scale_relevancy(s, keywords)

    pos_score = _score(pos_keywords)
    neg_score = _score(neg_keywords)

    return max(0, pos_score - 0.33 * neg_score)


def keyword_relevancy(response_html: str,
                      pos_keywords: List[str],
                      neg_keywords: List[str],
                      max_ngram=1):
    text = html2text(response_html).lower()
    return keyword_text_relevancy(text, pos_keywords, neg_keywords, max_ngram)


def max_ngram_length(keywords: List[str]) -> int:
    """
    >>> max_ngram_length(["foo"])
    1
    >>> max_ngram_length(["foo", "foo  bar"])
    2
    >>> max_ngram_length(["  foo", "foo bar", "foo bar baz "])
    3
    """
    return max(len(keyword.split()) for keyword in keywords)


def _scale_relevancy(score: float, keywords: List) -> float:
    """ Weird log scale to use for keyword occurance count """
    return math.log(score + 1, len(keywords) / 2 + 2)

