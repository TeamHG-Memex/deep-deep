# -*- coding: utf-8 -*-
from __future__ import absolute_import
from itertools import chain
from typing import Dict

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer
from formasaurus.text import normalize

from deepdeep.utils import url_path_query, html2text, canonicalize_url


def LinkVectorizer(use_url: bool=False,
                   use_full_url: bool=False,
                   use_same_domain: bool=True,
                   use_link_text: bool=True,
                   use_page_url: bool=False,
                   use_full_page_url: bool=False,
                   ):
    """
    Vectorizer for converting link dicts to feature vectors.
    """
    if use_url and use_full_url:
        raise ValueError("``use_url`` and ``use_full_url`` can't be both True")

    vectorizers = []

    if use_link_text:
        text_vec = HashingVectorizer(
            preprocessor=_link_inside_text,
            n_features=1024*1024,
            binary=True,
            norm='l2',
            # ngram_range=(1, 2),
            analyzer='char',
            ngram_range=(3, 5),
        )
        vectorizers.append(text_vec)

    if use_same_domain:
        same_domain = FunctionTransformer(_same_domain_feature, validate=False)
        vectorizers.append(same_domain)

    if use_url or use_full_url:
        preprocessor = _clean_url if use_url else _clean_url_keep_domain
        vectorizers.append(_url_vectorizer(preprocessor))

    if use_page_url or use_full_page_url:
        # It would be faster to run it only once per page
        preprocessor = (
            _clean_page_url if use_url else _clean_page_url_keep_domain)
        vectorizers.append(_url_vectorizer(preprocessor))

    if not vectorizers:
        raise ValueError('Please enable at least one vectorizer')

    return make_union(*vectorizers)


def _url_vectorizer(preprocessor):
    return HashingVectorizer(
        preprocessor=preprocessor,
        n_features=1024*1024,
        binary=True,
        analyzer='char',
        ngram_range=(4, 5),
    )


def PageVectorizer():
    """ Vectorizer for converting page HTML content to feature vectors """
    text_vec = HashingVectorizer(
        preprocessor=_html_text_lower,
        n_features=1024*1024,
        binary=False,
        ngram_range=(1, 1),
    )
    return text_vec


def LDAPageVctorizer(n_topics: int, batch_size: int, min_df: int, verbose=1,
                     max_features: int=None):
    """
    Vectorizer for converting page HTML content to feature vectors using LDA.
    Train it with scripts/train-lda.py script.
    """
    vec = CountVectorizer(
        preprocessor=_html_text_lower,
        stop_words=_get_stop_words(),
        min_df=min_df,
        max_features=max_features,
    )
    lda = LatentDirichletAllocation(
        n_topics=n_topics,
        batch_size=batch_size,
        evaluate_every=2,
        verbose=verbose,
    )

    # A workaround for scikit-learn 0.17 bug.
    # See https://github.com/scikit-learn/scikit-learn/issues/6320
    norm = Normalizer(norm='l1', copy=False)

    return make_pipeline(vec, lda, norm)


def _get_stop_words():
    import stop_words

    return set(chain.from_iterable(
        stop_words.get_stop_words(lang)
        for lang in stop_words.AVAILABLE_LANGUAGES
    ))


def _link_inside_text(link: Dict) -> str:
    text = link.get('inside_text', '')
    title = link.get('attrs', {}).get('title', '')
    return normalize(text + ' ' + title)


def _clean_url(link: Dict) -> str:
    return url_path_query(_clean_url_keep_domain(link))


def _clean_url_keep_domain(link: Dict) -> str:
    return canonicalize_url(link.get('url'))


def _clean_page_url(link: Dict) -> str:
    return url_path_query(_clean_page_url_keep_domain(link))


def _clean_page_url_keep_domain(link: Dict) -> str:
    return canonicalize_url(link.get('page_url'))


def _same_domain_feature(links):
    return np.asarray([
        link['domain_from'] == link['domain_to'] for link in links
    ]).reshape((-1, 1))


def _html_text_lower(html: str) -> str:
    return html2text(html).lower()
