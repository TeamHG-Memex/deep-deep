# -*- coding: utf-8 -*-
"""
Crawl objectives
================

Crawl objective (goal) classes define how is reward computed.
"""
from __future__ import absolute_import
import abc
from typing import Callable
from weakref import WeakKeyDictionary
from collections import defaultdict
import logging

from scrapy.http.response.text import TextResponse  # type: ignore
from scrapy.http import Response  # type: ignore

from deepdeep.score_pages import response_max_scores
from deepdeep.utils import get_response_domain, MaxScores


class BaseGoal(metaclass=abc.ABCMeta):
    """
    Abstract base class for crawling objectives.
    """

    @abc.abstractmethod
    def get_reward(self, response: Response) -> float:
        """
        Return a reward for a response.
        This method may be called several times; it shouldn't update
        internal goal state. Implement :meth:`response_observed` method
        to update internal goal state.
        """
        pass

    @abc.abstractmethod
    def response_observed(self, response: TextResponse) -> None:
        """
        Update internal state with the received response.
        This method is called after all :meth:`get_reward` calls.
        """
        pass

    def is_acheived_for(self, domain: str) -> bool:
        """
        This method should return True if a spider should stop
        processing the website.
        """
        return False

    def debug_print(self) -> None:
        """ Override this method to print debug information during the crawl """
        pass


class RelevancyGoal(BaseGoal):
    """
    The goal is two-fold:

    1) find new domains which has relevant information;
    2) find relevant information on a website.

    In order to prioritize (1) over (2) RelevancyGoal provides
    these options:

    a) it can stop crawling a domain after a certain
       number of pages (see ``max_requests_per_domain``);
    b) it can stop crawling a domain after a certain amount of relevant pages
       (see ``max_relevant_pages_per_domain`` and ``relevancy_threshold``);

    The idea behind (a) and (b) limits is to stop crawling a website after
    we're sure it is relevant, to free up resources for other websites.

    The difference between (a) and (b) is in how spider handles 'hub' websites
    with no relevant content, but with lots of links to other domains
    with relevant content: with (b) spider will keep crawling these hubs,
    while with (a) it won't.

    A third approach was also tried: add a larger bonus for the first
    relevant page on a website; this should encourage spider to go to
    new domains, but it didn't work.

    Parameters
    ----------

    relevancy : callable
        Function to compute relevancy score for a response. It should
        accept scrapy.http.Response and return a score (float value).
        This score is used as a reward.
    max_requests_per_domain: int, optional
        Maximum number of requests to send to a single domain, or None
        if there is no limit. Default is None.
    max_relevant_pages_per_domain: float, optional
        Maximum number of reward accumulated for a single domain, or None
        if there is no limit. Default is None.
    relevancy_threshold: float
        Minimum relevancy required to increase
        relevant pages count. See `max_relevant_pages_per_domain`.
        Default threshold is 0.1.
    """
    def __init__(self,
                 relevancy: Callable[[Response], float],
                 max_requests_per_domain: int = None,
                 max_relevant_pages_per_domain: float = None,
                 relevancy_threshold: float = 0.1
                 ) -> None:
        self.relevancy = relevancy
        self.relevancy_threshold = relevancy_threshold
        self.max_requests_per_domain = max_requests_per_domain
        self.max_relevant_pages_per_domain = max_relevant_pages_per_domain

        self.request_count = defaultdict(int)  # type: defaultdict
        self.relevant_pages_found = defaultdict(int)  # type: defaultdict

    def get_reward(self, response: Response) -> float:
        return self.relevancy(response)

    def response_observed(self, response: TextResponse) -> None:
        domain = get_response_domain(response)
        self.request_count[domain] += 1
        if self.relevancy(response) >= self.relevancy_threshold:
            self.relevant_pages_found[domain] += 1

    def is_acheived_for(self, domain: str):
        return (
            self._max_requests_reached(domain) or
            self._max_relevant_pages_reached(domain)
        )

    def _max_requests_reached(self, domain: str) -> bool:
        if self.max_requests_per_domain is None:
            return False
        return self.request_count[domain] >= self.max_requests_per_domain

    def _max_relevant_pages_reached(self, domain: str) -> bool:
        if self.max_relevant_pages_per_domain is None:
            return False
        return self.relevant_pages_found[domain] >= self.max_relevant_pages_per_domain


class FormasaurusGoal(BaseGoal):
    """
    The goal is to find a HTML form of a given type on each website.
    When the form is found, crawling is stopped for a domain.

    ``"password/login recovery"`` forms provide a nice testbed for
    crawling algorithms because a link to the password recovery page is usually
    present on a login page, but not on other website pages. So in order to
    find these forms efficiently crawler must learn to prioritize 'login'
    links, not only 'password recovery' links.

    Parameters
    ----------

    formtype : str
        Form type to look for. Allowed values:

        * "search"
        * "login"
        * "registration"
        * "password/login recovery"
        * "contact/comment"
        * "join mailing list"
        * "order/add to cart"
        * "other"

    threshold : float
         Probability threshold required to consider the goal acheived
         for a domain (default: 0.7).
    """
    def __init__(self, formtype: str, threshold: float=0.7) -> None:
        self.formtype = formtype
        self.threshold = threshold
        self._cache = WeakKeyDictionary()  # type: WeakKeyDictionary
        self._domain_scores = MaxScores()  # domain -> max score

    def get_reward(self, response: TextResponse) -> float:
        if response not in self._cache:
            if hasattr(response, 'text'):
                scores = response_max_scores(response)
                score = scores.get(self.formtype, 0.0)
                # score = score if score > 0.5 else 0
            else:
                score = 0.0
            self._cache[response] = score
        return self._cache[response]

    def response_observed(self, response: TextResponse) -> None:
        reward = self.get_reward(response)
        domain = get_response_domain(response)
        self._domain_scores.update(domain, reward)

    def is_acheived_for(self, domain: str) -> bool:
        score = self._domain_scores[domain]
        should_close = score > self.threshold
        if should_close:
            logging.debug(
                "Domain {} is going to be closed; score={:0.4f}.".format(
                domain, score))
        return should_close

    def debug_print(self) -> None:
        logging.debug("Scores: sum={:8.1f}, avg={:0.4f}".format(
            self._domain_scores.sum(), self._domain_scores.avg()
        ))
