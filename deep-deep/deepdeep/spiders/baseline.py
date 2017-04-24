# -*- coding: utf-8 -*-
import re
import random
from urllib.parse import urlsplit

import scrapy  # type: ignore

from deepdeep.utils import (
    get_response_domain,
    set_request_domain,
    decreasing_priority_iter,
)
from deepdeep.spiders._base import BaseSpider
from deepdeep.score_pages import forms_info, max_scores


class CrawlAllSpider(BaseSpider):
    """
    Spider for crawling experiments.

    It is written as a single spider with arguments (not as multiple spiders)
    in order to share HTTP cache.
    """
    name = 'all'

    shuffle = 1  # follow links in order or randomly
    heuristic = 0  # prefer registration/account links

    custom_settings = {
        'DEPTH_LIMIT': 1,  # override it using -s DEPTH_LIMIT=2
        'DEPTH_PRIORITY': 1,
        'CRAWLGRAPH_ENABLED': True,
        'CRAWLGRAPH_FILENAME': 'graph.pickle',
    }

    ALLOWED_ARGUMENTS = {'shuffle', 'heuristic'} | BaseSpider.ALLOWED_ARGUMENTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic_re = re.compile("(regi|join|create|sign|account|user|login|recover|password)")
        self.heuristic = int(self.heuristic)
        self.shuffle = int(self.shuffle)

    def parse(self, response):
        self.increase_response_count()
        node_id = response.meta['node_id']

        if not hasattr(response, 'text'):
            # can't decode the response
            # XXX: it should be set in midleware, why doesn't it work?
            # self.G[node_id]['ok'] = False
            return

        res = forms_info(response)
        self.G.node[node_id]['info'] = {
            'depth': response.meta['depth'],
            'forms': res,
            'scores': max_scores(res),
            'domain': get_response_domain(response),
        }

        yield from self.crawl_baseline(response,
            shuffle=self.shuffle,
            prioritize_re=None if not self.heuristic else self.heuristic_re
        )

    def crawl_baseline(self, response, shuffle, prioritize_re=None):
        """
        Baseline crawling algoritms.

        When shuffle=True, links are selected at random.
        When prioritize_re is not None, links which URLs follow specified
        regexes are prioritized.

        Raw link features are stored as edge data.
        """

        # limit crawl to the first domain
        links = list(self.le.iter_link_dicts(response, limit_by_domain=True))

        if shuffle:
            random.shuffle(links)

        domain = get_response_domain(response)
        for priority, link in zip(decreasing_priority_iter(), links):
            url = link['url']

            if prioritize_re:
                s = prioritize_re.search
                p = urlsplit(url)
                if s(p.path) or s(p.query) or s(p.fragment):
                    priority = 1

            req = scrapy.Request(url, priority=priority, meta={
                'edge_data': link,
            })
            set_request_domain(req, domain)
            yield req
