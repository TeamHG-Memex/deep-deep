# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import itertools

import networkx as nx  # type: ignore
import scrapy  # type: ignore
from scrapy import signals
from scrapy.dupefilters import RFPDupeFilter  # type: ignore
from scrapy.exceptions import NotConfigured  # type: ignore

logger = logging.getLogger(__name__)


class BaseExtension:
    def __init__(self, crawler):
        self.crawler = crawler
        self.init()

    def init(self):
        pass

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)


class CrawlGraphMiddleware(BaseExtension):
    """
    This spider middleware keeps track of crawl graph.
    The graph is accessible from spider as ``spider.G`` attribute;
    node ID of each response is available as ``response.meta['node_id']``.

    Enable this middleware in settings::

        SPIDER_MIDDLEWARES = {
            'deepdeep.spidermiddlewares.CrawlGraphMiddleware': 400,
        }

    By default each node contains the following information::

        {
            'url': <response url>,
            'original url': <request url (before redirects)>,
            'visited': True/False,  # this is False for links which are not visited yet
            'ok': True/False,       # True if response is a HTTP 200 HTML response
            'priority': <request.priority>
        }

    Spider can add more information to node in two ways:

    1. set ``request.meta['node_data']`` dict with additional node attributes
       when sending the request;
    2. update ``self.G.node[response.meta['node_id']]`` dict after response
       is received (usually in a ``parse_..`` callback).

    Edge data is empty by default; to attach information to edges send requests
    with non-empty ``request.meta['edge_data']`` dicts.
    """
    def init(self):
        if not self.crawler.settings.getbool('CRAWLGRAPH_ENABLED', True):
            raise NotConfigured()

        # fixme: it should be in spider state
        self.crawler.spider.G = self.G = nx.DiGraph(name='Crawl Graph')
        self.node_ids = itertools.count()
        self.crawler.signals.connect(self.on_spider_closed,
                                     signals.spider_closed)

        self.filename = self.crawler.settings.get('CRAWLGRAPH_FILENAME', None)

        # HACKHACKHACK
        self.dupefilter = RFPDupeFilter()

    def on_spider_closed(self):
        if self.filename:
            nx.write_gpickle(self.G, self.filename)

    def process_spider_input(self, response, spider):
        """
        Assign response.node_id attribute, make sure a node exists
        in a graph and update the node with received information.
        """
        if 'node_id' not in response.meta:
            # seed requests don't have node_id yet
            response.meta['node_id'] = next(self.node_ids)

        node_id = response.meta['node_id']
        data = dict(
            url=response.url,
            visited=True,
            ok=self._response_ok(response),
            priority=response.request.priority,
        )
        spider.G.add_node(node_id, data)
        logger.debug("VISITED NODE %s %s", node_id, data)

        self.crawler.stats.inc_value('graph_nodes/visited')
        if data['ok']:
            self.crawler.stats.inc_value('graph_nodes/visited/ok')
        else:
            self.crawler.stats.inc_value('graph_nodes/visited/err')

    def process_spider_output(self, response, result, spider):
        for request in result:
            if isinstance(request, scrapy.Request):
                ok = self._process_outgoing_request(response, request, spider)
                if not ok:
                    continue
            yield request

    def _process_outgoing_request(self, response, request, spider):
        """
        Create new nodes and edges for outgoing requests.
        Data can be attached to nodes and edges using
        ``request.meta['node_data']`` and ``request.meta['edge_data']``
        dicts; these keys are then removed by this middleware.
        """
        if self.dupefilter.request_seen(request):
            return False

        this_node_id = response.meta.get('node_id')
        new_node_id = next(self.node_ids)
        request.meta['node_id'] = new_node_id

        node_data = request.meta.pop('node_data', {})
        node_data.update(
            url=request.url,
            original_url=request.url,
            priority=request.priority,
            visited=False,
            ok=None,
        )
        edge_data = request.meta.pop('edge_data', {})
        spider.G.add_node(new_node_id, node_data)
        spider.G.add_edge(this_node_id, new_node_id, edge_data)
        logger.debug("Created node %s -> %s %s", this_node_id, new_node_id, node_data)
        self.crawler.stats.set_value('graph_nodes/created', len(spider.G))
        return True

    def _response_ok(self, response):
        return response.status == 200 and hasattr(response, 'text')
