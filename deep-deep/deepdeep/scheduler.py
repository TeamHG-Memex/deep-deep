# -*- coding: utf-8 -*-
from scrapy.utils.misc import load_object  # type: ignore

from deepdeep.queues import RequestsPriorityQueue, QueueClosed


class Scheduler:
    """
    This scheduler allows to customize request queue class:
    by default ``deepdeep.queues.RequestsPriorityQueue`` is used,
    but a spider can implement ``get_scheduler_queue()`` method
    which returns another queue class.
    """
    def __init__(self, dupefilter, stats):
        self.dupefilter = dupefilter
        self.stats = stats
        self.queue = None
        self.spider = None

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        dupefilter_cls = load_object(settings['DUPEFILTER_CLASS'])
        dupefilter = dupefilter_cls.from_settings(settings)
        return cls(
            dupefilter=dupefilter,
            stats=crawler.stats,
        )

    def has_pending_requests(self):
        return len(self.queue) > 0

    def open(self, spider):
        self.spider = spider
        if hasattr(spider, 'get_scheduler_queue'):
            self.queue = spider.get_scheduler_queue()
        else:
            self.queue = RequestsPriorityQueue(fifo=True)
        return self.dupefilter.open()

    def close(self, reason):
        return self.dupefilter.close(reason)

    def enqueue_request(self, request):
        if not request.dont_filter:
            if self.dupefilter.request_seen(request):
                self.dupefilter.log(request, self.spider)
                return False

        try:
            self.stats.inc_value('custom-scheduler/enqueued/', spider=self.spider)
            self.queue.push(request)
        except QueueClosed:
            self.stats.inc_value('custom-scheduler/dropped/', spider=self.spider)
        return True

    def next_request(self):
        request = self.queue.pop()
        if request:
            self.stats.inc_value('custom-scheduler/dequeued/', spider=self.spider)
        return request

    def close_slot(self, slot: str) -> None:
        """
        Stop processing requests for a given slot.
        This function doesn't work if scheduler queue is
        not a BalancedPriorityQueue.
        """
        num_dropped = self.queue.close_queue(slot)
        self.stats.inc_value('custom-scheduler/dropped/', num_dropped,
                             spider=self.spider)
