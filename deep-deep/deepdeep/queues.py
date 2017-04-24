# -*- coding: utf-8 -*-
"""
Queues
======

There are several conflicting goals which scheduler needs to acheive.
It should:

1. prefer to crawl all domains;
2. prefer more promising links;
3. allow less promising links to be crawled with some probability
   (ε-greedy policy);
4. allow to update link priorities dynamically.

This module contains custom Scrapy queues which allow to do that:
:class:`BalancedPriorityQueue` allows to have per-domain request queues and
sample from them, :class:`RequestsPriorityQueue` is a per-domain queue
which allows to update request priorities.
"""
import heapq
import itertools
import random
import csv
from typing import (
    List, Any, Iterable, Optional, Callable, Dict, Iterator, Set, TextIO,
    Sized,
)

import numpy as np  # type: ignore
import scrapy  # type: ignore

from deepdeep.utils import softmax, log_time, csr_nbytes


FLOAT_PRIORITY_MULTIPLIER = 10000


def score_to_priority(score: float) -> int:
    return int(score * FLOAT_PRIORITY_MULTIPLIER)


def priority_to_score(prio: int) -> float:
    return prio / FLOAT_PRIORITY_MULTIPLIER


class QueueClosed(Exception):
    pass


class RequestsPriorityQueue(Sized):
    """
    In-memory priority queue for requests.

    Unlike default Scrapy queues it supports high-cardinality priorities
    (but no float priorities because scrapy.Request doesn't support them).

    This queue allows to change request priorities. To do it

    1. iterate over queue.entries;
    2. call queue.change_priority(entry, new_priority) for each entry;
    3. call queue.heapify()

    It also allows to remove a request from a queue using remove_entry,
    and limit queue size with maxsize argument (queue is trimmed when
    updating request priorities).
    """

    REMOVED = object()

    REMOVED_PRIORITY = score_to_priority(15000)
    EMPTY_PRIORITY = score_to_priority(-15000)

    def __init__(self, fifo: bool=True, maxsize: Optional[int]=None) -> None:
        # entries are lists of [int, int, scrapy.Request]
        self.entries = []  # type: List[List]
        step = 1 if fifo else -1
        self.counter = itertools.count(step=step)
        self.maxsize = maxsize

    def push(self, request: scrapy.Request) -> List:
        count = next(self.counter)
        entry = [-request.priority, count, request]
        heapq.heappush(self.entries, entry)
        return entry

    def pop(self) -> Optional[scrapy.Request]:
        while self.entries:
            priority, count, request = heapq.heappop(self.entries)
            if request is not self.REMOVED:
                self._pop_empty()
                return request

    @classmethod
    def change_priority(cls,
                        entry: List,
                        new_priority: int) -> None:
        """
        Change priority of an existing entry.

        ``entry`` is an item from :attr:`entries` attribute.

        After priorities are changed it is necessary to call
        :meth:`heapify`.
        """
        entry[0] = -new_priority
        if cls.entry_is_active(entry):
            entry[2].priority = new_priority

    @classmethod
    def entry_is_active(cls, entry: List) -> bool:
        return entry[2] is not cls.REMOVED

    def iter_active_entries(self) -> Iterator[List]:
        return (e for e in self.entries if self.entry_is_active(e))

    def update_all_priorities(self,
                              compute_priority_func: Callable[[List[scrapy.Request]], List[int]]) -> None:
        """
        Update all request priorities.

        ``compute_priority_func`` is a function which returns
        new priority; it should accept a list of Requests and return a list of
        integer priorities.
        """
        requests = list(self.iter_requests())
        new_priorities = compute_priority_func(requests)
        n = len(new_priorities)
        if self.maxsize and n > self.maxsize:
            n_rm = n - self.maxsize
            to_remove_idx = np.array(new_priorities).argpartition(n_rm)[:n_rm]
            to_remove = np.zeros(n, dtype=np.bool)
            to_remove[to_remove_idx] = True
        else:
            to_remove = itertools.repeat(False)
        for entry, priority, remove in zip(self.iter_active_entries(),
                                           new_priorities,
                                           to_remove):
            if remove:
                self.remove_entry(entry)
            else:
                self.change_priority(entry, priority)
        self.heapify()

    def remove_entry(self, entry: List) -> scrapy.Request:
        """
        Mark an existing entry as removed.
        ``entry`` is an item from :attr:`entries` attribute.
        """
        request = entry[2]
        entry[2] = self.REMOVED
        # move removed entry to the top at next heapify call
        max_prio = 0 if not self.entries else -self.entries[0][0]
        entry[0] = - (max_prio + self.REMOVED_PRIORITY)
        return request

    def pop_random(self, n_attempts: int=10) -> Optional[scrapy.Request]:
        """ Pop random entry from a queue """
        self._pop_empty()
        if not self.entries:
            return

        # Because we've called _pop_empty it is guaranteed there is at least
        # one non-removed entry in a queue (the one at the top).
        for i in range(n_attempts):
            entry = random.choice(self.entries)
            if entry[2] is not self.REMOVED:
                request = self.remove_entry(entry)
                self._pop_empty()
                return request

    def max_priority(self) -> int:
        """ Return maximum request priority in this queue """
        if not self.entries:
            return self.EMPTY_PRIORITY
        return -self.entries[0][0]

    @property
    def next_request(self) -> Optional[scrapy.Request]:
        if not self.entries:
            return None
        return self.entries[0][2]

    def heapify(self) -> None:
        heapq.heapify(self.entries)
        self._pop_empty()

    def _pop_empty(self) -> None:
        """ Pop all removed entries from heap top """
        while self.entries and self.next_request is self.REMOVED:
            heapq.heappop(self.entries)

    def iter_requests(self) -> Iterable[scrapy.Request]:
        """
        Return all Request objects in a queue.
        The first request is guaranteed to have top priority;
        order of other requests is arbitrary.
        """
        return (e[2] for e in self.iter_active_entries())

    def __len__(self) -> int:
        return len(self.entries)

    def nbytes(self) -> int:
        """
        Memory taken by link vectors in requests stored in self.entries.
        """
        return sum(request_nbytes(request) for _, _, request in self.entries)


class BalancedPriorityQueue:
    """
    This queue samples other queues randomly, based on their weights
    (i.e. based on top request priority in a given queue).

    "Bins" to balance should be set in ``request.meta['scheduler_slot']``.
    For each ``scheduler_slot`` value a separate queue is created.

    queue_factory should be a function which returns a new
    RequestsPriorityQueue for a given slot name.

    ``eps`` is a probability of choosing random queue and
    returning random request from it. Because sampling is two-stage,
    it is biased towards queues with fewer requests.

    ``balancing_temperature`` is a parameter which controls how to
    choose the queue to get requests from. If the value is high,
    queue will be selected almost randomly. If the value is close to zero,
    queue with a highest request priority will be selected with a probability
    close to 1. Default value is 1.0; it means queues are selected randomly
    with probabilities proportional to max priority of their requests.

    Requests are fetched in batches; ``batch_size`` is a parameter
    which suggests a number of non-random requests in a batch.
    Average size of actual batch is ``batch_size*(1+eps)``.
    When ``batch_size`` is set to None (default), a heuristic algorithm
    is used to choose the batch size - the greater is a number of queues
    being balanced, the larger is a batch size.
    """
    def __init__(self,
                 queue_factory: Callable[[str], RequestsPriorityQueue],
                 eps: float=0.0,
                 balancing_temperature: float=1.0,
                 batch_size: Optional[int]=None,
                 ) -> None:
        assert balancing_temperature > 0
        self.queues = {}  # type: Dict[str, RequestsPriorityQueue]
        self.closed_slots = set()  # type: Set[str]
        self.eps = eps
        self.queue_factory = queue_factory
        self.balancing_temperature = balancing_temperature
        self._batch_size = batch_size
        self._buffer = []  # type: List[scrapy.Request]

    def push(self, request: scrapy.Request) -> None:
        slot = request.meta.get('scheduler_slot')
        if slot in self.closed_slots:
            raise QueueClosed()
        if slot not in self.queues:
            self.queues[slot] = self.queue_factory(slot)
        self.queues[slot].push(request)

    def pop(self) -> Optional[scrapy.Request]:
        if not self._buffer:
            self._buffer.extend(self._pop_many(self.batch_size))

        if self._buffer:
            return self._buffer.pop()

    @property
    def batch_size(self) -> int:
        if self._batch_size is not None:
            return self._batch_size
        # With small number of domains in a queue batching is not needed
        # and hurts sampling quality. With a large number of domains it is
        # crucial for fast sampling, and negative effects are much less
        # profound.
        return min(1000, max(1, len(self.queues) // 1000))

    @log_time
    def _pop_many(self, n: int) -> List[scrapy.Request]:
        all_slots = list(self.queues.keys())
        if not all_slots:
            return []

        weights = [q.max_priority() for q in self.queues.values()]
        temperature = FLOAT_PRIORITY_MULTIPLIER * self.balancing_temperature
        p = softmax(weights, t=temperature)
        chosen_slots = np.random.choice(all_slots, size=n, replace=True, p=p)

        # It is not possible to get a required amount of requests
        # from some domain queues - high-priority domain can be chosen too many
        # times. This changes a % of random requests in fixed-size batches:
        # there can be a much larger % of random requests because of these
        # unsuccessful attempts to get non-random requests.
        #
        # So instead of using a fixed batch size and making some requests
        # in it random, we're *adding* some amount of random requests
        # to the batch. The amount of random requests is chosen to make
        # average ratio of random requests equal to ``eps``.

        queues = np.asarray([self.queues[slot] for slot in chosen_slots])
        requests = [r for r in [q.pop() for q in queues] if r]

        # XXX: n_random is not 100% correct because there can be not enough
        # requests to pop from random queues as well. But it doesn't look
        # like a problem in practice (?); it makes effective ``eps`` slightly
        # smaller.
        n_random = np.random.binomial(
            n=len(requests) * (1 + self.eps),
            p=self.eps
        )
        random_queues = [
            self.queues[slot]
            for slot in np.random.choice(all_slots, size=n_random)
        ]
        for queue in random_queues:
            request = queue.pop_random()
            if request is not None:
                request.meta['from_random_policy'] = True
                requests.append(request)

        random.shuffle(requests)
        # print("======= Unique domains selected: %s" % len(set(chosen_slots)))
        # print("======= Random requests: %d/%d" % (n_random, len(requests)))
        return requests

    def get_active_slots(self) -> List[str]:
        return [key for key, queue in self.queues.items() if len(queue)]

    def get_queue(self, slot: str) -> RequestsPriorityQueue:
        return self.queues[slot]

    def close_queue(self, slot: str) -> int:
        """
        Close a queue. Requests for this queue are dropped,
        including requests which are already scheduled.

        Return a number of dropped requests.
        """
        self.closed_slots.add(slot)
        queue = self.queues.pop(slot, None) or []
        return len(queue)

    def debug_dump(self, fp: TextIO) -> None:
        """ Dump debug information about this queue to a .csv file """
        writer = csv.DictWriter(fp, ["priority", "slot", "url"])
        writer.writeheader()
        for req in self._buffer:
            writer.writerow({
                'url': req.url,
                'priority': req.priority,
                'slot': '<BUFFER>',
            })
        for slot, queue in self.queues.items():
            for req in queue.iter_requests():
                writer.writerow({
                    'url': req.url,
                    'priority': req.priority,
                    'slot': slot,
                })

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values()) + len(self._buffer)

    def nbytes(self) -> int:
        """
        Memory taken by link vectors in requests stored in all queues
        and self.buffer.
        """
        return (sum(q.nbytes() for q in self.queues.values()) +
                sum(map(request_nbytes, self._buffer)))


def request_nbytes(request):
    if hasattr(request, 'meta'):
        return csr_nbytes(request.meta.get('link_vector'))
    else:
        return 0
