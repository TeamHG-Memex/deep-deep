# -*- coding: utf-8 -*-
import scrapy  # type: ignore

from deepdeep.queues import RequestsPriorityQueue


def test_request_priority_queue():
    q = RequestsPriorityQueue(fifo=True)
    q.push(scrapy.Request('http://example.com/1', priority=1))
    q.push(scrapy.Request('http://example.com/1/1', priority=1))
    q.push(scrapy.Request('http://example.com/-1', priority=-1))
    q.push(scrapy.Request('http://example.com/2', priority=2))
    q.push(scrapy.Request('http://example.com/0', priority=0))

    assert -q.entries[0][0] == 2
    assert len(q) == 5

    assert q.pop().url == "http://example.com/2"
    assert len(q) == 4
    assert q.pop().url == "http://example.com/1"
    assert q.pop().url == "http://example.com/1/1"
    assert q.pop().url == "http://example.com/0"
    assert q.pop().url == "http://example.com/-1"
    assert len(q) == 0


def test_rpq_pop_random():
    requests = [
        scrapy.Request('http://example.com/1', priority=1),
        scrapy.Request('http://example.com/2', priority=2),
        scrapy.Request('http://example.com/3', priority=3),
    ]
    q = RequestsPriorityQueue(fifo=True)
    for req in requests:
        q.push(req)

    req1 = q.pop_random()
    req2 = q.pop_random()
    req3 = q.pop_random()

    assert {req1.url, req2.url, req3.url} == {r.url for r in requests}
    assert q.pop_random() is None
