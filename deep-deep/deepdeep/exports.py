# -*- coding: utf-8 -*-
import os
import gzip

from zope.interface import Interface, implementer  # type: ignore
from w3lib.url import file_uri_to_path  # type: ignore
from scrapy.extensions.feedexport import IFeedStorage  # type: ignore


@implementer(IFeedStorage)
class GzipFileFeedStorage(object):
    """
    Storage which exports data to a gzipped file.
    To use it, add

    ::

        FEED_STORAGES = {
            'gzip': 'deepdeep.exports.GzipFileFeedStorage',
        }

    to settings.py and then run scrapy crawl like this::

        scrapy crawl foo -o gzip:/path/to/items.jl

    The command above will create ``/path/to/items.jl.gz`` file
    (.gz extension is added automatically).

    Other export formats are also supported, but it is recommended to use .jl.
    If a spider is killed then gz archive may be partially broken.
    In this case it user should read the broken archive line-by-line and stop
    on gzip decoding errors, discarding the tail. It works OK with .jl exports.
    """
    COMPRESS_LEVEL = 4

    def __init__(self, uri):
        self.path = file_uri_to_path(uri) + ".gz"

    def open(self, spider):
        dirname = os.path.dirname(self.path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        return gzip.open(self.path, 'ab', compresslevel=self.COMPRESS_LEVEL)

    def store(self, file):
        file.close()
