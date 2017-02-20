#!/usr/bin/env python
import argparse
from itertools import islice
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List
import warnings

from eli5 import explain_prediction
from eli5.formatters import format_as_html, format_html_styles, fields
import joblib
import json_lines
from scrapy.http.response.text import TextResponse

from deepdeep.links import DictLinkExtractor


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('q_model', help='Path to Q.joblib (deep-deep link model)')
    arg('data', help='Path to jl.gz file in CDR format')
    arg('output_folder', help='Where to put html output files N.html')
    arg('--offset', type=int, default=0, help='0-based start index in data')
    arg('--limit', type=int, default=100, help='Number of documents to analyze')
    args = parser.parse_args()

    output_path = Path(args.output_folder)
    output_path.mkdir(exist_ok=True)
    q_model = joblib.load(args.q_model)
    assert not q_model.get('page_vectorizer'), 'TODO'
    le = DictLinkExtractor()
    styles = format_html_styles()

    with json_lines.open(args.data, broken=True) as items:
        items = islice(items, args.offset, None)
        if args.limit:
            items = islice(items, args.limit)
        with multiprocessing.Pool() as pool:
            for idx, expls in enumerate(pool.imap(
                    partial(links_expls, q_model, le), items)):
                expls.sort(reverse=True)
                (output_path.joinpath('{}.html'.format(idx + args.offset))
                 .write_text(styles + '\n'.join(expl for _, expl in expls)))


def item_links(le: DictLinkExtractor, url: str, raw_content: str) -> List[Dict]:
    return extract_links(
        le, TextResponse(url=url, body=raw_content, encoding='utf8'))


def extract_links(le: DictLinkExtractor, response: TextResponse) -> List[Dict]:
    return list(le.iter_link_dicts(
        response=response,
        limit_by_domain=False,
        deduplicate=False,
        deduplicate_local=True,
    ))


def links_expls(model, le, item):
    warnings.filterwarnings('ignore')  # FIXME - fit it!
    explanations = []
    for link in item_links(le, item['url'], item['raw_content']):
        expl = explain_prediction(
            model['Q'].clf_online,
            doc=link,
            vec=model['link_vectorizer'])
        explanations.append((
            expl.targets[0].score,
            format_as_html(
                 expl,
                 include_styles=False,
                 force_weights=False,
                 show=fields.WEIGHTS
            )))
    print(item['url'], len(explanations))
    return explanations


if __name__ == '__main__':
    main()
