#!/usr/bin/env python
import argparse
from itertools import islice
import multiprocessing
from functools import partial
from pathlib import Path

from eli5 import explain_prediction
from eli5.formatters import format_as_html, format_html_styles, fields
import joblib
import json_lines

from deepdeep.links import DictLinkExtractor, raw_html_links


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


def links_expls(model, le, item):
    explanations = []
    for link in raw_html_links(le, item['url'], item['raw_content']):
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
