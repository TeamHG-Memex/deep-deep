#!/usr/bin/env python
import argparse
from itertools import islice
import pickle

from eli5.sklearn.explain_weights import explain_weights
from eli5.formatters import format_as_text, format_as_html
import joblib
import json_lines
import numpy as np

from deepdeep.links import DictLinkExtractor
from deepdeep.explain import item_links, get_feature_names_scales


def main():
    parser = argparse.ArgumentParser(
        description='Export an explanation of deep-deep model '
                    'to html or pickle')
    arg = parser.add_argument
    arg('q_model', help='Path to Q.joblib (deep-deep link model)')
    arg('data', help='Path to jl.gz file in CDR format '
                     'to fit the hashing vectorizer')
    arg('--limit', type=int, default=1000,
        help='Limit number of documents for fitting hashing vectorizer')
    arg('--top', type=int, default=50, help='Top features (passed to eli5)')
    arg('--save-expl', help='Save pickled explanation')
    arg('--save-html', help='Save explanation in html')
    args = parser.parse_args()

    q_model = joblib.load(args.q_model)
    with json_lines.open(args.data) as items:
        if args.limit:
            items = islice(items, args.limit)

        print('Extracting links...')
        le = DictLinkExtractor()
        links = [
            link for item in items
            for link in item_links(le, item['url'], item['raw_content'])]
        print('Done.')
        assert not q_model.get('page_vectorizer'), 'TODO'
        all_features_names, coef_scale = get_feature_names_scales(
            q_model['link_vectorizer'], links)

        clf = q_model['Q'].clf_online
        expl = explain_weights(
            clf,
            feature_names=all_features_names,
            coef_scale=np.array(coef_scale),
            top=args.top)
        if args.save_expl:
            with open(args.save_expl, 'wb') as f:
                pickle.dump(expl, f)
            print('Pickled explanation saved to {}'.format(args.save_expl))
        if args.save_html:
            with open(args.save_html, 'wt') as f:
                f.write(format_as_html(expl))
            print('Explanation in html saved to {}'.format(args.save_html))
        if not args.save_expl and not args.save_html:
            print(format_as_text(expl))


if __name__ == '__main__':
    main()
