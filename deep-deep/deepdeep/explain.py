from typing import Dict, List, Tuple, Union

from scrapy.http.response.text import TextResponse
from eli5.base import WeightedSpans, FeatureWeights
from eli5.sklearn import explain_prediction_sklearn
from eli5.sklearn.text import get_weighted_spans
from eli5.sklearn.unhashing import InvertableHashingVectorizer
from eli5.sklearn.utils import FeatureNames
from eli5.formatters import FormattedFeatureName
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from deepdeep.links import DictLinkExtractor


def get_feature_names_scales(
        vectorizer: FeatureUnion, links: List[Dict], with_scales: bool=True)\
        -> Union[Tuple[FeatureNames, np.ndarray], FeatureNames]:
    """ Assemble feature names and coef scales (if with_scales is True)
    from individual vectorizers, fitting InvertableHashingVectorizer on given links.
    """
    all_features_names = {}
    coef_scales = []
    n_features = 0
    for name, vec in vectorizer.transformer_list:
        if isinstance(vec, HashingVectorizer):
            ivec = InvertableHashingVectorizer(vec)
            ivec.fit(links)
            feature_names = ivec.get_feature_names(always_signed=False)
            all_features_names.update(
                (n_features + idx, [dict(n, vec=vec) for n in name])
                for idx, name in feature_names.feature_names.items())
            if with_scales:
                coef_scales.append(ivec.column_signs_)
            n_features += feature_names.n_features
        elif isinstance(vec, FunctionTransformer):
            all_features_names[n_features] = vec.func.__name__
            n_features += 1
            if with_scales:
                coef_scales.append([1.])
    feature_names = FeatureNames(
        all_features_names, n_features=n_features, unkn_template='FEATURE[%d]')
    if with_scales:
        coef_scale = np.empty([sum(map(len, coef_scales))])
        start_idx = 0
        for arr in coef_scales:
            end_idx = start_idx + len(arr)
            coef_scales[start_idx: end_idx] = arr
            start_idx = end_idx
        return feature_names, coef_scale
    else:
        return feature_names


def links_explanations(clf, vec: FeatureUnion, links: List[Dict]) -> List[Dict]:
    all_expl = []
    feature_names = get_feature_names_scales(vec, links, with_scales=False)
    for link in links:
        expl = explain_prediction_sklearn(
            clf, link, vec, feature_names=feature_names, top=1000)
        target_expl = expl.targets[0]
        target_expl.weighted_spans = combined_weighted_spans(
            link, target_expl, vec)
        all_expl.append((target_expl.score, link, expl))
    return all_expl


def combined_weighted_spans(link, target_expl, vectorizer):
    ws_combined = WeightedSpans(
        analyzer='',
        document='',
        weighted_spans=[],
        other=FeatureWeights(pos=[], neg=[]))
    for vec_idx in [0, 2]:
        vec = vectorizer.transformer_list[vec_idx][1]
        vec_name = vec.preprocessor.__name__
        get_weights = lambda weights: [
            (name if isinstance(name, str) else
             [n for n in name if n['vec'] == vec],
             coef) for name, coef in weights]
        feature_weights = FeatureWeights(
            pos=get_weights(target_expl.feature_weights.pos),
            neg=get_weights(target_expl.feature_weights.neg))
        ws = get_weighted_spans(
            link, vec=vec, feature_weights=feature_weights)
        ws_combined.analyzer = ws.analyzer
        if ws_combined.document:
            ws_combined.document += ' | '
        s0 = len(ws_combined.document)
        ws_combined.document += ws.document
        shifted_spans = [
            (feature, [(s0 + s, s0 + e) for s, e in spans], weight)
            for feature, spans, weight in ws.weighted_spans]
        ws_combined.weighted_spans.extend(shifted_spans)
        for combined_other_weights, other_weights in [
                (ws_combined.other.pos, ws.other.pos),
                (ws_combined.other.neg, ws.other.neg)]:
            # Ignore pos_remaining and neg_remaining, top should be large enough
            other_features = {f for f, _ in combined_other_weights}
            for f, w in other_weights:
                if str(f) == 'Highlighted in text (sum)':
                    f = FormattedFeatureName('{}, {}'.format(f, vec_name))
                    combined_other_weights.append((f, w))
                elif f not in other_features:
                    combined_other_weights.append((f, w))
    found_features = {f for f, _, _ in ws_combined.weighted_spans}
    for other_weights in [ws_combined.other.pos, ws_combined.other.neg]:
        other_weights[:] = sorted(
            ((f, w) for f, w in other_weights if f not in found_features),
            key=lambda x: abs(x[1]), reverse=True)
    return ws_combined


def extract_links(le: DictLinkExtractor, response: TextResponse) -> List[Dict]:
    return list(le.iter_link_dicts(
        response=response,
        limit_by_domain=False,
        deduplicate=False,
        deduplicate_local=True,
    ))


def item_links(le: DictLinkExtractor, url: str, raw_content: str) -> List[Dict]:
    return extract_links(le, TextResponse(
        url=url,
        body=raw_content,
        encoding='utf8',
    ))
