import argparse
from itertools import islice
from pathlib import Path
import re
from typing import Dict
from urllib.parse import urlsplit

from scrapy.http.response.text import TextResponse
from deepdeep.utils import get_domain


def parse_rules(path: Path) -> Dict[str, Dict[str, str]]:
    rules = {}
    for pattern, username, rule_kind, rule_value in _rules_reader(path):
        domain = get_domain(pattern)
        rules[domain] = {
            'url_pattern': pattern,
            rule_kind: rule_value,
        }
    return rules


def _rules_reader(path: Path):
    with path.open() as f:
        for line in f:
            yield line.strip().split(', ', 3)


RULES_PATH = Path(__file__).parent / 'profiles.csv'  # type: Path
RULES = parse_rules(RULES_PATH)


def extract_username(response: TextResponse):
    if not hasattr(response, 'text'):
        return
    url = response.url
    domain = get_domain(url)
    rule = RULES.get(domain)
    if not rule:
        return
    pattern = rule['url_pattern'].replace('%s', '([^/?]+)').rstrip('/')
    match = re.match(pattern, url)
    if match:
        username = match.groups()[0]
        if 'css' in rule:
            if not response.css(rule['css']):
                return
        elif 'xpath' in rule:
            if not response.xpath(rule['xpath']):
                return
        else:
            raise ValueError('Unexpected rule: no css or xpath set')
        yield username, None


def merge_profiles():
    # FIXME - copied from ipython
    import json, csv, gzip
    from deepdeep.utils import get_domain
    from collections import defaultdict

    with open('all_items.jl') as f:
        all_items = []
        invalid = 0
        for line in f:
            try:
                all_items.append(json.loads(line))
            except Exception:
                invalid += 1
        print(invalid, 'invalid')

    with gzip.open('crawled_profiles.csv.gz', 'wt') as outf:
        writer = csv.writer(outf)
        by_domain = defaultdict(set)
        for item in all_items:
            url = item['url']
            # split by ? fixed a bug that is also fixed in extract_username
            name = item['key'].split('?')[0]
            domain = get_domain(url)
            if name in by_domain[domain]:
                continue
            by_domain[domain].add(name)
            writer.writerow([url, name])


def make_script(experiment_root: Path, limit: int, offset: int,
                use_page_urls: bool=True):
    print('set -v')
    for pattern, username, _, _ in islice(
            _rules_reader(RULES_PATH), offset, offset + limit):
        parsed = urlsplit(pattern)
        profile_url = pattern % username
        root_url = '{}://{}'.format(parsed.scheme, parsed.netloc)
        domain = get_domain(root_url)
        root = experiment_root / domain
        if root.exists():
            assert not any(root.iterdir())
        else:
            root.mkdir(parents=True)
        seeds_path = experiment_root.joinpath('{}-seeds.txt'.format(domain))
        seeds_path.write_text('\n'.join([root_url, profile_url, '']))
        print(
            'scrapy crawl extraction -a extractor=profiles:extract_username '
            "-a seeds_url='{seeds_url}' "
            '-a checkpoint_path={root} '
            '-a use_page_urls={use_page_urls} '
            '-s LOG_LEVEL=INFO -s LOG_FILE={root}/spider.log '
            '-s CLOSESPIDER_ITEMCOUNT=0 '  # no limit
            '-o {root}/{domain}.jl &'
            .format(seeds_url=seeds_path.absolute(),
                    domain=domain,
                    root=root,
                    use_page_urls=int(use_page_urls),
                    ))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('checkpoint_root', type=Path)
    arg('--limit', type=int, default=16)
    arg('--offset', type=int, default=0)
    arg('--use-page-urls', type=int, default=1)
    args = parser.parse_args()
    make_script(args.checkpoint_root, limit=args.limit, offset=args.offset,
                use_page_urls=args.use_page_urls)


if __name__ == '__main__':
    main()
