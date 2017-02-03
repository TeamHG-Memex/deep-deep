import argparse
from itertools import islice
import json
from pathlib import Path
import re
from typing import Any, Dict, List
from urllib.parse import urlsplit

import attr
from deepdeep.utils import get_domain
import requests
from scrapy.http.response.text import TextResponse


@attr.s
class Site:
    url = attr.ib()
    username = attr.ib()
    status_code = attr.ib()
    match_type = attr.ib()
    match_expr = attr.ib()


def parse_sites(path: Path) -> Dict[str, Site]:
    sites = {}
    with path.open() as f:
        for sdict in json.load(f):
            if sdict['valid']:
                site = Site(
                    url=sdict['url'],
                    username=sdict['test_username_pos'],
                    status_code=sdict['status_code'],
                    match_type=sdict['match_type'],
                    match_expr=sdict['match_expr'],
                )
                domain = get_domain(site.url)
                sites[domain] = site
    return sites


SITES_PATH = Path(__file__).parent / 'profiles.json'  # type: Path
SITES = parse_sites(SITES_PATH)


def extract_username(response: TextResponse):
    if not hasattr(response, 'text'):
        return
    url = response.url
    domain = get_domain(url)
    site = SITES.get(domain)
    if not site:
        return
    pattern = site.url_pattern.replace('%s', '([^/?]+)').rstrip('/')
    match = re.match(pattern, url)
    if match:
        username = match.groups()[0]
        if _check_response(site, response):
            yield username, {'html': response.text}


def _check_response(site: Site, response: TextResponse) -> bool:
    """
    Parse response and test against site criteria to determine
    whether username exists.
    """
    sel = response.selector
    status_ok = match_ok = True

    if site.status_code is not None:
        status_ok = site.status_code == response.status

    if site.match_expr is not None:
        if site.match_type == 'css':
            match_ok = len(sel.css(site.match_expr)) > 0
        elif site.match_type == 'text':
            text_nodes = sel.css(':not(script):not(style)::text').extract()
            text = []
            for text_node in text_nodes:
                stripped = text_node.strip()
                if stripped:
                    text.append(stripped)
            match_ok = site.match_expr in ' '.join(text)
        elif site.match_type == 'xpath':
            match_ok = len(sel.xpath(site.match_expr)) > 0
        else:
            raise ValueError('Unknown match_type: {}'.format(site.match_type))

    return status_ok and match_ok


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


def download_sites(api_url, username, password) -> List[Dict[str, Any]]:
    auth = requests.post('{}/api/authentication/'.format(api_url),
                         json={'email': username, 'password': password}).json()
    headers = {'X-Auth': auth['token']}
    site_url = '{}/api/site'.format(api_url)
    all_sites = []
    page = 1
    while True:
        sites = requests.get(site_url, headers=headers,
                             params={'rpp': 100, 'page': page}).json()
        if not sites['sites']:
            break
        all_sites.extend(sites['sites'])
        page += 1
    return all_sites


def make_script(experiment_root: Path, limit: int, offset: int,
                use_page_urls: bool=True):
    print('set -v')
    for _, site in islice(
            sorted(SITES.items()), offset, offset + limit):
        parsed = urlsplit(site.url)
        profile_url = site.url % site.username
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
            '-a checkpoint_latest=1 '
            '-s LOG_LEVEL=INFO -s LOG_FILE={root}/spider.log '
            '-s CLOSESPIDER_ITEMCOUNT=0 '  # no limit
            '-o gzip:{root}/{domain}.jl &'
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
