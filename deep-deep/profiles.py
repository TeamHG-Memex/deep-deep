import argparse
import json
import logging
from pathlib import Path
import re
import subprocess
import traceback
from typing import Any, Dict, List
from urllib.parse import urlsplit

import attr
from deepdeep.utils import get_domain
import requests
from scrapy.http.response.text import TextResponse


@attr.s
class Site:
    url_pattern = attr.ib()
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
                    url_pattern=sdict['url'],
                    username=sdict['test_username_pos'],
                    status_code=sdict['status_code'],
                    match_type=sdict['match_type'],
                    match_expr=sdict['match_expr'],
                )
                domain = get_domain(site.url_pattern)
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
    before, after = site.url_pattern.rstrip('/').split('%s')
    pattern = '{}([^/?&]+){}'.format(re.escape(before), re.escape(after))
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


def check_sites(sites=None):
    if sites is None:
        sites = list(SITES.values())
    valid = []
    invalid = []
    request_failed = []
    for site in sites:
        url = site.url_pattern % site.username
        print(url)
        try:
            r = requests.get(url)
        except requests.RequestException:
            traceback.print_exc()
            request_failed.append(site)
        else:
            response = TextResponse(
                url=url, body=r.content, headers=dict(r.headers))
            usernames = list(extract_username(response))
            if usernames:
                print('valid')
                valid.append(site)
            else:
                print('invalid')
                invalid.append(site)
    print('{} valid, {} invalid, {} failed to load'
          .format(len(valid), len(invalid), len(request_failed)))
    return valid, invalid, request_failed


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


def crawl_args(site: Site, experiment_root: Path,
               use_page_urls: bool=True, timeout: int=0, dry_run: bool=False):
    parsed = urlsplit(site.url_pattern)
    profile_url = site.url_pattern % site.username
    root_url = '{}://{}'.format(parsed.scheme, parsed.netloc)
    domain = get_domain(root_url)
    if '%' in root_url:
        root_url = '{}://{}'.format(parsed.scheme, domain)
        assert '%s' not in root_url
    root = (experiment_root / domain).absolute()
    if root.exists():
        assert not any(root.iterdir())
    elif not dry_run:
        root.mkdir(parents=True)
    seeds_path = experiment_root.joinpath('{}-seeds.txt'.format(domain))
    if not dry_run:
        seeds_path.write_text('\n'.join([root_url, profile_url, '']))
    return [
        'scrapy', 'crawl', 'extraction',
        '-a', 'extractor=profiles:extract_username',
        '-a', 'seeds_url={}'.format(seeds_path.absolute()),
        '-a', 'checkpoint_path={}'.format(root),
        '-a', 'use_page_urls={}'.format(int(use_page_urls)),
        '-a', 'checkpoint_latest=1',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'LOG_FILE={}/spider.log'.format(root),
        '-s', 'CLOSESPIDER_ITEMCOUNT=0',  # no limit
        '-s', 'CLOSESPIDER_TIMEOUT={}'.format(timeout),
        '-o', 'gzip:{}/{}.jl'.format(root, domain),
    ]


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('checkpoint_root', type=Path)
    arg('--max-workers', type=int, default=8)
    arg('--limit', type=int, default=0)
    arg('--offset', type=int, default=0)
    arg('--use-page-urls', type=int, default=1)
    arg('--timeout', type=int, default=86400)
    arg('--dry-run', action='store_true', help='do not run or start anything')
    args = parser.parse_args()
    sites = sorted(SITES.values())
    if args.offset:
        sites = sites[args.offset:]
    if args.limit:
        sites = sites[:args.limit]
    processes = {}  # type: Dict[Site, subprocess.Popen]
    for i, site in enumerate(sites, 1):
        while len(processes) >= args.max_workers:
            for s, p in list(processes.items()):
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    logging.info('Finished: {}'.format(s))
                    del processes[s]
        logging.info('[{}] Starting process for {}'.format(i, site))
        p_args = crawl_args(site, args.checkpoint_root,
                            use_page_urls=args.use_page_urls,
                            timeout=args.timeout,
                            dry_run=args.dry_run)
        logging.info(' '.join(p_args))
        if not args.dry_run:
            process = subprocess.Popen(p_args, stderr=subprocess.DEVNULL)
            processes[site] = process
    logging.info('Waiting for the last processes to finish')
    for s, p in processes.items():
        p.wait()
        logging.info('Finished: {}'.format(s))


if __name__ == '__main__':
    main()
