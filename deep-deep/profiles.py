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
    pattern = rule['url_pattern'].replace('%s', '([^/]+)').rstrip('/')
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


def make_script(top):
    print('set -v')
    for pattern, _, _, _ in islice(_rules_reader(RULES_PATH), top):
        parsed = urlsplit(pattern)
        seed_url = '{}://{}'.format(parsed.scheme, parsed.netloc)
        parent = Path('profiles') / get_domain(seed_url)
        i = 1
        while True:
            root = parent / str(i)
            if not root.exists():
                root.mkdir(parents=True)
                break
            elif not any(root.iterdir()):
                break
            i += 1
        print(
            'scrapy crawl extraction -a extractor=profiles:extract_username '
            '-a seed_url={seed_url} '
            '-a checkpoint_path={root} '
            '-s LOG_LEVEL=INFO -s LOG_FILE={root}/spider.log '
            '-o {root}/items.jl &'
            .format(seed_url=seed_url, root=root))


if __name__ == '__main__':
    make_script(top=16)
