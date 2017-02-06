import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path

import json_lines

from deepdeep.utils import get_domain


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='root directory (jl files are in root/*/*.jl.gz)')
    arg('output', help='.csv.gz forma')
    args = parser.parse_args()

    by_domain = defaultdict(int)
    with gzip.open(args.output, 'wt') as outf:
        writer = csv.writer(outf)
        for jl_path in Path(args.root).glob('*/*.jl.gz'):
            with json_lines.open(str(jl_path), broken=True) as f:
                for item in f:
                    url = item['url']
                    name = item['key']
                    domain = get_domain(url)
                    by_domain[domain] += 1
                    writer.writerow([domain, url, name])
    print('Stat by domain:')
    for domain, n_profiles in sorted(by_domain.items(),
                                     key=lambda x: x[1], reverse=True):
        print('{:<30} {:>8,}'.format(domain, n_profiles))


if __name__ == '__main__':
    main()
