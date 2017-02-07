import argparse
import gzip
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root', help='root directory (jl files are in root/*/*.jl.gz)')
    arg('output', help='output folder')
    arg('--n', type=int, default=10,
        help='number of profiles to sample for each site')
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(exist_ok=True, parents=True)

    for jl_path in Path(args.root).glob('*/*.jl.gz'):
        print('Reading {} ...'.format(jl_path))
        n_lines = sum(1 for _ in gzip_lines(jl_path))
        if n_lines == 0:
            continue
        sample_size = min(n_lines, args.n)
        lines = set(random.sample(range(n_lines), sample_size))
        sampled = 0
        with gzip.open(str(out_path.joinpath(jl_path.name)), 'wt') as outf:
            for idx, line in enumerate(gzip_lines(jl_path)):
                if idx in lines:
                    sampled += 1
                    outf.write(line)
        assert sampled == sample_size


def gzip_lines(path: Path):
    try:
        with gzip.open(str(path), 'rt') as f:
            yield from f
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    main()
