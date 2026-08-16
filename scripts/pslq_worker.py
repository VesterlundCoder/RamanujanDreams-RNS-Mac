#!/usr/bin/env python3
import argparse
import csv
import json
from mpmath import mp, pslq, matrix


def load_constants(path):
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return [(k, mp.mpf(v)) for k, v in obj.items()]


def main():
    ap = argparse.ArgumentParser(description='PSLQ scan of converged matrix-element ratios.')
    ap.add_argument('ratios_tsv')
    ap.add_argument('constants_json')
    ap.add_argument('out_tsv')
    ap.add_argument('--dps', type=int, default=160)
    ap.add_argument('--maxcoeff', type=int, default=10**8)
    args = ap.parse_args()
    mp.dps = args.dps
    consts = load_constants(args.constants_json)

    with open(args.ratios_tsv, newline='', encoding='utf-8') as fi, open(args.out_tsv, 'w', newline='', encoding='utf-8') as fo:
        rd = csv.DictReader(fi, delimiter='\t')
        limit_cols = [c for c in rd.fieldnames if c.startswith('limit_') and c.endswith('_200d')]
        if not limit_cols:
            raise RuntimeError('No limit_*_200d column found')
        final_limit_col = limit_cols[-1]

        wr = csv.writer(fo, delimiter='\t')
        wr.writerow([
            'trajectory_id', 'num_idx', 'num_row', 'num_col',
            'den_idx', 'den_row', 'den_col', 'constant', 'relation',
            'delta_rel_last'
        ])

        for row in rd:
            x = mp.mpf(row[final_limit_col])
            for name, c in consts:
                # First-stage basis: a*x + b*c + d = 0.
                # Extend with structured bases (pi^2, zeta(3), Gamma-values, etc.) as needed.
                rel = pslq(
                    matrix([x, c, mp.mpf(1)]),
                    tol=mp.mpf(10) ** (-(args.dps-20)),
                    maxcoeff=args.maxcoeff,
                    maxsteps=10000,
                )
                if rel:
                    wr.writerow([
                        row['trajectory_id'], row['num_idx'], row['num_row'], row['num_col'],
                        row['den_idx'], row['den_row'], row['den_col'], name,
                        ','.join(map(str, rel)), row['delta_rel_last']
                    ])
                    break


if __name__ == '__main__':
    main()
