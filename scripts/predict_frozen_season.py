"""Inference only, using hashed trusted packages in their original environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_model_package import load_research_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-root', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.queries)
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    readback = {'environment': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
                'input_sha256': hashlib.sha256(args.queries.read_bytes()).hexdigest(), 'models': {}}
    for kind, candidate in manifest['candidates'].items():
        directory = Path(candidate['remote_package']).resolve()
        if not directory.is_relative_to(args.trusted_root.resolve()):
            raise ValueError('Package outside explicitly trusted root')
        for name, expected in candidate['hashes'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen package hash mismatch')
        package = load_research_package(directory)
        probabilities = package.predict_proba(frame)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError('Invalid probabilities')
        rows = frame[['date', 'race_id', 'race_no', 'horse_id', 'horse_no', 'horse_name', 'field_size']].copy()
        rows['model'] = kind
        rows['model_probability'] = probabilities
        totals = rows.groupby('race_id').model_probability.sum()
        if not np.allclose(totals, 1.0, rtol=0, atol=1e-10):
            raise ValueError('Probabilities must sum to one within every race')
        path = args.output / f'{kind}.csv'
        rows.to_csv(path, index=False)
        readback['models'][kind] = {'package_id': package.manifest().package_id,
            'training_cutoff': package.feature_context.training_cutoff,
            'rows': len(rows), 'races': rows.race_id.nunique(),
            'probability_sum_min': float(totals.min()), 'probability_sum_max': float(totals.max()),
            'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({'completed': kind, **readback['models'][kind]}), flush=True)
    (args.output / 'readback.json').write_text(json.dumps(readback, indent=2) + '\n')


if __name__ == '__main__':
    main()
