import argparse
from typing import Dict, List

import numpy as np
import torch

from baselines.run_baselines import run as run_single, build_arg_parser

FIXED_DATA_SEED = 20010920
TEST_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def build_run_parser() -> argparse.ArgumentParser:
    """Extend the baseline parser with multi-seed options."""
    base_parser = build_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Run multiple seeds for baselines and report mean ± std metrics",
        parents=[base_parser]
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='List of training/init seeds to run.'
    )
    return parser


def aggregate_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in TEST_METRIC_KEYS:
        values = np.array([res[key] for res in results], dtype=np.float64)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary[key] = {
            "mean": float(values.mean()),
            "std": std,
        }
    return summary


def main():
    parser = build_run_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    if args.data_seed != FIXED_DATA_SEED:
        print(f"[info] Overriding data_seed from {args.data_seed} to fixed {FIXED_DATA_SEED} for consistent splits.")
    args.data_seed = FIXED_DATA_SEED

    if not args.seeds:
        parser.error("Please provide at least one seed via --seeds.")

    metrics_by_model: Dict[str, List[Dict]] = {}

    for seed in args.seeds:
        print("=" * 80)
        print(f"Running seed {seed} (data split seed fixed at {FIXED_DATA_SEED})")
        args.random_seed = seed
        run_results = run_single(args)
        for model_name, result in run_results.items():
            metrics_by_model.setdefault(model_name, []).append(result['test_final'])
            test_metrics = result['test_final']
            print(
                f"[{model_name}] "
                f"Acc={test_metrics['accuracy']:.4f}, "
                f"Prec={test_metrics['precision']:.4f}, "
                f"Rec={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"AUC={test_metrics['auc']:.4f}"
            )

    print("\n" + "=" * 80)
    for model_name, metrics_list in metrics_by_model.items():
        summary = aggregate_metrics(metrics_list)
        print(f"{model_name} aggregated over {len(metrics_list)} runs (mean ± std, data seed {FIXED_DATA_SEED}):")
        for key in TEST_METRIC_KEYS:
            stats = summary.get(key)
            if not stats:
                continue
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()