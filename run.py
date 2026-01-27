import torch
import argparse
import numpy as np
from typing import Dict, List
from main import main as run_single, build_arg_parser

FIXED_DATA_SEED = 20010920
TEST_METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def build_run_parser() -> argparse.ArgumentParser:
    """Extend the single-run parser with multi-seed options."""
    base_parser = build_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Run multiple seeds and report mean ± std for DIAL experiments",
        parents=[base_parser]
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='List of training/init seeds to run.'
    )
    return parser


def aggregate_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std for selected metrics across runs."""
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


def prepare_single_run_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    single_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != 'seeds'})
    single_args.random_seed = seed
    single_args.data_seed = FIXED_DATA_SEED
    return single_args


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

    test_results = []
    for seed in args.seeds:
        print("=" * 80)
        print(f"Running seed {seed} (data split seed fixed at {FIXED_DATA_SEED})")
        run_args = prepare_single_run_args(args, seed)
        result = run_single(run_args)
        test_metrics = result['test_final']
        test_results.append(test_metrics)
        print(
            "Test metrics: "
            f"Acc={test_metrics['accuracy']:.4f}, "
            f"Prec={test_metrics['precision']:.4f}, "
            f"Rec={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics['auc']:.4f}"
        )

    summary = aggregate_metrics(test_results)

    print("\n" + "=" * 80)
    print(f"Aggregated over {len(test_results)} runs (mean ± std, data seed {FIXED_DATA_SEED}):")
    for key in TEST_METRIC_KEYS:
        stats = summary.get(key)
        if not stats:
            continue
        print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
