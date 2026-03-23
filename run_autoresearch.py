#!/usr/bin/env python3
"""
Autoresearch CLI — Define MAPE as objective, let it rip.

Usage:
    python run_autoresearch.py                    # 50 iterations, default settings
    python run_autoresearch.py --iterations 200   # More iterations
    python run_autoresearch.py --iterations 500 --test-days 60 --seed 123

Inspired by Karpathy's autoresearch: define an objective function,
then systematically search over models and hyperparameters to minimize it.
"""

import argparse
import sys

from autoresearch.data import generate_retail_data, train_test_split_temporal
from autoresearch.engine import AutoResearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch: automated MAPE minimization for demand forecasting"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=50,
        help="Number of experiment iterations (default: 50)"
    )
    parser.add_argument(
        "--stores", type=int, default=3,
        help="Number of stores in synthetic data (default: 3)"
    )
    parser.add_argument(
        "--products", type=int, default=5,
        help="Number of products in synthetic data (default: 5)"
    )
    parser.add_argument(
        "--test-days", type=int, default=90,
        help="Number of days for the test set (default: 90)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-iteration output"
    )
    args = parser.parse_args()

    # Generate data
    print("Generating synthetic retail data...")
    df = generate_retail_data(
        n_stores=args.stores,
        n_products=args.products,
        seed=args.seed,
    )
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Stores: {df['store_id'].nunique()} | Products: {df['product_id'].nunique()}")

    # Split
    train, test = train_test_split_temporal(df, test_days=args.test_days)
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")

    # Run autoresearch
    engine = AutoResearchEngine(
        train=train,
        test=test,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    engine.run(n_iterations=args.iterations, verbose=not args.quiet)

    # Print results
    engine.print_leaderboard(top_n=15)

    print("\nStrategy Summary (best MAPE per strategy):")
    summary = engine.get_strategy_summary()
    if len(summary) > 0:
        for _, row in summary.iterrows():
            print(
                f"  {row['strategy']:<22} "
                f"best={row['best_mape']:.2f}%  "
                f"median={row['median_mape']:.2f}%  "
                f"trials={int(row['n_trials'])}"
            )

    print(f"\nResults saved to: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
