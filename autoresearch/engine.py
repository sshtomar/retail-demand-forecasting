"""
Autoresearch Engine — the core loop.

This is the Karpathy-inspired "let it rip" loop:
1. Maintain a leaderboard of (strategy, params, MAPE) tuples
2. Each iteration: pick a strategy, sample or mutate params, train, evaluate
3. If MAPE improves, record it as a new best
4. Periodically log progress and save results

The search uses a mix of:
- Random search (explore new strategy-param combos)
- Mutation (take a top-performing config and tweak it)
- Strategy rotation (ensure all strategies get tried)
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .objective import evaluate, EvalResult
from .strategies.base import Strategy
from .strategies.naive import NaiveLastValue, MovingAverage, SeasonalNaive
from .strategies.linear import LinearStrategy
from .strategies.tree import (
    RandomForestStrategy,
    GradientBoostingStrategy,
    LightGBMStrategy,
)

# Registry of all available strategies
STRATEGY_REGISTRY: list[type[Strategy]] = [
    NaiveLastValue,
    MovingAverage,
    SeasonalNaive,
    LinearStrategy,
    RandomForestStrategy,
    GradientBoostingStrategy,
    LightGBMStrategy,
]


def _get_strategy_by_name(name: str) -> type[Strategy]:
    for s in STRATEGY_REGISTRY:
        if s.name == name:
            return s
    raise ValueError(f"Unknown strategy: {name}")


class Experiment:
    """A single experiment result."""

    def __init__(
        self,
        strategy_name: str,
        params: dict[str, Any],
        result: EvalResult,
        duration_s: float,
        timestamp: str,
        error: str | None = None,
    ):
        self.strategy_name = strategy_name
        self.params = params
        self.result = result
        self.duration_s = duration_s
        self.timestamp = timestamp
        self.error = error

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "params": self.params,
            "metrics": self.result.to_dict() if self.result else None,
            "mape": self.result.mape if self.result else float("inf"),
            "duration_s": round(self.duration_s, 2),
            "timestamp": self.timestamp,
            "error": self.error,
        }


class AutoResearchEngine:
    """
    The main autoresearch loop.

    Usage:
        engine = AutoResearchEngine(train_df, test_df)
        engine.run(n_iterations=100)
        engine.print_leaderboard()
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str = "results",
        seed: int = 42,
    ):
        self.train = train
        self.test = test
        self.y_true = test["demand"].values
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.RandomState(seed)
        self.experiments: list[Experiment] = []
        self.best_mape = float("inf")
        self.best_experiment: Experiment | None = None
        self.iteration = 0

    def _pick_strategy_and_params(self) -> tuple[type[Strategy], dict[str, Any]]:
        """
        Decide what to try next. Uses a mix of:
        - Pure random (50% of time): pick random strategy + random params
        - Mutation (30%): take a top experiment and perturb its params
        - Exploitation (20%): re-run the best strategy with slightly tweaked params
        """
        roll = self.rng.random()

        if roll < 0.5 or len(self.experiments) < len(STRATEGY_REGISTRY):
            # Random exploration
            strategy_cls = STRATEGY_REGISTRY[self.rng.randint(len(STRATEGY_REGISTRY))]
            params = strategy_cls.sample_params(self.rng)
            return strategy_cls, params

        elif roll < 0.8 and len(self.experiments) >= 5:
            # Mutation: take one of the top 5 experiments and mutate
            sorted_exps = sorted(
                [e for e in self.experiments if e.result is not None],
                key=lambda e: e.result.mape,
            )
            donor = sorted_exps[self.rng.randint(min(5, len(sorted_exps)))]
            strategy_cls = _get_strategy_by_name(donor.strategy_name)
            params = self._mutate_params(strategy_cls, donor.params)
            return strategy_cls, params

        else:
            # Exploitation: use the best strategy, small mutation
            if self.best_experiment is None:
                strategy_cls = STRATEGY_REGISTRY[self.rng.randint(len(STRATEGY_REGISTRY))]
                return strategy_cls, strategy_cls.sample_params(self.rng)
            strategy_cls = _get_strategy_by_name(self.best_experiment.strategy_name)
            params = self._mutate_params(
                strategy_cls, self.best_experiment.params, scale=0.1
            )
            return strategy_cls, params

    def _mutate_params(
        self,
        strategy_cls: type[Strategy],
        base_params: dict[str, Any],
        scale: float = 0.3,
    ) -> dict[str, Any]:
        """Perturb parameters around a base configuration."""
        space = strategy_cls.search_space()
        params = dict(base_params)

        for name, spec in space.items():
            if self.rng.random() > 0.5:
                continue  # Only mutate ~half the params

            if spec["type"] == "int":
                delta = max(1, int((spec["high"] - spec["low"]) * scale))
                current = params.get(name, (spec["low"] + spec["high"]) // 2)
                new_val = current + self.rng.randint(-delta, delta + 1)
                params[name] = int(np.clip(new_val, spec["low"], spec["high"]))

            elif spec["type"] == "float":
                current = params.get(name, (spec["low"] + spec["high"]) / 2)
                if spec.get("log"):
                    log_val = np.log(current)
                    log_range = np.log(spec["high"]) - np.log(spec["low"])
                    log_val += self.rng.normal(0, log_range * scale)
                    new_val = np.exp(log_val)
                else:
                    val_range = spec["high"] - spec["low"]
                    new_val = current + self.rng.normal(0, val_range * scale)
                params[name] = float(np.clip(new_val, spec["low"], spec["high"]))

            elif spec["type"] == "categorical":
                if self.rng.random() < scale:
                    params[name] = spec["choices"][self.rng.randint(len(spec["choices"]))]

        return params

    def _run_one(
        self, strategy_cls: type[Strategy], params: dict[str, Any]
    ) -> Experiment:
        """Run a single experiment: instantiate, fit, predict, evaluate."""
        self.iteration += 1
        timestamp = datetime.now().isoformat()
        t0 = time.time()

        try:
            strategy = strategy_cls(params)
            strategy.fit(self.train)
            y_pred = strategy.predict(self.test)
            result = evaluate(self.y_true, y_pred)
            duration = time.time() - t0

            exp = Experiment(
                strategy_name=strategy_cls.name,
                params=params,
                result=result,
                duration_s=duration,
                timestamp=timestamp,
            )
        except Exception as e:
            duration = time.time() - t0
            exp = Experiment(
                strategy_name=strategy_cls.name,
                params=params,
                result=EvalResult(
                    mape=float("inf"), rmse=float("inf"), mae=float("inf"),
                    smape=float("inf"), median_ape=float("inf"),
                    p90_ape=float("inf"), n_samples=0,
                ),
                duration_s=duration,
                timestamp=timestamp,
                error=str(e),
            )

        return exp

    def run(self, n_iterations: int = 100, verbose: bool = True) -> None:
        """Run the autoresearch loop for n_iterations."""
        print(f"\n{'='*70}")
        print(f"  AUTORESEARCH ENGINE — MAPE Minimization")
        print(f"  Strategies: {len(STRATEGY_REGISTRY)} | Iterations: {n_iterations}")
        print(f"  Train: {len(self.train):,} rows | Test: {len(self.test):,} rows")
        print(f"{'='*70}\n")

        for i in range(n_iterations):
            strategy_cls, params = self._pick_strategy_and_params()
            exp = self._run_one(strategy_cls, params)
            self.experiments.append(exp)

            is_new_best = False
            if exp.result and exp.result.mape < self.best_mape:
                self.best_mape = exp.result.mape
                self.best_experiment = exp
                is_new_best = True

            if verbose:
                mape_str = f"{exp.result.mape:.2f}%" if exp.result else "FAILED"
                marker = " *** NEW BEST ***" if is_new_best else ""
                status = "ERR" if exp.error else "OK "
                print(
                    f"  [{i+1:>4}/{n_iterations}] {status} "
                    f"{exp.strategy_name:<22} "
                    f"MAPE={mape_str:<12} "
                    f"({exp.duration_s:.1f}s)"
                    f"{marker}"
                )

            # Periodic checkpoint
            if (i + 1) % 10 == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        if verbose:
            print(f"\n{'='*70}")
            print(f"  DONE. Best MAPE: {self.best_mape:.2f}%")
            print(f"  Strategy: {self.best_experiment.strategy_name}")
            print(f"  Params: {json.dumps(self.best_experiment.params, indent=2, default=str)}")
            print(f"{'='*70}\n")

    def get_leaderboard(self, top_n: int = 20) -> pd.DataFrame:
        """Return a leaderboard DataFrame sorted by MAPE."""
        rows = []
        for exp in self.experiments:
            if exp.result and exp.error is None:
                row = {"strategy": exp.strategy_name, **exp.result.to_dict()}
                row["duration_s"] = round(exp.duration_s, 2)
                # Add key params as string
                row["params"] = json.dumps(exp.params, default=str)
                rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        return df.sort_values("mape").head(top_n).reset_index(drop=True)

    def print_leaderboard(self, top_n: int = 20) -> None:
        """Print a formatted leaderboard."""
        lb = self.get_leaderboard(top_n)
        if len(lb) == 0:
            print("No successful experiments yet.")
            return

        print(f"\n{'='*90}")
        print(f"  LEADERBOARD (Top {min(top_n, len(lb))} by MAPE)")
        print(f"{'='*90}")
        print(f"  {'Rank':<5} {'Strategy':<22} {'MAPE':>8} {'RMSE':>10} "
              f"{'MAE':>10} {'sMAPE':>8} {'P90_APE':>8} {'Time':>6}")
        print(f"  {'-'*5} {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

        for idx, row in lb.iterrows():
            print(
                f"  {idx+1:<5} {row['strategy']:<22} "
                f"{row['mape']:>7.2f}% {row['rmse']:>10.2f} "
                f"{row['mae']:>10.2f} {row['smape']:>7.2f}% "
                f"{row['p90_ape']:>7.2f}% {row['duration_s']:>5.1f}s"
            )
        print()

    def get_strategy_summary(self) -> pd.DataFrame:
        """Summarize best MAPE per strategy."""
        lb = self.get_leaderboard(top_n=len(self.experiments))
        if len(lb) == 0:
            return pd.DataFrame()
        summary = lb.groupby("strategy").agg(
            best_mape=("mape", "min"),
            median_mape=("mape", "median"),
            n_trials=("mape", "count"),
        ).sort_values("best_mape").reset_index()
        return summary

    def _save_checkpoint(self) -> None:
        """Save experiment results to disk."""
        results = [e.to_dict() for e in self.experiments]
        path = self.output_dir / "experiments.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        lb = self.get_leaderboard(top_n=50)
        if len(lb) > 0:
            lb.to_csv(self.output_dir / "leaderboard.csv", index=False)

        summary = self.get_strategy_summary()
        if len(summary) > 0:
            summary.to_csv(self.output_dir / "strategy_summary.csv", index=False)
