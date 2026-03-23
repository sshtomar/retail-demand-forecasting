"""
Objective functions for the autoresearch loop.

MAPE is the primary objective. Additional metrics are tracked for analysis
but the search is driven by MAPE minimization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Full evaluation result for a single experiment run."""
    mape: float          # Primary objective — lower is better
    rmse: float
    mae: float
    smape: float
    median_ape: float
    p90_ape: float       # 90th percentile of absolute percentage errors
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mape": round(self.mape, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "smape": round(self.smape, 4),
            "median_ape": round(self.median_ape, 4),
            "p90_ape": round(self.p90_ape, 4),
            "n_samples": self.n_samples,
        }


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Filters out zeros in actuals."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — handles zeros better."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    """Run full evaluation suite. Returns EvalResult with MAPE as primary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    return EvalResult(
        mape=mape(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        smape=smape(y_true, y_pred),
        median_ape=float(np.median(ape)) if len(ape) > 0 else float("inf"),
        p90_ape=float(np.percentile(ape, 90)) if len(ape) > 0 else float("inf"),
        n_samples=len(y_true),
    )
