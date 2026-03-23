"""Base class for all forecasting strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    """
    A forecasting strategy that the autoresearch engine can instantiate,
    train, and evaluate. Each strategy defines its own hyperparameter
    search space.
    """

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Generate predictions for test data. Returns array of predicted demand."""

    @classmethod
    @abstractmethod
    def search_space(cls) -> dict[str, Any]:
        """
        Define the hyperparameter search space.

        Returns a dict where keys are param names and values are dicts like:
            {"type": "int", "low": 1, "high": 100}
            {"type": "float", "low": 0.01, "high": 1.0, "log": True}
            {"type": "categorical", "choices": ["a", "b", "c"]}
        """

    @classmethod
    def sample_params(cls, rng: np.random.RandomState) -> dict[str, Any]:
        """Sample random hyperparameters from the search space."""
        space = cls.search_space()
        params = {}
        for name, spec in space.items():
            if spec["type"] == "int":
                params[name] = int(rng.randint(spec["low"], spec["high"] + 1))
            elif spec["type"] == "float":
                if spec.get("log"):
                    log_low = np.log(spec["low"])
                    log_high = np.log(spec["high"])
                    params[name] = float(np.exp(rng.uniform(log_low, log_high)))
                else:
                    params[name] = float(rng.uniform(spec["low"], spec["high"]))
            elif spec["type"] == "categorical":
                params[name] = spec["choices"][rng.randint(len(spec["choices"]))]
        return params

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"
