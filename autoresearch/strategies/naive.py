"""Naive and simple baseline strategies."""

import numpy as np
import pandas as pd
from typing import Any

from .base import Strategy


class NaiveLastValue(Strategy):
    """Predict using the last known demand value per store-product."""

    name = "naive_last_value"

    def fit(self, train: pd.DataFrame) -> None:
        self.last_values = (
            train.sort_values("date")
            .groupby(["store_id", "product_id"])["demand"]
            .last()
            .to_dict()
        )
        self.global_mean = train["demand"].mean()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test.apply(
            lambda r: self.last_values.get(
                (r["store_id"], r["product_id"]), self.global_mean
            ),
            axis=1,
        ).values

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {}  # No hyperparameters


class MovingAverage(Strategy):
    """Predict using a rolling mean of the last N days per store-product."""

    name = "moving_average"

    def fit(self, train: pd.DataFrame) -> None:
        window = self.params.get("window", 7)
        self.averages = (
            train.sort_values("date")
            .groupby(["store_id", "product_id"])["demand"]
            .apply(lambda x: x.tail(window).mean())
            .to_dict()
        )
        self.global_mean = train["demand"].mean()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test.apply(
            lambda r: self.averages.get(
                (r["store_id"], r["product_id"]), self.global_mean
            ),
            axis=1,
        ).values

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "window": {"type": "int", "low": 3, "high": 90},
        }


class SeasonalNaive(Strategy):
    """Predict using the same day-of-week from N weeks ago."""

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame) -> None:
        weeks_back = self.params.get("weeks_back", 1)
        train = train.sort_values("date")

        self.lookup = {}
        for (store, prod), group in train.groupby(["store_id", "product_id"]):
            # Get the last `weeks_back` weeks and average by day_of_week
            cutoff = group["date"].max() - pd.Timedelta(weeks=weeks_back)
            recent = group[group["date"] > cutoff]
            dow_avg = recent.groupby("day_of_week")["demand"].mean().to_dict()
            self.lookup[(store, prod)] = dow_avg

        self.global_dow = train.groupby("day_of_week")["demand"].mean().to_dict()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test.iterrows():
            key = (row["store_id"], row["product_id"])
            dow = row["day_of_week"]
            if key in self.lookup and dow in self.lookup[key]:
                preds.append(self.lookup[key][dow])
            else:
                preds.append(self.global_dow.get(dow, 0))
        return np.array(preds)

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "weeks_back": {"type": "int", "low": 1, "high": 12},
        }
