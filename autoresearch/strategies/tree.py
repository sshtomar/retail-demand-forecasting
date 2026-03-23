"""Tree-based model strategies (Random Forest, Gradient Boosting, XGBoost, LightGBM)."""

import numpy as np
import pandas as pd
from typing import Any

from .base import Strategy


FEATURE_COLS = [
    "price", "is_promo", "day_of_week", "day_of_month", "month",
    "day_of_year", "is_weekend", "store_id", "product_id",
]


def _build_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add lag features grouped by store-product."""
    df = df.copy()
    df = df.sort_values(["store_id", "product_id", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "product_id"])["demand"].shift(lag)
    # Rolling stats
    df["rolling_7_mean"] = (
        df.groupby(["store_id", "product_id"])["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    df["rolling_28_mean"] = (
        df.groupby(["store_id", "product_id"])["demand"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
    )
    df["rolling_7_std"] = (
        df.groupby(["store_id", "product_id"])["demand"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    )
    return df


class RandomForestStrategy(Strategy):
    """Random Forest regressor with lag features."""

    name = "random_forest"

    def _get_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        lags = list(range(1, self.params.get("max_lag", 7) + 1))
        df = _build_lag_features(df, lags)
        feature_cols = FEATURE_COLS + [f"lag_{l}" for l in lags] + [
            "rolling_7_mean", "rolling_28_mean", "rolling_7_std"
        ]
        df[feature_cols] = df[feature_cols].fillna(0)
        return df, feature_cols

    def fit(self, train: pd.DataFrame) -> None:
        from sklearn.ensemble import RandomForestRegressor

        df, self.feature_cols = self._get_features(train)
        X = df[self.feature_cols].values
        y = df["demand"].values

        self.model = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        df, _ = self._get_features(test)
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "categorical", "choices": [5, 10, 15, 20]},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 50},
            "max_lag": {"type": "int", "low": 3, "high": 28},
        }


class GradientBoostingStrategy(Strategy):
    """Gradient Boosting (sklearn) with lag features."""

    name = "gradient_boosting"

    def _get_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        lags = list(range(1, self.params.get("max_lag", 7) + 1))
        df = _build_lag_features(df, lags)
        feature_cols = FEATURE_COLS + [f"lag_{l}" for l in lags] + [
            "rolling_7_mean", "rolling_28_mean", "rolling_7_std"
        ]
        df[feature_cols] = df[feature_cols].fillna(0)
        return df, feature_cols

    def fit(self, train: pd.DataFrame) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        df, self.feature_cols = self._get_features(train)
        X = df[self.feature_cols].values
        y = df["demand"].values

        self.model = GradientBoostingRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 5),
            learning_rate=self.params.get("learning_rate", 0.1),
            subsample=self.params.get("subsample", 0.8),
            min_samples_leaf=self.params.get("min_samples_leaf", 10),
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        df, _ = self._get_features(test)
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 50},
            "max_lag": {"type": "int", "low": 3, "high": 28},
        }


class LightGBMStrategy(Strategy):
    """LightGBM regressor — usually the strongest tree method."""

    name = "lightgbm"

    def _get_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        lags = list(range(1, self.params.get("max_lag", 14) + 1))
        df = _build_lag_features(df, lags)
        feature_cols = FEATURE_COLS + [f"lag_{l}" for l in lags] + [
            "rolling_7_mean", "rolling_28_mean", "rolling_7_std"
        ]
        df[feature_cols] = df[feature_cols].fillna(0)
        return df, feature_cols

    def fit(self, train: pd.DataFrame) -> None:
        import lightgbm as lgb

        df, self.feature_cols = self._get_features(train)
        X = df[self.feature_cols]
        y = df["demand"].values

        self.model = lgb.LGBMRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", -1),
            learning_rate=self.params.get("learning_rate", 0.05),
            num_leaves=self.params.get("num_leaves", 31),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            min_child_samples=self.params.get("min_child_samples", 20),
            reg_alpha=self.params.get("reg_alpha", 0.0),
            reg_lambda=self.params.get("reg_lambda", 0.0),
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        df, _ = self._get_features(test)
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "n_estimators": {"type": "int", "low": 100, "high": 1000},
            "max_depth": {"type": "categorical", "choices": [-1, 5, 10, 15, 20]},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
            "max_lag": {"type": "int", "low": 7, "high": 56},
        }
