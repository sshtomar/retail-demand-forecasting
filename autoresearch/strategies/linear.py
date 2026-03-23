"""Linear model strategies."""

import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

from .base import Strategy


FEATURE_COLS = [
    "price", "is_promo", "day_of_week", "day_of_month", "month",
    "day_of_year", "is_weekend", "store_id", "product_id",
]


class LinearStrategy(Strategy):
    """Ridge/Lasso/ElasticNet regression on tabular features."""

    name = "linear"

    def _get_model(self):
        model_type = self.params.get("model_type", "ridge")
        alpha = self.params.get("alpha", 1.0)
        if model_type == "ridge":
            return Ridge(alpha=alpha)
        elif model_type == "lasso":
            return Lasso(alpha=alpha, max_iter=5000)
        else:
            l1_ratio = self.params.get("l1_ratio", 0.5)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        add_fourier = self.params.get("add_fourier", True)
        n_harmonics = self.params.get("n_harmonics", 3)

        X = df[FEATURE_COLS].values.astype(float)

        if add_fourier:
            doy = df["day_of_year"].values.astype(float)
            fourier_feats = []
            for k in range(1, n_harmonics + 1):
                fourier_feats.append(np.sin(2 * np.pi * k * doy / 365))
                fourier_feats.append(np.cos(2 * np.pi * k * doy / 365))
            X = np.column_stack([X] + fourier_feats)

        return X

    def fit(self, train: pd.DataFrame) -> None:
        X = self._build_features(train)
        y = train["demand"].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._get_model()
        self.model.fit(X_scaled, y)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = self._build_features(test)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.maximum(preds, 0)

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        return {
            "model_type": {"type": "categorical", "choices": ["ridge", "lasso", "elasticnet"]},
            "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9},
            "add_fourier": {"type": "categorical", "choices": [True, False]},
            "n_harmonics": {"type": "int", "low": 1, "high": 10},
        }
