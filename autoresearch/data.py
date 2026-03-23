"""
Synthetic retail demand data generator.

Generates realistic retail time series with:
- Multiple stores and products
- Trend, seasonality (weekly + yearly), holiday effects
- Promotions, price elasticity, and random noise
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_retail_data(
    n_stores: int = 5,
    n_products: int = 10,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate synthetic retail demand data with realistic patterns."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    records = []
    for store_id in range(1, n_stores + 1):
        store_base = rng.uniform(50, 200)
        for product_id in range(1, n_products + 1):
            product_base = rng.uniform(5, 50)
            base_demand = store_base + product_base
            trend_slope = rng.uniform(-0.02, 0.05)

            # Seasonality parameters
            weekly_amp = rng.uniform(5, 20)
            yearly_amp = rng.uniform(10, 40)

            # Price
            base_price = rng.uniform(2.0, 50.0)
            price_elasticity = rng.uniform(-2.5, -0.5)

            for i, date in enumerate(dates):
                day_of_week = date.dayofweek
                day_of_year = date.dayofyear

                # Trend
                trend = trend_slope * i

                # Weekly seasonality (weekends higher)
                weekly = weekly_amp * np.sin(2 * np.pi * day_of_week / 7)

                # Yearly seasonality (holiday peaks)
                yearly = yearly_amp * (
                    np.sin(2 * np.pi * day_of_year / 365)
                    + 0.5 * np.sin(4 * np.pi * day_of_year / 365)
                )

                # Holiday boost (around Christmas, Black Friday, etc.)
                holiday_boost = 0.0
                if date.month == 12 and date.day >= 15:
                    holiday_boost = rng.uniform(30, 80)
                elif date.month == 11 and date.day >= 24 and date.day <= 30:
                    holiday_boost = rng.uniform(20, 60)

                # Random promotion (10% of days)
                is_promo = rng.random() < 0.10
                promo_lift = rng.uniform(15, 40) if is_promo else 0.0

                # Price variation
                price_mult = 1.0 + rng.uniform(-0.15, 0.15)
                price = base_price * price_mult
                price_effect = price_elasticity * (price - base_price) / base_price * base_demand

                # Noise
                noise = rng.normal(0, base_demand * 0.08)

                demand = max(
                    0,
                    base_demand + trend + weekly + yearly
                    + holiday_boost + promo_lift + price_effect + noise,
                )

                records.append({
                    "date": date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "demand": round(demand, 1),
                    "price": round(price, 2),
                    "is_promo": int(is_promo),
                    "day_of_week": day_of_week,
                    "day_of_month": date.day,
                    "month": date.month,
                    "day_of_year": day_of_year,
                    "is_weekend": int(day_of_week >= 5),
                })

    df = pd.DataFrame(records)
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally — last `test_days` days become the test set."""
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test
