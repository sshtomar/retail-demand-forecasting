# retail-demand-forecasting

Demand Forecasting MLOps Using Databricks

## Autoresearch: MAPE-Driven Automated Model Search

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — define an objective function (MAPE), then let the system iteratively discover the best model and hyperparameters.

### How it works

1. **Define the objective**: MAPE (Mean Absolute Percentage Error) on a held-out temporal test set
2. **Strategy pool**: 7 forecasting strategies from naive baselines to LightGBM
3. **Search loop**: Random exploration (50%) + mutation of top configs (30%) + exploitation of the best (20%)
4. **Leaderboard**: All experiments are tracked, ranked, and checkpointed to disk

### Strategies

| Strategy | Type | Description |
|---|---|---|
| `naive_last_value` | Baseline | Last known demand per store-product |
| `moving_average` | Baseline | Rolling mean of last N days |
| `seasonal_naive` | Baseline | Same day-of-week from N weeks ago |
| `linear` | ML | Ridge/Lasso/ElasticNet with Fourier features |
| `random_forest` | ML | Random Forest with lag + rolling features |
| `gradient_boosting` | ML | Sklearn GBM with lag + rolling features |
| `lightgbm` | ML | LightGBM with full feature engineering |

### Quick start

```bash
pip install -r requirements.txt
python run_autoresearch.py --iterations 50
```

### CLI options

```
--iterations N     Number of experiments to run (default: 50)
--stores N         Number of stores in synthetic data (default: 3)
--products N       Number of products (default: 5)
--test-days N      Days in test set (default: 90)
--seed N           Random seed (default: 42)
--output-dir DIR   Where to save results (default: results/)
--quiet            Suppress per-iteration output
```

### Results

Results are saved to `results/`:
- `experiments.json` — full experiment log
- `leaderboard.csv` — top experiments ranked by MAPE
- `strategy_summary.csv` — best MAPE per strategy
