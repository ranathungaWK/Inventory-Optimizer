"""
Lakehouse Artifact Path Registry
Resolves canonical paths to all data lake, model, diagnostic, and prediction artifacts.
"""
from pathlib import Path

def get_lakehouse_root() -> Path:
    """Finds the root directory of the inventory-lakehouse project."""
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "data_lake").is_dir() and ((p / "notebooks").is_dir() or (p / "app").is_dir()):
            return p
    # Fallback
    return Path("D:/Demand_forecasting/inventory-lakehouse")

ROOT_DIR = get_lakehouse_root()
DATA_LAKE_DIR = ROOT_DIR / "data_lake"
GOLD_DIR = DATA_LAKE_DIR / "gold"

# Upstream Artifact Directories
FEATURES_DIR = GOLD_DIR / "gold_forecasting_features"
BASELINE_DIR = GOLD_DIR / "forecasting_baseline_results"
LGBM_DIR = GOLD_DIR / "lightgbm_results"
DIAG_DIR = GOLD_DIR / "lightgbm_diagnostics"
PROD_PRED_DIR = GOLD_DIR / "production_predictions"

# JSON Contracts
MODEL_CONTRACT_PATH = LGBM_DIR / "model_contract.json"
BASELINE_CONTRACT_PATH = BASELINE_DIR / "baseline_contract.json"
DIAG_REPORT_PATH = DIAG_DIR / "diagnostic_report.json"
SCORING_CONTRACT_PATH = PROD_PRED_DIR / "scoring_run_contract.json"
SCORING_MANIFEST_PATH = PROD_PRED_DIR / "scoring_manifest.json"
APP_CONTRACT_PATH = GOLD_DIR / "app_contract.json"

# Model Booster & Tuning Results
MODEL_BOOSTER_PATH = LGBM_DIR / "lightgbm_model.txt"
HYPERPARAMETER_RESULTS_PATH = LGBM_DIR / "hyperparameter_results.parquet"
MODEL_COMPARISON_PATH = LGBM_DIR / "model_comparison.parquet"

# Diagnostic Parquet Files
STORE_METRICS_PATH = DIAG_DIR / "store_error_metrics.parquet"
FAMILY_METRICS_PATH = DIAG_DIR / "family_error_metrics.parquet"
DEMAND_TIER_METRICS_PATH = DIAG_DIR / "demand_tier_metrics.parquet"
DOW_METRICS_PATH = DIAG_DIR / "day_of_week_metrics.parquet"
PROMO_METRICS_PATH = DIAG_DIR / "promotion_metrics.parquet"
HOLIDAY_METRICS_PATH = DIAG_DIR / "holiday_metrics.parquet"
PERISHABILITY_METRICS_PATH = DIAG_DIR / "perishability_metrics.parquet"
FEATURE_IMPORTANCE_PATH = DIAG_DIR / "feature_importance.parquet"
CALIBRATION_DECILES_PATH = DIAG_DIR / "calibration_deciles.parquet"
WORST_ERRORS_PATH = DIAG_DIR / "worst_error_cases.parquet"
DIAG_SAMPLE_PATH = DIAG_DIR / "diagnostic_predictions_sample.parquet"
OVERALL_DIAG_PATH = DIAG_DIR / "overall_diagnostics.parquet"

# Production Predictions
OPERATIONAL_LATEST_PATH = PROD_PRED_DIR / "operational_simulation_latest.parquet"
PARTITIONED_PREDICTIONS_GLOB = str(PROD_PRED_DIR / "forecast_date=*/part-*.parquet").replace("\\", "/")
FEATURES_GLOB = str(FEATURES_DIR / "*.parquet").replace("\\", "/")
