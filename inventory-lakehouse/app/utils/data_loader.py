import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import duckdb
import pandas as pd
import streamlit as st

try:
    from app.utils.artifacts import (
        MODEL_CONTRACT_PATH,
        BASELINE_CONTRACT_PATH,
        DIAG_REPORT_PATH,
        SCORING_CONTRACT_PATH,
        SCORING_MANIFEST_PATH,
        APP_CONTRACT_PATH,
        OPERATIONAL_LATEST_PATH,
        PARTITIONED_PREDICTIONS_GLOB,
        FEATURES_GLOB,
        STORE_METRICS_PATH,
        FAMILY_METRICS_PATH,
        DEMAND_TIER_METRICS_PATH,
        DOW_METRICS_PATH,
        PROMO_METRICS_PATH,
        HOLIDAY_METRICS_PATH,
        PERISHABILITY_METRICS_PATH,
        FEATURE_IMPORTANCE_PATH,
        CALIBRATION_DECILES_PATH,
        WORST_ERRORS_PATH,
        OVERALL_DIAG_PATH,
        HYPERPARAMETER_RESULTS_PATH,
        MODEL_COMPARISON_PATH,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from utils.artifacts import (
            MODEL_CONTRACT_PATH,
            BASELINE_CONTRACT_PATH,
            DIAG_REPORT_PATH,
            SCORING_CONTRACT_PATH,
            SCORING_MANIFEST_PATH,
            APP_CONTRACT_PATH,
            OPERATIONAL_LATEST_PATH,
            PARTITIONED_PREDICTIONS_GLOB,
            FEATURES_GLOB,
            STORE_METRICS_PATH,
            FAMILY_METRICS_PATH,
            DEMAND_TIER_METRICS_PATH,
            DOW_METRICS_PATH,
            PROMO_METRICS_PATH,
            HOLIDAY_METRICS_PATH,
            PERISHABILITY_METRICS_PATH,
            FEATURE_IMPORTANCE_PATH,
            CALIBRATION_DECILES_PATH,
            WORST_ERRORS_PATH,
            OVERALL_DIAG_PATH,
            HYPERPARAMETER_RESULTS_PATH,
            MODEL_COMPARISON_PATH,
        )
    except (ImportError, ModuleNotFoundError):
        from .artifacts import (
            MODEL_CONTRACT_PATH,
            BASELINE_CONTRACT_PATH,
            DIAG_REPORT_PATH,
            SCORING_CONTRACT_PATH,
            SCORING_MANIFEST_PATH,
            APP_CONTRACT_PATH,
            OPERATIONAL_LATEST_PATH,
            PARTITIONED_PREDICTIONS_GLOB,
            FEATURES_GLOB,
            STORE_METRICS_PATH,
            FAMILY_METRICS_PATH,
            DEMAND_TIER_METRICS_PATH,
            DOW_METRICS_PATH,
            PROMO_METRICS_PATH,
            HOLIDAY_METRICS_PATH,
            PERISHABILITY_METRICS_PATH,
            FEATURE_IMPORTANCE_PATH,
            CALIBRATION_DECILES_PATH,
            WORST_ERRORS_PATH,
            OVERALL_DIAG_PATH,
            HYPERPARAMETER_RESULTS_PATH,
            MODEL_COMPARISON_PATH,
        )

# ── 1. JSON Contract Loaders ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_json(filepath_str: str) -> Dict[str, Any]:
    """Loads a JSON contract from disk with caching."""
    p = Path(filepath_str)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model_contract() -> Dict[str, Any]:
    return load_json(str(MODEL_CONTRACT_PATH))

def load_baseline_contract() -> Dict[str, Any]:
    return load_json(str(BASELINE_CONTRACT_PATH))

def load_diagnostic_report() -> Dict[str, Any]:
    return load_json(str(DIAG_REPORT_PATH))

def load_scoring_contract() -> Dict[str, Any]:
    return load_json(str(SCORING_CONTRACT_PATH))

def load_scoring_manifest() -> Dict[str, Any]:
    return load_json(str(SCORING_MANIFEST_PATH))

# ── 2. Parquet Dataset Loaders ───────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_parquet(filepath_str: str) -> pd.DataFrame:
    """Reads a standalone parquet file into Pandas."""
    p = Path(filepath_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

def load_store_metrics() -> pd.DataFrame:
    return load_parquet(str(STORE_METRICS_PATH))

def load_family_metrics() -> pd.DataFrame:
    return load_parquet(str(FAMILY_METRICS_PATH))

def load_demand_tier_metrics() -> pd.DataFrame:
    return load_parquet(str(DEMAND_TIER_METRICS_PATH))

def load_dow_metrics() -> pd.DataFrame:
    return load_parquet(str(DOW_METRICS_PATH))

def load_promo_metrics() -> pd.DataFrame:
    return load_parquet(str(PROMO_METRICS_PATH))

def load_holiday_metrics() -> pd.DataFrame:
    return load_parquet(str(HOLIDAY_METRICS_PATH))

def load_perishability_metrics() -> pd.DataFrame:
    return load_parquet(str(PERISHABILITY_METRICS_PATH))

def load_feature_importance() -> pd.DataFrame:
    return load_parquet(str(FEATURE_IMPORTANCE_PATH))

def load_calibration_deciles() -> pd.DataFrame:
    return load_parquet(str(CALIBRATION_DECILES_PATH))

def load_worst_error_cases() -> pd.DataFrame:
    return load_parquet(str(WORST_ERRORS_PATH))

def load_hyperparameter_results() -> pd.DataFrame:
    return load_parquet(str(HYPERPARAMETER_RESULTS_PATH))

def load_model_comparison() -> pd.DataFrame:
    return load_parquet(str(MODEL_COMPARISON_PATH))

def load_operational_slice() -> pd.DataFrame:
    return load_parquet(str(OPERATIONAL_LATEST_PATH))

# ── 3. Dynamic DuckDB Pushdown Query Functions ───────────────────────────────

def get_duckdb_con() -> duckdb.DuckDBPyConnection:
    """Creates a local in-memory DuckDB connection."""
    return duckdb.connect(":memory:")

@st.cache_data(ttl=86400)
def get_distinct_store_metadata() -> pd.DataFrame:
    """Extracts unique store metadata (store_id, city, store_type, cluster)."""
    con = get_duckdb_con()
    feat_glob_str = str(FEATURES_GLOB).replace("\\", "/")
    df = con.execute(f"""
        SELECT DISTINCT 
            store_id,
            city,
            store_type,
            cluster
        FROM read_parquet('{feat_glob_str}')
        ORDER BY store_id
    """).df()
    con.close()
    return df

@st.cache_data(ttl=3600)
def get_distinct_families(store_id: Optional[int] = None) -> List[str]:
    """Returns available product families dynamically."""
    con = get_duckdb_con()
    op_path_str = str(OPERATIONAL_LATEST_PATH).replace("\\", "/")
    filter_sql = f"WHERE store_id = {store_id}" if store_id is not None else ""
    df = con.execute(f"""
        SELECT DISTINCT family
        FROM read_parquet('{op_path_str}')
        {filter_sql}
        ORDER BY family
    """).df()
    con.close()
    return df["family"].tolist()

@st.cache_data(ttl=3600)
def get_distinct_skus(store_id: int, family: str) -> List[int]:
    """Returns available SKU IDs for a given store and family."""
    con = get_duckdb_con()
    op_path_str = str(OPERATIONAL_LATEST_PATH).replace("\\", "/")
    df = con.execute(f"""
        SELECT DISTINCT sku_id
        FROM read_parquet('{op_path_str}')
        WHERE store_id = {store_id} AND family = '{family}'
        ORDER BY sku_id
    """).df()
    con.close()
    return df["sku_id"].tolist()

@st.cache_data(ttl=1800)
def load_store_sku_forecast_series(store_id: int, sku_id: int) -> pd.DataFrame:
    """
    Queries actual sales from features and forecasts from partitioned predictions
    for a specific store-SKU combination across the holdout horizon.
    """
    con = get_duckdb_con()
    preds_glob_str = str(PARTITIONED_PREDICTIONS_GLOB).replace("\\", "/")
    feat_glob_str = str(FEATURES_GLOB).replace("\\", "/")
    
    # 1. Query holdout predictions (NB06 outputs)
    preds_df = con.execute(f"""
        SELECT 
            forecast_date,
            target_date,
            store_id,
            sku_id,
            family,
            raw_prediction,
            predicted_unit_sales,
            demand_tier,
            expected_error_mae,
            operational_lower_80,
            operational_upper_80
        FROM read_parquet('{preds_glob_str}')
        WHERE store_id = {store_id} AND sku_id = {sku_id}
        ORDER BY target_date
    """).df()

    if preds_df.empty:
        con.close()
        return pd.DataFrame()

    # 2. Query historical ground truth actuals & promo flags from feature store
    actuals_df = con.execute(f"""
        SELECT 
            target_date,
            target_unit_sales AS actual_unit_sales,
            on_promotion_today AS onpromotion
        FROM read_parquet('{feat_glob_str}')
        WHERE store_id = {store_id} 
          AND sku_id = {sku_id}
          AND target_date >= '2017-01-01' 
          AND target_date <= '2017-08-15'
        ORDER BY target_date
    """).df()

    con.close()

    # Merge predictions with actuals
    if not actuals_df.empty:
        merged = pd.merge(preds_df, actuals_df, on="target_date", how="left")
    else:
        merged = preds_df.copy()
        merged["actual_unit_sales"] = None
        merged["onpromotion"] = False

    merged["target_date"] = pd.to_datetime(merged["target_date"])
    merged["forecast_date"] = pd.to_datetime(merged["forecast_date"])
    return merged.sort_values("target_date")
