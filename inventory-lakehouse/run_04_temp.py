# Cell 2 — Setup, Dependencies, Imports & Path Discovery
import os
import sys
import json
import gc
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    display = print

print("Installing LightGBM, MLflow and Optuna...")

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"LightGBM version : {lgb.__version__}")
print(f"Optuna version   : {optuna.__version__}")
print(f"MLflow version   : {mlflow.__version__}")

# Dynamic lakehouse root resolution
lakehouse_root = None
for p in [Path.cwd()] + list(Path.cwd().parents):
    if (p / "data_lake").is_dir() and (p / "notebooks").is_dir():
        lakehouse_root = p
        break

if lakehouse_root is None:
    lakehouse_root = Path("D:/Demand_forecasting/inventory-lakehouse")

GOLD_DIR     = lakehouse_root / "data_lake" / "gold"
FEATURES_DIR = GOLD_DIR / "gold_forecasting_features"
BASELINE_DIR = GOLD_DIR / "forecasting_baseline_results"
OUTPUT_DIR   = GOLD_DIR / "lightgbm_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow Experiment Registration
MLFLOW_EXPERIMENT = "demand_forecasting_lightgbm"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print(f"Lakehouse root : {lakehouse_root}")
print(f"Features dir   : {FEATURES_DIR}")
print(f"Baseline dir   : {BASELINE_DIR}")
print(f"Output dir     : {OUTPUT_DIR}")
print(f"MLflow Exp     : {MLFLOW_EXPERIMENT}")

assert FEATURES_DIR.exists(), f"Features directory missing: {FEATURES_DIR}"
assert BASELINE_DIR.exists(), f"Baseline directory missing: {BASELINE_DIR}"

# Cell 3 — Load and Validate Baseline Contract
BASELINE_CONTRACT_PATH = BASELINE_DIR / "baseline_contract.json"
assert BASELINE_CONTRACT_PATH.exists(), f"Baseline contract missing at {BASELINE_CONTRACT_PATH}"

with open(BASELINE_CONTRACT_PATH, "r") as f:
    baseline_contract = json.load(f)

assert baseline_contract.get("produced_by") == "03_baselines_and_evaluation", "Contract was not produced by Notebook 03"

TRAIN_END  = baseline_contract["split"]["train_end"]
VAL_START  = baseline_contract["split"]["validation_start"]
VAL_END    = baseline_contract["split"]["validation_end"]
TEST_START = baseline_contract["split"]["test_start"].split(" ")[0]
TEST_END   = baseline_contract["split"]["test_end"].split(" ")[0]

BASELINE_WMAPE      = float(baseline_contract["model_selection"]["threshold_wmape"])
BASELINE_MAE        = float(baseline_contract["model_selection"]["threshold_mae"])
BASELINE_TEST_WMAPE = float(baseline_contract["final_evaluation"]["baseline_wmape"])
BASELINE_TEST_MAE   = float(baseline_contract["final_evaluation"]["baseline_mae"])

print("Loaded Baseline Contract:")
print(f"  Train End             : {TRAIN_END}")
print(f"  Validation Split      : {VAL_START} -> {VAL_END}")
print(f"  Test Split            : {TEST_START} -> {TEST_END}")
print(f"  Threshold Val WMAPE   : {BASELINE_WMAPE:.6f}")
print(f"  Threshold Val MAE     : {BASELINE_MAE:.6f}")
print(f"  Baseline Test WMAPE   : {BASELINE_TEST_WMAPE:.6f}")
print(f"  Baseline Test MAE     : {BASELINE_TEST_MAE:.6f}")

# Cell 4 — Schema Contract Validation
IDENTIFIERS = ["forecast_date", "target_date", "store_id", "sku_id"]
TARGET = "target_unit_sales"

CATEGORICAL_FEATURES = ["family", "city", "store_type", "cluster"]
NUMERIC_FEATURES = [
    "units_sold", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
    "zero_sales_count_7d", "zero_sales_count_28d",
    "is_perishable", "target_day_of_week", "target_month",
    "target_is_weekend", "target_is_holiday",
    "on_promotion_today", "on_promotion_tomorrow", "oil_price"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
ALL_COLUMNS = IDENTIFIERS + FEATURES + [TARGET]

con = duckdb.connect()
glob_pattern = str(FEATURES_DIR / "*.parquet").replace("\\", "/")
schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob_pattern}')").df()
con.close()

present_cols = set(schema_df["column_name"])
missing_cols = set(ALL_COLUMNS) - present_cols
assert len(missing_cols) == 0, f"Missing columns in Parquet feature store: {missing_cols}"
assert len(FEATURES) == 24, f"Expected 24 features, found {len(FEATURES)}"
assert TARGET not in FEATURES, "Target leakage alert: TARGET present in FEATURES list!"

print(f"Schema validated successfully: {len(FEATURES)} features, {len(IDENTIFIERS)} identifiers, 1 target.")

# Cell 5 — DuckDB Feature Views Setup
con = duckdb.connect()
glob_pattern = str(FEATURES_DIR / "*.parquet").replace("\\", "/")
con.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{glob_pattern}')")

con.execute(f"""
CREATE OR REPLACE VIEW train AS
SELECT * FROM features
WHERE target_date <= '{TRAIN_END}'
""")

con.execute(f"""
CREATE OR REPLACE VIEW val AS
SELECT * FROM features
WHERE target_date >= '{VAL_START}' AND target_date <= '{VAL_END}'
""")

train_max = con.execute("SELECT MAX(target_date) FROM train").fetchone()[0]
val_min = con.execute("SELECT MIN(target_date) FROM val").fetchone()[0]
val_max = con.execute("SELECT MAX(target_date) FROM val").fetchone()[0]

print(f"Train max date : {train_max}")
print(f"Val min date   : {val_min}")
print(f"Val max date   : {val_max}")

assert str(train_max)[:10] <= TRAIN_END, f"Train data leakage past {TRAIN_END}"
assert str(val_min)[:10] >= VAL_START, f"Val data leakage before {VAL_START}"
assert str(val_max)[:10] <= VAL_END, f"Val data leakage past {VAL_END}"

# Cell 6 — Row Cap & Hyperparameter Search Policy
MAX_TRAIN_ROWS        = 5_000_000
MAX_VAL_ROWS          = 1_000_000
MAX_TEST_ROWS         = 10_000_000
RANDOM_SEED           = 42
OPTUNA_SEED           = 42
N_TRIALS              = 10  # Initial pipeline-validation budget (scale to 20+ for production tuning)
EARLY_STOPPING_ROUNDS = 100
MAX_ESTIMATORS       = 3000

print("Execution & Tuning Policy:")
print(f"  MAX_TRAIN_ROWS : {MAX_TRAIN_ROWS:,}")
print(f"  MAX_VAL_ROWS   : {MAX_VAL_ROWS:,}")
print(f"  MAX_TEST_ROWS  : {MAX_TEST_ROWS:,}")
print(f"  N_TRIALS       : {N_TRIALS} (Initial pipeline-validation budget)")
print(f"  RANDOM_SEED    : {RANDOM_SEED}")

# Cell 7 — Materialize Optimized High-Speed System Samples
#
# Performance Optimization Note:
# We use DuckDB system block-level sampling ('system' mode) with explicit column projections.
# System sampling skips reading ~93% of unneeded Parquet row groups directly at the I/O layer,
# reducing table creation time from several minutes (reservoir method) down to ~25 seconds.
#
train_rows_available = con.execute("SELECT COUNT(*) FROM train").fetchone()[0]
val_rows_available   = con.execute("SELECT COUNT(*) FROM val").fetchone()[0]

print(f"Train rows available : {train_rows_available:,}")
print(f"Val rows available   : {val_rows_available:,}")

cols_sql = ", ".join(FEATURES + [TARGET])

con.execute(f"""
CREATE OR REPLACE TABLE train_ml AS
SELECT {cols_sql} FROM train
USING SAMPLE 7% (system, {RANDOM_SEED})
LIMIT {MAX_TRAIN_ROWS}
""")

con.execute(f"""
CREATE OR REPLACE TABLE val_ml AS
SELECT {cols_sql} FROM val
USING SAMPLE 10% (system, {RANDOM_SEED})
LIMIT {MAX_VAL_ROWS}
""")

train_rows_actual = con.execute("SELECT COUNT(*) FROM train_ml").fetchone()[0]
val_rows_actual   = con.execute("SELECT COUNT(*) FROM val_ml").fetchone()[0]

print(f"Materialized Train ML : {train_rows_actual:,} rows")
print(f"Materialized Val ML   : {val_rows_actual:,} rows")

# Cell 8 — Load ML Data into Pandas
train_df = con.execute("SELECT * FROM train_ml").df()
val_df   = con.execute("SELECT * FROM val_ml").df()

print(f"Loaded train_df in memory : {train_df.shape}")
print(f"Loaded val_df in memory   : {val_df.shape}")

# Cell 9 — Memory-Efficient Dtype Casting
for col in CATEGORICAL_FEATURES:
    train_df[col] = train_df[col].astype("category")
    val_df[col]   = val_df[col].astype("category")

for col in NUMERIC_FEATURES + [TARGET]:
    train_df[col] = train_df[col].astype("float32")
    val_df[col]   = val_df[col].astype("float32")

train_mem_mb = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
val_mem_mb   = val_df.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"Optimized train_df RAM usage : {train_mem_mb:.2f} MB")
print(f"Optimized val_df RAM usage   : {val_mem_mb:.2f} MB")

# Cell 10 — Data Quality & Integrity Validation
assert train_df.isna().sum().sum() == 0, "Nulls found in train_df"
assert val_df.isna().sum().sum() == 0, "Nulls found in val_df"

assert np.isfinite(train_df[NUMERIC_FEATURES + [TARGET]].values).all(), "Non-finite values in train_df"
assert np.isfinite(val_df[NUMERIC_FEATURES + [TARGET]].values).all(), "Non-finite values in val_df"

print("[PASS] Data quality checks passed: zero nulls, all numeric inputs finite.")

# Cell 11 — Separate Feature Matrices (X) and Target Vectors (y)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val     = val_df[FEATURES], val_df[TARGET]

print(f"X_train shape : {X_train.shape}, y_train shape : {y_train.shape}")
print(f"X_val shape   : {X_val.shape}, y_val shape   : {y_val.shape}")

# Cell 12 — Metric Definitions (WMAPE & MAE)
def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    denom = np.sum(y_t)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_t - y_p)) / denom)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.mean(np.abs(y_t - y_p)))

# Sanity check
assert abs(wmape(np.array([10, 20]), np.array([12, 18])) - 4.0 / 30.0) < 1e-6
print("[PASS] WMAPE & MAE metric functions initialized and verified.")

# Cell 13 — Optuna Search Space & Objective Function
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "regression",
        "metric": "l2",  # Explicit L2 loss for mean regression challenger
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127, step=16),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300, step=50),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "subsample_freq": 1,  # Required for subsample/bagging to take effect
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "n_estimators": MAX_ESTIMATORS,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    run_name = f"trial_{trial.number:03d}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        val_pred = model.predict(X_val)
        val_wmape_score = wmape(y_val, val_pred)
        val_mae_score   = mae(y_val, val_pred)
        best_iter       = int(model.best_iteration_)

        mlflow.log_metrics({
            "val_wmape": val_wmape_score,
            "val_mae": val_mae_score,
            "best_iteration": best_iter,
        })
        mlflow.set_tags({
            "stage": "hyperparameter_tuning",
            "promotion_gate": str(val_wmape_score < BASELINE_WMAPE),
        })

        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("val_mae", val_mae_score)

        del model
        gc.collect()

        return val_wmape_score

# Cell 14 — Execute Optuna Study within MLflow Parent Run
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=OPTUNA_SEED),
    study_name="lightgbm_wmape_tuning",
)

print(f"Launching Optuna TPE search over {N_TRIALS} trials (Initial pipeline-validation budget)...")
with mlflow.start_run(run_name="optuna_lightgbm_tuning") as parent_run:
    mlflow.log_params({
        "n_trials": N_TRIALS,
        "random_seed": RANDOM_SEED,
        "optuna_seed": OPTUNA_SEED,
        "train_end": TRAIN_END,
        "validation_start": VAL_START,
        "validation_end": VAL_END,
        "baseline_wmape": BASELINE_WMAPE,
        "max_train_rows": MAX_TRAIN_ROWS,
        "max_val_rows": MAX_VAL_ROWS,
        "train_rows_actual": len(X_train),
        "val_rows_actual": len(X_val),
    })

    study.optimize(objective, n_trials=N_TRIALS)

    mlflow.log_metric("best_val_wmape", study.best_trial.value)
    mlflow.log_metric("best_trial_number", study.best_trial.number)

PARENT_RUN_ID = parent_run.info.run_id
print(f"Optuna study finished. Parent MLflow Run ID: {PARENT_RUN_ID}")
print(f"Best Trial #{study.best_trial.number} WMAPE: {study.best_trial.value:.6f}")

# Cell 15 — Inspect Hyperparameter Search Results
trials_list = []
for t in study.trials:
    if t.value is not None:
        row = {
            "trial": t.number,
            "val_wmape": t.value,
            "val_mae": t.user_attrs.get("val_mae"),
            "best_iteration": t.user_attrs.get("best_iteration"),
            **t.params,
        }
        trials_list.append(row)

hyperparameter_results_df = pd.DataFrame(trials_list).sort_values("val_wmape").reset_index(drop=True)
print("Top Optuna Trials (Sorted by Val WMAPE):")
display(hyperparameter_results_df.head(10))

# Cell 16 — Model Promotion Gate (Hard Stop)
best_trial = study.best_trial
best_lgbm_val_wmape = float(best_trial.value)

print("=" * 70)
print("MODEL PROMOTION GATE EVALUATION")
print("=" * 70)
print(f"Best LightGBM Trial  : #{best_trial.number}")
print(f"Candidate Val WMAPE  : {best_lgbm_val_wmape:.6f}")
print(f"Baseline Threshold   : {BASELINE_WMAPE:.6f}")

assert best_lgbm_val_wmape < BASELINE_WMAPE, (
    f"[FAIL] Model Promotion Gate Failed! LightGBM did not beat baseline.\n"
    f"  LightGBM WMAPE = {best_lgbm_val_wmape:.6f}\n"
    f"  Baseline WMAPE = {BASELINE_WMAPE:.6f}\n"
    f"  TEST split remains strictly locked."
)

print("[PASS] MODEL PROMOTION GATE PASSED: LightGBM strictly outperforms baseline on validation.")

# Cell 17 — Freeze Model Configuration
BEST_PARAMS = {
    **best_trial.params,
    "objective": "regression",
    "metric": "l2",
    "n_estimators": int(best_trial.user_attrs["best_iteration"]),
    "subsample_freq": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

BEST_ITERATION = BEST_PARAMS["n_estimators"]
MODEL_FROZEN   = True
TEST_UNLOCKED  = False

print("FINAL FROZEN CONFIGURATION:")
print(json.dumps(BEST_PARAMS, indent=2))
print(f"[LOCK] Configuration frozen. Trees set to best_iteration = {BEST_ITERATION}.")

# Cell 18 — Train Final Model on TRAIN Split Only
#
# Conservative Retraining Protocol Rationale:
# Validation observations are used to select hyperparameters and the optimal number of boosting iterations,
# but are deliberately excluded from final fitting to preserve a strict out-of-sample holdout protocol.
# We intentionally train the final model exclusively on X_train / y_train up to TRAIN_END (2016-08-31).
#
assert MODEL_FROZEN, "Cannot train final model: Configuration is not frozen."

print(f"Fitting final model on TRAIN data only ({len(X_train):,} rows) with n_estimators={BEST_ITERATION}...")
final_model = lgb.LGBMRegressor(**BEST_PARAMS)
final_model.fit(
    X_train,
    y_train,
    categorical_feature=CATEGORICAL_FEATURES,
)

print("[PASS] Final model training completed.")
print(f"  Training Cutoff Date : {TRAIN_END}")
print(f"  Final Estimators     : {BEST_ITERATION}")

# Cell 19 — Diagnostic Validation Check on Frozen Model
final_val_pred  = final_model.predict(X_val)
final_val_wmape = wmape(y_val, final_val_pred)
final_val_mae   = mae(y_val, final_val_pred)

print(f"Final Frozen Model Val WMAPE : {final_val_wmape:.6f}")
print(f"Final Frozen Model Val MAE   : {final_val_mae:.6f}")
print(f"Baseline Threshold WMAPE     : {BASELINE_WMAPE:.6f}")

assert final_val_wmape < BASELINE_WMAPE, "Final frozen model failed validation re-check!"
print("[PASS] Final frozen model re-check verified against baseline.")

# Cell 20 — Extract Feature Importance
booster = final_model.booster_
importance_gain  = booster.feature_importance(importance_type="gain")
importance_split = booster.feature_importance(importance_type="split")

feature_importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_gain": importance_gain,
    "importance_split": importance_split,
}).sort_values("importance_gain", ascending=False).reset_index(drop=True)

print("Top 15 Feature Importances (by Gain):")
display(feature_importance_df.head(15))

# Cell 21 — Log Final Model MLflow Run
with mlflow.start_run(run_name="final_lightgbm_model") as final_run:
    mlflow.log_params(BEST_PARAMS)
    mlflow.log_metrics({
        "validation_wmape": final_val_wmape,
        "validation_mae": final_val_mae,
        "baseline_wmape": BASELINE_WMAPE,
        "baseline_mae": BASELINE_MAE,
        "wmape_improvement_vs_baseline": BASELINE_WMAPE - final_val_wmape,
    })
    mlflow.set_tags({
        "stage": "final_model",
        "training_split": "train_only",
        "promotion_gate": "passed",
        "produced_by": "04_lightgbm_model",
    })
    mlflow.lightgbm.log_model(final_model, name="lightgbm_model")

FINAL_RUN_ID = final_run.info.run_id
print(f"[OK] Logged final model run to MLflow. Run ID: {FINAL_RUN_ID}")

# Cell 22 — Unlock Test Split
assert MODEL_FROZEN, "Cannot unlock test: Model configuration not frozen."
assert final_val_wmape < BASELINE_WMAPE, "Cannot unlock test: Promotion gate not passed."

TEST_UNLOCKED = True
print("=" * 70)
print("[PASS] TEST SPLIT UNLOCKED")
print("All hyperparameter choices and model weights are locked.")
print("Test set will now be evaluated exactly once.")
print("=" * 70)

# Cell 23 — Create DuckDB Test View
con.execute(f"""
CREATE OR REPLACE VIEW test AS
SELECT * FROM features
WHERE target_date >= '{TEST_START}' AND target_date <= '{TEST_END}'
""")

test_rows_available = con.execute("SELECT COUNT(*) FROM test").fetchone()[0]
test_min = con.execute("SELECT MIN(target_date) FROM test").fetchone()[0]
test_max = con.execute("SELECT MAX(target_date) FROM test").fetchone()[0]

print(f"Test rows available : {test_rows_available:,}")
print(f"Test date range     : {test_min} -> {test_max}")

assert test_rows_available > 0, "Test view is empty!"
assert str(test_min)[:10] >= TEST_START, f"Test date leakage before {TEST_START}"

# Cell 24 — Load & Sample Test Data into Memory
test_rows_actual = con.execute(f"""
CREATE OR REPLACE TABLE test_ml AS
SELECT {', '.join(IDENTIFIERS + FEATURES + [TARGET])} FROM test
USING SAMPLE 70% (system, {RANDOM_SEED})
LIMIT {MAX_TEST_ROWS}
""")

test_rows_actual = con.execute("SELECT COUNT(*) FROM test_ml").fetchone()[0]
print(f"Materialized Test ML : {test_rows_actual:,} rows")

test_df = con.execute("SELECT * FROM test_ml").df()

for col in CATEGORICAL_FEATURES:
    test_df[col] = test_df[col].astype("category")

for col in NUMERIC_FEATURES + [TARGET]:
    test_df[col] = test_df[col].astype("float32")

X_test, y_test = test_df[FEATURES], test_df[TARGET]
print(f"X_test in-memory shape : {X_test.shape}")

# Cell 25 — ONE Final Holdout Evaluation on Test
assert TEST_UNLOCKED, "Test evaluation blocked: Test split is locked."

test_pred  = final_model.predict(X_test)
test_wmape = wmape(y_test, test_pred)
test_mae   = mae(y_test, test_pred)

print("=" * 70)
print("FINAL HOLDOUT TEST EVALUATION")
print("=" * 70)
print(f"LightGBM Test WMAPE  : {test_wmape:.6f}")
print(f"Baseline Test WMAPE  : {BASELINE_TEST_WMAPE:.6f}")
print(f"LightGBM Test MAE    : {test_mae:.6f}")
print(f"Baseline Test MAE    : {BASELINE_TEST_MAE:.6f}")

# Cell 26 — Baseline vs. LightGBM Model Comparison Table
model_comparison_df = pd.DataFrame([
    {"model": "MA-7 today-inclusive", "split": "validation", "WMAPE": BASELINE_WMAPE, "MAE": BASELINE_MAE},
    {"model": "LightGBM Mean Regression", "split": "validation", "WMAPE": final_val_wmape, "MAE": final_val_mae},
    {"model": "MA-7 today-inclusive", "split": "test", "WMAPE": BASELINE_TEST_WMAPE, "MAE": BASELINE_TEST_MAE},
    {"model": "LightGBM Mean Regression", "split": "test", "WMAPE": test_wmape, "MAE": test_mae},
])

print("Model Evaluation Comparison Summary:")
display(model_comparison_df)

# Cell 27 — Test Set Improvement Calculation
wmape_improvement_pct = (BASELINE_TEST_WMAPE - test_wmape) / BASELINE_TEST_WMAPE * 100.0
mae_improvement_pct   = (BASELINE_TEST_MAE - test_mae) / BASELINE_TEST_MAE * 100.0

print(f"Test WMAPE Relative Improvement : {wmape_improvement_pct:+.2f}%")
print(f"Test MAE Relative Improvement   : {mae_improvement_pct:+.2f}%")

# Cell 28 — Create Diagnostic Prediction Sample
sample_limit = min(1_000_000, len(test_df))
prediction_sample_df = test_df[IDENTIFIERS + [TARGET]].iloc[:sample_limit].copy()
prediction_sample_df["prediction"] = np.clip(test_pred[:sample_limit], 0, None).astype("float32")
prediction_sample_df["absolute_error"] = np.abs(prediction_sample_df[TARGET] - prediction_sample_df["prediction"]).astype("float32")

print(f"Diagnostic prediction sample created: {len(prediction_sample_df):,} rows")
display(prediction_sample_df.head(10))

# Cell 29 — Save Native LightGBM Booster Model
MODEL_PATH = OUTPUT_DIR / "lightgbm_model.txt"
final_model.booster_.save_model(str(MODEL_PATH))

assert MODEL_PATH.exists(), f"Model file failed to save at {MODEL_PATH}"
print(f"[OK] Saved LightGBM native booster to: {MODEL_PATH}")

# Cell 30 — Save Parquet Pipeline Result Tables
HYPERPARAMS_PATH = OUTPUT_DIR / "hyperparameter_results.parquet"
IMPORTANCE_PATH  = OUTPUT_DIR / "feature_importance.parquet"
COMPARISON_PATH  = OUTPUT_DIR / "model_comparison.parquet"
SAMPLE_PATH      = OUTPUT_DIR / "test_predictions_sample.parquet"

hyperparameter_results_df.to_parquet(HYPERPARAMS_PATH, index=False)
feature_importance_df.to_parquet(IMPORTANCE_PATH, index=False)
model_comparison_df.to_parquet(COMPARISON_PATH, index=False)
prediction_sample_df.to_parquet(SAMPLE_PATH, index=False)

print("[OK] Saved Parquet result tables:")
print(f"  Hyperparameters : {HYPERPARAMS_PATH}")
print(f"  Importance      : {IMPORTANCE_PATH}")
print(f"  Comparison      : {COMPARISON_PATH}")
print(f"  Sample Preds    : {SAMPLE_PATH}")

# Cell 31 — Construct & Save Model Contract JSON
model_contract = {
    "schema_version": 1,
    "produced_by": "04_lightgbm_model",
    "run_timestamp": datetime.utcnow().isoformat() + "Z",
    "mlflow": {
        "experiment": MLFLOW_EXPERIMENT,
        "tuning_parent_run_id": PARENT_RUN_ID,
        "final_model_run_id": FINAL_RUN_ID,
    },
    "model": {
        "algorithm": "LightGBM",
        "objective": "regression",
        "metric": "l2",
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "random_seed": RANDOM_SEED,
    },
    "split": {
        "train_end": TRAIN_END,
        "validation_start": VAL_START,
        "validation_end": VAL_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
    },
    "optuna": {
        "n_trials": N_TRIALS,
        "sampler": "TPESampler",
        "optuna_seed": OPTUNA_SEED,
        "best_trial_number": int(best_trial.number),
        "best_trial_val_wmape": best_lgbm_val_wmape,
        "direction": "minimize",
        "objective_metric": "val_wmape",
    },
    "sampling": {
        "method": "system",
        "seed": RANDOM_SEED,
        "train_rows_available": train_rows_available,
        "train_rows_used": train_rows_actual,
        "val_rows_available": val_rows_available,
        "val_rows_used": val_rows_actual,
        "test_rows_available": test_rows_available,
        "test_rows_used": test_rows_actual,
    },
    "hyperparameters": BEST_PARAMS,
    "validation": {
        "baseline_model": "MA-7 today-inclusive",
        "baseline_wmape": BASELINE_WMAPE,
        "baseline_mae": BASELINE_MAE,
        "lightgbm_wmape": final_val_wmape,
        "lightgbm_mae": final_val_mae,
        "beats_baseline": True,
    },
    "test": {
        "baseline_wmape": BASELINE_TEST_WMAPE,
        "baseline_mae": BASELINE_TEST_MAE,
        "lightgbm_wmape": test_wmape,
        "lightgbm_mae": test_mae,
        "wmape_improvement_pct": wmape_improvement_pct,
        "mae_improvement_pct": mae_improvement_pct,
    },
    "protocol": {
        "model_selection": "Optuna TPE, validation WMAPE strictly < baseline threshold_wmape.",
        "final_training": "Fitted on TRAIN split only (up to TRAIN_END) using best_iteration trees.",
        "test_evaluation": "Evaluated once on test split after gate pass and parameter freeze.",
        "leakage_rule": "target_unit_sales is excluded from FEATURES list.",
    },
}

MODEL_CONTRACT_PATH = OUTPUT_DIR / "model_contract.json"
with open(MODEL_CONTRACT_PATH, "w") as f:
    json.dump(model_contract, f, indent=2)

assert MODEL_CONTRACT_PATH.exists(), f"Model contract failed to save at {MODEL_CONTRACT_PATH}"
print(f"[OK] Saved model contract to: {MODEL_CONTRACT_PATH}")

# Cell 32 — Automated Final Verification
expected_files = [
    OUTPUT_DIR / "lightgbm_model.txt",
    OUTPUT_DIR / "model_contract.json",
    OUTPUT_DIR / "hyperparameter_results.parquet",
    OUTPUT_DIR / "feature_importance.parquet",
    OUTPUT_DIR / "model_comparison.parquet",
    OUTPUT_DIR / "test_predictions_sample.parquet",
]

for path in expected_files:
    assert path.exists(), f"Required artifact file missing: {path}"

assert MODEL_FROZEN == True, "State failure: MODEL_FROZEN is False"
assert TEST_UNLOCKED == True, "State failure: TEST_UNLOCKED is False"
assert final_val_wmape < BASELINE_WMAPE, "Gate failure: final_val_wmape >= BASELINE_WMAPE"

print("=" * 70)
print("[PASS] ALL PIPELINE CHECKS PASSED")
print(f"  Artifacts verified : {len(expected_files)} files present in {OUTPUT_DIR}")
print(f"  Val WMAPE          : {final_val_wmape:.6f} < {BASELINE_WMAPE:.6f}")
print(f"  Test WMAPE         : {test_wmape:.6f} (Baseline: {BASELINE_TEST_WMAPE:.6f})")
print(f"  Test Improvement   : {wmape_improvement_pct:+.2f}%")
print("=" * 70)