import json
import pickle
import hashlib

import joblib
import numpy as np
import pandas as pd

from teradataml import copy_to_sql, DataFrame
try:
    from tmo import (
        tmo_create_context,
        record_scoring_stats,
        ModelContext,
    )
except ImportError:
    from aoa import record_scoring_stats, ModelContext
    from aoa import aoa_create_context as tmo_create_context


def hash_bucket(val, n_buckets: int):
    """Stateless MD5 hash bucketing for VAS_DESC."""
    return int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16) % n_buckets


def load_cascade(artifact_path: str):
    stage1_model = joblib.load(f"{artifact_path}/stage1_model.joblib")
    stage2_model = joblib.load(f"{artifact_path}/stage2_model.joblib")
    with open(f"{artifact_path}/cascade_config.json", "r") as f:
        cfg = json.load(f)
    return stage1_model, stage2_model, cfg


def cascade_predict(stage1_model, stage2_model, X: pd.DataFrame, cfg: dict) -> np.ndarray:
    missing_s1 = [c for c in cfg["stage1_feature_cols"] if c not in X.columns]
    missing_s2 = [c for c in cfg["stage2_feature_cols"] if c not in X.columns]
    if missing_s1 or missing_s2:
        raise ValueError(f"Missing stage1 cols: {missing_s1}\nMissing stage2 cols: {missing_s2}")

    s1_probs = stage1_model.predict_proba(X[cfg["stage1_feature_cols"]])[:, 1]
    combined = np.zeros(len(X))
    passed = s1_probs >= cfg["stage1_cutoff"]

    if passed.sum() > 0:
        combined[passed] = stage2_model.predict_proba(
            X.loc[passed, cfg["stage2_feature_cols"]]
        )[:, 1]
    return combined


def score(context: ModelContext, **kwargs):

    tmo_create_context()

    # artifact_path   = context.artifact_input_path
    # entity_key      = context.dataset_info.entity_key       # "BILLING_ACCT_ID_NUM"
    # target_name     = context.dataset_info.target_names[0]  # "CHURN_PROB_30_DAY"
    SCORE_THRESHOLD = float(context.hyperparams.get("score_threshold", 0.648))
    STATE_DATE = context.hyperparams.get("state_date", "2026-08-30")
    OUTPUT_TABLE = f"{context.dataset_info.predictions_database}.{context.dataset_info.predictions_table}"

    artifact_path = "./model_modules"

    # ------------------------------------------------------------------
    # PHASE 1: DATA LOADING
    # The AOA dataset SQL is the LEFT JOIN of BB_FBB_SEG_MODEL_ADS and
    # FBB_FF_ADS, with ACCESS_TYPE derived via CASE WHEN.
    # ------------------------------------------------------------------
    print("Loading features via AOA dataset SQL (DSL LEFT JOIN FF)...")
    print(f"Dataset SQL: {context.dataset_info.sql}")
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)

    if features_pdf.empty:
        print("Dataset SQL returned 0 rows. Nothing to score.")
        return

    print(f"Loaded {len(features_pdf):,} subscriber records.")

    # Capture billing IDs before any column is dropped
    billing_ids = features_pdf["BILLING_ACCT_ID_NUM"].values.copy()

    # ------------------------------------------------------------------
    # PHASE 2: MERGING & TYPE NORMALIZATION
    # (Merge is handled in SQL; the steps below mirror scoring.py exactly)
    # ------------------------------------------------------------------
    df_merged = features_pdf.fillna(0)

    id_exclude = ['MDN', 'MDN_GPON', 'USER_NAME', 'BILLING_ACCT_ID']
    model_df = df_merged.drop(columns=id_exclude, errors="ignore")

    mixed_type_cols = [
        'HIGH_LOW_MRC_M1', 'HIGH_LOW_MRC_M2', 'HIGH_LOW_MRC_M3',
        'M1_FAILURE_REASON_DESC', 'MAPPED_PON_ID', 'PAIRED_PON_PROTECTION_STATUS',
        'PON_BW_MARGIN_FACTOR', 'PON_IS_IN_COMPLAINT_BASE', 'PON_PROTECTION_STATUS',
        'POWERLEVEL_M1_CAT', 'POWERLEVEL_M2_CAT', 'POWERLEVEL_M3_CAT',
        'SPEED_STEEPNESS_CAT', 'TOTAL_USAGE_M1_CAT', 'TOTAL_USAGE_M2_CAT',
        'TOTAL_USAGE_M3_CAT', 'VAS_DESC'
    ]
    for col in mixed_type_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype(object)  # widen dtype first
            mask = model_df[col].notna()
            model_df.loc[mask, col] = model_df.loc[mask, col].astype(str)

    # ------------------------------------------------------------------
    # PHASE 3: FEATURE ENGINEERING (USING FROZEN ARTIFACTS)
    # ------------------------------------------------------------------
    print("Loading preprocessing pipeline artifacts...")
    with open(f"{artifact_path}/tos_pipeline_artifacts.pkl", "rb") as f:
        tos_artifacts = pickle.load(f)

    constant_cols_dropped = tos_artifacts['constant_cols_dropped']
    categorical_cols      = tos_artifacts['categorical_cols']
    frozen_categories     = tos_artifacts['frozen_categories']
    N_HASH_BUCKETS        = tos_artifacts['vas_desc_hash_buckets']
    DATE_COL              = tos_artifacts['date_col']

    df_aug = model_df.copy()
    df_aug[DATE_COL] = pd.to_datetime(df_aug[DATE_COL])

    obj_cols = df_aug.select_dtypes(include='object').columns.tolist()
    for c in obj_cols:
        df_aug[c] = df_aug[c].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan})

    cols_to_drop = [c for c in constant_cols_dropped if c in df_aug.columns]
    df_aug = df_aug.drop(columns=cols_to_drop)

    sentinel_rules = {
        'AVG_MTBE_':     1.8e18,
        'HAPPINESS_IDX_': lambda s: s.where(s <= 100, np.nan),
        'MTBR_VAL_':     0.2778,
    }
    for pattern, rule in sentinel_rules.items():
        matching_cols = [c for c in df_aug.columns if pattern in c]
        for c in matching_cols:
            df_aug[c] = rule(df_aug[c]) if callable(rule) else df_aug[c].replace(rule, np.nan)

    for c in categorical_cols:
        if c not in df_aug.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_aug[c]):
            df_aug[c] = (
                df_aug[c].astype('Int64').astype(str).replace('<NA>', np.nan)
                if (df_aug[c].dropna() % 1 == 0).all()
                else df_aug[c].astype(str)
            )
        df_aug[c] = pd.Categorical(df_aug[c], categories=frozen_categories[c])

    if 'VAS_DESC' in df_aug.columns:
        hash_vals = df_aug['VAS_DESC'].map(lambda x: hash_bucket(x, N_HASH_BUCKETS))
        df_aug['VAS_DESC_HASH'] = pd.Categorical(hash_vals, categories=list(range(N_HASH_BUCKETS)))
        df_aug = df_aug.drop(columns=['VAS_DESC'])

    # ------------------------------------------------------------------
    # PHASE 4: CASCADE MODEL INFERENCE
    # ------------------------------------------------------------------
    print("Executing two-stage cascade inference...")
    stage1_model, stage2_model, cfg = load_cascade(artifact_path)
    combined_probs = cascade_predict(stage1_model, stage2_model, df_aug, cfg)

    # ------------------------------------------------------------------
    # PHASE 5: WRITE FLAGGED PREDICTIONS TO TERADATA VIA AOA
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        "BILLING_ACCT_ID_NUM": df_merged["BILLING_ACCT_ID_NUM"].values,
        "STATE_DATE": STATE_DATE,
        "PREDICTION": combined_probs,
        "FLAGGED": (combined_probs >= SCORE_THRESHOLD).astype(int),
        "SCORED_AT": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        # ,"JOB_ID": context.job_id 
    })

    flagged_df = results[results["FLAGGED"] == 1].sort_values("PREDICTION", ascending=False)
    print(f"Scoring complete. Total flagged accounts (>= {SCORE_THRESHOLD}): {len(flagged_df):,}")

    if flagged_df.empty:
        print("No flagged accounts to persist.")
        return

    copy_to_sql(
        df=flagged_df,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        # index=False,
        if_exists="append",
    )

    print(f"SUCCESS: Pipeline successfully loaded into {OUTPUT_TABLE}.")