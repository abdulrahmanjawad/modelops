import os
import sys
import time
import json
import pickle
import hashlib
import joblib
import numpy as np
import pandas as pd
from teradatasql import connect

def get_artifact_path(filename: str) -> str:
    """Resolves artifact paths relative to this script."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    return os.path.join(base_dir, filename)

def hash_bucket(val, n_buckets: int):
    """Stateless MD5 hash bucketing for VAS_DESC."""
    return int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16) % n_buckets

def load_cascade():
    """Loads stage 1, stage 2, and cascade configuration."""
    stage1_model = joblib.load(get_artifact_path("stage1_model.joblib"))
    stage2_model = joblib.load(get_artifact_path("stage2_model.joblib"))
    with open(get_artifact_path("cascade_config.json"), "r") as f:
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

def run_cascade_scoring():
    # ------------------------------------------------------------------
    # CONFIGURATION & PARAMETERS
    # ------------------------------------------------------------------
    TD_HOST = os.environ.get("TD_HOST", "172.16.15.19")
    TD_USER = os.environ.get("TD_USER", "UP_LOAD_SAS")
    TD_PASSWORD = os.environ.get("TD_PASSWORD")
    STATE_DATE = os.environ.get("STATE_DATE", "2026-08-30")
    OUTPUT_TABLE = os.environ.get("OUTPUT_TABLE", "AD_VEW_SAS_ETL.TOS_PREDICTIONS_OUTPUT")
    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.648"))
    SAMPLE_SIZE = os.environ.get("SAMPLE_SIZE", "")  # Set e.g. "SAMPLE 2000" for dry-runs

    if not TD_PASSWORD:
        print("CRITICAL: TD_PASSWORD environment variable is missing.")
        sys.exit(1)

    print(f"--- Starting TOS Stay Scoring ---")
    print(f"STATE_DATE: {STATE_DATE} | HOST: {TD_HOST} | OUTPUT: {OUTPUT_TABLE}")

    # ------------------------------------------------------------------
    # PHASE 1: DATA EXTRACTION FROM TERADATA
    # ------------------------------------------------------------------
    sample_clause = f"SAMPLE {SAMPLE_SIZE}" if SAMPLE_SIZE else ""
    dsl_sql = f"SELECT * FROM AD_VEW_SAS_ETL.BB_FBB_SEG_MODEL_ADS WHERE STATE_DATE = '{STATE_DATE}' {sample_clause}"
    ff_sql = f"SELECT * FROM AD_VEW_SAS_ETL.FBB_FF_ADS WHERE STATE_DATE = '{STATE_DATE}' {sample_clause}"

    print("Connecting to Teradata Vantage for ADS extraction...")
    with connect(host=TD_HOST, user=TD_USER, password=TD_PASSWORD) as conn:
        df_dsl = pd.read_sql(dsl_sql, conn)
        df_ff = pd.read_sql(ff_sql, conn)

    if df_dsl.empty:
        print(f"No DSL records returned for {STATE_DATE}. Halting execution.")
        sys.exit(0)

    print(f"Loaded {len(df_dsl):,} DSL records and {len(df_ff):,} FF records.")

    # ------------------------------------------------------------------
    # PHASE 2: MERGING & TYPE NORMALIZATION
    # ------------------------------------------------------------------
    ff_keep_cols = [
        'STATE_DATE', 'BILLING_ACCT_ID_NUM', 'AAA_SESSION_COMPLETED_FLAG',
        'AVG_MTBE_M1','AVG_MTBE_M2','AVG_MTBE_M3','AVG_NUM_INTERRUPT_ALARMS_M1',
        'AVG_NUM_INTERRUPT_ALARMS_M2','AVG_NUM_INTERRUPT_ALARMS_M3','BLACKLISTED_FLAG',
        'BOLT_ON_FLAG','CORPORATE_CUSTOMER_FLAG','DAYTME_OTG_DUR_M1','DAYTME_OTG_DUR_M2',
        'DAYTME_OTG_DUR_M3','HIGH_LOW_MRC_M1','HIGH_LOW_MRC_M2','HIGH_LOW_MRC_M3',
        'INSTALLATION_CHARGES_FLAG','INTL_INC_DUR_M1','INTL_INC_DUR_M2','INTL_INC_DUR_M3',
        'M1_FAILURE_REASON_CD','M1_FAILURE_REASON_DESC','MAPPED_PON_ID','MOB_INC_DUR_M1',
        'MOB_INC_DUR_M2','MOB_INC_DUR_M3','NGHTTME_OTG_DUR_M1','NGHTTME_OTG_DUR_M2',
        'NGHTTME_OTG_DUR_M3','NON_BOLT_ON_FLAG','NON_RESIDENTIAL_FLAG','OFFNET_INC_DUR_M1',
        'OFFNET_INC_DUR_M2','OFFNET_INC_DUR_M3','ONNET_INC_DUR_M1','ONNET_INC_DUR_M2',
        'ONNET_INC_DUR_M3','PAIRED_PON_PROTECTION_STATUS','PON_AVG_DOWNSTREAM_BANDWIDTH',
        'PON_AVG_UPSTREAM_BANDWIDTH','PON_BW_MARGIN_FACTOR','PON_IS_IN_COMPLAINT_BASE',
        'PON_MAX_DOWNSTREAM_BANDWIDTH','PON_MAX_UPSTREAM_BANDWIDTH','PON_PROTECTION_STATUS',
        'POWERLEVEL_M1_CAT','POWERLEVEL_M2_CAT','POWERLEVEL_M3_CAT','SPEED_STEEPNESS_CAT',
        'SPEED_TREND_SLOPE_MBPS','TARGET_COMPLAINT_FLAG','TOTAL_USAGE_M1_CAT',
        'TOTAL_USAGE_M2_CAT','TOTAL_USAGE_M3_CAT','VAS_DESC','WKDAY_DUR_M1','WKDAY_DUR_M2',
        'WKDAY_DUR_M3','WKND_DUR_M1','WKND_DUR_M2','WKND_DUR_M3',
    ]

    df_ff = df_ff[[c for c in ff_keep_cols if c in df_ff.columns]].copy()
    df_ff['ACCESS_TYPE'] = 'FF'

    df_merged = df_dsl.merge(df_ff, on=["BILLING_ACCT_ID_NUM", "STATE_DATE"], how="left")
    df_merged['ACCESS_TYPE'] = df_merged['ACCESS_TYPE'].fillna('xDSL')
    df_merged = df_merged.fillna(0)

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
            mask = model_df[col].notna()
            model_df.loc[mask, col] = model_df.loc[mask, col].astype(str)

    # ------------------------------------------------------------------
    # PHASE 3: FEATURE ENGINEERING (USING FROZEN ARTIFACTS)
    # ------------------------------------------------------------------
    print("Loading preprocessing pipeline artifacts...")
    with open(get_artifact_path("tos_pipeline_artifacts.pkl"), "rb") as f:
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
        'AVG_MTBE_': 1.8e18,
        'HAPPINESS_IDX_': lambda s: s.where(s <= 100, np.nan),
        'MTBR_VAL_': 0.2778,
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
    stage1_model, stage2_model, cfg = load_cascade()
    combined_probs = cascade_predict(stage1_model, stage2_model, df_aug, cfg)

    # ------------------------------------------------------------------
    # PHASE 5: WRITE-BACK TO TERADATA VANTAGE
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        "BILLING_ACCT_ID_NUM": df_merged["BILLING_ACCT_ID_NUM"].values,
        "STATE_DATE": STATE_DATE,
        "PREDICTION": combined_probs,
        "FLAGGED": (combined_probs >= SCORE_THRESHOLD).astype(int),
        "SCORED_AT": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    flagged_df = results[results["FLAGGED"] == 1].sort_values("PREDICTION", ascending=False)
    print(f"Scoring complete. Total flagged accounts (>= {SCORE_THRESHOLD}): {len(flagged_df):,}")

    if flagged_df.empty:
        print("No flagged accounts to persist.")
        return

    print(f"Connecting to write {len(flagged_df):,} records into {OUTPUT_TABLE}...")
    with connect(host=TD_HOST, user=TD_USER, password=TD_PASSWORD) as conn:
        with conn.cursor() as cursor:
            insert_sql = f"""
                INSERT INTO {OUTPUT_TABLE} 
                (BILLING_ACCT_ID_NUM, STATE_DATE, PREDICTION, FLAGGED, SCORED_AT) 
                VALUES (?, ?, ?, ?, ?)
            """
            
            # Chunk insertions to prevent socket timeouts on large subscriber bases
            CHUNK_SIZE = 50000
            total_records = len(flagged_df)
            rows = flagged_df.values.tolist()
            
            for i in range(0, total_records, CHUNK_SIZE):
                chunk = rows[i:i + CHUNK_SIZE]
                cursor.executemany(insert_sql, chunk)
                conn.commit()
                print(f"Committed batch: {min(i + CHUNK_SIZE, total_records):,} / {total_records:,}")

    print(f"SUCCESS: Pipeline successfully loaded into {OUTPUT_TABLE}.")

if __name__ == "__main__":
    run_cascade_scoring()