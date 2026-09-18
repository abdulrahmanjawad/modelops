import os
import sys
import json
import pickle
import hashlib

import numpy as np
import pandas as pd

from teradataml import DataFrame, copy_to_sql
from tmo import (
    tmo_create_context, 
    ModelContext
)


def hash_bucket(val, n_buckets: int):
    """Stateless MD5 hash bucketing for VAS_DESC."""
    return int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16) % n_buckets


def run_task(context: ModelContext, **kwargs):
    tmo_create_context()

    # ------------------------------------------------------------------
    # CONFIGURATION & PARAMETERS
    # ------------------------------------------------------------------
    artifact_path   = context.artifact_input_path

    # OUTPUT_TABLE = os.environ.get("OUTPUT_TABLE", "DEMO_USER.TOS_ADS_INPUT_DATA")
    SCHEMA_NAME = "DEMO_USER"
    OUTPUT_TABLE_NAME = "TOS_ADS_INPUT_DATA"
    SAMPLE_SIZE = os.environ.get("SAMPLE_SIZE", "")
    STATE_DATE = os.environ.get("STATE_DATE", "2026-08-30")

    print(f"--- Starting TOS Stay Feature Engineering ---")
    print(f"STATE_DATE: {STATE_DATE}")

    # ------------------------------------------------------------------
    # PHASE 1: DATA EXTRACTION FROM TERADATA
    # ------------------------------------------------------------------
    sample_clause = f"SAMPLE {SAMPLE_SIZE}" if SAMPLE_SIZE else ""
    dsl_sql = f"SELECT * FROM DEMO_USER.BB_FBB_SEG_MODEL_ADS WHERE STATE_DATE = '{STATE_DATE}' {sample_clause}"
    ff_sql = f"SELECT * FROM DEMO_USER.FBB_FF_ADS WHERE STATE_DATE = '{STATE_DATE}' {sample_clause}"

    df_dsl = DataFrame.from_query(dsl_sql).to_pandas(all_rows=True)
    df_ff = DataFrame.from_query(ff_sql).to_pandas(all_rows=True)

    if df_dsl.empty:
        print(f"No DSL records returned for {STATE_DATE}. Halting execution.")
        return 0

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
    # PHASE 4: WRITE FEATURE ENGINEERED DATA TO TERADATA VIA TMO
    # ------------------------------------------------------------------
    print(f"Writing {len(df_aug)} records into {SCHEMA_NAME}.{OUTPUT_TABLE_NAME}.")
    copy_to_sql(
        df=df_aug,
        schema_name=SCHEMA_NAME,
        table_name=OUTPUT_TABLE_NAME,
        if_exists="append", # "append" to existing table if it exists, "replace" to drop and recreate
        primary_index=["BILLING_ACCT_ID_NUM"]
    )