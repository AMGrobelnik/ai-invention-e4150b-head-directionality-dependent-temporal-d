#!/usr/bin/env python3
"""
Sentence-Length-Stratified R² Analysis, Head-Type Count Reconciliation,
and 5.25% Residual Variance Contextualization.

Evaluates the Phase 1 experiment output (870K sentences, 335 treebanks) with:
  1. Sentence-length-stratified R² (4 bins × head type)
  2. Head-type count reconciliation (all vs n≥30, WALS vs left_proportion)
  3. Residual variance contextualization (incremental R², benchmarks)
  4. Monotonicity test (Spearman, linear slope, bootstrap CI)
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scipy import stats as scipy_stats
import statsmodels.api as sm

# ============================================================
# LOGGING SETUP
# ============================================================
WORKSPACE = Path(__file__).parent.resolve()
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION & MEMORY LIMITS
# ============================================================

def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.60 * 1024**3)  # 60% for safety

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET_BYTES / 1e9:.1f}GB")

try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
    logger.info("Memory limit set via resource.setrlimit")
except (ValueError, OSError) as e:
    logger.warning(f"Could not set memory limit: {e}")

# ============================================================
# PATHS
# ============================================================
EXP_DIR = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus")
DATA_DIR = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
SENTENCE_CSV_1 = EXP_DIR / "sentence_imb_features" / "sentence_imb_features_1.csv"
SENTENCE_CSV_2 = EXP_DIR / "sentence_imb_features" / "sentence_imb_features_2.csv"
METHOD_OUT = EXP_DIR / "full_method_out.json"
TYPOLOGY_FILE = DATA_DIR / "full_data_out.json"

# Length bins as specified in the plan
LENGTH_BINS = [(8, 12), (13, 20), (21, 35), (36, None)]
BIN_LABELS = ["8-12", "13-20", "21-35", "36+"]
BIN_MIDPOINTS = [10.0, 16.5, 28.0, 50.0]  # midpoint for 36+ approximated

# Regression predictors
PREDICTORS = ["mean_dd", "dd_variance", "dd_skewness", "sentence_length", "tree_depth", "max_arity", "mean_arity"]

# Benchmark effect sizes from quantitative linguistics literature
BENCHMARK_EFFECTS = {
    "heavy_np_shift": {
        "r2_range": "0.02-0.05",
        "r2_mid": 0.035,
        "source": "Wasow 2002; Arnold et al. 2000 (mixed-effects models for heavy-NP shift ordering)",
        "notes": "Typical incremental R2 for constituent weight in ordering alternations"
    },
    "given_before_new": {
        "r2_range": "0.03-0.08",
        "r2_mid": 0.055,
        "source": "Bresnan et al. 2007 (dative alternation); Arnold et al. 2000",
        "notes": "Information status contribution to ordering in dative alternation logistic models"
    },
    "uid_word_order": {
        "r2_range": "0.01-0.04",
        "r2_mid": 0.025,
        "source": "Clark et al. 2023 (UID and word order); Maurits et al. 2010",
        "notes": "Effect sizes reported as z-scores/Cohen's d; converted to approximate R2 via r2~d2/(d2+4)"
    },
    "ddm_dependency_length": {
        "r2_range": "0.02-0.06",
        "r2_mid": 0.04,
        "source": "Futrell et al. 2015 (DDM); Gildea & Temperley 2010",
        "notes": "Dependency length minimization effect in ordering alternations; z-scores converted"
    },
}

# ============================================================
# DATA LOADING
# ============================================================

def load_sentence_data(max_rows: int | None = None) -> pd.DataFrame:
    """Load sentence-level IMB features from CSV shards."""
    logger.info("Loading sentence-level CSV data...")
    t0 = time.time()

    cols_needed = [
        "treebank_id", "sentence_length", "tree_depth",
        "max_arity", "mean_arity", "mean_dd", "dd_variance", "dd_skewness",
        "max_imb", "max_imb_enc", "head_type", "modality", "wals_word_order",
        "language_family",
    ]

    dtypes = {
        "treebank_id": str,
        "head_type": str,
        "modality": str,
        "wals_word_order": str,
        "language_family": str,
        "sentence_length": "int32",
        "tree_depth": "int16",
        "max_arity": "int16",
        "max_imb": "float32",
        "max_imb_enc": "float32",
        "mean_dd": "float32",
        "dd_variance": "float32",
        "dd_skewness": "float32",
        "mean_arity": "float32",
    }

    dfs = []
    for csv_path in [SENTENCE_CSV_1, SENTENCE_CSV_2]:
        df_chunk = pd.read_csv(
            csv_path, usecols=cols_needed, dtype=dtypes,
            nrows=max_rows,
        )
        dfs.append(df_chunk)
        logger.info(f"  Loaded {len(df_chunk)} rows from {csv_path.name}")

    df = pd.concat(dfs, ignore_index=True)
    # Convert string columns to category after concat to avoid chunk mismatch
    for col in ["treebank_id", "head_type", "modality", "wals_word_order", "language_family"]:
        df[col] = df[col].astype("category")
    elapsed = time.time() - t0
    logger.info(f"Loaded {len(df)} sentences total in {elapsed:.1f}s, mem={df.memory_usage(deep=True).sum() / 1e6:.1f}MB")
    return df


def load_method_output() -> dict:
    """Load per-treebank method output."""
    logger.info("Loading method output...")
    data = json.loads(METHOD_OUT.read_text())
    logger.info(f"Loaded method output: {len(data['datasets'][0]['examples'])} treebanks")
    return data


def load_typology() -> dict[str, dict]:
    """Load typological metadata, keyed by treebank_id."""
    logger.info("Loading typological metadata...")
    raw = json.loads(TYPOLOGY_FILE.read_text())
    typo = {}
    for ex in raw["datasets"][0]["examples"]:
        out = json.loads(ex["output"])
        typo[out["treebank_id"]] = out
    logger.info(f"Loaded typology for {len(typo)} treebanks")
    return typo


# ============================================================
# METRIC GROUP 1: Sentence-Length-Stratified R²
# ============================================================

def assign_length_bin(lengths: pd.Series) -> pd.Series:
    """Assign each sentence to a length bin."""
    conditions = [
        (lengths >= 8) & (lengths <= 12),
        (lengths >= 13) & (lengths <= 20),
        (lengths >= 21) & (lengths <= 35),
        (lengths >= 36),
    ]
    return pd.Series(
        np.select(conditions, BIN_LABELS, default="excluded"),
        index=lengths.index,
        dtype="category",
    )


def fit_ols_r2(df: pd.DataFrame, y_col: str) -> dict:
    """Fit OLS regression and return R², coefficients, residuals."""
    if len(df) < 10:
        return {"r2": np.nan, "n": len(df), "residuals": np.array([])}

    X = df[PREDICTORS].values.astype(np.float64)
    y = df[y_col].values.astype(np.float64)

    # Remove rows with NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return {"r2": np.nan, "n": len(y), "residuals": np.array([])}

    X_const = sm.add_constant(X)
    try:
        model = sm.OLS(y, X_const).fit()
        residuals = model.resid
        return {
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "n": len(y),
            "residuals": residuals,
            "coefficients": {name: float(model.params[i]) for i, name in enumerate(["const"] + PREDICTORS)},
        }
    except Exception as e:
        logger.warning(f"OLS failed: {e}")
        return {"r2": np.nan, "n": len(y), "residuals": np.array([])}


def compute_length_stratified_r2(df: pd.DataFrame) -> dict:
    """Compute R² within each length bin, overall and by head type."""
    logger.info("Computing length-stratified R² analysis...")
    t0 = time.time()

    df = df.copy()
    df["length_bin"] = assign_length_bin(df["sentence_length"])
    df_binned = df[df["length_bin"] != "excluded"]
    logger.info(f"  Sentences in bins: {len(df_binned)} / {len(df)} ({100*len(df_binned)/len(df):.1f}%)")

    results = {
        "r2_std_by_bin": {},
        "r2_enc_by_bin": {},
        "residual_variance_by_bin": {},
        "mean_abs_residualized_max_imb_by_bin": {},
        "n_sentences_per_bin": {},
        # Cross-stratified
        "r2_std_by_headtype_bin": {},
        "r2_enc_by_headtype_bin": {},
        "n_by_headtype_bin": {},
    }

    # Overall by bin
    for label in BIN_LABELS:
        bin_df = df_binned[df_binned["length_bin"] == label]
        logger.info(f"  Bin {label}: {len(bin_df)} sentences")

        res_std = fit_ols_r2(bin_df, "max_imb")
        res_enc = fit_ols_r2(bin_df, "max_imb_enc")

        results["r2_std_by_bin"][label] = res_std["r2"]
        results["r2_enc_by_bin"][label] = res_enc["r2"]
        results["n_sentences_per_bin"][label] = res_std["n"]

        if len(res_std["residuals"]) > 0:
            results["residual_variance_by_bin"][label] = float(np.var(res_std["residuals"]))
            results["mean_abs_residualized_max_imb_by_bin"][label] = float(np.mean(np.abs(res_std["residuals"])))
        else:
            results["residual_variance_by_bin"][label] = np.nan
            results["mean_abs_residualized_max_imb_by_bin"][label] = np.nan

    # R² deltas
    r2_std_vals = results["r2_std_by_bin"]
    r2_enc_vals = results["r2_enc_by_bin"]
    results["r2_delta_std"] = (r2_std_vals.get("8-12", np.nan) or np.nan) - (r2_std_vals.get("36+", np.nan) or np.nan)
    results["r2_delta_enc"] = (r2_enc_vals.get("8-12", np.nan) or np.nan) - (r2_enc_vals.get("36+", np.nan) or np.nan)

    # Cross-stratification by head type × length bin
    head_types = ["head_final", "head_initial", "mixed"]
    for ht in head_types:
        results["r2_std_by_headtype_bin"][ht] = {}
        results["r2_enc_by_headtype_bin"][ht] = {}
        results["n_by_headtype_bin"][ht] = {}
        ht_df = df_binned[df_binned["head_type"] == ht]
        for label in BIN_LABELS:
            bin_ht_df = ht_df[ht_df["length_bin"] == label]
            res_std = fit_ols_r2(bin_ht_df, "max_imb")
            res_enc = fit_ols_r2(bin_ht_df, "max_imb_enc")
            results["r2_std_by_headtype_bin"][ht][label] = res_std["r2"]
            results["r2_enc_by_headtype_bin"][ht][label] = res_enc["r2"]
            results["n_by_headtype_bin"][ht][label] = res_std["n"]

    elapsed = time.time() - t0
    logger.info(f"Length-stratified R² done in {elapsed:.1f}s")
    return results


# ============================================================
# METRIC GROUP 2: Head-Type Count Reconciliation
# ============================================================

def compute_head_type_reconciliation(
    method_data: dict,
    typology: dict[str, dict],
) -> dict:
    """Reconcile head-type counts: all 335 vs n>=30, WALS vs left_proportion."""
    logger.info("Computing head-type count reconciliation...")

    examples = method_data["datasets"][0]["examples"]

    # Build per-treebank info
    treebanks = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        tb_id = inp["treebank_id"]
        typo = typology.get(tb_id, {})

        treebanks.append({
            "treebank_id": tb_id,
            "head_type": inp["head_type"],
            "wals_word_order": inp.get("wals_word_order", "") or "",
            "n_sentences": out["n_sentences"],
            "r2_std": out.get("r2_std"),
            "r2_enc": out.get("r2_enc"),
            "left_proportion": typo.get("head_direction_left_proportion"),
        })

    df_tb = pd.DataFrame(treebanks)

    # Group 2a: head_type_counts_all (ALL 335 treebanks)
    head_type_counts_all = df_tb["head_type"].value_counts().to_dict()

    # Group 2b: head_type_counts_with_r2 (n >= 30 for regression)
    df_with_r2 = df_tb[df_tb["n_sentences"] >= 30]
    head_type_counts_with_r2 = df_with_r2["head_type"].value_counts().to_dict()

    # Group 2c: Classification method breakdown
    # WALS-direct: SOV -> head_final, SVO/VSO -> head_initial
    # left_proportion fallback: lp<0.4 -> head_final, lp>0.6 -> head_initial, else -> mixed
    wals_classified = df_tb[df_tb["wals_word_order"].isin(["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"])]
    no_wals = df_tb[~df_tb["wals_word_order"].isin(["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"])]

    counts_by_method = {
        "wals_direct_count": len(wals_classified),
        "left_proportion_fallback_count": len(no_wals),
        "wals_head_final": len(wals_classified[wals_classified["wals_word_order"] == "SOV"]),
        "wals_head_initial": len(wals_classified[wals_classified["wals_word_order"].isin(["SVO", "VSO", "VOS"])]),
        "wals_other": len(wals_classified[wals_classified["wals_word_order"].isin(["OVS", "OSV"])]),
    }

    # Left-proportion classification for fallback treebanks
    lp_valid = no_wals[no_wals["left_proportion"].notna()]
    counts_by_method["lp_fallback_head_final"] = int((lp_valid["left_proportion"] < 0.4).sum())
    counts_by_method["lp_fallback_head_initial"] = int((lp_valid["left_proportion"] > 0.6).sum())
    counts_by_method["lp_fallback_mixed"] = int(
        ((lp_valid["left_proportion"] >= 0.4) & (lp_valid["left_proportion"] <= 0.6)).sum()
    )
    counts_by_method["lp_fallback_no_data"] = int(no_wals["left_proportion"].isna().sum())

    # Group 2d: Classification consistency check
    # For treebanks with BOTH WALS and left_proportion
    both = df_tb[
        df_tb["wals_word_order"].isin(["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"])
        & df_tb["left_proportion"].notna()
    ].copy()

    def wals_to_headtype(wo: str) -> str:
        if wo == "SOV":
            return "head_final"
        elif wo in ("SVO", "VSO", "VOS"):
            return "head_initial"
        else:
            return "other"

    def lp_to_headtype(lp: float) -> str:
        if lp < 0.4:
            return "head_final"
        elif lp > 0.6:
            return "head_initial"
        else:
            return "mixed"

    if len(both) > 0:
        both["wals_ht"] = both["wals_word_order"].apply(wals_to_headtype)
        both["lp_ht"] = both["left_proportion"].apply(lp_to_headtype)
        agree = (both["wals_ht"] == both["lp_ht"]).sum()
        disagree = len(both) - agree
        consistency = {
            "n_treebanks_with_both": len(both),
            "agree": int(agree),
            "disagree": int(disagree),
            "agreement_rate": round(float(agree / len(both)), 4) if len(both) > 0 else 0.0,
            "disagreement_details": [],
        }
        # Log disagreements
        disagreements = both[both["wals_ht"] != both["lp_ht"]]
        for _, row in disagreements.iterrows():
            consistency["disagreement_details"].append({
                "treebank_id": row["treebank_id"],
                "wals_word_order": row["wals_word_order"],
                "wals_ht": row["wals_ht"],
                "left_proportion": round(float(row["left_proportion"]), 4),
                "lp_ht": row["lp_ht"],
                "assigned_head_type": row["head_type"],
            })
    else:
        consistency = {
            "n_treebanks_with_both": 0,
            "agree": 0,
            "disagree": 0,
            "agreement_rate": 0.0,
            "disagreement_details": [],
        }

    # Group 2e: R² by head type (corrected, both groupings)
    r2_by_head_type_all = {}
    r2_by_head_type_n30 = {}
    for ht in ["head_final", "head_initial", "mixed"]:
        # All treebanks
        ht_all = df_tb[df_tb["head_type"] == ht]
        ht_r2 = ht_all["r2_std"].dropna()
        r2_by_head_type_all[ht] = {
            "n_treebanks": len(ht_all),
            "n_with_r2": len(ht_r2),
            "r2_std_mean": round(float(ht_r2.mean()), 4) if len(ht_r2) > 0 else None,
            "r2_std_median": round(float(ht_r2.median()), 4) if len(ht_r2) > 0 else None,
        }
        # n>=30 only
        ht_n30 = df_with_r2[df_with_r2["head_type"] == ht]
        ht_r2_n30 = ht_n30["r2_std"].dropna()
        ht_r2_enc_n30 = ht_n30["r2_enc"].dropna()
        r2_by_head_type_n30[ht] = {
            "n_treebanks": len(ht_n30),
            "r2_std_mean": round(float(ht_r2_n30.mean()), 4) if len(ht_r2_n30) > 0 else None,
            "r2_std_median": round(float(ht_r2_n30.median()), 4) if len(ht_r2_n30) > 0 else None,
            "r2_enc_mean": round(float(ht_r2_enc_n30.mean()), 4) if len(ht_r2_enc_n30) > 0 else None,
            "r2_enc_median": round(float(ht_r2_enc_n30.median()), 4) if len(ht_r2_enc_n30) > 0 else None,
        }

    return {
        "head_type_counts_all": head_type_counts_all,
        "head_type_counts_with_r2": head_type_counts_with_r2,
        "n_treebanks_all": len(df_tb),
        "n_treebanks_with_r2": len(df_with_r2),
        "head_type_counts_by_method": counts_by_method,
        "classification_consistency_check": consistency,
        "r2_by_head_type_all_335": r2_by_head_type_all,
        "r2_by_head_type_n30_subset": r2_by_head_type_n30,
    }


# ============================================================
# METRIC GROUP 3: Residual Variance Contextualization
# ============================================================

def compute_incremental_r2(df: pd.DataFrame) -> dict:
    """Compute incremental R² from adding each predictor one at a time (hierarchical)."""
    logger.info("Computing incremental R² (hierarchical regression)...")

    # Remove rows with any NaN/Inf
    cols = PREDICTORS + ["max_imb", "max_imb_enc"]
    df_clean = df[cols].dropna()
    mask = np.isfinite(df_clean.values).all(axis=1)
    df_clean = df_clean[mask]
    logger.info(f"  Clean rows for incremental R²: {len(df_clean)}")

    y_std = df_clean["max_imb"].values.astype(np.float64)
    y_enc = df_clean["max_imb_enc"].values.astype(np.float64)

    incremental_std = {}
    incremental_enc = {}
    cumulative_r2_std = 0.0
    cumulative_r2_enc = 0.0

    # Order predictors by their individual R² contribution (descending) for standard
    individual_r2 = {}
    for pred in PREDICTORS:
        X = sm.add_constant(df_clean[pred].values.astype(np.float64))
        model = sm.OLS(y_std, X).fit()
        individual_r2[pred] = model.rsquared

    sorted_preds = sorted(PREDICTORS, key=lambda p: individual_r2[p], reverse=True)

    # Hierarchical: add predictors in order of decreasing individual R²
    added = []
    for pred in sorted_preds:
        added.append(pred)
        X = sm.add_constant(df_clean[added].values.astype(np.float64))

        model_std = sm.OLS(y_std, X).fit()
        model_enc = sm.OLS(y_enc, X).fit()

        delta_std = model_std.rsquared - cumulative_r2_std
        delta_enc = model_enc.rsquared - cumulative_r2_enc

        incremental_std[pred] = {
            "individual_r2": round(individual_r2[pred], 6),
            "cumulative_r2": round(float(model_std.rsquared), 6),
            "incremental_r2": round(delta_std, 6),
        }
        incremental_enc[pred] = {
            "cumulative_r2": round(float(model_enc.rsquared), 6),
            "incremental_r2": round(delta_enc, 6),
        }

        cumulative_r2_std = model_std.rsquared
        cumulative_r2_enc = model_enc.rsquared

    return {
        "predictor_order": sorted_preds,
        "incremental_r2_std": incremental_std,
        "incremental_r2_enc": incremental_enc,
        "final_r2_std": round(cumulative_r2_std, 6),
        "final_r2_enc": round(cumulative_r2_enc, 6),
    }


def compute_residual_contextualization(
    method_data: dict,
    incremental: dict,
) -> dict:
    """Contextualize the 5.25% residual variance."""
    logger.info("Computing residual variance contextualization...")

    meta = method_data["metadata"]
    pooled_std_r2 = meta["novelty_test_pooled"]["standard"]["r2"]
    pooled_enc_r2 = meta["novelty_test_pooled"]["encounter_only"]["r2"]

    residual_r2_std = round(1.0 - pooled_std_r2, 4)
    residual_r2_enc = round(1.0 - pooled_enc_r2, 4)

    # Per-treebank residual R² distribution
    examples = method_data["datasets"][0]["examples"]
    per_tb_residual = []
    for ex in examples:
        out = json.loads(ex["output"])
        r2 = out.get("r2_std")
        if r2 is not None and not np.isnan(r2):
            per_tb_residual.append(1.0 - r2)

    per_tb_arr = np.array(per_tb_residual)

    per_tb_stats = {
        "n_treebanks": len(per_tb_arr),
        "mean": round(float(np.mean(per_tb_arr)), 4),
        "median": round(float(np.median(per_tb_arr)), 4),
        "q25": round(float(np.percentile(per_tb_arr, 25)), 4),
        "q75": round(float(np.percentile(per_tb_arr, 75)), 4),
        "pct_above_005": round(float((per_tb_arr > 0.05).mean()), 4),
        "pct_above_010": round(float((per_tb_arr > 0.10).mean()), 4),
        "pct_above_015": round(float((per_tb_arr > 0.15).mean()), 4),
    }

    # Benchmark comparison table
    benchmark_comparison = {}
    for name, bm in BENCHMARK_EFFECTS.items():
        benchmark_comparison[name] = {
            "r2_mid": bm["r2_mid"],
            "r2_range": bm["r2_range"],
            "source": bm["source"],
            "ratio_to_residual_std": round(residual_r2_std / bm["r2_mid"], 2) if bm["r2_mid"] > 0 else None,
            "ratio_to_residual_enc": round(residual_r2_enc / bm["r2_mid"], 2) if bm["r2_mid"] > 0 else None,
        }

    return {
        "residual_r2_pct_std": residual_r2_std,
        "residual_r2_pct_enc": residual_r2_enc,
        "residual_r2_std_pct_display": f"{residual_r2_std * 100:.2f}%",
        "residual_r2_enc_pct_display": f"{residual_r2_enc * 100:.2f}%",
        "incremental_r2": incremental,
        "benchmark_effect_sizes": benchmark_comparison,
        "per_treebank_residual_r2": per_tb_stats,
    }


# ============================================================
# METRIC GROUP 4: Monotonicity Test
# ============================================================

def compute_monotonicity_test(
    r2_by_bin: dict[str, float],
    df: pd.DataFrame,
) -> dict:
    """Test whether R² decreases monotonically with sentence length."""
    logger.info("Computing monotonicity test...")

    # Spearman correlation between bin midpoint and within-bin R²
    midpoints = BIN_MIDPOINTS
    r2_values = [r2_by_bin.get(label, np.nan) for label in BIN_LABELS]

    valid = [(m, r) for m, r in zip(midpoints, r2_values) if not np.isnan(r)]
    if len(valid) < 3:
        logger.warning("Not enough valid bins for monotonicity test")
        return {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "linear_slope": np.nan,
            "bootstrap_ci_slope_lower": np.nan,
            "bootstrap_ci_slope_upper": np.nan,
        }

    mids = [v[0] for v in valid]
    r2s = [v[1] for v in valid]

    spearman_r, spearman_p = scipy_stats.spearmanr(mids, r2s)

    # Linear regression of R² on bin midpoint
    slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(mids, r2s)

    # Bootstrap CI for the slope
    logger.info("  Running bootstrap (1000 resamples per bin)...")
    df = df.copy()
    df["length_bin"] = assign_length_bin(df["sentence_length"])
    df_binned = df[df["length_bin"] != "excluded"]

    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    bootstrap_slopes = []

    for b in range(n_bootstrap):
        boot_r2s = []
        for label in BIN_LABELS:
            bin_df = df_binned[df_binned["length_bin"] == label]
            if len(bin_df) < 10:
                boot_r2s.append(np.nan)
                continue
            # Resample within bin
            boot_idx = rng.choice(len(bin_df), size=len(bin_df), replace=True)
            boot_df = bin_df.iloc[boot_idx]
            res = fit_ols_r2(boot_df, "max_imb")
            boot_r2s.append(res["r2"])

        valid_boot = [(m, r) for m, r in zip(BIN_MIDPOINTS, boot_r2s)
                       if r is not None and not np.isnan(r)]
        if len(valid_boot) >= 2:
            bm = [v[0] for v in valid_boot]
            br = [v[1] for v in valid_boot]
            bs, _, _, _, _ = scipy_stats.linregress(bm, br)
            bootstrap_slopes.append(bs)

    bootstrap_slopes = np.array(bootstrap_slopes)
    if len(bootstrap_slopes) > 0:
        ci_lower = float(np.percentile(bootstrap_slopes, 2.5))
        ci_upper = float(np.percentile(bootstrap_slopes, 97.5))
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        "spearman_r_length_r2": round(float(spearman_r), 4),
        "spearman_p_length_r2": round(float(spearman_p), 4),
        "linear_slope_length_r2": round(float(slope), 8),
        "linear_intercept": round(float(intercept), 6),
        "linear_r": round(float(r_val), 4),
        "linear_p": round(float(p_val), 4),
        "bootstrap_ci_slope_lower": round(ci_lower, 8),
        "bootstrap_ci_slope_upper": round(ci_upper, 8),
        "n_bootstrap_resamples": n_bootstrap,
        "n_valid_slopes": len(bootstrap_slopes),
        "r2_values_by_bin": dict(zip(BIN_LABELS, r2_values)),
        "midpoints_by_bin": dict(zip(BIN_LABELS, midpoints)),
    }


# ============================================================
# OUTPUT FORMATTING (exp_eval_sol_out.json schema)
# ============================================================

def build_output(
    *,
    length_strat: dict,
    head_type_recon: dict,
    residual_context: dict,
    monotonicity: dict,
    method_data: dict,
) -> dict:
    """Build output conforming to exp_eval_sol_out.json schema."""
    logger.info("Building output...")

    # ---- metrics_agg: flat numeric metrics ----
    metrics_agg = {}

    # Group 1: Length-stratified R²
    for i, label in enumerate(BIN_LABELS):
        safe_label = label.replace("-", "_").replace("+", "plus")
        r2_std = length_strat["r2_std_by_bin"].get(label)
        r2_enc = length_strat["r2_enc_by_bin"].get(label)
        n_sent = length_strat["n_sentences_per_bin"].get(label, 0)
        resid_var = length_strat["residual_variance_by_bin"].get(label)
        mean_abs = length_strat["mean_abs_residualized_max_imb_by_bin"].get(label)

        if r2_std is not None and not np.isnan(r2_std):
            metrics_agg[f"r2_std_bin_{safe_label}"] = round(r2_std, 6)
        if r2_enc is not None and not np.isnan(r2_enc):
            metrics_agg[f"r2_enc_bin_{safe_label}"] = round(r2_enc, 6)
        metrics_agg[f"n_sentences_bin_{safe_label}"] = n_sent
        if resid_var is not None and not np.isnan(resid_var):
            metrics_agg[f"residual_var_bin_{safe_label}"] = round(resid_var, 4)
        if mean_abs is not None and not np.isnan(mean_abs):
            metrics_agg[f"mean_abs_resid_bin_{safe_label}"] = round(mean_abs, 4)

    r2_delta_std = length_strat.get("r2_delta_std")
    r2_delta_enc = length_strat.get("r2_delta_enc")
    if r2_delta_std is not None and not np.isnan(r2_delta_std):
        metrics_agg["r2_delta_std_short_minus_long"] = round(r2_delta_std, 6)
    if r2_delta_enc is not None and not np.isnan(r2_delta_enc):
        metrics_agg["r2_delta_enc_short_minus_long"] = round(r2_delta_enc, 6)

    # Group 2: Head-type counts
    for ht, cnt in head_type_recon["head_type_counts_all"].items():
        metrics_agg[f"head_type_all_{ht}"] = cnt
    for ht, cnt in head_type_recon["head_type_counts_with_r2"].items():
        metrics_agg[f"head_type_n30_{ht}"] = cnt
    metrics_agg["n_treebanks_all"] = head_type_recon["n_treebanks_all"]
    metrics_agg["n_treebanks_with_r2"] = head_type_recon["n_treebanks_with_r2"]
    consistency = head_type_recon["classification_consistency_check"]
    metrics_agg["classification_agreement_rate"] = consistency["agreement_rate"]
    metrics_agg["classification_n_with_both"] = consistency["n_treebanks_with_both"]
    metrics_agg["classification_n_agree"] = consistency["agree"]
    metrics_agg["classification_n_disagree"] = consistency["disagree"]

    # Group 3: Residual variance
    metrics_agg["residual_r2_pct_std"] = residual_context["residual_r2_pct_std"]
    metrics_agg["residual_r2_pct_enc"] = residual_context["residual_r2_pct_enc"]
    per_tb = residual_context["per_treebank_residual_r2"]
    metrics_agg["per_tb_residual_mean"] = per_tb["mean"]
    metrics_agg["per_tb_residual_median"] = per_tb["median"]
    metrics_agg["per_tb_residual_q25"] = per_tb["q25"]
    metrics_agg["per_tb_residual_q75"] = per_tb["q75"]
    metrics_agg["per_tb_pct_above_5pct"] = per_tb["pct_above_005"]
    metrics_agg["per_tb_pct_above_10pct"] = per_tb["pct_above_010"]
    metrics_agg["per_tb_pct_above_15pct"] = per_tb["pct_above_015"]

    # Group 4: Monotonicity
    mono = monotonicity
    if not np.isnan(mono["spearman_r_length_r2"]):
        metrics_agg["spearman_r_length_r2"] = mono["spearman_r_length_r2"]
    if not np.isnan(mono["spearman_p_length_r2"]):
        metrics_agg["spearman_p_length_r2"] = mono["spearman_p_length_r2"]
    if not np.isnan(mono["linear_slope_length_r2"]):
        metrics_agg["linear_slope_length_r2"] = mono["linear_slope_length_r2"]
    if not np.isnan(mono["bootstrap_ci_slope_lower"]):
        metrics_agg["bootstrap_ci_slope_lower"] = mono["bootstrap_ci_slope_lower"]
    if not np.isnan(mono["bootstrap_ci_slope_upper"]):
        metrics_agg["bootstrap_ci_slope_upper"] = mono["bootstrap_ci_slope_upper"]

    # ---- Per-treebank examples ----
    examples = []
    method_examples = method_data["datasets"][0]["examples"]
    for ex in method_examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        tb_id = inp["treebank_id"]
        r2_std = out.get("r2_std")
        r2_enc = out.get("r2_enc")

        example_entry = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_treebank_id": tb_id,
            "metadata_head_type": inp.get("head_type", ""),
            "metadata_n_sentences": out.get("n_sentences", 0),
        }

        # Carry over predict_* fields from the original experiment output
        if "predict_imb_method" in ex:
            example_entry["predict_imb_method"] = ex["predict_imb_method"]
        if "predict_baseline" in ex:
            example_entry["predict_baseline"] = ex["predict_baseline"]

        # Always include eval fields (use 0.0 as fallback for treebanks without R²)
        if r2_std is not None and not np.isnan(r2_std):
            example_entry["eval_r2_std"] = round(r2_std, 6)
            example_entry["eval_residual_r2_std"] = round(1.0 - r2_std, 6)
        else:
            example_entry["eval_r2_std"] = 0.0
            example_entry["eval_residual_r2_std"] = 1.0
        if r2_enc is not None and not np.isnan(r2_enc):
            example_entry["eval_r2_enc"] = round(r2_enc, 6)
            example_entry["eval_residual_r2_enc"] = round(1.0 - r2_enc, 6)
        else:
            example_entry["eval_r2_enc"] = 0.0
            example_entry["eval_residual_r2_enc"] = 1.0

        examples.append(example_entry)

    # ---- Build final output ----
    output = {
        "metadata": {
            "evaluation": "sentence_length_stratified_r2_head_type_reconciliation_residual_contextualization",
            "description": (
                "Three complementary evaluations: (1) sentence-length-stratified R2 in 4 bins with head-type "
                "cross-stratification, (2) head-type count reconciliation between 335-treebank and 294-treebank "
                "groupings, (3) residual variance (5.25%) contextualization via incremental R2 and literature benchmarks."
            ),
            "length_bins": BIN_LABELS,
            "n_predictors": len(PREDICTORS),
            "predictors": PREDICTORS,
            "length_stratified_r2": length_strat,
            "head_type_reconciliation": head_type_recon,
            "residual_contextualization": residual_context,
            "monotonicity_test": monotonicity,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "commul/universal_dependencies",
                "examples": examples,
            }
        ],
    }

    return output


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main(max_rows: int | None = None):
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting evaluation: Length-Stratified R², Head-Type Reconciliation, Residual Contextualization")
    logger.info("=" * 60)

    # 1. Load data
    df_sentences = load_sentence_data(max_rows=max_rows)
    method_data = load_method_output()
    typology = load_typology()

    # 2. Metric Group 1: Sentence-Length-Stratified R²
    length_strat = compute_length_stratified_r2(df_sentences)

    # 3. Metric Group 2: Head-Type Count Reconciliation
    head_type_recon = compute_head_type_reconciliation(method_data, typology)

    # 4. Metric Group 3: Residual Variance Contextualization
    incremental = compute_incremental_r2(df_sentences)
    residual_context = compute_residual_contextualization(method_data, incremental)

    # 5. Metric Group 4: Monotonicity Test (with bootstrap)
    monotonicity = compute_monotonicity_test(length_strat["r2_std_by_bin"], df_sentences)

    # 6. Build output
    output = build_output(
        length_strat=length_strat,
        head_type_recon=head_type_recon,
        residual_context=residual_context,
        monotonicity=monotonicity,
        method_data=method_data,
    )

    # 7. Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f}KB)")

    # Free memory
    del df_sentences
    gc.collect()

    elapsed = time.time() - t_start
    logger.info(f"Total evaluation time: {elapsed:.1f}s")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows per CSV shard (for testing)")
    args = parser.parse_args()
    main(max_rows=args.max_rows)
