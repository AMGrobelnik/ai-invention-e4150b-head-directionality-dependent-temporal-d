#!/usr/bin/env python3
"""Phase 3: IC Independence Test — LLF/Residualized max(IMB) Beyond Intervener Complexity.

Tests whether LLF and residualized max(IMB) provide significant incremental
explanatory power beyond intervener complexity (IC) in nested logistic
regression models predicting real-vs-baseline sentence status.
Uses per-treebank logistic regressions with likelihood-ratio tests plus
random-effects meta-analysis across 30+ diverse treebanks.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Loguru setup
# ---------------------------------------------------------------------------
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).resolve().parent
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
# ---------------------------------------------------------------------------

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
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")

# RAM budget: ~20 GB (leave headroom)
RAM_BUDGET = int(min(20, TOTAL_RAM_GB * 0.70) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))  # 2 hours CPU time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA2_DIR = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/"
                 "3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
DATA3_DIR = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/"
                 "3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
DATA2_SHARDS = sorted((DATA2_DIR / "full_data_out").glob("full_data_out_*.json"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPHAS = [1.0, 1.2, 1.5, 2.0, 3.0]

# Scaling config (will be adjusted by gradual scaling logic)
PILOT_N_TREEBANKS = 5
PILOT_N_SENTENCES = 100
PILOT_N_BASELINES = 20
FULL_N_TREEBANKS = 35
FULL_N_SENTENCES = 400
FULL_N_BASELINES = 50
MIN_TREEBANK_SENTENCES = 400

# Priority treebanks by word order
PRIORITY_SVO = ["en_ewt", "cs_pdt", "fr_gsd", "de_gsd", "es_ancora", "it_isdt",
                "pt_bosque", "ru_syntagrus", "zh_gsd", "bg_btb", "ro_rrt", "nl_alpino"]
PRIORITY_SOV = ["hi_hdtb", "ja_gsd", "ko_gsd", "tr_imst", "fa_perdt", "ta_ttb",
                "eu_bdt", "la_proiel", "ur_udtb", "mr_ufal"]
PRIORITY_VSO = ["ar_padt", "he_htb", "ga_idt", "cy_ccg"]
PRIORITY_OTHER = ["id_gsd", "vi_vtb", "fi_tdt", "et_edt"]
ALL_PRIORITY = PRIORITY_SVO + PRIORITY_SOV + PRIORITY_VSO + PRIORITY_OTHER

# =========================================================================
# STEP 1: LOAD TREEBANK METADATA FROM data_id3
# =========================================================================

def load_metadata() -> dict[str, dict]:
    """Load typological metadata for all 336 treebanks."""
    logger.info("Loading treebank metadata from data_id3...")
    data = json.loads((DATA3_DIR / "full_data_out.json").read_text())
    meta: dict[str, dict] = {}
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            tb_id = ex["input"]
            try:
                rec = json.loads(ex["output"])
            except json.JSONDecodeError:
                continue
            # Also pull metadata_ fields
            rec["_word_order"] = ex.get("metadata_wals_word_order", "") or ""
            rec["_family"] = ex.get("metadata_glottolog_family_name", "") or ""
            rec["_macroarea"] = ex.get("metadata_macroarea", "") or ""
            meta[tb_id] = rec
    logger.info(f"Loaded metadata for {len(meta)} treebanks")
    return meta


def load_treebank_stats() -> dict[str, dict]:
    """Load treebank_stats from data_id2 metadata (first shard)."""
    logger.info("Loading treebank_stats from data_id2 shard 1...")
    shard = json.loads(DATA2_SHARDS[0].read_text())
    stats = shard.get("metadata", {}).get("treebank_stats", {})
    del shard
    gc.collect()
    logger.info(f"Loaded stats for {len(stats)} treebanks")
    return stats

# =========================================================================
# STEP 2: SELECT DIVERSE TREEBANKS
# =========================================================================

def select_treebanks(meta: dict, treebank_stats: dict,
                     target: int = FULL_N_TREEBANKS,
                     min_sents: int = MIN_TREEBANK_SENTENCES) -> list[str]:
    """Select diverse treebanks with sufficient data."""
    # Candidates with enough filtered sentences
    candidates = {}
    for tb_id, stats in treebank_stats.items():
        if stats.get("filtered", 0) >= min_sents and tb_id in meta:
            candidates[tb_id] = stats["filtered"]

    logger.info(f"Candidates with >= {min_sents} sentences: {len(candidates)}")

    # Start with priority treebanks that qualify
    selected = []
    for tb_id in ALL_PRIORITY:
        if tb_id in candidates and tb_id not in selected:
            selected.append(tb_id)

    # Fill remaining slots with diverse families
    if len(selected) < target:
        families_covered = {meta[t]["_family"] for t in selected if t in meta}
        remaining = [(tb_id, cnt) for tb_id, cnt in candidates.items()
                     if tb_id not in selected]
        remaining.sort(key=lambda x: -x[1])  # largest first

        for tb_id, cnt in remaining:
            if len(selected) >= target:
                break
            fam = meta.get(tb_id, {}).get("_family", "")
            if fam and fam not in families_covered:
                selected.append(tb_id)
                families_covered.add(fam)

        # If still need more, add by size
        for tb_id, cnt in remaining:
            if len(selected) >= target:
                break
            if tb_id not in selected:
                selected.append(tb_id)

    # If still not enough, lower threshold
    if len(selected) < 25 and min_sents > 200:
        logger.warning(f"Only {len(selected)} treebanks at threshold {min_sents}. "
                       f"Lowering to 200.")
        return select_treebanks(meta, treebank_stats, target=target, min_sents=200)

    logger.info(f"Selected {len(selected)} treebanks")
    # Log word order distribution
    wo_dist: dict[str, int] = defaultdict(int)
    for tb_id in selected:
        wo = meta.get(tb_id, {}).get("_word_order", "unknown") or "unknown"
        wo_dist[wo] += 1
    logger.info(f"Word order distribution: {dict(wo_dist)}")
    return selected

# =========================================================================
# STEP 3: CORE COMPUTATION FUNCTIONS
# =========================================================================

def compute_imb_profile(heads: list[int]) -> list[float]:
    """Compute IMB(j) at each position j. heads is 0-indexed list, values are 1-indexed head positions, 0=root."""
    n = len(heads)
    # Build dependency arcs as (open, close) pairs (1-indexed)
    deps = []
    for j in range(n):
        h = heads[j]
        if h != 0:
            dep_pos = j + 1
            deps.append((min(dep_pos, h), max(dep_pos, h)))

    imb = [0.0] * n
    for j_pos in range(1, n + 1):
        burden = 0.0
        for op, cl in deps:
            if op <= j_pos <= cl:
                burden += (j_pos - op)
        imb[j_pos - 1] = burden
    return imb


def compute_llf(imb: list[float]) -> float:
    """Compute Load Levelling Factor from IMB profile."""
    if not imb:
        return 0.0
    max_imb = max(imb)
    mean_imb = sum(imb) / len(imb)
    return mean_imb / max_imb if max_imb > 0 else 0.0


def compute_dd_features(heads: list[int]) -> tuple[float, float, float, list[int]]:
    """Compute dependency distance features from heads array."""
    dd_list = []
    for i, h in enumerate(heads):
        if h != 0:
            dd_list.append(abs((i + 1) - h))
    if not dd_list:
        return 0.0, 0.0, 0.0, dd_list
    arr = np.array(dd_list, dtype=np.float64)
    mean_dd = float(np.mean(arr))
    dd_var = float(np.var(arr))
    if len(dd_list) >= 3 and np.std(arr) > 0:
        dd_skew = float(np.mean(((arr - np.mean(arr)) / np.std(arr)) ** 3))
    else:
        dd_skew = 0.0
    return mean_dd, dd_var, dd_skew, dd_list


def compute_ic_features(heads: list[int]) -> tuple[float, float, int]:
    """Compute intervener complexity features from heads array."""
    n = len(heads)
    head_set = set()
    for h in heads:
        if h != 0:
            head_set.add(h)
    ic_list = []
    for i in range(n):
        h = heads[i]
        if h == 0:
            continue
        pos = i + 1
        a, b = min(pos, h), max(pos, h)
        ic_list.append(sum(1 for k in range(a + 1, b) if k in head_set))
    if not ic_list:
        return 0.0, 0.0, 0
    arr = np.array(ic_list, dtype=np.float64)
    return float(np.mean(arr)), float(np.var(arr)), int(np.max(arr))


def build_children_map(heads: list[int]) -> tuple[dict[int, list[int]], int | None]:
    """Build children map from heads. Returns (children_map, root_node)."""
    children: dict[int, list[int]] = defaultdict(list)
    root = None
    for i, h in enumerate(heads):
        node = i + 1
        if h == 0:
            root = node
        else:
            children[h].append(node)
    return dict(children), root


def sample_projective_linearization(children_map: dict, root: int, rng) -> list[int]:
    """Sample one random projective linearization of the dependency tree."""
    def _lin(node: int) -> list[int]:
        kids = children_map.get(node, [])
        if not kids:
            return [node]
        left, right = [], []
        for kid in kids:
            if rng.random() < 0.5:
                left.append(kid)
            else:
                right.append(kid)
        rng.shuffle(left)
        rng.shuffle(right)
        result = []
        for kid in left:
            result.extend(_lin(kid))
        result.append(node)
        for kid in right:
            result.extend(_lin(kid))
        return result
    return _lin(root)


def linearization_to_heads(lin: list[int], original_heads: list[int]) -> list[int]:
    """Convert a linearization to a new heads array."""
    new_pos_of = {orig: new_idx + 1 for new_idx, orig in enumerate(lin)}
    new_heads = []
    for orig_pos in lin:
        orig_head = original_heads[orig_pos - 1]
        if orig_head == 0:
            new_heads.append(0)
        else:
            new_heads.append(new_pos_of[orig_head])
    return new_heads


def compute_alpha_costs(imb_profile: list[float], alphas: list[float] = ALPHAS) -> dict[float, float]:
    """Compute superlinear costs for various alpha values."""
    return {a: sum(v ** a for v in imb_profile) for a in alphas}


def validate_tree(heads: list[int]) -> bool:
    """Validate that heads form a valid tree (single root, no cycles)."""
    n = len(heads)
    roots = [i for i, h in enumerate(heads) if h == 0]
    if len(roots) != 1:
        return False
    # Check for cycles via visited set
    for i in range(n):
        visited = set()
        cur = i
        while heads[cur] != 0:
            if cur in visited:
                return False
            visited.add(cur)
            cur = heads[cur] - 1
            if cur < 0 or cur >= n:
                return False
    return True

# =========================================================================
# STEP 4: LOAD SENTENCE DATA (memory-safe streaming)
# =========================================================================

def load_sentences_for_treebanks(target_tbs: set[str], max_per_tb: int) -> dict[str, list[dict]]:
    """Stream through shards and collect sentences for target treebanks."""
    logger.info(f"Loading sentences for {len(target_tbs)} treebanks (max {max_per_tb}/tb)...")
    result: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()

    for shard_idx, shard_path in enumerate(DATA2_SHARDS):
        # Early exit if all treebanks have enough
        if all(len(result.get(tb, [])) >= max_per_tb for tb in target_tbs):
            break

        logger.debug(f"Loading shard {shard_idx + 1}/{len(DATA2_SHARDS)}...")
        raw = json.loads(shard_path.read_text())
        for ds in raw.get("datasets", []):
            tb_id = ds.get("dataset", "")
            if tb_id not in target_tbs:
                continue
            if len(result[tb_id]) >= max_per_tb:
                continue
            for ex in ds.get("examples", []):
                if len(result[tb_id]) >= max_per_tb:
                    break
                try:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                except (json.JSONDecodeError, KeyError):
                    continue
                result[tb_id].append({
                    "heads": inp["heads"],
                    "sent_id": ex.get("metadata_sent_id", ""),
                    "sentence_length": out["sentence_length"],
                    "tree_depth": out["tree_depth"],
                    "mean_arity": out["mean_arity"],
                    "mean_dd": out["mean_dd"],
                    "dd_variance": out["dd_variance"],
                    "dd_skewness": out["dd_skewness"],
                    "mean_ic": out["mean_ic"],
                    "ic_variance": out["ic_variance"],
                    "max_ic": out["max_ic"],
                })
        del raw
        gc.collect()

    elapsed = time.time() - t0
    total_loaded = sum(len(v) for v in result.values())
    logger.info(f"Loaded {total_loaded} sentences from {len(result)} treebanks in {elapsed:.1f}s")
    return dict(result)

# =========================================================================
# STEP 5: PROCESS SENTENCES - GENERATE BASELINES & COMPUTE FEATURES
# =========================================================================

def process_one_sentence(sent_data: dict, n_baselines: int, rng_seed: int) -> list[dict]:
    """Process one real sentence: compute real features + N baselines."""
    heads = sent_data["heads"]
    children_map, root = build_children_map(heads)
    if root is None:
        return []

    n = len(heads)
    # Real sentence features
    imb_real = compute_imb_profile(heads)
    llf_real = compute_llf(imb_real)
    max_imb_real = max(imb_real) if imb_real else 0.0
    mean_imb_real = sum(imb_real) / len(imb_real) if imb_real else 0.0
    costs_real = compute_alpha_costs(imb_real)

    rows = []
    # Row for real sentence
    row_real = {
        "is_real": 1,
        "sent_id": sent_data["sent_id"],
        "sentence_length": sent_data["sentence_length"],
        "tree_depth": sent_data["tree_depth"],
        "mean_arity": sent_data["mean_arity"],
        "mean_dd": sent_data["mean_dd"],
        "dd_variance": sent_data["dd_variance"],
        "dd_skewness": sent_data["dd_skewness"],
        "mean_ic": sent_data["mean_ic"],
        "ic_variance": sent_data["ic_variance"],
        "max_ic": sent_data["max_ic"],
        "max_imb": max_imb_real,
        "mean_imb": mean_imb_real,
        "llf": llf_real,
    }
    for a, c in costs_real.items():
        row_real[f"cost_alpha_{str(a).replace('.', '_')}"] = c
    rows.append(row_real)

    # Generate baselines
    rng = np.random.RandomState(rng_seed)
    seen_lins: set[tuple[int, ...]] = set()
    original_order = tuple(range(1, n + 1))

    for _ in range(n_baselines * 3):
        if len(rows) - 1 >= n_baselines:
            break
        lin = sample_projective_linearization(children_map, root, rng)
        lin_key = tuple(lin)
        if lin_key in seen_lins or lin_key == original_order:
            continue
        seen_lins.add(lin_key)

        new_heads = linearization_to_heads(lin, heads)
        mean_dd_b, dd_var_b, dd_skew_b, _ = compute_dd_features(new_heads)
        mean_ic_b, ic_var_b, max_ic_b = compute_ic_features(new_heads)
        imb_b = compute_imb_profile(new_heads)
        llf_b = compute_llf(imb_b)
        max_imb_b = max(imb_b) if imb_b else 0.0
        mean_imb_b = sum(imb_b) / len(imb_b) if imb_b else 0.0
        costs_b = compute_alpha_costs(imb_b)

        row_b = {
            "is_real": 0,
            "sent_id": sent_data["sent_id"],
            "sentence_length": sent_data["sentence_length"],
            "tree_depth": sent_data["tree_depth"],
            "mean_arity": sent_data["mean_arity"],
            "mean_dd": mean_dd_b,
            "dd_variance": dd_var_b,
            "dd_skewness": dd_skew_b,
            "mean_ic": mean_ic_b,
            "ic_variance": ic_var_b,
            "max_ic": max_ic_b,
            "max_imb": max_imb_b,
            "mean_imb": mean_imb_b,
            "llf": llf_b,
        }
        for a, c in costs_b.items():
            row_b[f"cost_alpha_{str(a).replace('.', '_')}"] = c
        rows.append(row_b)

    return rows


def process_treebank_worker(args: tuple) -> tuple[str, list[dict]]:
    """Worker function for multiprocessing: process all sentences in one treebank."""
    tb_id, sentences, n_baselines = args
    all_rows = []
    for i, sent in enumerate(sentences):
        seed = hash((tb_id, i)) & 0xFFFFFFFF
        rows = process_one_sentence(sent, n_baselines, rng_seed=seed)
        for row in rows:
            row["treebank_id"] = tb_id
        all_rows.extend(rows)
    return tb_id, all_rows

# =========================================================================
# STEP 6: RESIDUALIZATION
# =========================================================================

def residualize_max_imb(df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """Residualize max_imb against DD features via OLS."""
    import statsmodels.api as sm

    predictors = ["mean_dd", "dd_variance", "dd_skewness",
                  "sentence_length", "tree_depth", "mean_arity"]
    X = sm.add_constant(df[predictors].values.astype(float))
    y = df["max_imb"].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    df = df.copy()
    df["resid_max_imb"] = model.resid
    r_squared = model.rsquared
    summary_str = f"R²={r_squared:.4f}, adj_R²={model.rsquared_adj:.4f}, F={model.fvalue:.1f}"
    logger.info(f"Residualization {summary_str} (R²<0.90 needed for novelty)")
    return df, r_squared, summary_str

# =========================================================================
# STEP 7: NESTED LOGISTIC REGRESSION MODELS (per-treebank)
# =========================================================================

def _safe_float(v: Any) -> float:
    """Convert to float, replacing inf/nan with 0."""
    f = float(v)
    if not math.isfinite(f):
        return 0.0
    return f


def fit_nested_models_for_treebank(tb_df: pd.DataFrame) -> tuple[dict, dict]:
    """Fit 7 nested logistic regression models and run LR tests."""
    import statsmodels.api as sm
    from scipy.stats import chi2

    y = tb_df["is_real"].values

    # Check for degenerate case
    if y.sum() == 0 or y.sum() == len(y):
        return {}, {}

    base_cols = ["mean_dd", "dd_variance", "dd_skewness",
                 "sentence_length", "tree_depth", "mean_arity"]
    ic_cols = ["mean_ic", "ic_variance", "max_ic"]

    models_spec = {
        "A": base_cols,
        "B": base_cols + ic_cols,
        "C": base_cols + ic_cols + ["llf"],
        "D": base_cols + ic_cols + ["resid_max_imb"],
        "E": base_cols + ic_cols + ["llf", "resid_max_imb"],
        "F": base_cols + ["llf"],
        "G": base_cols + ["llf"] + ic_cols,
    }

    results: dict[str, dict] = {}
    fitted: dict[str, Any] = {}

    for name, cols in models_spec.items():
        try:
            # Standardize features to help convergence
            X_raw = tb_df[cols].values.astype(float)
            means = X_raw.mean(axis=0)
            stds = X_raw.std(axis=0)
            stds[stds == 0] = 1.0
            X_std = (X_raw - means) / stds
            X = sm.add_constant(X_std)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try lbfgs first, fallback to newton then bfgs
                model = None
                for method in ["lbfgs", "newton", "bfgs"]:
                    try:
                        model = sm.Logit(y, X).fit(disp=0, maxiter=500, method=method)
                        if model.mle_retvals.get("converged", False):
                            break
                    except Exception:
                        continue
                if model is None:
                    # Last resort: regularized
                    model = sm.Logit(y, X).fit_regularized(alpha=0.01, disp=0, maxiter=500)

            converged = getattr(model, 'mle_retvals', {}).get("converged", True)
            fitted[name] = model
            res = {
                "converged": converged,
                "llf": _safe_float(model.llf),
                "aic": _safe_float(model.aic),
                "bic": _safe_float(model.bic),
                "n_params": len(cols) + 1,
                "pseudo_r2": _safe_float(model.prsquared),
            }
            # Extract key variable coefficients (on standardized scale)
            param_names = ["const"] + cols
            for pn, coef, se, pval in zip(param_names, model.params, model.bse, model.pvalues):
                if pn in ("llf", "resid_max_imb", "mean_ic"):
                    res[f"coeff_{pn}"] = _safe_float(coef)
                    res[f"se_{pn}"] = _safe_float(se)
                    res[f"pval_{pn}"] = _safe_float(pval)
                    res[f"ci_lower_{pn}"] = _safe_float(coef - 1.96 * se)
                    res[f"ci_upper_{pn}"] = _safe_float(coef + 1.96 * se)
            results[name] = res

        except Exception as e:
            logger.debug(f"Model {name} failed: {e}")
            results[name] = {"error": str(e)[:200], "converged": False}

    # Likelihood ratio tests
    comparisons = [("B", "C"), ("B", "D"), ("B", "E"), ("A", "B"), ("F", "G")]
    lr_tests: dict[str, dict] = {}
    for restricted, full in comparisons:
        if restricted in fitted and full in fitted:
            try:
                lr_stat = max(0.0, -2.0 * (fitted[restricted].llf - fitted[full].llf))
                df_diff = results[full]["n_params"] - results[restricted]["n_params"]
                if df_diff <= 0:
                    continue
                p_val = float(chi2.sf(lr_stat, df=df_diff))
                lr_tests[f"{restricted}_vs_{full}"] = {
                    "lr_statistic": _safe_float(lr_stat),
                    "df": df_diff,
                    "p_value": _safe_float(p_val),
                    "delta_aic": _safe_float(results[full]["aic"] - results[restricted]["aic"]),
                    "delta_bic": _safe_float(results[full]["bic"] - results[restricted]["bic"]),
                }
            except Exception as e:
                logger.debug(f"LR test {restricted} vs {full} failed: {e}")

    return results, lr_tests

# =========================================================================
# STEP 8: META-ANALYSIS (DerSimonian-Laird)
# =========================================================================

def meta_analysis_dersimonian_laird(effects: list[float], ses: list[float]) -> dict:
    """Random-effects meta-analysis using DerSimonian-Laird estimator."""
    from scipy.stats import chi2 as chi2_dist, norm

    eff = np.array(effects, dtype=np.float64)
    se_arr = np.array(ses, dtype=np.float64)

    # Filter out zero/tiny SEs
    valid = se_arr > 1e-10
    if valid.sum() < 2:
        return {"pooled_effect": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "z": 0.0, "p_value": 1.0, "tau2": 0.0, "I2": 0.0,
                "Q": 0.0, "Q_df": 0, "Q_pvalue": 1.0, "k": int(valid.sum())}
    eff = eff[valid]
    se_arr = se_arr[valid]

    weights_fe = 1.0 / (se_arr ** 2)
    theta_fe = np.sum(weights_fe * eff) / np.sum(weights_fe)

    # Q-statistic
    Q = float(np.sum(weights_fe * (eff - theta_fe) ** 2))
    k = len(eff)
    df = k - 1
    p_het = float(chi2_dist.sf(Q, df)) if df > 0 else 1.0

    # DerSimonian-Laird tau²
    C = float(np.sum(weights_fe) - np.sum(weights_fe ** 2) / np.sum(weights_fe))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects
    weights_re = 1.0 / (se_arr ** 2 + tau2)
    theta_re = float(np.sum(weights_re * eff) / np.sum(weights_re))
    se_re = float(1.0 / np.sqrt(np.sum(weights_re)))

    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    z = theta_re / se_re if se_re > 0 else 0.0
    p_re = float(2.0 * norm.sf(abs(z)))

    return {
        "pooled_effect": _safe_float(theta_re),
        "se": _safe_float(se_re),
        "ci_lower": _safe_float(theta_re - 1.96 * se_re),
        "ci_upper": _safe_float(theta_re + 1.96 * se_re),
        "z": _safe_float(z),
        "p_value": _safe_float(p_re),
        "tau2": _safe_float(tau2),
        "I2": _safe_float(I2),
        "Q": _safe_float(Q),
        "Q_df": df,
        "Q_pvalue": _safe_float(p_het),
        "k": k,
    }


def run_meta_analysis(per_tb_results: dict, variable: str, model_name: str) -> dict:
    """Extract per-treebank effects and SEs for a variable and run meta-analysis."""
    effects, ses = [], []
    for tb_id, res in per_tb_results.items():
        models = res.get("models", {})
        m = models.get(model_name, {})
        coeff_key = f"coeff_{variable}"
        se_key = f"se_{variable}"
        if coeff_key in m and se_key in m:
            eff = m[coeff_key]
            se = m[se_key]
            if math.isfinite(eff) and math.isfinite(se) and se > 1e-10:
                effects.append(eff)
                ses.append(se)
    if len(effects) < 2:
        return {"pooled_effect": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "z": 0.0, "p_value": 1.0, "tau2": 0.0, "I2": 0.0,
                "Q": 0.0, "Q_df": 0, "Q_pvalue": 1.0, "k": len(effects)}
    return meta_analysis_dersimonian_laird(effects, ses)

# =========================================================================
# STEP 9: DISCONFIRMATION ASSESSMENT
# =========================================================================

def assess_disconfirmation(per_tb_results: dict, meta_llf: dict,
                           meta_resid: dict) -> dict:
    """Evaluate preregistered confirmation/disconfirmation criteria."""
    n_tbs = len(per_tb_results)
    if n_tbs == 0:
        return {"status": "error", "reason": "no treebank results"}

    bonferroni_threshold = 0.01 / n_tbs

    n_sig_llf = 0
    n_sig_resid = 0
    n_sig_ic_bidir = 0

    for tb_id, res in per_tb_results.items():
        lr = res.get("lr_tests", {})
        bc = lr.get("B_vs_C", {})
        bd = lr.get("B_vs_D", {})
        fg = lr.get("F_vs_G", {})
        if bc.get("p_value", 1.0) < bonferroni_threshold:
            n_sig_llf += 1
        if bd.get("p_value", 1.0) < bonferroni_threshold:
            n_sig_resid += 1
        if fg.get("p_value", 1.0) < bonferroni_threshold:
            n_sig_ic_bidir += 1

    pct_sig_llf = n_sig_llf / n_tbs
    pct_sig_resid = n_sig_resid / n_tbs
    pct_sig_ic_bidir = n_sig_ic_bidir / n_tbs

    meta_llf_sig = meta_llf.get("p_value", 1.0) < 0.01
    meta_resid_sig = meta_resid.get("p_value", 1.0) < 0.01

    confirmed = (pct_sig_llf >= 0.70 or pct_sig_resid >= 0.70) and meta_llf_sig
    disconfirmed = pct_sig_llf < 0.30 and pct_sig_resid < 0.30

    if confirmed:
        status = "confirmed"
    elif disconfirmed:
        status = "disconfirmed"
    else:
        status = "inconclusive"

    return {
        "status": status,
        "n_treebanks": n_tbs,
        "bonferroni_threshold": bonferroni_threshold,
        "pct_significant_llf_B_vs_C": round(pct_sig_llf, 4),
        "n_significant_llf": n_sig_llf,
        "pct_significant_resid_max_imb_B_vs_D": round(pct_sig_resid, 4),
        "n_significant_resid": n_sig_resid,
        "pct_significant_ic_bidirectional_F_vs_G": round(pct_sig_ic_bidir, 4),
        "n_significant_ic_bidir": n_sig_ic_bidir,
        "meta_llf_significant": meta_llf_sig,
        "meta_llf_p_value": meta_llf.get("p_value", 1.0),
        "meta_resid_significant": meta_resid_sig,
        "meta_resid_p_value": meta_resid.get("p_value", 1.0),
    }

# =========================================================================
# STEP 10: OUTPUT ASSEMBLY
# =========================================================================

def build_output(per_tb_results: dict, meta_llf: dict, meta_resid: dict,
                 meta_ic_bidir: dict, assessment: dict, r2: float,
                 resid_summary: str, meta_data: dict,
                 n_sentences_per_tb: int, n_baselines: int,
                 selected_tbs: list[str]) -> dict:
    """Build output JSON in exp_gen_sol_out schema."""
    # Dataset 1: model_comparison_summary
    model_names = ["A", "B", "C", "D", "E", "F", "G"]
    model_descriptions = {
        "A": "DD only (mean_dd + dd_variance + dd_skewness + sentence_length + tree_depth + mean_arity)",
        "B": "DD + IC (Model A + mean_ic + ic_variance + max_ic)",
        "C": "DD + IC + LLF (Model B + llf)",
        "D": "DD + IC + resid_max_imb (Model B + resid_max_imb)",
        "E": "Full (Model B + llf + resid_max_imb)",
        "F": "DD + LLF (Model A + llf, no IC)",
        "G": "DD + LLF + IC (Model F + mean_ic + ic_variance + max_ic)",
    }

    model_summary_examples = []
    for mn in model_names:
        # Aggregate stats across treebanks
        aics, bics, pr2s, converged_count = [], [], [], 0
        for tb_id, res in per_tb_results.items():
            m = res.get("models", {}).get(mn, {})
            if m.get("converged", False):
                converged_count += 1
                if "aic" in m:
                    aics.append(m["aic"])
                if "bic" in m:
                    bics.append(m["bic"])
                if "pseudo_r2" in m:
                    pr2s.append(m["pseudo_r2"])

        summary = {
            "model": mn,
            "description": model_descriptions[mn],
            "n_converged": converged_count,
            "n_treebanks": len(per_tb_results),
            "mean_aic": round(float(np.mean(aics)), 2) if aics else None,
            "mean_bic": round(float(np.mean(bics)), 2) if bics else None,
            "mean_pseudo_r2": round(float(np.mean(pr2s)), 4) if pr2s else None,
            "median_pseudo_r2": round(float(np.median(pr2s)), 4) if pr2s else None,
        }
        model_summary_examples.append({
            "input": f"model_{mn}",
            "output": json.dumps(summary),
            "predict_mean_pseudo_r2": str(round(float(np.mean(pr2s)), 4)) if pr2s else "NA",
            "metadata_model_name": mn,
        })

    # Dataset 2: per_treebank_results
    per_tb_examples = []
    for tb_id in selected_tbs:
        if tb_id not in per_tb_results:
            continue
        res = per_tb_results[tb_id]
        m_data = meta_data.get(tb_id, {})
        wo = m_data.get("_word_order", "") or ""
        fam = m_data.get("_family", "") or ""

        tb_output = {
            "treebank_id": tb_id,
            "word_order": wo,
            "family": fam,
            "models": res.get("models", {}),
            "lr_tests": res.get("lr_tests", {}),
        }

        # Determine significance flags
        lr = res.get("lr_tests", {})
        bc_p = lr.get("B_vs_C", {}).get("p_value", 1.0)
        bd_p = lr.get("B_vs_D", {}).get("p_value", 1.0)
        bonf = 0.01 / len(per_tb_results)

        per_tb_examples.append({
            "input": tb_id,
            "output": json.dumps(tb_output, default=str),
            "predict_llf_significant": str(bc_p < bonf).lower(),
            "predict_resid_max_imb_significant": str(bd_p < bonf).lower(),
            "metadata_treebank_id": tb_id,
            "metadata_word_order": wo,
            "metadata_family": fam,
        })

    # Dataset 3: meta_analysis
    meta_examples = [
        {
            "input": "llf_meta",
            "output": json.dumps(meta_llf, default=str),
            "predict_pooled_effect": str(round(meta_llf.get("pooled_effect", 0.0), 6)),
            "predict_p_value": str(round(meta_llf.get("p_value", 1.0), 6)),
            "metadata_variable": "llf",
            "metadata_test": "B_vs_C",
        },
        {
            "input": "resid_max_imb_meta",
            "output": json.dumps(meta_resid, default=str),
            "predict_pooled_effect": str(round(meta_resid.get("pooled_effect", 0.0), 6)),
            "predict_p_value": str(round(meta_resid.get("p_value", 1.0), 6)),
            "metadata_variable": "resid_max_imb",
            "metadata_test": "B_vs_D",
        },
        {
            "input": "ic_bidirectional_meta",
            "output": json.dumps(meta_ic_bidir, default=str),
            "predict_pooled_effect": str(round(meta_ic_bidir.get("pooled_effect", 0.0), 6)),
            "predict_p_value": str(round(meta_ic_bidir.get("p_value", 1.0), 6)),
            "metadata_variable": "mean_ic",
            "metadata_test": "F_vs_G",
        },
    ]

    # Dataset 4: disconfirmation_assessment
    disconf_examples = [
        {
            "input": "phase3_assessment",
            "output": json.dumps(assessment, default=str),
            "predict_status": assessment.get("status", "unknown"),
            "metadata_status": assessment.get("status", "unknown"),
        }
    ]

    output = {
        "metadata": {
            "experiment": "phase3_ic_independence",
            "n_treebanks": len(per_tb_results),
            "n_sentences_per_tb": n_sentences_per_tb,
            "n_baselines_per_sentence": n_baselines,
            "residualization_r2": round(r2, 4),
            "residualization_summary": resid_summary,
            "alphas": ALPHAS,
            "selected_treebanks": selected_tbs[:len(per_tb_results)],
        },
        "datasets": [
            {"dataset": "model_comparison_summary", "examples": model_summary_examples},
            {"dataset": "per_treebank_results", "examples": per_tb_examples},
            {"dataset": "meta_analysis", "examples": meta_examples},
            {"dataset": "disconfirmation_assessment", "examples": disconf_examples},
        ],
    }
    return output

# =========================================================================
# GATE 1: SMOKE TEST
# =========================================================================

def run_smoke_test():
    """Validate IMB computation against worked examples and basic pipeline."""
    logger.info("=== GATE 1: SMOKE TEST ===")

    # Worked example A: [0, 1, 2, 2, 3]
    heads_a = [0, 1, 2, 2, 3]
    imb_a = compute_imb_profile(heads_a)
    # Expected: deps are (1,2), (2,3), (2,4), (3,5)
    # j=1: arcs covering 1: (1,2) -> 0, (1,3) not exists... Actually let me recalc
    # deps: h[0]=0 -> skip; h[1]=1 -> dep_pos=2, arc(1,2); h[2]=2 -> dep_pos=3, arc(2,3);
    #        h[3]=2 -> dep_pos=4, arc(2,4); h[4]=3 -> dep_pos=5, arc(3,5)
    # j=1: (1,2) covers 1: 1-1=0; (2,3) no; (2,4) no; (3,5) no -> 0
    # j=2: (1,2) covers 2: 2-1=1; (2,3) covers 2: 2-2=0; (2,4) covers 2: 2-2=0; (3,5) no -> 1
    # j=3: (1,2) no; (2,3) covers 3: 3-2=1; (2,4) covers 3: 3-2=1; (3,5) covers 3: 3-3=0 -> 2
    # j=4: (1,2) no; (2,3) no; (2,4) covers 4: 4-2=2; (3,5) covers 4: 4-3=1 -> 3
    # j=5: (3,5) covers 5: 5-3=2 -> 2
    expected_imb_a = [0, 1, 2, 3, 2]
    assert imb_a == expected_imb_a, f"IMB A mismatch: {imb_a} != {expected_imb_a}"
    assert max(imb_a) == 3, f"max_imb_a should be 3, got {max(imb_a)}"
    llf_a = compute_llf(imb_a)
    assert abs(llf_a - 8/15) < 0.01, f"LLF A mismatch: {llf_a} != {8/15:.4f}"

    # Worked example B: [0, 1, 1, 3, 3]
    heads_b = [0, 1, 1, 3, 3]
    imb_b = compute_imb_profile(heads_b)
    # deps: h[1]=1 -> (1,2); h[2]=1 -> (1,3); h[3]=3 -> (3,4); h[4]=3 -> (3,5)
    # j=1: (1,2)->0, (1,3)->0 -> 0
    # j=2: (1,2)->1, (1,3)->1 -> 2
    # j=3: (1,3)->2, (3,4)->0, (3,5)->0 -> 2
    # j=4: (3,4)->1, (3,5)->1 -> 2
    # j=5: (3,5)->2 -> 2
    expected_imb_b = [0, 2, 2, 2, 2]
    assert imb_b == expected_imb_b, f"IMB B mismatch: {imb_b} != {expected_imb_b}"
    assert max(imb_b) == 2
    llf_b = compute_llf(imb_b)
    assert abs(llf_b - 0.8) < 0.01, f"LLF B mismatch: {llf_b}"

    # Validate tree operations
    children_map, root = build_children_map(heads_a)
    assert root == 1
    rng = np.random.RandomState(42)
    lin = sample_projective_linearization(children_map, root, rng)
    assert sorted(lin) == [1, 2, 3, 4, 5]
    new_heads = linearization_to_heads(lin, heads_a)
    assert validate_tree(new_heads), f"Invalid tree from linearization: {new_heads}"
    assert len(new_heads) == len(heads_a)

    # Check DD/IC recomputation on baseline
    dd_orig = compute_dd_features(heads_a)
    dd_new = compute_dd_features(new_heads)
    # DD features should generally differ for different linearizations
    ic_orig = compute_ic_features(heads_a)
    ic_new = compute_ic_features(new_heads)

    logger.info(f"  IMB A: {imb_a}, max={max(imb_a)}, LLF={llf_a:.4f}")
    logger.info(f"  IMB B: {imb_b}, max={max(imb_b)}, LLF={llf_b:.4f}")
    logger.info(f"  Linearization test: orig_heads={heads_a}, lin={lin}, new_heads={new_heads}")
    logger.info("  GATE 1 PASSED")

# =========================================================================
# MAIN EXECUTION PIPELINE
# =========================================================================

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 3: IC Independence Test")
    logger.info("=" * 60)

    # Gate 1: Smoke test
    run_smoke_test()

    # Step 1: Load metadata
    meta = load_metadata()
    treebank_stats = load_treebank_stats()

    # Step 2: Select treebanks
    selected_tbs = select_treebanks(meta, treebank_stats)

    # =====================================================================
    # GATE 2: PILOT (5 treebanks, 100 sentences, 20 baselines)
    # =====================================================================
    logger.info("=== GATE 2: PILOT STATISTICAL TEST ===")
    pilot_tbs = selected_tbs[:PILOT_N_TREEBANKS]
    logger.info(f"Pilot treebanks: {pilot_tbs}")

    t_pilot_start = time.time()
    pilot_sents = load_sentences_for_treebanks(set(pilot_tbs), PILOT_N_SENTENCES)

    # Process pilot sequentially (small enough)
    pilot_rows: list[dict] = []
    for tb_id in pilot_tbs:
        if tb_id not in pilot_sents:
            logger.warning(f"  No sentences for pilot treebank {tb_id}")
            continue
        sents = pilot_sents[tb_id]
        logger.info(f"  Processing pilot {tb_id}: {len(sents)} sentences, {PILOT_N_BASELINES} baselines")
        _, rows = process_treebank_worker((tb_id, sents, PILOT_N_BASELINES))
        pilot_rows.extend(rows)
        logger.info(f"    -> {len(rows)} rows")

    pilot_df = pd.DataFrame(pilot_rows)
    logger.info(f"Pilot DataFrame: {len(pilot_df)} rows, {pilot_df['is_real'].sum()} real")

    # Residualize
    pilot_df, pilot_r2, pilot_resid_summary = residualize_max_imb(pilot_df)

    # Fit models for each pilot treebank
    pilot_all_converged = 0
    for tb_id in pilot_tbs:
        tb_df = pilot_df[pilot_df["treebank_id"] == tb_id]
        if len(tb_df) < 10:
            continue
        results, lr_tests = fit_nested_models_for_treebank(tb_df)
        n_conv = sum(1 for r in results.values() if r.get("converged", False))
        if n_conv == 7:
            pilot_all_converged += 1
        bc_p = lr_tests.get("B_vs_C", {}).get("p_value", "N/A")
        logger.info(f"  {tb_id}: {n_conv}/7 converged, B→C p={bc_p}")

    t_pilot = time.time() - t_pilot_start
    logger.info(f"Pilot completed in {t_pilot:.1f}s")
    logger.info(f"Pilot R²={pilot_r2:.4f}")
    logger.info(f"  {pilot_all_converged}/{len(pilot_tbs)} treebanks with all 7 models converged")

    # Extrapolate full run time
    n_pilot_rows = len(pilot_rows)
    pilot_per_tb = t_pilot / max(len(pilot_tbs), 1)
    estimated_full_time = pilot_per_tb * len(selected_tbs) * (FULL_N_SENTENCES / PILOT_N_SENTENCES) * (FULL_N_BASELINES / PILOT_N_BASELINES)
    # Account for parallelism
    estimated_full_time_parallel = estimated_full_time / max(NUM_CPUS - 1, 1)
    logger.info(f"Estimated full run: {estimated_full_time_parallel / 60:.1f} min "
                f"(serial: {estimated_full_time / 60:.1f} min)")

    # Free pilot data
    del pilot_rows, pilot_df, pilot_sents
    gc.collect()

    # =====================================================================
    # GATE 3+4: FULL RUN with adaptive scaling
    # =====================================================================
    # Determine budget-safe parameters
    remaining_minutes = (353.0 * 60 - (time.time() - t_start)) / 60
    logger.info(f"Remaining time budget: ~{remaining_minutes:.0f} min")

    # Adjust parameters based on time budget
    actual_n_sentences = FULL_N_SENTENCES
    actual_n_baselines = FULL_N_BASELINES
    actual_n_treebanks = min(len(selected_tbs), FULL_N_TREEBANKS)

    if estimated_full_time_parallel > remaining_minutes * 60 * 0.7:
        # Scale down
        scale_factor = (remaining_minutes * 60 * 0.5) / estimated_full_time_parallel
        actual_n_sentences = max(100, int(FULL_N_SENTENCES * min(scale_factor, 1.0)))
        actual_n_baselines = max(20, int(FULL_N_BASELINES * min(scale_factor, 1.0)))
        logger.warning(f"Scaling down: {actual_n_sentences} sentences, "
                       f"{actual_n_baselines} baselines")

    logger.info("=== GATE 3+4: FULL RUN ===")
    logger.info(f"Parameters: {actual_n_treebanks} treebanks, "
                f"{actual_n_sentences} sentences, {actual_n_baselines} baselines")

    final_tbs = selected_tbs[:actual_n_treebanks]

    # Load all sentences
    t_load = time.time()
    all_sents = load_sentences_for_treebanks(set(final_tbs), actual_n_sentences)
    logger.info(f"Data loading: {time.time() - t_load:.1f}s")

    # Filter treebanks with enough sentences
    valid_tbs = [tb for tb in final_tbs
                 if tb in all_sents and len(all_sents[tb]) >= 50]
    logger.info(f"Valid treebanks with >= 50 sentences: {len(valid_tbs)}")

    # Parallel processing
    t_proc = time.time()
    all_rows: list[dict] = []
    n_workers = max(1, NUM_CPUS - 1)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for tb_id in valid_tbs:
            fut = pool.submit(process_treebank_worker,
                              (tb_id, all_sents[tb_id], actual_n_baselines))
            futures[fut] = tb_id

        for fut in as_completed(futures):
            try:
                tb_id, rows = fut.result()
                all_rows.extend(rows)
                n_real = sum(1 for r in rows if r["is_real"] == 1)
                logger.info(f"  Completed {tb_id}: {len(rows)} rows "
                            f"({n_real} real, {len(rows) - n_real} baseline)")
            except Exception as e:
                logger.error(f"  Failed {futures[fut]}: {e}")

    # Free sentence data
    del all_sents
    gc.collect()

    t_proc_elapsed = time.time() - t_proc
    logger.info(f"Processing: {t_proc_elapsed:.1f}s, {len(all_rows)} total rows")

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    logger.info(f"DataFrame: {len(df)} rows, {df['is_real'].sum()} real, "
                f"{len(df) - df['is_real'].sum()} baseline")

    # Step 6: Residualize
    df, r2, resid_summary = residualize_max_imb(df)

    # Step 7: Fit nested models per treebank
    logger.info("Fitting nested logistic regression models per treebank...")
    per_tb_results: dict[str, dict] = {}
    for tb_id in valid_tbs:
        tb_df = df[df["treebank_id"] == tb_id]
        if len(tb_df) < 100:
            logger.warning(f"  Skipping {tb_id}: only {len(tb_df)} rows")
            continue
        try:
            results, lr_tests = fit_nested_models_for_treebank(tb_df)
            if results:
                per_tb_results[tb_id] = {"models": results, "lr_tests": lr_tests}
                n_conv = sum(1 for r in results.values() if r.get("converged", False))
                bc_p = lr_tests.get("B_vs_C", {}).get("p_value", "N/A")
                bd_p = lr_tests.get("B_vs_D", {}).get("p_value", "N/A")
                logger.info(f"  {tb_id}: {n_conv}/7 converged, "
                            f"B→C p={bc_p}, B→D p={bd_p}")
        except Exception as e:
            logger.error(f"  {tb_id} model fitting failed: {e}")

    logger.info(f"Successful treebanks: {len(per_tb_results)}")

    # Free DataFrame
    del df
    gc.collect()

    # Step 8: Meta-analysis
    logger.info("Running meta-analysis...")
    meta_llf = run_meta_analysis(per_tb_results, "llf", "C")
    meta_resid = run_meta_analysis(per_tb_results, "resid_max_imb", "D")
    meta_ic_bidir = run_meta_analysis(per_tb_results, "mean_ic", "G")

    logger.info(f"Meta LLF: effect={meta_llf['pooled_effect']:.4f}, "
                f"p={meta_llf['p_value']:.6f}, I²={meta_llf['I2']:.1f}%")
    logger.info(f"Meta resid_max_imb: effect={meta_resid['pooled_effect']:.4f}, "
                f"p={meta_resid['p_value']:.6f}, I²={meta_resid['I2']:.1f}%")
    logger.info(f"Meta IC bidir: effect={meta_ic_bidir['pooled_effect']:.4f}, "
                f"p={meta_ic_bidir['p_value']:.6f}, I²={meta_ic_bidir['I2']:.1f}%")

    # Step 9: Disconfirmation assessment
    assessment = assess_disconfirmation(per_tb_results, meta_llf, meta_resid)
    logger.info(f"Disconfirmation status: {assessment['status']}")
    logger.info(f"  LLF significant in {assessment['pct_significant_llf_B_vs_C']*100:.1f}% treebanks")
    logger.info(f"  resid_max_imb significant in {assessment['pct_significant_resid_max_imb_B_vs_D']*100:.1f}% treebanks")

    # Step 10: Build and save output
    output = build_output(
        per_tb_results=per_tb_results,
        meta_llf=meta_llf,
        meta_resid=meta_resid,
        meta_ic_bidir=meta_ic_bidir,
        assessment=assessment,
        r2=r2,
        resid_summary=resid_summary,
        meta_data=meta,
        n_sentences_per_tb=actual_n_sentences,
        n_baselines=actual_n_baselines,
        selected_tbs=valid_tbs,
    )

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path}")

    total_examples = sum(len(ds["examples"]) for ds in output["datasets"])
    logger.info(f"Total examples in output: {total_examples}")

    t_total = time.time() - t_start
    logger.info(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info("=" * 60)
    logger.info("Phase 3 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
