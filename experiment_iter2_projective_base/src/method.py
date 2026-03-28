#!/usr/bin/env python3
"""Phase 2: Projective Baseline LLF Comparison with Parametric Convexity Analysis.

Compares real sentence linearizations against random projective baselines to test
whether natural languages optimize for temporal load smoothing (measured by LLF,
max_imb, and parametric convexity costs at multiple alpha exponents).

Output: method_out.json conforming to exp_gen_sol_out schema.
"""

import json
import math
import os
import gc
import sys
import time
import random
import resource
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats as scipy_stats
from loguru import logger

# ============================================================
# CONFIGURATION
# ============================================================
WORKSPACE = Path(__file__).parent
DATA_ID2_DIR = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out"
)
DATA_ID3_PATH = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/full_data_out.json"
)

ALPHAS = [1.0, 1.2, 1.5, 2.0, 3.0]
N_BASELINES = 100
MAX_SENTENCES = 500
MIN_TREEBANK_SIZE = 200
PILOT_SIZE = 5
TARGET_TREEBANKS = 50
NUM_SHARDS = 23

# Logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# Reproducibility
random.seed(42)
np.random.seed(42)
sys.setrecursionlimit(5000)


# ============================================================
# HARDWARE DETECTION & RESOURCE LIMITS
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


def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 29.0


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")


# ============================================================
# CORE: IMB PROFILE COMPUTATION
# ============================================================
def compute_imb_profile(heads: list[int]) -> list[float]:
    """Compute IMB profile from heads array (1-indexed, 0=root).

    IMB(j) = sum of ages of all open dependencies at position j.
    An open dependency from dep_pos to head_pos has age = j - span_start at position j.
    """
    n = len(heads)
    imb = [0.0] * n
    for i in range(n):
        h = heads[i]
        if h == 0:
            continue
        dep_pos = i + 1
        head_pos = h
        s = min(dep_pos, head_pos)
        e = max(dep_pos, head_pos)
        for j in range(s, e + 1):
            imb[j - 1] += (j - s)
    return imb


def verify_identity(imb_profile: list[float], dd_list: list[int]) -> tuple[bool, float, float]:
    """Verify total_imb == sum(d*(d+1)/2 for d in dd_list)."""
    total_imb = sum(imb_profile)
    expected = sum(d * (d + 1) / 2.0 for d in dd_list)
    return abs(total_imb - expected) < 1e-6, total_imb, expected


# ============================================================
# CORE: PROJECTIVE BASELINE GENERATION
# ============================================================
def build_tree(heads: list[int]) -> tuple[int, dict]:
    """Build tree from heads (1-indexed, 0=root). Returns (root_0idx, children_dict)."""
    children = defaultdict(list)
    root = None
    for i, h in enumerate(heads):
        if h == 0:
            if root is None:
                root = i
        else:
            children[h - 1].append(i)
    return root, dict(children)


def random_projective_linearization(node: int, children_dict: dict) -> list[int]:
    """Recursively generate a random projective linearization of the subtree."""
    kids = children_dict.get(node, [])
    if not kids:
        return [node]

    left_kids, right_kids = [], []
    for kid in kids:
        if random.random() < 0.5:
            left_kids.append(kid)
        else:
            right_kids.append(kid)

    random.shuffle(left_kids)
    random.shuffle(right_kids)

    result = []
    for kid in left_kids:
        result.extend(random_projective_linearization(kid, children_dict))
    result.append(node)
    for kid in right_kids:
        result.extend(random_projective_linearization(kid, children_dict))
    return result


def generate_baseline_heads(heads: list[int]) -> list[int] | None:
    """Generate a random projective baseline linearization, returning new heads array."""
    n = len(heads)
    root, children = build_tree(heads)
    if root is None:
        return None

    lin_order = random_projective_linearization(root, children)
    if len(lin_order) != n:
        return None

    # Map old node index -> new 1-indexed position
    pos_map = [0] * n
    for new_pos, old_node in enumerate(lin_order):
        pos_map[old_node] = new_pos + 1

    new_heads = [0] * n
    for old_node in range(n):
        old_head = heads[old_node]
        new_pos = pos_map[old_node]
        if old_head == 0:
            new_heads[new_pos - 1] = 0
        else:
            new_heads[new_pos - 1] = pos_map[old_head - 1]
    return new_heads


# ============================================================
# PROCESS ONE SENTENCE
# ============================================================
def process_sentence(sent: dict, n_baselines: int, alphas: list[float]) -> dict | None:
    """Process a single sentence: compute real IMB metrics + projective baselines."""
    heads = sent["heads"]
    dd_list = sent["dd_list"]
    n = len(heads)
    if n < 3:
        return None

    # Real IMB
    real_imb = compute_imb_profile(heads)
    ok, total_imb, expected = verify_identity(real_imb, dd_list)

    real_arr = np.array(real_imb)
    real_max = float(np.max(real_arr))
    real_mean = float(np.mean(real_arr))
    real_llf = real_mean / real_max if real_max > 0 else 1.0

    real_alpha_costs = {}
    for a in alphas:
        real_alpha_costs[f"{a:.1f}"] = float(np.sum(np.power(real_arr, a)))

    # Generate baselines
    b_maxs, b_means, b_llfs = [], [], []
    b_alpha_costs = {f"{a:.1f}": [] for a in alphas}

    for _ in range(n_baselines):
        bh = generate_baseline_heads(heads)
        if bh is None:
            continue
        b_imb = compute_imb_profile(bh)
        b_arr = np.array(b_imb)
        bmax = float(np.max(b_arr))
        bmean = float(np.mean(b_arr))
        b_maxs.append(bmax)
        b_means.append(bmean)
        b_llfs.append(bmean / bmax if bmax > 0 else 1.0)
        for a in alphas:
            b_alpha_costs[f"{a:.1f}"].append(float(np.sum(np.power(b_arr, a))))

    if len(b_maxs) < 10:
        return None

    result = {
        "sent_id": sent["sent_id"],
        "sentence_length": sent["sentence_length"],
        "tree_depth": sent["tree_depth"],
        "mean_arity": sent["mean_arity"],
        "mean_dd": sent["mean_dd"],
        "dd_variance": sent["dd_variance"],
        "dd_skewness": sent["dd_skewness"],
        "projectivity_proportion": sent["projectivity_proportion"],
        "mean_ic": sent["mean_ic"],
        "ic_variance": sent["ic_variance"],
        "max_ic": sent["max_ic"],
        "identity_ok": ok,
        "identity_error": abs(total_imb - expected),
        "real_max_imb": real_max,
        "real_mean_imb": real_mean,
        "real_llf": real_llf,
        "baseline_mean_max_imb": float(np.mean(b_maxs)),
        "baseline_std_max_imb": float(np.std(b_maxs)),
        "baseline_mean_llf": float(np.mean(b_llfs)),
        "baseline_std_llf": float(np.std(b_llfs)),
        "baseline_mean_mean_imb": float(np.mean(b_means)),
        "n_valid_baselines": len(b_maxs),
    }
    for a in alphas:
        akey = f"{a:.1f}"
        result[f"real_cost_{akey}"] = real_alpha_costs[akey]
        result[f"baseline_mean_cost_{akey}"] = float(np.mean(b_alpha_costs[akey]))
        result[f"baseline_std_cost_{akey}"] = float(np.std(b_alpha_costs[akey]))

    return result


# ============================================================
# PROCESS ONE TREEBANK (for multiprocessing)
# ============================================================
def process_treebank(args: tuple) -> tuple[str, list, int]:
    """Process all sentences in a treebank. Designed for ProcessPoolExecutor."""
    tb_id, sentences, n_baselines, alphas = args

    # Deterministic seed per treebank
    rng_seed = sum(ord(c) * (i + 1) for i, c in enumerate(tb_id)) % (2**31)
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2**32))

    results = []
    n_violations = 0
    for sent in sentences:
        try:
            r = process_sentence(sent, n_baselines, alphas)
            if r is not None:
                results.append(r)
                if not r["identity_ok"]:
                    n_violations += 1
        except Exception:
            continue

    return tb_id, results, n_violations


# ============================================================
# DATA LOADING
# ============================================================
def load_metadata() -> dict:
    """Load typological metadata from data_id3."""
    logger.info(f"Loading metadata from {DATA_ID3_PATH.name}")
    data = json.loads(DATA_ID3_PATH.read_text())
    metadata = {}
    for ds in data["datasets"]:
        for ex in ds["examples"]:
            tb_id = ex["input"]
            meta = json.loads(ex["output"])
            metadata[tb_id] = meta
    logger.info(f"Loaded metadata for {len(metadata)} treebanks")
    return metadata


def load_treebank_stats() -> dict:
    """Load treebank_stats from the first shard's metadata."""
    shard1_path = DATA_ID2_DIR / "full_data_out_1.json"
    logger.info("Loading treebank stats from shard 1 metadata")
    with open(shard1_path) as f:
        shard1 = json.load(f)
    stats = shard1.get("metadata", {}).get("treebank_stats", {})
    del shard1
    gc.collect()
    logger.info(f"Loaded stats for {len(stats)} treebanks")
    return stats


def select_treebanks(metadata: dict, treebank_stats: dict) -> list[str]:
    """Select 50+ treebanks stratified by word order, family, and size."""
    # Key treebanks to force-include for pilot and coverage
    FORCE_INCLUDE = {"en_ewt", "de_gsd", "zh_gsd", "ja_gsd", "ar_padt",
                     "cs_pdt", "ru_syntagrus", "fi_tdt", "tr_imst", "hi_hdtb"}

    eligible = {}
    for tb_id, meta in metadata.items():
        filt = treebank_stats.get(tb_id, {}).get("filtered", 0)
        if filt >= MIN_TREEBANK_SIZE:
            eligible[tb_id] = {**meta, "_filt": filt}

    logger.info(f"Eligible treebanks (>={MIN_TREEBANK_SIZE} filtered): {len(eligible)}")

    def _wo(m):
        wo = m.get("wals_word_order")
        if wo and wo not in ("", "None", "No dominant order"):
            return wo
        return "other"

    by_wo = defaultdict(list)
    for tb_id, meta in eligible.items():
        by_wo[_wo(meta)].append((tb_id, meta))
    for wo, tbs in sorted(by_wo.items()):
        logger.info(f"  Word order {wo}: {len(tbs)} eligible")

    selected_set = set()
    selected = []

    # Force-include key treebanks first
    for tb_id in FORCE_INCLUDE:
        if tb_id in eligible and tb_id not in selected_set:
            selected.append(tb_id)
            selected_set.add(tb_id)
    logger.info(f"Force-included {len(selected)} key treebanks")

    # Stratified selection: up to 4 per family per word order (increased from 3)
    for wo, tbs in by_wo.items():
        tbs_sorted = sorted(tbs, key=lambda x: x[1]["_filt"], reverse=True)
        family_count = defaultdict(int)
        # Count force-included treebanks toward family limits
        for tb_id in selected:
            if tb_id in eligible:
                fam = eligible[tb_id].get("glottolog_family_name") or "Unknown"
                if _wo(eligible[tb_id]) == wo:
                    family_count[fam] += 1
        for tb_id, meta in tbs_sorted:
            if tb_id in selected_set:
                continue
            fam = meta.get("glottolog_family_name") or "Unknown"
            if family_count[fam] < 4:
                selected.append(tb_id)
                selected_set.add(tb_id)
                family_count[fam] += 1

    # Fill remaining if needed
    if len(selected) < TARGET_TREEBANKS:
        remaining = [t for t in eligible if t not in selected_set]
        remaining.sort(key=lambda x: eligible[x]["_filt"], reverse=True)
        for tb in remaining:
            if len(selected) >= TARGET_TREEBANKS:
                break
            selected.append(tb)
            selected_set.add(tb)

    logger.info(f"Selected {len(selected)} treebanks for analysis")
    return selected


def load_sentences(selected_treebanks: list[str], max_per_tb: int = MAX_SENTENCES) -> dict:
    """Load sentence data from data_id2 shards, one shard at a time."""
    selected_set = set(selected_treebanks)
    sentences_by_tb = defaultdict(list)

    for shard_idx in range(1, NUM_SHARDS + 1):
        shard_path = DATA_ID2_DIR / f"full_data_out_{shard_idx}.json"
        logger.info(f"Loading shard {shard_idx}/{NUM_SHARDS}")
        try:
            with open(shard_path) as f:
                shard = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed shard {shard_idx}: {e}")
            continue

        for ds_entry in shard.get("datasets", []):
            tb_id = ds_entry["dataset"]
            if tb_id not in selected_set:
                continue
            for ex in ds_entry["examples"]:
                try:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                    sentences_by_tb[tb_id].append({
                        "heads": inp["heads"],
                        "sent_id": ex.get("metadata_sent_id", ""),
                        "sentence_length": out["sentence_length"],
                        "tree_depth": out["tree_depth"],
                        "mean_arity": out["mean_arity"],
                        "mean_dd": out["mean_dd"],
                        "dd_variance": out["dd_variance"],
                        "dd_skewness": out["dd_skewness"],
                        "dd_list": out["dd_list"],
                        "projectivity_proportion": out["projectivity_proportion"],
                        "mean_ic": out["mean_ic"],
                        "ic_variance": out["ic_variance"],
                        "max_ic": out["max_ic"],
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skip malformed example in {tb_id}: {e}")
        del shard
        gc.collect()

    # Sample up to max_per_tb per treebank
    for tb_id in sentences_by_tb:
        if len(sentences_by_tb[tb_id]) > max_per_tb:
            sentences_by_tb[tb_id] = random.sample(sentences_by_tb[tb_id], max_per_tb)

    total = sum(len(v) for v in sentences_by_tb.values())
    logger.info(f"Loaded {total} sentences across {len(sentences_by_tb)} treebanks")
    return dict(sentences_by_tb)


# ============================================================
# REGRESSION & RESIDUALIZATION
# ============================================================
def ols_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS regression with intercept. Returns (beta, residuals, R^2)."""
    n = X.shape[0]
    X_int = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ beta
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, resid, r2


def residualize_all(all_results: list[dict], alphas: list[float]) -> dict:
    """Residualize real-baseline differences controlling for DD moments + tree properties.

    Approach: regress the DIFFERENCE (real - baseline) on features X.
    The residual is the portion of the difference NOT explained by X.
    """
    n = len(all_results)
    X = np.array([
        [r["mean_dd"], r["dd_variance"], r["dd_skewness"],
         r["sentence_length"], r["tree_depth"], r["mean_arity"],
         r["projectivity_proportion"]]
        for r in all_results
    ])

    out = {}

    # LLF difference
    diff_llf = np.array([r["real_llf"] - r["baseline_mean_llf"] for r in all_results])
    beta_l, resid_l, r2_l = ols_regression(X, diff_llf)
    out["llf"] = {"r2_diff": r2_l, "beta": beta_l.tolist(), "resid_diff": resid_l}

    # Also R^2 of LLF level from features (how well features predict LLF itself)
    y_real_llf = np.array([r["real_llf"] for r in all_results])
    _, _, r2_llf_level = ols_regression(X, y_real_llf)
    out["llf"]["r2_level"] = r2_llf_level

    # max_imb difference
    diff_max = np.array([r["real_max_imb"] - r["baseline_mean_max_imb"] for r in all_results])
    beta_m, resid_m, r2_m = ols_regression(X, diff_max)
    out["max_imb"] = {"r2_diff": r2_m, "beta": beta_m.tolist(), "resid_diff": resid_m}

    y_real_max = np.array([r["real_max_imb"] for r in all_results])
    _, _, r2_max_level = ols_regression(X, y_real_max)
    out["max_imb"]["r2_level"] = r2_max_level

    # Alpha cost differences
    for alpha in alphas:
        akey = f"{alpha:.1f}"
        diff_c = np.array([
            r[f"real_cost_{akey}"] - r[f"baseline_mean_cost_{akey}"]
            for r in all_results
        ])
        beta_c, resid_c, r2_c = ols_regression(X, diff_c)
        out[f"cost_{akey}"] = {"r2_diff": r2_c, "beta": beta_c.tolist(), "resid_diff": resid_c}

        # R^2 of cost level from DD features only (for negative control)
        y_real_cost = np.array([r[f"real_cost_{akey}"] for r in all_results])
        X_dd = X[:, :3]  # mean_dd, dd_variance, dd_skewness only
        _, _, r2_cost_dd = ols_regression(X_dd, y_real_cost)
        out[f"cost_{akey}"]["r2_cost_from_dd"] = r2_cost_dd

    return out


# ============================================================
# PER-TREEBANK STATISTICAL COMPARISON
# ============================================================
def _safe_cohen_d(diff: np.ndarray) -> float:
    """Compute Cohen's d = mean(diff)/std(diff) safely."""
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return float(np.mean(diff)) / sd if sd > 1e-12 else 0.0


def _safe_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test with NaN handling."""
    if len(a) < 3:
        return 0.0, 1.0
    try:
        t, p = scipy_stats.ttest_rel(a, b)
        return float(t) if not np.isnan(t) else 0.0, float(p) if not np.isnan(p) else 1.0
    except Exception:
        return 0.0, 1.0


def _safe_ttest_1samp(a: np.ndarray) -> tuple[float, float]:
    """One-sample t-test against 0."""
    if len(a) < 3:
        return 0.0, 1.0
    try:
        t, p = scipy_stats.ttest_1samp(a, 0)
        return float(t) if not np.isnan(t) else 0.0, float(p) if not np.isnan(p) else 1.0
    except Exception:
        return 0.0, 1.0


def compute_treebank_stats(
    tb_results: list[dict],
    resid_data: dict,
    idx_start: int,
    idx_end: int,
    alphas: list[float],
) -> dict:
    """Compute paired statistics for one treebank."""
    # --- RAW LLF ---
    real_llfs = np.array([r["real_llf"] for r in tb_results])
    base_llfs = np.array([r["baseline_mean_llf"] for r in tb_results])
    raw_diff = real_llfs - base_llfs
    _, raw_p = _safe_ttest(real_llfs, base_llfs)

    # --- RESIDUALIZED LLF ---
    resid_llf = resid_data["llf"]["resid_diff"][idx_start:idx_end]
    _, resid_llf_p = _safe_ttest_1samp(resid_llf)

    # --- RAW max_imb ---
    real_maxs = np.array([r["real_max_imb"] for r in tb_results])
    base_maxs = np.array([r["baseline_mean_max_imb"] for r in tb_results])
    _, raw_max_p = _safe_ttest(real_maxs, base_maxs)

    # --- RESIDUALIZED max_imb ---
    resid_max = resid_data["max_imb"]["resid_diff"][idx_start:idx_end]
    _, resid_max_p = _safe_ttest_1samp(resid_max)

    # --- CONVEXITY: raw + residualized per alpha ---
    convexity = {}
    for alpha in alphas:
        akey = f"{alpha:.1f}"
        rc = np.array([r[f"real_cost_{akey}"] for r in tb_results])
        bc = np.array([r[f"baseline_mean_cost_{akey}"] for r in tb_results])
        raw_c_diff = rc - bc
        _, raw_c_p = _safe_ttest(rc, bc)

        resid_c = resid_data[f"cost_{akey}"]["resid_diff"][idx_start:idx_end]
        _, resid_c_p = _safe_ttest_1samp(resid_c)

        convexity[f"alpha_{akey}"] = {
            "raw_cost_diff": float(np.mean(raw_c_diff)),
            "raw_cohen_d": _safe_cohen_d(raw_c_diff),
            "raw_p_value": raw_c_p,
            "resid_cohen_d": _safe_cohen_d(resid_c),
            "resid_p_value": resid_c_p,
        }

    return {
        "n_sentences": len(tb_results),
        "raw_llf": {
            "real_mean": float(np.mean(real_llfs)),
            "baseline_mean": float(np.mean(base_llfs)),
            "cohen_d": _safe_cohen_d(raw_diff),
            "p_value": raw_p,
        },
        "residualized_llf": {
            "mean_resid_diff": float(np.mean(resid_llf)),
            "cohen_d": _safe_cohen_d(resid_llf),
            "p_value": resid_llf_p,
        },
        "raw_max_imb": {
            "real_mean": float(np.mean(real_maxs)),
            "baseline_mean": float(np.mean(base_maxs)),
            "cohen_d": _safe_cohen_d(real_maxs - base_maxs),
            "p_value": raw_max_p,
        },
        "residualized_max_imb": {
            "mean_resid_diff": float(np.mean(resid_max)),
            "cohen_d": _safe_cohen_d(resid_max),
            "p_value": resid_max_p,
        },
        "convexity": convexity,
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================
def compute_aggregate(
    per_tb: list[dict], metadata: dict, alphas: list[float], resid_data: dict
) -> dict:
    """Compute aggregate statistics across all treebanks."""
    n_tbs = len(per_tb)
    if n_tbs == 0:
        return {}
    bonf = 0.01 / n_tbs

    def _wo(tb_id):
        wo = metadata.get(tb_id, {}).get("wals_word_order")
        if wo and wo not in ("", "None", "No dominant order"):
            return wo
        return "other"

    # --- Residualized LLF ---
    llf_ds = [r["stats"]["residualized_llf"]["cohen_d"] for r in per_tb]
    llf_ps = [r["stats"]["residualized_llf"]["p_value"] for r in per_tb]
    # For LLF: real > baseline (higher = more smoothed), so d > 0 is good
    prop_d_gt = sum(1 for d in llf_ds if d > 0.2) / n_tbs
    prop_sig = sum(1 for p in llf_ps if p < bonf) / n_tbs

    by_wo = defaultdict(list)
    for r in per_tb:
        by_wo[_wo(r["treebank_id"])].append(r["stats"]["residualized_llf"]["cohen_d"])
    wo_stats = {
        wo: {"n": len(ds), "median_d": float(np.median(ds)), "mean_d": float(np.mean(ds))}
        for wo, ds in by_wo.items()
    }

    # --- Residualized max_imb ---
    # For max_imb: real < baseline (lower peak = better), so d < 0 is good
    max_ds = [r["stats"]["residualized_max_imb"]["cohen_d"] for r in per_tb]
    max_ps = [r["stats"]["residualized_max_imb"]["p_value"] for r in per_tb]
    prop_max_d = sum(1 for d in max_ds if d < -0.2) / n_tbs
    prop_max_sig = sum(1 for p in max_ps if p < bonf) / n_tbs

    # --- Convexity: raw effects per alpha ---
    alpha_effects_raw = []
    for alpha in alphas:
        akey = f"alpha_{alpha:.1f}"
        ds = [r["stats"]["convexity"][akey]["raw_cohen_d"] for r in per_tb]
        ps = [r["stats"]["convexity"][akey]["raw_p_value"] for r in per_tb]
        alpha_effects_raw.append({
            "alpha": alpha,
            "median_d": float(np.median(ds)),
            "mean_d": float(np.mean(ds)),
            "prop_sig": sum(1 for p in ps if p < bonf) / n_tbs,
        })

    # --- Convexity: residualized effects per alpha ---
    alpha_effects_resid = []
    for alpha in alphas:
        akey = f"alpha_{alpha:.1f}"
        ds = [r["stats"]["convexity"][akey]["resid_cohen_d"] for r in per_tb]
        ps = [r["stats"]["convexity"][akey]["resid_p_value"] for r in per_tb]
        alpha_effects_resid.append({
            "alpha": alpha,
            "median_d": float(np.median(ds)),
            "mean_d": float(np.mean(ds)),
            "prop_sig": sum(1 for p in ps if p < bonf) / n_tbs,
        })

    # --- R^2 of cost from DD at each alpha (KEY evidence) ---
    # At alpha=1: R^2 should be high (DD explains cost level)
    # At alpha>1: R^2 drops (temporal overlap matters beyond DD)
    r2_by_alpha = {}
    for alpha in alphas:
        akey = f"{alpha:.1f}"
        r2_by_alpha[akey] = resid_data[f"cost_{akey}"]["r2_cost_from_dd"]

    r2_values = [r2_by_alpha[f"{a:.1f}"] for a in alphas]
    r2_drop = r2_values[0] - r2_values[-1] if r2_values else 0.0

    # --- Monotonicity: R^2 from DD should DECREASE with alpha ---
    # (DD explains less at higher alpha because temporal overlap matters more)
    a_arr = np.array(alphas)
    r2_arr = np.array(r2_values)
    if len(a_arr) > 2:
        sp_rho, sp_p = scipy_stats.spearmanr(a_arr, r2_arr)
    else:
        sp_rho, sp_p = 0.0, 1.0
    mono_r2_decrease = all(r2_arr[i + 1] <= r2_arr[i] for i in range(len(r2_arr) - 1))

    # Also check monotonicity of raw effect sizes (|d| should increase with alpha)
    raw_abs_ds = [abs(ae["median_d"]) for ae in alpha_effects_raw]

    # --- Negative control: R^2-based ---
    # R^2 at alpha=1 should be high (DD dominates cost at alpha=1)
    # R^2 drop should be substantial (temporal overlap matters at higher alpha)
    # R^2 should decrease monotonically with alpha
    neg_control_passes = (r2_values[0] > 0.7 and r2_drop > 0.3 and mono_r2_decrease)

    # --- Alpha* estimation ---
    # Smallest alpha where R^2 from DD drops below 0.5
    # (i.e., DD alone explains less than half the cost variance)
    alpha_star = None
    for i, alpha in enumerate(alphas):
        if r2_values[i] < 0.5:
            alpha_star = alpha
            break

    return {
        "residualized_llf": {
            "proportion_d_gt_0.2": prop_d_gt,
            "proportion_significant": prop_sig,
            "median_cohen_d": float(np.median(llf_ds)),
            "mean_cohen_d": float(np.mean(llf_ds)),
            "by_word_order": wo_stats,
        },
        "residualized_max_imb": {
            "proportion_abs_d_gt_0.2": prop_max_d,
            "proportion_significant": prop_max_sig,
            "median_cohen_d": float(np.median(max_ds)),
            "mean_cohen_d": float(np.mean(max_ds)),
        },
        "convexity_raw": alpha_effects_raw,
        "convexity_residualized": alpha_effects_resid,
        "convexity_monotonicity": {
            "test": "R^2 of cost from DD decreases with alpha",
            "r2_by_alpha": list(zip([float(a) for a in alphas], [float(r) for r in r2_values])),
            "spearman_rho": float(sp_rho) if not np.isnan(sp_rho) else 0.0,
            "spearman_p": float(sp_p) if not np.isnan(sp_p) else 1.0,
            "r2_monotonic_decrease": mono_r2_decrease,
            "r2_drop_alpha1_to_max": r2_drop,
            "raw_abs_d_by_alpha": list(zip([float(a) for a in alphas], raw_abs_ds)),
        },
        "negative_control_alpha1": {
            "r2_alpha1_from_dd": r2_values[0] if r2_values else 0.0,
            "r2_alpha_max_from_dd": r2_values[-1] if r2_values else 0.0,
            "r2_drop": r2_drop,
            "r2_monotonic_decrease": mono_r2_decrease,
            "passes": neg_control_passes,
        },
        "alpha_star": alpha_star,
        "r2_cost_from_dd_by_alpha": r2_by_alpha,
    }


# ============================================================
# UNIT TESTS
# ============================================================
def run_unit_tests() -> bool:
    """Run quick correctness checks before the main pipeline."""
    logger.info("=== UNIT TESTS ===")

    # Test 1: IMB on worked examples
    heads_a = [0, 1, 2, 2, 3]
    imb_a = compute_imb_profile(heads_a)
    assert imb_a == [0, 1, 2, 3, 2], f"Tree A IMB: {imb_a}"
    assert max(imb_a) == 3 and abs(sum(imb_a) / 5 / 3 - 8 / 15) < 0.01
    dd_a = [abs(i + 1 - heads_a[i]) for i in range(5) if heads_a[i] != 0]
    ok_a, _, _ = verify_identity(imb_a, dd_a)
    assert ok_a, "Tree A identity failed"

    heads_b = [0, 1, 1, 3, 3]
    imb_b = compute_imb_profile(heads_b)
    assert imb_b == [0, 2, 2, 2, 2], f"Tree B IMB: {imb_b}"
    assert max(imb_b) == 2 and abs(sum(imb_b) / 5 / 2 - 0.8) < 0.01
    dd_b = [abs(i + 1 - heads_b[i]) for i in range(5) if heads_b[i] != 0]
    ok_b, _, _ = verify_identity(imb_b, dd_b)
    assert ok_b, "Tree B identity failed"
    logger.info("  [PASS] IMB profiles correct for Trees A & B")

    # Test 2: Alpha costs
    arr_a, arr_b = np.array(imb_a, dtype=float), np.array(imb_b, dtype=float)
    assert abs(np.sum(arr_a) - np.sum(arr_b)) < 1e-6, "Alpha=1 costs differ"
    assert abs(np.sum(arr_a**2) - 18) < 1e-6, f"Tree A alpha=2: {np.sum(arr_a**2)}"
    assert abs(np.sum(arr_b**2) - 16) < 1e-6, f"Tree B alpha=2: {np.sum(arr_b**2)}"
    logger.info("  [PASS] Alpha costs correct (a=1 equal, a=2 Tree B < Tree A)")

    # Test 3: Projective baseline
    test_heads = [2, 0, 2, 5, 2]
    root, children = build_tree(test_heads)
    assert root == 1, f"Root wrong: {root}"
    n_valid = 0
    distinct = set()
    for _ in range(200):
        bh = generate_baseline_heads(test_heads)
        if bh is not None:
            n_valid += 1
            distinct.add(tuple(bh))
            assert bh.count(0) == 1, f"Multiple roots: {bh}"
    assert n_valid >= 190, f"Too many failures: {n_valid}/200"
    assert len(distinct) > 1, "All baselines identical"
    logger.info(f"  [PASS] Baselines: {n_valid}/200 valid, {len(distinct)} distinct")

    # Test 4: OLS regression
    rng = np.random.RandomState(42)
    X_t = rng.randn(100, 2)
    y_t = 2 * X_t[:, 0] + 3 * X_t[:, 1] + rng.randn(100) * 0.01
    beta_t, _, r2_t = ols_regression(X_t, y_t)
    assert r2_t > 0.99, f"R^2 too low: {r2_t}"
    assert abs(beta_t[1] - 2) < 0.1 and abs(beta_t[2] - 3) < 0.1
    logger.info(f"  [PASS] OLS regression R^2={r2_t:.4f}, betas=[{beta_t[1]:.2f}, {beta_t[2]:.2f}]")

    logger.info("=== ALL UNIT TESTS PASSED ===")
    return True


# ============================================================
# MAIN PIPELINE
# ============================================================
@logger.catch
def main():
    t0 = time.time()

    # --- Unit tests ---
    if not run_unit_tests():
        logger.error("Unit tests failed, aborting")
        return

    # --- Step 1: Load metadata ---
    metadata = load_metadata()
    treebank_stats = load_treebank_stats()

    # --- Step 2: Select treebanks ---
    selected = select_treebanks(metadata, treebank_stats)

    # --- Step 3: Load sentences ---
    sentences_by_tb = load_sentences(selected)
    # Remove treebanks with too few post-load sentences
    sentences_by_tb = {k: v for k, v in sentences_by_tb.items() if len(v) >= 30}
    logger.info(f"Treebanks after size filter: {len(sentences_by_tb)}")

    # --- Define pilot & full treebank lists ---
    pilot_candidates = ["en_ewt", "de_gsd", "zh_gsd", "ja_gsd", "ar_padt"]
    pilot_tbs = [t for t in pilot_candidates if t in sentences_by_tb]
    if len(pilot_tbs) < PILOT_SIZE:
        for t in sorted(sentences_by_tb, key=lambda x: len(sentences_by_tb[x]), reverse=True):
            if t not in pilot_tbs:
                pilot_tbs.append(t)
            if len(pilot_tbs) >= PILOT_SIZE:
                break
    logger.info(f"Pilot treebanks ({len(pilot_tbs)}): {pilot_tbs}")

    # --- Helper: run treebanks with multiprocessing ---
    def run_treebanks(tb_list: list[str], label: str) -> tuple[dict, int]:
        tasks = [(tb, sentences_by_tb[tb], N_BASELINES, ALPHAS) for tb in tb_list if tb in sentences_by_tb]
        logger.info(f"[{label}] Processing {len(tasks)} treebanks ({NUM_CPUS} workers)")
        results = {}
        violations = 0
        with ProcessPoolExecutor(max_workers=NUM_CPUS) as pool:
            futs = {pool.submit(process_treebank, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                tb_id = futs[fut]
                try:
                    ret_id, ret_results, ret_viol = fut.result()
                    results[ret_id] = ret_results
                    violations += ret_viol
                    done += 1
                    if done % 5 == 0 or done == len(tasks):
                        logger.info(f"  [{label}] {done}/{len(tasks)} done ({time.time()-t0:.0f}s)")
                except Exception as e:
                    logger.error(f"  [{label}] {tb_id} failed: {e}")
        return results, violations

    # ===== PILOT =====
    logger.info("=" * 60)
    logger.info("PILOT RUN")
    logger.info("=" * 60)
    t_pilot = time.time()
    pilot_results, pilot_violations = run_treebanks(pilot_tbs, "PILOT")
    pilot_elapsed = time.time() - t_pilot

    pilot_sents = sum(len(v) for v in pilot_results.values())
    logger.info(f"Pilot: {pilot_sents} sentences, {pilot_violations} identity violations, {pilot_elapsed:.1f}s")

    for tb_id, res in pilot_results.items():
        if res:
            rl = np.mean([r["real_llf"] for r in res])
            bl = np.mean([r["baseline_mean_llf"] for r in res])
            logger.info(f"  {tb_id}: n={len(res)}, real_LLF={rl:.3f}, base_LLF={bl:.3f}, diff={rl-bl:.4f}")

    # Extrapolate
    n_remaining = len([t for t in sentences_by_tb if t not in pilot_results])
    if pilot_tbs:
        est_per_tb = pilot_elapsed / len(pilot_tbs)
        est_total = est_per_tb * n_remaining
        logger.info(f"Estimated remaining: {n_remaining} treebanks, ~{est_total:.0f}s ({est_total/60:.1f}min)")

    # ===== FULL RUN =====
    remaining_tbs = [t for t in sentences_by_tb if t not in pilot_results]
    logger.info("=" * 60)
    logger.info(f"FULL RUN ({len(remaining_tbs)} treebanks)")
    logger.info("=" * 60)
    t_full = time.time()
    full_results, full_violations = run_treebanks(remaining_tbs, "FULL")
    full_elapsed = time.time() - t_full
    logger.info(f"Full run: {full_elapsed:.1f}s, {full_violations} violations")

    # Merge
    all_tb_results = {**pilot_results, **full_results}
    total_violations = pilot_violations + full_violations

    # --- Flatten all sentence results with treebank tracking ---
    all_sentence_results = []
    tb_indices = {}
    for tb_id in sorted(all_tb_results.keys()):
        res = all_tb_results[tb_id]
        if not res:
            continue
        start = len(all_sentence_results)
        all_sentence_results.extend(res)
        tb_indices[tb_id] = (start, len(all_sentence_results))

    total_sents = len(all_sentence_results)
    logger.info(f"Total sentences: {total_sents}, treebanks with data: {len(tb_indices)}")

    # --- Step 7: Residualization ---
    logger.info("=" * 60)
    logger.info("RESIDUALIZATION")
    logger.info("=" * 60)
    resid = residualize_all(all_sentence_results, ALPHAS)
    logger.info(f"R^2 LLF level from features: {resid['llf']['r2_level']:.4f}")
    logger.info(f"R^2 max_imb level from features: {resid['max_imb']['r2_level']:.4f}")
    logger.info(f"R^2 LLF diff from features: {resid['llf']['r2_diff']:.4f}")
    for alpha in ALPHAS:
        akey = f"{alpha:.1f}"
        logger.info(f"  alpha={akey}: R^2 cost from DD={resid[f'cost_{akey}']['r2_cost_from_dd']:.4f}, "
                     f"R^2 cost diff from features={resid[f'cost_{akey}']['r2_diff']:.4f}")

    # --- Step 8: Per-treebank stats ---
    logger.info("=" * 60)
    logger.info("PER-TREEBANK STATISTICS")
    logger.info("=" * 60)
    per_tb_final = []
    for tb_id, (s, e) in tb_indices.items():
        tb_res = all_sentence_results[s:e]
        if len(tb_res) < 10:
            continue
        stats = compute_treebank_stats(tb_res, resid, s, e, ALPHAS)
        meta = metadata.get(tb_id, {})
        per_tb_final.append({
            "treebank_id": tb_id,
            "language": meta.get("language_name", ""),
            "word_order": meta.get("wals_word_order") or "unknown",
            "family": meta.get("glottolog_family_name") or "unknown",
            "stats": stats,
        })

    logger.info(f"Per-treebank results: {len(per_tb_final)} treebanks")

    # --- Step 9: Aggregate ---
    logger.info("=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    aggregate = compute_aggregate(per_tb_final, metadata, ALPHAS, resid)

    logger.info(f"Residualized LLF: median d={aggregate['residualized_llf']['median_cohen_d']:.4f}, "
                f"prop d>0.2={aggregate['residualized_llf']['proportion_d_gt_0.2']:.3f}, "
                f"prop sig={aggregate['residualized_llf']['proportion_significant']:.3f}")
    logger.info(f"Residualized max_imb: median d={aggregate['residualized_max_imb']['median_cohen_d']:.4f}")
    logger.info(f"Convexity monotonicity (R2 from DD vs alpha): rho={aggregate['convexity_monotonicity']['spearman_rho']:.3f}, "
                f"R2 mono decrease={aggregate['convexity_monotonicity']['r2_monotonic_decrease']}")
    nc = aggregate['negative_control_alpha1']
    logger.info(f"Negative control: passes={nc['passes']} "
                f"(R2 a1={nc['r2_alpha1_from_dd']:.3f}, R2 amax={nc['r2_alpha_max_from_dd']:.3f}, "
                f"drop={nc['r2_drop']:.3f})")
    logger.info(f"Alpha*: {aggregate['alpha_star']}")
    logger.info(f"R^2 cost from DD: " + ", ".join(
        f"a={a:.1f}:{aggregate['r2_cost_from_dd_by_alpha'][f'{a:.1f}']:.3f}" for a in ALPHAS))

    # Pilot subset aggregate
    pilot_tb_final = [r for r in per_tb_final if r["treebank_id"] in set(pilot_tbs)]
    pilot_agg = compute_aggregate(pilot_tb_final, metadata, ALPHAS, resid) if pilot_tb_final else {}

    # --- Step 10: Build output (exp_gen_sol_out format) ---
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"FORMATTING OUTPUT (total runtime: {elapsed:.0f}s)")
    logger.info("=" * 60)

    examples = []
    for r in per_tb_final:
        tb_output = {
            "treebank_id": r["treebank_id"],
            "language": r["language"],
            "word_order": r["word_order"],
            "family": r["family"],
            "n_sentences": r["stats"]["n_sentences"],
            "raw_llf": r["stats"]["raw_llf"],
            "residualized_llf": r["stats"]["residualized_llf"],
            "raw_max_imb": r["stats"]["raw_max_imb"],
            "residualized_max_imb": r["stats"]["residualized_max_imb"],
            "convexity": r["stats"]["convexity"],
        }
        examples.append({
            "input": r["treebank_id"],
            "output": json.dumps(tb_output),
            "metadata_language": r["language"],
            "metadata_word_order": str(r["word_order"]),
            "metadata_family": r["family"],
            "metadata_n_sentences": r["stats"]["n_sentences"],
            "predict_residualized_llf_cohen_d": f"{r['stats']['residualized_llf']['cohen_d']:.4f}",
            "predict_raw_llf_cohen_d": f"{r['stats']['raw_llf']['cohen_d']:.4f}",
            "predict_residualized_max_imb_cohen_d": f"{r['stats']['residualized_max_imb']['cohen_d']:.4f}",
        })

    # Serialize residualization betas (not the full residual arrays)
    resid_summary = {
        "llf": {"r2_diff": resid["llf"]["r2_diff"], "r2_level": resid["llf"]["r2_level"],
                "beta": resid["llf"]["beta"]},
        "max_imb": {"r2_diff": resid["max_imb"]["r2_diff"], "r2_level": resid["max_imb"]["r2_level"],
                    "beta": resid["max_imb"]["beta"]},
    }
    for alpha in ALPHAS:
        akey = f"{alpha:.1f}"
        resid_summary[f"cost_{akey}"] = {
            "r2_diff": resid[f"cost_{akey}"]["r2_diff"],
            "r2_cost_from_dd": resid[f"cost_{akey}"]["r2_cost_from_dd"],
            "beta": resid[f"cost_{akey}"]["beta"],
        }

    output = {
        "metadata": {
            "experiment": "phase2_projective_baseline_llf",
            "n_treebanks": len(per_tb_final),
            "n_sentences_total": total_sents,
            "n_baselines_per_sentence": N_BASELINES,
            "alphas_tested": ALPHAS,
            "max_sentences_per_treebank": MAX_SENTENCES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": elapsed,
            "novelty_test": resid_summary,
            "identity_verification": {
                "n_sentences_checked": total_sents,
                "n_violations": total_violations,
                "max_relative_error": float(max(
                    (r["identity_error"] for r in all_sentence_results), default=0.0
                )),
            },
            "aggregate_results": aggregate,
            "pilot_results": pilot_agg,
        },
        "datasets": [
            {
                "dataset": "phase2_projective_baseline_llf",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path.name} ({size_mb:.1f} MB)")

    logger.info("=" * 60)
    logger.info(f"DONE - {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
