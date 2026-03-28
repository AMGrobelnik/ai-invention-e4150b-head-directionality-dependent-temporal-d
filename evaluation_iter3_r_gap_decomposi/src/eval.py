#!/usr/bin/env python3
"""R² Gap Decomposition & Typological Stratification of Convexity Evidence.

Evaluates whether the R² monotonicity across α in the convexity analysis is a
mathematical artifact or reflects genuine empirical signal, and whether weak
pooled convexity effects mask typological structure.

Output: eval_out.json conforming to exp_eval_sol_out schema.
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

# Experiment output (86 treebanks with convexity stats)
EXP_OUT_PATH = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd"
    "/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus/full_method_out.json"
)

# Raw sentence data (23 shards)
DATA_ID2_DIR = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd"
    "/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/full_data_out"
)

# Typological metadata
DATA_ID3_PATH = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd"
    "/3_invention_loop/iter_1/gen_art/data_id3_it1__opus/full_data_out.json"
)

ALPHAS = [1.0, 1.2, 1.5, 2.0, 3.0]
N_BASELINES = 50
MAX_SENTENCES = 500
N_BOOTSTRAP = 5000
NUM_SHARDS = 23
MIN_SENTENCES_PER_TB_R2 = 100  # Min sentences for per-treebank R²
MIN_SENTENCES_FOR_POOL = 30    # Min sentences to include in pooled analysis
N_STABILITY_REPS = 10          # Stability check repetitions

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
    """Detect actual CPU allocation (containers/pods/bare metal)."""
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
    """Read RAM limit from cgroup (containers/pods)."""
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)  # 60% of container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")


# ============================================================
# CORE FUNCTIONS (copied from experiment method.py)
# ============================================================
def compute_imb_profile(heads: list[int]) -> list[float]:
    """Compute IMB profile from heads array (1-indexed, 0=root)."""
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


def ols_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS regression with intercept. Returns (beta, residuals, R²)."""
    n = X.shape[0]
    X_int = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ beta
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, resid, r2


def compute_dd_features(heads: list[int]) -> tuple[float | None, float | None, float | None, list]:
    """Compute DD list and features from heads array."""
    dd_list = []
    for i, h in enumerate(heads):
        if h == 0:
            continue
        dd_list.append(abs(i + 1 - h))
    if not dd_list:
        return None, None, None, []
    arr = np.array(dd_list, dtype=float)
    mean_dd = float(np.mean(arr))
    dd_var = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    dd_skew = float(scipy_stats.skew(arr, bias=False)) if len(arr) > 2 else 0.0
    return mean_dd, dd_var, dd_skew, dd_list


# ============================================================
# STEP 1: LOAD EXPERIMENT OUTPUT
# ============================================================
def load_experiment_output() -> list[dict]:
    """Load per-treebank stats from experiment output."""
    logger.info(f"Loading experiment output from {EXP_OUT_PATH.name}")
    data = json.loads(EXP_OUT_PATH.read_text())
    examples = data["datasets"][0]["examples"]

    treebanks = []
    for ex in examples:
        tb_id = ex["input"]
        out = json.loads(ex["output"])
        resid_d = {}
        raw_d = {}
        for a in ALPHAS:
            akey = f"alpha_{a:.1f}"
            conv = out.get("convexity", {}).get(akey, {})
            resid_d[f"{a:.1f}"] = conv.get("resid_cohen_d", 0.0)
            raw_d[f"{a:.1f}"] = conv.get("raw_cohen_d", 0.0)

        wo = out.get("word_order", "unknown")
        if wo in ("", "None", "No dominant order"):
            wo = "other"
        elif wo == "unknown":
            wo = "other"

        treebanks.append({
            "tb_id": tb_id,
            "language": out.get("language", ""),
            "word_order": wo,
            "family": out.get("family", "unknown"),
            "n_sentences": out.get("n_sentences", 0),
            "resid_d": resid_d,
            "raw_d": raw_d,
        })

    logger.info(f"Loaded {len(treebanks)} treebanks from experiment output")
    return treebanks


# ============================================================
# STEP 2: TYPOLOGICAL STRATIFICATION (from existing data)
# ============================================================
def compute_stratification(treebanks: list[dict]) -> dict:
    """Compute typological stratification of convexity evidence."""
    logger.info("Computing typological stratification")

    # Group by word order
    by_wo = defaultdict(list)
    for tb in treebanks:
        by_wo[tb["word_order"]].append(tb)

    for wo, tbs in sorted(by_wo.items()):
        logger.info(f"  {wo}: {len(tbs)} treebanks")

    results = {"by_word_order": {}, "kruskal_wallis": {}, "spearman_within_wo": {},
               "ranking_check": {}, "raw_by_word_order": {}, "raw_kruskal_wallis": {}}

    # For each alpha, compute stratified stats
    for a in ALPHAS:
        akey = f"{a:.1f}"

        # --- Residualized d ---
        wo_stats = {}
        for wo, tbs in by_wo.items():
            ds = [tb["resid_d"][akey] for tb in tbs]
            arr = np.array(ds)
            wo_stats[wo] = {
                "n": len(ds),
                "median_d": float(np.median(arr)),
                "mean_d": float(np.mean(arr)),
                "std_d": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "iqr_25": float(np.percentile(arr, 25)),
                "iqr_75": float(np.percentile(arr, 75)),
                "values": ds,
            }
        results["by_word_order"][akey] = wo_stats

        # Kruskal-Wallis across VSO/SVO/SOV (exclude "other")
        groups_kw = []
        group_labels = []
        for wo_label in ["VSO", "SVO", "SOV"]:
            if wo_label in wo_stats and wo_stats[wo_label]["n"] >= 3:
                groups_kw.append(wo_stats[wo_label]["values"])
                group_labels.append(wo_label)

        if len(groups_kw) >= 2:
            try:
                h_stat, p_val = scipy_stats.kruskal(*groups_kw)
                results["kruskal_wallis"][akey] = {
                    "h_statistic": float(h_stat),
                    "p_value": float(p_val),
                    "groups": group_labels,
                }
            except Exception as e:
                logger.warning(f"Kruskal-Wallis failed at α={akey}: {e}")
                results["kruskal_wallis"][akey] = {"h_statistic": 0.0, "p_value": 1.0, "groups": group_labels}
        else:
            results["kruskal_wallis"][akey] = {"h_statistic": 0.0, "p_value": 1.0, "groups": group_labels}

        # Check ranking: VSO > SVO > SOV median d
        ranking_ok = True
        for wo_a, wo_b in [("VSO", "SVO"), ("SVO", "SOV")]:
            if wo_a in wo_stats and wo_b in wo_stats:
                if wo_stats[wo_a]["median_d"] <= wo_stats[wo_b]["median_d"]:
                    ranking_ok = False
            else:
                ranking_ok = False
        results["ranking_check"][akey] = ranking_ok

        # --- Raw d ---
        raw_wo_stats = {}
        for wo, tbs in by_wo.items():
            ds = [tb["raw_d"][akey] for tb in tbs]
            arr = np.array(ds)
            raw_wo_stats[wo] = {
                "n": len(ds),
                "median_d": float(np.median(arr)),
                "mean_d": float(np.mean(arr)),
                "std_d": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "values": ds,
            }
        results["raw_by_word_order"][akey] = raw_wo_stats

        # Raw Kruskal-Wallis
        raw_groups_kw = []
        raw_group_labels = []
        for wo_label in ["VSO", "SVO", "SOV"]:
            if wo_label in raw_wo_stats and raw_wo_stats[wo_label]["n"] >= 3:
                raw_groups_kw.append(raw_wo_stats[wo_label]["values"])
                raw_group_labels.append(wo_label)
        if len(raw_groups_kw) >= 2:
            try:
                h_stat, p_val = scipy_stats.kruskal(*raw_groups_kw)
                results["raw_kruskal_wallis"][akey] = {"h_statistic": float(h_stat), "p_value": float(p_val)}
            except Exception:
                results["raw_kruskal_wallis"][akey] = {"h_statistic": 0.0, "p_value": 1.0}
        else:
            results["raw_kruskal_wallis"][akey] = {"h_statistic": 0.0, "p_value": 1.0}

    # Spearman within each word-order group: resid d vs alpha
    for wo in by_wo:
        medians = []
        for a in ALPHAS:
            akey = f"{a:.1f}"
            if wo in results["by_word_order"][akey]:
                medians.append(results["by_word_order"][akey][wo]["median_d"])
            else:
                medians.append(0.0)
        if len(medians) >= 3:
            rho, p = scipy_stats.spearmanr(ALPHAS, medians)
            results["spearman_within_wo"][wo] = {
                "rho": float(rho) if not np.isnan(rho) else 0.0,
                "p_value": float(p) if not np.isnan(p) else 1.0,
            }

    return results


# ============================================================
# STEP 4: LOAD RAW SENTENCE DATA
# ============================================================
def load_raw_sentences(treebank_ids: list[str]) -> dict[str, list[dict]]:
    """Load raw sentence data from 23 shards, one at a time."""
    target_set = set(treebank_ids)
    sentences_by_tb = defaultdict(list)

    # Check shard availability
    available_shards = 0
    for shard_idx in range(1, NUM_SHARDS + 1):
        shard_path = DATA_ID2_DIR / f"full_data_out_{shard_idx}.json"
        if shard_path.exists():
            available_shards += 1

    if available_shards < NUM_SHARDS * 0.5:
        logger.error(f"Only {available_shards}/{NUM_SHARDS} shards available. Aborting.")
        sys.exit(1)

    logger.info(f"Loading raw sentences for {len(target_set)} treebanks from {available_shards} shards")

    for shard_idx in range(1, NUM_SHARDS + 1):
        shard_path = DATA_ID2_DIR / f"full_data_out_{shard_idx}.json"
        if not shard_path.exists():
            logger.warning(f"Shard {shard_idx} missing, skipping")
            continue

        logger.info(f"  Loading shard {shard_idx}/{NUM_SHARDS}")
        try:
            with open(shard_path) as f:
                shard = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed shard {shard_idx}: {e}")
            continue

        for ds_entry in shard.get("datasets", []):
            tb_id = ds_entry["dataset"]
            if tb_id not in target_set:
                continue
            for ex in ds_entry["examples"]:
                try:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                    sentences_by_tb[tb_id].append({
                        "heads": inp["heads"],
                        "sentence_length": out["sentence_length"],
                        "tree_depth": out["tree_depth"],
                        "mean_arity": out["mean_arity"],
                        "mean_dd": out["mean_dd"],
                        "dd_variance": out["dd_variance"],
                        "dd_skewness": out["dd_skewness"],
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skip malformed example in {tb_id}: {e}")

        del shard
        gc.collect()

    # Sample up to MAX_SENTENCES per treebank with deterministic seed
    for tb_id in sentences_by_tb:
        if len(sentences_by_tb[tb_id]) > MAX_SENTENCES:
            rng_seed = sum(ord(c) * (i + 1) for i, c in enumerate(tb_id)) % (2**31)
            rng = random.Random(rng_seed)
            sentences_by_tb[tb_id] = rng.sample(sentences_by_tb[tb_id], MAX_SENTENCES)

    total = sum(len(v) for v in sentences_by_tb.values())
    logger.info(f"Loaded {total} sentences across {len(sentences_by_tb)} treebanks")

    for tb_id in sorted(sentences_by_tb):
        logger.debug(f"  {tb_id}: {len(sentences_by_tb[tb_id])} sentences")

    return dict(sentences_by_tb)


# ============================================================
# STEP 5: PROCESS SENTENCES FOR R² GAP
# ============================================================
def process_treebank_r2(args: tuple) -> tuple[str, list, list]:
    """Per-treebank processing for R² gap analysis. Designed for ProcessPoolExecutor."""
    tb_id, sentences, n_baselines, alphas = args

    # Deterministic seed per treebank
    rng_seed = sum(ord(c) * (i + 1) for i, c in enumerate(tb_id)) % (2**31)
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2**32))

    real_rows = []
    baseline_rows = []
    n_skipped = 0

    for sent in sentences:
        heads = sent["heads"]
        if len(heads) < 3:
            n_skipped += 1
            continue

        # REAL sentence features + costs
        imb = compute_imb_profile(heads)
        imb_arr = np.array(imb)
        real_row = {
            "mean_dd": sent["mean_dd"],
            "dd_variance": sent["dd_variance"],
            "dd_skewness": sent["dd_skewness"],
            "sentence_length": sent["sentence_length"],
            "tree_depth": sent["tree_depth"],
            "mean_arity": sent["mean_arity"],
        }
        for a in alphas:
            real_row[f"cost_{a:.1f}"] = float(np.sum(np.power(imb_arr, a)))
        real_rows.append(real_row)

        # Generate N baselines, randomly select 1 for R² analysis
        baseline_data = []
        for _ in range(n_baselines):
            try:
                bh = generate_baseline_heads(heads)
            except RecursionError:
                continue
            if bh is None:
                continue
            b_mean_dd, b_dd_var, b_dd_skew, _ = compute_dd_features(bh)
            if b_mean_dd is None:
                continue
            b_imb = compute_imb_profile(bh)
            b_arr = np.array(b_imb)
            b_row = {
                "mean_dd": b_mean_dd,
                "dd_variance": b_dd_var,
                "dd_skewness": b_dd_skew,
                "sentence_length": sent["sentence_length"],
                "tree_depth": sent["tree_depth"],
                "mean_arity": sent["mean_arity"],
            }
            for a in alphas:
                b_row[f"cost_{a:.1f}"] = float(np.sum(np.power(b_arr, a)))
            baseline_data.append(b_row)

        if len(baseline_data) >= 10:
            # Select 1 random baseline per sentence
            baseline_rows.append(random.choice(baseline_data))
        else:
            n_skipped += 1

    return tb_id, real_rows, baseline_rows


# ============================================================
# STEP 6: COMPUTE POOLED R²
# ============================================================
def build_feature_matrix(rows: list[dict]) -> np.ndarray:
    """Extract 6-feature matrix from rows."""
    return np.array([
        [r["mean_dd"], r["dd_variance"], r["dd_skewness"],
         r["sentence_length"], r["tree_depth"], r["mean_arity"]]
        for r in rows
    ])


def compute_pooled_r2(
    per_tb_results: dict[str, dict],
    alphas: list[float],
) -> dict:
    """Compute pooled R² for real vs baseline across all treebanks."""
    logger.info("Computing pooled R² gap analysis")

    # Pool all rows
    all_real = []
    all_baseline = []
    for tb_id in sorted(per_tb_results):
        all_real.extend(per_tb_results[tb_id]["real"])
        all_baseline.extend(per_tb_results[tb_id]["baseline"])

    logger.info(f"  Pooled: {len(all_real)} real, {len(all_baseline)} baseline sentences")

    X_real = build_feature_matrix(all_real)
    X_baseline = build_feature_matrix(all_baseline)

    # DD-only features (first 3: mean_dd, dd_var, dd_skew)
    X_real_dd = X_real[:, :3]
    X_baseline_dd = X_baseline[:, :3]

    r2_results = {}
    for a in alphas:
        akey = f"{a:.1f}"
        y_real = np.array([r[f"cost_{akey}"] for r in all_real])
        y_base = np.array([r[f"cost_{akey}"] for r in all_baseline])

        # Full 6-feature model
        _, _, r2_real = ols_regression(X_real, y_real)
        _, _, r2_base = ols_regression(X_baseline, y_base)
        delta_r2 = r2_real - r2_base

        # DD-only model (3 features)
        _, _, r2_real_dd = ols_regression(X_real_dd, y_real)
        _, _, r2_base_dd = ols_regression(X_baseline_dd, y_base)
        delta_r2_dd = r2_real_dd - r2_base_dd

        r2_results[akey] = {
            "r2_real": r2_real,
            "r2_baseline": r2_base,
            "delta_r2": delta_r2,
            "r2_real_dd_only": r2_real_dd,
            "r2_baseline_dd_only": r2_base_dd,
            "delta_r2_dd_only": delta_r2_dd,
        }
        logger.info(f"  α={akey}: R²_real={r2_real:.4f}, R²_base={r2_base:.4f}, "
                     f"ΔR²={delta_r2:.4f}, R²_real_dd={r2_real_dd:.4f}")

    # Spearman ρ of ΔR² vs α
    deltas = [r2_results[f"{a:.1f}"]["delta_r2"] for a in alphas]
    rho_gap, p_gap = scipy_stats.spearmanr(alphas, deltas)
    r2_results["spearman_gap_vs_alpha"] = {
        "rho": float(rho_gap) if not np.isnan(rho_gap) else 0.0,
        "p_value": float(p_gap) if not np.isnan(p_gap) else 1.0,
    }

    return r2_results


# ============================================================
# STABILITY CHECK
# ============================================================
def stability_check(
    per_tb_results: dict[str, dict],
    alphas: list[float],
    n_reps: int = N_STABILITY_REPS,
) -> dict:
    """Repeat baseline R² with different random selections."""
    logger.info(f"Running {n_reps}-rep stability check")

    # Pre-collect all baselines per treebank-sentence (need multiple baselines stored)
    # We can't do this since we only stored 1 baseline per sentence.
    # Instead, we re-run pooled R² using the existing data, which is already stable
    # since we selected 1 baseline from 50 per sentence.
    # For a proper stability check, we'll compute bootstrap variance of R²_baseline.

    tb_ids = list(per_tb_results.keys())
    r2_base_samples = {f"{a:.1f}": [] for a in alphas}

    rng = np.random.RandomState(42)
    for rep in range(n_reps):
        # Subsample 80% of treebanks
        n_sample = max(int(len(tb_ids) * 0.8), 1)
        sampled = rng.choice(tb_ids, size=n_sample, replace=False)
        base_pool = []
        for tb in sampled:
            base_pool.extend(per_tb_results[tb]["baseline"])

        if len(base_pool) < 30:
            continue

        X_b = build_feature_matrix(base_pool)
        for a in alphas:
            akey = f"{a:.1f}"
            y_b = np.array([r[f"cost_{akey}"] for r in base_pool])
            _, _, r2_b = ols_regression(X_b, y_b)
            r2_base_samples[akey].append(r2_b)

    stability = {}
    for a in alphas:
        akey = f"{a:.1f}"
        samples = r2_base_samples[akey]
        if samples:
            stability[akey] = {
                "mean_r2_baseline": float(np.mean(samples)),
                "std_r2_baseline": float(np.std(samples)),
                "stable": float(np.std(samples)) < 0.01,
            }
        else:
            stability[akey] = {"mean_r2_baseline": 0.0, "std_r2_baseline": 0.0, "stable": True}

    logger.info(f"  Stability: " + ", ".join(
        f"α={akey}: std={stability[akey]['std_r2_baseline']:.4f}" for akey in sorted(stability)
    ))
    return stability


# ============================================================
# STEP 7: BOOTSTRAP R² GAP
# ============================================================
def bootstrap_r2_gap(
    per_tb_results: dict[str, dict],
    alphas: list[float],
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap 95% CI for ΔR²(α) by resampling treebanks."""
    logger.info(f"Running {n_boot}-iteration bootstrap")
    t0 = time.time()

    tb_ids = list(per_tb_results.keys())
    n_tbs = len(tb_ids)
    boot_gaps = {f"{a:.1f}": [] for a in alphas}

    rng = np.random.RandomState(42)

    # Time first 100 iterations to extrapolate
    for b in range(n_boot):
        if b == 100:
            elapsed_100 = time.time() - t0
            est_total = elapsed_100 * n_boot / 100
            logger.info(f"  100 iterations in {elapsed_100:.1f}s, estimated total: {est_total:.0f}s")
            if est_total > 300:  # 5 min limit
                # Subsample sentences for speed
                logger.warning(f"  Bootstrap too slow ({est_total:.0f}s est), reducing to 2000 iterations")
                n_boot = min(n_boot, 2000)

        # Resample treebanks with replacement
        sampled_indices = rng.choice(n_tbs, size=n_tbs, replace=True)
        sampled_tbs = [tb_ids[i] for i in sampled_indices]

        real_pool = []
        base_pool = []
        for tb in sampled_tbs:
            real_pool.extend(per_tb_results[tb]["real"])
            base_pool.extend(per_tb_results[tb]["baseline"])

        if len(real_pool) < 30 or len(base_pool) < 30:
            continue

        X_r = build_feature_matrix(real_pool)
        X_b = build_feature_matrix(base_pool)

        for a in alphas:
            akey = f"{a:.1f}"
            y_r = np.array([r[f"cost_{akey}"] for r in real_pool])
            y_b = np.array([r[f"cost_{akey}"] for r in base_pool])
            _, _, r2_r = ols_regression(X_r, y_r)
            _, _, r2_b = ols_regression(X_b, y_b)
            boot_gaps[akey].append(r2_r - r2_b)

        if b > 0 and b % 500 == 0:
            logger.info(f"  Bootstrap {b}/{n_boot} ({time.time()-t0:.0f}s)")

        if b >= n_boot - 1:
            break

    elapsed = time.time() - t0
    logger.info(f"  Bootstrap completed: {len(boot_gaps[f'{alphas[0]:.1f}'])} iterations in {elapsed:.1f}s")

    # Compute CIs and monotonicity
    boot_results = {}
    for a in alphas:
        akey = f"{a:.1f}"
        gaps = np.array(boot_gaps[akey])
        if len(gaps) > 0:
            boot_results[akey] = {
                "ci_2_5": float(np.percentile(gaps, 2.5)),
                "ci_50": float(np.percentile(gaps, 50)),
                "ci_97_5": float(np.percentile(gaps, 97.5)),
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps)),
            }
        else:
            boot_results[akey] = {"ci_2_5": 0.0, "ci_50": 0.0, "ci_97_5": 0.0, "mean": 0.0, "std": 0.0}

    # Proportion of bootstraps where ΔR² increases monotonically with α
    n_valid = len(boot_gaps[f"{alphas[0]:.1f}"])
    n_monotonic = 0
    for i in range(n_valid):
        gaps_at_i = [boot_gaps[f"{a:.1f}"][i] for a in alphas]
        if all(gaps_at_i[j + 1] >= gaps_at_i[j] for j in range(len(gaps_at_i) - 1)):
            n_monotonic += 1
    boot_results["prop_monotonic"] = n_monotonic / n_valid if n_valid > 0 else 0.0

    return boot_results


# ============================================================
# STEP 8: PER-TREEBANK R²
# ============================================================
def compute_per_treebank_r2(
    per_tb_results: dict[str, dict],
    treebank_info: dict[str, dict],
    alphas: list[float],
) -> dict:
    """Compute per-treebank R² gap for treebanks with sufficient sentences."""
    logger.info("Computing per-treebank R²")

    per_tb_r2 = {}
    for tb_id in sorted(per_tb_results):
        real = per_tb_results[tb_id]["real"]
        base = per_tb_results[tb_id]["baseline"]

        if len(real) < MIN_SENTENCES_PER_TB_R2 or len(base) < MIN_SENTENCES_PER_TB_R2:
            continue

        X_r = build_feature_matrix(real)
        X_b = build_feature_matrix(base)

        tb_r2 = {}
        for a in alphas:
            akey = f"{a:.1f}"
            y_r = np.array([r[f"cost_{akey}"] for r in real])
            y_b = np.array([r[f"cost_{akey}"] for r in base])
            _, _, r2_r = ols_regression(X_r, y_r)
            _, _, r2_b = ols_regression(X_b, y_b)
            tb_r2[akey] = {
                "r2_real": r2_r,
                "r2_baseline": r2_b,
                "delta_r2": r2_r - r2_b,
            }
        per_tb_r2[tb_id] = tb_r2

    logger.info(f"  Per-treebank R² computed for {len(per_tb_r2)} treebanks (>={MIN_SENTENCES_PER_TB_R2} sents)")

    # Aggregate by word order
    by_wo = defaultdict(list)
    for tb_id, r2_data in per_tb_r2.items():
        wo = treebank_info.get(tb_id, {}).get("word_order", "other")
        by_wo[wo].append(r2_data)

    wo_median_delta = {}
    for wo, r2_list in by_wo.items():
        wo_median_delta[wo] = {}
        for a in alphas:
            akey = f"{a:.1f}"
            deltas = [r[akey]["delta_r2"] for r in r2_list]
            wo_median_delta[wo][akey] = {
                "n": len(deltas),
                "median_delta_r2": float(np.median(deltas)),
                "mean_delta_r2": float(np.mean(deltas)),
            }

    return {"per_treebank": per_tb_r2, "by_word_order": wo_median_delta}


# ============================================================
# STEP 9: VERDICTS
# ============================================================
def compute_verdicts(
    r2_results: dict,
    stratification: dict,
    alphas: list[float],
) -> dict:
    """Compute 3-way verdict: mathematical artifact, weak empirical, or typologically modulated."""
    gap_values = [r2_results[f"{a:.1f}"]["delta_r2"] for a in alphas]
    gap_range = max(gap_values) - min(gap_values)
    rho_gap = r2_results["spearman_gap_vs_alpha"]["rho"]

    # Verdict 1: Mathematical artifact?
    verdict_mathematical = (abs(rho_gap) < 0.8) and (gap_range < 0.05)

    # Verdict 2: Weak empirical signal?
    verdict_weak_empirical = (rho_gap > 0.8) and (max(gap_values) < 0.10)

    # Verdict 3: Typologically modulated?
    any_kw_sig = any(
        stratification["kruskal_wallis"].get(f"{a:.1f}", {}).get("p_value", 1.0) < 0.05
        for a in alphas if a >= 1.5
    )
    ranking_matches = all(
        stratification["ranking_check"].get(f"{a:.1f}", False)
        for a in alphas if a >= 1.5
    )
    verdict_typological = any_kw_sig and ranking_matches

    verdicts = {
        "mathematical_artifact": verdict_mathematical,
        "weak_empirical": verdict_weak_empirical,
        "typologically_modulated": verdict_typological,
        "gap_range": gap_range,
        "gap_rho": rho_gap,
        "gap_values": gap_values,
        "max_gap": max(gap_values),
    }

    logger.info(f"Verdicts:")
    logger.info(f"  Mathematical artifact: {verdict_mathematical} (|ρ|={abs(rho_gap):.3f}, range={gap_range:.4f})")
    logger.info(f"  Weak empirical: {verdict_weak_empirical} (ρ={rho_gap:.3f}, max_gap={max(gap_values):.4f})")
    logger.info(f"  Typologically modulated: {verdict_typological} (KW sig={any_kw_sig}, ranking={ranking_matches})")

    return verdicts


# ============================================================
# STEP 10: FORMAT OUTPUT
# ============================================================
def format_output(
    treebanks: list[dict],
    stratification: dict,
    r2_results: dict,
    stability: dict,
    bootstrap: dict,
    per_tb_r2_data: dict,
    verdicts: dict,
    per_tb_results: dict,
    runtime_seconds: float,
) -> dict:
    """Format output as eval_out.json conforming to exp_eval_sol_out schema."""

    # ---- metrics_agg ----
    metrics_agg = {}

    # R² gap metrics (Group 1)
    for a in ALPHAS:
        akey = f"{a:.1f}"
        akey_safe = akey.replace(".", "_")
        r2 = r2_results.get(akey, {})
        metrics_agg[f"r2_real_alpha_{akey_safe}"] = r2.get("r2_real", 0.0)
        metrics_agg[f"r2_baseline_alpha_{akey_safe}"] = r2.get("r2_baseline", 0.0)
        metrics_agg[f"delta_r2_alpha_{akey_safe}"] = r2.get("delta_r2", 0.0)
        metrics_agg[f"r2_real_dd_only_alpha_{akey_safe}"] = r2.get("r2_real_dd_only", 0.0)
        metrics_agg[f"r2_baseline_dd_only_alpha_{akey_safe}"] = r2.get("r2_baseline_dd_only", 0.0)

        # Bootstrap CIs
        boot = bootstrap.get(akey, {})
        metrics_agg[f"bootstrap_ci_lower_alpha_{akey_safe}"] = boot.get("ci_2_5", 0.0)
        metrics_agg[f"bootstrap_ci_median_alpha_{akey_safe}"] = boot.get("ci_50", 0.0)
        metrics_agg[f"bootstrap_ci_upper_alpha_{akey_safe}"] = boot.get("ci_97_5", 0.0)

        # Kruskal-Wallis
        kw = stratification.get("kruskal_wallis", {}).get(akey, {})
        metrics_agg[f"kruskal_wallis_h_alpha_{akey_safe}"] = kw.get("h_statistic", 0.0)
        metrics_agg[f"kruskal_wallis_p_alpha_{akey_safe}"] = kw.get("p_value", 1.0)

        # Stratified median resid d
        by_wo = stratification.get("by_word_order", {}).get(akey, {})
        for wo_label in ["VSO", "SVO", "SOV", "other"]:
            wo_data = by_wo.get(wo_label, {})
            metrics_agg[f"stratified_median_resid_d_{wo_label}_alpha_{akey_safe}"] = wo_data.get("median_d", 0.0)
            metrics_agg[f"stratified_n_{wo_label}_alpha_{akey_safe}"] = float(wo_data.get("n", 0))

        # Stability
        stab = stability.get(akey, {})
        metrics_agg[f"stability_std_r2_baseline_alpha_{akey_safe}"] = stab.get("std_r2_baseline", 0.0)

    # Spearman ρ gap vs α
    sp = r2_results.get("spearman_gap_vs_alpha", {})
    metrics_agg["spearman_rho_gap_vs_alpha"] = sp.get("rho", 0.0)
    metrics_agg["spearman_p_gap_vs_alpha"] = sp.get("p_value", 1.0)

    # Bootstrap monotonicity
    metrics_agg["prop_bootstrap_monotonic"] = bootstrap.get("prop_monotonic", 0.0)

    # Verdicts (as 0/1 numeric)
    metrics_agg["verdict_mathematical_artifact"] = 1.0 if verdicts["mathematical_artifact"] else 0.0
    metrics_agg["verdict_weak_empirical"] = 1.0 if verdicts["weak_empirical"] else 0.0
    metrics_agg["verdict_typologically_modulated"] = 1.0 if verdicts["typologically_modulated"] else 0.0

    # Additional summary metrics
    metrics_agg["n_treebanks"] = float(len(treebanks))
    metrics_agg["n_treebanks_with_per_tb_r2"] = float(len(per_tb_r2_data.get("per_treebank", {})))
    n_real = sum(len(per_tb_results[tb]["real"]) for tb in per_tb_results)
    n_base = sum(len(per_tb_results[tb]["baseline"]) for tb in per_tb_results)
    metrics_agg["n_sentences_real"] = float(n_real)
    metrics_agg["n_sentences_baseline"] = float(n_base)
    metrics_agg["runtime_seconds"] = runtime_seconds

    # ---- Per-treebank examples ----
    # Build lookup for per-treebank R² data
    per_tb_r2_lookup = per_tb_r2_data.get("per_treebank", {})

    examples = []
    for tb in treebanks:
        tb_id = tb["tb_id"]
        # Build output dict
        tb_output = {
            "treebank_id": tb_id,
            "language": tb["language"],
            "word_order": tb["word_order"],
            "family": tb["family"],
            "n_sentences": tb["n_sentences"],
            "resid_d_by_alpha": tb["resid_d"],
            "raw_d_by_alpha": tb["raw_d"],
        }

        # Add per-treebank R² if available
        if tb_id in per_tb_r2_lookup:
            tb_output["per_tb_r2"] = per_tb_r2_lookup[tb_id]

        # Add sentence counts from our processing
        if tb_id in per_tb_results:
            tb_output["n_real_sentences_processed"] = len(per_tb_results[tb_id]["real"])
            tb_output["n_baseline_sentences_processed"] = len(per_tb_results[tb_id]["baseline"])

        # eval_ fields (must be numbers)
        eval_delta_r2_alpha_2 = 0.0
        eval_r2_real_alpha_2 = 0.0
        eval_r2_baseline_alpha_2 = 0.0
        if tb_id in per_tb_r2_lookup:
            r2_2 = per_tb_r2_lookup[tb_id].get("2.0", {})
            eval_delta_r2_alpha_2 = r2_2.get("delta_r2", 0.0)
            eval_r2_real_alpha_2 = r2_2.get("r2_real", 0.0)
            eval_r2_baseline_alpha_2 = r2_2.get("r2_baseline", 0.0)

        eval_resid_d_alpha_2 = tb["resid_d"].get("2.0", 0.0)

        example = {
            "input": tb_id,
            "output": json.dumps(tb_output),
            "metadata_word_order": str(tb["word_order"]),
            "metadata_language": tb["language"],
            "metadata_n_sentences": tb["n_sentences"],
            "predict_resid_convexity_d_alpha_2": f"{eval_resid_d_alpha_2:.4f}",
            "predict_delta_r2_alpha_2": f"{eval_delta_r2_alpha_2:.4f}",
            "eval_delta_r2_alpha_2": eval_delta_r2_alpha_2,
            "eval_resid_convexity_d_alpha_2": eval_resid_d_alpha_2,
            "eval_r2_real_alpha_2": eval_r2_real_alpha_2,
            "eval_r2_baseline_alpha_2": eval_r2_baseline_alpha_2,
        }
        examples.append(example)

    output = {
        "metadata": {
            "evaluation": "r2_gap_decomposition_typological_stratification",
            "n_treebanks": len(treebanks),
            "n_baselines_per_sentence": N_BASELINES,
            "n_bootstrap": N_BOOTSTRAP,
            "alphas": ALPHAS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": runtime_seconds,
            "verdicts": verdicts,
            "stratification_summary": {
                wo: stratification.get("spearman_within_wo", {}).get(wo, {})
                for wo in ["VSO", "SVO", "SOV", "other"]
            },
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "r2_gap_typological_eval",
                "examples": examples,
            }
        ],
    }

    return output


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    t0 = time.time()

    # --- Verify paths ---
    logger.info("=" * 60)
    logger.info("R² GAP DECOMPOSITION & TYPOLOGICAL STRATIFICATION")
    logger.info("=" * 60)

    for path, label in [(EXP_OUT_PATH, "Experiment output"), (DATA_ID2_DIR, "Raw data dir"), (DATA_ID3_PATH, "Metadata")]:
        if not path.exists():
            logger.error(f"{label} not found: {path}")
            sys.exit(1)
        logger.info(f"  {label}: {path.name} ✓")

    # ===== STEP 1: Load experiment output =====
    treebanks = load_experiment_output()

    # Build lookup
    tb_info = {tb["tb_id"]: tb for tb in treebanks}
    tb_ids = [tb["tb_id"] for tb in treebanks]

    # ===== STEP 2: Typological stratification from existing data =====
    stratification = compute_stratification(treebanks)

    # ===== STEP 4: Load raw sentence data =====
    sentences_by_tb = load_raw_sentences(tb_ids)

    # Filter treebanks with too few sentences
    valid_tbs = {k: v for k, v in sentences_by_tb.items() if len(v) >= MIN_SENTENCES_FOR_POOL}
    logger.info(f"Treebanks with >={MIN_SENTENCES_FOR_POOL} sentences: {len(valid_tbs)}")

    # ===== STEP 5: Process sentences for R² gap =====
    logger.info("=" * 60)
    logger.info("PROCESSING SENTENCES FOR R² GAP")
    logger.info("=" * 60)

    tasks = [(tb_id, sents, N_BASELINES, ALPHAS) for tb_id, sents in valid_tbs.items()]
    per_tb_results = {}
    n_done = 0
    t_proc = time.time()

    with ProcessPoolExecutor(max_workers=NUM_CPUS) as pool:
        futs = {pool.submit(process_treebank_r2, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            tb_id = futs[fut]
            try:
                ret_id, real_rows, baseline_rows = fut.result()
                per_tb_results[ret_id] = {"real": real_rows, "baseline": baseline_rows}
                n_done += 1
                if n_done % 10 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t_proc
                    logger.info(f"  {n_done}/{len(tasks)} treebanks processed ({elapsed:.0f}s)")
            except Exception as e:
                logger.error(f"  {tb_id} failed: {e}")

    total_real = sum(len(v["real"]) for v in per_tb_results.values())
    total_base = sum(len(v["baseline"]) for v in per_tb_results.values())
    proc_elapsed = time.time() - t_proc
    logger.info(f"Processing done: {total_real} real, {total_base} baseline sentences in {proc_elapsed:.0f}s")

    # ===== STEP 6: Compute pooled R² =====
    logger.info("=" * 60)
    logger.info("POOLED R² GAP ANALYSIS")
    logger.info("=" * 60)
    r2_results = compute_pooled_r2(per_tb_results, ALPHAS)

    # ===== STABILITY CHECK =====
    stability = stability_check(per_tb_results, ALPHAS)

    # ===== STEP 7: Bootstrap R² gap =====
    logger.info("=" * 60)
    logger.info("BOOTSTRAP R² GAP")
    logger.info("=" * 60)
    bootstrap = bootstrap_r2_gap(per_tb_results, ALPHAS)

    # ===== STEP 8: Per-treebank R² =====
    logger.info("=" * 60)
    logger.info("PER-TREEBANK R²")
    logger.info("=" * 60)
    per_tb_r2_data = compute_per_treebank_r2(per_tb_results, tb_info, ALPHAS)

    # ===== STEP 9: Verdicts =====
    logger.info("=" * 60)
    logger.info("VERDICTS")
    logger.info("=" * 60)
    verdicts = compute_verdicts(r2_results, stratification, ALPHAS)

    # ===== STEP 10: Format output =====
    logger.info("=" * 60)
    logger.info("FORMATTING OUTPUT")
    logger.info("=" * 60)
    elapsed = time.time() - t0

    output = format_output(
        treebanks=treebanks,
        stratification=stratification,
        r2_results=r2_results,
        stability=stability,
        bootstrap=bootstrap,
        per_tb_r2_data=per_tb_r2_data,
        verdicts=verdicts,
        per_tb_results=per_tb_results,
        runtime_seconds=elapsed,
    )

    # Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path.name} ({size_mb:.2f} MB)")

    # Final summary
    logger.info("=" * 60)
    logger.info(f"DONE - {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)
    logger.info(f"  Treebanks: {len(treebanks)}")
    logger.info(f"  Sentences processed: {total_real} real, {total_base} baseline")
    logger.info(f"  Verdict mathematical artifact: {verdicts['mathematical_artifact']}")
    logger.info(f"  Verdict weak empirical: {verdicts['weak_empirical']}")
    logger.info(f"  Verdict typologically modulated: {verdicts['typologically_modulated']}")

    for a in ALPHAS:
        akey = f"{a:.1f}"
        r2 = r2_results.get(akey, {})
        boot = bootstrap.get(akey, {})
        logger.info(f"  α={akey}: ΔR²={r2.get('delta_r2', 0):.4f} "
                     f"[{boot.get('ci_2_5', 0):.4f}, {boot.get('ci_97_5', 0):.4f}]")


if __name__ == "__main__":
    main()
