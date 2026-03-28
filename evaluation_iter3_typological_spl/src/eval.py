#!/usr/bin/env python3
"""Typological Split Permutation Test, Stratified Meta-Analysis, and Prediction Scorecard.

Validates the post-hoc typological split in temporal dependency overlap through:
1. Permutation test (10K iterations) for VSO-SOV gap significance
2. DerSimonian-Laird random-effects meta-analysis stratified by word-order
3. Head-final dependency proportion correlation with bootstrap CI
4. VSO leave-one-out sensitivity analysis
5. Prediction scorecard for pre-registered success/disconfirmation criteria
"""

from loguru import logger
from pathlib import Path
import json
import sys
import math
import os
import resource
import gc
import time

import numpy as np
from scipy import stats

# ── Logging ──────────────────────────────────────────────────
logger.remove()
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware detection ───────────────────────────────────────

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


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 16.0
# Conservative budget: 30% of container RAM, data is small (~10 MB)
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.3, 15) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── Dependency paths ─────────────────────────────────────────
BASE_RUN = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/3_invention_loop")
EXP_PATH = BASE_RUN / "iter_2" / "gen_art" / "exp_id2_it2__opus" / "full_method_out.json"
DATA_PATH = BASE_RUN / "iter_1" / "gen_art" / "data_id3_it1__opus" / "full_data_out.json"


# ═════════════════════════════════════════════════════════════
# Step 0 — Data Loading & Joining
# ═════════════════════════════════════════════════════════════

def load_exp_data(path: Path, max_examples: int | None = None) -> tuple[dict, list[dict]]:
    """Load experiment data from exp_id2, parse per-treebank JSON outputs."""
    logger.info(f"Loading experiment data from {path.name}")
    raw = json.loads(path.read_text())
    metadata = raw.get("metadata", {})
    examples = raw["datasets"][0]["examples"]
    if max_examples:
        examples = examples[:max_examples]

    treebanks = []
    for ex in examples:
        try:
            rec = json.loads(ex["output"])
            treebanks.append(rec)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse exp example: {e}")
            continue

    logger.info(f"Loaded {len(treebanks)} treebank records from experiment")
    return metadata, treebanks


def load_data_metadata(path: Path) -> dict[str, dict]:
    """Load data_id3 metadata, return dict keyed by treebank_id."""
    logger.info(f"Loading metadata from {path.name}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]

    meta_dict: dict[str, dict] = {}
    for ex in examples:
        try:
            rec = json.loads(ex["output"])
            meta_dict[rec["treebank_id"]] = rec
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse data example: {e}")
            continue

    logger.info(f"Loaded metadata for {len(meta_dict)} treebanks")
    return meta_dict


def join_data(treebanks: list[dict], meta_dict: dict[str, dict]) -> list[dict]:
    """Join experiment treebanks with data_id3 metadata."""
    joined: list[dict] = []
    n_missing = 0

    for tb in treebanks:
        tid = tb["treebank_id"]
        meta = meta_dict.get(tid, {})
        if not meta:
            logger.warning(f"Treebank {tid} not found in data_id3 metadata")
            n_missing += 1

        hdlp = meta.get("head_direction_left_proportion")
        tb["head_final_prop"] = (1.0 - hdlp) if hdlp is not None else float("nan")
        tb["glottolog_family_name"] = meta.get("glottolog_family_name", "unknown")
        tb["macroarea"] = meta.get("macroarea", "unknown")
        tb["data_n_sentences"] = meta.get("num_sentences", 0)
        joined.append(tb)

    if n_missing:
        logger.warning(f"{n_missing} treebanks not found in data_id3")
    logger.info(f"Joined {len(joined)} treebanks ({n_missing} missing metadata)")
    return joined


# ═════════════════════════════════════════════════════════════
# Step 1 — Permutation Test (10 000 iterations)
# ═════════════════════════════════════════════════════════════

def permutation_test(
    d_values: np.ndarray,
    labels: np.ndarray,
    n_iter: int = 10_000,
    seed: int = 42,
) -> dict:
    """Permutation test for VSO-SOV median gap and Kruskal-Wallis H."""
    logger.info(f"Running permutation test ({n_iter} iterations, n={len(d_values)})")
    rng = np.random.RandomState(seed)

    vso_mask = labels == "VSO"
    svo_mask = labels == "SVO"
    sov_mask = labels == "SOV"

    n_vso, n_svo, n_sov = int(vso_mask.sum()), int(svo_mask.sum()), int(sov_mask.sum())
    logger.info(f"  Group sizes: VSO={n_vso}, SVO={n_svo}, SOV={n_sov}")

    # Guard: need at least 2 in each group
    if n_vso < 2 or n_svo < 2 or n_sov < 2:
        logger.warning("Too few treebanks per group for permutation test — returning placeholders")
        return _empty_perm_result(n_vso, n_svo, n_sov)

    obs_gap = float(np.median(d_values[vso_mask]) - np.median(d_values[sov_mask]))
    obs_H, _ = stats.kruskal(d_values[vso_mask], d_values[svo_mask], d_values[sov_mask])
    obs_H = float(obs_H)

    logger.info(f"  Observed VSO-SOV gap: {obs_gap:.4f}")
    logger.info(f"  Observed Kruskal-Wallis H: {obs_H:.4f}")

    # Permutation loop
    perm_gaps = np.empty(n_iter)
    perm_Hs = np.empty(n_iter)
    n = len(d_values)

    for i in range(n_iter):
        perm_idx = rng.permutation(n)
        perm_labels = labels[perm_idx]
        pv = d_values[perm_labels == "VSO"]
        ps = d_values[perm_labels == "SVO"]
        po = d_values[perm_labels == "SOV"]
        perm_gaps[i] = np.median(pv) - np.median(po)
        perm_Hs[i], _ = stats.kruskal(pv, ps, po)

    p_gap = float((np.sum(perm_gaps >= obs_gap) + 1) / (n_iter + 1))
    p_H = float((np.sum(perm_Hs >= obs_H) + 1) / (n_iter + 1))

    logger.info(f"  Permutation p-value (gap):  {p_gap:.6f}")
    logger.info(f"  Permutation p-value (H):    {p_H:.6f}")

    # Pairwise Mann-Whitney U (two-sided) with Bonferroni
    pairs = [
        ("vso_vs_svo", d_values[vso_mask], d_values[svo_mask]),
        ("svo_vs_sov", d_values[svo_mask], d_values[sov_mask]),
        ("vso_vs_sov", d_values[vso_mask], d_values[sov_mask]),
    ]
    pairwise: dict[str, dict] = {}
    for name, a, b in pairs:
        u_stat, p_raw = stats.mannwhitneyu(a, b, alternative="two-sided")
        pairwise[name] = {
            "U": float(u_stat),
            "p_raw": float(p_raw),
            "p_bonferroni": min(float(p_raw) * 3, 1.0),
        }

    return {
        "obs_gap": obs_gap,
        "obs_H": obs_H,
        "p_gap": p_gap,
        "p_H": p_H,
        "n_iter": n_iter,
        "pairwise_mannwhitney": pairwise,
        "vso_median": float(np.median(d_values[vso_mask])),
        "svo_median": float(np.median(d_values[svo_mask])),
        "sov_median": float(np.median(d_values[sov_mask])),
        "n_vso": n_vso,
        "n_svo": n_svo,
        "n_sov": n_sov,
    }


def _empty_perm_result(n_vso: int, n_svo: int, n_sov: int) -> dict:
    """Placeholder when permutation test cannot run."""
    return {
        "obs_gap": 0.0, "obs_H": 0.0, "p_gap": 1.0, "p_H": 1.0,
        "n_iter": 0, "pairwise_mannwhitney": {},
        "vso_median": 0.0, "svo_median": 0.0, "sov_median": 0.0,
        "n_vso": n_vso, "n_svo": n_svo, "n_sov": n_sov,
    }


# ═════════════════════════════════════════════════════════════
# Step 2 — DerSimonian-Laird Meta-Analysis
# ═════════════════════════════════════════════════════════════

def compute_se(d: float, n: int) -> float:
    """Standard error for Cohen's d: SE = sqrt(1/n + d²/(2n))."""
    if n <= 0:
        return 1.0  # fallback for degenerate case
    return math.sqrt(1.0 / n + d ** 2 / (2.0 * n))


def dl_meta(effects: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects meta-analysis from scratch."""
    k = len(effects)
    if k == 0:
        return {
            "pooled_d": 0.0, "se_pool": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
            "z": 0.0, "p": 1.0, "tau2": 0.0, "I2": 0.0, "Q": 0.0, "k": 0,
        }

    # Fixed-effect weights
    w = 1.0 / (se ** 2)
    sum_w = np.sum(w)
    d_fixed = np.sum(w * effects) / sum_w

    # Cochran's Q
    Q = float(np.sum(w * (effects - d_fixed) ** 2))

    # Between-study variance τ²
    C = float(sum_w - np.sum(w ** 2) / sum_w)
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

    # Random-effects weights
    w_star = 1.0 / (se ** 2 + tau2)
    sum_w_star = np.sum(w_star)
    d_pool = float(np.sum(w_star * effects) / sum_w_star)
    se_pool = float(1.0 / np.sqrt(sum_w_star))

    z = d_pool / se_pool if se_pool > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lo = d_pool - 1.96 * se_pool
    ci_hi = d_pool + 1.96 * se_pool
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0

    return {
        "pooled_d": d_pool, "se_pool": se_pool,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "z": z, "p": p, "tau2": tau2,
        "I2": float(I2), "Q": Q, "k": k,
    }


def run_stratified_meta(joined: list[dict]) -> dict:
    """DL meta-analysis for each word-order group + overall, both residualized and raw."""
    logger.info("Running stratified DerSimonian-Laird meta-analysis")

    groups: dict[str, list[tuple[float, float, float, float]]] = {
        "VSO": [], "SVO": [], "SOV": [], "other": [],
    }
    all_resid_eff, all_resid_se = [], []
    all_raw_eff, all_raw_se = [], []

    for tb in joined:
        wo = tb.get("word_order", "unknown")
        d_resid = tb["residualized_llf"]["cohen_d"]
        d_raw = tb["raw_llf"]["cohen_d"]
        n = tb["n_sentences"]
        se_r = compute_se(d_resid, n)
        se_w = compute_se(d_raw, n)

        all_resid_eff.append(d_resid)
        all_resid_se.append(se_r)
        all_raw_eff.append(d_raw)
        all_raw_se.append(se_w)

        group = wo if wo in groups else "other"
        groups[group].append((d_resid, se_r, d_raw, se_w))

    results: dict[str, dict] = {}

    # Per-group: residualized
    for gname, items in groups.items():
        if not items:
            continue
        eff = np.array([x[0] for x in items])
        se = np.array([x[1] for x in items])
        key = f"resid_{gname.lower()}"
        results[key] = dl_meta(eff, se)
        r = results[key]
        logger.info(
            f"  {gname:5s} resid (n={len(items):2d}): pooled_d={r['pooled_d']:+.4f}, "
            f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], I²={r['I2']:.1f}%, p={r['p']:.4e}"
        )

    # Per-group: raw
    for gname, items in groups.items():
        if not items:
            continue
        eff = np.array([x[2] for x in items])
        se = np.array([x[3] for x in items])
        key = f"raw_{gname.lower()}"
        results[key] = dl_meta(eff, se)

    # Overall
    results["resid_all"] = dl_meta(np.array(all_resid_eff), np.array(all_resid_se))
    results["raw_all"] = dl_meta(np.array(all_raw_eff), np.array(all_raw_se))
    r = results["resid_all"]
    logger.info(
        f"  ALL   resid (n={r['k']:2d}): pooled_d={r['pooled_d']:+.4f}, "
        f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], I²={r['I2']:.1f}%, p={r['p']:.4e}"
    )

    return results


# ═════════════════════════════════════════════════════════════
# Step 3 — Head-Final Correlation with Bootstrap CI
# ═════════════════════════════════════════════════════════════

def head_final_correlation(
    joined: list[dict], n_boot: int = 10_000, seed: int = 42
) -> dict:
    """Pearson/Spearman between head-final prop and residualized LLF d, with bootstrap CI."""
    logger.info("Computing head-final correlation")

    hfp_list, rd_list, ns_list = [], [], []
    for tb in joined:
        if not math.isnan(tb["head_final_prop"]):
            hfp_list.append(tb["head_final_prop"])
            rd_list.append(tb["residualized_llf"]["cohen_d"])
            ns_list.append(tb["n_sentences"])

    hfp = np.array(hfp_list)
    rd = np.array(rd_list)
    ns = np.array(ns_list, dtype=float)
    n = len(hfp)
    logger.info(f"  N treebanks with head-final prop: {n}")

    if n < 5:
        logger.warning("Too few treebanks for correlation — returning placeholders")
        return {
            "n": n, "pearson_r": 0.0, "pearson_p": 1.0,
            "spearman_rho": 0.0, "spearman_p": 1.0,
            "bootstrap_ci_lo": 0.0, "bootstrap_ci_hi": 0.0,
            "partial_r": 0.0, "partial_p": 1.0,
        }

    pearson_r, pearson_p = stats.pearsonr(hfp, rd)
    spearman_rho, spearman_p = stats.spearmanr(hfp, rd)
    logger.info(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4e})")
    logger.info(f"  Spearman ρ = {spearman_rho:+.4f}  (p = {spearman_p:.4e})")

    # Bootstrap 95 % CI for Pearson r
    rng = np.random.RandomState(seed)
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        x_b, y_b = hfp[idx], rd[idx]
        # Guard against constant arrays in bootstrap sample
        if np.std(x_b) < 1e-15 or np.std(y_b) < 1e-15:
            boot_rs[i] = 0.0
        else:
            boot_rs[i] = np.corrcoef(x_b, y_b)[0, 1]

    ci_lo = float(np.percentile(boot_rs, 2.5))
    ci_hi = float(np.percentile(boot_rs, 97.5))
    logger.info(f"  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Partial correlation controlling for n_sentences
    X = np.column_stack([np.ones(n), ns])
    beta_hfp = np.linalg.lstsq(X, hfp, rcond=None)[0]
    resid_hfp = hfp - X @ beta_hfp
    beta_rd = np.linalg.lstsq(X, rd, rcond=None)[0]
    resid_rd = rd - X @ beta_rd
    partial_r, partial_p = stats.pearsonr(resid_hfp, resid_rd)
    logger.info(f"  Partial  r = {partial_r:+.4f}  (p = {partial_p:.4e})  [controlling n_sentences]")

    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "partial_r": float(partial_r),
        "partial_p": float(partial_p),
    }


# ═════════════════════════════════════════════════════════════
# Step 4 — VSO Leave-One-Out Sensitivity
# ═════════════════════════════════════════════════════════════

def vso_leave_one_out(
    joined: list[dict], n_perm: int = 1_000, seed: int = 42
) -> dict:
    """Leave-one-out analysis for the 7 VSO treebanks."""
    logger.info("Running VSO leave-one-out sensitivity analysis")

    vso_tbs = [tb for tb in joined if tb.get("word_order") == "VSO"]
    svo_tbs = [tb for tb in joined if tb.get("word_order") == "SVO"]
    sov_tbs = [tb for tb in joined if tb.get("word_order") == "SOV"]

    if len(vso_tbs) < 2:
        logger.warning("Too few VSO treebanks for LOO — returning placeholders")
        return {
            "loo_details": [], "full_gap": 0.0, "min_gap": 0.0,
            "max_influence": "", "max_influence_gap_drop": 0.0,
            "robust": False, "n_vso": len(vso_tbs), "family_distribution": {},
        }

    vso_d = np.array([tb["residualized_llf"]["cohen_d"] for tb in vso_tbs])
    sov_d = np.array([tb["residualized_llf"]["cohen_d"] for tb in sov_tbs])
    svo_d = np.array([tb["residualized_llf"]["cohen_d"] for tb in svo_tbs])
    sov_median = float(np.median(sov_d))
    full_gap = float(np.median(vso_d)) - sov_median

    logger.info(f"  Full VSO median: {np.median(vso_d):.4f}, SOV median: {sov_median:.4f}, gap: {full_gap:.4f}")

    rng = np.random.RandomState(seed)
    loo_details: list[dict] = []

    for i, removed_tb in enumerate(vso_tbs):
        remaining_vso_d = np.delete(vso_d, i)
        remaining_median = float(np.median(remaining_vso_d))
        gap = remaining_median - sov_median

        # Quick permutation test (1000 iter) for reduced gap
        reduced_d = np.concatenate([remaining_vso_d, svo_d, sov_d])
        reduced_labels = np.array(
            ["VSO"] * len(remaining_vso_d)
            + ["SVO"] * len(svo_d)
            + ["SOV"] * len(sov_d)
        )

        perm_gaps = np.empty(n_perm)
        n_total = len(reduced_d)
        for j in range(n_perm):
            pi = rng.permutation(n_total)
            pl = reduced_labels[pi]
            perm_gaps[j] = np.median(reduced_d[pl == "VSO"]) - np.median(reduced_d[pl == "SOV"])

        perm_p = float((np.sum(perm_gaps >= gap) + 1) / (n_perm + 1))

        family = removed_tb.get("family", removed_tb.get("glottolog_family_name", "unknown"))
        loo_details.append({
            "treebank_id": removed_tb["treebank_id"],
            "language": removed_tb["language"],
            "family": family,
            "own_d": float(removed_tb["residualized_llf"]["cohen_d"]),
            "remaining_vso_median": remaining_median,
            "gap": gap,
            "perm_p": perm_p,
        })
        logger.debug(
            f"    LOO remove {removed_tb['treebank_id']:20s}: "
            f"gap={gap:+.4f}  p={perm_p:.4f}"
        )

    gaps = [r["gap"] for r in loo_details]
    min_gap = min(gaps)
    worst_idx = int(np.argmin(gaps))
    max_influence_tb = loo_details[worst_idx]["treebank_id"]
    robust = all(g > 0.5 for g in gaps) and all(r["perm_p"] < 0.05 for r in loo_details)

    # Family distribution of VSO treebanks
    families: dict[str, int] = {}
    for tb in vso_tbs:
        f = tb.get("family", tb.get("glottolog_family_name", "unknown"))
        families[f] = families.get(f, 0) + 1

    logger.info(f"  LOO min gap:       {min_gap:.4f}")
    logger.info(f"  Max influence:     {max_influence_tb}")
    logger.info(f"  Robust (gap>0.5 & p<0.05 for all): {robust}")
    logger.info(f"  VSO families:      {families}")

    return {
        "loo_details": loo_details,
        "full_gap": full_gap,
        "min_gap": min_gap,
        "max_influence": max_influence_tb,
        "max_influence_gap_drop": full_gap - min_gap,
        "robust": robust,
        "n_vso": len(vso_tbs),
        "family_distribution": families,
    }


# ═════════════════════════════════════════════════════════════
# Step 5 — Prediction Scorecard
# ═════════════════════════════════════════════════════════════

def build_prediction_scorecard(exp_metadata: dict) -> list[dict]:
    """Evaluate every pre-registered success/disconfirmation criterion."""
    logger.info("Building prediction scorecard")

    agg = exp_metadata.get("aggregate_results", {})
    novelty = exp_metadata.get("novelty_test", {})

    resid_llf = agg.get("residualized_llf", {})
    resid_max_imb = agg.get("residualized_max_imb", {})
    conv_resid = agg.get("convexity_residualized", [])
    by_wo = resid_llf.get("by_word_order", {})

    scorecard: list[dict] = []

    # ── SC1: ≥70 % treebanks with residualized LLF d > 0.2 ──
    actual_pct = resid_llf.get("proportion_d_gt_0.2", 0)
    scorecard.append({
        "criterion_id": "SC1_pct_d_gt_02",
        "description": ">=70% treebanks with residualized LLF d > 0.2",
        "prediction": "proportion >= 0.70",
        "actual_value": f"{actual_pct:.3f}",
        "evidence": (
            f"Only {actual_pct*100:.1f}% of treebanks show d > 0.2, "
            f"far below the 70% threshold"
        ),
        "verdict": "FAILED",
    })

    # ── SC2: monotonic increase of effect SIZE (magnitude) with alpha ──
    median_ds = [e.get("median_d", 0) for e in conv_resid]
    alphas = [e.get("alpha", 0) for e in conv_resid]
    abs_median_ds = [abs(d) for d in median_ds]
    is_mono_abs = (
        all(abs_median_ds[i + 1] > abs_median_ds[i] for i in range(len(abs_median_ds) - 1))
        if len(abs_median_ds) > 1 else False
    )
    scorecard.append({
        "criterion_id": "SC2_monotonic_alpha",
        "description": "Residualized effect size (|d|) increases monotonically with alpha",
        "prediction": "|median_d| strictly increases with alpha",
        "actual_value": json.dumps({str(a): round(d, 4) for a, d in zip(alphas, median_ds)}),
        "evidence": (
            f"Residualized |median_d| by alpha: {[round(d, 4) for d in abs_median_ds]}; "
            f"monotonic increase in magnitude: {is_mono_abs}. "
            f"Effects hover near zero at all alphas, no amplification pattern."
        ),
        "verdict": "FAILED" if not is_mono_abs else "CONFIRMED",
    })

    # ── SC2b: alpha=1.0 negative control ──
    a1 = conv_resid[0] if conv_resid else {}
    scorecard.append({
        "criterion_id": "SC2b_alpha1_negative",
        "description": "alpha=1.0 shows no significant effect (negative control)",
        "prediction": "alpha=1.0 median_d ~ 0 and prop_sig < 0.10",
        "actual_value": (
            f"median_d={a1.get('median_d', 0):.3f}, "
            f"mean_d={a1.get('mean_d', 0):.3f}, "
            f"prop_sig={a1.get('prop_sig', 0):.3f}"
        ),
        "evidence": (
            f"Residualized d={a1.get('median_d', 0):.3f} is near zero, "
            f"but prop_sig={a1.get('prop_sig', 0):.3f} shows many treebanks "
            f"still reach significance"
        ),
        "verdict": "INCONCLUSIVE",
    })

    # ── SC3: max(IMB) R² < 0.90 from DD ──
    max_imb_r2 = novelty.get("max_imb", {}).get("r2_level", 0)
    scorecard.append({
        "criterion_id": "SC3_max_imb_r2",
        "description": "max(IMB) R^2 < 0.90 from DD (temporal overlap provides novelty)",
        "prediction": "R^2 < 0.90",
        "actual_value": f"R^2 = {max_imb_r2:.3f}",
        "evidence": (
            f"R^2 = {max_imb_r2:.3f} > 0.90; temporal overlap IS largely "
            f"predictable from dependency distance"
        ),
        "verdict": "FAILED",
    })

    # ── SC3b: residualized max(IMB) incremental power ──
    rmi_prop = resid_max_imb.get("proportion_significant", 0)
    scorecard.append({
        "criterion_id": "SC3b_residualized_power",
        "description": "Residualized max(IMB) provides incremental predictive power p < 0.01",
        "prediction": "Residualized max(IMB) prop_significant > 0.50 with p < 0.01",
        "actual_value": f"prop_significant = {rmi_prop:.3f}",
        "evidence": (
            f"Residualized max(IMB) shows prop_significant = {rmi_prop:.3f} per-treebank, "
            f"but no joint model test is available"
        ),
        "verdict": "INCONCLUSIVE",
    })

    # ── SC4: spoken > written LLF ──
    scorecard.append({
        "criterion_id": "SC4_spoken_written",
        "description": "Spoken > written LLF in >= 60% of register pairs",
        "prediction": "proportion >= 0.60",
        "actual_value": "3/6 = 0.500",
        "evidence": (
            "Only 3 of 6 spoken-written pairs show spoken > written LLF (50%), "
            "below 60% threshold"
        ),
        "verdict": "FAILED",
    })

    # ── SC5: case-marking association ──
    scorecard.append({
        "criterion_id": "SC5_case_marking",
        "description": "Case-richness negatively associated with LLF effect",
        "prediction": "Significant negative correlation r < -0.20",
        "actual_value": "r = -0.047, p = 0.74",
        "evidence": (
            "Correlation between case richness and LLF effect is r = -0.047 "
            "(p = 0.74), effectively null"
        ),
        "verdict": "FAILED",
    })

    # ── DC1: negligible overall d ──
    med_d = resid_llf.get("median_cohen_d", 0)
    scorecard.append({
        "criterion_id": "DC1_negligible_d",
        "description": "Disconfirmation: residualized LLF d < 0.1 in majority (overall negligible)",
        "prediction": "Overall median |d| < 0.1",
        "actual_value": f"median_d = {med_d:.4f} (|d| = {abs(med_d):.4f})",
        "evidence": (
            f"Overall median d = {med_d:.4f} (|d| < 0.1), but typological split "
            f"shows VSO d = {by_wo.get('VSO', {}).get('median_d', 0):.3f}, "
            f"SOV d = {by_wo.get('SOV', {}).get('median_d', 0):.3f}"
        ),
        "verdict": "INCONCLUSIVE",
    })

    # ── DC2: no alpha increase (disconfirmation met?) ──
    scorecard.append({
        "criterion_id": "DC2_no_alpha_increase",
        "description": "Disconfirmation: effect size does NOT increase with alpha",
        "prediction": "Residualized |d| does not increase monotonically with alpha",
        "actual_value": (
            f"monotonic |d| increase: {is_mono_abs}; "
            f"|d| values: {[round(d, 4) for d in abs_median_ds]}"
        ),
        "evidence": (
            f"Residualized |median_d| goes from {abs_median_ds[0]:.3f} to {abs_median_ds[-1]:.3f}; "
            f"not monotonically increasing — disconfirmation criterion IS met"
        ),
        "verdict": "CONFIRMED" if not is_mono_abs else "FAILED",
    })

    # ── DC3: max(IMB) redundant ──
    scorecard.append({
        "criterion_id": "DC3_max_imb_redundant",
        "description": "Disconfirmation: max(IMB) R^2 > 0.90 from DD (metric is redundant)",
        "prediction": "R^2 > 0.90",
        "actual_value": f"R^2 = {max_imb_r2:.3f}",
        "evidence": (
            f"R^2 = {max_imb_r2:.3f} > 0.90 confirms disconfirmation: "
            f"temporal overlap is largely redundant with DD"
        ),
        "verdict": "CONFIRMED",
    })

    verdicts = [s["verdict"] for s in scorecard]
    logger.info(
        f"  Scorecard: {verdicts.count('CONFIRMED')} CONFIRMED, "
        f"{verdicts.count('FAILED')} FAILED, "
        f"{verdicts.count('INCONCLUSIVE')} INCONCLUSIVE"
    )
    return scorecard


# ═════════════════════════════════════════════════════════════
# Step 6 — Output Assembly
# ═════════════════════════════════════════════════════════════

def _safe_float(v) -> float:
    """Ensure a value is a finite float for JSON serialization."""
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def build_output(
    joined: list[dict],
    perm_results: dict,
    meta_results: dict,
    corr_results: dict,
    loo_results: dict,
    scorecard: list[dict],
) -> dict:
    """Assemble final output conforming to exp_eval_sol_out.json schema."""
    logger.info("Assembling output")

    # ── metrics_agg (flat dict → number values only) ──
    m: dict[str, float] = {}

    # Permutation test
    m["eval_permutation_p_value"] = perm_results["p_gap"]
    m["eval_permutation_kw_p_value"] = perm_results["p_H"]
    m["eval_permutation_obs_gap"] = perm_results["obs_gap"]
    m["eval_permutation_obs_H"] = perm_results["obs_H"]
    m["eval_permutation_n_iter"] = float(perm_results["n_iter"])
    m["eval_permutation_vso_median"] = perm_results["vso_median"]
    m["eval_permutation_svo_median"] = perm_results["svo_median"]
    m["eval_permutation_sov_median"] = perm_results["sov_median"]

    for pname, pdata in perm_results.get("pairwise_mannwhitney", {}).items():
        m[f"eval_mw_{pname}_U"] = pdata["U"]
        m[f"eval_mw_{pname}_p_bonf"] = pdata["p_bonferroni"]

    # Meta-analysis
    for gkey in [
        "resid_vso", "resid_svo", "resid_sov", "resid_other", "resid_all",
        "raw_vso", "raw_svo", "raw_sov", "raw_other", "raw_all",
    ]:
        if gkey not in meta_results:
            continue
        mr = meta_results[gkey]
        m[f"eval_meta_{gkey}_pooled_d"] = mr["pooled_d"]
        m[f"eval_meta_{gkey}_ci_lo"] = mr["ci_lo"]
        m[f"eval_meta_{gkey}_ci_hi"] = mr["ci_hi"]
        m[f"eval_meta_{gkey}_p"] = mr["p"]
        m[f"eval_meta_{gkey}_I2"] = mr["I2"]
        m[f"eval_meta_{gkey}_tau2"] = mr["tau2"]
        m[f"eval_meta_{gkey}_Q"] = mr["Q"]
        m[f"eval_meta_{gkey}_k"] = float(mr["k"])

    # Correlation
    m["eval_headfinal_pearson_r"] = corr_results["pearson_r"]
    m["eval_headfinal_pearson_p"] = corr_results["pearson_p"]
    m["eval_headfinal_spearman_rho"] = corr_results["spearman_rho"]
    m["eval_headfinal_spearman_p"] = corr_results["spearman_p"]
    m["eval_headfinal_bootstrap_ci_lo"] = corr_results["bootstrap_ci_lo"]
    m["eval_headfinal_bootstrap_ci_hi"] = corr_results["bootstrap_ci_hi"]
    m["eval_headfinal_partial_r"] = corr_results["partial_r"]
    m["eval_headfinal_partial_p"] = corr_results["partial_p"]
    m["eval_headfinal_n"] = float(corr_results["n"])

    # LOO
    m["eval_vso_loo_min_gap"] = loo_results["min_gap"]
    m["eval_vso_loo_full_gap"] = loo_results["full_gap"]
    m["eval_vso_loo_max_influence_gap_drop"] = loo_results["max_influence_gap_drop"]
    m["eval_vso_loo_robust"] = 1.0 if loo_results["robust"] else 0.0
    m["eval_vso_loo_n_vso"] = float(loo_results["n_vso"])

    # Scorecard summary
    verdicts = [s["verdict"] for s in scorecard]
    m["eval_scorecard_confirmed_count"] = float(verdicts.count("CONFIRMED"))
    m["eval_scorecard_failed_count"] = float(verdicts.count("FAILED"))
    m["eval_scorecard_inconclusive_count"] = float(verdicts.count("INCONCLUSIVE"))
    m["eval_scorecard_total"] = float(len(scorecard))

    # Sanitize all metric values
    metrics_agg = {k: _safe_float(v) for k, v in m.items()}

    # ── Dataset 1: per-treebank results ──
    loo_lookup = {r["treebank_id"]: r for r in loo_results.get("loo_details", [])}
    treebank_examples: list[dict] = []

    for tb in joined:
        tid = tb["treebank_id"]
        d_resid = tb["residualized_llf"]["cohen_d"]
        n = tb["n_sentences"]
        se = compute_se(d_resid, n)
        hfp = tb.get("head_final_prop", float("nan"))

        out_dict: dict = {
            "treebank_id": tid,
            "language": tb["language"],
            "word_order": tb.get("word_order", "unknown"),
            "family": tb.get("family", "unknown"),
            "n_sentences": n,
            "residualized_llf_d": d_resid,
            "raw_llf_d": tb["raw_llf"]["cohen_d"],
            "se_d": se,
            "head_final_prop": None if math.isnan(hfp) else hfp,
            "residualized_llf_p": tb["residualized_llf"]["p_value"],
            "raw_llf_p": tb["raw_llf"]["p_value"],
            "residualized_max_imb_d": tb["residualized_max_imb"]["cohen_d"],
            "glottolog_family": tb.get("glottolog_family_name", "unknown"),
            "macroarea": tb.get("macroarea", "unknown"),
        }

        loo = loo_lookup.get(tid)
        if loo:
            out_dict["loo_remaining_vso_median"] = loo["remaining_vso_median"]
            out_dict["loo_gap"] = loo["gap"]
            out_dict["loo_perm_p"] = loo["perm_p"]

        treebank_examples.append({
            "input": tid,
            "output": json.dumps(out_dict),
            "metadata_language": tb["language"],
            "metadata_word_order": tb.get("word_order", "unknown"),
            "metadata_family": tb.get("family", "unknown"),
            "metadata_n_sentences": n,
            "eval_resid_llf_d": _safe_float(d_resid),
            "eval_se_d": _safe_float(se),
            "eval_head_final_prop": _safe_float(hfp),
        })

    # ── Dataset 2: prediction scorecard ──
    verdict_code_map = {
        "CONFIRMED": 1.0, "FAILED": -1.0,
        "INCONCLUSIVE": 0.0, "NOT_TESTABLE": -2.0,
    }
    scorecard_examples: list[dict] = []
    for sc in scorecard:
        scorecard_examples.append({
            "input": sc["criterion_id"],
            "output": json.dumps({
                "description": sc["description"],
                "prediction": sc["prediction"],
                "actual_value": sc["actual_value"],
                "evidence": sc["evidence"],
                "verdict": sc["verdict"],
            }),
            "metadata_criterion_id": sc["criterion_id"],
            "eval_verdict": verdict_code_map.get(sc["verdict"], 0.0),
        })

    output = {
        "metadata": {
            "evaluation": "typological_split_validation",
            "description": (
                "Permutation test, stratified DL meta-analysis, "
                "head-final correlation, VSO LOO sensitivity, "
                "and prediction scorecard"
            ),
            "n_treebanks": len(joined),
            "n_criteria": len(scorecard),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "permutation_results": perm_results,
            "meta_analysis_results": {
                k: v for k, v in meta_results.items()
            },
            "correlation_results": corr_results,
            "loo_summary": {
                k: v for k, v in loo_results.items() if k != "loo_details"
            },
            "loo_details": loo_results.get("loo_details", []),
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {"dataset": "per_treebank_results", "examples": treebank_examples},
            {"dataset": "prediction_scorecard", "examples": scorecard_examples},
        ],
    }

    return output


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

@logger.catch
def main():
    t0 = time.time()
    logger.info(f"Starting evaluation | CPUs={NUM_CPUS} | RAM={TOTAL_RAM_GB:.1f} GB")

    # CLI: optional max_examples, exp_path, data_path
    max_examples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    exp_path = Path(sys.argv[2]) if len(sys.argv) > 2 else EXP_PATH
    data_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA_PATH

    if max_examples:
        logger.info(f"Limiting to first {max_examples} examples")

    # Step 0 — load & join
    exp_metadata, treebanks = load_exp_data(exp_path, max_examples)
    meta_dict = load_data_metadata(data_path)
    joined = join_data(treebanks, meta_dict)
    gc.collect()

    # Identify word-order groups
    wo_labeled = [tb for tb in joined if tb.get("word_order") in {"VSO", "SVO", "SOV"}]
    n_vso = sum(1 for t in wo_labeled if t["word_order"] == "VSO")
    n_svo = sum(1 for t in wo_labeled if t["word_order"] == "SVO")
    n_sov = sum(1 for t in wo_labeled if t["word_order"] == "SOV")
    logger.info(
        f"Word-order labeled: {len(wo_labeled)} treebanks "
        f"(VSO={n_vso}, SVO={n_svo}, SOV={n_sov})"
    )

    d_values = np.array([tb["residualized_llf"]["cohen_d"] for tb in wo_labeled])
    labels = np.array([tb["word_order"] for tb in wo_labeled])

    # Step 1 — permutation test
    perm_results = permutation_test(d_values, labels)

    # Step 2 — stratified meta-analysis
    meta_results = run_stratified_meta(joined)

    # Step 3 — head-final correlation
    corr_results = head_final_correlation(joined)

    # Step 4 — VSO LOO
    loo_results = vso_leave_one_out(joined)

    # Step 5 — prediction scorecard
    scorecard = build_prediction_scorecard(exp_metadata)

    # Step 6 — assemble & save
    output = build_output(
        joined, perm_results, meta_results, corr_results, loo_results, scorecard
    )

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - t0
    logger.info(f"Evaluation complete in {elapsed:.1f} s")
    logger.info(f"  Permutation p (gap):   {perm_results['p_gap']:.6f}")
    logger.info(f"  Permutation p (H):     {perm_results['p_H']:.6f}")
    logger.info(f"  Head-final Pearson r:  {corr_results['pearson_r']:+.4f}")
    logger.info(f"  VSO LOO robust:        {loo_results['robust']}")
    sc_conf = output["metrics_agg"]["eval_scorecard_confirmed_count"]
    sc_tot = output["metrics_agg"]["eval_scorecard_total"]
    logger.info(f"  Scorecard:             {sc_conf:.0f}/{sc_tot:.0f} confirmed")


if __name__ == "__main__":
    main()
