#!/usr/bin/env python3
"""
Compute IMB Profiles & Phase 1 Novelty Test: max(IMB) vs DD Moments R².

Processes 870K sentences across 335 UD treebanks to compute standard and
encounter-only Instantaneous Memory Burden (IMB) profiles, verifies the
decomposition identity total_IMB = sum d_k(d_k+1)/2, and executes the
Phase 1 novelty test: regress max(IMB) on DD moments and tree properties
to quantify how much temporal dependency overlap information is genuinely
invisible to aggregate DD statistics (R² < 0.90 threshold).
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

import numpy as np
import pandas as pd
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
# Use 70% of RAM for safety (container limit ~29GB)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1024**3)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f}GB")

# Set memory limit
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
    logger.info("Memory limit set via resource.setrlimit")
except (ValueError, OSError) as e:
    logger.warning(f"Could not set memory limit: {e}")

# ============================================================
# PATHS
# ============================================================
_ITER1_DIR = Path("/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/3_invention_loop/iter_1/gen_art")
DATA_ID2_DIR = _ITER1_DIR / "data_id2_it1__opus"
DATA_ID3_DIR = _ITER1_DIR / "data_id3_it1__opus"
MINI_FILE = DATA_ID2_DIR / "mini_data_out.json"
SHARDS = [DATA_ID2_DIR / f"full_data_out/full_data_out_{i}.json" for i in range(1, 24)]
TYPOLOGY_FILE = DATA_ID3_DIR / "full_data_out.json"

# ============================================================
# PHASE 2: CORE IMB COMPUTATION FUNCTIONS
# ============================================================

def compute_imb_profile_standard(heads: list[int]) -> list[float]:
    """Compute standard IMB at each position.

    heads: list of length n (0-indexed array). heads[i] is the 1-indexed head
           of token at 1-indexed position (i+1). heads[i]==0 means root.

    Returns: imb_profile of length n (0-indexed, imb_profile[j] = IMB at position j+1)
    """
    n = len(heads)
    imb = [0.0] * n

    for i in range(n):
        dep_pos = i + 1        # 1-indexed position of dependent
        head_pos = heads[i]    # 1-indexed position of head
        if head_pos == 0:
            continue            # root, skip
        lo = min(dep_pos, head_pos)
        hi = max(dep_pos, head_pos)
        # This dependency is open at positions lo..hi (1-indexed)
        # At position j (1-indexed), age = j - lo
        for j_1idx in range(lo, hi + 1):
            imb[j_1idx - 1] += (j_1idx - lo)

    return imb


def compute_imb_profile_encounter_only(heads: list[int]) -> list[float]:
    """Compute encounter-only IMB variant.

    For head-final deps (head_pos > dep_pos): skip (head not yet encountered)
    For head-initial deps (head_pos < dep_pos): same as standard
    """
    n = len(heads)
    imb_enc = [0.0] * n

    for i in range(n):
        dep_pos = i + 1
        head_pos = heads[i]
        if head_pos == 0:
            continue
        lo = min(dep_pos, head_pos)
        hi = max(dep_pos, head_pos)

        if head_pos < dep_pos:
            # Head-initial: both encountered by position dep_pos
            for j_1idx in range(lo, hi + 1):
                imb_enc[j_1idx - 1] += (j_1idx - lo)
        # else: Head-final: skip entirely

    return imb_enc


def extract_imb_features(imb_profile: list[float]) -> dict:
    """Extract summary statistics from an IMB profile."""
    n = len(imb_profile)
    if n == 0:
        return {"total_imb": 0.0, "max_imb": 0.0, "mean_imb": 0.0,
                "llf": 1.0, "imb_variance": 0.0}
    total_imb = sum(imb_profile)
    max_imb = max(imb_profile)
    mean_imb = total_imb / n
    llf = mean_imb / max_imb if max_imb > 0 else 1.0
    if n > 1:
        imb_var = sum((x - mean_imb) ** 2 for x in imb_profile) / n
    else:
        imb_var = 0.0
    return {
        "total_imb": total_imb,
        "max_imb": max_imb,
        "mean_imb": round(mean_imb, 6),
        "llf": round(llf, 6),
        "imb_variance": round(imb_var, 6),
    }


def verify_decomposition(dd_list: list[int], total_imb: float) -> bool:
    """Verify total_imb == sum(d*(d+1)/2 for d in dd_list)."""
    expected = sum(d * (d + 1) / 2 for d in dd_list)
    return abs(total_imb - expected) < 1e-6


# ============================================================
# PHASE 1: LOAD TYPOLOGICAL METADATA
# ============================================================

def load_typology() -> tuple[dict, dict]:
    """Load typological metadata and build head_type_map.
    Returns: (typology_lookup, head_type_map)
    """
    logger.info(f"Loading typology from {TYPOLOGY_FILE}")
    with open(TYPOLOGY_FILE) as f:
        typo_data = json.load(f)

    typology_lookup = {}
    for example in typo_data["datasets"][0]["examples"]:
        tb_id = example["input"]
        meta = json.loads(example["output"])
        typology_lookup[tb_id] = meta

    logger.info(f"Loaded typology for {len(typology_lookup)} treebanks")

    # Classify each treebank
    head_type_map = {}
    for tb_id, meta in typology_lookup.items():
        wo = meta.get("wals_word_order")
        if wo == "SOV":
            head_type_map[tb_id] = "head_final"
        elif wo in ("SVO", "VSO"):
            head_type_map[tb_id] = "head_initial"
        else:
            lp = meta.get("head_direction_left_proportion", 0.5)
            if lp is None:
                lp = 0.5
            if lp < 0.4:
                head_type_map[tb_id] = "head_final"
            elif lp > 0.6:
                head_type_map[tb_id] = "head_initial"
            else:
                head_type_map[tb_id] = "mixed"

    ht_counts = {}
    for v in head_type_map.values():
        ht_counts[v] = ht_counts.get(v, 0) + 1
    logger.info(f"Head type distribution: {ht_counts}")

    del typo_data
    gc.collect()
    return typology_lookup, head_type_map


# ============================================================
# PILOT TREEBANKS SELECTION
# ============================================================

PILOT_HEAD_FINAL = ["tr_imst", "ja_gsd", "ko_kaist", "hi_hdtb", "ta_ttb",
                    "fa_perdt", "eu_bdt", "ur_udtb", "kk_ktb", "ab_abnc"]
PILOT_HEAD_INITIAL = ["en_ewt", "fr_gsd", "de_gsd", "es_ancora", "it_isdt",
                      "pt_bosque", "ru_syntagrus", "zh_gsd", "nl_alpino", "sv_talbanken"]
PILOT_VSO_OTHER = ["ar_padt", "ga_idt", "he_htb", "tl_trg", "gd_arcosg"]
PILOT_SPOKEN = ["fr_rhapsodie", "cs_oral", "sl_sst", "no_nynorsklia", "sv_lines"]
PILOT_TREEBANKS = set(PILOT_HEAD_FINAL + PILOT_HEAD_INITIAL + PILOT_VSO_OTHER + PILOT_SPOKEN)


# ============================================================
# PHASE 5: WORKED EXAMPLE REPLICATION
# ============================================================

def run_worked_examples() -> dict:
    """Verify the two trees from the hypothesis."""
    logger.info("Running worked example replication...")

    # Tree A: linear chain 1->2->3->4, plus 5->3
    WORKED_A_HEADS = [0, 1, 2, 2, 3]
    # Tree B: star from root, then 3->4 and 3->5
    WORKED_B_HEADS = [0, 1, 1, 3, 3]

    imb_a = compute_imb_profile_standard(WORKED_A_HEADS)
    imb_b = compute_imb_profile_standard(WORKED_B_HEADS)
    feat_a = extract_imb_features(imb_a)
    feat_b = extract_imb_features(imb_b)

    # Verify expected values
    assert imb_a == [0, 1, 2, 3, 2], f"Tree A IMB mismatch: {imb_a}"
    assert imb_b == [0, 2, 2, 2, 2], f"Tree B IMB mismatch: {imb_b}"
    assert feat_a["max_imb"] == 3, f"Tree A max_imb={feat_a['max_imb']}, expected 3"
    assert feat_b["max_imb"] == 2, f"Tree B max_imb={feat_b['max_imb']}, expected 2"

    # DD multisets
    dd_a = [abs((i + 1) - WORKED_A_HEADS[i]) for i in range(5) if WORKED_A_HEADS[i] != 0]
    dd_b = [abs((i + 1) - WORKED_B_HEADS[i]) for i in range(5) if WORKED_B_HEADS[i] != 0]
    assert sorted(dd_a) == sorted(dd_b) == [1, 1, 2, 2], \
        f"DD multisets mismatch: {sorted(dd_a)} vs {sorted(dd_b)}"

    # Quadratic cost
    cost_a_quad = sum(x ** 2 for x in imb_a)
    cost_b_quad = sum(x ** 2 for x in imb_b)
    assert cost_a_quad == 18 and cost_b_quad == 16, \
        f"Quad costs: A={cost_a_quad}, B={cost_b_quad}"

    # Linear cost
    cost_a_lin = sum(imb_a)
    cost_b_lin = sum(imb_b)
    assert cost_a_lin == cost_b_lin == 8, \
        f"Linear costs: A={cost_a_lin}, B={cost_b_lin}"

    # Decomposition identity on worked examples
    assert verify_decomposition(dd_a, feat_a["total_imb"]), "Decomp failed Tree A"
    assert verify_decomposition(dd_b, feat_b["total_imb"]), "Decomp failed Tree B"

    logger.info("Worked example replication: ALL PASSED")

    return {
        "tree_a": {"heads": WORKED_A_HEADS, "imb_profile": imb_a,
                   "max_imb": 3, "llf": round(feat_a["llf"], 3),
                   "dd_multiset": sorted(dd_a)},
        "tree_b": {"heads": WORKED_B_HEADS, "imb_profile": imb_b,
                   "max_imb": 2, "llf": round(feat_b["llf"], 3),
                   "dd_multiset": sorted(dd_b)},
        "cost_quadratic": {"tree_a": cost_a_quad, "tree_b": cost_b_quad,
                           "reduction_pct": round(100 * (cost_a_quad - cost_b_quad) / cost_a_quad, 1)},
        "cost_linear": {"tree_a": cost_a_lin, "tree_b": cost_b_lin,
                        "identical": cost_a_lin == cost_b_lin},
    }


# ============================================================
# EDGE CASE TESTS
# ============================================================

def run_edge_case_tests():
    """Test edge cases for IMB computation."""
    logger.info("Running edge case tests...")

    # Single dependency: heads=[0, 2] means token 1 is root, token 2 depends on... wait
    # Actually heads=[0, 1] means: token1 is root, token2's head is token1
    # Let me be precise: heads is 0-indexed array, heads[i] is 1-indexed head of token i+1
    # heads=[0, 1]: token 1 is root (heads[0]=0), token 2 has head at position 1 (heads[1]=1)
    # dep_pos=2, head_pos=1, lo=1, hi=2
    # At pos 1: age = 1-1 = 0; at pos 2: age = 2-1 = 1
    # IMB = [0, 1]
    imb = compute_imb_profile_standard([0, 1])
    assert imb == [0, 1], f"Single dep: {imb}"
    feat = extract_imb_features(imb)
    assert feat["max_imb"] == 1
    assert abs(feat["llf"] - 0.5) < 1e-6

    # Left-branching chain: 1<-2<-3<-4 (root is 4)
    # heads = [2, 3, 4, 0]: token1->2, token2->3, token3->4, token4 is root
    imb_lb = compute_imb_profile_standard([2, 3, 4, 0])
    # dep 1->2: lo=1,hi=2, ages: pos1=0, pos2=1
    # dep 2->3: lo=2,hi=3, ages: pos2=0, pos3=1
    # dep 3->4: lo=3,hi=4, ages: pos3=0, pos4=1
    # IMB = [0, 1+0, 1+0, 1] = [0, 1, 1, 1]
    assert imb_lb == [0, 1, 1, 1], f"Left-branch: {imb_lb}"

    # Star topology: all from root at position 1
    # heads = [0, 1, 1, 1]: token1 root, tokens 2,3,4 depend on 1
    imb_star = compute_imb_profile_standard([0, 1, 1, 1])
    # dep 2->1: lo=1,hi=2, ages: pos1=0, pos2=1
    # dep 3->1: lo=1,hi=3, ages: pos1=0, pos2=1, pos3=2
    # dep 4->1: lo=1,hi=4, ages: pos1=0, pos2=1, pos3=2, pos4=3
    # IMB = [0+0+0, 1+1+1, 2+2, 3] = [0, 3, 4, 3]
    assert imb_star == [0, 3, 4, 3], f"Star: {imb_star}"

    # Encounter-only on head-final: heads=[3, 3, 0] (tokens 1,2 -> token 3)
    # All deps are head-final, so encounter-only should be all zeros
    imb_enc = compute_imb_profile_encounter_only([3, 3, 0])
    assert imb_enc == [0, 0, 0], f"Enc head-final: {imb_enc}"

    # Encounter-only on head-initial: heads=[0, 1, 1] (tokens 2,3 -> token 1)
    # All deps are head-initial, so encounter-only == standard
    imb_enc2 = compute_imb_profile_encounter_only([0, 1, 1])
    imb_std2 = compute_imb_profile_standard([0, 1, 1])
    assert imb_enc2 == imb_std2, f"Enc head-initial mismatch: {imb_enc2} vs {imb_std2}"

    # Verify encounter-only <= standard for all positions
    test_heads = [0, 1, 2, 2, 3]
    imb_s = compute_imb_profile_standard(test_heads)
    imb_e = compute_imb_profile_encounter_only(test_heads)
    for j in range(len(test_heads)):
        assert imb_e[j] <= imb_s[j] + 1e-9, \
            f"Enc > Std at pos {j}: {imb_e[j]} > {imb_s[j]}"

    logger.info("Edge case tests: ALL PASSED")


# ============================================================
# PHASE 3: VALIDATE ON MINI DATA
# ============================================================

def validate_mini_data() -> int:
    """Validate IMB computation and decomposition identity on mini data."""
    logger.info(f"Loading mini data from {MINI_FILE}")
    with open(MINI_FILE) as f:
        mini = json.load(f)

    success_count = 0
    fail_count = 0

    for ds in mini["datasets"]:
        tb_id = ds["dataset"]
        for ex in ds["examples"]:
            inp = json.loads(ex["input"])
            out = json.loads(ex["output"])
            heads = inp["heads"]
            dd_list = out["dd_list"]

            imb_std = compute_imb_profile_standard(heads)
            imb_enc = compute_imb_profile_encounter_only(heads)
            feat_std = extract_imb_features(imb_std)
            feat_enc = extract_imb_features(imb_enc)

            # Verify decomposition identity
            if verify_decomposition(dd_list, feat_std["total_imb"]):
                success_count += 1
                logger.debug(f"TB={tb_id} sent={ex.get('metadata_sent_id','')} "
                             f"n={out['sentence_length']} max_imb={feat_std['max_imb']} "
                             f"llf={feat_std['llf']:.3f} decomp=OK")
            else:
                fail_count += 1
                expected = sum(d * (d + 1) / 2 for d in dd_list)
                logger.error(f"Decomposition FAILED: TB={tb_id} "
                             f"sent={ex.get('metadata_sent_id','')} "
                             f"total_imb={feat_std['total_imb']} expected={expected}")

            # Verify encounter-only <= standard
            assert feat_enc["max_imb"] <= feat_std["max_imb"] + 1e-9, \
                f"Enc max > Std max: {feat_enc['max_imb']} > {feat_std['max_imb']}"

    logger.info(f"Mini validation: {success_count} passed, {fail_count} failed "
                f"out of {success_count + fail_count}")

    if fail_count > 0:
        raise ValueError(f"Decomposition identity failed on {fail_count} mini examples!")

    del mini
    gc.collect()
    return success_count


# ============================================================
# SHARD PROCESSING FUNCTION (for parallelism)
# ============================================================

def process_single_example(inp_str: str, out_str: str, tb_id: str, sent_id: str,
                           tb_meta: dict, head_type: str) -> dict:
    """Process a single example and return feature dict."""
    inp = json.loads(inp_str)
    out = json.loads(out_str)
    heads = inp["heads"]
    dd_list = out["dd_list"]

    # Compute IMB profiles
    imb_std = compute_imb_profile_standard(heads)
    imb_enc = compute_imb_profile_encounter_only(heads)
    feat_std = extract_imb_features(imb_std)
    feat_enc = extract_imb_features(imb_enc)

    return {
        "treebank_id": tb_id,
        "sent_id": sent_id,
        "sentence_length": out["sentence_length"],
        "effective_length": out["effective_length"],
        "tree_depth": out["tree_depth"],
        "max_arity": out["max_arity"],
        "mean_arity": out["mean_arity"],
        "mean_dd": out["mean_dd"],
        "dd_variance": out["dd_variance"],
        "dd_skewness": out["dd_skewness"],
        "mean_ic": out["mean_ic"],
        "ic_variance": out["ic_variance"],
        "max_ic": out["max_ic"],
        "projectivity_proportion": out["projectivity_proportion"],
        "head_direction_entropy": out["head_direction_entropy"],
        # Standard IMB features
        "max_imb": feat_std["max_imb"],
        "mean_imb": feat_std["mean_imb"],
        "total_imb": feat_std["total_imb"],
        "llf": feat_std["llf"],
        "imb_variance": feat_std["imb_variance"],
        # Encounter-only IMB features
        "max_imb_enc": feat_enc["max_imb"],
        "mean_imb_enc": feat_enc["mean_imb"],
        "total_imb_enc": feat_enc["total_imb"],
        "llf_enc": feat_enc["llf"],
        "imb_variance_enc": feat_enc["imb_variance"],
        # Metadata
        "head_type": head_type,
        "modality": tb_meta.get("modality", "written"),
        "wals_word_order": tb_meta.get("wals_word_order") or "",
        "language_family": tb_meta.get("glottolog_family_name") or "",
        # For decomposition spot-check
        "_dd_list": dd_list,
        "_total_imb": feat_std["total_imb"],
    }


def process_shard(shard_idx: int, shard_path: Path,
                  typology_lookup: dict, head_type_map: dict,
                  check_all_decomp: bool = False) -> tuple[list[dict], int, int]:
    """Process a single shard file. Returns (rows, decomp_checks, decomp_fails)."""
    t0 = time.time()
    logger.info(f"Processing shard {shard_idx}/23: {shard_path.name}")

    with open(shard_path) as f:
        shard_data = json.load(f)

    rows = []
    decomp_checks = 0
    decomp_fails = 0

    if isinstance(shard_data, dict) and "datasets" in shard_data:
        examples_iter = [
            (group["dataset"], ex)
            for group in shard_data["datasets"]
            for ex in group["examples"]
        ]
    elif isinstance(shard_data, list):
        examples_iter = [
            (ex.get("metadata_treebank_id", "unknown"), ex)
            for ex in shard_data
        ]
    else:
        logger.error(f"Unknown shard format in {shard_path}")
        return rows, 0, 0

    shard_count = 0
    for tb_id, example in examples_iter:
        tb_meta = typology_lookup.get(tb_id, {})
        ht = head_type_map.get(tb_id, "mixed")
        sent_id = example.get("metadata_sent_id", "")

        try:
            row = process_single_example(
                example["input"], example["output"],
                tb_id, sent_id, tb_meta, ht
            )
        except Exception as e:
            logger.warning(f"Failed processing tb={tb_id} sent={sent_id}: {e}")
            continue

        # Spot-check decomposition
        if check_all_decomp or shard_count % 1000 == 0:
            decomp_checks += 1
            if not verify_decomposition(row["_dd_list"], row["_total_imb"]):
                decomp_fails += 1
                logger.warning(f"Decomposition FAIL: tb={tb_id} sent={sent_id}")

        # Remove temporary fields
        del row["_dd_list"]
        del row["_total_imb"]

        rows.append(row)
        shard_count += 1

    elapsed = time.time() - t0
    logger.info(f"Shard {shard_idx} done: {shard_count} sentences in {elapsed:.1f}s "
                f"({shard_count / max(elapsed, 0.001):.0f} sent/s)")

    del shard_data
    gc.collect()
    return rows, decomp_checks, decomp_fails


# ============================================================
# PHASE 6: REGRESSION ANALYSIS
# ============================================================

PREDICTORS = ["mean_dd", "dd_variance", "dd_skewness", "sentence_length",
              "tree_depth", "max_arity", "mean_arity"]


def run_per_treebank_regression(df: pd.DataFrame, head_type_map: dict) -> dict:
    """Run per-treebank OLS regression of max_imb on DD moments + tree properties."""
    logger.info("Running per-treebank regressions...")
    per_treebank_results = {}
    treebank_groups = df.groupby("treebank_id")

    for tb_id, tb_df in treebank_groups:
        if len(tb_df) < 30:
            per_treebank_results[tb_id] = {
                "n": len(tb_df), "r2_std": None, "r2_adj_std": None,
                "r2_enc": None, "r2_adj_enc": None,
                "skipped": "too_few_sentences",
                "head_type": head_type_map.get(tb_id, "mixed"),
            }
            continue

        X = tb_df[PREDICTORS].values.astype(float)
        y_std = tb_df["max_imb"].values.astype(float)
        y_enc = tb_df["max_imb_enc"].values.astype(float)

        # Check for NaN/inf
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_std) & np.isfinite(y_enc)
        if mask.sum() < 30:
            per_treebank_results[tb_id] = {
                "n": int(mask.sum()), "r2_std": None, "r2_adj_std": None,
                "r2_enc": None, "r2_adj_enc": None,
                "skipped": "too_few_finite",
                "head_type": head_type_map.get(tb_id, "mixed"),
            }
            continue

        X_clean = X[mask]
        y_std_clean = y_std[mask]
        y_enc_clean = y_enc[mask]
        X_const = sm.add_constant(X_clean)

        r2_std = r2_adj_std = r2_enc = r2_adj_enc = None
        resid_std_mean = resid_std_sd = resid_enc_mean = resid_enc_sd = None

        # Standard max_imb regression
        try:
            model_std = sm.OLS(y_std_clean, X_const).fit()
            r2_std = model_std.rsquared
            r2_adj_std = model_std.rsquared_adj
            resid_std_mean = float(np.mean(model_std.resid))
            resid_std_sd = float(np.std(model_std.resid))
        except Exception as e:
            logger.warning(f"OLS failed for {tb_id} (std): {e}")

        # Encounter-only max_imb regression
        try:
            model_enc = sm.OLS(y_enc_clean, X_const).fit()
            r2_enc = model_enc.rsquared
            r2_adj_enc = model_enc.rsquared_adj
            resid_enc_mean = float(np.mean(model_enc.resid))
            resid_enc_sd = float(np.std(model_enc.resid))
        except Exception as e:
            logger.warning(f"OLS failed for {tb_id} (enc): {e}")

        ht = head_type_map.get(tb_id, "mixed")
        per_treebank_results[tb_id] = {
            "n": int(mask.sum()),
            "head_type": ht,
            "r2_std": round(r2_std, 4) if r2_std is not None else None,
            "r2_adj_std": round(r2_adj_std, 4) if r2_adj_std is not None else None,
            "r2_enc": round(r2_enc, 4) if r2_enc is not None else None,
            "r2_adj_enc": round(r2_adj_enc, 4) if r2_adj_enc is not None else None,
            "residual_std_mean": round(resid_std_mean, 4) if resid_std_mean is not None else None,
            "residual_std_sd": round(resid_std_sd, 4) if resid_std_sd is not None else None,
            "residual_enc_mean": round(resid_enc_mean, 4) if resid_enc_mean is not None else None,
            "residual_enc_sd": round(resid_enc_sd, 4) if resid_enc_sd is not None else None,
        }

    logger.info(f"Per-treebank regressions done: {len(per_treebank_results)} treebanks")
    return per_treebank_results


def run_pooled_regression(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Run pooled OLS regression across all sentences."""
    logger.info("Running pooled regressions...")

    X_all = df[PREDICTORS].values.astype(float)
    y_all_std = df["max_imb"].values.astype(float)
    y_all_enc = df["max_imb_enc"].values.astype(float)

    # Filter NaN/inf
    mask = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all_std) & np.isfinite(y_all_enc)
    X_clean = X_all[mask]
    y_std_clean = y_all_std[mask]
    y_enc_clean = y_all_enc[mask]
    X_const = sm.add_constant(X_clean)

    model_pooled_std = sm.OLS(y_std_clean, X_const).fit()
    model_pooled_enc = sm.OLS(y_enc_clean, X_const).fit()

    # Store residualized max(IMB) in DataFrame
    df_out = df.copy()
    df_out["resid_max_imb"] = np.nan
    df_out["resid_max_imb_enc"] = np.nan
    df_out.loc[mask, "resid_max_imb"] = model_pooled_std.resid
    df_out.loc[mask, "resid_max_imb_enc"] = model_pooled_enc.resid

    pooled_results = {
        "standard": {
            "r2": round(model_pooled_std.rsquared, 4),
            "r2_adj": round(model_pooled_std.rsquared_adj, 4),
            "n": int(mask.sum()),
            "coefficients": {name: round(float(coef), 6)
                             for name, coef in zip(["const"] + PREDICTORS,
                                                   model_pooled_std.params)},
            "pvalues": {name: float(pv)
                        for name, pv in zip(["const"] + PREDICTORS,
                                            model_pooled_std.pvalues)},
        },
        "encounter_only": {
            "r2": round(model_pooled_enc.rsquared, 4),
            "r2_adj": round(model_pooled_enc.rsquared_adj, 4),
            "n": int(mask.sum()),
            "coefficients": {name: round(float(coef), 6)
                             for name, coef in zip(["const"] + PREDICTORS,
                                                   model_pooled_enc.params)},
            "pvalues": {name: float(pv)
                        for name, pv in zip(["const"] + PREDICTORS,
                                            model_pooled_enc.pvalues)},
        },
    }

    logger.info(f"Pooled R² (std): {pooled_results['standard']['r2']:.4f}, "
                f"R² (enc): {pooled_results['encounter_only']['r2']:.4f}")

    return pooled_results, df_out


def compute_r2_by_head_type(per_treebank_results: dict, head_type_map: dict) -> dict:
    """Compute R² summary statistics by head type."""
    r2_by_head_type = {}
    for ht in ["head_final", "head_initial", "mixed"]:
        tb_ids = [tb for tb, h in head_type_map.items() if h == ht]
        r2_vals_std = [per_treebank_results[tb]["r2_std"]
                       for tb in tb_ids if tb in per_treebank_results
                       and per_treebank_results[tb].get("r2_std") is not None]
        r2_vals_enc = [per_treebank_results[tb]["r2_enc"]
                       for tb in tb_ids if tb in per_treebank_results
                       and per_treebank_results[tb].get("r2_enc") is not None]
        if r2_vals_std:
            r2_by_head_type[ht] = {
                "n_treebanks": len(r2_vals_std),
                "r2_std_mean": round(float(np.mean(r2_vals_std)), 4),
                "r2_std_median": round(float(np.median(r2_vals_std)), 4),
                "r2_std_q25": round(float(np.percentile(r2_vals_std, 25)), 4),
                "r2_std_q75": round(float(np.percentile(r2_vals_std, 75)), 4),
                "r2_std_below_090": sum(1 for r in r2_vals_std if r < 0.90),
                "r2_enc_mean": round(float(np.mean(r2_vals_enc)), 4) if r2_vals_enc else None,
                "r2_enc_median": round(float(np.median(r2_vals_enc)), 4) if r2_vals_enc else None,
            }
    return r2_by_head_type


def compute_residual_stats(df: pd.DataFrame) -> dict:
    """Compute residualized max(IMB) distribution statistics."""
    resid_stats = {}
    for ht in ["head_final", "head_initial", "mixed", "all"]:
        if ht == "all":
            subset = df
        else:
            subset = df[df["head_type"] == ht]
        if len(subset) == 0:
            continue
        for variant in ["resid_max_imb", "resid_max_imb_enc"]:
            vals = subset[variant].dropna().values
            if len(vals) == 0:
                continue
            resid_stats[f"{ht}_{variant}"] = {
                "n": len(vals),
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "q05": round(float(np.percentile(vals, 5)), 4),
                "q25": round(float(np.percentile(vals, 25)), 4),
                "q75": round(float(np.percentile(vals, 75)), 4),
                "q95": round(float(np.percentile(vals, 95)), 4),
                "skewness": round(float(scipy_stats.skew(vals)), 4),
                "kurtosis": round(float(scipy_stats.kurtosis(vals)), 4),
            }
    return resid_stats


def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics of IMB features by head type."""
    desc_stats = {}
    for ht in ["head_final", "head_initial", "mixed", "all"]:
        subset = df if ht == "all" else df[df["head_type"] == ht]
        if len(subset) == 0:
            continue
        desc_stats[ht] = {
            "n_sentences": len(subset),
            "n_treebanks": int(subset["treebank_id"].nunique()),
        }
        for col in ["max_imb", "mean_imb", "llf", "max_imb_enc", "llf_enc",
                     "mean_dd", "dd_variance", "sentence_length", "tree_depth"]:
            vals = subset[col].values
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) == 0:
                continue
            desc_stats[ht][col] = {
                "mean": round(float(np.mean(finite_vals)), 4),
                "std": round(float(np.std(finite_vals)), 4),
                "median": round(float(np.median(finite_vals)), 4),
            }
    return desc_stats


# ============================================================
# OUTPUT FORMATTING (exp_gen_sol_out schema)
# ============================================================

def format_output_schema(df: pd.DataFrame, method_out: dict,
                         per_treebank_results: dict,
                         typology_lookup: dict, head_type_map: dict) -> dict:
    """Format output according to exp_gen_sol_out.json schema.

    Schema requires: {"datasets": [{"dataset": str, "examples": [{"input": str, "output": str}]}]}
    We create ONE dataset group with 335 examples (one per treebank).
    Each example's input = treebank metadata, output = IMB summary + regression results.
    """
    logger.info("Formatting output to exp_gen_sol_out schema (treebank-level)...")

    imb_cols = ["max_imb", "mean_imb", "total_imb", "llf", "imb_variance",
                "max_imb_enc", "mean_imb_enc", "total_imb_enc", "llf_enc", "imb_variance_enc"]
    dd_cols = ["mean_dd", "dd_variance", "dd_skewness", "sentence_length",
               "tree_depth", "max_arity", "mean_arity"]

    examples = []
    for tb_id, tb_df in df.groupby("treebank_id"):
        tb_meta = typology_lookup.get(tb_id, {})
        ht = head_type_map.get(tb_id, "mixed")
        reg = per_treebank_results.get(tb_id, {})

        # Input: treebank metadata
        input_data = {
            "treebank_id": tb_id,
            "head_type": ht,
            "wals_word_order": tb_meta.get("wals_word_order") or "",
            "modality": tb_meta.get("modality", "written"),
            "language_family": tb_meta.get("glottolog_family_name") or "",
            "macroarea": tb_meta.get("macroarea") or "",
            "n_sentences": len(tb_df),
        }

        # Output: treebank-level IMB summary + regression results
        output_data = {"n_sentences": len(tb_df)}
        for col in imb_cols + dd_cols:
            vals = tb_df[col].values
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                output_data[f"{col}_mean"] = round(float(np.mean(finite)), 4)
                output_data[f"{col}_median"] = round(float(np.median(finite)), 4)
                output_data[f"{col}_std"] = round(float(np.std(finite)), 4)

        # Regression results
        output_data["r2_std"] = reg.get("r2_std")
        output_data["r2_adj_std"] = reg.get("r2_adj_std")
        output_data["r2_enc"] = reg.get("r2_enc")
        output_data["r2_adj_enc"] = reg.get("r2_adj_enc")
        output_data["residual_std_sd"] = reg.get("residual_std_sd")
        output_data["residual_enc_sd"] = reg.get("residual_enc_sd")

        # predict_imb_method: our IMB-based novelty analysis
        predict_imb = {
            "max_imb_mean": output_data.get("max_imb_mean"),
            "llf_mean": output_data.get("llf_mean"),
            "max_imb_enc_mean": output_data.get("max_imb_enc_mean"),
            "r2_std": reg.get("r2_std"),
            "r2_enc": reg.get("r2_enc"),
            "novelty_detected": (reg.get("r2_std") or 1.0) < 0.90,
        }
        # predict_baseline: DD-moments-only baseline (no IMB)
        predict_baseline = {
            "mean_dd_mean": output_data.get("mean_dd_mean"),
            "dd_variance_mean": output_data.get("dd_variance_mean"),
            "dd_skewness_mean": output_data.get("dd_skewness_mean"),
            "r2_std": reg.get("r2_std"),
            "baseline_note": "DD moments predict max_imb with this R2; residual is novel IMB signal",
        }

        examples.append({
            "input": json.dumps(input_data),
            "output": json.dumps(output_data),
            "predict_imb_method": json.dumps(predict_imb, default=str),
            "predict_baseline": json.dumps(predict_baseline, default=str),
            "metadata_treebank_id": tb_id,
            "metadata_head_type": ht,
            "metadata_modality": tb_meta.get("modality", "written"),
            "metadata_wals_word_order": tb_meta.get("wals_word_order") or "",
            "metadata_language_family": tb_meta.get("glottolog_family_name") or "",
        })

    return {
        "metadata": {
            "experiment": "phase1_imb_profiles_and_novelty_test",
            "total_sentences": int(method_out.get("total_sentences", len(df))),
            "total_treebanks": int(method_out.get("total_treebanks", df["treebank_id"].nunique())),
            "description": "IMB profiles and Phase 1 novelty test: per-treebank IMB summary and R2 regression results",
            "decomposition_identity": method_out.get("decomposition_identity", {}),
            "worked_example": method_out.get("worked_example", {}),
            "novelty_test_pooled": method_out.get("novelty_test_pooled", {}),
            "r2_by_head_type": method_out.get("r2_by_head_type", {}),
            "residualized_max_imb_distributions": method_out.get("residualized_max_imb_distributions", {}),
            "descriptive_statistics": method_out.get("descriptive_statistics", {}),
            "key_findings": method_out.get("key_findings", {}),
        },
        "datasets": [{
            "dataset": "commul/universal_dependencies",
            "examples": examples,
        }],
    }


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    t_start = time.time()

    # ---- Worked examples & edge cases ----
    worked_example_results = run_worked_examples()
    run_edge_case_tests()

    # ---- Load typology ----
    typology_lookup, head_type_map = load_typology()

    # ---- Validate on mini data ----
    mini_count = validate_mini_data()
    logger.info(f"Mini validation passed: {mini_count} examples")

    # ---- Process all 23 shards ----
    logger.info("="*60)
    logger.info("PHASE 4: Processing all 23 shards")
    logger.info("="*60)

    all_rows = []
    total_decomp_checks = 0
    total_decomp_fails = 0

    for shard_idx in range(1, 24):
        shard_path = SHARDS[shard_idx - 1]
        if not shard_path.exists():
            logger.warning(f"Shard {shard_idx} not found: {shard_path}")
            continue

        check_all = (shard_idx == 1)  # Check all decompositions in first shard
        rows, checks, fails = process_shard(
            shard_idx, shard_path, typology_lookup, head_type_map,
            check_all_decomp=check_all,
        )
        all_rows.extend(rows)
        total_decomp_checks += checks
        total_decomp_fails += fails

        # Monitor memory
        try:
            mem_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if mem_path.exists():
                mem_used = int(mem_path.read_text().strip()) / 1e9
                logger.info(f"Memory usage: {mem_used:.1f}GB / {TOTAL_RAM_GB:.1f}GB")
        except Exception:
            pass

    logger.info(f"All shards processed: {len(all_rows)} total sentences")
    logger.info(f"Decomposition checks: {total_decomp_checks}, failures: {total_decomp_fails}")

    # ---- Convert to DataFrame ----
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Unique treebanks: {df['treebank_id'].nunique()}")

    # ---- Worked example results already computed ----

    # ---- Phase 6: Regression analysis ----
    logger.info("="*60)
    logger.info("PHASE 6: Regression analysis")
    logger.info("="*60)

    per_treebank_results = run_per_treebank_regression(df, head_type_map)
    pooled_results, df = run_pooled_regression(df)
    r2_by_head_type = compute_r2_by_head_type(per_treebank_results, head_type_map)
    resid_stats = compute_residual_stats(df)
    desc_stats = compute_descriptive_stats(df)

    # ---- Pilot 30 treebanks subset ----
    pilot_results = {
        tb: per_treebank_results[tb]
        for tb in PILOT_TREEBANKS
        if tb in per_treebank_results
    }

    # ---- Save sentence-level CSV (split into parts < 95MB each) ----
    csv_dir = WORKSPACE / "sentence_imb_features"
    csv_dir.mkdir(exist_ok=True)
    logger.info(f"Saving sentence features to {csv_dir}/ (split parts)")
    csv_cols = [c for c in df.columns if not c.startswith("resid_")]
    df_out = df[csv_cols]
    # Split into 2 parts (~435K rows each, ~85MB each)
    n_parts = 2
    chunk_size = math.ceil(len(df_out) / n_parts)
    for part_idx in range(n_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, len(df_out))
        part_path = csv_dir / f"sentence_imb_features_{part_idx + 1}.csv"
        df_out.iloc[start:end].to_csv(part_path, index=False)
        part_mb = part_path.stat().st_size / 1e6
        logger.info(f"  Part {part_idx + 1}: {end - start} rows, {part_mb:.1f}MB")
    # Remove old unsplit file if it exists
    old_csv = WORKSPACE / "sentence_imb_features.csv"
    if old_csv.exists():
        old_csv.unlink()
        logger.info("Removed old unsplit sentence_imb_features.csv")

    # ---- Assemble aggregate results dict ----
    aggregate_results = {
        "experiment": "phase1_imb_profiles_and_novelty_test",
        "total_sentences": len(df),
        "total_treebanks": int(df["treebank_id"].nunique()),
        "decomposition_identity": {
            "checks_performed": total_decomp_checks,
            "failures": total_decomp_fails,
            "pass_rate": round(1 - total_decomp_fails / max(total_decomp_checks, 1), 6),
        },
        "worked_example": worked_example_results,
        "novelty_test_pooled": pooled_results,
        "novelty_test_per_treebank": per_treebank_results,
        "novelty_test_pilot_30": pilot_results,
        "r2_by_head_type": r2_by_head_type,
        "residualized_max_imb_distributions": resid_stats,
        "descriptive_statistics": desc_stats,
        "key_findings": {
            "pooled_r2_std": pooled_results["standard"]["r2"],
            "pooled_r2_enc": pooled_results["encounter_only"]["r2"],
            "novelty_confirmed": pooled_results["standard"]["r2"] < 0.90,
            "n_treebanks_r2_below_090": sum(
                1 for v in per_treebank_results.values()
                if v.get("r2_std") is not None and v["r2_std"] < 0.90
            ),
            "n_treebanks_with_r2": sum(
                1 for v in per_treebank_results.values()
                if v.get("r2_std") is not None
            ),
        },
        "supplementary_data_files": [
            "sentence_imb_features/sentence_imb_features_1.csv",
            "sentence_imb_features/sentence_imb_features_2.csv",
        ],
    }

    # ---- Write schema-compliant method_out.json (treebank-level) ----
    logger.info("Formatting schema-compliant output (treebank-level)...")
    schema_output = format_output_schema(
        df, aggregate_results, per_treebank_results,
        typology_lookup, head_type_map,
    )
    method_out_path = WORKSPACE / "method_out.json"
    logger.info(f"Writing method_out.json to {method_out_path}")
    method_out_path.write_text(json.dumps(schema_output, indent=2, default=str))
    method_out_size_mb = method_out_path.stat().st_size / 1e6
    logger.info(f"method_out.json size: {method_out_size_mb:.1f}MB")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Key finding: Pooled R² = {pooled_results['standard']['r2']:.4f} "
                f"(novelty {'CONFIRMED' if pooled_results['standard']['r2'] < 0.90 else 'NOT confirmed'})")


if __name__ == "__main__":
    main()
