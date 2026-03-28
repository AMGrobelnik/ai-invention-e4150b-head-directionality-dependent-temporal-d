#!/usr/bin/env python3
"""Phases 4-5: Modality Comparison, Head-Final Encounter-Only IMB, and Case-Marking Analysis.

Implements the load-smoothing hypothesis analysis:
  Phase 4:  Spoken vs written LLF comparison across verified language pairs
  Phase 5a: Standard vs encounter-only IMB across SOV and SVO treebanks
  Phase 5b: Case-marking richness correlation with LLF in SOV languages
  Phase 5c: Within-language DOM analysis (case-marked vs unmarked sentences)
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE = Path(__file__).parent

DATA_ID2_DIR = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/"
    "3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)
DATA_ID3_DIR = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/"
    "3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)

DISFLUENCY_DEPRELS = frozenset({"reparandum", "discourse:filler"})

SPOKEN_WRITTEN_PAIRS: dict[str, dict] = {
    "sl": {"spoken": ["sl_sst"], "written": "sl_ssj", "lang": "Slovenian"},
    "fr": {"spoken": ["fr_rhapsodie", "fr_parisstories"], "written": "fr_gsd", "lang": "French"},
    "it": {"spoken": ["it_valico"], "written": "it_isdt", "lang": "Italian"},
    "tr": {"spoken": ["tr_atis"], "written": "tr_boun", "lang": "Turkish"},
    "en": {"spoken": ["en_atis"], "written": "en_ewt", "lang": "English"},
}

PILOT_SOV_TREEBANKS = ["tr_boun", "ko_kaist", "hi_hdtb", "ja_gsd", "fa_perdt"]

LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


# =============================================================================
# HARDWARE AND MEMORY SETUP
# =============================================================================

def _detect_cpus() -> int:
    """Detect actual CPU allocation (container-aware)."""
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
    """Read RAM limit from cgroup (container-aware)."""
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 16.0  # conservative fallback


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1e9)

try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
except ValueError:
    pass

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, "
            f"budget {RAM_BUDGET_BYTES / 1e9:.1f} GB")


# =============================================================================
# CORE IMB COMPUTATION
# =============================================================================

def compute_imb_profile(
    heads: list[int],
    deprels: list[str],
    *,
    exclude_disfluency: bool = False,
    encounter_only: bool = False,
) -> list[float]:
    """Compute age-weighted IMB profile from 1-indexed heads (0 = root).

    Standard mode: every dependency contributes.
    Encounter-only mode: head-final deps (head right of dependent) are zeroed
    out because the head has not yet been encountered during left-to-right
    processing.
    """
    n = len(heads)
    if n == 0:
        return []

    opens: list[int] = []
    closes: list[int] = []

    for i in range(n):
        h = heads[i]
        if h == 0:
            continue
        if exclude_disfluency and deprels[i] in DISFLUENCY_DEPRELS:
            continue
        dep_pos = i + 1   # 1-indexed
        head_pos = h       # 1-indexed

        if encounter_only and head_pos > dep_pos:
            # head-final: head not yet seen → skip
            continue

        opens.append(min(dep_pos, head_pos))
        closes.append(max(dep_pos, head_pos))

    if not opens:
        return [0.0] * n

    # Vectorised computation
    ops = np.array(opens, dtype=np.int32)
    cls = np.array(closes, dtype=np.int32)
    j_pos = np.arange(1, n + 1, dtype=np.int32)[:, None]   # (n, 1)

    mask = (ops[None, :] <= j_pos) & (j_pos <= cls[None, :])  # (n, d)
    ages = j_pos - ops[None, :]                                # (n, d)

    imb = np.sum(ages * mask, axis=1).astype(float)
    return imb.tolist()


def compute_llf_metrics(imb_profile: list[float],
                        dd_list: list[int]) -> dict[str, Any]:
    """Derive LLF and related metrics from an IMB profile."""
    n = len(imb_profile)
    if n == 0:
        return {"max_imb": 0.0, "mean_imb": 0.0, "llf": 0.0,
                "total_imb": 0.0, "total_imb_formula": 0.0, "costs": {}}

    max_imb = float(max(imb_profile))
    total_imb = float(sum(imb_profile))
    mean_imb = total_imb / n
    llf = mean_imb / max_imb if max_imb > 0 else 0.0

    alphas = [1.0, 1.2, 1.5, 2.0, 3.0]
    costs = {str(a): sum(v ** a for v in imb_profile) for a in alphas}

    total_imb_formula = sum(d * (d + 1) / 2.0 for d in dd_list)

    return {
        "max_imb": max_imb,
        "mean_imb": mean_imb,
        "llf": llf,
        "total_imb": total_imb,
        "total_imb_formula": total_imb_formula,
        "costs": costs,
    }


def compute_head_final_proportion(heads: list[int]) -> float:
    """Fraction of dependencies where head is to the right of dependent."""
    n_deps = 0
    n_hf = 0
    for i, h in enumerate(heads):
        if h == 0:
            continue
        n_deps += 1
        if h > i + 1:
            n_hf += 1
    return n_hf / n_deps if n_deps > 0 else 0.0


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute Cohen's d with pooled SD."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = float(np.var(g1, ddof=1))
    var2 = float(np.var(g2, ddof=1))
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd < 1e-15:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_sd)


def bootstrap_cohens_d_ci(
    g1: np.ndarray, g2: np.ndarray,
    n_boot: int = 10000, ci: float = 0.95,
) -> list[float]:
    """Bootstrap 95 % CI for Cohen's d."""
    rng = np.random.RandomState(42)
    ds = np.empty(n_boot)
    n1, n2 = len(g1), len(g2)
    for i in range(n_boot):
        s1 = g1[rng.randint(0, n1, n1)]
        s2 = g2[rng.randint(0, n2, n2)]
        v1 = float(np.var(s1, ddof=1))
        v2 = float(np.var(s2, ddof=1))
        psd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        ds[i] = (np.mean(s1) - np.mean(s2)) / psd if psd > 1e-15 else 0.0
    alpha = (1 - ci) / 2
    return [float(np.percentile(ds, alpha * 100)),
            float(np.percentile(ds, (1 - alpha) * 100))]


def residualize_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residualisation: y ~ X  →  residuals."""
    try:
        # Drop constant columns
        col_var = np.var(X, axis=0)
        good = col_var > 1e-10
        if not np.any(good):
            return y - np.mean(y)
        X_clean = X[:, good]
        X_const = sm.add_constant(X_clean)
        model = sm.OLS(y, X_const).fit()
        return model.resid
    except Exception as e:
        logger.warning(f"OLS residualisation failed ({e}), returning centred y")
        return y - np.mean(y)


def partial_correlation(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray,
) -> tuple[float, float]:
    """Partial Pearson r of x and y controlling for covariates."""
    try:
        x_r = residualize_ols(x, covariates)
        y_r = residualize_ols(y, covariates)
        return tuple(float(v) for v in pearsonr(x_r, y_r))  # type: ignore
    except Exception as e:
        logger.warning(f"Partial correlation failed: {e}")
        return (float("nan"), float("nan"))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_metadata() -> dict[str, dict]:
    """Load treebank metadata from data_id3."""
    logger.info("Loading metadata from data_id3 ...")
    path = DATA_ID3_DIR / "full_data_out.json"
    data = json.loads(path.read_text())

    metadata: dict[str, dict] = {}
    for ex in data["datasets"][0]["examples"]:
        tb_id = ex["input"]
        out = json.loads(ex["output"])
        metadata[tb_id] = {
            "wals_word_order":                out.get("wals_word_order"),
            "modality":                       out.get("modality", "written"),
            "spoken_written_pair_id":         out.get("spoken_written_pair_id"),
            "case_richness_count":            out.get("case_richness_count", 0),
            "case_token_proportion":          out.get("case_token_proportion", 0.0),
            "has_differential_object_marking": out.get("has_differential_object_marking", False),
            "disfluency_deprels_present":     out.get("disfluency_deprels_present", []),
            "head_direction_entropy_binary":  out.get("head_direction_entropy_binary", 0.0),
            "head_direction_left_proportion": out.get("head_direction_left_proportion", 0.5),
            "glottolog_family_name":          out.get("glottolog_family_name", ""),
            "glottolog_family_id":            out.get("glottolog_family_id", ""),
            "num_sentences":                  out.get("num_sentences", 0),
            "num_tokens":                     out.get("num_tokens", 0),
            "mean_sentence_length":           out.get("mean_sentence_length", 0.0),
            "lang_code":                      out.get("lang_code", ""),
            "language_name":                  out.get("language_name", ""),
        }
    del data
    gc.collect()

    # Summary
    modalities: dict[str, int] = defaultdict(int)
    word_orders: dict[str, int] = defaultdict(int)
    n_dom = 0
    for m in metadata.values():
        modalities[m["modality"]] += 1
        word_orders[m["wals_word_order"] or "unknown"] += 1
        if m["has_differential_object_marking"]:
            n_dom += 1

    logger.info(f"Loaded metadata for {len(metadata)} treebanks")
    logger.info(f"  Modalities: {dict(modalities)}")
    logger.info(f"  Word orders: {dict(word_orders)}")
    logger.info(f"  DOM treebanks: {n_dom}")
    return metadata


def identify_targets(metadata: dict[str, dict]) -> dict:
    """Identify every treebank set needed across Phases 4-5."""
    pairs = {k: dict(v) for k, v in SPOKEN_WRITTEN_PAIRS.items()}
    # deep-copy spoken lists
    for k in pairs:
        pairs[k]["spoken"] = list(pairs[k]["spoken"])

    # Scan for additional spoken-written pairs
    for tb_id, m in metadata.items():
        if m["modality"] != "spoken":
            continue
        pid = m.get("spoken_written_pair_id")
        if not pid:
            continue
        # Already covered?
        already = False
        for p in pairs.values():
            if tb_id in p["spoken"]:
                already = True
                break
        if already:
            continue
        # Check extra English treebanks
        if tb_id in ("en_childes", "en_sptoken") and pid == "en_ewt":
            pairs["en"]["spoken"].append(tb_id)
            logger.info(f"  Added {tb_id} to English spoken group")
            continue
        # Other new pairs
        if pid in metadata:
            key = m["lang_code"] + "_extra"
            if key not in pairs:
                pairs[key] = {
                    "spoken": [tb_id],
                    "written": pid,
                    "lang": m["language_name"] + " (extra)",
                }
                logger.info(f"  Found extra pair: {tb_id} → {pid}")

    sov = [t for t, m in metadata.items() if m["wals_word_order"] == "SOV"]
    svo = [t for t, m in metadata.items() if m["wals_word_order"] == "SVO"]
    dom = [t for t, m in metadata.items() if m["has_differential_object_marking"]]

    all_needed: set[str] = set()
    for p in pairs.values():
        all_needed.update(p["spoken"])
        all_needed.add(p["written"])
    all_needed.update(sov)
    all_needed.update(svo)
    all_needed.update(dom)

    logger.info(f"Target treebanks: SOV={len(sov)}, SVO={len(svo)}, "
                f"DOM={len(dom)}, pairs={len(pairs)}, "
                f"total unique={len(all_needed)}")
    return {
        "spoken_written_pairs": pairs,
        "sov_treebanks": sov,
        "svo_treebanks": svo,
        "dom_treebanks": dom,
        "all_needed": all_needed,
    }


def load_sentences(
    needed: set[str],
    dom_set: set[str],
) -> dict[str, list[dict]]:
    """Stream-load sentence data from data_id2 split files.

    Only keeps treebanks in *needed*; stores feats only for DOM treebanks.
    """
    logger.info(f"Loading sentences for {len(needed)} treebanks …")
    sents: dict[str, list[dict]] = defaultdict(list)

    for idx in range(1, 24):
        fpath = DATA_ID2_DIR / f"full_data_out/full_data_out_{idx}.json"
        if not fpath.exists():
            logger.warning(f"  Missing: {fpath.name}")
            continue

        t0 = time.time()
        try:
            raw = fpath.read_text()
            data = json.loads(raw)
            del raw
        except (json.JSONDecodeError, MemoryError) as exc:
            logger.error(f"  Failed loading {fpath.name}: {exc}")
            continue

        for dsg in data.get("datasets", []):
            tb = dsg.get("dataset", "")
            if tb not in needed:
                continue
            want_feats = tb in dom_set
            for ex in dsg.get("examples", []):
                try:
                    inp = json.loads(ex["input"])
                    out = json.loads(ex["output"])
                except (json.JSONDecodeError, KeyError):
                    continue
                s: dict[str, Any] = {
                    "heads":      inp["heads"],
                    "deprels":    inp["deprels"],
                    "sent_len":   out["sentence_length"],
                    "tree_depth": out["tree_depth"],
                    "mean_dd":    out["mean_dd"],
                    "dd_var":     out["dd_variance"],
                    "dd_skew":    out.get("dd_skewness", 0.0),
                    "dd_list":    out["dd_list"],
                    "max_arity":  out.get("max_arity", 0),
                    "mean_arity": out.get("mean_arity", 0.0),
                    "hde":        out.get("head_direction_entropy", 0.0),
                    "sid":        ex.get("metadata_sent_id", ""),
                }
                if want_feats:
                    s["feats"] = inp.get("feats", [])
                sents[tb].append(s)

        del data
        gc.collect()
        logger.info(f"  Split {idx:>2}/23 done in {time.time()-t0:.1f}s  "
                     f"({len(sents)} treebanks so far)")

    total = sum(len(v) for v in sents.values())
    logger.info(f"Loaded {total:,} sentences across {len(sents)} treebanks")

    missing = needed - set(sents.keys())
    if missing:
        logger.warning(f"  Missing treebanks ({len(missing)}): "
                       f"{sorted(missing)[:20]} ...")
    return dict(sents)


# =============================================================================
# PHASE 1 — PILOT VALIDATION
# =============================================================================

def pilot_validation(sents: dict, metadata: dict) -> dict:
    """Micro-validation on hardcoded examples + pilot on real data."""
    logger.info("=" * 60)
    logger.info("PHASE 1: PILOT VALIDATION")
    logger.info("=" * 60)

    res: dict[str, Any] = {"micro_tests": {}, "pilot_slovenian": {},
                           "pilot_sov": {}}

    # ---- Tier 1: micro tests ----
    logger.info("Tier 1 – hardcoded micro-validation …")

    # Tree A
    h_a, d_a = [0, 1, 2, 2, 3], ["root", "dep", "dep", "dep", "dep"]
    imb_a = compute_imb_profile(h_a, d_a)
    exp_a = [0.0, 1.0, 2.0, 3.0, 2.0]
    ok_a_imb = imb_a == exp_a
    m_a = compute_llf_metrics(imb_a, [1, 1, 2, 2])
    ok_a_llf = abs(m_a["llf"] - 8 / 15) < 1e-6
    ok_a_id = abs(m_a["total_imb"] - m_a["total_imb_formula"]) < 1e-6
    logger.info(f"  A IMB  {imb_a}  expect {exp_a}  {'OK' if ok_a_imb else 'FAIL'}")
    logger.info(f"  A LLF  {m_a['llf']:.6f}  expect {8/15:.6f}  {'OK' if ok_a_llf else 'FAIL'}")
    logger.info(f"  A identity  {m_a['total_imb']} == {m_a['total_imb_formula']}  {'OK' if ok_a_id else 'FAIL'}")

    # Tree B
    h_b, d_b = [0, 1, 1, 3, 3], ["root", "dep", "dep", "dep", "dep"]
    imb_b = compute_imb_profile(h_b, d_b)
    exp_b = [0.0, 2.0, 2.0, 2.0, 2.0]
    ok_b_imb = imb_b == exp_b
    m_b = compute_llf_metrics(imb_b, [1, 2, 1, 2])
    ok_b_llf = abs(m_b["llf"] - 0.8) < 1e-6
    ok_b_id = abs(m_b["total_imb"] - m_b["total_imb_formula"]) < 1e-6
    logger.info(f"  B IMB  {imb_b}  expect {exp_b}  {'OK' if ok_b_imb else 'FAIL'}")
    logger.info(f"  B LLF  {m_b['llf']:.6f}  expect 0.800000  {'OK' if ok_b_llf else 'FAIL'}")
    logger.info(f"  B identity  {m_b['total_imb']} == {m_b['total_imb_formula']}  {'OK' if ok_b_id else 'FAIL'}")

    # Head-final tree: encounter-only should be all zeros
    h_hf, d_hf = [5, 5, 5, 5, 0], ["dep"] * 4 + ["root"]
    std_hf = compute_imb_profile(h_hf, d_hf, encounter_only=False)
    enc_hf = compute_imb_profile(h_hf, d_hf, encounter_only=True)
    ok_hf = all(v == 0.0 for v in enc_hf) and any(v > 0 for v in std_hf)
    logger.info(f"  Head-final: std={std_hf} enc={enc_hf}  {'OK' if ok_hf else 'FAIL'}")

    # Head-initial tree: encounter-only should equal standard
    h_hi, d_hi = [0, 1, 1, 1, 1], ["root"] + ["dep"] * 4
    std_hi = compute_imb_profile(h_hi, d_hi, encounter_only=False)
    enc_hi = compute_imb_profile(h_hi, d_hi, encounter_only=True)
    ok_hi = std_hi == enc_hi
    logger.info(f"  Head-initial: std={std_hi} enc={enc_hi}  {'OK' if ok_hi else 'FAIL'}")

    # Disfluency exclusion
    h_df, d_df = [0, 1, 1, 3], ["root", "dep", "reparandum", "dep"]
    no_ex = compute_imb_profile(h_df, d_df, exclude_disfluency=False)
    ex_df = compute_imb_profile(h_df, d_df, exclude_disfluency=True)
    ok_df = no_ex != ex_df
    logger.info(f"  Disfluency: no_excl={no_ex} excl={ex_df}  {'OK' if ok_df else 'FAIL'}")

    all_ok = all([ok_a_imb, ok_a_llf, ok_a_id,
                  ok_b_imb, ok_b_llf, ok_b_id,
                  ok_hf, ok_hi, ok_df])

    res["micro_tests"] = {
        "all_passed": all_ok,
        "tree_a_imb": ok_a_imb, "tree_a_llf": ok_a_llf, "identity_a": ok_a_id,
        "tree_b_imb": ok_b_imb, "tree_b_llf": ok_b_llf, "identity_b": ok_b_id,
        "encounter_head_final": ok_hf, "encounter_head_initial": ok_hi,
        "disfluency_exclusion": ok_df,
    }
    if not all_ok:
        logger.error("MICRO VALIDATION FAILED — aborting")
        return res
    logger.info("All micro tests PASSED")

    # ---- Tier 2: pilot on Slovenian pair ----
    logger.info("Tier 2 – pilot on Slovenian + 5 SOV treebanks …")

    sp = sents.get("sl_sst", [])
    wr = sents.get("sl_ssj", [])
    if sp and wr:
        sp_llfs, wr_llfs = [], []
        id_ok, id_tot = 0, 0
        for s in sp[:200]:
            imb = compute_imb_profile(s["heads"], s["deprels"],
                                      exclude_disfluency=True)
            m = compute_llf_metrics(imb, s["dd_list"])
            sp_llfs.append(m["llf"])
            id_tot += 1
            if abs(m["total_imb"] - m["total_imb_formula"]) < 0.5:
                id_ok += 1
        for s in wr[:200]:
            imb = compute_imb_profile(s["heads"], s["deprels"])
            m = compute_llf_metrics(imb, s["dd_list"])
            wr_llfs.append(m["llf"])
            id_tot += 1
            if abs(m["total_imb"] - m["total_imb_formula"]) < 0.5:
                id_ok += 1

        res["pilot_slovenian"] = {
            "n_spoken": len(sp), "n_written": len(wr),
            "mean_llf_spoken": float(np.mean(sp_llfs)),
            "mean_llf_written": float(np.mean(wr_llfs)),
            "identity_rate": id_ok / id_tot if id_tot else 0,
            "all_in_01": all(0 <= x <= 1 for x in sp_llfs + wr_llfs),
        }
        logger.info(f"  Slovenian spoken={len(sp)} written={len(wr)}  "
                     f"LLF spoken={np.mean(sp_llfs):.4f} written={np.mean(wr_llfs):.4f}  "
                     f"identity {id_ok}/{id_tot}")
    else:
        logger.warning("  Slovenian data not available")

    # Pilot SOV treebanks
    sov_pilot: dict[str, dict] = {}
    for tb in PILOT_SOV_TREEBANKS:
        ss = sents.get(tb, [])
        if not ss:
            logger.warning(f"  {tb}: no data")
            continue
        std_mx, enc_mx = [], []
        for s in ss[:100]:
            si = compute_imb_profile(s["heads"], s["deprels"])
            ei = compute_imb_profile(s["heads"], s["deprels"], encounter_only=True)
            std_mx.append(max(si) if si else 0)
            enc_mx.append(max(ei) if ei else 0)
        red = sum(1 for a, b in zip(std_mx, enc_mx) if b < a) / len(std_mx)
        sov_pilot[tb] = {
            "n": len(ss), "piloted": min(len(ss), 100),
            "mean_std_max": float(np.mean(std_mx)),
            "mean_enc_max": float(np.mean(enc_mx)),
            "pct_reduced": red,
        }
        logger.info(f"  {tb}: {len(ss)} sents, enc reduces in {red*100:.1f}%")

    res["pilot_sov"] = sov_pilot
    res["encounter_only_reduces_imb_in_sov"] = bool(
        sov_pilot and all(r["pct_reduced"] > 0.3 for r in sov_pilot.values()))
    res["identity_check_passed"] = bool(
        res.get("pilot_slovenian", {}).get("identity_rate", 0) > 0.95)
    res["n_sentences_checked"] = (
        400 + sum(r["piloted"] for r in sov_pilot.values()))

    logger.info(f"Pilot done.  Identity OK={res['identity_check_passed']}  "
                f"Enc reduces SOV={res['encounter_only_reduces_imb_in_sov']}")
    return res


# =============================================================================
# HELPER: sentence-level IMB metrics
# =============================================================================

def _sent_imb_metrics(
    sentences: list[dict],
    *,
    exclude_disfluency: bool = False,
    encounter_only: bool = False,
) -> list[dict]:
    """Compute IMB/LLF + covariates for a list of sentences."""
    out: list[dict] = []
    for s in sentences:
        imb = compute_imb_profile(
            s["heads"], s["deprels"],
            exclude_disfluency=exclude_disfluency,
            encounter_only=encounter_only,
        )
        m = compute_llf_metrics(imb, s["dd_list"])
        out.append({
            "llf":      m["llf"],
            "max_imb":  m["max_imb"],
            "mean_imb": m["mean_imb"],
            "sent_len": s["sent_len"],
            "tree_depth": s["tree_depth"],
            "mean_dd":  s["mean_dd"],
            "dd_var":   s["dd_var"],
            "dd_skew":  s.get("dd_skew", 0.0),
            "max_arity": s.get("max_arity", 0),
        })
    return out


# =============================================================================
# PHASE 4 — MODALITY COMPARISON
# =============================================================================

def _phase4_pair(
    sp_m: list[dict], wr_m: list[dict],
    pair_key: str, pair_info: dict,
) -> dict:
    """Analyse one spoken/written pair."""
    n_sp = len(sp_m)
    n_wr = len(wr_m)
    logger.info(f"  {pair_key} ({pair_info['lang']}): spoken={n_sp} written={n_wr}")

    base = {
        "spoken_tb": pair_info["spoken"],
        "written_tb": pair_info["written"],
        "lang": pair_info["lang"],
        "n_spoken": n_sp,
        "n_written": n_wr,
    }

    if n_sp < 10 or n_wr < 10:
        logger.warning(f"    underpowered ({n_sp}/{n_wr})")
        sp_llf = np.array([x["llf"] for x in sp_m]) if sp_m else np.array([])
        wr_llf = np.array([x["llf"] for x in wr_m]) if wr_m else np.array([])
        base.update({
            "mean_llf_spoken":  float(np.mean(sp_llf)) if len(sp_llf) else None,
            "mean_llf_written": float(np.mean(wr_llf)) if len(wr_llf) else None,
            "residual_llf_cohens_d": None, "residual_llf_p": None,
            "residual_llf_ci95": None, "spoken_higher": None,
            "residual_max_imb_cohens_d": None, "residual_max_imb_p": None,
            "spoken_lower_max_imb": None, "underpowered": True,
        })
        return base

    # Pool and build arrays
    all_m = sp_m + wr_m
    llf      = np.array([x["llf"] for x in all_m])
    max_imb  = np.array([x["max_imb"] for x in all_m])
    sl       = np.array([x["sent_len"] for x in all_m])
    td       = np.array([x["tree_depth"] for x in all_m])
    md       = np.array([x["mean_dd"] for x in all_m])
    dv       = np.array([x["dd_var"] for x in all_m])
    ds       = np.array([x["dd_skew"] for x in all_m])
    ma       = np.array([x["max_arity"] for x in all_m])

    # Residualise LLF
    X_llf = np.column_stack([sl, td, md, dv])
    r_llf = residualize_ols(llf, X_llf)

    # Residualise max_imb
    X_mx = np.column_stack([sl, td, md, dv, ds, ma])
    r_mx = residualize_ols(max_imb, X_mx)

    sp_rl, wr_rl = r_llf[:n_sp], r_llf[n_sp:]
    sp_rm, wr_rm = r_mx[:n_sp],  r_mx[n_sp:]

    _, p_llf = stats.ttest_ind(sp_rl, wr_rl, equal_var=False)
    d_llf = cohens_d(sp_rl, wr_rl)
    ci_llf = bootstrap_cohens_d_ci(sp_rl, wr_rl)

    _, p_mx = stats.ttest_ind(sp_rm, wr_rm, equal_var=False)
    d_mx = cohens_d(sp_rm, wr_rm)

    base.update({
        "mean_llf_spoken":  float(np.mean([x["llf"] for x in sp_m])),
        "mean_llf_written": float(np.mean([x["llf"] for x in wr_m])),
        "residual_llf_cohens_d":  float(d_llf),
        "residual_llf_p":         float(p_llf),
        "residual_llf_ci95":      [float(ci_llf[0]), float(ci_llf[1])],
        "spoken_higher":          bool(np.mean(sp_rl) > np.mean(wr_rl)),
        "residual_max_imb_cohens_d": float(d_mx),
        "residual_max_imb_p":        float(p_mx),
        "spoken_lower_max_imb":      bool(np.mean(sp_rm) < np.mean(wr_rm)),
        "underpowered": False,
    })
    logger.info(f"    LLF d={d_llf:.4f} p={p_llf:.2e}  "
                f"spoken_higher={base['spoken_higher']}")
    return base


def phase4_modality(sents: dict, metadata: dict, targets: dict) -> dict:
    """Phase 4: Spoken vs Written LLF comparison."""
    logger.info("=" * 60)
    logger.info("PHASE 4: MODALITY COMPARISON")
    logger.info("=" * 60)
    t_phase = time.time()

    pairs_results: dict[str, dict] = {}

    for pk, pi in targets["spoken_written_pairs"].items():
        t0 = time.time()
        # Gather spoken sentences
        sp_sents: list[dict] = []
        for sp_tb in pi["spoken"]:
            sp_sents.extend(sents.get(sp_tb, []))
        wr_sents = sents.get(pi["written"], [])

        if not sp_sents or not wr_sents:
            logger.warning(f"  {pk}: no data (sp={len(sp_sents)} wr={len(wr_sents)})")
            continue

        # Main: disfluency-excluded
        sp_m = _sent_imb_metrics(sp_sents, exclude_disfluency=True)
        wr_m = _sent_imb_metrics(wr_sents, exclude_disfluency=True)
        pr = _phase4_pair(sp_m, wr_m, pk, pi)

        # Sensitivity 1: no disfluency exclusion
        sp_m2 = _sent_imb_metrics(sp_sents, exclude_disfluency=False)
        wr_m2 = _sent_imb_metrics(wr_sents, exclude_disfluency=False)
        s1 = _phase4_pair(sp_m2, wr_m2, pk + "_noExcl", pi)
        pr["sensitivity_no_disfluency_exclusion"] = {
            "residual_llf_cohens_d": s1.get("residual_llf_cohens_d"),
            "residual_llf_p": s1.get("residual_llf_p"),
            "spoken_higher": s1.get("spoken_higher"),
        }

        # Sensitivity 2: encounter-only
        sp_m3 = _sent_imb_metrics(sp_sents, exclude_disfluency=True,
                                  encounter_only=True)
        wr_m3 = _sent_imb_metrics(wr_sents, exclude_disfluency=True,
                                  encounter_only=True)
        s2 = _phase4_pair(sp_m3, wr_m3, pk + "_enc", pi)
        pr["sensitivity_encounter_only"] = {
            "residual_llf_cohens_d": s2.get("residual_llf_cohens_d"),
            "residual_llf_p": s2.get("residual_llf_p"),
            "spoken_higher": s2.get("spoken_higher"),
        }

        # French sub-treebank check
        if pk == "fr" and len(pi["spoken"]) > 1:
            sub: dict[str, dict] = {}
            for sp_tb in pi["spoken"]:
                sp_sub = sents.get(sp_tb, [])
                if not sp_sub:
                    continue
                sm_ = _sent_imb_metrics(sp_sub, exclude_disfluency=True)
                sr_ = _phase4_pair(sm_, wr_m, f"fr_{sp_tb}", {
                    "spoken": [sp_tb], "written": pi["written"],
                    "lang": f"French ({sp_tb})"})
                sub[sp_tb] = {
                    "n_spoken": sr_["n_spoken"],
                    "cohens_d": sr_.get("residual_llf_cohens_d"),
                    "p": sr_.get("residual_llf_p"),
                    "spoken_higher": sr_.get("spoken_higher"),
                }
            pr["french_individual_spoken"] = sub

        # English extra spoken treebanks check
        if pk == "en" and len(pi["spoken"]) > 1:
            sub_en: dict[str, dict] = {}
            for sp_tb in pi["spoken"]:
                sp_sub = sents.get(sp_tb, [])
                if not sp_sub:
                    continue
                sm_ = _sent_imb_metrics(sp_sub, exclude_disfluency=True)
                sr_ = _phase4_pair(sm_, wr_m, f"en_{sp_tb}", {
                    "spoken": [sp_tb], "written": pi["written"],
                    "lang": f"English ({sp_tb})"})
                sub_en[sp_tb] = {
                    "n_spoken": sr_["n_spoken"],
                    "cohens_d": sr_.get("residual_llf_cohens_d"),
                    "p": sr_.get("residual_llf_p"),
                    "spoken_higher": sr_.get("spoken_higher"),
                }
            pr["english_individual_spoken"] = sub_en

        pairs_results[pk] = pr
        logger.info(f"  {pk} finished in {time.time()-t0:.1f}s")

    # Aggregate
    valid = {k: v for k, v in pairs_results.items()
             if not v.get("underpowered", False)}
    n_higher = sum(1 for v in valid.values() if v.get("spoken_higher"))
    n_sig = sum(1 for v in valid.values()
                if v.get("residual_llf_p") is not None
                and v["residual_llf_p"] < 0.01)
    ds = [v["residual_llf_cohens_d"] for v in valid.values()
          if v.get("residual_llf_cohens_d") is not None
          and np.isfinite(v["residual_llf_cohens_d"])]

    agg = {
        "n_pairs": len(valid),
        "n_spoken_higher": n_higher,
        "proportion_spoken_higher": n_higher / len(valid) if valid else 0,
        "mean_cohens_d": float(np.mean(ds)) if ds else None,
        "bonferroni_threshold": 0.01,
        "n_significant": n_sig,
    }
    logger.info(f"Phase 4 aggregate: {n_higher}/{len(valid)} spoken>written, "
                f"{n_sig} significant (Bonf p<0.01)  "
                f"elapsed={time.time()-t_phase:.1f}s")
    return {"pairs": pairs_results, "aggregate": agg}


# =============================================================================
# PHASE 5A — HEAD-FINAL EFFECTS (standard vs encounter-only IMB)
# =============================================================================

def phase5a_head_final(sents: dict, metadata: dict, targets: dict) -> dict:
    """Phase 5a: compare standard and encounter-only IMB across SOV/SVO."""
    logger.info("=" * 60)
    logger.info("PHASE 5A: HEAD-FINAL LANGUAGE EFFECTS")
    logger.info("=" * 60)
    t_phase = time.time()

    wo_tbs = set(targets["sov_treebanks"]) | set(targets["svo_treebanks"])
    tb_res: dict[str, dict] = {}

    for idx, tb in enumerate(sorted(wo_tbs)):
        ss = sents.get(tb, [])
        if not ss:
            continue
        wo = metadata[tb]["wals_word_order"]

        std_llfs, enc_llfs = [], []
        std_maxs, enc_maxs = [], []
        hf_props = []
        n_undef = 0

        for s in ss:
            si = compute_imb_profile(s["heads"], s["deprels"])
            ei = compute_imb_profile(s["heads"], s["deprels"],
                                     encounter_only=True)
            sm_ = compute_llf_metrics(si, s["dd_list"])
            em_ = compute_llf_metrics(ei, s["dd_list"])

            std_llfs.append(sm_["llf"])
            enc_llfs.append(em_["llf"])
            std_maxs.append(sm_["max_imb"])
            enc_maxs.append(em_["max_imb"])
            if em_["max_imb"] == 0:
                n_undef += 1
            hf_props.append(compute_head_final_proportion(s["heads"]))

        tb_res[tb] = {
            "word_order": wo,
            "n_sentences": len(ss),
            "mean_llf_standard":     float(np.mean(std_llfs)),
            "mean_llf_encounter":    float(np.mean(enc_llfs)),
            "llf_delta":             float(np.mean(std_llfs) - np.mean(enc_llfs)),
            "mean_max_imb_standard": float(np.mean(std_maxs)),
            "mean_max_imb_encounter":float(np.mean(enc_maxs)),
            "max_imb_delta":         float(np.mean(std_maxs) - np.mean(enc_maxs)),
            "proportion_head_final": float(np.mean(hf_props)),
            "pct_undefined_enc_llf": n_undef / len(ss),
            "language": metadata[tb].get("language_name", ""),
            "family":   metadata[tb].get("glottolog_family_name", ""),
        }
        if (idx + 1) % 25 == 0 or idx + 1 == len(wo_tbs):
            logger.info(f"  Phase 5a: {idx+1}/{len(wo_tbs)} treebanks  "
                        f"({time.time()-t_phase:.0f}s)")

    # --- Compare SOV vs SVO ---
    sov_r = [r for r in tb_res.values() if r["word_order"] == "SOV"]
    svo_r = [r for r in tb_res.values() if r["word_order"] == "SVO"]

    sov_d = np.array([r["llf_delta"] for r in sov_r])
    svo_d = np.array([r["llf_delta"] for r in svo_r])

    if len(sov_d) > 1 and len(svo_d) > 1:
        _, p_cmp = stats.ttest_ind(sov_d, svo_d, equal_var=False)
        d_cmp = cohens_d(sov_d, svo_d)
    else:
        p_cmp, d_cmp = float("nan"), float("nan")

    cmp = {
        "mean_sov": float(np.mean(sov_d)) if len(sov_d) else None,
        "mean_svo": float(np.mean(svo_d)) if len(svo_d) else None,
        "cohens_d": float(d_cmp),
        "p": float(p_cmp),
        "sov_larger": bool(np.mean(sov_d) > np.mean(svo_d))
                      if len(sov_d) and len(svo_d) else None,
    }

    # Head-final proportion distributions
    sov_hf = np.array([r["proportion_head_final"] for r in sov_r])
    svo_hf = np.array([r["proportion_head_final"] for r in svo_r])
    logger.info(f"  SOV head-final prop: {np.mean(sov_hf):.3f} +/- {np.std(sov_hf):.3f}")
    logger.info(f"  SVO head-final prop: {np.mean(svo_hf):.3f} +/- {np.std(svo_hf):.3f}")

    # Correlation: head_final_prop vs llf_delta
    all_hf = np.array([r["proportion_head_final"] for r in tb_res.values()])
    all_dl = np.array([r["llf_delta"] for r in tb_res.values()])
    if len(all_hf) > 2:
        pr_, pp_ = pearsonr(all_hf, all_dl)
        sr_, sp_ = spearmanr(all_hf, all_dl)
    else:
        pr_, pp_, sr_, sp_ = [float("nan")] * 4

    corr = {
        "pearson_r": float(pr_), "pearson_p": float(pp_),
        "spearman_r": float(sr_), "spearman_p": float(sp_),
    }

    hf_by_wo = {
        "SOV_mean": float(np.mean(sov_hf)) if len(sov_hf) else None,
        "SOV_std":  float(np.std(sov_hf))  if len(sov_hf) else None,
        "SVO_mean": float(np.mean(svo_hf)) if len(svo_hf) else None,
        "SVO_std":  float(np.std(svo_hf))  if len(svo_hf) else None,
    }

    logger.info(f"  SOV vs SVO llf_delta: d={d_cmp:.4f} p={p_cmp:.2e}")
    logger.info(f"  head-final vs llf_delta: r={pr_:.4f} p={pp_:.2e}")
    logger.info(f"  Phase 5a done in {time.time()-t_phase:.1f}s")

    sov_list = [{"tb_id": t, **r} for t, r in tb_res.items()
                if r["word_order"] == "SOV"]
    svo_list = [{"tb_id": t, **r} for t, r in tb_res.items()
                if r["word_order"] == "SVO"]
    return {
        "sov_treebanks": sov_list,
        "svo_treebanks": svo_list,
        "sov_vs_svo_llf_delta": cmp,
        "head_final_prop_vs_llf_delta_correlation": corr,
        "head_final_proportions_by_wo": hf_by_wo,
        "n_sov": len(sov_r),
        "n_svo": len(svo_r),
    }


# =============================================================================
# PHASE 5B — CASE-MARKING RICHNESS
# =============================================================================

def phase5b_case_richness(p5a: dict, metadata: dict) -> dict:
    """Phase 5b: correlate case richness with LLF in SOV languages."""
    logger.info("=" * 60)
    logger.info("PHASE 5B: CASE-MARKING RICHNESS AND LLF")
    logger.info("=" * 60)

    scatter: list[dict] = []
    for tb in p5a["sov_treebanks"]:
        tid = tb["tb_id"]
        m = metadata.get(tid, {})
        scatter.append({
            "tb_id": tid,
            "case_richness_count": m.get("case_richness_count", 0),
            "case_token_proportion": m.get("case_token_proportion", 0.0),
            "mean_llf_standard": tb["mean_llf_standard"],
            "mean_llf_encounter": tb["mean_llf_encounter"],
            "mean_sentence_length": m.get("mean_sentence_length", 0.0),
            "num_sentences": m.get("num_sentences", 0),
            "hde": m.get("head_direction_entropy_binary", 0.0),
            "language": tb.get("language", ""),
            "family":   tb.get("family", ""),
        })

    if len(scatter) < 5:
        logger.warning("Too few SOV treebanks for correlation")
        return {"n_sov_treebanks": len(scatter), "insufficient_data": True,
                "scatter_data": scatter}

    cr  = np.array([d["case_richness_count"] for d in scatter], dtype=float)
    cp  = np.array([d["case_token_proportion"] for d in scatter])
    llf = np.array([d["mean_llf_standard"] for d in scatter])
    lle = np.array([d["mean_llf_encounter"] for d in scatter])
    msl = np.array([d["mean_sentence_length"] for d in scatter])
    ns  = np.array([d["num_sentences"] for d in scatter], dtype=float)
    hde = np.array([d["hde"] for d in scatter])

    # Main correlations
    pr, pp   = pearsonr(cr, llf)
    sr, sp_  = spearmanr(cr, llf)

    # Partial correlation controlling length + size
    cov1 = np.column_stack([msl, np.log1p(ns)])
    pcr1, pcp1 = partial_correlation(cr, llf, cov1)

    # Partial + head-direction entropy
    cov2 = np.column_stack([msl, np.log1p(ns), hde])
    pcr2, pcp2 = partial_correlation(cr, llf, cov2)

    # Encounter-only LLF
    pr_e, pp_e = pearsonr(cr, lle)
    sr_e, sp_e = spearmanr(cr, lle)

    # case_token_proportion
    pr_p, pp_p = pearsonr(cp, llf)
    sr_p, sp_p = spearmanr(cp, llf)

    res = {
        "n_sov_treebanks": len(scatter),
        "case_richness_count": {
            "pearson_r": float(pr),  "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp_),
            "partial_r_controlling_length": float(pcr1),
            "partial_p_controlling_length": float(pcp1),
            "partial_r_with_hde": float(pcr2),
            "partial_p_with_hde": float(pcp2),
        },
        "case_richness_encounter_only": {
            "pearson_r": float(pr_e), "pearson_p": float(pp_e),
            "spearman_r": float(sr_e), "spearman_p": float(sp_e),
        },
        "case_token_proportion": {
            "pearson_r": float(pr_p), "pearson_p": float(pp_p),
            "spearman_r": float(sr_p), "spearman_p": float(sp_p),
        },
        "scatter_data": scatter,
    }
    logger.info(f"  case_richness vs LLF: r={pr:.4f} p={pp:.2e}")
    logger.info(f"  case_proportion vs LLF: r={pr_p:.4f} p={pp_p:.2e}")
    logger.info(f"  partial (ctrl length): r={pcr1:.4f} p={pcp1:.2e}")
    return res


# =============================================================================
# PHASE 5C — DIFFERENTIAL OBJECT MARKING
# =============================================================================

def _classify_dom(sent: dict) -> str:
    """Classify sentence by case-marking status on obj tokens."""
    deprels = sent["deprels"]
    feats = sent.get("feats", [])
    if not feats:
        return "no_feats"

    has_cm = False   # has case-marked obj
    has_um = False   # has unmarked obj
    has_obj = False

    for i, dr in enumerate(deprels):
        if not dr.startswith("obj"):
            continue
        has_obj = True
        f = feats[i] if i < len(feats) else None
        if f and "Case=" in str(f):
            has_cm = True
        else:
            has_um = True

    if not has_obj:
        return "no_obj"
    if has_cm and not has_um:
        return "case_marked"
    if has_um and not has_cm:
        return "unmarked"
    return "mixed"


def phase5c_dom(sents: dict, metadata: dict, targets: dict) -> dict:
    """Phase 5c: within-language DOM — case-marked vs unmarked obj sentences."""
    logger.info("=" * 60)
    logger.info("PHASE 5C: DIFFERENTIAL OBJECT MARKING")
    logger.info("=" * 60)

    MIN_N = 30
    per_tb: list[dict] = []
    pool_rows: list[dict] = []     # for mixed-effects model
    n_suf = 0

    for tb in sorted(targets["dom_treebanks"]):
        ss = sents.get(tb, [])
        if not ss:
            continue

        cm_s, um_s = [], []
        for s in ss:
            cat = _classify_dom(s)
            if cat == "case_marked":
                cm_s.append(s)
            elif cat == "unmarked":
                um_s.append(s)

        n_cm, n_um = len(cm_s), len(um_s)
        logger.debug(f"  {tb}: case_marked={n_cm} unmarked={n_um}")

        if n_cm < 15 or n_um < 15:
            per_tb.append({
                "tb_id": tb,
                "lang": metadata[tb].get("language_name", ""),
                "n_case_marked": n_cm, "n_unmarked": n_um,
                "sufficient_data": False,
                "cohens_d": None, "p": None, "direction": None,
            })
            continue

        # Use lower threshold if needed but flag
        used_lower = n_cm < MIN_N or n_um < MIN_N
        n_suf += 1

        cm_m = _sent_imb_metrics(cm_s)
        um_m = _sent_imb_metrics(um_s)

        all_m = cm_m + um_m
        llf = np.array([x["llf"] for x in all_m])
        sl  = np.array([x["sent_len"] for x in all_m])
        td  = np.array([x["tree_depth"] for x in all_m])
        md  = np.array([x["mean_dd"] for x in all_m])
        dv  = np.array([x["dd_var"] for x in all_m])
        X = np.column_stack([sl, td, md, dv])
        resid = residualize_ols(llf, X)

        cm_r = resid[:len(cm_m)]
        um_r = resid[len(cm_m):]
        _, pv = stats.ttest_ind(cm_r, um_r, equal_var=False)
        dv_ = cohens_d(cm_r, um_r)

        direction = ("case_marked_higher"
                     if np.mean(cm_r) > np.mean(um_r)
                     else "unmarked_higher")

        per_tb.append({
            "tb_id": tb,
            "lang": metadata[tb].get("language_name", ""),
            "n_case_marked": n_cm, "n_unmarked": n_um,
            "sufficient_data": True, "used_lower_threshold": used_lower,
            "cohens_d": float(dv_), "p": float(pv),
            "direction": direction,
            "mean_resid_llf_cm": float(np.mean(cm_r)),
            "mean_resid_llf_um": float(np.mean(um_r)),
        })

        # Accumulate for mixed-effects
        for i in range(len(cm_m)):
            pool_rows.append({"resid_llf": float(resid[i]),
                              "case_status": 1, "tb_id": tb})
        for i in range(len(um_m)):
            pool_rows.append({"resid_llf": float(resid[len(cm_m) + i]),
                              "case_status": 0, "tb_id": tb})

    suf = [r for r in per_tb if r["sufficient_data"]]
    n_pos = sum(1 for r in suf if r["direction"] == "case_marked_higher")
    logger.info(f"  DOM sufficient: {n_suf}, positive direction: {n_pos}/{len(suf)}")

    # Mixed-effects model
    mx_coef, mx_p = None, None
    if len(pool_rows) > 100 and n_suf >= 3:
        try:
            df = pd.DataFrame(pool_rows)
            mdl = MixedLM.from_formula("resid_llf ~ case_status",
                                       df, groups=df["tb_id"])
            fit = mdl.fit(reml=True)
            mx_coef = float(fit.params.get("case_status", float("nan")))
            mx_p    = float(fit.pvalues.get("case_status", float("nan")))
            logger.info(f"  MixedLM: coef={mx_coef:.4f} p={mx_p:.2e}")
        except Exception as e:
            logger.warning(f"  MixedLM failed: {e}")
            # Fallback: weighted mean Cohen's d
            d_vals = [r["cohens_d"] for r in suf
                      if r["cohens_d"] is not None and np.isfinite(r["cohens_d"])]
            n_vals = [r["n_case_marked"] + r["n_unmarked"] for r in suf
                      if r["cohens_d"] is not None and np.isfinite(r["cohens_d"])]
            if d_vals:
                w = np.sqrt(n_vals)
                mx_coef = float(np.average(d_vals, weights=w))
                logger.info(f"  Fallback weighted d: {mx_coef:.4f}")

    return {
        "n_dom_treebanks_analyzed": len(per_tb),
        "n_with_sufficient_data": n_suf,
        "min_sentences_threshold": MIN_N,
        "per_treebank": per_tb,
        "aggregate_proportion_positive": n_pos / len(suf) if suf else 0,
        "mixed_effects_coefficient": mx_coef,
        "mixed_effects_p": mx_p,
    }


# =============================================================================
# RESULTS ASSEMBLY  →  exp_gen_sol_out schema
# =============================================================================

def _s(obj: Any) -> str:
    """Safe JSON-serialise to string (replace NaN with null)."""
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not serialisable: {type(o)}")


def assemble_output(p4: dict, p5a: dict, p5b: dict, p5c: dict,
                    pilot: dict) -> dict:
    """Build the final method_out.json conforming to exp_gen_sol_out."""
    logger.info("Assembling output …")
    datasets: list[dict] = []

    # --- Phase 4 ---
    ex4: list[dict] = []
    for pk, pd_ in p4.get("pairs", {}).items():
        d_val = pd_.get("residual_llf_cohens_d")
        pred = _s({"spoken_higher": pd_.get("spoken_higher"),
                    "cohens_d": d_val,
                    "p": pd_.get("residual_llf_p")})
        ex4.append({"input": pk, "output": _s(pd_),
                     "predict_method": pred,
                     "metadata_phase": "4",
                     "metadata_analysis": "modality_comparison",
                     "metadata_language": pd_.get("lang", "")})
    agg4 = p4.get("aggregate", {})
    ex4.append({"input": "aggregate", "output": _s(agg4),
                "predict_method": _s({"proportion_spoken_higher":
                    agg4.get("proportion_spoken_higher"),
                    "mean_cohens_d": agg4.get("mean_cohens_d")}),
                "metadata_phase": "4",
                "metadata_analysis": "modality_aggregate"})
    datasets.append({"dataset": "phase4_modality_comparison", "examples": ex4})

    # --- Phase 5a ---
    ex5a: list[dict] = []
    for tb in p5a.get("sov_treebanks", []):
        ex5a.append({"input": tb["tb_id"], "output": _s(tb),
                      "predict_method": _s({"llf_delta": tb.get("llf_delta"),
                          "mean_llf_standard": tb.get("mean_llf_standard")}),
                      "metadata_phase": "5a", "metadata_word_order": "SOV"})
    for tb in p5a.get("svo_treebanks", []):
        ex5a.append({"input": tb["tb_id"], "output": _s(tb),
                      "predict_method": _s({"llf_delta": tb.get("llf_delta"),
                          "mean_llf_standard": tb.get("mean_llf_standard")}),
                      "metadata_phase": "5a", "metadata_word_order": "SVO"})
    cmp5a = p5a.get("sov_vs_svo_llf_delta", {})
    ex5a.append({"input": "sov_vs_svo_comparison",
                 "output": _s({
                     "sov_vs_svo_llf_delta": cmp5a,
                     "head_final_prop_vs_llf_delta_correlation":
                         p5a.get("head_final_prop_vs_llf_delta_correlation"),
                     "head_final_proportions_by_wo":
                         p5a.get("head_final_proportions_by_wo"),
                     "n_sov": p5a.get("n_sov", 0),
                     "n_svo": p5a.get("n_svo", 0),
                 }),
                 "predict_method": _s({"sov_larger": cmp5a.get("sov_larger"),
                     "cohens_d": cmp5a.get("cohens_d"),
                     "p": cmp5a.get("p")}),
                 "metadata_phase": "5a",
                 "metadata_analysis": "aggregate_comparison"})
    datasets.append({"dataset": "phase5a_head_final_effects",
                     "examples": ex5a})

    # --- Phase 5b ---
    ex5b: list[dict] = []
    for pt in p5b.get("scatter_data", []):
        ex5b.append({"input": pt["tb_id"], "output": _s(pt),
                      "predict_method": _s({
                          "mean_llf_standard": pt.get("mean_llf_standard"),
                          "case_richness_count": pt.get("case_richness_count")}),
                      "metadata_phase": "5b",
                      "metadata_analysis": "case_richness_scatter"})
    corr_only = {k: v for k, v in p5b.items() if k != "scatter_data"}
    cr_res = p5b.get("case_richness_count", {})
    ex5b.append({"input": "correlation_analysis", "output": _s(corr_only),
                 "predict_method": _s({"pearson_r": cr_res.get("pearson_r"),
                     "pearson_p": cr_res.get("pearson_p")}),
                 "metadata_phase": "5b",
                 "metadata_analysis": "case_richness_correlation"})
    datasets.append({"dataset": "phase5b_case_richness", "examples": ex5b})

    # --- Phase 5c ---
    ex5c: list[dict] = []
    for tb in p5c.get("per_treebank", []):
        ex5c.append({"input": tb["tb_id"], "output": _s(tb),
                      "predict_method": _s({"direction": tb.get("direction"),
                          "cohens_d": tb.get("cohens_d"),
                          "p": tb.get("p")}),
                      "metadata_phase": "5c",
                      "metadata_analysis": "dom_within_language"})
    agg5c = {k: v for k, v in p5c.items() if k != "per_treebank"}
    ex5c.append({"input": "dom_aggregate", "output": _s(agg5c),
                 "predict_method": _s({
                     "proportion_positive": p5c.get("aggregate_proportion_positive"),
                     "mixed_effects_coef": p5c.get("mixed_effects_coefficient"),
                     "mixed_effects_p": p5c.get("mixed_effects_p")}),
                 "metadata_phase": "5c",
                 "metadata_analysis": "dom_aggregate"})
    datasets.append({"dataset": "phase5c_dom_analysis", "examples": ex5c})

    # --- Pilot ---
    datasets.append({"dataset": "pilot_validation",
                     "examples": [{"input": "pilot_validation_results",
                                   "output": _s(pilot),
                                   "predict_method": _s({
                                       "all_micro_passed": pilot.get(
                                           "micro_tests", {}).get("all_passed"),
                                       "enc_reduces_sov": pilot.get(
                                           "encounter_only_reduces_imb_in_sov")}),
                                   "metadata_phase": "pilot",
                                   "metadata_analysis": "validation"}]})

    # --- Comprehensive summary ---
    summary = {
        "phase4_modality": p4,
        "phase5a_head_final": {
            k: v for k, v in p5a.items()
            if k not in ("sov_treebanks", "svo_treebanks")
        },
        "phase5b_case_richness": {k: v for k, v in p5b.items()
                                   if k != "scatter_data"},
        "phase5c_dom": {k: v for k, v in p5c.items()
                        if k != "per_treebank"},
        "pilot_validation": pilot,
    }
    datasets.append({"dataset": "comprehensive_summary",
                     "examples": [{"input": "all_phases_summary",
                                   "output": _s(summary),
                                   "predict_method": _s({
                                       "p4_proportion_spoken_higher":
                                           p4.get("aggregate", {}).get(
                                               "proportion_spoken_higher"),
                                       "p5a_sov_larger":
                                           p5a.get("sov_vs_svo_llf_delta",
                                                   {}).get("sov_larger"),
                                       "p5c_proportion_positive":
                                           p5c.get(
                                               "aggregate_proportion_positive")
                                   }),
                                   "metadata_phase": "all",
                                   "metadata_analysis": "comprehensive"}]})

    return {
        "metadata": {
            "description": ("Load-smoothing hypothesis analysis: "
                            "Phases 4-5 (modality, head-final, "
                            "case-richness, DOM)"),
            "phases": ["4_modality", "5a_head_final",
                       "5b_case_richness", "5c_dom"],
        },
        "datasets": datasets,
    }


# =============================================================================
# MAIN
# =============================================================================

@logger.catch
def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("LOAD-SMOOTHING HYPOTHESIS  —  Phases 4-5")
    logger.info("=" * 60)

    # --- Phase 0 ---
    metadata = load_metadata()
    targets  = identify_targets(metadata)
    dom_set  = set(targets["dom_treebanks"])
    sdata    = load_sentences(targets["all_needed"], dom_set)

    # --- Phase 1: pilot ---
    pilot = pilot_validation(sdata, metadata)
    if not pilot.get("micro_tests", {}).get("all_passed", False):
        logger.error("Micro validation FAILED — aborting")
        sys.exit(1)

    # --- Phase 4 ---
    p4 = phase4_modality(sdata, metadata, targets)

    # --- Phase 5a ---
    p5a = phase5a_head_final(sdata, metadata, targets)

    # --- Phase 5b ---
    p5b = phase5b_case_richness(p5a, metadata)

    # --- Phase 5c ---
    p5c = phase5c_dom(sdata, metadata, targets)

    # --- assemble ---
    output = assemble_output(p4, p5a, p5b, p5c, pilot)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                   default=_json_default))
    sz = out_path.stat().st_size
    logger.info(f"Wrote {out_path}  ({sz/1e6:.2f} MB)")
    logger.info(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
