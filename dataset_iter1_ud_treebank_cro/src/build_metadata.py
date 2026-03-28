#!/usr/bin/env python3
"""Build cross-linguistic typological and modality metadata for all UD treebanks.

Compiles a single joinable metadata table covering all UD treebanks from
commul/universal_dependencies with WALS word-order, Glottolog families,
morphological case richness, head-direction entropy, modality labels, and more.
"""

import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware Detection ───────────────────────────────────────────────────────
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

def _container_ram_gb() -> Optional[float]:
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

# Set RAM limit to 80% of container limit
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.80 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
MAX_SENTENCES_PER_TREEBANK = 5000  # sufficient for distributional stats

WALS_VALUES_URL = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/values.csv"
WALS_LANGS_URL = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/languages.csv"
GLOTTOLOG_LANGS_URL = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"

# WALS 81A value mapping
WALS_81A_MAP = {
    "1": "SOV", "2": "SVO", "3": "VSO", "4": "VOS",
    "5": "OVS", "6": "OSV", "7": "No dominant order",
}

# ── Spoken Treebanks (from Dobrovoljc 2022 + UD documentation) ───────────────
SPOKEN_TREEBANKS = {
    # French spoken (Rhapsodie) ↔ French GSD
    "fr_rhapsodie": "fr_gsd",
    # French spoken (Paris Stories) ↔ French GSD
    "fr_parisstories": "fr_gsd",
    # Slovenian spoken ↔ Slovenian SSJ
    "sl_sst": "sl_ssj",
    # English ATIS (flight queries) ↔ English EWT
    "en_atis": "en_ewt",
    # English ESL spoken ↔ English EWT (renamed from en_esl in commul v2.0)
    "en_eslspok": "en_ewt",
    # English CHILDES (child speech) ↔ English EWT
    "en_childes": "en_ewt",
    # Italian VALICO (learner) ↔ Italian ISDT
    "it_valico": "it_isdt",
    # Turkish ATIS (flight queries) ↔ Turkish BOUN
    "tr_atis": "tr_boun",
    # Naija NSC — no written counterpart
    "pcm_nsc": None,
    # Cantonese HK — no written counterpart
    "yue_hk": None,
    # Bambara CRB — no written counterpart
    "bm_crb": None,
    # Komi-Zyrian — no written counterpart
    "kpv_lattice": None,
    # Telugu-English code-switch — no counterpart
    "qte_tect": None,
    # Turkish-German code-switch — no counterpart
    "qtd_sagt": None,
}

# Mixed treebanks (contain some spoken portions)
MIXED_TREEBANKS = {
    "en_gum", "da_ddt", "en_lines", "el_gdt", "lv_lvtb",
    "fa_seraji", "pl_lfg", "gd_arcosg", "sms_giellagas",
    "sv_lines", "ajp_madar",
}

# DOM languages (from hypothesis plan)
DOM_TREEBANK_PREFIXES = {
    "tr", "hi", "ko", "fa", "ur", "he", "es", "ro",
}

# ── ISO 639-1 → 639-3 Mapping ───────────────────────────────────────────────
# Comprehensive mapping covering all UD language prefixes
ISO_639_1_TO_3 = {
    "ab": "abk", "af": "afr", "ajp": "ajp", "akk": "akk", "am": "amh",
    "an": "arg", "apu": "apu", "aqz": "aqz", "ar": "ara", "be": "bel",
    "bg": "bul", "bho": "bho", "bm": "bam", "bn": "ben", "br": "bre",
    "bxr": "bxr", "ca": "cat", "ceb": "ceb", "ckb": "ckb", "ckt": "ckt",
    "cop": "cop", "cs": "ces", "cu": "chu", "cy": "cym", "da": "dan",
    "de": "deu", "el": "ell", "en": "eng", "es": "spa", "et": "est",
    "eu": "eus", "fa": "fas", "fi": "fin", "fo": "fao", "fr": "fra",
    "fro": "fro", "ga": "gle", "gd": "gla", "gl": "glg", "got": "got",
    "grc": "grc", "gsw": "gsw", "gun": "gun", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "hr": "hrv", "hsb": "hsb", "hu": "hun",
    "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "jv": "jav", "ka": "kat", "kfm": "kfm", "kk": "kaz", "kmr": "kmr",
    "ko": "kor", "koi": "koi", "kpv": "kpv", "krl": "krl", "la": "lat",
    "lt": "lit", "lv": "lav", "lzh": "lzh", "mdf": "mdf", "mk": "mkd",
    "ml": "mal", "mn": "mon", "mr": "mar", "mt": "mlt", "my": "mya",
    "myv": "myv", "nb": "nob", "nds": "nds", "nl": "nld", "nn": "nno",
    "no": "nor", "nyq": "nyq", "olo": "olo", "orv": "orv", "pcm": "pcm",
    "pl": "pol", "pt": "por", "qtd": "qtd", "qte": "qte", "ro": "ron",
    "ru": "rus", "sa": "san", "sk": "slk", "sl": "slv", "sme": "sme",
    "sms": "sms", "soj": "soj", "sq": "sqi", "sr": "srp", "sv": "swe",
    "swl": "swl", "ta": "tam", "te": "tel", "th": "tha", "tl": "tgl",
    "tr": "tur", "ug": "uig", "uk": "ukr", "ur": "urd", "uz": "uzb",
    "vi": "vie", "wbp": "wbp", "wo": "wol", "yo": "yor", "yue": "yue",
    "zh": "zho", "abq": "abq", "aii": "aii", "arr": "arr", "bej": "bej",
    "gub": "gub", "hit": "hit", "mpu": "mpu", "nhi": "nhi",
    "otk": "otk", "qfn": "qfn", "quc": "quc", "say": "say", "shp": "shp",
    "tpn": "tpn", "xav": "xav", "xnr": "xnr",
    # Missing 2-letter codes
    "az": "azj", "ms": "zsm", "nb": "nob", "nn": "nno",
    "bar": "bar", "gn": "grn", "rm": "roh", "se": "sme",
}

# Macrolanguage → individual language fallbacks for Glottolog/WALS matching
# ISO 639-3 macrolanguages don't appear in Glottolog directly
MACROLANG_FALLBACKS = {
    "ara": "arb",  # Arabic → Standard Arabic
    "zho": "cmn",  # Chinese → Mandarin
    "fas": "pes",  # Persian → Western Farsi (Iranian Persian)
    "msa": "zsm",  # Malay → Standard Malay
    "nor": "nob",  # Norwegian → Bokmål
    "sqi": "als",  # Albanian → Tosk Albanian
    "hye": "hye",  # Armenian (same)
    "eus": "eus",  # Basque (same)
    "grn": "gug",  # Guarani → Paraguayan Guarani
    "azj": "azj",  # North Azerbaijani (same)
}


# ── Step 1: Enumerate Treebank Configs ───────────────────────────────────────
def enumerate_treebank_configs() -> pd.DataFrame:
    """Get all treebank config names from commul/universal_dependencies."""
    from datasets import get_dataset_config_names
    logger.info("Enumerating treebank configs from commul/universal_dependencies...")
    try:
        configs = get_dataset_config_names("commul/universal_dependencies")
    except Exception:
        logger.warning("commul/universal_dependencies failed, trying universal-dependencies/universal_dependencies")
        configs = get_dataset_config_names("universal-dependencies/universal_dependencies")

    rows = []
    for cfg in configs:
        parts = cfg.split("_", 1)
        lang_code = parts[0]
        treebank_suffix = parts[1] if len(parts) > 1 else ""
        iso3 = ISO_639_1_TO_3.get(lang_code, lang_code if len(lang_code) == 3 else None)
        rows.append({
            "treebank_id": cfg,
            "lang_code": lang_code,
            "treebank_suffix": treebank_suffix,
            "iso_639_3": iso3,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Found {len(df)} treebank configs")
    return df


# ── Step 2-3: Download WALS Data ────────────────────────────────────────────
def download_wals_data() -> pd.DataFrame:
    """Download WALS Feature 81A values and language table, join them."""
    logger.info("Downloading WALS values (81A) and languages...")

    # Download values
    resp_vals = requests.get(WALS_VALUES_URL, timeout=60)
    resp_vals.raise_for_status()
    df_vals = pd.read_csv(StringIO(resp_vals.text))
    df_81a = df_vals[df_vals["Parameter_ID"] == "81A"].copy()
    df_81a["wals_word_order"] = df_81a["Value"].astype(str).map(WALS_81A_MAP)
    logger.info(f"WALS 81A: {len(df_81a)} language entries")

    # Download languages
    resp_langs = requests.get(WALS_LANGS_URL, timeout=60)
    resp_langs.raise_for_status()
    df_langs = pd.read_csv(StringIO(resp_langs.text))
    logger.info(f"WALS languages: {len(df_langs)} entries")

    # Join values to languages on Language_ID == ID
    merged = df_81a.merge(
        df_langs[["ID", "Name", "ISO639P3code", "Glottocode", "Family", "Genus"]],
        left_on="Language_ID", right_on="ID", how="left", suffixes=("", "_lang")
    )

    # Build ISO-639-3 → WALS info mapping (handle many-to-one by taking first)
    wals_by_iso = merged.dropna(subset=["ISO639P3code"]).groupby("ISO639P3code").first().reset_index()
    result = wals_by_iso[["ISO639P3code", "Language_ID", "Name", "wals_word_order", "Family", "Genus"]].copy()
    result.columns = ["iso_639_3", "wals_code", "wals_language_name", "wals_word_order", "wals_family", "wals_genus"]
    logger.info(f"WALS mapped to {len(result)} unique ISO-639-3 codes")
    return result


# ── Step 4: Download Glottolog Data ─────────────────────────────────────────
def download_glottolog_data() -> pd.DataFrame:
    """Download Glottolog CLDF languages.csv and extract family info."""
    logger.info("Downloading Glottolog languages...")
    resp = requests.get(GLOTTOLOG_LANGS_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), low_memory=False)
    logger.info(f"Glottolog: {len(df)} languoid entries")

    # Filter to language-level entries with ISO codes
    df_lang = df[df["ISO639P3code"].notna()].copy()

    # Build family name lookup: Family_ID -> Name
    family_lookup = df.set_index("ID")["Name"].to_dict()

    df_lang["glottolog_family_name"] = df_lang["Family_ID"].map(family_lookup)
    df_lang["is_isolate"] = df_lang["Family_ID"].isna()

    result = df_lang[["ISO639P3code", "ID", "Family_ID", "glottolog_family_name", "Macroarea", "Name"]].copy()
    result.columns = ["iso_639_3", "glottocode", "glottolog_family_id", "glottolog_family_name", "macroarea", "glottolog_language_name"]

    # Deduplicate: keep first per ISO code
    result = result.drop_duplicates(subset=["iso_639_3"], keep="first")
    result["is_isolate"] = result["glottolog_family_id"].isna()
    logger.info(f"Glottolog mapped to {len(result)} unique ISO-639-3 codes")
    return result


# ── Step 5: Modality Labels ─────────────────────────────────────────────────
def get_modality_label(treebank_id: str) -> dict:
    """Return modality info for a treebank."""
    if treebank_id in SPOKEN_TREEBANKS:
        return {
            "modality": "spoken",
            "spoken_written_pair_id": SPOKEN_TREEBANKS[treebank_id],
        }
    elif treebank_id in MIXED_TREEBANKS:
        return {
            "modality": "mixed",
            "spoken_written_pair_id": None,
        }
    else:
        return {
            "modality": "written",
            "spoken_written_pair_id": None,
        }


# ── Step 6-8: Process Single Treebank (case, head-direction, stats) ─────────
def process_single_treebank(treebank_id: str, max_sentences: int = MAX_SENTENCES_PER_TREEBANK) -> dict:
    """Process a single treebank: compute case richness, head-direction entropy, basic stats.

    This function runs in a subprocess via ProcessPoolExecutor.
    """
    from datasets import load_dataset

    result = {
        "treebank_id": treebank_id,
        "case_values_distinct": [],
        "case_richness_count": 0,
        "case_token_proportion": 0.0,
        "case_token_count": 0,
        "head_direction_entropy_binary": None,
        "head_direction_entropy_deprel": None,
        "head_direction_left_proportion": None,
        "num_sentences": 0,
        "num_tokens": 0,
        "mean_sentence_length": 0.0,
        "deprel_inventory": [],
        "has_reparandum": False,
        "has_discourse": False,
        "error": None,
    }

    try:
        # Try loading train split first, fall back to any available split
        ds = None
        for split in ["train", "test", "validation", "dev"]:
            try:
                ds = load_dataset(
                    "commul/universal_dependencies",
                    treebank_id,
                    split=split,
                    streaming=True,
                )
                break
            except (ValueError, KeyError):
                continue

        if ds is None:
            result["error"] = "No valid split found"
            return result

        # Accumulators
        case_values = set()
        case_token_count = 0
        total_tokens = 0
        num_sentences = 0
        left_headed = 0
        right_headed = 0
        deprel_set = set()
        deprel_left = {}  # deprel -> count of left-headed
        deprel_right = {}  # deprel -> count of right-headed
        sentence_lengths = []

        for sent_idx, sentence in enumerate(ds):
            if sent_idx >= max_sentences:
                break

            num_sentences += 1
            tokens = sentence.get("tokens", [])
            feats_list = sentence.get("feats", [])
            head_list = sentence.get("head", [])
            deprel_list = sentence.get("deprel", [])

            n_tokens = len(tokens)
            total_tokens += n_tokens
            sentence_lengths.append(n_tokens)

            # Process feats for case values
            for feat_str in feats_list:
                if feat_str and feat_str != "_" and feat_str is not None:
                    for feat_pair in str(feat_str).split("|"):
                        if feat_pair.startswith("Case="):
                            case_val = feat_pair.split("=", 1)[1]
                            # Handle multiple case values (e.g., "Case=Acc,Nom")
                            for cv in case_val.split(","):
                                case_values.add(cv)
                            case_token_count += 1
                            break  # only count token once

            # Process head for direction + deprels
            for tok_idx, (head_str, deprel) in enumerate(zip(head_list, deprel_list)):
                if deprel:
                    deprel_set.add(str(deprel))

                try:
                    head_int = int(head_str) if head_str is not None else 0
                except (ValueError, TypeError):
                    continue

                if head_int == 0:  # root
                    continue

                # Token positions are 1-indexed; tok_idx is 0-indexed in the list
                token_pos = tok_idx + 1
                dep_str = str(deprel) if deprel else "unknown"

                if head_int < token_pos:
                    left_headed += 1
                    deprel_left[dep_str] = deprel_left.get(dep_str, 0) + 1
                elif head_int > token_pos:
                    right_headed += 1
                    deprel_right[dep_str] = deprel_right.get(dep_str, 0) + 1

        # Compute results
        result["num_sentences"] = num_sentences
        result["num_tokens"] = total_tokens
        result["mean_sentence_length"] = round(np.mean(sentence_lengths), 2) if sentence_lengths else 0.0
        result["case_values_distinct"] = sorted(case_values)
        result["case_richness_count"] = len(case_values)
        result["case_token_count"] = case_token_count
        result["case_token_proportion"] = round(case_token_count / total_tokens, 4) if total_tokens > 0 else 0.0
        result["deprel_inventory"] = sorted(deprel_set)
        result["has_reparandum"] = "reparandum" in deprel_set
        result["has_discourse"] = "discourse" in deprel_set

        # Head-direction entropy (binary)
        total_deps = left_headed + right_headed
        if total_deps > 0:
            p = left_headed / total_deps
            result["head_direction_left_proportion"] = round(p, 4)
            if 0 < p < 1:
                result["head_direction_entropy_binary"] = round(
                    -p * math.log2(p) - (1 - p) * math.log2(1 - p), 4
                )
            else:
                result["head_direction_entropy_binary"] = 0.0

            # Head-direction entropy (per-deprel weighted)
            all_deprels = set(list(deprel_left.keys()) + list(deprel_right.keys()))
            weighted_entropy = 0.0
            for dr in all_deprels:
                l_count = deprel_left.get(dr, 0)
                r_count = deprel_right.get(dr, 0)
                dr_total = l_count + r_count
                if dr_total == 0:
                    continue
                weight = dr_total / total_deps
                dr_p = l_count / dr_total
                if 0 < dr_p < 1:
                    dr_entropy = -dr_p * math.log2(dr_p) - (1 - dr_p) * math.log2(1 - dr_p)
                else:
                    dr_entropy = 0.0
                weighted_entropy += weight * dr_entropy
            result["head_direction_entropy_deprel"] = round(weighted_entropy, 4)

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


# ── Main Pipeline ────────────────────────────────────────────────────────────
@logger.catch
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build UD treebank metadata table")
    parser.add_argument("--max-treebanks", type=int, default=None,
                        help="Limit number of treebanks to process (for gradual scaling)")
    parser.add_argument("--workers", type=int, default=max(1, NUM_CPUS),
                        help="Number of parallel workers")
    parser.add_argument("--max-sentences", type=int, default=MAX_SENTENCES_PER_TREEBANK,
                        help="Max sentences per treebank for stats")
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Enumerate configs
    treebank_df = enumerate_treebank_configs()
    all_configs = list(treebank_df["treebank_id"])
    valid_configs_set = set(all_configs)

    # Apply limit for scaling
    if args.max_treebanks is not None:
        configs_to_process = all_configs[:args.max_treebanks]
        logger.info(f"Processing {len(configs_to_process)}/{len(all_configs)} treebanks (scaling limit)")
    else:
        configs_to_process = all_configs
        logger.info(f"Processing all {len(configs_to_process)} treebanks")

    # Steps 2-4: Download external data (sequential, fast HTTP calls)
    wals_df = download_wals_data()
    glottolog_df = download_glottolog_data()

    # Step 5: Verify spoken treebanks exist
    verified_spoken = {}
    for tb, pair in SPOKEN_TREEBANKS.items():
        if tb in valid_configs_set:
            if pair is None or pair in valid_configs_set:
                verified_spoken[tb] = pair
            else:
                logger.warning(f"Spoken pair {pair} not found for {tb}, setting to null")
                verified_spoken[tb] = None
        else:
            logger.warning(f"Spoken treebank {tb} not found in configs")
    logger.info(f"Verified {len(verified_spoken)}/{len(SPOKEN_TREEBANKS)} spoken treebanks")

    # Steps 6-8: Process treebanks in parallel
    logger.info(f"Processing {len(configs_to_process)} treebanks with {args.workers} workers, max {args.max_sentences} sentences each")
    treebank_stats = {}
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_treebank, cfg, args.max_sentences): cfg
            for cfg in configs_to_process
        }

        for i, future in enumerate(as_completed(futures)):
            cfg = futures[future]
            try:
                result = future.result(timeout=300)  # 5 min timeout per treebank
                treebank_stats[cfg] = result
                if result.get("error"):
                    failed.append(cfg)
                    logger.warning(f"[{i+1}/{len(configs_to_process)}] {cfg}: ERROR - {result['error'][:100]}")
                else:
                    if (i + 1) % 20 == 0 or (i + 1) == len(configs_to_process):
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed * 60
                        logger.info(f"[{i+1}/{len(configs_to_process)}] {cfg} done | {elapsed:.0f}s elapsed | {rate:.1f} tb/min")
            except Exception as e:
                failed.append(cfg)
                treebank_stats[cfg] = {"treebank_id": cfg, "error": str(e)[:200]}
                logger.error(f"[{i+1}/{len(configs_to_process)}] {cfg}: EXCEPTION - {str(e)[:100]}")

    logger.info(f"Processed {len(treebank_stats)} treebanks, {len(failed)} failed")

    # Step 9: Join all sources
    logger.info("Joining all data sources...")

    output_rows = []
    wals_match = 0
    glottolog_match = 0

    for _, row in treebank_df.iterrows():
        tb_id = row["treebank_id"]

        # Skip if not in processed set (scaling limit)
        if tb_id not in treebank_stats:
            continue

        stats = treebank_stats[tb_id]
        iso3 = row.get("iso_639_3")

        # WALS join (try primary ISO, then macrolang fallback)
        wals_row = None
        if iso3:
            candidates = [iso3]
            if iso3 in MACROLANG_FALLBACKS:
                candidates.append(MACROLANG_FALLBACKS[iso3])
            for candidate_iso in candidates:
                matches = wals_df[wals_df["iso_639_3"] == candidate_iso]
                if not matches.empty:
                    wals_row = matches.iloc[0]
                    break
        if wals_row is not None:
            wals_match += 1

        # Glottolog join (try primary ISO, then macrolang fallback)
        glot_row = None
        if iso3:
            candidates = [iso3]
            if iso3 in MACROLANG_FALLBACKS:
                candidates.append(MACROLANG_FALLBACKS[iso3])
            for candidate_iso in candidates:
                matches = glottolog_df[glottolog_df["iso_639_3"] == candidate_iso]
                if not matches.empty:
                    glot_row = matches.iloc[0]
                    break
        if glot_row is not None:
            glottolog_match += 1

        # Modality
        modality_info = get_modality_label(tb_id)
        # Override with verified spoken data
        if tb_id in verified_spoken:
            modality_info["spoken_written_pair_id"] = verified_spoken[tb_id]

        # DOM flag
        lang_prefix = row["lang_code"]
        has_dom = lang_prefix in DOM_TREEBANK_PREFIXES

        # Build output record
        record = {
            "treebank_id": tb_id,
            "lang_code": row["lang_code"],
            "iso_639_3": iso3,
            "language_name": (
                glot_row["glottolog_language_name"] if glot_row is not None and pd.notna(glot_row.get("glottolog_language_name"))
                else (wals_row["wals_language_name"] if wals_row is not None and pd.notna(wals_row.get("wals_language_name"))
                      else None)
            ),
            # WALS fields
            "wals_word_order": _safe_str(wals_row["wals_word_order"]) if wals_row is not None else None,
            "wals_code": _safe_str(wals_row["wals_code"]) if wals_row is not None else None,
            "wals_family": _safe_str(wals_row["wals_family"]) if wals_row is not None else None,
            "wals_genus": _safe_str(wals_row["wals_genus"]) if wals_row is not None else None,
            # Glottolog fields
            "glottocode": _safe_str(glot_row["glottocode"]) if glot_row is not None else None,
            "glottolog_family_id": _safe_str(glot_row["glottolog_family_id"]) if glot_row is not None else None,
            "glottolog_family_name": _safe_str(glot_row["glottolog_family_name"]) if glot_row is not None else None,
            "macroarea": _safe_str(glot_row["macroarea"]) if glot_row is not None else None,
            "is_isolate": bool(glot_row["is_isolate"]) if glot_row is not None and pd.notna(glot_row.get("is_isolate")) else None,
            # Modality
            "modality": modality_info["modality"],
            "spoken_written_pair_id": modality_info["spoken_written_pair_id"],
            # Case richness
            "case_values_distinct": stats.get("case_values_distinct", []),
            "case_richness_count": stats.get("case_richness_count", 0),
            "case_token_proportion": stats.get("case_token_proportion", 0.0),
            # DOM
            "has_differential_object_marking": has_dom,
            # Disfluency
            "disfluency_deprels_present": _build_disfluency_list(stats),
            # Head direction
            "head_direction_entropy_binary": stats.get("head_direction_entropy_binary"),
            "head_direction_entropy_deprel": stats.get("head_direction_entropy_deprel"),
            "head_direction_left_proportion": stats.get("head_direction_left_proportion"),
            # Basic stats
            "num_sentences": stats.get("num_sentences", 0),
            "num_tokens": stats.get("num_tokens", 0),
            "mean_sentence_length": stats.get("mean_sentence_length", 0.0),
            "deprel_inventory": stats.get("deprel_inventory", []),
        }
        output_rows.append(record)

    # Log join quality
    n = len(output_rows)
    logger.info(f"Join quality: WALS matched {wals_match}/{n} ({100*wals_match/n:.1f}%), "
                f"Glottolog matched {glottolog_match}/{n} ({100*glottolog_match/n:.1f}%)")

    no_match = [r["treebank_id"] for r in output_rows
                if r["wals_word_order"] is None and r["glottocode"] is None]
    if no_match:
        logger.warning(f"No external metadata: {no_match[:20]}{'...' if len(no_match) > 20 else ''}")

    # Step 10: Format as required output schema
    final_output = []
    for rec in output_rows:
        final_output.append({
            "input": rec["treebank_id"],
            "output": rec,
            "metadata_fold": "metadata",
        })

    # Validate
    _validate_output(final_output)

    # Save
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(final_output, indent=2, ensure_ascii=False))

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(final_output)} treebank metadata records to {out_path}")
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return final_output


def _safe_str(val) -> Optional[str]:
    """Convert value to string, handling NaN/None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val)


def _build_disfluency_list(stats: dict) -> list:
    """Build list of disfluency-related deprels present."""
    result = []
    if stats.get("has_reparandum"):
        result.append("reparandum")
    if stats.get("has_discourse"):
        result.append("discourse")
    return result


def _validate_output(data: list) -> None:
    """Run validation checks on the output."""
    logger.info("Running validation checks...")
    valid_orders = {"SOV", "SVO", "VSO", "VOS", "OVS", "OSV", "No dominant order", None}
    valid_modalities = {"spoken", "written", "mixed"}
    errors = 0

    for item in data:
        rec = item["output"]
        if not rec.get("treebank_id"):
            logger.error(f"Missing treebank_id")
            errors += 1
        if not rec.get("lang_code"):
            logger.error(f"Missing lang_code for {rec.get('treebank_id')}")
            errors += 1
        if rec.get("wals_word_order") not in valid_orders:
            logger.error(f"Invalid word_order '{rec.get('wals_word_order')}' for {rec['treebank_id']}")
            errors += 1
        if rec.get("case_richness_count", 0) != len(rec.get("case_values_distinct", [])):
            logger.error(f"Case count mismatch for {rec['treebank_id']}")
            errors += 1
        hde = rec.get("head_direction_entropy_binary")
        if hde is not None and not (0 <= hde <= 1):
            logger.error(f"Invalid entropy {hde} for {rec['treebank_id']}")
            errors += 1
        if rec.get("modality") not in valid_modalities:
            logger.error(f"Invalid modality '{rec.get('modality')}' for {rec['treebank_id']}")
            errors += 1

    if errors == 0:
        logger.info("Validation PASSED (all checks OK)")
    else:
        logger.warning(f"Validation found {errors} issues")


if __name__ == "__main__":
    main()
