#!/usr/bin/env python3
"""Download and process Universal Dependencies treebanks for IMB/LLF analysis.

Extracts raw dependency tree structures and computes per-dependency and
per-sentence linguistic features from commul/universal_dependencies on HuggingFace.

Memory-safe design:
  - Each treebank is processed independently and saved to a temp file
  - Final output is assembled by streaming temp files to disk
  - Per-treebank cap prevents any single treebank from dominating memory
"""

import json
import math
import sys
import gc
import os
import resource
import time
from pathlib import Path
from collections import defaultdict

from loguru import logger
import numpy as np
from datasets import load_dataset, get_dataset_config_names

# === Configuration ===
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
TEMP_DIR = WORKSPACE / "temp" / "treebanks"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")


# === Hardware Detection ===
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


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


TOTAL_RAM_GB = _container_ram_gb() or 29.0
NUM_CPUS = _detect_cpus()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# === Constants ===
DATASET_ID = "commul/universal_dependencies"
UPOS_NAMES = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART",
    "DET", "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
]
FUNCTIONAL_DEPRELS = {"aux", "case", "cc", "clf", "cop", "det", "mark", "punct"}
MIN_EFFECTIVE_LENGTH = 8
MAX_EXAMPLES_PER_TREEBANK = 10000  # Cap to keep output manageable


# === Feature Computation (unchanged, well-tested) ===
def upos_int_to_str(upos_int: int) -> str:
    """Convert UPOS integer index to string label."""
    if 0 <= upos_int < len(UPOS_NAMES):
        return UPOS_NAMES[upos_int]
    return "_"


def validate_tree(heads: list[int]) -> bool:
    """Validate tree well-formedness: single root, valid indices, no cycles."""
    n = len(heads)
    if n == 0:
        return False
    roots = [i for i, h in enumerate(heads) if h == 0]
    if len(roots) != 1:
        return False
    for h in heads:
        if h < 0 or h > n:
            return False
    for i in range(n):
        if heads[i] == 0:
            continue
        visited = set()
        cur = i
        while cur not in visited:
            visited.add(cur)
            h = heads[cur]
            if h == 0:
                break
            cur = h - 1
            if cur < 0 or cur >= n:
                return False
        else:
            return False
    return True


def compute_tree_depth_and_arity(heads: list[int]) -> tuple[int, int, float]:
    """Compute tree depth, max arity, and mean arity via BFS from root."""
    n = len(heads)
    children = defaultdict(list)
    root_idx = -1
    for i in range(n):
        if heads[i] == 0:
            root_idx = i
        else:
            children[heads[i] - 1].append(i)
    if root_idx == -1:
        return -1, 0, 0.0
    max_depth = 0
    queue = [(root_idx, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            max_depth = depth
        for child in children[node]:
            queue.append((child, depth + 1))
    non_leaf_arities = [len(ch) for ch in children.values() if len(ch) > 0]
    max_arity = max(non_leaf_arities) if non_leaf_arities else 0
    mean_arity = sum(non_leaf_arities) / len(non_leaf_arities) if non_leaf_arities else 0.0
    return max_depth, max_arity, mean_arity


def get_descendants(heads: list[int]) -> dict[int, set[int]]:
    """Return dict: node (1-indexed) -> set of all descendants (1-indexed)."""
    n = len(heads)
    children = defaultdict(list)
    for i, h in enumerate(heads):
        if h != 0:
            children[h].append(i + 1)
    desc: dict[int, set[int]] = {}
    for node in range(1, n + 1):
        desc[node] = set()
        stack = list(children[node])
        while stack:
            c = stack.pop()
            desc[node].add(c)
            stack.extend(children[c])
    return desc


def is_projective(dep_pos: int, head_pos: int, descendants: dict[int, set[int]]) -> bool:
    """Check if a single dependency arc is projective."""
    a, b = min(dep_pos, head_pos), max(dep_pos, head_pos)
    desc_a = descendants.get(a, set())
    desc_b = descendants.get(b, set())
    for k in range(a + 1, b):
        if k not in desc_a and k not in desc_b:
            return False
    return True


def compute_sentence_features(
    tokens: list[str],
    heads: list[int],
    deprels: list[str],
    upos_strs: list[str],
    feats: list[str | None],
    text: str,
    sent_id: str,
    split_name: str,
    config_name: str,
    lang_code: str,
) -> dict | None:
    """Compute all features for one sentence. Returns formatted example or None."""
    n = len(tokens)
    effective_length = sum(1 for pos in upos_strs if pos != "PUNCT")
    if effective_length < MIN_EFFECTIVE_LENGTH:
        return None
    if not validate_tree(heads):
        return None
    tree_depth, max_arity, mean_arity = compute_tree_depth_and_arity(heads)
    if tree_depth < 0:
        return None

    descendants = get_descendants(heads)
    head_set = {h for h in heads if h != 0}

    dd_list: list[int] = []
    left_count = 0
    right_count = 0
    proj_count = 0
    ic_list: list[int] = []
    func_count = 0
    total_deps = 0

    for i in range(n):
        pos = i + 1
        h = heads[i]
        if h == 0:
            continue
        total_deps += 1
        dd_list.append(abs(pos - h))
        if h < pos:
            left_count += 1
        else:
            right_count += 1
        if is_projective(pos, h, descendants):
            proj_count += 1
        a, b = min(pos, h), max(pos, h)
        ic_list.append(sum(1 for k in range(a + 1, b) if k in head_set))
        if deprels[i].split(":")[0] in FUNCTIONAL_DEPRELS:
            func_count += 1

    if total_deps == 0:
        return None

    dd_arr = np.array(dd_list, dtype=np.float64)
    mean_dd = round(float(np.mean(dd_arr)), 4)
    dd_variance = round(float(np.var(dd_arr)), 4)
    if len(dd_list) >= 3:
        m, s = np.mean(dd_arr), np.std(dd_arr)
        dd_skewness = round(float(np.mean(((dd_arr - m) / s) ** 3)), 4) if s > 0 else 0.0
    else:
        dd_skewness = 0.0

    ic_arr = np.array(ic_list, dtype=np.float64)
    functional_ratio = round(func_count / total_deps, 4)
    total_dir = left_count + right_count
    if total_dir > 0 and left_count > 0 and right_count > 0:
        p_l, p_r = left_count / total_dir, right_count / total_dir
        hde = round(-(p_l * math.log2(p_l) + p_r * math.log2(p_r)), 4)
    else:
        hde = 0.0

    input_data = {
        "text": text, "tokens": tokens, "heads": heads,
        "deprels": deprels, "upos": upos_strs, "feats": feats,
    }
    output_features = {
        "sentence_length": n, "effective_length": effective_length,
        "tree_depth": tree_depth, "max_arity": max_arity,
        "mean_arity": round(mean_arity, 4), "mean_dd": mean_dd,
        "dd_variance": dd_variance, "dd_skewness": dd_skewness,
        "dd_list": dd_list, "projectivity_proportion": round(proj_count / total_deps, 4),
        "n_nonprojective": total_deps - proj_count,
        "mean_ic": round(float(np.mean(ic_arr)), 4),
        "ic_variance": round(float(np.var(ic_arr)), 4),
        "max_ic": int(np.max(ic_arr)),
        "functional_ratio": functional_ratio,
        "content_ratio": round(1.0 - functional_ratio, 4),
        "head_direction_entropy": hde,
    }
    return {
        "input": json.dumps(input_data, ensure_ascii=False),
        "output": json.dumps(output_features, ensure_ascii=False),
        "metadata_fold": split_name,
        "metadata_treebank_id": config_name,
        "metadata_language_code": lang_code,
        "metadata_treebank_name": config_name,
        "metadata_sent_id": sent_id,
    }


# === Treebank Processing (saves to temp file) ===
def process_treebank(config_name: str) -> dict | None:
    """Process one treebank. Saves dataset entry to temp file. Returns stats or None."""
    lang_code = config_name.split("_")[0]
    examples: list[dict] = []
    total_sents = 0
    malformed = 0

    try:
        ds = load_dataset(DATASET_ID, config_name)
    except Exception as e:
        logger.error(f"Failed to load {config_name}: {e}")
        return None

    available_splits = list(ds.keys())
    capped = False

    for split_name in available_splits:
        for row in ds[split_name]:
            total_sents += 1
            if len(examples) >= MAX_EXAMPLES_PER_TREEBANK:
                capped = True
                continue  # Still count total_sents but skip processing
            try:
                tokens = row["tokens"]
                upos_strs = [upos_int_to_str(u) for u in row["upos"]]
                heads = [int(h) for h in row["head"]]
                deprels = row["deprel"]
                feats = [f if f is not None else None for f in row["feats"]]
                example = compute_sentence_features(
                    tokens=tokens, heads=heads, deprels=deprels,
                    upos_strs=upos_strs, feats=feats, text=row["text"],
                    sent_id=row["sent_id"], split_name=split_name,
                    config_name=config_name, lang_code=lang_code,
                )
                if example is not None:
                    examples.append(example)
            except Exception as e:
                malformed += 1
                if malformed <= 3:
                    logger.warning(f"  Error in {config_name}/{split_name}: {e}")

    del ds
    gc.collect()

    filtered = len(examples)
    if not examples:
        return None

    # Save to temp file and free memory
    temp_path = TEMP_DIR / f"{config_name}.json"
    entry = {"dataset": config_name, "examples": examples}
    temp_path.write_text(json.dumps(entry, ensure_ascii=False))
    entry_size = temp_path.stat().st_size

    del examples, entry
    gc.collect()

    stats = {
        "total": total_sents, "filtered": filtered,
        "malformed": malformed, "splits": available_splits,
        "capped": capped, "temp_size_bytes": entry_size,
    }
    return stats


def stream_assemble_output(
    metadata: dict,
    successful_configs: list[str],
    out_path: Path,
) -> int:
    """Stream-assemble final JSON from temp files. Returns bytes written."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"metadata":')
        json.dump(metadata, f, ensure_ascii=False)
        f.write(',"datasets":[')
        first = True
        for config_name in successful_configs:
            temp_path = TEMP_DIR / f"{config_name}.json"
            if not temp_path.exists():
                continue
            if not first:
                f.write(",")
            first = False
            f.write(temp_path.read_text())
        f.write("]}")
    return out_path.stat().st_size


def split_output(
    metadata: dict,
    successful_configs: list[str],
    max_part_bytes: int = 95_000_000,
) -> list[Path]:
    """Split output into part files under max_part_bytes each."""
    out_dir = WORKSPACE / "full_data_out"
    out_dir.mkdir(exist_ok=True)

    part_num = 1
    part_configs: list[str] = []
    part_size = 0
    meta_overhead = len(json.dumps({"metadata": metadata, "datasets": []}).encode("utf-8"))
    part_files: list[Path] = []

    def write_part(pnum: int, configs: list[str]) -> Path:
        part_path = out_dir / f"full_data_out_{pnum}.json"
        with open(part_path, "w", encoding="utf-8") as f:
            f.write('{"metadata":')
            json.dump(metadata, f, ensure_ascii=False)
            f.write(',"datasets":[')
            first = True
            for cn in configs:
                tp = TEMP_DIR / f"{cn}.json"
                if not tp.exists():
                    continue
                if not first:
                    f.write(",")
                first = False
                f.write(tp.read_text())
            f.write("]}")
        logger.info(f"  Wrote part {pnum}: {part_path.name} ({part_path.stat().st_size / 1e6:.1f} MB, {len(configs)} treebanks)")
        return part_path

    for config_name in successful_configs:
        temp_path = TEMP_DIR / f"{config_name}.json"
        if not temp_path.exists():
            continue
        entry_size = temp_path.stat().st_size
        if part_size + entry_size > max_part_bytes - meta_overhead and part_configs:
            part_files.append(write_part(part_num, part_configs))
            part_num += 1
            part_configs = []
            part_size = 0
        part_configs.append(config_name)
        part_size += entry_size

    if part_configs:
        part_files.append(write_part(part_num, part_configs))

    return part_files


@logger.catch
def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Universal Dependencies Treebank Feature Extraction")
    logger.info(f"Source: {DATASET_ID}")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET / 1e9:.1f} GB")
    logger.info(f"Max examples per treebank: {MAX_EXAMPLES_PER_TREEBANK}")
    logger.info("=" * 60)

    # Get all configs
    logger.info("Fetching config names...")
    all_configs = get_dataset_config_names(DATASET_ID)
    logger.info(f"Found {len(all_configs)} treebank configs")

    # Phase 1: Process each treebank → temp file
    treebank_stats: dict = {}
    failed_treebanks: list[str] = []
    successful_configs: list[str] = []
    total_before = 0
    total_after = 0
    total_temp_bytes = 0

    for i, config in enumerate(all_configs):
        t_config = time.time()
        logger.info(f"[{i + 1}/{len(all_configs)}] {config}")

        stats = process_treebank(config)

        if stats is None:
            failed_treebanks.append(config)
            continue

        successful_configs.append(config)
        treebank_stats[config] = stats
        total_before += stats["total"]
        total_after += stats["filtered"]
        total_temp_bytes += stats["temp_size_bytes"]

        elapsed_config = time.time() - t_config
        logger.info(
            f"  -> {stats['filtered']} examples in {elapsed_config:.1f}s "
            f"{'[CAPPED]' if stats['capped'] else ''}"
            f"(cumulative: {total_after} ex, ~{total_temp_bytes / 1e6:.0f} MB)"
        )

        if (i + 1) % 20 == 0:
            gc.collect()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta_min = (len(all_configs) - (i + 1)) / rate if rate > 0 else 0
            mem_gb = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()) / 1e9
            logger.info(
                f"  Progress: {i + 1}/{len(all_configs)}, "
                f"{total_after} examples, ~{total_temp_bytes / 1e6:.0f} MB on disk, "
                f"mem={mem_gb:.1f}GB, elapsed={elapsed / 60:.1f}min, ETA={eta_min:.1f}min"
            )

    elapsed_phase1 = time.time() - t0
    logger.info(f"Phase 1 complete in {elapsed_phase1 / 60:.1f} min")
    logger.info(f"  Successful: {len(successful_configs)}, Failed: {len(failed_treebanks)}")
    logger.info(f"  Total before filter: {total_before}, after: {total_after}")
    logger.info(f"  Total temp size: {total_temp_bytes / 1e6:.1f} MB")

    # Build metadata
    metadata = {
        "source": DATASET_ID,
        "total_treebanks": len(all_configs),
        "successful_treebanks": len(successful_configs),
        "failed_treebanks": failed_treebanks,
        "min_sentence_length": MIN_EFFECTIVE_LENGTH,
        "length_filter": f"effective_length >= {MIN_EFFECTIVE_LENGTH} (excluding PUNCT)",
        "max_examples_per_treebank": MAX_EXAMPLES_PER_TREEBANK,
        "total_sentences_before_filter": total_before,
        "total_sentences_after_filter": total_after,
        "treebank_stats": treebank_stats,
    }

    # Phase 2: Assemble output
    out_path = WORKSPACE / "full_data_out.json"
    logger.info("Phase 2: Assembling output...")

    if total_temp_bytes < 95_000_000:
        # Small enough for a single file
        nbytes = stream_assemble_output(metadata, successful_configs, out_path)
        logger.info(f"  Single file: {out_path.name} ({nbytes / 1e6:.1f} MB)")
    else:
        # Need to split
        logger.info(f"  Output ~{total_temp_bytes / 1e6:.0f} MB, splitting into parts...")
        part_files = split_output(metadata, successful_configs)
        logger.info(f"  Created {len(part_files)} part files in full_data_out/")

        # Also write a small first-part as full_data_out.json for validation/mini/preview
        # Use just the first few treebanks (enough for 10+ examples)
        mini_configs = successful_configs[:5]
        stream_assemble_output(metadata, mini_configs, out_path)
        logger.info(f"  Also wrote mini version as full_data_out.json ({out_path.stat().st_size / 1e6:.1f} MB)")

    elapsed_total = time.time() - t0
    logger.info(f"Total runtime: {elapsed_total / 60:.1f} minutes")
    logger.info("Done!")


if __name__ == "__main__":
    main()
