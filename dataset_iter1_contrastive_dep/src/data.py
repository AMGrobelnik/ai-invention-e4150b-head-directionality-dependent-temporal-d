#!/usr/bin/env python3
"""Generate synthetic dataset of contrastive dependency tree pairs.

Enumerates all projective linearizations of small rooted dependency trees (n=5-8),
computes IMB profiles, DD multisets, LLF, and superlinear costs, then identifies
contrastive pairs sharing identical DD multisets but differing in max(IMB)/LLF.
Validates the mathematical identity total_imb == sum(d_k*(d_k+1)/2).
"""

import itertools
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

# =========================================================================
# Configuration
# =========================================================================
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/data/runs/comp-ling-dobrovoljc_bnd/"
    "3_invention_loop/iter_1/gen_art/data_id5_it1__opus"
)
ALPHAS = [1.0, 1.2, 1.5, 2.0, 3.0]
OEIS_A000081 = {1: 1, 2: 1, 3: 2, 4: 4, 5: 9, 6: 20, 7: 48, 8: 115}

# =========================================================================
# Logging
# =========================================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

# =========================================================================
# Hardware detection & memory limits
# =========================================================================

def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return 16.0


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.6, 30) * 1024**3)

try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
except Exception as e:
    logger.warning(f"Could not set memory limit: {e}")


# =========================================================================
# 1. Tree Enumeration  (OEIS A000081)
# =========================================================================
_tree_cache: dict[int, list[tuple]] = {}


def _partitions(n: int, min_part: int = 1):
    """Yield integer partitions of n with parts in non-decreasing order."""
    if n == 0:
        yield ()
        return
    for first in range(min_part, n + 1):
        for rest in _partitions(n - first, first):
            yield (first,) + rest


def enum_rooted_trees(n: int) -> list[tuple]:
    """All non-isomorphic rooted trees on n nodes as canonical nested tuples."""
    if n in _tree_cache:
        return _tree_cache[n]
    if n == 1:
        _tree_cache[1] = [()]
        return [()]

    results: set[tuple] = set()
    for partition in _partitions(n - 1):
        # Group consecutive equal parts
        groups: list[tuple[int, int]] = []
        i = 0
        while i < len(partition):
            sz = partition[i]
            cnt = 0
            while i < len(partition) and partition[i] == sz:
                cnt += 1
                i += 1
            groups.append((sz, cnt))

        group_opts = []
        for sz, cnt in groups:
            trees = enum_rooted_trees(sz)
            group_opts.append(
                list(itertools.combinations_with_replacement(trees, cnt))
            )

        for combo in itertools.product(*group_opts):
            children: list[tuple] = []
            for grp in combo:
                children.extend(grp)
            results.add(tuple(sorted(children)))

    result_list = sorted(results)
    _tree_cache[n] = result_list
    return result_list


def tree_to_str(t: tuple) -> str:
    """Canonical string like '(()())'."""
    if not t:
        return "()"
    return "(" + "".join(tree_to_str(c) for c in t) + ")"


def count_proj_lins(t: tuple) -> int:
    """Count projective linearizations without enumerating."""
    if not t:
        return 1
    k = len(t)
    return math.factorial(k + 1) * math.prod(count_proj_lins(c) for c in t)


# =========================================================================
# 2. Labeling & Projective Linearization
# =========================================================================

def assign_labels(tree: tuple, start: int = 0):
    """DFS-label tree nodes. Returns ((label, [children]), next_label)."""
    label = start
    nxt = start + 1
    children = []
    for sub in tree:
        child, nxt = assign_labels(sub, nxt)
        children.append(child)
    return (label, children), nxt


def get_parent_map(labeled_tree) -> dict[int, int]:
    """node_label -> parent_label  (root -> -1)."""
    parent: dict[int, int] = {}

    def _visit(node, par):
        lbl, ch = node
        parent[lbl] = par
        for c in ch:
            _visit(c, lbl)

    _visit(labeled_tree, -1)
    return parent


def projective_lins(labeled_tree):
    """Yield all projective linearizations as lists of node labels."""
    lbl, children = labeled_tree
    if not children:
        yield [lbl]
        return
    k = len(children)
    child_lins = [list(projective_lins(c)) for c in children]

    for perm in itertools.permutations(range(k)):
        for head_pos in range(k + 1):
            ordered = [child_lins[perm[i]] for i in range(k)]
            for combo in itertools.product(*ordered):
                result: list[int] = []
                for i in range(head_pos):
                    result.extend(combo[i])
                result.append(lbl)
                for i in range(head_pos, k):
                    result.extend(combo[i])
                yield result


# =========================================================================
# 3. Metric Computation
# =========================================================================

def compute_metrics_from_heads(heads_array: list[int]) -> dict:
    """Compute all metrics directly from a 1-indexed heads_array (0=root)."""
    n = len(heads_array)
    dd_list: list[int] = []
    deps: list[tuple[int, int]] = []
    for j in range(n):
        h = heads_array[j]
        if h != 0:
            dep_pos = j + 1
            dd = abs(dep_pos - h)
            dd_list.append(dd)
            deps.append((min(dep_pos, h), max(dep_pos, h)))

    dd_sorted = sorted(dd_list)
    dd_multiset = ",".join(str(d) for d in dd_sorted)

    # IMB profile
    imb = [0.0] * n
    for j_pos in range(1, n + 1):
        burden = 0.0
        for op, cl in deps:
            if op <= j_pos <= cl:
                burden += (j_pos - op)
        imb[j_pos - 1] = burden

    max_imb = max(imb) if imb else 0.0
    total_imb = sum(imb)
    mean_imb = total_imb / n if n > 0 else 0.0
    llf = mean_imb / max_imb if max_imb > 0 else 0.0

    total_imb_formula = sum(d * (d + 1) / 2.0 for d in dd_list)
    formula_verified = abs(total_imb - total_imb_formula) < 1e-9

    costs = {a: sum(v ** a for v in imb) for a in ALPHAS}

    return {
        "dd_list": dd_sorted,
        "dd_multiset": dd_multiset,
        "imb_profile": imb,
        "max_imb": max_imb,
        "mean_imb": mean_imb,
        "llf": llf,
        "total_imb": total_imb,
        "total_imb_formula": total_imb_formula,
        "formula_verified": formula_verified,
        "cost_alpha_1_0": costs[1.0],
        "cost_alpha_1_2": costs[1.2],
        "cost_alpha_1_5": costs[1.5],
        "cost_alpha_2_0": costs[2.0],
        "cost_alpha_3_0": costs[3.0],
    }


def lin_to_heads(linearization: list[int], parent_map: dict[int, int]) -> list[int]:
    """Convert a linearization + parent_map to a 1-indexed heads_array."""
    pos_of = {label: i + 1 for i, label in enumerate(linearization)}
    heads: list[int] = []
    for label in linearization:
        par = parent_map[label]
        heads.append(0 if par == -1 else pos_of[par])
    return heads


# =========================================================================
# 4. Process one tree shape  (worker for multiprocessing)
# =========================================================================

def process_tree_shape(n: int, shape_idx: int, tree_shape: tuple) -> list[dict]:
    """Enumerate unique projective linearizations for one shape, compute metrics."""
    tree_id = f"t{n}_{shape_idx:03d}"
    tree_str = tree_to_str(tree_shape)

    labeled_tree, _ = assign_labels(tree_shape)
    parent_map = get_parent_map(labeled_tree)

    seen_heads: set[tuple[int, ...]] = set()
    rows: list[dict] = []
    lin_id = 0

    for lin in projective_lins(labeled_tree):
        ha = lin_to_heads(lin, parent_map)
        ha_key = tuple(ha)
        if ha_key in seen_heads:
            continue
        seen_heads.add(ha_key)

        metrics = compute_metrics_from_heads(ha)
        rows.append({
            "n_words": n,
            "tree_topology": tree_str,
            "tree_topology_id": tree_id,
            "linearization_id": lin_id,
            "heads_array": ha,
            **metrics,
        })
        lin_id += 1

    return rows


# =========================================================================
# 5. Worked-example validation
# =========================================================================
WORKED_A_HEADS = [0, 1, 2, 2, 3]
WORKED_B_HEADS = [0, 1, 1, 3, 3]


def validate_worked_example():
    """Compute and log worked-example verification."""
    ma = compute_metrics_from_heads(WORKED_A_HEADS)
    mb = compute_metrics_from_heads(WORKED_B_HEADS)

    expected_a = {"imb": [0, 1, 2, 3, 2], "llf": 8 / 15, "cost2": 18.0}
    expected_b = {"imb": [0, 2, 2, 2, 2], "llf": 0.8, "cost2": 16.0}

    ok_a = (
        ma["imb_profile"] == expected_a["imb"]
        and abs(ma["llf"] - expected_a["llf"]) < 1e-4
        and ma["cost_alpha_2_0"] == expected_a["cost2"]
    )
    ok_b = (
        mb["imb_profile"] == expected_b["imb"]
        and abs(mb["llf"] - expected_b["llf"]) < 1e-4
        and mb["cost_alpha_2_0"] == expected_b["cost2"]
    )

    logger.info(f"Worked-example Tree A: IMB={ma['imb_profile']} LLF={ma['llf']:.4f} "
                f"cost_a2={ma['cost_alpha_2_0']}  PASS={ok_a}")
    logger.info(f"Worked-example Tree B: IMB={mb['imb_profile']} LLF={mb['llf']:.4f} "
                f"cost_a2={mb['cost_alpha_2_0']}  PASS={ok_b}")
    logger.info(f"Same DD multiset: {ma['dd_multiset']}=={mb['dd_multiset']} "
                f"→ {ma['dd_multiset'] == mb['dd_multiset']}")
    logger.info(f"Same cost@1.0: {ma['cost_alpha_1_0']}=={mb['cost_alpha_1_0']} "
                f"→ {abs(ma['cost_alpha_1_0'] - mb['cost_alpha_1_0']) < 1e-9}")

    if not (ok_a and ok_b):
        logger.error("Worked-example numbers do NOT match hypothesis!")
    return ma, mb


# =========================================================================
# 6. Main pipeline
# =========================================================================

@logger.catch
def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    logger.info("=== Contrastive Dependency Tree Pairs Dataset ===")
    logger.info(f"max_n={max_n}  CPUs={NUM_CPUS}  RAM={TOTAL_RAM_GB:.1f}GB")

    # ---- validate worked example -----------------------------------
    worked_a_m, worked_b_m = validate_worked_example()

    # ---- Phase 1: enumerate shapes, predict scale ------------------
    logger.info("Phase 1: Enumerating tree shapes and predicting scale...")
    all_shapes: dict[int, list[tuple]] = {}
    for n in range(5, max_n + 1):
        shapes = enum_rooted_trees(n)
        expected = OEIS_A000081[n]
        if len(shapes) != expected:
            logger.error(f"n={n}: {len(shapes)} shapes vs expected {expected}")
            raise ValueError(f"Tree enumeration bug at n={n}")
        total_lins = sum(count_proj_lins(s) for s in shapes)
        all_shapes[n] = shapes
        logger.info(f"  n={n}: {len(shapes)} shapes, "
                     f"{total_lins:,} raw projective linearizations")

    # ---- Phase 2: process shapes -----------------------------------
    logger.info("Phase 2: Computing metrics for all unique linearizations...")
    all_rows: list[dict] = []

    for n in sorted(all_shapes):
        shapes = all_shapes[n]
        t0 = time.time()
        before = len(all_rows)

        if n >= 7 and NUM_CPUS > 1:
            tasks = [(n, idx, s) for idx, s in enumerate(shapes)]
            with ProcessPoolExecutor(max_workers=min(NUM_CPUS - 1, len(tasks))) as pool:
                futs = {pool.submit(process_tree_shape, *t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        all_rows.extend(fut.result())
                    except Exception:
                        logger.error(f"Worker failed for {futs[fut]}")
                        raise
        else:
            for idx, shape in enumerate(shapes):
                all_rows.extend(process_tree_shape(n, idx, shape))

        added = len(all_rows) - before
        elapsed = time.time() - t0
        logger.info(f"  n={n}: {added:,} unique heads_arrays in {elapsed:.1f}s")

    logger.info(f"Total unique linearizations: {len(all_rows):,}")

    # ---- formula verification --------------------------------------
    n_ok = sum(1 for r in all_rows if r["formula_verified"])
    n_fail = len(all_rows) - n_ok
    logger.info(f"Formula verified: {n_ok:,} OK, {n_fail:,} FAIL  "
                f"({100 * n_ok / max(len(all_rows), 1):.2f}%)")
    if n_fail:
        logger.error("FORMULA VERIFICATION FAILURES DETECTED")

    # ---- Phase 3: contrastive groups --------------------------------
    logger.info("Phase 3: Identifying contrastive groups...")

    # Within-topology: (topology_id, dd_multiset)
    within_grp: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(all_rows):
        within_grp[(r["tree_topology_id"], r["dd_multiset"])].append(i)

    # Cross-topology: (n_words, dd_multiset)
    cross_grp: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(all_rows):
        cross_grp[(r["n_words"], r["dd_multiset"])].append(i)

    # Tag within-topology contrastive (different max_imb)
    wg_id = 0
    within_contrastive: dict[tuple, str] = {}
    for key, idxs in within_grp.items():
        max_imbs = {all_rows[i]["max_imb"] for i in idxs}
        if len(max_imbs) > 1:
            within_contrastive[key] = f"wg_{wg_id:05d}"
            wg_id += 1

    # Tag cross-topology contrastive (different max_imb across topologies)
    xg_id = 0
    cross_contrastive: dict[tuple, str] = {}
    for key, idxs in cross_grp.items():
        max_imbs = {all_rows[i]["max_imb"] for i in idxs}
        if len(max_imbs) > 1:
            cross_contrastive[key] = f"xg_{xg_id:05d}"
            xg_id += 1

    logger.info(f"  Within-topology contrastive groups: {len(within_contrastive):,}")
    logger.info(f"  Cross-topology contrastive groups:  {len(cross_contrastive):,}")

    # Annotate rows
    for i, r in enumerate(all_rows):
        wk = (r["tree_topology_id"], r["dd_multiset"])
        xk = (r["n_words"], r["dd_multiset"])
        r["contrastive_group_id"] = within_contrastive.get(wk)
        r["cross_topo_group_id"] = cross_contrastive.get(xk)
        r["is_contrastive"] = wk in within_contrastive
        r["is_cross_contrastive"] = xk in cross_contrastive
        r["is_worked_example"] = False

    # ---- Phase 4: flag worked example in enumerated data -----------
    logger.info("Phase 4: Flagging worked-example rows...")
    tree_b_found = False
    for r in all_rows:
        if r["heads_array"] == WORKED_B_HEADS and r["n_words"] == 5:
            r["is_worked_example"] = True
            r["cross_topo_group_id"] = "worked_example"
            r["is_cross_contrastive"] = True
            tree_b_found = True
            logger.info(f"  Tree B found: {r['tree_topology_id']} lin={r['linearization_id']}")

    if not tree_b_found:
        logger.warning("  Tree B NOT found in projective enumeration (unexpected)")

    # Tree A is non-projective → add as an explicit special row
    tree_a_row = {
        "n_words": 5,
        "tree_topology": "nonprojective_worked_example_A",
        "tree_topology_id": "worked_A",
        "linearization_id": 0,
        "heads_array": WORKED_A_HEADS,
        **worked_a_m,
        "contrastive_group_id": None,
        "cross_topo_group_id": "worked_example",
        "is_contrastive": False,
        "is_cross_contrastive": True,
        "is_worked_example": True,
    }
    all_rows.append(tree_a_row)
    logger.info("  Tree A added as non-projective worked-example row")

    # ---- Phase 5: validate contrastive properties -------------------
    logger.info("Phase 5: Validating contrastive properties...")
    cost10_violations = 0
    jensen_checks = 0
    jensen_violations = 0

    for key, gid in within_contrastive.items():
        idxs = within_grp[key]
        c10s = [all_rows[i]["cost_alpha_1_0"] for i in idxs]
        if max(c10s) - min(c10s) > 1e-9:
            cost10_violations += 1

        for alpha_field in ["cost_alpha_2_0", "cost_alpha_3_0"]:
            pairs = [(all_rows[i]["llf"], all_rows[i][alpha_field]) for i in idxs]
            pairs.sort(key=lambda x: x[0], reverse=True)
            for j in range(len(pairs) - 1):
                jensen_checks += 1
                if pairs[j][1] > pairs[j + 1][1] + 1e-9:
                    jensen_violations += 1

    logger.info(f"  cost@1.0 identity violations: {cost10_violations}")
    logger.info(f"  Jensen checks: {jensen_checks}, violations: {jensen_violations}")

    # ---- Phase 6: summary ------------------------------------------
    logger.info("=== SUMMARY ===")
    for n in sorted(all_shapes):
        prefix = f"t{n}_"
        nr = sum(1 for r in all_rows if r["tree_topology_id"].startswith(prefix))
        nc = sum(1 for r in all_rows
                 if r["tree_topology_id"].startswith(prefix) and r["is_contrastive"])
        nx = sum(1 for r in all_rows
                 if r["tree_topology_id"].startswith(prefix) and r["is_cross_contrastive"])
        logger.info(f"  n={n}: {nr:,} rows, {nc:,} within-contrastive, "
                     f"{nx:,} cross-contrastive")

    n_contrastive = sum(1 for r in all_rows if r["is_contrastive"])
    n_cross = sum(1 for r in all_rows if r["is_cross_contrastive"])
    logger.info(f"  TOTAL: {len(all_rows):,} rows | "
                f"{n_contrastive:,} within-contrastive | "
                f"{n_cross:,} cross-contrastive")
    logger.info(f"  Within-topo groups: {len(within_contrastive)} | "
                f"Cross-topo groups: {len(cross_contrastive)}")

    # ---- Phase 7: convert to exp_sel_data_out schema ----------------
    logger.info("Phase 7: Building output JSON...")
    examples: list[dict] = []
    for r in all_rows:
        inp = {
            "n_words": r["n_words"],
            "tree_topology": r["tree_topology"],
            "heads_array": r["heads_array"],
            "words": list(range(1, r["n_words"] + 1)),
        }
        out = {
            "dd_list": r["dd_list"],
            "dd_multiset": r["dd_multiset"],
            "imb_profile": r["imb_profile"],
            "max_imb": r["max_imb"],
            "mean_imb": r["mean_imb"],
            "llf": r["llf"],
            "total_imb": r["total_imb"],
            "total_imb_formula": r["total_imb_formula"],
            "formula_verified": r["formula_verified"],
            "cost_alpha_1_0": r["cost_alpha_1_0"],
            "cost_alpha_1_2": r["cost_alpha_1_2"],
            "cost_alpha_1_5": r["cost_alpha_1_5"],
            "cost_alpha_2_0": r["cost_alpha_2_0"],
            "cost_alpha_3_0": r["cost_alpha_3_0"],
        }
        examples.append({
            "input": json.dumps(inp, separators=(",", ":")),
            "output": json.dumps(out, separators=(",", ":")),
            "metadata_fold": "all",
            "metadata_tree_topology_id": r["tree_topology_id"],
            "metadata_linearization_id": r["linearization_id"],
            "metadata_contrastive_group_id": r.get("contrastive_group_id") or "",
            "metadata_cross_topo_group_id": r.get("cross_topo_group_id") or "",
            "metadata_is_contrastive": r.get("is_contrastive", False),
            "metadata_is_cross_contrastive": r.get("is_cross_contrastive", False),
            "metadata_is_worked_example": r.get("is_worked_example", False),
            "metadata_n_deps_in_tree": r["n_words"] - 1,
            "metadata_n_words": r["n_words"],
            "metadata_dd_multiset": r["dd_multiset"],
            "metadata_max_imb": r["max_imb"],
            "metadata_llf": r["llf"],
        })

    output = {
        "datasets": [{
            "dataset": "contrastive_dependency_tree_pairs",
            "examples": examples,
        }]
    }

    out_path = WORKSPACE / "full_data_out.json"
    logger.info(f"Writing {len(examples):,} examples to {out_path}...")
    out_path.write_text(json.dumps(output))
    fsize = out_path.stat().st_size
    logger.info(f"Output: {fsize / 1e6:.1f} MB  ({fsize:,} bytes)")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
