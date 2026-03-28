# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru"]
# ///
"""Format compiled UD treebank metadata into exp_sel_data_out.json schema.

Loads raw treebank metadata from data_out_flat.json (produced by build_metadata.py),
standardizes each treebank row into a schema-compliant example with proper metadata_*
fields, and saves to full_data_out.json.

Each of the ~336 treebank rows becomes one example in the output.
"""

import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent


def main() -> None:
    # ── Load raw flat data ───────────────────────────────────────────────
    flat_path = WORKSPACE / "data_out_flat.json"
    if not flat_path.exists():
        logger.error(f"Source file not found: {flat_path}")
        logger.info("Run build_metadata.py first to compile treebank metadata.")
        sys.exit(1)

    raw_data: list[dict] = json.loads(flat_path.read_text())
    logger.info(f"Loaded {len(raw_data)} raw treebank records from {flat_path.name}")

    # ── Convert each row into a schema-compliant example ─────────────────
    # Schema requires: input (str), output (str), optional metadata_* fields
    # Each treebank row = one example
    examples: list[dict] = []

    for row in raw_data:
        rec: dict = row["output"]  # the full metadata dict
        treebank_id: str = row["input"]

        # Build the example with input/output as strings
        example: dict = {
            # input: the treebank identifier (the key for joining)
            "input": treebank_id,
            # output: full metadata record as JSON string
            "output": json.dumps(rec, ensure_ascii=False),
            # ── Per-example metadata_* fields for easy filtering ──────
            "metadata_fold": "metadata",
            "metadata_task_type": "metadata_compilation",
            "metadata_row_index": len(examples),
            "metadata_lang_code": rec.get("lang_code", ""),
            "metadata_iso_639_3": rec.get("iso_639_3") or "",
            "metadata_language_name": rec.get("language_name") or "",
            "metadata_wals_word_order": rec.get("wals_word_order") or "",
            "metadata_glottolog_family_name": rec.get("glottolog_family_name") or "",
            "metadata_macroarea": rec.get("macroarea") or "",
            "metadata_modality": rec.get("modality", "written"),
            "metadata_spoken_written_pair_id": rec.get("spoken_written_pair_id") or "",
            "metadata_case_richness_count": rec.get("case_richness_count", 0),
            "metadata_head_direction_entropy_binary": rec.get("head_direction_entropy_binary"),
            "metadata_has_dom": rec.get("has_differential_object_marking", False),
            "metadata_num_sentences": rec.get("num_sentences", 0),
            "metadata_num_tokens": rec.get("num_tokens", 0),
        }
        examples.append(example)

    logger.info(f"Converted {len(examples)} examples")

    # ── Wrap in schema-compliant structure ────────────────────────────────
    output = {
        "metadata": {
            "description": (
                "Cross-linguistic typological and modality metadata for all UD treebanks. "
                "Compiled from commul/universal_dependencies (HuggingFace), WALS Feature 81A "
                "(word order), Glottolog CLDF (language families), and curated spoken treebank "
                "labels from Dobrovoljc (2022)."
            ),
            "source": "commul/universal_dependencies",
            "version": "1.0",
            "total_treebanks": len(examples),
            "external_sources": [
                "WALS Feature 81A (word order classification)",
                "Glottolog CLDF (language family/macroarea)",
                "Dobrovoljc 2022 (spoken treebank survey)",
            ],
            "coverage": {
                "wals_word_order_pct": round(
                    100 * sum(1 for e in examples if e["metadata_wals_word_order"]) / len(examples), 1
                ),
                "glottolog_family_pct": round(
                    100 * sum(1 for e in examples if e["metadata_glottolog_family_name"]) / len(examples), 1
                ),
            },
        },
        "datasets": [
            {
                "dataset": "commul/universal_dependencies",
                "examples": examples,
            }
        ],
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(examples)} examples to {out_path.name}")

    # Also overwrite data_out.json (the canonical output)
    canonical = WORKSPACE / "data_out.json"
    canonical.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Saved canonical copy to {canonical.name}")

    # ── Quick validation summary ─────────────────────────────────────────
    n = len(examples)
    modalities = {}
    word_orders = {}
    for e in examples:
        m = e["metadata_modality"]
        modalities[m] = modalities.get(m, 0) + 1
        wo = e["metadata_wals_word_order"]
        if wo:
            word_orders[wo] = word_orders.get(wo, 0) + 1

    logger.info(f"Modalities: {modalities}")
    logger.info(f"Word orders: {word_orders}")
    logger.info(
        f"WALS coverage: {sum(1 for e in examples if e['metadata_wals_word_order'])}/{n} "
        f"({output['metadata']['coverage']['wals_word_order_pct']}%)"
    )
    logger.info(
        f"Glottolog coverage: {sum(1 for e in examples if e['metadata_glottolog_family_name'])}/{n} "
        f"({output['metadata']['coverage']['glottolog_family_pct']}%)"
    )


if __name__ == "__main__":
    main()
