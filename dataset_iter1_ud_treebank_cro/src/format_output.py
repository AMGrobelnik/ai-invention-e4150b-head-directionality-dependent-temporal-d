#!/usr/bin/env python3
"""Convert raw data_out.json to schema-compliant format for exp_sel_data_out validation."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent


def main():
    raw_path = WORKSPACE / "data_out.json"
    logger.info(f"Reading raw output from {raw_path}")
    raw_data = json.loads(raw_path.read_text())
    logger.info(f"Loaded {len(raw_data)} records")

    # Convert to schema-compliant format:
    # {"datasets": [{"dataset": "...", "examples": [{"input": "...", "output": "...", "metadata_fold": "..."}]}]}
    examples = []
    for item in raw_data:
        examples.append({
            "input": item["input"],
            "output": json.dumps(item["output"], ensure_ascii=False),
            "metadata_fold": item.get("metadata_fold", "metadata"),
        })

    schema_output = {
        "metadata": {
            "source": "commul/universal_dependencies",
            "description": "Cross-linguistic typological and modality metadata for all UD treebanks",
            "version": "1.0",
            "external_sources": [
                "WALS Feature 81A (word order)",
                "Glottolog CLDF (language families)",
                "Dobrovoljc 2022 (spoken treebank survey)",
            ],
            "total_treebanks": len(examples),
        },
        "datasets": [
            {
                "dataset": "commul/universal_dependencies",
                "examples": examples,
            }
        ],
    }

    # Save schema-compliant version
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(schema_output, indent=2, ensure_ascii=False))
    logger.info(f"Saved schema-compliant output ({len(examples)} examples) to {out_path}")

    # Also save the raw flat array for easy programmatic use
    raw_flat_path = WORKSPACE / "data_out_flat.json"
    raw_flat_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False))
    logger.info(f"Saved flat array version to {raw_flat_path}")


if __name__ == "__main__":
    main()
