"""Build train_labeled.jsonl + swe_smith_filtered.jsonl (run: `uv run --extra train python triage/data/build_dataset.py`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent
ANNOTATIONS_CSV = DATA_DIR / "ensembled_annotations_public.csv"
LABELED_OUT = DATA_DIR / "train_labeled.jsonl"
SWE_SMITH_OUT = DATA_DIR / "swe_smith_filtered.jsonl"

# Human annotation buckets -> the classifier's 3-class vocabulary.
DIFFICULTY_MAP = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "hard",
}


def build_labeled() -> None:
    with ANNOTATIONS_CSV.open() as f:
        annotations = {
            row["instance_id"]: DIFFICULTY_MAP[row["difficulty"]]
            for row in csv.DictReader(f)
            if row["difficulty"] in DIFFICULTY_MAP
        }
    print(f"[build] {len(annotations)} human difficulty annotations")

    sweb = load_dataset("SWE-bench/SWE-bench", split="test")
    n = 0
    with LABELED_OUT.open("w") as out:
        for row in sweb:
            label = annotations.get(row["instance_id"])
            if label is None or not row["problem_statement"].strip():
                continue
            out.write(json.dumps({
                "instance_id": row["instance_id"],
                "text": row["problem_statement"],
                "label": label,
            }) + "\n")
            n += 1
    print(f"[build] wrote {n} labeled instances -> {LABELED_OUT.name}")


def build_swe_smith() -> None:
    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    n = 0
    with SWE_SMITH_OUT.open("w") as out:
        for row in ds:
            if ".pr_" not in row["instance_id"]:
                continue
            out.write(json.dumps({
                "instance_id": row["instance_id"],
                "text": row["problem_statement"],
                "label": None,
            }) + "\n")
            n += 1
    print(f"[build] wrote {n} pr_mirror instances -> {SWE_SMITH_OUT.name}")


if __name__ == "__main__":
    build_labeled()
    build_swe_smith()
