#!/usr/bin/env python3
"""CPU-only, stdlib-only split audit and pre-result kcross subset freeze.

Run with ``python -I tools/audit_racer_data.py --freeze`` before any formal
results, then ``--verify`` against the same source data before using indices.
This audits question text, not paraphrases, provenance, or model pretraining.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"
MANIFEST = ROOT / "experiments" / "racer_data_audit_v1.json"
TASKS = {
    "nlgraph": ("nlgraph",),
    "gsm8k": ("gsm8k",),
    "hellaswag": ("hellaswag",),
    "mmlu": ("mmlu",),
    "mmlu_pro": ("mmlu_pro",),
    "truthfulqa": ("truthfulqa",),
    "medical": ("medqa", "medmcqa"),
    "science": ("scifact", "stem"),
    "culture": ("normad_country", "normad_value"),
}
KCROSS = "knowledge_crosswords"
NORMALIZATION = "NFKC + casefold + all Unicode whitespace collapsed to ASCII spaces; kcross removes terminal Options: instruction"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def question(row):
    value = row["question"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing or empty question")
    return value


def stem(text, dataset):
    if dataset == KCROSS:
        # Choices are stored outside question; exclude its generic trailing
        # instruction, retaining the full constraints and their original order.
        suffix = "\nOptions:\nPlease select the correct answer from A, B, C, D."
        if not text.endswith(suffix):
            raise ValueError("unrecognized kcross question footer")
        text = text[:-len(suffix)]
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def overlap(dev, test, key):
    """Count test rows and distinct keys also present in dev (not pairs)."""
    left, right = Counter(map(key, dev)), Counter(map(key, test))
    shared = left.keys() & right.keys()
    return {
        "dev_rows": len(dev), "test_rows": len(test),
        "dev_unique": len(left), "test_unique": len(right),
        "shared_unique": len(shared),
        "dev_overlap_rows": sum(left[k] for k in shared),
        "test_overlap_rows": sum(right[k] for k in shared),
    }


def read_data(data_dir, name):
    path = data_dir / (name + ".json")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"invalid dataset: {name}")
    for split in ("dev", "test"):
        if not isinstance(data.get(split), list) or not data[split]:
            raise ValueError(f"missing/non-list/empty {name}.{split}")
        for row in data[split]:
            question(row)
    return data, {"path": "data/eval/" + path.name,
                  "bytes": len(raw), "sha256": sha(raw)}


def build_manifest(data_dir=DATA):
    names = sorted({name for pair in TASKS.values() for name in pair} | {KCROSS})
    data, sources = {}, {}
    datasets = {}
    for name in names:
        data[name], sources[name] = read_data(data_dir, name)
        dev, test = (data[name][split] for split in ("dev", "test"))
        datasets[name] = {
            "exact_question": overlap(dev, test, question),
            "normalized_stem": overlap(dev, test, lambda r: stem(question(r), name)),
        }

    tasks = {}
    for task, components in TASKS.items():
        # Task-level overlap also catches cross-component collisions. Prefixing
        # with dataset here would conceal those. Never compare answer labels.
        dev = [(name, row) for name in components for row in data[name]["dev"]]
        test = [(name, row) for name in components for row in data[name]["test"]]
        tasks[task] = {
            "components": list(components),
            "exact_question": overlap(dev, test, lambda item: question(item[1])),
            "normalized_stem": overlap(
                dev, test, lambda item: stem(question(item[1]), item[0])),
        }

    dev = data[KCROSS]["dev"]
    test = data[KCROSS]["test"]
    dev_stems = {stem(question(row), KCROSS) for row in dev}
    seen = set(dev_stems)
    selected = []
    for index, row in enumerate(test):
        key = stem(question(row), KCROSS)
        if key not in seen:
            selected.append({"index": index, "stem_sha256": sha(key.encode("utf-8")),
                             "question_sha256": sha(question(row).encode("utf-8"))})
            seen.add(key)
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": 1,
        "scope": "question-text dev/test overlap only; no test answers or model results used for selection",
        "normalization": NORMALIZATION,
        "tasks": tasks,
        "datasets": datasets,
        "sources": sources,
        "kcross_subset": {
            "dataset": KCROSS,
            "rule": "ascending test index; first occurrence of each normalized stem absent from all dev stems",
            "selected": selected,
            "selected_sha256": sha(payload),
        },
        "limitations": [
            "Normalized text matching does not identify paraphrases, shared underlying instances with changed wording, or training/pretraining leakage.",
            "The kcross subset excludes identical normalized constraint strings, not re-ordered/equivalent graph constraints or shared entity families.",
            "Subset indices are non-contiguous: current contiguous split evaluator does not automatically consume this manifest; a consumer must enforce source hash and selected indices explicitly before claiming subset results.",
            "Audits the current on-disk data only; does not establish historical experiment data identity or statistical independence.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true", help="create immutable manifest (fails if it exists)")
    action.add_argument("--verify", action="store_true", help="recompute and check frozen manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    current = build_manifest()
    if args.freeze:
        with args.manifest.open("x", encoding="utf-8") as output:
            json.dump(current, output, ensure_ascii=False, indent=2)
            output.write("\n")
    else:
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        if frozen != current:
            raise SystemExit("FAIL: audit or data changed since freeze")
    for name, row in current["tasks"].items():
        print(f"{name}: exact {row['exact_question']['test_overlap_rows']}/{row['exact_question']['test_rows']} test rows; normalized {row['normalized_stem']['test_overlap_rows']}/{row['normalized_stem']['test_rows']}")
    sub = current["kcross_subset"]
    print(f"kcross: {len(sub['selected'])} stem-deduplicated non-dev test rows; subset SHA256 {sub['selected_sha256']}")
    print(f"{'Frozen' if args.freeze else 'Verified'}: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
