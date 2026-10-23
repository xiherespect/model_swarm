#!/usr/bin/env python3
"""Read only explicitly allowlisted legacy evidence; never discover or rank runs.

Run with python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json.
No model imports, rescoring, GPU, globbing, eval, or original-run writes.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re


class EvidenceError(ValueError):
    """Evidence is ambiguous, changed, unsafe, or inconsistent."""


def literal_mapping(text):
    """Parse a literal dict or Namespace(key=<literal>); never execute calls."""
    try:
        tree = ast.parse(text.strip(), mode="eval").body
        if isinstance(tree, ast.Call):
            if not isinstance(tree.func, ast.Name) or tree.func.id != "Namespace" or tree.args:
                raise EvidenceError("only literal Namespace keywords are allowed")
            keys = [kw.arg for kw in tree.keywords]
            if None in keys or len(set(keys)) != len(keys):
                raise EvidenceError("invalid Namespace keys")
            value = {kw.arg: ast.literal_eval(kw.value) for kw in tree.keywords}
        else:
            value = ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise EvidenceError("not a literal mapping") from exc
    if not isinstance(value, dict):
        raise EvidenceError("expected a mapping")
    return value


def score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("score must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise EvidenceError("score outside [0, 1]")
    return float(value)


def harmonic(values):
    if len(values) != 2:
        raise EvidenceError("expected exactly two domain components")
    a, b = map(score, values)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def stable_particle(scratch):
    pairs = sorted((int(m.group(1)), score(value)) for key, value in scratch.items()
                   for m in [re.fullmatch(r"particle_(\d+)_best", key)] if m)
    if not pairs or [i for i, _ in pairs] != list(range(len(pairs))):
        raise EvidenceError("particle indices must be contiguous from zero")
    # Python max retains the first item for a tie, as legacy list.index(max()).
    return max(pairs, key=lambda pair: pair[1])


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise EvidenceError("invalid evidence path")
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or any(c in relative for c in "*?[]"):
        raise EvidenceError("evidence paths must be explicit relative paths")
    path = (root / p).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceError("evidence path escapes root") from exc
    return path


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise EvidenceError("unsupported manifest schema")
    if manifest.get("selection_policy") != "frozen_explicit_runs_no_test_selection":
        raise EvidenceError("test-based or unstated run selection is forbidden")
    if manifest.get("evidence_track") != "legacy_aggregation_corrected_v1":
        raise EvidenceError("this reader cannot produce strict rescoring")
    cohorts = manifest.get("cohorts", [])
    if not cohorts or len({c["id"] for c in cohorts}) != len(cohorts):
        raise EvidenceError("missing or duplicate cohorts")
    definitions = {c["id"]: c for c in cohorts}
    seen, ids = set(), set()
    for row in manifest.get("runs", []):
        if row["id"] in ids:
            raise EvidenceError("duplicate run id")
        ids.add(row["id"])
        cohort = definitions.get(row["cohort"])
        if cohort is None or row["task"] not in cohort["tasks"]:
            raise EvidenceError("unknown cohort/task")
        method = cohort["methods"].get(row["method"])
        if method is None:
            raise EvidenceError("undeclared method")
        if row["config_id"] != method["config_id"] or row["protocol"] != method["protocol"]:
            raise EvidenceError("mixed configuration or protocol: split into labelled cohorts")
        if row["seed"] not in method["seeds"]:
            raise EvidenceError("undeclared seed")
        key = (row["cohort"], row["task"], row["method"], row["seed"])
        if key in seen:
            raise EvidenceError("duplicate task/method/seed row")
        seen.add(key)
        if row["selection"] != "frozen_run_dev_selected_candidate":
            raise EvidenceError("only frozen runs with dev-selected candidates allowed")
        if row["code_version"] != "unknown" or not row["missing_metadata"]:
            raise EvidenceError("legacy code version must not be inferred")
        if row["data_independence"] != manifest["task_independence"][row["task"]]:
            raise EvidenceError("task independence label mismatch")
        if row["task"] == "legal" and row["data_independence"] != "not_independent_shared_questions":
            raise EvidenceError("legal cannot be independent generalization evidence")
        if row["protocol"] != "legacy_scoring_unknown_code":
            raise EvidenceError("strict and legacy scoring must not be mixed")
        domain = row["task"] in manifest["metric_policy"]["domain_tasks"]
        wanted_metric = ("logged_route_event_not_counterfactual_correctness" if row["extractor"] == "router"
                         else "domain_harmonic_legacy_components" if domain
                         else "legacy_logged_test_or_saved_scores")
        if row["metric_definition"] != wanted_metric:
            raise EvidenceError("metric definition disagrees with task/extractor")
        if row["extractor"] == "pso_final" and domain and row.get("components") != manifest["metric_policy"]["domain_tasks"][row["task"]]:
            raise EvidenceError("domain components disagree with declared metric")
        if row["method"] == "SUMO":
            flag = row["args"].get("no_pair_seed", "absent")
            expected_flag = {"sumo_original": "absent", "sumo2_pair_enabled": False,
                             "sumo2_no_pair": True, "sumo43_no_pair": True}.get(row["cohort"])
            if flag != expected_flag:
                raise EvidenceError("SUMO logged flag disagrees with labelled cohort")
        if row["method"] in {"CMA", "Soup", "Random", "Anchors"} and row["args"].get("method") != row["method"].lower():
            raise EvidenceError("baseline method disagrees with logged arguments")
        dataset = {"kcross": "knowledge_crosswords", "abstainqa": "mmlu"}.get(row["task"], row["task"])
        if row["args"].get("dataset") != dataset:
            raise EvidenceError("dataset disagrees with task")
        sources = row["sources"]
        roles = [s["role"] for s in sources]
        if not sources or len(set(roles)) != len(roles):
            raise EvidenceError("missing or duplicate source roles")
        for src in sources:
            safe_path(Path("/manifest-root"), src["path"])
            if not re.fullmatch("[0-9a-f]{64}", src["sha256"]):
                raise EvidenceError("missing source fingerprint")
            if "line" in src and (type(src["line"]) is not int or src["line"] < 1):
                raise EvidenceError("source line must be positive")
    expected = {(c["id"], t, m, s) for c in cohorts for t in c["tasks"]
                for m, spec in c["methods"].items() for s in spec["seeds"]}
    if seen != expected:
        raise EvidenceError("cohort matrix incomplete")
    return manifest


def _read_sources(root, row):
    out = {}
    for source in row["sources"]:
        path = safe_path(root, source["path"])
        if path.stat().st_size > 32 * 1024 * 1024:
            raise EvidenceError("source too large")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise EvidenceError("source changed: " + source["path"])
        text = raw.decode("utf-8")
        if "line" in source:
            lines = text.splitlines()
            if source["line"] > len(lines):
                raise EvidenceError("missing source line")
            text = lines[source["line"] - 1]
        out[source["role"]] = text
    return out


def extract_row(root, row):
    sources = _read_sources(root, row)
    args_text = sources["args"].split("args: ", 1)[-1]
    args = literal_mapping(args_text)
    if args != row["args"]:
        raise EvidenceError("argument snapshot mismatch")
    if args.get("seed") is not None and args["seed"] != row["seed"]:
        raise EvidenceError("seed mismatch")
    if args.get("fake_eval", False):
        raise EvidenceError("fake evaluations cannot be evidence")
    kind = row["extractor"]
    result = {"id": row["id"], "task": row["task"], "method": row["method"],
              "seed": row["seed"], "cohort": row["cohort"], "config_id": row["config_id"],
              "data_independence": row["data_independence"], "sources": row["sources"],
              "protocol": row["protocol"], "test": None, "dev": None}
    if kind == "final_fields":
        fields = dict(re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", sources["metrics"].split("final metrics: ", 1)[1]))
        result.update(test=score(float(fields["ending_best_test"])),
                      dev=score(float(fields["ending_best_dev"])))
        result["logged_cost"] = {k: int(fields[k]) for k in
                                 ["full_evals", "coarse_evals", "holdout_evals", "equiv_total"] if k in fields}
    elif kind in ("pso_final", "pso_scores"):
        scratch = json.loads(sources["scratchpad"])
        index, dev = stable_particle(scratch)
        if index != row["selected_particle"]:
            raise EvidenceError("dev-selected particle mismatch")
        result.update(selected_particle=index, dev=dev)
        if kind == "pso_final":
            fields = literal_mapping(sources["metrics"].split("final metrics for test: ", 1)[1])
            if fields["ending_best_particle_on_validation"] != index or not math.isclose(
                    score(fields["ending_best_validation_utility"]), dev, abs_tol=1e-12):
                raise EvidenceError("legacy selection disagrees with scratchpad")
            result["legacy_reported_test"] = score(fields["ending_best_single_test_accuracy"])
            components = row.get("components", [])
            if components:
                result["test_components"] = {c: score(fields["ending_best_single_test_" + c]) for c in components}
                result["test"] = harmonic(list(result["test_components"].values()))
            else:
                result["test"] = result["legacy_reported_test"]
        else:
            selected_path = next(s["path"] for s in row["sources"] if s["role"] == "scores")
            if not selected_path.endswith(f"/particle_{index}/personal_best/scores.json"):
                raise EvidenceError("scores not from selected particle")
            values = json.loads(sources["scores"])
            if not isinstance(values, list) or not values:
                raise EvidenceError("empty/non-list selected scores")
            result["test"] = sum(map(score, values)) / len(values)
            result["test_count"] = len(values)
    elif kind == "router":
        decision = re.search(r"\b(KEEP|SWITCH)\b", sources["decision"])
        if not decision:
            raise EvidenceError("no routing event")
        result["decision"] = decision.group(1)
        result["counterfactual_accuracy"] = None
    else:
        raise EvidenceError("unsupported extractor")
    if row["metric_definition"] == "domain_harmonic_legacy_components" and kind == "pso_final" and len(row.get("components", [])) != 2:
        raise EvidenceError("missing domain components")
    for key, expected in row["expected"].items():
        observed = result.get(key)
        if isinstance(expected, (float, int)):
            if observed is None or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
                raise EvidenceError("reconstruction mismatch for " + row["id"] + ": " + key)
        elif observed != expected:
            raise EvidenceError("reconstruction mismatch for " + row["id"] + ": " + key)
    return result


def build_report(manifest, root):
    validate_manifest(manifest)
    root = Path(root)
    audits = []
    for entry in manifest.get("data_audit", []):
        contents = _read_sources(root, {"sources": [entry["source"]]})
        data = json.loads(contents[entry["source"]["role"]])
        dev, test = data["dev"], data["test"]
        dq, tq = {x["question"] for x in dev}, {x["question"] for x in test}
        dr = {json.dumps(x, sort_keys=True) for x in dev}
        tr = {json.dumps(x, sort_keys=True) for x in test}
        actual = {"dev_count": len(dev), "test_count": len(test),
                  "unique_dev_questions": len(dq), "unique_test_questions": len(tq),
                  "shared_unique_questions": len(dq & tq), "identical_full_records": len(dr & tr),
                  "dev_test_equal": dev == test}
        if any(entry.get(k) != v for k, v in actual.items()):
            raise EvidenceError("current data audit mismatch")
        audits.append(entry)
    supplemental = []
    for entry in manifest.get("supplemental_evidence", []):
        contents = _read_sources(root, entry)
        item = dict(entry)
        if "aggregate" in contents:
            item["values"] = json.loads(contents["aggregate"])
        supplemental.append(item)
    if "expert_pool" in manifest:
        _read_sources(root, {"sources": [manifest["expert_pool"]["source"]]})
    rows = [extract_row(root, row) for row in manifest["runs"]]
    tables = {}
    for cohort in manifest["cohorts"]:
        table = {"tasks": cohort["tasks"], "methods": {}, "scope": cohort["scope"]}
        for method in cohort["methods"]:
            selected = [r for r in rows if r["cohort"] == cohort["id"] and r["method"] == method]
            per_task = {}
            for task in cohort["tasks"]:
                scores = [r["test"] for r in selected if r["task"] == task]
                per_task[task] = None if any(v is None for v in scores) else sum(scores) / len(scores)
            values = list(per_task.values())
            table["methods"][method] = {"per_task": per_task,
                "macro_mean": None if any(v is None for v in values) else sum(values) / len(values),
                "run_ids": [r["id"] for r in selected]}
        tables[cohort["id"]] = table
    return {"evidence_track": manifest["evidence_track"],
            "selection_policy": manifest["selection_policy"],
            "strict_rescoring": False, "tables": tables, "rows": rows,
            "data_audit": audits, "supplemental_evidence": supplemental}


def markdown_table(report, cohort="main"):
    table = report["tables"][cohort]
    methods = table["methods"]
    lines = ["| task | " + " | ".join(methods) + " | independence |",
             "|---|" + "---|" * (len(methods) + 1)]
    for task in table["tasks"]:
        labels = {r["data_independence"] for r in report["rows"] if r["cohort"] == cohort and r["task"] == task}
        cells = ["—" if spec["per_task"][task] is None else f'{spec["per_task"][task]:.6f}' for spec in methods.values()]
        lines.append("| " + task + " | " + " | ".join(cells) + " | " + ", ".join(sorted(labels)) + " |")
    cells = ["—" if spec["macro_mean"] is None else f'{spec["macro_mean"]:.7f}' for spec in methods.values()]
    lines.append("| macro mean | " + " | ".join(cells) + " | descriptive only |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/legacy_manifest.json")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="new JSON path only; never an existing file or inside search/")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_report(manifest, args.root)
    if args.output:
        target = args.output.resolve()
        root = args.root.resolve()
        if target == root / "search" or root / "search" in target.parents:
            raise EvidenceError("never write inside legacy search/")
        protected = {safe_path(root, s["path"]) for row in manifest["runs"] for s in row["sources"]}
        protected.add(args.manifest.resolve())
        if target in protected:
            raise EvidenceError("output would replace evidence")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(markdown_table(report))


if __name__ == "__main__":
    main()
