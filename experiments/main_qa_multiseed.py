"""Repeat native main-table QA with explicit seeds and frozen retrieval inputs.

The user-provided table is checked against historical predictions, not used to
generate scores. Full new predictions are preserved in lossless bounded archives.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import mean, stdev
import subprocess
from tempfile import TemporaryDirectory

from experiments.component_artifacts import publish_tree, restore_tree, verify_archive, sha256
from experiments.encoder_robustness import json_save

BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")
REFERENCE = BASE / "optimization_simplified_amor_seed42_20260929"
FIXTURE = Path(__file__).with_name("fixtures") / "main_table_seed42_20261003.json"
SEEDS = (42, 52, 62)
BUDGET = 2 * 1024**3
NATIVE_ROOTS = {
    "Qwen/Qwen3.5-4B": BASE / "optimization_native_readers_seed42_20260920/main/Qwen_Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": BASE / "optimization_native_readers_seed42_20260920/main/Qwen_Qwen3.5-9B",
    "google/gemma-3-4b-it": BASE / "optimization_native_readers_seed42_20260920/main/google_gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct": BASE / "optimization_context_reranking_seed42_20260920/one_shot/main/meta-llama_Llama-3.1-8B-Instruct",
}
AMOR_METHODS = {"full", "without_propagation", "without_context_augmentation"}


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def comparable_reader(metadata):
    return {key: value for key, value in metadata.items() if key not in {"gpu", "seed"}}


def code_hashes():
    import optimization.ircot as native
    import experiments.runner as evaluator
    return {str(path): sha256(path) for path in
            (Path(__file__), Path(native.__file__), Path(evaluator.__file__),
             Path(__file__).with_name("component_artifacts.py"), FIXTURE)}


def check_scope(table):
    tasks, methods, models = table["tasks"], table["methods"], table["scores_percent"]
    if len(tasks) != 6 or len(set(tasks)) != 6 or len(methods) != 10 or len(set(methods)) != 10:
        raise ValueError("The complete six-task, ten-method main table is required")
    if set(models) != set(NATIVE_ROOTS) or table["seed"] != 42:
        raise ValueError("Historical reader/seed scope differs")
    for model, scores in models.items():
        if len(scores) != len(methods) or any(len(row) != len(tasks) for row in scores):
            raise ValueError(f"Incomplete displayed scores: {model}")
        if any(not math.isfinite(x) or not 0 <= x <= 100 for row in scores for x in row):
            raise ValueError("Invalid displayed score")


def verify_usage(cost, case_id, generation, seed, *, new_run):
    if cost["case_id"] != case_id or cost["seed"] != seed:
        raise ValueError("Wrong QA identity or seed")
    if cost["generation_settings"] != generation:
        raise ValueError("Native generation settings changed")
    for key in ("input_tokens", "output_tokens"):
        if not isinstance(cost[key], int) or cost[key] < 0:
            raise ValueError("Invalid token count")
    if cost["input_tokens"] + generation["max_tokens"] > 32768:
        raise ValueError("Reader context overflow")
    if new_run and (cost.get("sampling_seed") != seed or not cost.get("raw_answer")):
        raise ValueError("Missing actual sampling seed or raw completion")


def audit(directory, source, model, task, seed, *, smoke=False, new_run=False):
    from experiments.runner import (_read_retrieval_records, _score, _official_generation,
                                    _answer_prompt, _locomo_category_5_answer)
    from optimization.report_results import TASK_METRICS
    directory, source = Path(directory), Path(source)
    rows = list(_read_retrieval_records(directory / "retrieval.jsonl"))
    expected = list(_read_retrieval_records(source))
    if smoke:
        categories = dict.fromkeys(row.case.category for row in expected)
        expected = [next(row for row in expected if row.case.category == c) for c in categories]
    count, metric = TASK_METRICS[task]
    if not smoke and len(expected) != count:
        raise ValueError("Wrong complete task population")
    marker = json.loads((directory / "qa_complete.json").read_text())
    if not marker["complete"] or bool(marker["pilot"]) != smoke:
        raise ValueError("Missing QA completion or wrong smoke/full status")
    if not marker["native_evaluator"] or not marker["scores_recomputed"]:
        raise ValueError("Non-native QA result")
    evaluation = directory / "evaluations" / model.replace("/", "_")
    predictions, usage = records(evaluation / "predictions.jsonl"), records(evaluation / "qa_usage.jsonl")
    if not len(rows) == len(expected) == len(predictions) == len(usage) == marker["questions"]:
        raise ValueError("Incomplete QA coverage")
    if new_run and marker.get("seed") != seed:
        raise ValueError("Completion seed differs")
    identities, scores = set(), []
    rng = random.Random(seed)
    for row, original, prediction, cost in zip(rows, expected, predictions, usage, strict=True):
        key = row.group_id, row.case.case_id
        if key in identities or key != (prediction["group_id"], prediction["case_id"]):
            raise ValueError("Changed or duplicated case identity")
        identities.add(key)
        if (row.group_id, row.case, row.top_k, row.retrieved) != (
                original.group_id, original.case, original.top_k, original.retrieved):
            raise ValueError("Input differs from its frozen retrieval source")
        if [item.text for item in row.retrieved] != [item["text"] for item in prediction["retrieved"]]:
            raise ValueError("QA prediction used a different context")
        actual = _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
        if actual != prediction["metrics"]:
            raise ValueError("Saved metrics differ from native recomputation")
        score = actual[metric]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid native score")
        scores.append(score)
        verify_usage(cost, row.case.case_id, _official_generation(row.case), seed, new_run=new_run)
        _, answer_key = _answer_prompt(row.case, row.retrieved, rng)
        if new_run:
            answer = cost["raw_answer"]
            if row.case.metric == "hipporag" and "Answer:" in answer:
                answer = answer.split("Answer:", 1)[1].strip()
            if answer_key is not None:
                answer = _locomo_category_5_answer(answer, answer_key)
            if answer != prediction["prediction"]:
                raise ValueError("Native answer postprocessing or seeded option mapping differs")
    score = mean(scores)
    summary = json.loads((evaluation / "summary.json").read_text())
    if not math.isclose(score, summary[metric], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Summary score differs")
    if not smoke and not math.isclose(score, marker["score"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Completion score differs")
    totals = {k: sum(row[k] for row in usage) for k in ("input_tokens", "output_tokens")}
    if any(totals[k] != marker[k] for k in totals):
        raise ValueError("Token totals differ")
    paths = [directory / "retrieval.jsonl", directory / "qa_complete.json",
             evaluation / "predictions.jsonl", evaluation / "qa_usage.jsonl", evaluation / "summary.json"]
    return dict(questions=len(rows), metric=metric, score=100 * score,
                seed=seed, smoke=smoke, native_scores_recomputed=True,
                hashes={str(p.relative_to(directory)): sha256(p) for p in paths}, **totals)


def prepare(root):
    from optimization.ircot import MODELS
    from experiments.runner import _read_retrieval_records
    table = json.loads(FIXTURE.read_text())
    check_scope(table)
    with (REFERENCE / "qa_results.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    entries, identities = [], {}
    for model, displayed in table["scores_percent"].items():
        reader_paths = [NATIVE_ROOTS[model] / "reader.json"]
        reader_paths.extend(sorted((REFERENCE / "one_shot" / model.replace("/", "_")).glob("reader_*.json")))
        if len(reader_paths) < 2:
            raise ValueError("Missing historical native and AMOR reader metadata")
        metadata = [json.loads(p.read_text()) for p in reader_paths]
        if any(m["seed"] != 42 or m["revision"] != MODELS[model] or
               comparable_reader(m) != comparable_reader(metadata[0]) for m in metadata):
            raise ValueError(f"Historical reader runtimes disagree: {model}")
        for method, method_scores in zip(table["methods"], displayed, strict=True):
            for task, expected_score in zip(table["tasks"], method_scores, strict=True):
                if method in AMOR_METHODS:
                    match, = [r for r in manifest if r["model"] == model and r["task"] == task
                              and r["setting"] == "one_shot" and r["variant"] == method]
                    source = Path(match["source"])
                else:
                    source = NATIVE_ROOTS[model] / method
                    if model == "meta-llama/Llama-3.1-8B-Instruct":
                        source = source / "original"
                    source = source / task
                result = audit(source, source / "retrieval.jsonl", model, task, 42)
                if abs(result["score"] - expected_score) > .005000001:
                    raise ValueError(f"Main table mismatch: {model}/{method}/{task}: {result['score']} != {expected_score}")
                rows = list(_read_retrieval_records(source / "retrieval.jsonl"))
                current = [(r.group_id, r.case) for r in rows]
                if task in identities and current != identities[task]:
                    raise ValueError(f"Task identity/order differs across main-table configurations: {task}")
                identities[task] = current
                entry = dict(model=model, method=method, task=task, directory=str(source),
                             retrieval_sha256=sha256(source / "retrieval.jsonl"), reference=result,
                             reader=comparable_reader(metadata[0]),
                             reader_metadata_hashes={str(p): sha256(p) for p in reader_paths})
                entries.append(entry)
                print("reference_verified", model, method, task, result["score"], flush=True)
    if len(entries) != 240 or sum(e["reference"]["questions"] for e in entries) != 135440:
        raise ValueError("Main table coverage incomplete")
    protocol = dict(experiment="main_table_fixed_context_qa_seeds", seeds=SEEDS,
                    tasks=table["tasks"], methods=table["methods"], models=MODELS,
                    questions_per_seed=135440, conditions_per_seed=240,
                    archive_budget_bytes=BUDGET, fixed_retrieval=True,
                    decoding="native unchanged; 2Wiki greedy temperature zero",
                    scope="QA randomness, including native LoCoMo option order; not construction/retrieval variance",
                    code_sha256=code_hashes())
    json_save(root / "protocol.json", protocol)
    json_save(root / "inputs.json", dict(complete=True, entries=entries))
    print("main_qa_preflight_complete", len(entries), flush=True)


def storage_gate(root, additional_bytes=0):
    if not isinstance(additional_bytes, int) or not 0 <= additional_bytes <= BUDGET:
        raise ValueError("Invalid archive publication size")
    if not root.resolve().is_relative_to(Path("/oscar/scratch/zliu328/agent-memory-outputs").resolve()):
        raise ValueError("This bounded run must be stored in the declared scratch output area")
    result = subprocess.run(["checkquota"], capture_output=True, text=True, check=True)
    rows = [line.split() for line in result.stdout.splitlines() if "/oscar/scratch" in line]
    if len(rows) != 1 or rows[0][1] != "/oscar/scratch":
        raise ValueError("Unrecognized scratch quota format")
    used, hard = float(rows[0][2]), float(rows[0][5])
    if used + BUDGET / 10**9 + 1 > hard:
        raise RuntimeError("Insufficient scratch hard-quota margin")
    if "SOFT_EXCEEDED" in rows[0]:
        grace = rows[0][rows[0].index("SOFT_EXCEEDED") + 1:]
        if len(grace) != 2 or not grace[0].isdigit() or int(grace[0]) < 1:
            raise RuntimeError("Scratch soft-quota grace is less than one day or unreadable")
    print("storage_gate", json.dumps(dict(used_g=used, hard_limit_g=hard,
                                         archive_budget_bytes=BUDGET)), flush=True)


def archive_name(entry, seed, smoke):
    return "_".join((str(seed), entry["model"].replace("/", "_"), entry["method"],
                     entry["task"], "smoke" if smoke else "full"))


def validate_inputs(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["code_sha256"] != code_hashes() or protocol["seeds"] != list(SEEDS):
        raise ValueError("Frozen run implementation or seeds changed")
    inputs = json.loads((root / "inputs.json").read_text())
    if not inputs["complete"] or len(inputs["entries"]) != 240:
        raise ValueError("Missing complete reference preflight")
    return protocol, inputs["entries"]


def run(root, model, seed):
    from optimization.ircot import Reader, BATCH_SIZE, evaluate_task
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation
    from experiments.ablate_components import save_rows
    _, entries = validate_inputs(root)
    if model not in NATIVE_ROOTS or seed not in SEEDS[1:]:
        raise ValueError("Only the approved new QA seeds/readers can run")
    storage_gate(root)
    selected = [entry for entry in entries if entry["model"] == model]
    archives = root / "archives"
    reader = Reader(model, seed=seed)
    try:
        if any(comparable_reader(reader.metadata) != e["reader"] for e in selected):
            raise ValueError("New reader runtime differs from historical seed 42")
        for smoke in (True, False):
            for entry in selected:
                source = Path(entry["directory"]) / "retrieval.jsonl"
                if sha256(source) != entry["retrieval_sha256"]:
                    raise ValueError("Frozen retrieval input changed")
                name = archive_name(entry, seed, smoke)
                with TemporaryDirectory(prefix="amor-seed-", dir=os.environ["TMPDIR"]) as temporary:
                    directory = Path(temporary) / "result"
                    if (archives / (name + ".manifest.json")).exists():
                        restore_tree(archives, name, directory)
                        audit(directory, source, model, entry["task"], seed, smoke=smoke, new_run=True)
                        continue
                    directory.mkdir()
                    rows = list(_read_retrieval_records(source))
                    if smoke:
                        categories = dict.fromkeys(row.case.category for row in rows)
                        rows = [next(row for row in rows if row.case.category == c) for c in categories]
                        save_rows(directory / "retrieval.jsonl", rows)
                    else:
                        shutil.copy2(source, directory / "retrieval.jsonl")
                    json_save(directory / "reader.json", reader.metadata)
                    json_save(directory / "source.json", entry)
                    shutil.copy2(root / "protocol.json", directory / "protocol.json")
                    rng = random.Random(seed)
                    requests = [dict(phase="answer", messages=_answer_prompt(row.case, row.retrieved, rng)[0],
                                     generation=_official_generation(row.case)) for row in rows]
                    for start in range(0, len(requests), BATCH_SIZE):
                        if reader.request(phase="check_context", items=requests[start:start + BATCH_SIZE])["overflow"]:
                            raise ValueError("QA context exceeds native limit")
                    evaluate_task(directory, entry["task"], reader, model, smoke, seed=seed)
                    result = audit(directory, source, model, entry["task"], seed, smoke=smoke, new_run=True)
                    json_save(directory / "verified.json", result)
                    published = publish_tree(directory, ["retrieval.jsonl", "reader.json", "source.json",
                                             "protocol.json", "evaluations", "qa_complete.json", "verified.json"],
                                             archives, name, budget_bytes=BUDGET,
                                             quota_check=lambda size: storage_gate(root, size))
                    restored = Path(temporary) / "readback"
                    restore_tree(archives, name, restored)
                    audit(restored, source, model, entry["task"], seed, smoke=smoke, new_run=True)
                    print("qa_archived", name, result["score"], published["archive_bytes"], flush=True)
    finally:
        reader.close()


def report(root):
    _, entries = validate_inputs(root)
    rows, audited = [], []
    for entry in entries:
        source = Path(entry["directory"]) / "retrieval.jsonl"
        if sha256(source) != entry["retrieval_sha256"]:
            raise ValueError("Reference changed before reporting")
        scores = []
        for seed in SEEDS:
            if seed == 42:
                result = audit(Path(entry["directory"]), source, entry["model"], entry["task"], seed)
            else:
                name = archive_name(entry, seed, False)
                with TemporaryDirectory(prefix="amor-seed-report-", dir=os.environ["TMPDIR"]) as temporary:
                    directory = Path(temporary)
                    restore_tree(root / "archives", name, directory)
                    if json.loads((directory / "source.json").read_text()) != entry:
                        raise ValueError("Archived source identity differs")
                    reader = json.loads((directory / "reader.json").read_text())
                    if reader["seed"] != seed or comparable_reader(reader) != entry["reader"]:
                        raise ValueError("Archived reader configuration differs")
                    if sha256(directory / "protocol.json") != sha256(root / "protocol.json"):
                        raise ValueError("Archived protocol differs")
                    result = audit(directory, source, entry["model"], entry["task"], seed, new_run=True)
                    if result != json.loads((directory / "verified.json").read_text()):
                        raise ValueError("Independent archive audit differs")
                    audited.append(dict(name=name, archive=verify_archive(root / "archives", name)["archive_sha256"]))
            rows.append(dict(model=entry["model"], method=entry["method"], task=entry["task"],
                             seed=seed, score=result["score"], questions=result["questions"], metric=result["metric"]))
            scores.append(result["score"])
        print("three_seeds_verified", entry["model"], entry["method"], entry["task"], scores, flush=True)
    summary = []
    for entry in entries:
        values = [r["score"] for r in rows if (r["model"], r["method"], r["task"]) ==
                  (entry["model"], entry["method"], entry["task"])]
        summary.append(dict(model=entry["model"], method=entry["method"], task=entry["task"],
                            mean=mean(values), sample_sd=stdev(values), seeds=len(values)))
    if len(rows) != 720 or sum(r["questions"] for r in rows) != 406320 or len(audited) != 480:
        raise ValueError("Incomplete three-seed report")
    target = root / "report"
    target.mkdir(exist_ok=True)
    for name, values in (("seed_scores.csv", rows), ("mean_sd.csv", summary)):
        with (target / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    json_save(target / "audit.json", dict(archives=audited, native_scores_recomputed=True,
                                        seeds=SEEDS, all_720_conditions=True))
    json_save(target / "complete.json", dict(complete=True, conditions=720, answers=406320,
        hashes={name: sha256(target / name) for name in ("seed_scores.csv", "mean_sd.csv", "audit.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qa", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=tuple(NATIVE_ROOTS))
    parser.add_argument("--seed", choices=SEEDS[1:], type=int)
    args = parser.parse_args()
    if args.phase == "qa" and (args.model is None or args.seed is None):
        parser.error("QA requires a declared model and new seed")
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "qa":
        run(args.output, args.model, args.seed)
    else:
        report(args.output)
