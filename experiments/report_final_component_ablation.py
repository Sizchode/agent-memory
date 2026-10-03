"""Independently score all final-graph component results against the fresh reference."""
import argparse
import csv
import json
from pathlib import Path

from experiments.encoder_robustness import digest, json_save
from experiments.final_component_ablation import VARIANTS, TASKS, READERS
from experiments.report_final_parameter_sensitivity import verify_condition, paired_interval


def report(root):
    from experiments.runner import _read_retrieval_records
    protocol = json.loads((root / "protocol.json").read_text())
    reference = Path(protocol["reference_qa"])
    reference_protocol = json.loads((reference / "protocol.json").read_text())
    if protocol["source_protocol_sha256"] != reference_protocol["source_protocol_sha256"]:
        raise ValueError("Reference and treatment are from different AMOR versions")
    if not reference_protocol["fresh_qa_reference"]:
        raise ValueError("Reference QA was not freshly generated")
    rows, audits = [], []
    for model in READERS:
        slug = model.replace("/", "_")
        for task in TASKS:
            score, base, audit = verify_condition(reference, model, "reference", task)
            audits.append(audit)
            ref_dir = reference / "qa" / slug / "reference" / task
            ref_rows = list(_read_retrieval_records(ref_dir / "retrieval.jsonl"))
            replay = list(_read_retrieval_records(root / "inputs/reference" / task / "retrieval.jsonl"))
            for actual, expected in zip(replay, ref_rows, strict=True):
                if (actual.group_id, actual.case) != (expected.group_id, expected.case):
                    raise ValueError("Reference question identity differs")
                if [i.text for i in actual.retrieved] != [i.text for i in expected.retrieved]:
                    raise ValueError("Fresh QA reference did not use the exact replayed context")
            reader_meta = json.loads((ref_dir / "reader.json").read_text())
            rows.append(dict(model=model, task=task, variant="reference", questions=len(base),
                score=score, reference_score=score, difference_pp=0.0, paired_low=0.0, paired_high=0.0))
            for variant in VARIANTS:
                target = root / "qa" / slug / variant / task
                if json.loads((target / "reader.json").read_text()) != reader_meta:
                    raise ValueError("Reader runtime or generation configuration differs")
                mean, values, audit = verify_condition(root, model, variant, task)
                audits.append(audit)
                low, high = paired_interval(base, values, cluster=task == "LoCoMo")
                rows.append(dict(model=model, task=task, variant=variant, questions=len(values),
                    score=mean, reference_score=score, difference_pp=mean-score, paired_low=low, paired_high=high))
    directory = root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(directory / "audit.json", dict(conditions=audits, all_96_treatments_and_24_references=True,
        native_metrics_recomputed=True, exact_fresh_reference_contexts=True,
        reporting="All four components, six tasks and four readers; no cross-task averaging",
        paired_intervals="5000 bootstrap replicates; conversation clusters for LoCoMo, questions otherwise; seed42"))
    json_save(directory / "complete.json", dict(complete=True, conditions=len(rows),
        scores_sha256=digest(directory / "scores.csv"), audit_sha256=digest(directory / "audit.json")))
    print("Verified 96 component conditions and 24 reference conditions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
