"""Source-verified, explicitly exemplary interventions for the FC-MH case.

This is a declarative experiment on a previously selected benchmark example,
not a deployed retrieval rule or a claim about case prevalence. All inserted
facts and removed spans must occur verbatim in the frozen source artifacts.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def remove_rendered_span(record, span, expected):
    result = deepcopy(record)
    occurrences = sum(item["text"].count(span) for item in result["retrieved"])
    if occurrences != expected:
        raise ValueError(f"Expected {expected} exact rendered occurrences, found {occurrences}")
    for item in result["retrieved"]:
        item["text"] = item["text"].replace(span, "")
    assert not any(span in item["text"] for item in result["retrieved"])
    return result


def exact_fact(record, text):
    matches = [item for item in record["retrieved"] if item["text"] == text]
    if len(matches) != 1:
        raise ValueError("Intervention donor must be one verified existing fact entry")
    return deepcopy(matches[0])


def append_fact(record, donor, label):
    result = deepcopy(record)
    copied = deepcopy(donor)
    copied["metadata"]["diagnostic_intervention"] = label
    result["retrieved"].append(copied)
    return result


def prepare(source, output):
    case_id = "factconsolidation_mh_262k_no45"
    task = "FactConsolidation-MH"
    source_paths = {name: source/"inputs"/task/name/"retrieval.jsonl"
                    for name in ("current", "old_authorship")}
    records = {}
    for name, path in source_paths.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["case"]["case_id"] == case_id
        records[name] = rows[0]
    obsolete = "4873. The author of Dubliners is James Joyce."
    death_span = "13652. George Eliot died in the city of London."
    correction = exact_fact(records["current"], "(Dubliners, author, George Eliot)")
    death = exact_fact(records["current"], "(George Eliot, died in, London)")
    assert exact_fact(records["old_authorship"], correction["text"])["text"] == correction["text"]
    current = remove_rendered_span(records["current"], obsolete, 1)
    old = remove_rendered_span(records["old_authorship"], obsolete, 1)
    assert sum(death_span in item["text"] for item in current["retrieved"]) == 1
    assert not any("George Eliot died in" in item["text"] or item["text"] == death["text"]
                   for item in old["retrieved"])
    deprived = remove_rendered_span(current, death_span, 1)
    deprived["retrieved"] = [item for item in deprived["retrieved"] if item["text"] != death["text"]]
    assert len(deprived["retrieved"]) == len(current["retrieved"]) - 1
    conditions = dict(current_filtered=current, old_routing_filtered=old,
        old_routing_repeat_correction=append_fact(old, correction, "repeat existing current correction"),
        old_routing_supply_death=append_fact(old, death, "oracle restore missing downstream fact"),
        current_remove_death=deprived,
        current_remove_death_restore=deepcopy(current))
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for condition, row in conditions.items():
        assert row["case"] == records["current"]["case"]
        assert not any(obsolete in item["text"] for item in row["retrieved"])
        assert any(item["text"] == correction["text"] for item in row["retrieved"])
        path = output/"inputs"/condition/"retrieval.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row, ensure_ascii=False)+"\n")
        entries.append(dict(task=task, condition=condition, questions=1, path=str(path)))
    design = dict(case_id=case_id, scope="Previously chosen illustrative case, not a population estimate",
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths.values()},
        obsolete_span=obsolete, downstream_span=death_span,
        fixed="Within old-routing rescue arms: original five sources, current correction, other facts, reader prompt",
        filter="Exact oracle obsolete authorship-span removal from rendered input; metadata retains provenance",
        comparison="One appended current correction versus one appended missing downstream fact; token counts measured, not matched",
        routing_limit="Existing old accepted tuple also changes relation wording and argument direction",
        restored="current_remove_death_restore exactly repeats current_filtered input",
        new_retrieval=False, new_extraction=False)
    (output/"design.json").write_text(json.dumps(design,indent=2)+"\n")
    (output/"prepared.json").write_text(json.dumps(dict(complete=True, entries=entries),indent=2)+"\n")
    print(json.dumps(design,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source,args.output)
