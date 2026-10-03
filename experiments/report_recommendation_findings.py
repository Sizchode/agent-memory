"""Audit completed native-reader outputs against their exact intervention inputs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def report(root):
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation, _score
    from optimization.ircot import MODELS, SEED
    import random
    protocol=json.loads((root/"prepared.json").read_text())
    assert protocol["complete"]
    expected={}
    for entry in protocol["entries"]:
        # Resolve within the copied experiment, not a stale absolute CCV path.
        path=root/"inputs"/entry["condition"]/"retrieval.jsonl"
        rows=list(_read_retrieval_records(path))
        assert len(rows)==entry["questions"]
        rng=random.Random(SEED)
        for row in rows:
            messages, answer_key=_answer_prompt(row.case,row.retrieved,rng)
            assert answer_key is None
            key=(entry["condition"],row.case.case_id)
            assert key not in expected
            expected[key]=(row,dict(phase="answer",messages=messages,generation=_official_generation(row.case)))
    summaries, pairs, audits=[],[],[]
    for model, revision in MODELS.items():
        directory=root/"qa"/model.replace("/","_")
        marker=directory/"complete.json"
        if not marker.exists():
            continue
        complete=json.loads(marker.read_text())
        assert complete["complete"] and complete["revision"]==revision and complete["seed"]==SEED
        predictions=lines(directory/"predictions.jsonl")
        assert len(predictions)==complete["requests"]==len(expected)
        observed={}
        conditions=defaultdict(list)
        for prediction in predictions:
            key=(prediction["condition"],prediction["case_id"])
            assert key in expected and key not in observed
            row, request=expected[key]
            assert prediction["request"]==request
            actual=_score(row.case,prediction["prediction"],row.retrieved,row.top_k)
            assert actual==prediction["metrics"]
            metric="answer_f1" if row.case.metric=="hipporag" else "substring_exact_match"
            value=actual[metric]
            assert math.isfinite(value) and 0<=value<=1
            usage=prediction["usage"]
            assert usage["input_tokens"]>0 and usage["output_tokens"]>=0
            observed[key]=dict(score=value,**usage)
            conditions[prediction["condition"]].append(observed[key])
        for condition, values in conditions.items():
            summaries.append(dict(experiment=root.name,model=model,condition=condition,questions=len(values),
                score=100*mean(v["score"] for v in values),
                input_tokens=mean(v["input_tokens"] for v in values),
                output_tokens=mean(v["output_tokens"] for v in values)))
        contrasts=[("amor_k5_augmented","without_recommendation_k15_augmented"),
                   ("amor_k5_central","without_recommendation_k15_central"),
                   ("swap_support","swap_unannotated"),
                   ("swap_support","swap_original"),
                   ("seeds_current_graph_current","seeds_current_graph_all"),
                   ("seeds_current_graph_current","seeds_all_graph_current"),
                   ("seeds_current_graph_current","seeds_all_graph_all"),
                   ("old_routing_supply_death","old_routing_repeat_correction"),
                   ("current_filtered","old_routing_filtered"),
                   ("current_filtered","current_remove_death")]
        for treatment,control in contrasts:
            if treatment not in conditions or control not in conditions:
                continue
            ids={cid for condition,cid in observed if condition==treatment}
            assert ids=={cid for condition,cid in observed if condition==control}
            deltas=[observed[treatment,cid]["score"]-observed[control,cid]["score"] for cid in sorted(ids)]
            pairs.append(dict(experiment=root.name,model=model,treatment=treatment,control=control,
                questions=len(ids),difference_pp=100*mean(deltas),
                higher=sum(d>0 for d in deltas),equal=sum(d==0 for d in deltas),lower=sum(d<0 for d in deltas),
                input_token_difference=mean(observed[treatment,cid]["input_tokens"]-observed[control,cid]["input_tokens"] for cid in ids)))
        audits.append(dict(model=model,requests=len(predictions),exact_prompts=True,native_scores_recomputed=True,
            predictions_sha256=hashlib.sha256((directory/"predictions.jsonl").read_bytes()).hexdigest()))
    assert summaries, "No completed readers to report"
    for name,rows in (("qa_summary.csv",summaries),("qa_pairs.csv",pairs)):
        if rows:
            with (root/name).open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/"qa_audit.json").write_text(json.dumps(dict(readers=audits,all_four_readers=len(audits)==len(MODELS)),indent=2)+"\n")
    for row in summaries:
        print(json.dumps(row))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    report(parser.parse_args().root)
