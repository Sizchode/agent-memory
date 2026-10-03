"""Recompute native encoder QA metrics and verify every input and prediction."""
import argparse
import csv
import json
import math
from pathlib import Path

from experiments.encoder_robustness import READERS, digest, json_save, protocol_tasks


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_predictions(rows, expected, predictions, usage, *, score_fn,
                      generation_fn, count, metric):
    if not len(rows) == len(expected) == len(predictions) == len(usage) == count:
        raise ValueError('Incomplete question coverage')
    values, input_tokens, output_tokens = {}, 0, 0
    for row, reference, prediction, cost in zip(rows, expected, predictions, usage, strict=True):
        key = (row.group_id, row.case.case_id)
        if key in values or key != (prediction['group_id'], prediction['case_id']):
            raise ValueError('Duplicated or changed question identity')
        if (row.group_id, row.case) != (reference.group_id, reference.case):
            raise ValueError('Evaluation question differs from prepared input')
        source_keys = []
        for record in (row, reference):
            keys = [item.metadata['source_passage'] for item in record.retrieved
                    if item.metadata['context_representation'] == 'original_source_and_frozen_window']
            if len(keys) != reference.top_k or len(set(keys)) != len(keys):
                raise ValueError('Central passage budget or identities differ')
            if any(item.metadata['context_representation'] not in
                   {'original_source_and_frozen_window', 'retrieved_triples'}
                   for item in record.retrieved):
                raise ValueError('Unknown context-entry representation')
            source_keys.append(keys)
        if source_keys[0] != source_keys[1]:
            raise ValueError('Central passage identities differ')
        texts = [item.text for item in row.retrieved]
        if texts != [item.text for item in reference.retrieved]:
            raise ValueError('QA context differs from prepared input')
        if texts != [item['text'] for item in prediction['retrieved']]:
            raise ValueError('Prediction context differs from QA input')
        actual = score_fn(row.case, prediction['prediction'], row.retrieved, row.top_k)
        if actual != prediction['metrics']:
            raise ValueError('Saved scores disagree with native evaluator')
        value = actual[metric]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid native score')
        values[key] = value
        if cost['case_id'] != row.case.case_id or cost['seed'] != 42:
            raise ValueError('Generation identity or seed changed')
        if cost['generation_settings'] != generation_fn(row.case):
            raise ValueError('Task-specific generation settings changed')
        for field in ('input_tokens', 'output_tokens'):
            if not isinstance(cost[field], int) or cost[field] < 0:
                raise ValueError('Invalid token count')
        if cost['input_tokens'] + cost['generation_settings']['max_tokens'] > 32768:
            raise ValueError('Reader context limit exceeded')
        input_tokens += cost['input_tokens']
        output_tokens += cost['output_tokens']
    return values, input_tokens, output_tokens


def audit_condition(directory, prepared_path, model, task):
    from experiments.runner import _read_retrieval_records, _score, _official_generation
    from optimization.report_results import TASK_METRICS
    count, metric = TASK_METRICS[task]
    marker = json.loads((directory / 'qa_complete.json').read_text())
    if not marker['complete'] or marker['pilot'] or marker['questions'] != count:
        raise ValueError('Missing full native QA completion')
    if not marker['native_evaluator'] or not marker['scores_recomputed']:
        raise ValueError('Non-native evaluation is not reportable')
    output = directory / 'evaluations' / model.replace('/', '_')
    rows = list(_read_retrieval_records(directory / 'retrieval.jsonl'))
    expected = list(_read_retrieval_records(prepared_path))
    predictions = read_jsonl(output / 'predictions.jsonl')
    usage = read_jsonl(output / 'qa_usage.jsonl')
    values, inputs, outputs = check_predictions(rows, expected, predictions, usage,
        score_fn=_score, generation_fn=_official_generation, count=count, metric=metric)
    mean = sum(values.values()) / count
    summary = json.loads((output / 'summary.json').read_text())
    if not math.isclose(mean, marker['score'], abs_tol=1e-12, rel_tol=0):
        raise ValueError('Completion score disagrees with recomputed scores')
    if not math.isclose(mean, summary[metric], abs_tol=1e-12, rel_tol=0):
        raise ValueError('Summary disagrees with recomputed scores')
    if (inputs, outputs) != (marker['input_tokens'], marker['output_tokens']):
        raise ValueError('Token totals disagree')
    paths = [directory / 'qa_complete.json', directory / 'retrieval.jsonl', prepared_path,
             output / 'predictions.jsonl', output / 'qa_usage.jsonl', output / 'summary.json']
    audit = dict(directory=str(directory), questions=count, metric=metric,
                 native_scores_recomputed=True, exact_contexts=True,
                 recorded_top_k=sorted({r.top_k for r in rows}),
                 prepared_passage_cutoffs=sorted({r.top_k for r in expected}),
                 cutoff_policy='Verify central-source metadata and complete rendered context; historical top_k counts rendered entries. Rescore each file using its original top_k.',
                 hashes={str(p): digest(p) for p in paths})
    result = dict(model=model, task=task, questions=count, metric=metric,
                  score=100 * mean, mean_input_tokens=inputs / count,
                  mean_output_tokens=outputs / count)
    return result, values, audit


def reader_metadata(reference, model):
    paths = sorted((reference / 'one_shot' / model.replace('/', '_')).glob('reader_*.json'))
    if not paths:
        raise ValueError('Missing historical reader metadata')
    records = [json.loads(p.read_text()) for p in paths]
    # GPU allocation identity is not a generation setting; preserve it in the audit.
    comparable = [{k: v for k, v in record.items() if k != 'gpu'} for record in records]
    if any(record != comparable[0] for record in comparable):
        raise ValueError('Historical reader configurations disagree')
    return comparable[0], {str(p): digest(p) for p in paths}


def report(root, reference_only=False):
    from experiments.report_final_parameter_sensitivity import paired_interval
    protocol = json.loads((root / 'protocol.json').read_text())
    tasks = protocol_tasks(root)
    if protocol['readers'] != list(READERS):
        raise ValueError('Frozen experiment scope changed')
    reference = Path(protocol['source_results'])
    with (reference / 'qa_results.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    rows, audits, contrasts = [], [], []
    encoders = ('qwen',) if reference_only else ('qwen', 'bge', 'nv')
    for encoder in encoders:
        prepared = json.loads((root / 'qa_inputs' / encoder / 'prepared.json').read_text())
        if not prepared['complete']:
            raise ValueError('Retrieval not complete')
        entries = {(e['task'], e['method']): e for e in prepared['entries']}
        if len(entries) != len(prepared['entries']) or set(entries) != {
                (t, m) for t in tasks for m in ('amor', 'without_recommendation')}:
            raise ValueError('Incomplete retrieval conditions')
        for model in READERS:
            metadata, metadata_hashes = reader_metadata(reference, model)
            for task in tasks:
                scores = {}
                for method, variant in (('amor', 'full'), ('without_recommendation', 'without_propagation')):
                    prepared_path = Path(entries[task, method]['path'])
                    if encoder == 'qwen':
                        match, = [r for r in manifest if r['model'] == model and r['task'] == task
                                  and r['setting'] == 'one_shot' and r['variant'] == variant]
                        directory = Path(match['source'])
                    else:
                        directory = root / 'qa_runs' / encoder / model.replace('/', '_') / method / task
                        current = json.loads((directory / 'reader.json').read_text())
                        if {k: v for k, v in current.items() if k != 'gpu'} != metadata:
                            raise ValueError('Reader runtime or generation defaults changed')
                        metadata_hashes = dict(metadata_hashes, **{str(directory / 'reader.json'): digest(directory / 'reader.json')})
                    result, values, audit = audit_condition(directory, prepared_path, model, task)
                    result.update(encoder=encoder, method=method, historical_reference=encoder == 'qwen')
                    rows.append(result)
                    audit.update(encoder=encoder, method=method, model=model, task=task,
                                 reader_metadata_hashes=metadata_hashes.copy())
                    audits.append(audit)
                    scores[method] = values
                    print(json.dumps(result), flush=True)
                low, high = paired_interval(scores['without_recommendation'], scores['amor'], cluster=task == 'LoCoMo')
                contrasts.append(dict(encoder=encoder, model=model, task=task,
                    difference_pp=100 * sum(scores['amor'][k] - scores['without_recommendation'][k]
                                            for k in scores['amor']) / len(scores['amor']),
                    paired_low=low, paired_high=high))
    destination = root / ('reference_qa_audit' if reference_only else 'report_qa')
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (('scores.csv', rows), ('contrasts.csv', contrasts)):
        with (destination / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    json_save(destination / 'audit.json', dict(conditions=audits, scope=list(encoders),
        reference_only=reference_only, native_scores_recomputed=True,
        bootstrap='5000 paired replicates; conversation clusters for LoCoMo, questions for other tasks; seed42'))
    json_save(destination / 'complete.json', dict(complete=True, reference_only=reference_only,
        conditions=len(rows), answers=sum(r['questions'] for r in rows),
        hashes={name: digest(destination / name) for name in ('scores.csv', 'contrasts.csv', 'audit.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    report(args.root, args.reference_only)
