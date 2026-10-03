"""Audit fixed graphs and complete central-source rankings for all encoders."""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
from statistics import mean
import tempfile

from experiments.encoder_robustness import ENCODERS, RecognitionAudit, digest, json_save, protocol_tasks


def audit_recognition_log(path, policy):
    """Reparse original responses with the native parser, without inference."""
    from baseline.official import _OFFICIAL_ALGORITHMS, _prepend
    _prepend(_OFFICIAL_ALGORITHMS / 'HippoRAG' / 'src')
    from hipporag.rerank import DSPyFilter
    native = DSPyFilter.__new__(DSPyFilter)
    def forbidden_inference(*args, **kwargs):
        raise RuntimeError('Reporting must never request inference')
    native.llm_infer_fn = forbidden_inference
    with tempfile.TemporaryDirectory(prefix='amor-recognition-audit-') as temporary:
        audit = RecognitionAudit(native, Path(temporary) / 'unused_inference.jsonl', policy=policy)
        calls = 0
        try:
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    response = record['response']
                    if policy == 'historical_replay' and response[-1] is not True:
                        raise ValueError('Historical replay contains uncached inference')
                    native.parse_filter(response[0])
                    audit.check()
                    calls += 1
            if not calls:
                raise ValueError('Missing recognition calls')
            return dict(policy=policy, calls=calls, responses=audit.responses)
        finally:
            audit.close()


def evidence_metrics(gold, retrieved):
    gold, retrieved = set(gold), list(retrieved)
    if not gold:
        raise ValueError('No positive source annotations')
    if not retrieved or len(set(retrieved)) != len(retrieved):
        raise ValueError('Empty or duplicated retrieved sources')
    hits = len(gold.intersection(retrieved))
    return dict(precision=100 * hits / len(retrieved), recall=100 * hits / len(gold),
                complete=100 * float(gold <= set(retrieved)))


def validate_rankings(traces, questions, passage_texts, task, group):
    expected = set(itertools.product(questions, ('amor', 'without_recommendation', 'dense'), (5, 10, 15)))
    seen = set()
    for trace in traces:
        key = trace['case_id'], trace['method'], trace['k']
        if key in seen or key not in expected:
            raise ValueError('Unexpected or duplicated ranking condition')
        seen.add(key)
        if trace['group_id'] != group or trace['task'] != task:
            raise ValueError('Ranking task/group differs')
        ids = trace['ids']
        if len(ids) != trace['k'] or len(set(ids)) != len(ids):
            raise ValueError('Wrong retrieval budget or duplicated sources')
        if [passage_texts[i] for i in ids] != trace['passages']:
            raise ValueError('Ranked source identities and text disagree')
    if seen != expected:
        raise ValueError('Missing ranking conditions')


def locomo_annotations(path):
    from dataset_loader import load_locomo
    raw = {str(s['sample_id']): s for s in json.loads(path.read_text())}
    mappings, gold, cases = {}, {}, {}
    for conversation in load_locomo(path):
        sample = raw[conversation.sample_id]
        history = sample['conversation']
        sessions = sorted((k for k in history if k.startswith('session_') and k[8:].isdigit()),
                          key=lambda k: int(k[8:]))
        turns = [turn for session in sessions for turn in history[session]]
        texts, ids = conversation.memory_items, [t['dia_id'] for t in turns]
        if len(texts) != len(ids) or len(set(texts)) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError('Ambiguous native LoCoMo source identities')
        group = 'locomo-' + conversation.sample_id
        mappings[group] = dict(zip(texts, ids, strict=True))
        for index, question in enumerate(conversation.questions):
            key = group, f'{conversation.sample_id}-{index}'
            cases[key] = question
            # Category 5 annotations describe distractors, not positive evidence.
            if question.category != 5 and question.evidence:
                gold[key] = set(question.evidence)
    return mappings, gold, cases


def report(root, locomo_path, reference_only=False):
    import igraph as ig
    from experiments.runner import _read_retrieval_records
    protocol = json.loads((root / 'protocol.json').read_text())
    tasks = protocol_tasks(root)
    if protocol['encoders'] != ENCODERS:
        raise ValueError('Frozen encoder scope changed')
    if protocol['code_sha256'] != digest(Path(__file__).with_name('encoder_robustness.py')):
        raise ValueError('Experiment implementation changed')
    reference = Path(protocol['source_results'])
    original_protocol = json.loads((reference / 'protocol.json').read_text())
    sources = json.loads((root / 'sources.json').read_text())
    if not sources['complete']:
        raise ValueError('Incomplete source export')
    locomo_map, locomo_gold, locomo_cases = ({}, {}, {})
    if 'LoCoMo' in tasks:
        locomo_map, locomo_gold, locomo_cases = locomo_annotations(locomo_path)
    reference_rows = {}
    for task in tasks:
        path = root / 'qa_inputs/qwen/amor' / task / 'retrieval.jsonl'
        rows = list(_read_retrieval_records(path))
        reference_rows[task] = {(r.group_id, r.case.case_id): r for r in rows}
        if len(reference_rows[task]) != len(rows):
            raise ValueError('Duplicate reference questions')
    if set(reference_rows.get('LoCoMo', {})) != set(locomo_cases):
        raise ValueError('LoCoMo native population differs')
    for key, row in reference_rows.get('LoCoMo', {}).items():
        native = locomo_cases[key]
        if (row.case.question, row.case.answers, row.case.category) != (native.question, (native.answer,), native.category):
            raise ValueError('LoCoMo annotation/question mismatch')
    groups, observations, audits = set(), [], []
    encoders = ('qwen',) if reference_only else ('qwen', 'bge', 'nv')
    for entry in sources['entries']:
        task, group = entry['task'], entry['group']
        if (task, group) in groups:
            raise ValueError('Duplicated source group')
        groups.add((task, group))
        source_path = Path(entry['directory']) / 'source.json'
        source = json.loads(source_path.read_text())
        text_by_id = dict(zip(source['passage_ids'], source['passage_texts'], strict=True))
        if len(text_by_id) != len(source['passage_ids']) or len(set(text_by_id.values())) != len(text_by_id):
            raise ValueError('Ambiguous passage identities')
        questions = {cid: row for (g, cid), row in reference_rows[task].items() if g == group}
        graph_path = Path(original_protocol['graph_source']) / 'groups' / task / group / (original_protocol['graph_variant'] + '.pickle')
        graph = ig.Graph.Read_Pickle(str(graph_path))
        if (graph.vs['name'] != source['graph_names'] or
                [list(e) for e in graph.get_edgelist()] != source['graph_edges'] or
                graph.es['weight'] != source['graph_weights']):
            raise ValueError('Graph identities, connectivity or weights changed')
        del graph
        if task == 'LoCoMo':
            gold = {cid: locomo_gold[group, cid] for cid in questions if (group, cid) in locomo_gold}
        else:
            gold = {cid: set(row.case.gold_passages) for cid, row in questions.items()
                    if row.case.gold_passages}
            if task == '2WikiMultiHopQA' and len(gold) != len(questions):
                raise ValueError('2Wiki requires annotations for every question')
        for encoder in encoders:
            directory = root / 'retrieval' / encoder / task / group
            marker = json.loads((directory / 'complete.json').read_text())
            vector_dir = root / 'vectors' / encoder / task / group
            vectors = json.loads((vector_dir / 'complete.json').read_text())
            if not marker['complete'] or marker['smoke'] or marker['questions'] != len(questions):
                raise ValueError('Incomplete full retrieval')
            if not marker['graph_fixed'] or marker['encoder'] != encoder:
                raise ValueError('Retrieval graph/encoder mismatch')
            if marker['reference_replayed'] != (encoder == 'qwen'):
                raise ValueError('Reference replay status differs')
            if vectors['source_sha256'] != digest(source_path):
                raise ValueError('Vector source changed')
            vector_hash = digest(vector_dir / 'embeddings.npz')
            if vector_hash != vectors['embeddings_sha256'] or vector_hash != marker['embeddings_sha256']:
                raise ValueError('Vector hash changed after retrieval')
            expected_policy = 'historical_replay' if encoder == 'qwen' else 'strict'
            recognition_path = directory / 'recognition.jsonl'
            recognition = audit_recognition_log(recognition_path, expected_policy)
            failures = sum(r['status'] != 'valid' for r in recognition['responses'].values())
            if encoder != 'qwen' and failures:
                raise ValueError('Fresh recognition contains parsing failures')
            traces = json.loads((directory / 'rankings.json').read_text())
            validate_rankings(traces, questions, text_by_id, task, group)
            for trace in traces:
                cid = trace['case_id']
                if cid not in gold:
                    continue
                selected = ([locomo_map[group][text] for text in trace['passages']]
                            if task == 'LoCoMo' else trace['passages'])
                observations.append(dict(encoder=encoder, task=task, group=group, case_id=cid,
                    method=trace['method'], k=trace['k'], **evidence_metrics(gold[cid], selected)))
            paths = [source_path, graph_path, directory / 'complete.json', directory / 'rankings.json',
                     recognition_path, vector_dir / 'complete.json']
            audits.append(dict(encoder=encoder, task=task, group=group, questions=len(questions),
                annotated_questions=len(gold), graph_fixed=True, recognition_failures=failures,
                recognition_calls=recognition['calls'], recognition_policy=expected_policy,
                embeddings_sha256=vector_hash, hashes={str(p): digest(p) for p in paths}))
            print('retrieval_verified', encoder, task, group, len(questions), flush=True)
    if groups != {(task, g) for task, rows in reference_rows.items() for g, _ in rows}:
        raise ValueError('Missing source groups')
    buckets = defaultdict(list)
    for row in observations:
        buckets[row['encoder'], row['task'], row['method'], row['k']].append(row)
    annotated_tasks = {a['task'] for a in audits if a['annotated_questions']}
    expected_metrics = set(itertools.product(encoders, annotated_tasks,
                                            ('amor', 'without_recommendation', 'dense'), (5, 10, 15)))
    if set(buckets) != expected_metrics:
        raise ValueError('Missing retrieval metric conditions')
    expected_conditions = set(itertools.product(encoders, tasks,
                                               ('amor', 'without_recommendation', 'dense'), (5, 10, 15)))
    summary = []
    for e, t, m, k in sorted(expected_conditions):
        rows = buckets[e, t, m, k]
        summary.append(dict(encoder=e, task=t, method=m, k=k, questions=len(rows),
                            **{metric: mean(r[metric] for r in rows) if rows else None
                               for metric in ('precision', 'recall', 'complete')}))
    target = root / ('reference_retrieval_audit' if reference_only else 'report_retrieval')
    target.mkdir(parents=True, exist_ok=True)
    for name, records in (('scores.csv', summary), ('per_question.csv', observations)):
        with (target / name).open('w', newline='') as stream:
            fields = (list(summary[0]) if name == 'scores.csv' else
                      ['encoder', 'task', 'group', 'case_id', 'method', 'k', 'precision', 'recall', 'complete'])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    json_save(target / 'audit.json', dict(groups=audits, reference_only=reference_only,
        annotation_sha256=digest(locomo_path) if 'LoCoMo' in tasks else None,
        policy='Central sources only. All rankings, graphs and cases are audited. Metrics use available native positive annotations; unannotated tasks have empty metrics, not zero scores. LoCoMo unresolved IDs remain misses; category5 is excluded only from annotated retrieval metrics, never QA.'))
    json_save(target / 'complete.json', dict(complete=True, reference_only=reference_only,
        conditions=len(summary), hashes={name: digest(target / name) for name in ('scores.csv', 'per_question.csv', 'audit.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--locomo', type=Path, default=Path('/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json'))
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    report(args.root, args.locomo, args.reference_only)
