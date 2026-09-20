"""Compare native 2Wiki gold evidence with frozen retrieved evidence across readers."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import zipfile

from baseline.base import RetrievedItem
from experiments.runner import _read_retrieval_records, _retrieval_record
from optimization.ircot import BASE, MODELS, Reader, evaluate_task, groups_for, write_json
from optimization.report_results import BASELINES, TASK_METRICS


ROOT = BASE / 'optimization_gold_evidence_seed42_20260919'
PREVIOUS = BASE / 'optimization_fact_graph_main_qa_seed42_20260919'
FACTS = BASE / 'optimization_graph_edges_seed42_20260919/main/without_synonym_edges'
TASK = '2WikiMultiHopQA'
CONTROLS = ('gold_only', 'bm25', 'fact_graph')


def prepare(phase, pilot):
    groups = groups_for(TASK)
    native = [(group.group_id, case) for group in groups for case in group.cases]
    assert len(native) == TASK_METRICS[TASK][0] == 1000
    corpora = {g.group_id: set(g.memory_items) for g in groups}
    controls = {name: list(_read_retrieval_records(source / TASK / 'retrieval.jsonl'))
                for name, source in (('bm25', BASELINES['bm25']), ('fact_graph', FACTS))}
    assert all([(r.group_id, r.case) for r in rows] == native for rows in controls.values())
    sizes = Counter()
    for group_id, case in native:
        assert case.metric == 'hipporag' and case.gold_passages
        assert len(set(case.gold_passages)) == len(case.gold_passages)
        assert set(case.gold_passages).issubset(corpora[group_id])
        sizes[len(case.gold_passages)] += 1
    controls['gold_only'] = [replace(row,
        retrieved=tuple(RetrievedItem(text, metadata={'oracle_annotation': True})
                        for text in row.case.gold_passages),
        top_k=len(row.case.gold_passages), retrieval_seconds=None) for row in controls['bm25']]
    for name, rows in controls.items():
        if pilot:
            rows = rows[:2]
        directory = phase / 'inputs' / name / TASK
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'retrieval.jsonl'
        content = ''.join(json.dumps(_retrieval_record(r), ensure_ascii=False) + '\n' for r in rows)
        if output.exists():
            assert output.read_text() == content
        else:
            output.write_text(content)
    write_json(phase / 'protocol.json', dict(task=TASK, pilot=pilot, complete_native_questions=1000,
        evaluated_questions=2 if pilot else 1000, models=MODELS, controls=CONTROLS, seed=42,
        full_gold_passage_counts=dict(sizes), gold_order='Unchanged native case.gold_passages order',
        reference='https://aclanthology.org/2024.tacl-1.9/',
        scope='Oracle-context diagnostic on the HippoRAG-released 1000-query 2Wiki subset, not a deployable retriever',
        fixed='Original questions, labels, QA prompts, generation and answer F1; frozen BM25 and fact-graph evidence',
        caveats=['Gold-only uses test annotations and cannot be a main-method score or tuning rule',
                 'Gold evidence is not a guaranteed upper bound and non-perfect F1 does not alone prove semantic error',
                 'Evidence amount and content change together; this does not isolate ordering or distractor effects',
                 'No pseudo evidence labels for the other five tasks; six-task main scope unchanged',
                 'This diagnostic does not change the extractor, retriever, reader or IRCoT controller']))


def run(phase, pilot):
    reports = {}
    for model in MODELS:
        slug = model.replace('/', '_')
        if not pilot:
            checked = json.loads((ROOT / 'pilot' / slug / 'complete.json').read_text())
            assert checked['complete'] and set(checked['controls']) == set(CONTROLS)
        reader = Reader(model)
        try:
            previous = json.loads((PREVIOUS / slug / 'main/reader.json').read_text())
            assert reader.metadata == previous
            target = phase / slug
            write_json(target / 'reader.json', reader.metadata)
            reports[model], predictions = {}, {}
            for control in CONTROLS:
                directory = target / control / TASK
                directory.mkdir(parents=True, exist_ok=True)
                source = phase / 'inputs' / control / TASK / 'retrieval.jsonl'
                path = directory / 'retrieval.jsonl'
                if not path.exists():
                    path.symlink_to(source)
                assert path.resolve() == source.resolve()
                marker = directory / 'qa_complete.json'
                if not marker.exists():
                    evaluate_task(directory, TASK, reader, model, pilot)
                result = json.loads(marker.read_text())
                assert result['complete'] and result['native_evaluator'] and result['scores_recomputed']
                assert result['questions'] == (2 if pilot else 1000)
                reports[model][control] = result
                predictions[control] = [json.loads(line) for line in
                    (directory / 'evaluations' / slug / 'predictions.jsonl').open()]
                print(model, control, result['summary'], flush=True)
            pairs = []
            for rows in zip(*(predictions[name] for name in CONTROLS), strict=True):
                assert len({(r['group_id'], r['case_id']) for r in rows}) == 1
                pairs.append(dict(group_id=rows[0]['group_id'], case_id=rows[0]['case_id'],
                    controls={name: dict(prediction=row['prediction'], metrics=row['metrics'])
                              for name, row in zip(CONTROLS, rows, strict=True)}))
            write_json(target / 'paired_predictions.json', pairs)
            write_json(target / 'complete.json', dict(complete=True, model=model, pilot=pilot,
                questions=len(pairs), controls=reports[model]))
        finally:
            reader.close()
    write_json(phase / 'complete.json', dict(complete=True, pilot=pilot, models=reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    phase = ROOT / ('pilot' if args.pilot else 'main')
    phase.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(phase / f'code_{os.environ["SLURM_JOB_ID"]}.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), Path('optimization/ircot.py'),
                     Path('experiments/runner.py'), Path('dataset_loader/loader.py'), Path('utils/hipporag_metrics.py')):
            archive.write(path, path.name if path.is_absolute() else str(path))
        assert archive.testzip() is None
    prepare(phase, args.pilot)
    run(phase, args.pilot)
