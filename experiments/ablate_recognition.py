"""Ablate LLM triple filtering on the original and explicit-fact graphs."""

import argparse
from collections import Counter
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

from experiments.runner import _read_retrieval_records, _retrieval_record
from optimization.ircot import Reader, evaluate_task, write_json
from optimization.report_results import BASELINES, TASK_METRICS
from optimization.retriever.fact_incidence import install_fact_graph
from optimization.retriever.hipporag import CacheMissGuard, load_memory
from optimization.run_graph import BASE, SOURCE, TASKS, retrieval_config


ROOT = BASE / 'optimization_recognition_ablation_seed42_20260919'
RUN = ROOT / 'exact_context'
PREVIOUS = BASE / 'optimization_graph_edges_seed42_20260919'
GRAPHS = BASE / 'optimization_graph_index_factorial_seed42_20260919/graph'
CONTROLS = ('bm25', 'hipporag2', 'hipporag2_without_filter', 'fact_graph', 'fact_graph_without_filter')
MODEL = 'google/gemma-3-4b-it'


class FactFilter:
    def __init__(self, original, enabled):
        self.original, self.enabled, self.calls = original, enabled, []

    def __call__(self, query, candidate_facts, candidate_indices, len_after_rerank):
        self.calls.append(dict(query=query, facts=candidate_facts, indices=candidate_indices,
                               limit=len_after_rerank))
        if self.enabled:
            return self.original(query, candidate_facts, candidate_indices,
                                 len_after_rerank=len_after_rerank)
        return candidate_indices, candidate_facts, {}


def retrieve(phase, pilot):
    import igraph as ig
    import numpy as np
    from utils.models import release_accelerator_memory

    report = {}
    for task, (expected, _) in TASK_METRICS.items():
        sources = dict(bm25=BASELINES['bm25'], hipporag2=BASELINES['hipporag2'],
                       fact_graph=PREVIOUS / 'main/without_synonym_edges')
        frozen = {name: list(_read_retrieval_records(source / task / 'retrieval.jsonl'))
                  for name, source in sources.items()}
        identities = [(r.group_id, r.case) for r in frozen['hipporag2']]
        assert len(identities) == expected
        assert all([(r.group_id, r.case) for r in rows] == identities for rows in frozen.values())
        counts, selected = Counter(), []
        for row in frozen['hipporag2']:
            if not pilot or counts[row.group_id] < 2:
                selected.append(row)
            counts[row.group_id] += 1
        keys = {(r.group_id, r.case.case_id) for r in selected}
        frozen = {name: {(r.group_id, r.case.case_id): r for r in rows
                         if (r.group_id, r.case.case_id) in keys} for name, rows in frozen.items()}
        results = dict(bm25=list(frozen['bm25'].values()))
        results.update({name: [] for name in CONTROLS[1:]})
        candidates, score_differences, changed, replayed = [], [], Counter(), 0
        native_task = {name.replace(' ', '_'): name for name in TASKS}[task]
        config, _ = retrieval_config(SimpleNamespace(task=native_task, output_root=ROOT,
            generator_base_url='http://127.0.0.1:1/v1',
            path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
            data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
        for group_id in dict.fromkeys(r.group_id for r in selected):
            runtime = phase / 'runtime' / task / group_id
            cache = runtime / 'llm_cache'
            if not cache.exists():
                shutil.copytree(PREVIOUS / 'runtime' / task / group_id / 'llm_cache', cache)
            memory = load_memory(config, SOURCE / 'hipporag2' / task / 'hipporag_indices' / group_id, runtime)
            try:
                hippo = memory._memory
                original_filter = hippo.rerank_filter
                guard = CacheMissGuard(memory._generator)
                assert hippo.global_config.linking_top_k == 5 and hippo.global_config.damping == 0.5
                group_rows = [r for r in selected if r.group_id == group_id]
                for graph_name in ('hipporag2', 'fact_graph'):
                    if graph_name == 'fact_graph':
                        graph = ig.Graph.Read_Pickle(str(GRAPHS / task / group_id / 'graph.pickle'))
                        install_fact_graph(hippo, graph, keep_synonym_edges=False)
                        del graph
                    for row in group_rows:
                        calls, outputs = [], []
                        key = (group_id, row.case.case_id)
                        for enabled in (True, False):
                            wrapper = FactFilter(original_filter, enabled)
                            hippo.rerank_filter = wrapper
                            items = tuple(memory.retrieve(row.case.question, 5))
                            guard.check()
                            assert len(wrapper.calls) == 1 and len(items) == 5
                            calls.append(wrapper.calls)
                            outputs.append([item.text for item in items])
                            name = graph_name if enabled else graph_name + '_without_filter'
                            if enabled:
                                reference = frozen[graph_name][key].retrieved
                                assert outputs[-1] == [item.text for item in reference]
                                actual_scores = [i.score for i in items]
                                reference_scores = [i.score for i in reference]
                                assert np.isfinite(actual_scores).all() and np.isfinite(reference_scores).all()
                                if actual_scores != reference_scores:
                                    score_differences.append(dict(group_id=group_id, case_id=row.case.case_id,
                                        graph=graph_name, actual=actual_scores, reference=reference_scores))
                                replayed += 1
                            results[name].append(replace(row, retrieved=items, retrieval_seconds=None))
                        assert calls[0] == calls[1]
                        changed[graph_name] += outputs[0] != outputs[1]
                        candidates.append(dict(group_id=group_id, case_id=row.case.case_id,
                                               graph=graph_name, candidates=calls[0]))
                hippo.rerank_filter = original_filter
            finally:
                memory.close()
                memory = hippo = guard = wrapper = original_filter = None
                gc.collect()
                release_accelerator_memory()
        order = {(r.group_id, r.case.case_id): i for i, r in enumerate(selected)}
        for name, rows in results.items():
            rows.sort(key=lambda r: order[r.group_id, r.case.case_id])
            assert [(r.group_id, r.case) for r in rows] == [(r.group_id, r.case) for r in selected]
            target = phase / name / task
            target.mkdir(parents=True, exist_ok=True)
            content = ''.join(json.dumps(_retrieval_record(r), ensure_ascii=False) + '\n' for r in rows)
            output = target / 'retrieval.jsonl'
            if output.exists():
                assert output.read_text() == content
            else:
                output.write_text(content)
        write_json(phase / 'candidates' / (task + '.json'), candidates)
        write_json(phase / 'score_differences' / (task + '.json'), score_differences)
        report[task] = dict(questions=len(selected), replayed=replayed, changed_rankings=dict(changed),
                            nonidentical_score_vectors=len(score_differences))
        print(task, report[task], flush=True)
    assert sum(row['questions'] for row in report.values()) == (30 if pilot else 3386)
    write_json(phase / 'retrieval_complete.json', dict(complete=True, tasks=report,
        candidates_exact=True, recognition_cache_misses=0, new_extraction=False))


def evaluate(phase, pilot):
    reader = Reader(MODEL)
    try:
        target = phase / MODEL.replace('/', '_')
        write_json(target / 'reader.json', reader.metadata)
        results = {}
        for task in TASK_METRICS:
            results[task] = {}
            for control in CONTROLS:
                directory = target / control / task
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / 'retrieval.jsonl'
                source = phase / control / task / 'retrieval.jsonl'
                if not path.exists():
                    path.symlink_to(source)
                assert path.resolve() == source.resolve()
                marker = directory / 'qa_complete.json'
                if not marker.exists():
                    evaluate_task(directory, task, reader, MODEL, pilot)
                result = json.loads(marker.read_text())
                assert result['complete'] and result['native_evaluator'] and result['scores_recomputed']
                results[task][control] = result
                print(task, control, result['summary'], flush=True)
        write_json(target / 'complete.json', dict(complete=True, pilot=pilot, model=MODEL, tasks=results))
    finally:
        reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if not args.pilot:
        checked = json.loads((RUN / 'pilot/google_gemma-3-4b-it/complete.json').read_text())
        assert checked['complete'] and all(set(v) == set(CONTROLS) for v in checked['tasks'].values())
    phase = RUN / ('pilot' if args.pilot else 'main')
    phase.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(phase / f'code_{os.environ["SLURM_JOB_ID"]}.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), Path('optimization/ircot.py'),
                     Path('optimization/retriever/fact_incidence.py'), Path('optimization/retriever/hipporag.py'),
                     Path('experiments/runner.py'), Path('baseline/official.py'),
                     Path('baseline_algorithms/HippoRAG/src/hipporag/HippoRAG.py')):
            archive.write(path, path.name if path.is_absolute() else str(path))
        assert archive.testzip() is None
    write_json(phase / 'protocol.json', dict(controls=CONTROLS, seed=42, test_as_dev=True,
        model=MODEL, reference='https://arxiv.org/html/2502.14802v2#S6.SS1',
        intervention='Replace LLM triple filter with identity; keep original top-5 embedding candidates',
        fixed='OpenIE, embeddings, candidate count, graph within each pair, PPR, raw evidence, native QA and metrics',
        graphs='Original HippoRAG 2 vs current no-synonym explicit-fact graph with uniform fact seeds',
        caveats=['Graph comparison includes the existing graph and seed changes, not topology alone',
                 'Removing filtering also removes filter-empty dense fallback when candidates are nonempty',
                 'No new recognition generation; filtered controls require cached recognition and exact source text and order',
                 'Reconstruction requires exact source text and order, not identical floating point scores; all score differences saved',
                 'This is a published component ablation, not a new algorithm or zero-LLM pipeline',
                 'Does not evaluate fixed-query multi-round replay; no new IRCoT claims']))
    retrieve(phase, args.pilot)
    evaluate(phase, args.pilot)
