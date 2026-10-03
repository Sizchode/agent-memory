"""Inspect frozen embeddings and replay graph scores for reviewed QA cases."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BASE = Path('/oscar/scratch/zliu328/agent-memory-outputs')
CASES = BASE / 'analysis_cases_20260928/cases'
GRAPH = BASE / 'optimization_retained_fact_index_seed42_20260914/statement_projection_loop_free_retained_index_rrf_window'
QUERIES = BASE / 'optimization_query_embeddings_seed42_20260919'
COMPONENTS = BASE / 'optimization_component_ablation_seed42_20260927/groups'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ranks(scores):
    order = np.argsort(scores)[::-1]
    result = np.empty(len(order), dtype=int)
    result[order] = np.arange(1, len(order) + 1)
    return result


def extract(name, output, results=None):
    print('Loading analysis dependencies', name, flush=True)
    import igraph as ig
    from sklearn.decomposition import PCA
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.run_graph import retrieval_config

    if results is None:
        case = json.loads((CASES / (name + '.json')).read_text())
        reference = case['conditions']['libra']['input']
        reference_path = CASES / (name + '.json')
    else:
        if name != '2wiki_projection_1':
            raise ValueError('Current-result replay requires the reviewed 2Wiki case')
        reference_path = results / 'analysis/reviewed_cases.json'
        reviewed = json.loads(reference_path.read_text())
        current, = [r for r in reviewed
                    if r['case_id'] == 'fcdafe320bdb11eba7f7acde48001122'
                    and r['model'] == 'Qwen_Qwen3.5-4B' and r['method'] == 'libra']
        reference = current['context']
        case = {'task': current['task'], 'conditions': {}}
    task, group = case['task'], reference['group_id']
    q = reference['case']['question']
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    config, _ = retrieval_config(SimpleNamespace(task=task.replace('_', ' '), output_root=directory,
        generator_base_url='http://127.0.0.1:1/v1',
        path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
        data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
    runtime = directory / 'runtime'
    cache, = (GRAPH / task / 'runtime' / group / 'llm_cache').glob('*.sqlite')
    dest = runtime / 'llm_cache' / cache.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        assert not cache.with_name(cache.name + '-wal').exists()
        shutil.copy2(cache, dest)
    memory = load_optimized_memory(config, GRAPH / task / 'memory' / group, runtime)
    print('Loaded frozen graph', name, flush=True)
    try:
        hippo = memory._memory
        if results is not None:
            protocol = json.loads((results / 'protocol.json').read_text())
            graph_path = (Path(protocol['graph_source']) / 'groups' / task / group
                          / (protocol['graph_variant'] + '.pickle'))
            graph = ig.Graph.Read_Pickle(str(graph_path))
            assert graph.vs['name'] == hippo.graph.vs['name']
            hippo.graph = graph
        guard = CacheMissGuard(memory._generator)
        load_query_embeddings(hippo, QUERIES / task / group / 'queries.npz')

        def reject(*args, **kwargs):
            raise RuntimeError('No new embedding is allowed in this analysis')

        hippo.embedding_model.batch_encode = reject
        captured = {}
        old_ppr, old_recognition = hippo.run_ppr, hippo.rerank_facts

        def capture_ppr(reset_prob, damping=None):
            captured['reset'] = reset_prob.copy()
            captured['damping'] = damping
            result = old_ppr(reset_prob, damping)
            captured['ranking'], captured['scores'] = result
            return result

        def capture_recognition(*args, **kwargs):
            result = old_recognition(*args, **kwargs)
            captured['recognition'] = result[2]
            return result

        hippo.run_ppr, hippo.rerank_facts = capture_ppr, capture_recognition
        selected = memory.retrieve(q, 5)
        guard.check()
        expected_keys = [r['metadata']['source_passage'] for r in reference['retrieved']
                         if 'source_passage' in r['metadata']]
        actual_keys = [r.metadata['source_passage'] for r in selected]
        assert expected_keys == actual_keys
        graph = hippo.graph
        reset = captured['reset']
        reset = reset / reset.sum()
        alpha = captured['damping']
        pi = np.asarray(graph.personalized_pagerank(damping=alpha, directed=False,
                         weights='weight', reset=reset, implementation='prpack'))
        doc_nodes = np.asarray(hippo.passage_node_idxs)
        np.testing.assert_allclose(pi[doc_nodes][captured['ranking']], captured['scores'], atol=1e-12)
        binary = ig.Graph.Read_Pickle(str(COMPONENTS / task / group / 'without_projection.pickle'))
        assert binary.vs['name'] == graph.vs['name']
        binary_pi = np.asarray(binary.personalized_pagerank(damping=alpha, directed=False,
                                weights='weight', reset=reset, implementation='prpack'))
        metadata = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
        original = ig.Graph.Read_Pickle(metadata['source_graph'])
        assert original.vs['name'] == graph.vs['name']
        original_pi = np.asarray(original.personalized_pagerank(damping=alpha, directed=False,
                                  weights='weight', reset=reset, implementation='prpack'))
        if 'without_projection' in case['conditions']:
            hippo.graph = binary
            control_items = memory.retrieve(q, 5)
            guard.check()
            control_keys = [r['metadata']['source_passage'] for r in
                            case['conditions']['without_projection']['input']['retrieved']
                            if 'source_passage' in r['metadata']]
            assert [r.metadata['source_passage'] for r in control_items] == control_keys
            np.testing.assert_allclose(captured['reset'] / captured['reset'].sum(), reset, atol=0)
            hippo.graph = graph
        query = hippo.query_to_embedding['passage'][q].reshape(-1)
        embeddings = hippo.passage_embeddings
        cosine = embeddings @ query
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(query), 1, atol=1e-5)
        dense_rank, ppr_rank, binary_rank = ranks(cosine), ranks(pi[doc_nodes]), ranks(binary_pi[doc_nodes])
        original_rank = ranks(original_pi[doc_nodes])
        # PCA is a visualization only; all scores above use the original vectors.
        pca = PCA(n_components=2, svd_solver='randomized', random_state=42)
        xy = pca.fit_transform(embeddings)
        qxy = pca.transform(query[None, :])[0]
        sources = hippo.chunk_embedding_store.get_all_id_to_rows()
        entities = hippo.entity_embedding_store.get_all_id_to_rows()
        names = graph.vs['name']
        texts = {key: row['content'] for key, row in {**sources, **entities}.items()}
        gold = reference['case'].get('gold_passages') or []
        records = []
        for i, key in enumerate(hippo.passage_node_keys):
            records.append(dict(key=key, node=int(doc_nodes[i]), text=sources[key]['content'],
                xy=xy[i].tolist(), cosine=float(cosine[i]), dense_rank=int(dense_rank[i]),
                graph_rank=int(ppr_rank[i]), binary_rank=int(binary_rank[i]),
                original_graph_same_seeds_rank=int(original_rank[i]),
                graph_score=float(pi[doc_nodes[i]]), initial_score=float(reset[doc_nodes[i]]),
                final_rank=actual_keys.index(key) + 1 if key in actual_keys else None,
                annotated_support=sources[key]['content'] in gold))
        # Capture the full incoming neighborhood, not just the edges shown in a figure.
        targets = [r for r in records if r['annotated_support']]
        if not targets:
            targets = [r for r in records if r['key'] == actual_keys[0]]
        strength = np.asarray(graph.strength(weights='weight', mode='all'))
        neighborhoods = []
        for target in targets:
            j = target['node']
            incoming = []
            for eid in graph.incident(j):
                edge = graph.es[eid]
                i = edge.target if edge.source == j else edge.source
                assert i != j
                bid = binary.get_eid(i, j, directed=False, error=False)
                weight = float(edge['weight'])
                mass = alpha * pi[i] * weight / strength[i]
                incoming.append(dict(node=i, key=names[i], text=texts.get(names[i], names[i]),
                    type='record' if names[i] in sources else 'entity', weight=weight,
                    binary_weight=float(binary.es[bid]['weight']) if bid >= 0 else 0.0,
                    transition=weight / strength[i], contribution=float(mass),
                    initial_score=float(reset[i]), graph_score=float(pi[i])))
            incoming.sort(key=lambda r: -r['contribution'])
            direct = (1 - alpha) * reset[j]
            residual = pi[j] - direct - sum(r['contribution'] for r in incoming)
            # Dangling nodes restart with personalization; preserve this separately.
            dangling = alpha * pi[strength == 0].sum() * reset[j]
            assert abs(residual - dangling) < 1e-10
            neighborhoods.append(dict(target=target['key'], incoming=incoming,
                direct_restart=float(direct), dangling_restart=float(dangling),
                residual=float(residual), graph_score=float(pi[j]),
                retained_triples=memory.contents[target['key']]['retained_triples']))
        seeds = [dict(key=names[i], text=texts.get(names[i], names[i]), score=float(reset[i]))
                 for i in np.flatnonzero(reset) if names[i] not in sources]
        local = {r['node'] for r in targets}
        local.update(graph.vs.find(name=s['key']).index for s in seeds)
        for neighborhood in neighborhoods:
            local.update(r['node'] for r in neighborhood['incoming'][:3])
        local_nodes = [dict(node=i, key=names[i], text=texts.get(names[i], names[i]),
            type='record' if names[i] in sources else 'entity',
            initial_score=float(reset[i]), graph_score=float(pi[i])) for i in sorted(local)]
        local_edges = []
        for edge in graph.es:
            i, j = edge.tuple
            if i in local and j in local:
                local_edges.append(dict(source=names[i], target=names[j], weight=float(edge['weight']),
                    forward=float(alpha * pi[i] * edge['weight'] / strength[i]),
                    backward=float(alpha * pi[j] * edge['weight'] / strength[j])))
        result = dict(name=name, task=task, group=group, question=q,
            reference_case=str(reference_path), records=records,
            query_xy=qxy.tolist(), pca_variance=pca.explained_variance_ratio_.tolist(),
            neighborhoods=neighborhoods, entity_seeds=seeds,
            local_nodes=local_nodes, local_edges=local_edges,
            recognition=captured['recognition'], damping=alpha,
            audit=dict(exact_original_selection=True, original_vectors=True,
                       new_llm_calls=guard.misses, new_embeddings=0,
                       same_reset_projection_control=True,
                       note='Original-graph control uses AMOR seeds, not native HippoRAG. Selected case; no population or causal-path claim.'))
        write_json(directory / 'geometry.json', result)
        print(name, 'verified', len(records), 'records', flush=True)
        for r in targets:
            print(r['text'][:70], {k: r[k] for k in ('dense_rank','graph_rank','binary_rank','final_rank')}, flush=True)
    finally:
        memory.close()


def graph_controls(results, output, synonym_control=False, recommendation_control=False, step_control=False):
    """Replay existing graph variants with identical query personalization."""
    import csv
    from collections import defaultdict
    import igraph as ig
    from baseline.base import RetrievedItem
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.retriever.hybrid_graph import fuse_rankings
    from optimization.run_graph import retrieval_config

    task, group = '2WikiMultiHopQA', 'hipporag-2wikimultihopqa'
    output.mkdir(parents=True, exist_ok=True)
    with (results / 'qa_results.csv').open() as stream:
        entry, = [r for r in csv.DictReader(stream) if r['setting'] == 'one_shot'
                  and r['variant'] == 'full' and r['task'] == task and r['model'] == 'Qwen/Qwen3.5-4B']
    references = [json.loads(line) for line in (Path(entry['source']) / 'retrieval.jsonl').open()]
    native_path = Path('baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa.json')
    native = {r['_id']: r for r in json.loads(native_path.read_text())}
    assert len(references) == len(native) == 1000
    assert {r['case']['case_id'] for r in references} == set(native)
    metadata = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
    protocol = json.loads((results / 'protocol.json').read_text())
    current_path = Path(protocol['graph_source']) / 'groups' / task / group / (protocol['graph_variant'] + '.pickle')
    paths = dict(original=Path(metadata['source_graph']),
                 binary=COMPONENTS / task / group / 'without_projection.pickle',
                 projected=Path(metadata['constructed_graph_file']), amor=current_path)
    if recommendation_control or step_control:
        paths = {'amor': current_path}
    graphs = {name: ig.Graph.Read_Pickle(str(path)) for name, path in paths.items()}
    if 'projected' in graphs:
        graphs['projected'].es['weight'] = np.load(GRAPH / task / 'memory' / group / 'edge_weights.npy').tolist()
    assert all(g.vs['name'] == graphs['amor'].vs['name'] for g in graphs.values())
    if synonym_control:
        original = graphs['original']
        without_synonyms = original.copy()
        without_synonyms.delete_edges([e.index for e in original.es if e['edge_kind'] == 'synonym'])
        graphs = dict(amor=graphs['amor'], original_without_synonym_edges=without_synonyms)
    if recommendation_control:
        graphs.update(without_entity_seeds=graphs['amor'], without_record_seeds=graphs['amor'])
    if step_control:
        graph = graphs['amor']
        assert not graph.is_directed() and not any(graph.is_loop())
        adjacency = graph.get_adjacency_sparse(attribute='weight').tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree>0)
        graphs.update({f'steps_{steps}': graph for steps in (1, 2, 4)})
    config, _ = retrieval_config(SimpleNamespace(task=task, output_root=output,
        generator_base_url='http://127.0.0.1:1/v1',
        path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
        data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
    runtime = output / '2wiki_projection_1/runtime'
    cache, = (GRAPH / task / 'runtime' / group / 'llm_cache').glob('*.sqlite')
    dest = runtime / 'llm_cache' / cache.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        assert not cache.with_name(cache.name + '-wal').exists()
        shutil.copy2(cache, dest)
    prefix = ('propagation_steps' if step_control else 'recommendation_controls' if recommendation_control else
              'synonym_controls' if synonym_control else 'graph_controls')
    path = output / (prefix + '.jsonl')
    rows = [json.loads(line) for line in path.open()] if path.exists() else []
    completed = {r['case_id']: r for r in rows}
    assert len(completed) == len(rows) and set(completed).issubset(native)
    memory = load_optimized_memory(config, GRAPH / task / 'memory' / group, runtime)
    try:
        hippo, hybrid = memory._memory, memory.base
        guard = CacheMissGuard(memory._generator)
        load_query_embeddings(hippo, QUERIES / task / group / 'queries.npz')
        def reject(*args, **kwargs):
            raise RuntimeError('Graph controls must reuse frozen embeddings')
        hippo.embedding_model.batch_encode = reject
        captured = {}
        old_ppr, old_recognition = hippo.run_ppr, hippo.rerank_facts
        def capture_ppr(reset_prob, damping=None):
            captured.update(reset=reset_prob.copy(), damping=damping)
            return old_ppr(reset_prob, damping)
        def capture_recognition(*args, **kwargs):
            result = old_recognition(*args, **kwargs)
            captured['recognition'] = result[2]
            return result
        hippo.run_ppr, hippo.rerank_facts = capture_ppr, capture_recognition
        sources = hippo.chunk_embedding_store.get_all_id_to_rows()
        text_to_key = {r['content']: key for key, r in sources.items()}
        assert len(text_to_key) == len(sources)
        step_verified = False
        for reference in references:
            cid, query = reference['case']['case_id'], reference['case']['question']
            assert query == native[cid]['question']
            expected = [r['metadata']['source_passage'] for r in reference['retrieved']
                        if 'source_passage' in r['metadata']]
            if cid in completed:
                assert completed[cid]['question'] == query and completed[cid]['selected']['amor'] == expected
                continue
            captured.clear()
            hippo.graph = graphs['amor']
            actual = memory.retrieve(query, 5)
            guard.check()
            assert [r.metadata['source_passage'] for r in actual] == expected
            gold = [text_to_key[text] for text in reference['case']['gold_passages']]
            lexical = hybrid.lexical.retrieve(query, 5)
            selected, gold_ranks, rankings = {}, {}, {}
            if recommendation_control:
                order, scores = hippo.dense_passage_retrieval(query)
                rankings['dense'] = [RetrievedItem(sources[hippo.passage_node_keys[i]]['content'], float(score),
                                      {'source_passage': hippo.passage_node_keys[i]})
                                     for i, score in zip(order[:15], scores[:15])]
            for name, graph in graphs.items():
                if 'reset' not in captured:
                    selected[name], gold_ranks[name] = expected, None
                    if recommendation_control:
                        rankings[name] = rankings['dense']
                    continue
                hippo.graph = graph
                reset = captured['reset'].copy()
                if name == 'without_entity_seeds':
                    reset[[i for i, key in enumerate(graph.vs['name']) if key.startswith('entity-')]] = 0
                elif name == 'without_record_seeds':
                    reset[hippo.passage_node_idxs] = 0
                assert reset.sum() > 0
                if name.startswith('steps_'):
                    alpha = captured['damping']
                    v = reset / reset.sum()
                    pi = v.copy()
                    for _ in range(int(name.split('_')[1])):
                        pi = alpha*(adjacency.T @ (pi*inverse_degree) + pi[degree==0].sum()*v) + (1-alpha)*v
                    assert np.isclose(pi.sum(), 1.0)
                    scores = pi[hippo.passage_node_idxs]
                    order = np.argsort(scores)[::-1]
                    scores = scores[order]
                    if not step_verified:
                        check = v.copy()
                        for _ in range(100):
                            check = alpha*(adjacency.T @ (check*inverse_degree) + check[degree==0].sum()*v) + (1-alpha)*v
                        expected_pi = np.asarray(graph.personalized_pagerank(damping=alpha, directed=False,
                            weights='weight', reset=reset, implementation='prpack'))
                        assert np.allclose(check, expected_pi, rtol=1e-8, atol=1e-12)
                        step_verified = True
                else:
                    order, scores = old_ppr(reset, captured['damping'])
                keys = [hippo.passage_node_keys[i] for i in order]
                positions = {key: rank + 1 for rank, key in enumerate(keys)}
                gold_ranks[name] = {key: positions[key] for key in gold}
                items = [RetrievedItem(sources[key]['content'], float(value), {'source_passage': key})
                         for key, value in zip(keys[:5], scores[:5])]
                fused = fuse_rankings(items, lexical, 5, hybrid.rank_constant)
                selected[name] = [text_to_key[item.text] for item in fused]
                if recommendation_control:
                    rankings[name] = [RetrievedItem(sources[key]['content'], float(value), {'source_passage': key})
                                      for key, value in zip(keys[:15], scores[:15])]
            assert selected['amor'] == expected
            seeds = []
            if 'reset' in captured:
                seeds = [dict(entity=hippo.entity_embedding_store.get_row(graphs['amor'].vs[i]['name'])['content'],
                              initial_score=float(captured['reset'][i]))
                         for i in np.flatnonzero(captured['reset'])
                         if graphs['amor'].vs[i]['name'].startswith('entity-')]
            row = dict(case_id=cid, question=query, question_type=native[cid]['type'], gold=gold,
                       selected=selected, gold_graph_ranks=gold_ranks,
                       recognition=captured.get('recognition'), entity_seeds=seeds,
                       dense_fallback='reset' not in captured)
            if recommendation_control:
                depth = {}
                for k in (5, 10, 15):
                    lexical_k = hybrid.lexical.retrieve(query, k)
                    selections = {name: fuse_rankings(items[:k], lexical_k, k, hybrid.rank_constant)
                                  for name, items in rankings.items() if name != 'dense'}
                    selections['dense'] = rankings['dense'][:k]
                    selections['bm25'] = lexical_k
                    selections['without_recommendation'] = fuse_rankings(rankings['dense'][:k], lexical_k, k, hybrid.rank_constant)
                    depth[str(k)] = {name: [text_to_key[item.text] for item in items]
                                     for name, items in selections.items()}
                assert depth['5']['amor'] == expected
                assert all(depth['5'][name] == selected[name] for name in graphs)
                row['depth'] = depth
            with path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)
            if len(rows) % 25 == 0:
                print('Graph controls verified', len(rows), '/1000', flush=True)
        guard.check()
    finally:
        memory.close()
    grouped = defaultdict(list)
    for row in rows:
        grouped['ALL'].append(row)
        grouped[row['question_type']].append(row)
    summary = []
    for kind, values in grouped.items():
        for name in graphs:
            golds = [set(r['gold']) for r in values]
            selections = [set(r['selected'][name]) for r in values]
            amor = [set(r['selected']['amor']) for r in values]
            summary.append(dict(question_type=kind, graph=name, questions=len(values),
                recall_at_5=100 * np.mean([len(g & s) / len(g) for g, s in zip(golds, selections)]),
                complete_at_5=100 * np.mean([g <= s for g, s in zip(golds, selections)]),
                recovered=sum(len((g - s) & a) for g, s, a in zip(golds, selections, amor)),
                lost=sum(len((g & s) - a) for g, s, a in zip(golds, selections, amor))))
    with (output / (prefix + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if recommendation_control:
        depth_summary = []
        for kind, values in grouped.items():
            for k in (5, 10, 15):
                for name in values[0]['depth'][str(k)]:
                    pairs = [(set(r['gold']), set(r['depth'][str(k)][name])) for r in values]
                    depth_summary.append(dict(question_type=kind, method=name, cutoff=k, questions=len(values),
                        recall=100*np.mean([len(g & s)/len(g) for g,s in pairs]),
                        complete=100*np.mean([g <= s for g,s in pairs])))
        with (output / 'retrieval_depth.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(depth_summary[0]))
            writer.writeheader()
            writer.writerows(depth_summary)
    assert len(rows) == len({r['case_id'] for r in rows}) == 1000
    write_json(output / (prefix + '_audit.json'), dict(complete=True, questions=len(rows),
        new_llm_calls=guard.misses, new_embeddings=0, current_selections_exact=True,
        graphs={name: str(path) for name, path in paths.items()},
        fixed=('graph, retained facts, recognition, embeddings, damping and BM25 implementation; vary the named seed component or retrieval count'
               if recommendation_control else 'retained facts, recognition, personalization, embeddings, damping, BM25, cutoff'),
        fallback_questions=sum(r['dense_fallback'] for r in rows),
        removed_edge_kind='synonym' if synonym_control else None,
        note=('Apply the existing PageRank update 1, 2 or 4 times from the same normalized personalization. Dangling mass returns to personalization; power iteration checked against native igraph. Same graph, initial scores, BM25 and top five. No new QA.'
              if step_control else 'Seed component removal and retrieval count sensitivity; graph, cached recognition and embeddings fixed. Both ranking branches return k candidates before native RRF; dense fallback preserved. No new QA.'
              if recommendation_control else 'Graph interventions within AMOR; original is not the native HippoRAG pipeline.')))


def case_counterfactual(results, output):
    """Delete actual edges or a seed in the fixed, previously reviewed case."""
    import igraph as ig
    from baseline.base import RetrievedItem
    from baseline.bm25 import BM25Baseline
    from optimization.retriever.hybrid_graph import fuse_rankings

    directory = output / '2wiki_projection_1'
    data = json.loads((directory / 'geometry.json').read_text())
    task, group = data['task'], data['group']
    meta = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
    protocol = json.loads((results / 'protocol.json').read_text())
    graph_path = Path(protocol['graph_source']) / 'groups' / task / group / (protocol['graph_variant'] + '.pickle')
    current = ig.Graph.Read_Pickle(str(graph_path))
    original = ig.Graph.Read_Pickle(meta['source_graph'])
    assert current.vs['name'] == original.vs['name']
    names = {name: i for i, name in enumerate(current.vs['name'])}
    reset = np.zeros(current.vcount())
    for row in data['records']:
        reset[row['node']] = row['initial_score']
    for row in data['entity_seeds']:
        reset[names[row['key']]] = row['score']
    np.testing.assert_allclose(reset.sum(), 1, atol=1e-12)
    director, = [s for s in data['entity_seeds'] if s['text'] == 'albert s rogell']
    d = names[director['key']]
    film, = [r for r in data['records'] if r['annotated_support'] and r['dense_rank'] == 1]
    bio, = [r for r in data['records'] if r['annotated_support'] and r['dense_rank'] != 1]
    film_edge = current.get_eid(d, film['node'], directed=False)
    bio_edge = current.get_eid(d, bio['node'], directed=False)
    conditions = [('amor', current.copy(), reset.copy())]
    no_seed = reset.copy()
    no_seed[d] = 0
    conditions.append(('without_director_seed', current.copy(), no_seed))
    for name, edges in [('without_film_director_edge', [film_edge]),
                        ('without_biography_director_edge', [bio_edge]),
                        ('without_both_edges', [film_edge, bio_edge])]:
        graph = current.copy()
        graph.delete_edges(edges)
        conditions.append((name, graph, reset.copy()))
    conditions.append(('original', original.copy(), reset.copy()))
    assert 'edge_kind' in original.es.attribute_names()
    synonyms = [e.index for e in original.es if e['edge_kind'] == 'synonym']
    local_synonyms = [e for e in original.incident(d) if original.es[e]['edge_kind'] == 'synonym']
    for name, edges in [('original_without_director_synonym_edges', local_synonyms),
                        ('original_without_synonym_edges', synonyms)]:
        graph = original.copy()
        graph.delete_edges(edges)
        conditions.append((name, graph, reset.copy()))
    records = data['records']
    doc_nodes = np.array([r['node'] for r in records])
    texts = {r['key']: r['text'] for r in records}
    keys = json.loads((GRAPH / task / 'memory' / group / 'lexical_source_keys.json').read_text())
    lexical = BM25Baseline()
    lexical.build([texts[key] for key in keys])
    lexical_items = lexical.retrieve(data['question'], 5)
    rows = []
    for name, graph, seeds in conditions:
        pi = np.asarray(graph.personalized_pagerank(damping=data['damping'], directed=False,
                        weights='weight', reset=seeds, implementation='prpack'))
        rank = ranks(pi[doc_nodes])
        order = np.argsort(pi[doc_nodes])[::-1][:5]
        items = [RetrievedItem(records[i]['text'], float(pi[doc_nodes[i]]), {}) for i in order]
        fused = fuse_rankings(items, lexical_items, 5)
        selected = [r.text for r in fused]
        row = dict(condition=name, graph_rank=int(rank[next(i for i,r in enumerate(records) if r['key']==bio['key'])]),
            final_rank=selected.index(bio['text'])+1 if bio['text'] in selected else None,
            biography_score=float(pi[bio['node']]), edges=graph.ecount(),
            director_degree=graph.degree(d), director_strength=graph.strength(d, weights='weight'),
            selected_titles=[text.splitlines()[0] for text in selected])
        if name == 'amor':
            assert row['graph_rank'] == bio['graph_rank'] and row['final_rank'] == bio['final_rank']
        if name == 'original':
            assert row['graph_rank'] == bio['original_graph_same_seeds_rank']
        rows.append(row)
    lexical.close()
    write_json(directory / 'counterfactual.json', dict(question=data['question'], conditions=rows,
        local_synonym_edges=len(local_synonyms), total_synonym_edges=len(synonyms),
        removed_synonyms=[dict(entity=original.vs[original.es[e].target if original.es[e].source == d else original.es[e].source]['content'],
                              weight=original.es[e]['weight']) for e in local_synonyms],
        audit=dict(new_llm_calls=0, new_embeddings=0, original_ranks_reproduced=True,
                   interpretation='Deterministic graph/seed interventions on one reviewed case; no new QA or population claim')))
    print(json.dumps(rows, indent=2), flush=True)


def baseline_depth(method, output):
    """Measure native passage rankings and audit their original top-five outputs."""
    import io
    from experiments.mine_cases import manifest, read_lines
    from optimization.retriever.hipporag import CacheMissGuard, load_memory, load_query_embeddings
    from optimization.run_graph import retrieval_config

    output.mkdir(parents=True, exist_ok=True)
    entry = manifest(BASE)['one_shot', 'Qwen_Qwen3.5-4B', '2WikiMultiHopQA', method]
    reference_path = Path(entry['directory']) / 'retrieval.jsonl'
    references = list(read_lines(reference_path))
    assert len(references) == len({r['case']['case_id'] for r in references}) == 1000
    runtime = output / ('depth_' + method)
    runtime.mkdir(exist_ok=True)
    stream = io.StringIO()
    if method == 'dense':
        from utils.models import HuggingFaceEmbedder
        from baseline.dense import _cosine
        from main import _load_groups
        _, common = retrieval_config(SimpleNamespace(task='2WikiMultiHopQA', output_root=runtime,
            generator_base_url='http://127.0.0.1:1/v1',
            path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
            data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
        group, = list(_load_groups(common))
        documents = list(group.memory_items)
        embed = HuggingFaceEmbedder('Qwen/Qwen3-Embedding-0.6B')
        cache = runtime / 'passages.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                assert saved['texts'].tolist() == documents
                vectors = saved['vectors']
        else:
            vectors = np.asarray(embed(documents), dtype=np.float64)
            np.savez_compressed(cache, texts=np.asarray(documents), vectors=vectors)
        norm = np.linalg.norm(vectors, axis=1)
        ids = {text: i for i,text in enumerate(documents)}
        assert len(ids) == len(documents)
        path = output / 'depth_dense.jsonl'
        rows = list(read_lines(path)) if path.exists() else []
        completed = {r['case_id']:r for r in rows}
        assert len(rows) == len(completed)
        for ref in references:
            case = ref['case']
            cid, query = case['case_id'], case['question']
            expected = [ids[r['text']] for r in ref['retrieved']]
            if cid in completed:
                assert completed[cid]['question'] == query
                completed[cid]['original_top_five'] = expected
                continue
            q = np.asarray(embed([query])[0], dtype=np.float64)
            denominator = norm*np.linalg.norm(q)
            scores = np.divide(vectors @ q, denominator, out=np.zeros(len(vectors)), where=denominator>0)
            if not rows:
                actual = np.asarray([_cosine(q,v) for v in vectors])
                np.testing.assert_allclose(scores, actual, atol=1e-12, rtol=1e-12)
            selected = np.argsort(-scores, kind='stable')[:15].tolist()
            row = dict(case_id=cid, question=query, gold=[ids[t] for t in case['gold_passages']],
                       selected=selected, original_top_five=expected)
            with path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            rows.append(row)
            if len(rows)%25 == 0:
                print(method, 'scored', len(rows), '/1000', flush=True)
        summarize_baseline_depth(method, output, rows, references, reference_path,
            'Original HuggingFaceEmbedder and corpus order. Cosine arithmetic vectorized in float64 and checked against baseline _cosine. Embeddings recomputed; no LLM or QA calls.')
        return
    if method == 'hipporag2':
        config, _ = retrieval_config(SimpleNamespace(task='2WikiMultiHopQA', output_root=runtime,
            generator_base_url='http://127.0.0.1:1/v1',
            path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
            data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
        meta = json.loads((GRAPH / '2WikiMultiHopQA/memory/hipporag-2wikimultihopqa/graph.json').read_text())
        memory = load_memory(config, Path(meta['source_graph']).parent.parent, runtime)
        native, generator = memory._memory, memory._generator
        load_query_embeddings(native, QUERIES / '2WikiMultiHopQA/hipporag-2wikimultihopqa/queries.npz')
    else:
        from baseline.catrag_memory import CatRAGMemory
        source = BASE / 'catrag_hypermem_top5_seed42_20260912_h100_batch8/catrag_top5/2WikiMultiHopQA/memory/hipporag-2wikimultihopqa'
        memory = CatRAGMemory(source, stream, group_id='hipporag-2wikimultihopqa',
            generator_model='Qwen/Qwen3-30B-A3B-Instruct-2507',
            generator_base_url='http://127.0.0.1:1/v1', embedding_model='Qwen/Qwen3-Embedding-0.6B',
            load_existing=True)
        native, generator = memory.native, memory.native.llm_model
        # Native reload rebuilds provenance maps but omits this initialization.
        native.ent_node_to_fact_ids = {}
        native.prepare_retrieval_objects()
        native.query_to_embedding_store = str(runtime / 'queries.pkl')
        cache = Path(generator.cache_file_name)
        destination = runtime / cache.name
        if not destination.exists():
            assert not cache.with_name(cache.name + '-wal').exists()
            shutil.copy2(cache, destination)
        generator.cache_file_name = str(destination)
    guard = CacheMissGuard(generator)
    if hasattr(generator, 'async_openai_client'):
        generator.async_openai_client.chat.completions.create = guard.reject
    def reject(*args, **kwargs):
        raise RuntimeError('Baseline depth requires cached embeddings')
    if method == 'hipporag2':
        native.embedding_model.batch_encode = reject
    else:
        import torch
        assert torch.cuda.is_available(), 'Native CatRAG NER embedding requires GPU replay'
    path = output / ('depth_' + method + '.jsonl')
    rows = list(read_lines(path)) if path.exists() else []
    completed = {r['case_id']: r for r in rows}
    assert len(completed) == len(rows)
    try:
        texts = native.chunk_embedding_store.get_all_id_to_rows()
        ids = {r['content']: key for key,r in texts.items()}
        assert len(ids) == len(texts)
        for ref in references:
            case = ref['case']
            cid, query = case['case_id'], case['question']
            expected = [ids[r['text']] for r in ref['retrieved']]
            if cid in completed:
                assert completed[cid]['question'] == query and completed[cid]['selected'][:5] == expected
                continue
            assert all(query in native.query_to_embedding[k] for k in ('triple', 'passage'))
            items = memory.retrieve(query, 15)
            guard.check()
            selected = [ids[r.text] for r in items]
            assert len(selected) == len(set(selected)) == 15
            assert selected[:5] == expected, (method, cid, 'native top-five differs')
            row = dict(case_id=cid, question=query, gold=[ids[t] for t in case['gold_passages']], selected=selected)
            with path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            rows.append(row)
            if len(rows) % 25 == 0:
                print(method, 'verified', len(rows), '/1000', flush=True)
        guard.check()
    finally:
        memory.close()
    summarize_baseline_depth(method, output, rows, references, reference_path,
        'Native graph rankings and cached LLM calls. CatRAG entity embeddings recomputed by its native encoder; HippoRAG embeddings frozen. No QA or context augmentation.')


def summarize_baseline_depth(method, output, rows, references, reference_path, note):
    import csv
    assert len(rows) == len({r['case_id'] for r in rows}) == 1000
    assert {r['case_id'] for r in rows} == {r['case']['case_id'] for r in references}
    summary = []
    for k in (5,10,15):
        pairs = [(set(r['gold']), set(r['selected'][:k])) for r in rows]
        summary.append(dict(method=method, cutoff=k, questions=len(rows),
            recall=100*sum(len(g&s)/len(g) for g,s in pairs)/len(rows),
            complete=100*sum(g <= s for g,s in pairs)/len(rows)))
    with (output / ('depth_' + method + '.csv')).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    differences = [r['case_id'] for r in rows if 'original_top_five' in r
                   and r['selected'][:5] != r['original_top_five']]
    write_json(output / ('depth_' + method + '_audit.json'), dict(complete=True,
        questions=len(rows), native_top_five_exact=not differences, differing_top_five= differences,
        new_llm_calls=0,
        embeddings_recomputed=method in ('dense', 'catrag'), reference=str(reference_path), note=note))


def plot_retrieval_depth(output):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    with (output / 'retrieval_depth.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['question_type'] == 'ALL'
                and r['method'] in ('bm25', 'without_recommendation', 'amor')]
    for method in ('dense', 'hipporag2'):
        audit = json.loads((output / f'depth_{method}_audit.json').read_text())
        assert audit['complete'] and audit['questions'] == 1000
        with (output / f'depth_{method}.csv').open() as stream:
            rows.extend(csv.DictReader(stream))
    styles = [
        ('bm25', 'BM25', '#E6A06B', 'o'),
        ('dense', 'Dense', '#DAB94E', 'D'),
        ('hipporag2', 'HippoRAG 2', '#8A94A3', '^'),
        ('without_recommendation', 'AMOR w/o recommendation', '#83B48A', 's'),
        ('amor', 'AMOR', '#547DA7', 'h'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    for method, label, color, marker in styles:
        values = sorted((r for r in rows if r['method'] == method), key=lambda r: int(r['cutoff']))
        assert [int(r['cutoff']) for r in values] == [5, 10, 15]
        assert all(int(r['questions']) == 1000 for r in values)
        for ax, metric in zip(axes, ('recall', 'complete')):
            ax.plot([5, 10, 15], [float(r[metric]) for r in values],
                    label=label, color=color, marker=marker, markersize=6,
                    linewidth=2 if method == 'amor' else 1.5)
    for ax, label, limits in zip(axes, ('Recall (%)', 'Complete (%)'), ((60, 95), (25, 85))):
        ax.set(xlabel='Retrieved passages', ylabel=label, xticks=[5, 10, 15],
               xlim=(4, 16), ylim=limits)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.grid(color='#D8DEE7', linewidth=.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color('#8993A4')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
    fig.tight_layout(pad=.45, w_pad=1.0)
    figures = output.parent / 'figures'
    figures.mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(figures / f'retrieval_depth.{extension}', dpi=200, bbox_inches='tight', pad_inches=.03)
    handles, labels = axes[0].get_legend_handles_labels()
    plt.close(fig)
    legend = plt.figure(figsize=(7.0, .8))
    legend.legend(handles, labels, loc='center', ncol=3, frameon=False,
                  prop={'size': 10, 'weight': 'bold'}, columnspacing=1.1, handlelength=1.6)
    for extension in ('pdf', 'png'):
        legend.savefig(figures / f'legend_retrieval_depth.{extension}', dpi=200,
                       bbox_inches='tight', pad_inches=.03)
    plt.close(legend)


def summarize_weight_controls(output):
    import csv
    rows = [json.loads(line) for line in (output / 'graph_controls.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['case_id'] for r in rows}) == 1000
    summary = []
    for kind in ['ALL'] + sorted({r['question_type'] for r in rows}):
        selected = [r for r in rows if kind == 'ALL' or r['question_type'] == kind]
        counts = dict(question_type=kind, questions=len(selected), binary_complete=0,
                      amor_complete=0, gained_complete=0, lost_complete=0,
                      partial_to_complete=0, none_to_complete=0)
        for r in selected:
            gold = set(r['gold'])
            binary, amor = (set(r['selected'][m]) for m in ('binary', 'amor'))
            before, after = gold <= binary, gold <= amor
            counts['binary_complete'] += int(before)
            counts['amor_complete'] += int(after)
            counts['gained_complete'] += int(after and not before)
            counts['lost_complete'] += int(before and not after)
            counts['partial_to_complete'] += int(after and not before and bool(gold & binary))
            counts['none_to_complete'] += int(after and not bool(gold & binary))
        summary.append(counts)
    with (output / 'weight_completion.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


def recommendation_cases(results, output):
    """Inspect actual source recommendations under fixed query processing."""
    from collections import defaultdict
    import igraph as ig
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.run_graph import retrieval_config

    rows = [json.loads(line) for line in (output / 'graph_controls.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['case_id'] for r in rows}) == 1000
    gains, losses = [], []
    for row in rows:
        gold = set(row['gold'])
        before = gold <= set(row['selected']['binary'])
        after = gold <= set(row['selected']['amor'])
        if after and not before:
            gains.append(row)
        elif before and not after:
            losses.append(row)
    selected = [next(r for r in gains if r['question_type'] == 'bridge_comparison')] + losses
    task, group = '2WikiMultiHopQA', 'hipporag-2wikimultihopqa'
    metadata = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
    protocol = json.loads((results / 'protocol.json').read_text())
    path = Path(protocol['graph_source']) / 'groups' / task / group / (protocol['graph_variant'] + '.pickle')
    graphs = {'amor': ig.Graph.Read_Pickle(str(path)),
              'binary': ig.Graph.Read_Pickle(str(COMPONENTS / task / group / 'without_projection.pickle'))}
    assert graphs['amor'].vs['name'] == graphs['binary'].vs['name']
    assert set(map(tuple, graphs['amor'].get_edgelist())) == set(map(tuple, graphs['binary'].get_edgelist()))
    assert set(graphs['binary'].es['weight']) == {1.0}
    config, _ = retrieval_config(SimpleNamespace(task=task, output_root=output,
        generator_base_url='http://127.0.0.1:1/v1',
        path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
        data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
    runtime = output / '2wiki_projection_1/runtime'
    memory = load_optimized_memory(config, GRAPH / task / 'memory' / group, runtime)
    try:
        from hipporag.utils.misc_utils import text_processing
        hippo = memory._memory
        guard = CacheMissGuard(memory._generator)
        load_query_embeddings(hippo, QUERIES / task / group / 'queries.npz')

        def reject(*args, **kwargs):
            raise RuntimeError('Recommendation cases must use frozen embeddings')

        hippo.embedding_model.batch_encode = reject
        contents = memory.contents
        sources = hippo.chunk_embedding_store.get_all_id_to_rows()
        entities = hippo.entity_embedding_store.get_all_id_to_rows()
        names = graphs['amor'].vs['name']
        positions = {name: i for i, name in enumerate(names)}
        entity_keys = {r['content']: key for key, r in entities.items()}
        supports = defaultdict(set)
        for key, record in contents.items():
            for triple in record['retained_triples']:
                supports[tuple(text_processing(triple))].add(key)
        source_rows = []
        totals = {m: dict(questions=0, gold_occurrences=0, directly_matched_gold=0,
                         selected_gold=0, selected_gold_without_matched_fact=0,
                         weight_gains_without_matched_fact=0, weight_gains_with_matched_fact=0)
                  for m in ('original', 'binary', 'amor')}
        for row in rows:
            if row['dense_fallback']:
                continue
            matched_sources = set()
            for fact in row['recognition']['facts_after_rerank']:
                triple = tuple(text_processing(fact))
                assert triple in supports, (row['case_id'], triple)
                matched_sources.update(supports[triple])
            gold = set(row['gold'])
            gained = (gold & set(row['selected']['amor'])) - set(row['selected']['binary'])
            for method, counts in totals.items():
                found = gold & set(row['selected'][method])
                counts['questions'] += 1
                counts['gold_occurrences'] += len(gold)
                counts['directly_matched_gold'] += len(gold & matched_sources)
                counts['selected_gold'] += len(found)
                counts['selected_gold_without_matched_fact'] += len(found - matched_sources)
                counts['weight_gains_without_matched_fact'] += len(gained - matched_sources)
                counts['weight_gains_with_matched_fact'] += len(gained & matched_sources)
            source_rows.append(dict(case_id=row['case_id'], recognized_fact_sources=sorted(matched_sources),
                                    gold=row['gold'], selected=row['selected']))
        write_json(output / 'recommendation_source_coverage.json', dict(
            interpretation='Source membership of recognized retained facts; not a new ranking baseline. '
                           'Counts exclude identical dense fallback queries. No QA or augmentation.',
            fallback_questions=sum(r['dense_fallback'] for r in rows), summary=totals, questions=source_rows))
        native = {r['_id']: r for r in json.loads(Path(
            'baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa.json').read_text())}
        captured = {}
        old_ppr, old_recognition = hippo.run_ppr, hippo.rerank_facts

        def capture_ppr(reset_prob, damping=None):
            captured['reset'] = reset_prob.copy()
            captured['damping'] = damping
            result = old_ppr(reset_prob, damping)
            captured['ranking'], captured['scores'] = result
            return result

        def capture_recognition(*args, **kwargs):
            result = old_recognition(*args, **kwargs)
            captured['recognition'] = result[2]
            return result

        hippo.run_ppr, hippo.rerank_facts = capture_ppr, capture_recognition
        cases = []
        for row in selected:
            conditions = {}
            query = row['question']
            assert native[row['case_id']]['question'] == query
            reset = None
            for method, graph in graphs.items():
                hippo.graph = graph
                captured.clear()
                retrieved = memory.retrieve(query, 5)
                guard.check()
                keys = [r.metadata['source_passage'] for r in retrieved]
                assert keys == row['selected'][method]
                assert json.loads(json.dumps(captured['recognition'])) == row['recognition']
                if reset is None:
                    reset = captured['reset'].copy()
                else:
                    np.testing.assert_array_equal(reset, captured['reset'])
                graph_rank = {hippo.passage_node_keys[int(i)]: rank + 1
                              for rank, i in enumerate(captured['ranking'])}
                assert all(graph_rank[k] == v for k, v in row['gold_graph_ranks'][method].items())
                conditions[method] = dict(selected=keys,
                    selected_titles=[sources[k]['content'].splitlines()[0] for k in keys],
                    supporting_passages_found=len(set(keys) & set(row['gold'])),
                    graph_ranks={k: graph_rank[k] for k in set(row['gold']) |
                                 set(row['selected']['binary']) | set(row['selected']['amor'])})
            source_details = []
            for key in dict.fromkeys(row['gold'] + row['selected']['binary'] + row['selected']['amor']):
                triples = set(tuple(text_processing(t)) for t in contents[key]['retained_triples'])
                neighbors = []
                for seed in row['entity_seeds']:
                    entity = seed['entity']
                    i, j = positions[entity_keys[entity]], positions[key]
                    eid = graphs['amor'].get_eid(i, j, directed=False, error=False)
                    if eid < 0:
                        continue
                    facts = sorted(t for t in triples if entity in (t[0], t[2]))
                    contributions = [1 / (len({entity_keys[t[0]], entity_keys[t[2]]} | supports[t]) - 1)
                                     for t in facts]
                    actual = float(graphs['amor'].es[eid]['weight'])
                    np.testing.assert_allclose(actual, 1 + sum(contributions), atol=1e-12)
                    neighbors.append(dict(entity=entity, weight=actual, unit_weight=1.0,
                                          facts=facts, fact_contributions=contributions))
                source_details.append(dict(key=key, title=sources[key]['content'].splitlines()[0],
                    original_text=sources[key]['content'], annotated_support=key in row['gold'],
                    ranks={m: conditions[m]['selected'].index(key)+1 if key in conditions[m]['selected'] else None
                           for m in conditions}, seed_connections=neighbors))
            interventions = []
            if row is selected[0]:
                target, = (set(row['gold']) & set(row['selected']['amor'])) - set(row['selected']['binary'])
                target_detail, = [r for r in source_details if r['key'] == target]
                for label, base, replacement in (
                        ('amor_with_target_connections_unit', 'amor', 'binary'),
                        ('binary_with_target_connections_restored', 'binary', 'amor')):
                    changed = graphs[base].copy()
                    changed_edges = []
                    for connection in target_detail['seed_connections']:
                        entity_key = entity_keys[connection['entity']]
                        i, j = positions[entity_key], positions[target]
                        eid = changed.get_eid(i, j, directed=False)
                        source_eid = graphs[replacement].get_eid(i, j, directed=False)
                        weight = graphs[replacement].es[source_eid]['weight']
                        changed_edges.append(dict(entity=connection['entity'], source=target,
                                                  before=changed.es[eid]['weight'], after=weight))
                        changed.es[eid]['weight'] = weight
                    hippo.graph = changed
                    captured.clear()
                    retrieved = memory.retrieve(query, 5)
                    guard.check()
                    np.testing.assert_array_equal(reset, captured['reset'])
                    assert json.loads(json.dumps(captured['recognition'])) == row['recognition']
                    keys = [r.metadata['source_passage'] for r in retrieved]
                    graph_rank = {hippo.passage_node_keys[int(i)]: rank + 1
                                  for rank, i in enumerate(captured['ranking'])}
                    interventions.append(dict(condition=label, changed_edges=changed_edges,
                        selected=keys, selected_titles=[sources[k]['content'].splitlines()[0] for k in keys],
                        target_graph_rank=graph_rank[target],
                        target_final_rank=keys.index(target)+1 if target in keys else None,
                        supporting_passages_found=len(set(keys) & set(row['gold']))))
            cases.append(dict(case_id=row['case_id'], question=query, question_type=row['question_type'],
                answer=native[row['case_id']]['answer'], recognition=row['recognition'],
                entity_seeds=row['entity_seeds'], conditions=conditions, sources=source_details,
                edge_weight_interventions=interventions))
        write_json(output / 'recommendation_cases.json', dict(
            selection='First bridge-comparison gain in original evaluation order, plus every completeness loss.',
            gained_case_ids=[r['case_id'] for r in gains], lost_case_ids=[r['case_id'] for r in losses],
            graph_nodes=graphs['amor'].vcount(), graph_edges=graphs['amor'].ecount(),
            same_nodes_and_edges=True, same_initial_scores=True, exact_saved_rankings=True,
            checked_fact_weight_contributions=True, new_llm_calls=guard.misses, new_embeddings=0,
            cases=cases))
        for case in cases:
            print(case['question'], flush=True)
            print(json.dumps(case['conditions'], indent=2), flush=True)
    finally:
        memory.close()


def memory_case_traces(results, output, selection):
    """Replay reviewed memory cases using frozen recognition and embeddings."""
    from collections import defaultdict
    from tempfile import TemporaryDirectory
    import igraph as ig
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.run_graph import retrieval_config

    requests = json.loads(selection.read_text())
    groups = defaultdict(list)
    for request in requests:
        groups[request['task'], request['group_id']].append(request)
    protocol = json.loads((results / 'protocol.json').read_text())
    reports = []
    for (task, group), cases in groups.items():
        config, _ = retrieval_config(SimpleNamespace(task=task.replace('_', ' '), output_root=output,
            generator_base_url='http://127.0.0.1:1/v1',
            path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
            data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
        with TemporaryDirectory(prefix='amor-case-trace-') as temporary:
            runtime = Path(temporary)
            cache, = (GRAPH / task / 'runtime' / group / 'llm_cache').glob('*.sqlite')
            assert not cache.with_name(cache.name + '-wal').exists()
            (runtime / 'llm_cache').mkdir()
            shutil.copy2(cache, runtime / 'llm_cache' / cache.name)
            memory = load_optimized_memory(config, GRAPH / task / 'memory' / group, runtime)
            try:
                hippo = memory._memory
                graph_path = (Path(protocol['graph_source']) / 'groups' / task / group
                              / (protocol['graph_variant'] + '.pickle'))
                graph = ig.Graph.Read_Pickle(str(graph_path))
                assert graph.vs['name'] == hippo.graph.vs['name']
                hippo.graph = graph
                guard = CacheMissGuard(memory._generator)
                load_query_embeddings(hippo, QUERIES / task / group / 'queries.npz')

                def reject(*args, **kwargs):
                    raise RuntimeError('Case analysis must use frozen embeddings')

                hippo.embedding_model.batch_encode = reject
                old_ppr, old_recognition = hippo.run_ppr, hippo.rerank_facts
                captured = {}

                def capture_ppr(reset_prob, damping=None):
                    captured['reset'] = reset_prob.copy()
                    result = old_ppr(reset_prob, damping)
                    captured['ranking'], captured['scores'] = result
                    return result

                def capture_recognition(*args, **kwargs):
                    result = old_recognition(*args, **kwargs)
                    captured['recognition'] = result[2]
                    return result

                hippo.run_ppr, hippo.rerank_facts = capture_ppr, capture_recognition
                entities = hippo.entity_embedding_store.get_all_id_to_rows()
                entity_by_key = {key: row['content'] for key, row in entities.items()}
                for request in cases:
                    captured.clear()
                    case_path = output / 'cases' / (request['name'] + '.json')
                    case = json.loads(case_path.read_text())
                    reference = case['conditions']['libra']['input']
                    question = reference['case']['question']
                    selected = memory.retrieve(question, 5)
                    guard.check()
                    expected = [r['metadata']['source_passage'] for r in reference['retrieved']
                                if 'source_passage' in r['metadata']]
                    actual = [r.metadata['source_passage'] for r in selected]
                    assert actual == expected, (request['name'], actual, expected)
                    seeds, ranking = [], {}
                    if 'reset' in captured:
                        reset = captured['reset'] / captured['reset'].sum()
                        seeds = [dict(entity=entity_by_key[graph.vs[i]['name']], score=float(reset[i]))
                                 for i in np.flatnonzero(reset) if graph.vs[i]['name'] in entity_by_key]
                        ranking = {hippo.passage_node_keys[int(index)]: rank
                                   for rank, index in enumerate(captured['ranking'], 1)}
                    selected_sources = []
                    for rank, key in enumerate(actual, 1):
                        source = memory.contents[key]
                        selected_sources.append(dict(key=key, final_rank=rank, graph_rank=ranking.get(key),
                            original_source_text=source['original_source_text'],
                            source_position=source['source_position'], retained_triples=source['retained_triples']))
                    reports.append(dict(name=request['name'], task=task, case_id=request['case_id'],
                        question=question, recognition=captured.get('recognition'), entity_seeds=seeds,
                        selected=selected_sources, reference=str(case_path), graph=str(graph_path),
                        exact_saved_selection=True, new_llm_calls=guard.misses, new_embeddings=0))
                    print(request['name'], 'exact selection replayed', flush=True)
            finally:
                memory.close()
    assert len(reports) == len(requests)
    write_json(output / 'memory_case_traces.json', reports)


def prepare_memory_interventions(results, output):
    """Prepare population controls and source-verified local interventions."""
    from dataclasses import replace
    from tempfile import TemporaryDirectory
    import igraph as ig
    from experiments.runner import _read_retrieval_records
    from experiments.ablate_components import save_rows
    from experiments.mine_cases import csv_rows
    from optimization.retriever.hipporag import CacheMissGuard, load_optimized_memory, load_query_embeddings
    from optimization.retriever.query_fact_context import QueryFactContext
    from optimization.graph_construction.compiled_sources import compile_sources
    from optimization.run_graph import retrieval_config

    target = output / 'memory_interventions'
    target.mkdir(exist_ok=True)
    design = dict(tasks=['FactConsolidation-MH', 'FactConsolidation-SH'], questions_per_task=100,
        population_controls=dict(all_fact_candidates='FC-MH: all extracted recognition candidates; current graph and retained context facts',
            all_context_facts='FC-SH: all extracted facts from the same five sources; same embedding ranking and top-ten budget'),
        local_cases=dict(author_death='Replace the accepted current authorship with its extracted older counterpart at fixed fact score; restore',
            show_country='Remove only the appended current country fact; replace it with the extracted older country fact; restore'),
        fixed='Current graph, original queries, source corpus, embeddings, rank fusion, native reader prompts and scoring',
        local_scope='Preselected diagnostic cases; not a prevalence estimate or a deployable intervention',
        result_root=str(results), new_extraction=False, new_embeddings=False)
    design_path = target / 'design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text()) == design
    else:
        write_json(design_path, design)
    protocol = json.loads((results / 'protocol.json').read_text())
    entries = csv_rows(results / 'qa_results.csv')
    all_root = BASE / 'optimization_support_ablation_seed42_20260917/graph_all_index_all'
    prepared, traces = [], []
    for task in design['tasks']:
        entry, = [r for r in entries if r['setting'] == 'one_shot' and r['task'] == task
                  and r['variant'] == 'full' and r['model'] == 'Qwen/Qwen3.5-4B']
        rows = list(_read_retrieval_records(Path(entry['source']) / 'retrieval.jsonl'))
        assert len(rows) == 100 and {r.group_id for r in rows} == {'mab-0'}
        group = 'mab-0'
        config, _ = retrieval_config(SimpleNamespace(task=task, output_root=target,
            generator_base_url='http://127.0.0.1:1/v1',
            path='/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json',
            data_root=str(Path.cwd() / 'baseline_algorithms/HippoRAG/reproduce/dataset')))
        metadata = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
        openie, = Path(metadata['source_graph']).parent.parent.glob('openie_results_ner_*.json')
        documents = json.loads(openie.read_text())['docs']
        with TemporaryDirectory(prefix='amor-intervention-') as temporary:
            memories, guards = {}, {}
            try:
                for name, root in [('current', GRAPH), ('all', all_root)]:
                    runtime = Path(temporary) / name
                    cache, = (root / task / 'runtime' / group / 'llm_cache').glob('*.sqlite')
                    assert not cache.with_name(cache.name + '-wal').exists()
                    (runtime / 'llm_cache').mkdir(parents=True)
                    shutil.copy2(cache, runtime / 'llm_cache' / cache.name)
                    memory = load_optimized_memory(config, root / task / 'memory' / group, runtime)
                    memories[name] = memory
                    graph_path = Path(protocol['graph_source']) / 'groups' / task / group / (protocol['graph_variant'] + '.pickle')
                    graph = ig.Graph.Read_Pickle(str(graph_path))
                    assert graph.vs['name'] == memory._memory.graph.vs['name']
                    memory._memory.graph = graph
                    guards[name] = CacheMissGuard(memory._generator)
                    load_query_embeddings(memory._memory, QUERIES / task / group / 'queries.npz')

                    def reject(*args, **kwargs):
                        raise RuntimeError('Interventions must use frozen embeddings')

                    memory._memory.embedding_model.batch_encode = reject
                current, all_memory = memories['current'], memories['all']
                assert current.contents == all_memory.contents
                selector = QueryFactContext(current._memory, current.contents)
                all_selector = QueryFactContext(all_memory._memory, current.contents)
                from hipporag.utils.misc_utils import text_processing
                ordered = json.loads((GRAPH / task / 'memory' / group / 'lexical_source_keys.json').read_text())
                all_statements = [(d['idx'], tuple(text_processing(f))) for d in documents for f in d['extracted_triples']]
                all_contents = compile_sources(documents, ordered, all_statements, {}, text_processing)
                assert all(all_contents[k]['original_source_text'] == current.contents[k]['original_source_text'] for k in ordered)
                raw_selector = QueryFactContext(all_memory._memory, all_contents)
                captured = {}
                for name, memory in memories.items():
                    native = memory._memory.rerank_facts

                    def record_recognition(*args, _native=native, _name=name, **kwargs):
                        answer = _native(*args, **kwargs)
                        captured[_name] = answer[2]
                        return answer

                    memory._memory.rerank_facts = record_recognition
                population, local = [], {}
                variant = 'all_fact_candidates' if task.endswith('-MH') else 'all_context_facts'
                for row in rows:
                    query = row.case.question
                    centers = current.retrieve(query, 5)
                    replay = selector.render(query, centers)
                    assert [i.text for i in replay] == [i.text for i in row.retrieved], row.case.case_id
                    if task.endswith('-MH'):
                        changed_centers = all_memory.retrieve(query, 5)
                        changed = all_selector.render(query, changed_centers)
                    else:
                        changed_centers = centers
                        changed = raw_selector.render(query, centers)
                        assert [i.text for i in changed[:5]] == [i.text for i in replay[:5]]
                    assert len(changed_centers) == 5 and len(changed) <= 15
                    population.append(replace(row, retrieved=changed))
                    traces.append(dict(task=task, case_id=row.case.case_id, variant=variant,
                        current_recognition=captured.get('current'), all_recognition=captured.get('all'),
                        current_sources=[i.metadata['source_passage'] for i in centers],
                        changed_sources=[i.metadata['source_passage'] for i in changed_centers],
                        current_facts=[i.text for i in replay[5:]], changed_facts=[i.text for i in changed[5:]],
                        current_replay_exact=True))
                    if row.case.case_id == 'factconsolidation_mh_262k_no45':
                        old = ['James Joyce', 'is the author of', 'Dubliners']
                        source, = [d for d in documents if old in d['extracted_triples']]
                        assert '4873. The author of Dubliners is James Joyce.' in source['passage']
                        rerank = current._memory.rerank_facts

                        def replace_authorship(*args, **kwargs):
                            indices, facts, log = rerank(*args, **kwargs)
                            assert list(map(list, facts)) == [['dubliners', 'author', 'george eliot']]
                            return indices, [tuple(text_processing(old))], dict(log, intervention='older extracted authorship at unchanged current fact score')

                        current._memory.rerank_facts = replace_authorship
                        try:
                            old_centers = current.retrieve(query, 5)
                            old_context = selector.render(query, old_centers)
                        finally:
                            current._memory.rerank_facts = rerank
                        restored = selector.render(query, current.retrieve(query, 5))
                        assert [i.text for i in restored] == [i.text for i in replay]
                        local = dict(current=[row], old_authorship=[replace(row, retrieved=old_context)],
                                     restored=[replace(row, retrieved=restored)])
                    if row.case.case_id == 'factconsolidation_sh_262k_no12':
                        old = ['Witches of East End', 'created in', 'United States of America']
                        source, = [d for d in documents if old in d['extracted_triples']]
                        assert '1836. Witches of East End was created in the country of United States of America.' in source['passage']
                        index, = [i for i, item in enumerate(row.retrieved)
                                  if item.text == '(Witches of East End, created in, United Kingdom)']
                        assert index >= 5
                        removed = tuple(item for i, item in enumerate(row.retrieved) if i != index)
                        replaced = list(row.retrieved)
                        replaced[index] = replace(replaced[index], text='(' + ', '.join(old) + ')',
                            metadata=dict(replaced[index].metadata, intervention='older extracted country assertion',
                                          source_passages=[source['idx']]))
                        restored = list(replaced)
                        restored[index] = row.retrieved[index]
                        assert restored == list(row.retrieved)
                        local = dict(current=[row], remove_current_fact=[replace(row, retrieved=removed)],
                                     old_country_fact=[replace(row, retrieved=replaced)],
                                     restored=[replace(row, retrieved=restored)])
                for guard in guards.values():
                    guard.check()
                assert local
                for name, values in [(variant, population), *local.items()]:
                    path = target / 'inputs' / task / name / 'retrieval.jsonl'
                    save_rows(path, values)
                    prepared.append(dict(task=task, condition=name, questions=len(values), path=str(path),
                        scope='population' if name == variant else 'local_case'))
                print(task, 'prepared', len(population), 'population and', len(local), 'local conditions', flush=True)
            finally:
                for memory in memories.values():
                    memory.close()
    write_json(target / 'traces.json', traces)
    write_json(target / 'prepared.json', dict(complete=True, entries=prepared, new_llm_calls=0, new_embeddings=0,
        current_source_replays=200, native_context_constructor=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('2wiki_projection_1', 'locomo_adoption'))
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--results', type=Path, help='Verified result directory for current AMOR cases and graph')
    parser.add_argument('--compact', action='store_true', help='Show Dense, HippoRAG 2, and AMOR in one row')
    parser.add_argument('--graph-controls', action='store_true')
    parser.add_argument('--synonym-control', action='store_true')
    parser.add_argument('--recommendation-controls', action='store_true')
    parser.add_argument('--propagation-steps', action='store_true')
    parser.add_argument('--case-counterfactual', action='store_true')
    parser.add_argument('--baseline-depth', choices=('hipporag2', 'catrag', 'dense'))
    parser.add_argument('--plot-depth', action='store_true')
    parser.add_argument('--weight-completion', action='store_true')
    parser.add_argument('--recommendation-cases', action='store_true')
    parser.add_argument('--memory-case-traces', type=Path, help='Replay exported cases listed in this selection file')
    parser.add_argument('--prepare-memory-interventions', action='store_true')
    args = parser.parse_args()
    if args.prepare_memory_interventions:
        if args.results is None:
            parser.error('--prepare-memory-interventions requires --results')
        prepare_memory_interventions(args.results, args.output)
    elif args.memory_case_traces:
        if args.results is None:
            parser.error('--memory-case-traces requires --results')
        memory_case_traces(args.results, args.output, args.memory_case_traces)
    elif args.recommendation_cases:
        if args.results is None:
            parser.error('--recommendation-cases requires --results')
        recommendation_cases(args.results, args.output)
    elif args.plot_depth:
        plot_retrieval_depth(args.output)
    elif args.weight_completion:
        summarize_weight_controls(args.output)
    elif args.baseline_depth:
        baseline_depth(args.baseline_depth, args.output)
    elif args.case_counterfactual:
        if args.results is None:
            parser.error('--case-counterfactual requires --results')
        case_counterfactual(args.results, args.output)
    elif args.graph_controls or args.synonym_control or args.recommendation_controls or args.propagation_steps:
        if args.results is None:
            parser.error('--graph-controls requires --results')
        graph_controls(args.results, args.output, args.synonym_control, args.recommendation_controls, args.propagation_steps)
    elif args.tsne:
        plot_tsne(args.output, args.results, args.compact)
        if args.results is None and not args.compact:
            plot_conversation(args.output)
    elif args.case is None:
        parser.error('--case is required unless --tsne is used')
    elif args.plot:
        plot(args.case, args.output)
    else:
        extract(args.case, args.output, args.results)


def plot(name, output):
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    data = json.loads((output / name / 'geometry.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    gray, blue, orange, green = '#CBD1DA', '#547DA7', '#D9914C', '#3B8565'
    records = data['records']
    targets = [r for r in records if r['annotated_support']]
    if not targets:
        targets = [r for r in records if r['final_rank'] == 1]
    target = max(targets, key=lambda r: r['dense_rank'])
    neighborhood, = [n for n in data['neighborhoods'] if n['target'] == target['key']]
    target_label = ('Director biography' if name.startswith('2wiki') else 'Adoption conversation')
    labels = {r['key']: ('Film description' if r != target else target_label) for r in targets}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), gridspec_kw={'width_ratios': [1.0, 1.15]})
    ax = axes[0]
    x = np.asarray([r['cosine'] for r in records])
    y = np.asarray([r['graph_score'] for r in records]) * 1000
    ax.scatter(x, y, s=10, color=gray, alpha=.52, edgecolors='none', rasterized=True)
    dense = sorted(records, key=lambda r: r['dense_rank'])[:5]
    ax.scatter([r['cosine'] for r in dense], [r['graph_score'] * 1000 for r in dense],
               s=58, facecolors='none', edgecolors=orange, linewidths=1.6, zorder=3)
    for j, r in enumerate(targets):
        ax.scatter(r['cosine'], r['graph_score'] * 1000, s=95, marker='*',
                   color=blue, edgecolor='white', linewidth=.6, zorder=5)
        offset = (-15, 20) if r == target else (-8, -42)
        text = f"{labels[r['key']]}\nDense #{r['dense_rank']}  /  Graph #{r['graph_rank']}"
        ax.annotate(text, (r['cosine'], r['graph_score'] * 1000), xytext=offset,
                    textcoords='offset points', ha='right', va='center', fontsize=9,
                    fontweight='bold', arrowprops=dict(arrowstyle='-', color=blue, lw=.8),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.0))
    ax.set_xlabel('Cosine similarity to question')
    ax.set_ylabel(r'PageRank score ($\times 10^{-3}$)')
    ax.set_title('(a) Direct similarity and graph relevance', loc='left', fontsize=11, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.spines[['right', 'top']].set_visible(False)
    ax.margins(x=.12, y=.20)
    ax.grid(alpha=.16, linewidth=.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_title('(b) Contributions from connected entities', loc='left', fontsize=11, pad=12)
    incoming = neighborhood['incoming'][:4]
    positions = np.linspace(.83, .17, len(incoming))
    denom = neighborhood['graph_score']
    for item, ypos in zip(incoming, positions):
        fraction = item['contribution'] / denom
        ax.annotate('', xy=(.82, .5), xytext=(.35, ypos),
            arrowprops=dict(arrowstyle='-|>', lw=.8 + 8 * fraction, color=blue,
                            shrinkA=8, shrinkB=13, mutation_scale=12), zorder=1)
        ax.scatter(.35, ypos, s=145, color=green if item['initial_score'] > 0 else '#AFBBC8',
                   edgecolors='white', zorder=3)
        label = textwrap.fill(item['text'], 24, max_lines=3, placeholder='...')
        ax.text(.29, ypos, label, ha='right', va='center', fontsize=9, fontweight='bold')
        ax.text(.56, .5 + (ypos - .5) * .56, f'{fraction:.1%}', ha='center', va='center',
                fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=4)
    ax.scatter(.82, .5, marker='s', s=700, color=blue, edgecolor='white', zorder=3)
    ax.text(.83, .36, target_label + f"\nGraph rank {target['graph_rank']}", ha='center', va='top',
            fontsize=10, fontweight='bold')
    shown = sum(r['contribution'] for r in incoming) / denom
    direct = neighborhood['direct_restart'] / denom
    remainder = 1 - shown - direct
    ax.text(.05, -.02, f"Direct restart: {direct:.1%}    Remaining incoming / restart: {remainder:.1%}",
            fontsize=9, transform=ax.transAxes)
    ax.set(xlim=(0, 1.04), ylim=(0, 1))
    ax.axis('off')
    fig.subplots_adjust(left=.07, right=.98, bottom=.17, top=.76, wspace=.26)
    fig.suptitle(data['question'], fontsize=12, fontweight='bold', y=.98)
    accepted = data['recognition']['facts_after_rerank'][0]
    fig.text(.5, .89, 'Recognized fact: (' + ', '.join(accepted) + ')', ha='center', fontsize=10)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', markerfacecolor='none',
                    markeredgecolor=orange, label='Top 5 by embedding similarity'),
                Line2D([], [], marker='*', linestyle='', color=blue,
                    label='Annotated support' if name.startswith('2wiki') else 'Answer-bearing record'),
                Line2D([], [], marker='o', linestyle='', color=green, label='Entity with initial relevance')],
                loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.015), fontsize=9)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(figures / f'{name}_geometry.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # The two panels use exactly the same projection; only selected records change.
    xy = np.asarray([r['xy'] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True, sharey=True)
    for ax, title, field in zip(axes, ('Direct embedding matching', 'AMOR context recommendation'),
                               ('dense_rank', 'final_rank')):
        ax.scatter(xy[:, 0], xy[:, 1], s=9, color=gray, alpha=.42, edgecolor='none', rasterized=True)
        chosen = [r for r in records if r[field] is not None and r[field] <= 5]
        ax.scatter([r['xy'][0] for r in chosen], [r['xy'][1] for r in chosen],
                   s=75, facecolors='none', edgecolors=orange if field == 'dense_rank' else blue,
                   linewidth=1.8, zorder=4)
        for j, r in enumerate(targets):
            ax.scatter(*r['xy'], s=75, marker='*', color=green, zorder=5)
            offset = (-8, 22) if name == 'locomo_adoption' else (7, 13 + 18*j)
            ax.annotate(labels[r['key']], r['xy'], xytext=offset, textcoords='offset points',
                        ha='right' if name == 'locomo_adoption' else 'left',
                        fontsize=9, fontweight='bold', arrowprops=dict(arrowstyle='-', color=green),
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
        ax.scatter(*data['query_xy'], marker='X', s=95, color='#423C48', edgecolor='white', zorder=6)
        ax.annotate('Question', data['query_xy'], xytext=(5, -16), textcoords='offset points', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC 1 ({100 * data['pca_variance'][0]:.1f}% variance)")
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel(f"PC 2 ({100 * data['pca_variance'][1]:.1f}% variance)")
    fig.suptitle('Same frozen embeddings; different selected context', fontsize=12, fontweight='bold')
    fig.tight_layout()
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(figures / f'{name}_pca.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    write_analysis(data, target, neighborhood, output)


def write_analysis(data, target, neighborhood, output):
    name = data['name']
    top = neighborhood['incoming'][0]
    detail = dict(record=target['text'], cosine=target['cosine'], dense_rank=target['dense_rank'],
        graph_rank=target['graph_rank'], final_rank=target['final_rank'],
        binary_graph_rank=target['binary_rank'], top_incoming_entity=top['text'],
        top_incoming_share=top['contribution']/neighborhood['graph_score'],
        selected_case=True, attribution='Stationary incoming score, not causal path importance')
    write_json(output / (name + '_summary.json'), detail)


def generated_memory_case(output, entries, task, cid):
    """Verify exact saved memories and outputs for the manually selected case."""
    import pickle
    import sqlite3

    specifications = [
        ('mem0', 'Mem0', 'final_qwen3_30b_seed42_clean_20260910/mem0/2WikiMultiHopQA/mem0_indices/hipporag-2wikimultihopqa/collection/mem0/storage.sqlite',
         '74d1f135-b258-4249-a965-5d6c5f7c1eb6', 'data',
         'The Circus Cyclone is a 1925 American Western film written and directed by Albert S. Rogell',
         'Albert S. Rogell was an American film director born on August 21, 1901, in Oklahoma City, Oklahoma'),
        ('lightmem_offline', 'LightMem', 'lightmem_offline_20260911T184409Z/lightmem/2WikiMultiHopQA/lightmem_indices/hipporag-2wikimultihopqa/collection/hipporag-2wikimultihopqa/storage.sqlite',
         'ef872b5d-c562-453e-9536-0efe24a72912', 'memory',
         'Circus Cyclone is a 1925 American Western film directed by Albert S. Rogell',
         'Albert S. Rogell was born on August 21, 1901, in Oklahoma City')]
    with (output / 'failure_questions.jsonl').open() as stream:
        outcomes = next(r for line in stream if (r := json.loads(line))['case_id'] == cid)
    reports = []
    for method, label, relative, memory_id, field, selected_quote, omitted_quote in specifications:
        path = BASE / relative
        fields, match = set(), None
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
            for blob, in connection.execute('SELECT point FROM points'):
                point = pickle.loads(blob)
                fields.update(point.payload)
                if str(point.id) == memory_id:
                    match = point.payload
        assert match is not None and omitted_quote in match[field]
        retrieval = Path(entries['Qwen_Qwen3.5-4B', task, method]['directory']) / 'retrieval.jsonl'
        with retrieval.open() as stream:
            case = next(r for line in stream if (r := json.loads(line))['case']['case_id'] == cid)
        texts = [r['text'] for r in case['retrieved']]
        assert len(texts) == 5 and selected_quote in texts[0]
        assert all(omitted_quote not in text and 'Oklahoma City' not in text for text in texts)
        scores = {model: values[method]['score'] for model, values in outcomes['conditions'].items()}
        assert len(scores) == 4 and all(score == 0 for score in scores.values())
        reports.append(dict(method=method, label=label, store=str(path), payload_fields=sorted(fields),
                            memory_id=memory_id, stored_payload=match, retrieved_texts=texts,
                            selected_quote=selected_quote, omitted_quote=omitted_quote,
                            scores=scores, retrieval_file=str(retrieval)))
    write_json(output / 'generated_memory_case_audit.json', dict(case_id=cid, methods=reports,
        note='Exact excerpts manually selected and verified. No inferred source-passage mapping.'))
    return reports


def answer_comparisons(output):
    """Count paired native F1 comparisons without converting partial scores to labels."""
    import csv

    methods = ['bm25', 'dense', 'hipporag2', 'catrag', 'mem0', 'lightmem_offline', 'anchormem_official']
    counts = {m: dict(method=m, higher=0, equal=0, lower=0) for m in methods}
    per_model, questions = {}, set()
    with (output / 'failure_questions.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['task'] != '2WikiMultiHopQA':
                continue
            assert row['case_id'] not in questions
            questions.add(row['case_id'])
            assert len(row['conditions']) == 4
            for model, values in row['conditions'].items():
                for method in methods:
                    a, b = values['libra']['score'], values[method]['score']
                    outcome = 'higher' if a > b else 'lower' if a < b else 'equal'
                    counts[method][outcome] += 1
                    group = per_model.setdefault((model, method), dict(model=model, method=method, higher=0, equal=0, lower=0))
                    group[outcome] += 1
    assert len(questions) == 1000
    assert all(sum(r[k] for k in ('higher', 'equal', 'lower')) == 4000 for r in counts.values())
    for name, rows in [('answer_comparisons', list(counts.values())),
                       ('answer_comparisons_by_model', list(per_model.values()))]:
        with (output / (name + '.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def case_memory_vectors(output, cases):
    """Load stored vectors by exact memory identity, never by nearest passage."""
    import pickle
    import sqlite3
    import pandas as pd

    cache = output / 'generated_case_vectors.npz'
    descriptions = output / 'generated_case_vectors.json'
    if cache.exists() and descriptions.exists():
        metadata = json.loads(descriptions.read_text())
        for item in metadata:
            if item['rank'] is not None:
                assert cases[item['method']]['retrieved'][item['rank'] - 1]['text'] == item['rendered']
        with np.load(cache, allow_pickle=False) as saved:
            assert saved['keys'].tolist() == [r['key'] for r in metadata]
            return metadata, saved['vectors']
    audit = json.loads((output / 'generated_memory_case_audit.json').read_text())
    rows, vectors = [], []
    for report in audit['methods']:
        method = report['method']
        retrieved = cases[method]['retrieved']
        requested = {r['text']: i + 1 for i, r in enumerate(retrieved)}
        found = []
        path = Path(report['store'])
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
            for blob, in connection.execute('SELECT point FROM points'):
                point = pickle.loads(blob)
                payload = point.payload
                text = payload['data'] if method == 'mem0' else payload['memory']
                rendered = text if method == 'mem0' else f"{payload.get('time_stamp', '')} {payload.get('weekday', '')} {text}"
                rank = requested.get(rendered)
                if rank is None and str(point.id) != report['memory_id']:
                    continue
                if method == 'mem0' and rank is not None:
                    assert str(point.id) == retrieved[rank - 1]['metadata']['id']
                stored_vector = point.vector.get('') if isinstance(point.vector, dict) else point.vector
                vector = np.asarray(stored_vector, dtype=np.float32)
                assert vector.shape == (1024,) and np.isfinite(vector).all()
                found.append((dict(key=method + ':' + str(point.id), method=method, text=text,
                                   rendered=rendered, rank=rank, source=str(path),
                                   birthplace_memory=str(point.id) == report['memory_id']), vector))
        assert sorted(r['rank'] for r, _ in found if r['rank'] is not None) == list(range(1, 6))
        assert len(found) == 6
        for row, vector in sorted(found, key=lambda pair: pair[0]['rank'] or 6):
            rows.append(row)
            vectors.append(vector)
    directory = BASE / 'anchormem_full_seed42_20260911/anchormem/2WikiMultiHopQA/build_oom_retry/hipporag-2wikimultihopqa/Qwen_Qwen3-30B-A3B-Instruct-2507_Transformers_Qwen_Qwen3-Embedding-0.6B/event_embeddings/vdb_event.parquet'
    events = pd.read_parquet(directory).set_index('hash_id')
    for rank, item in enumerate(cases['anchormem_official']['retrieved'], 1):
        if item['metadata']['memory_type'] != 'event':
            continue
        key = item['metadata']['memory_id']
        event = events.loc[key]
        assert item['text'] == event['content']
        rows.append(dict(key='anchormem_official:' + key, method='anchormem_official',
                         text=item['text'], rendered=item['text'], rank=rank, source=str(directory)))
        vectors.append(event['embedding'])
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez_compressed(cache, keys=np.asarray([r['key'] for r in rows]), vectors=vectors)
    write_json(descriptions, rows)
    return rows, vectors


def plot_tsne(output, results=None, compact=False):
    """One fixed t-SNE fit per previously reviewed case, using original vectors."""
    import csv
    import textwrap
    import pandas as pd
    import igraph as ig
    import sklearn
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def read_lines(path):
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)

    output.mkdir(parents=True, exist_ok=True)
    with (BASE / 'paper_costs_20260925/data/one_shot_qa_cost.csv').open() as stream:
        entries = {(r['model'], r['task'], r['method']): r for r in csv.DictReader(stream)}
    requests = [('2WikiMultiHopQA', 'fcdafe320bdb11eba7f7acde48001122', 'albert s rogell', 'Albert S. Rogell')]
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'font.weight': 'bold', 'axes.titleweight': 'bold',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9)) if compact else plt.subplots(2, 4, figsize=(10.4, 5.2))
    methods = [('bm25', 'BM25', '#B59B52'), ('dense', 'Dense', '#CD8649'),
               ('hipporag2', 'HippoRAG 2', '#9B709A'), ('catrag', 'CatRAG', '#688F84'),
               ('mem0', 'Mem0', '#AA6871'), ('lightmem_offline', 'LightMem', '#738F5B'),
               ('anchormem_official', 'AnchorMem', '#7C829E'), ('libra', 'AMOR', '#547DA7')]
    all_cases = []
    for row, (task, cid, entity_text, entity_label) in enumerate(requests):
        path = Path(entries['Qwen_Qwen3.5-4B', task, 'libra']['directory']) / 'retrieval.jsonl'
        case = next(r for r in read_lines(path) if r['case']['case_id'] == cid)
        if results is not None:
            path = results / 'analysis/reviewed_cases.json'
            reviewed = json.loads(path.read_text())
            current, = [r for r in reviewed if r['case_id'] == cid
                        and r['model'] == 'Qwen_Qwen3.5-4B' and r['method'] == 'libra']
            case = current['context']
        group, question = case['group_id'], case['case']['question']
        metadata = json.loads((GRAPH / task / 'memory' / group / 'graph.json').read_text())
        source = Path(metadata['source_graph']).parent
        records = pd.read_parquet(source / 'chunk_embeddings/vdb_chunk.parquet')
        entities = pd.read_parquet(source / 'entity_embeddings/vdb_entity.parquet')
        entity, = entities[entities['content'] == entity_text].to_dict('records')
        contents = json.loads(Path(metadata['compiled_source_file']).read_text())
        with np.load(QUERIES / task / group / 'queries.npz', allow_pickle=False) as cached:
            queries = cached['queries'].tolist()
            query = cached['passage'][queries.index(question)].reshape(-1)
            identity = json.loads(str(cached['identity']))
        vectors = np.stack(records.embedding).astype(np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(query), 1, atol=1e-5)
        dense_ranks = ranks(vectors @ query)
        dense_path = Path(entries['Qwen_Qwen3.5-4B', task, 'dense']['directory']) / 'retrieval.jsonl'
        dense_case = next(r for r in read_lines(dense_path) if r['case']['case_id'] == cid)
        frozen_top = records.iloc[np.argsort(dense_ranks)[:5]].content.tolist()
        native_top = [r['text'] for r in dense_case['retrieved']]
        native_set_equal = set(frozen_top) == set(native_top)
        native_order_equal = frozen_top == native_top
        selected = [r for r in case['retrieved'] if 'source_passage' in r.get('metadata', {})]
        keys = records.hash_id.tolist()
        texts = records.content.tolist()
        cases = {'libra': case}
        for method, _, _ in methods[:-1]:
            other_path = Path(entries['Qwen_Qwen3.5-4B', task, method]['directory']) / 'retrieval.jsonl'
            cases[method] = next(r for r in read_lines(other_path) if r['case']['case_id'] == cid)
        generated_memory_case(output, entries, task, cid)
        extra, extra_vectors = case_memory_vectors(output, cases)
        all_keys = keys + [r['key'] for r in extra] + ['question:' + cid, entity['hash_id']]
        memory_positions = {(r['method'], r['rank']): len(keys) + i for i, r in enumerate(extra) if r['rank'] is not None}
        selections = {}
        for method, _, _ in methods:
            if method == 'libra':
                selections[method] = [keys.index(r['metadata']['source_passage']) for r in selected]
            else:
                selections[method] = [memory_positions[method, rank] if (method, rank) in memory_positions
                                      else texts.index(item['text'])
                                      for rank, item in enumerate(cases[method]['retrieved'], 1)]
        assert all(len(selections[m]) == (10 if m == 'anchormem_official' else 5) for m, _, _ in methods)
        if task == '2WikiMultiHopQA':
            support = [texts.index(t) for t in case['case']['gold_passages']]
        else:
            support = [keys.index(selected[0]['metadata']['source_passage'])]
        target = max(support, key=lambda i: dense_ranks[i])
        # The displayed cases are fixed before fitting; no layout or seed search.
        cache_path = output / (cid + '_all_methods_tsne.npz')
        params = dict(n_components=2, perplexity=30, metric='cosine', init='pca',
                      learning_rate='auto', max_iter=1000, random_state=42,
                      method='barnes_hut', angle=.5, n_jobs=1)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                assert saved['keys'].tolist() == all_keys
                assert str(saved['case_id']) == cid
                assert json.loads(str(saved['parameters'])) == params
                xy = saved['xy']
                kl = float(saved['kl'])
        else:
            model = TSNE(**params)
            xy = model.fit_transform(np.vstack([vectors, extra_vectors, query, entity['embedding']]))
            kl = float(model.kl_divergence_)
            np.savez_compressed(cache_path, keys=np.asarray(all_keys), xy=xy, kl=kl, case_id=cid,
                                parameters=json.dumps(params))
        assert xy.shape == (len(all_keys), 2) and np.isfinite(xy).all()
        graph_path = Path(metadata['constructed_graph_file'])
        if results is not None:
            protocol = json.loads((results / 'protocol.json').read_text())
            graph_path = Path(protocol['graph_source']) / 'groups' / task / group / (protocol['graph_variant'] + '.pickle')
        graph = ig.Graph.Read_Pickle(graph_path)
        positions = {name: i for i, name in enumerate(graph.vs['name'])}
        edges = []
        for i in support:
            eid = graph.get_eid(positions[entity['hash_id']], positions[keys[i]], directed=False, error=False)
            assert eid >= 0
            triples = [triple for triple in contents[keys[i]]['retained_triples']
                       if entity_label in (triple[0], triple[2])]
            assert triples
            edges.append(dict(record=keys[i], entity=entity['hash_id'],
                              weight=graph.es[eid]['weight'], supporting_triples=triples))
        labels = []
        for i in support:
            item, = [r for r in selected if r['metadata']['source_passage'] == keys[i]]
            graph_rank = item['metadata']['fusion_ranks'].get('graph')
            label = ('Director biography' if i == target else 'Film description') if task == '2WikiMultiHopQA' else 'Adoption conversation'
            text = f'{label}\nSimilarity #{dense_ranks[i]} / AMOR #{selected.index(item)+1}'
            labels.append(dict(index=i, text=text, dense_rank=int(dense_ranks[i]),
                               graph_rank=graph_rank, final_rank=selected.index(item)+1))
        low, high = xy.min(axis=0), xy.max(axis=0)
        span = high - low
        displayed = [m for m in methods if m[0] in ('dense', 'hipporag2', 'libra')] if compact else methods
        for col, (method, title, color) in enumerate(displayed):
            ax = axes.flat[col]
            ax.scatter(xy[:len(keys), 0], xy[:len(keys), 1], s=6, c='#BDC5CF',
                       alpha=.5, linewidths=0, rasterized=True)
            if not compact:
                ax.scatter(xy[len(keys):-2, 0], xy[len(keys):-2, 1], marker='s', s=12,
                           c='#8D98A4', alpha=.8, linewidths=0, zorder=3)
            chosen = selections[method]
            ax.scatter(xy[chosen, 0], xy[chosen, 1], s=100, facecolors='none',
                       edgecolors=color, linewidths=1.65, zorder=5)
            ax.scatter(xy[support, 0], xy[support, 1], s=34, c='#DEAD48',
                       edgecolors='#876922', linewidths=.6, zorder=6)
            ax.scatter(*xy[-2], marker='*', s=100, c='#222222',
                       edgecolors='white', linewidths=.4, zorder=7)
            if method == 'libra':
                for i in support:
                    ax.plot([xy[i, 0], xy[-1, 0]], [xy[i, 1], xy[-1, 1]],
                            color='#3B8565', lw=1.4, linestyle='--', zorder=4)
                ax.scatter(*xy[-1], marker='D', s=50, c='#3B8565',
                           edgecolor='white', linewidths=.5, zorder=7)
                ax.annotate(entity_label, xy[-1], xytext=(.98, .22),
                            textcoords='axes fraction', ha='right', va='bottom', fontsize=10,
                            color='#246345', arrowprops=dict(arrowstyle='-', color='#3B8565', lw=.8),
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=2), zorder=8)
            # Generated memories are separate vectors, not assigned passage coordinates.
            for j, item in enumerate(labels):
                name = item['text'].split('\n')[0]
                position = item['index']
                if method in ('mem0', 'lightmem_offline'):
                    is_birthplace = item['index'] == target
                    position, = [len(keys) + i for i, r in enumerate(extra) if r['method'] == method
                                 and (r['birthplace_memory'] if is_birthplace else r['rank'] == 1)]
                    name = 'Birthplace memory' if is_birthplace else 'Film memory'
                    ax.scatter(*xy[position], marker='s', s=34, c='#DEAD48',
                               edgecolors='#876922', linewidths=.6, zorder=6)
                rank = f"Rank {chosen.index(position) + 1}" if position in chosen else 'Not selected'
                location = (.02, .96) if j == 0 else (.02, .04)
                ax.annotate(name + '\n' + rank, xy[position], xytext=location,
                            textcoords='axes fraction', ha='left',
                            va='top' if j == 0 else 'bottom', fontsize=10, color='#503F15',
                            arrowprops=dict(arrowstyle='-', color='#876922', lw=.8),
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=2), zorder=8)
            ax.set_xlim(low[0] - span[0]*.10, high[0] + span[0]*.10)
            ax.set_ylim(low[1] - span[1]*.15, high[1] + span[1]*.15)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title + f'  ({len(chosen)} selected)', color=color, fontsize=11, pad=5)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.text(.5, .99, '\n'.join(textwrap.wrap(question, width=85)),
                 ha='center', va='top', fontsize=11)
        all_cases.append(dict(task=task, case_id=cid, question=question, records=len(keys),
            corpus_file=str(source / 'chunk_embeddings/vdb_chunk.parquet'), source_predictions=str(path),
            graph_file=str(graph_path), displayed_methods=[m[0] for m in displayed],
            query_encoder=identity, labels=labels, actual_graph_edges=edges,
            parameters=params, sklearn_version=sklearn.__version__, kl_divergence=kl,
            native_dense_top5_set_equal=native_set_equal, native_dense_order_equal=native_order_equal,
            selections=selections, keys=all_keys, generated_memories=extra,
            rank_source='Displayed ranks and selected circles use each method\'s saved native retrieval output. Frozen cosine ranks are audit metadata only.',
            note='Shared qualitative projection of all 6119 source passages, 17 generated memories, query and entity. Generated subset contains actual returned memories plus two verified unselected birthplace memories; not the full generated-memory corpus. No clusters inferred or layout tuning.'))
        print('Finished fixed t-SNE', task, len(keys), flush=True)
    fig.subplots_adjust(left=.01, right=.99, bottom=.015, top=.90, hspace=.13, wspace=.13)
    figures = output.parent / 'figures'
    figures.mkdir(exist_ok=True)
    suffix = '_compact' if compact else ''
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(figures / f'context_tsne{suffix}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    plot_tsne_legend(figures, compact)
    write_json(output / f'context_tsne{suffix}_audit.json', all_cases)
    if results is None and not compact:
        answer_comparisons(output)


def plot_tsne_legend(figures, compact=False):
    """Render the standalone legend without changing the case visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    suffix = '_compact' if compact else ''
    fig = plt.figure(figsize=(6, .7) if compact else (12, .65))
    handles = [Line2D([], [], marker='o', linestyle='', color='#BDC5CF', label='Source text'),
        Line2D([], [], marker='s', linestyle='', color='#8D98A4', label='Generated memory'),
        Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor='#666666', label='Selected text'),
        Line2D([], [], marker='o', linestyle='', color='#DEAD48', label='Supporting text'),
        Line2D([], [], marker='s', linestyle='', color='#DEAD48', label='Corresponding memory'),
        Line2D([], [], marker='D', linestyle='', color='#3B8565', label='Entity'),
        Line2D([], [], marker='*', linestyle='', color='#333333', label='Question'),
        Line2D([], [], linestyle='--', color='#3B8565', label='AMOR graph connection')]
    if compact:
        handles = [h for h in handles if h.get_label() not in ('Generated memory', 'Corresponding memory')]
        handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=handles,
        ncol=3 if compact else 4, loc='center', frameon=False, handletextpad=.4,
        columnspacing=.9 if compact else 1.5,
        prop={'weight':'bold','size':9})
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(figures / f'legend_context_tsne{suffix}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_conversation(output):
    """Show the inspected adjacent reply and actual scores, without fitting embeddings."""
    import csv
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cid = 'conv-47-119'
    with (output / 'failure_questions.jsonl').open() as stream:
        audit = next(r for line in stream if (r := json.loads(line))['case_id'] == cid)
    with (BASE / 'paper_costs_20260925/data/one_shot_qa_cost.csv').open() as stream:
        entries = {r['method']: r for r in csv.DictReader(stream)
                   if r['model'] == 'Qwen_Qwen3.5-4B' and r['task'] == 'LoCoMo'}
    cases = {}
    for method, entry in entries.items():
        path = Path(entry['directory']) / 'retrieval.jsonl'
        with path.open() as stream:
            cases[method] = next(r for line in stream if (r := json.loads(line))['case']['case_id'] == cid)
    original = cases['libra']['retrieved'][0]['metadata']['original_source_text']
    augmented = cases['libra']['retrieved'][0]['text']
    excerpts = ["I've been teaching my siblings coding.",
                "What kind of programs are they making?",
                "They're starting small, making basic games and stories."]
    assert excerpts[0] in original
    assert all(x in augmented for x in excerpts)
    assert cases['hipporag2']['retrieved'][0]['text'] == original
    assert all(excerpts[1] in cases[m]['retrieved'][3]['text'] for m in ('dense', 'catrag'))
    assert all(excerpts[2] not in r['metadata']['original_source_text']
               for r in cases['libra']['retrieved'] if 'original_source_text' in r.get('metadata', {}))
    labels = [('bm25', 'BM25'), ('dense', 'Dense'), ('hipporag2', 'HippoRAG 2'),
              ('catrag', 'CatRAG'), ('mem0', 'Mem0'), ('lightmem_offline', 'LightMem'),
              ('anchormem_official', 'AnchorMem'),
              ('without_context_augmentation', 'AMOR w/o augmentation'), ('libra', 'AMOR')]
    values = []
    for key, _ in labels:
        scores = [model[key]['score'] for model in audit['conditions'].values()]
        assert len(scores) == 4 and len(set(scores)) == 1
        values.append(scores[0] * 100)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (ax, table) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={'width_ratios': [1.4, 1]})
    for a in (ax, table):
        a.set(xlim=(0, 1), ylim=(0, 1))
        a.axis('off')
    fig.suptitle(audit['question'], fontsize=10, fontweight='bold', y=.99)
    turns = [('John', excerpts[0], 'HippoRAG 2: rank 1', '#9B709A'),
             ('James', excerpts[1], 'Dense / CatRAG: rank 4', '#B16C3A'),
             ('John', excerpts[2], 'Reply added by AMOR context augmentation', '#547DA7')]
    for y, (speaker, quote, selection, color) in zip((.98, .65, .32), turns):
        ax.text(.03, y, speaker, fontsize=9, color=color, weight='bold', va='top')
        ax.text(.03, y-.073, '\n'.join(textwrap.wrap('"' + quote + '"', width=51)),
                fontsize=9, va='top', linespacing=1.2)
        ax.text(.03, y-.22, selection, fontsize=8, color=color, va='top', weight='bold')
    ax.annotate('', xy=(.006, .12), xytext=(.006, .92),
                arrowprops=dict(arrowstyle='->', color='#A7ADB5', lw=1))
    table.text(.0, .98, 'Method', weight='bold', va='top')
    table.text(.99, .98, 'F1 (%)', weight='bold', va='top', ha='right')
    table.axhline(.90, color='#777777', lw=.6)
    for i, ((_, label), value) in enumerate(zip(labels, values)):
        y = .865 - i*.094
        color = '#547DA7' if label.startswith('AMOR') else '#222222'
        table.text(0, y, label, va='top', fontsize=8.5, color=color, weight='bold' if label == 'AMOR' else 'normal')
        table.text(.99, y, f'{value:.1f}', va='top', ha='right', fontsize=8.5, color=color,
                   weight='bold' if label == 'AMOR' else 'normal')
    fig.subplots_adjust(left=.01, right=.98, top=.85, bottom=.04, wspace=.13)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for a in (ax, table):
        boxes = [t.get_window_extent(renderer) for t in a.texts]
        for i, box in enumerate(boxes):
            assert box.x1 <= a.bbox.x1 + 2, a.texts[i].get_text()
            assert all(not box.overlaps(other) for other in boxes[i+1:]), a.texts[i].get_text()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(output.parent / 'figures' / f'failure_cases.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    write_json(output / 'conversation_context_audit.json', dict(case_id=cid, question=audit['question'],
        excerpts=excerpts, selected_contexts=cases, predictions=audit['conditions'],
        note='One selected case. Scores are identical across all four LLMs; not a dataset average.'))


if __name__ == '__main__':
    main()
