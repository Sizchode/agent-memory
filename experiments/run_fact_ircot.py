"""Use the existing batched official IRCoT controller with explicit fact graphs."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np

from experiments.runner import _read_retrieval_records
from optimization import ircot
from optimization.retriever.fact_incidence import UniformFactSeeds, install_fact_graph, load_fact_memory
from optimization.retriever.hipporag import CacheMissGuard
from optimization.run_graph import BASE, SOURCE, TASKS, retrieval_config


ROOT = BASE / "optimization_ircot_fact_graph_seed42_20260919"
GRAPH = BASE / "optimization_graph_index_factorial_seed42_20260919/graph"
PREVIOUS = BASE / "optimization_graph_edges_seed42_20260919"
VARIANTS = ("bm25", "fact_graph", "fact_graph_without_synonyms")
CAPS = (1, 3, 5)


def endpoint(job_id):
    import requests

    saved = json.loads((ircot.SERVICE_ROOT / "generator.json").read_text())
    assert saved["job_id"] == str(job_id) and saved["model"] == ircot.GENERATOR
    response = requests.get(saved["base_url"] + "/models", timeout=30,
                            headers={"Authorization": "Bearer local-graph-generator"})
    response.raise_for_status()
    assert ircot.GENERATOR in [row["id"] for row in response.json()["data"]]
    os.environ["OPENAI_API_KEY"] = "local-graph-generator"
    return saved["base_url"]


def elasticsearch_endpoint():
    from elasticsearch import Elasticsearch

    saved = json.loads((ircot.SERVICE_ROOT / "elasticsearch.json").read_text())
    client = Elasticsearch([saved["host"]], scheme="http", port=saved["port"], timeout=30)
    try:
        assert client.info()["version"]["number"] == "7.10.2"
    except BaseException:
        client.close()
        raise
    return saved, client


def prepare():
    tasks = []
    for task in TASKS:
        slug = task.replace(" ", "_")
        groups = ircot.groups_for(task)
        original = list(_read_retrieval_records(SOURCE / "hipporag2" / slug / "retrieval.jsonl"))
        assert [(r.group_id, r.case) for r in original] == [(g.group_id, c) for g in groups for c in g.cases]
        assert all((GRAPH / slug / group.group_id / "graph.pickle").is_file() for group in groups)
        tasks.append(dict(task=task, questions=len(original), groups=len(groups)))
    config = ircot.official_config()
    retrieval = config["models"][config["start_state"]]
    assert retrieval["retrieval_count"] == 6 and retrieval["global_max_num_paras"] == 15
    revision = subprocess.check_output(["git", "-C", str(ircot.OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(seed=42, caps=CAPS, variants=VARIANTS, tasks=tasks, models=ircot.MODELS,
        official_revision=revision, official_config=config, original_qa_prompts_and_scoring=True,
        batch_size=ircot.BATCH_SIZE, graph_source=str(GRAPH), engine="vllm", dtype="bfloat16",
        prefix_caching=True, max_model_len=32768, new_extraction=False, test_as_dev=True,
        reuse_legacy_trajectories=False, reuse_legacy_qa=False,
        reuse_recognition_cache=True, new_subquery_recognition="local Qwen3-30B-A3B; provider failures abort",
        graph_method="uniform accepted fact seeds on the full fact index; original PPR and raw passages",
        ablation="keep or delete inherited synonym edges; all source and fact memberships fixed",
        no_fusion_or_context_appendix=True, bm25_backend="official ElasticsearchRetriever")
    target = ROOT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == json.loads(json.dumps(protocol))
    else:
        ircot.write_json(target, protocol)


def load_backend(variant, config, task, group, directory):
    if variant == "bm25":
        return ircot.ElasticsearchMemory(task, group)
    if variant not in VARIANTS[1:]:
        raise ValueError("Unknown graph control")
    slug = task.replace(" ", "_")
    runtime = directory / "runtime" / slug / group.group_id
    if not (runtime / "llm_cache").exists():
        pilot_cache = ROOT / "pilot" / directory.name / "runtime" / slug / group.group_id / "llm_cache"
        source_cache = pilot_cache if pilot_cache.exists() else PREVIOUS / "runtime" / slug / group.group_id / "llm_cache"
        shutil.copytree(source_cache, runtime / "llm_cache")
    return load_fact_memory(config, SOURCE / "hipporag2" / slug / "hipporag_indices" / group.group_id,
                            runtime, GRAPH / slug / group.group_id / "graph.pickle",
                            keep_synonym_edges=variant == "fact_graph")


def check_graph_interface():
    import igraph as ig

    graph = ig.Graph(n=5, edges=[(4, 0), (4, 1), (4, 2), (0, 2), (0, 1)])
    graph.vs["name"] = ["a", "b", "p", "isolated", "fact"]
    graph.vs["statement"] = [None] * 4 + [("a", "r", "b")]
    graph.es["weight"] = [1, 1, 1, 1, 0.8]
    graph.es["edge_kind"] = ["statement_incidence"] * 3 + ["passage", "fact+synonym"]
    graph.es["passage_source"] = [None, None, None, "p", None]
    graph.es["synonym_score"] = [None, None, None, 0, 0.8]
    kwargs = dict(query="", link_top_k=5, query_fact_scores=np.array([0.4, 0.4]),
                  top_k_fact_indices=[0, 0], passage_node_weight=0.05)
    for keep in (True, False):
        original = ig.Graph(n=4)
        original.vs["name"] = graph.vs["name"][:4]
        hippo = SimpleNamespace(graph=original, fact_node_keys=["f"], passage_node_idxs=[2], passage_node_keys=["p"],
            global_config=SimpleNamespace(damping=0.5), ppr_time=0, run_ppr=lambda reset, damping: reset)
        install_fact_graph(hippo, graph, keep_synonym_edges=keep)
        assert hippo.graph.ecount() == (5 if keep else 4) and hippo.graph.degree(3) == 0
        search = hippo.graph_search_with_fact_entities
        np.testing.assert_array_equal(search(top_k_facts=[("a", "r", "b")] * 2, **kwargs), [0, 0, 0, 0, 1])
        for facts, error in (([], ValueError), ([("a", "other", "b")], KeyError)):
            try:
                search(top_k_facts=facts, **kwargs)
            except error:
                pass
            else:
                raise AssertionError("Invalid recognized facts must fail")
    assert graph.ecount() == 5


def verify_cached_retrieval(model, service):
    from utils.models import release_accelerator_memory

    checked = 0
    directory = ROOT / "verification" / model.replace("/", "_")
    for task in TASKS:
        slug = task.replace(" ", "_")
        config, _ = retrieval_config(SimpleNamespace(task=task, output_root=ROOT, generator_base_url=service,
            path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
            data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
        for variant, control in zip(VARIANTS[1:], ("uniform_facts", "without_synonym_edges"), strict=True):
            expected = {(row.group_id, row.case.case_id): row for row in
                        _read_retrieval_records(PREVIOUS / "main" / control / slug / "retrieval.jsonl")}
            for group in ircot.groups_for(task):
                memory = load_backend(variant, config, task, group, directory)
                try:
                    guard = CacheMissGuard(memory._generator)
                    for case in group.cases[:2]:
                        actual = tuple(memory.retrieve(case.question, 5))
                        guard.check()
                        saved = expected[group.group_id, case.case_id].retrieved
                        assert [item.text for item in actual] == [item.text for item in saved]
                        np.testing.assert_allclose([item.score for item in actual], [item.score for item in saved], rtol=1e-6, atol=1e-10)
                        checked += 1
                finally:
                    memory.close()
                    memory = guard = None
                    gc.collect()
                    release_accelerator_memory()
    assert checked == 60
    ircot.write_json(directory / "complete.json", dict(complete=True, retrievals=checked, recognition_cache_misses=0))


def run(model, generator_job, pilot):
    ircot.ROOT, ircot.CAPS, ircot.VARIANTS = ROOT, CAPS, VARIANTS
    ircot.prepare, ircot.load_backend = prepare, load_backend
    ircot.endpoint, ircot.elasticsearch_endpoint = endpoint, elasticsearch_endpoint
    phase = "pilot" if pilot else "main"
    directory = ROOT / phase / model.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / f"adapter_code_{os.environ['SLURM_JOB_ID']}.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in (Path(__file__), Path("optimization/ircot.py"),
                     Path("optimization/retriever/fact_incidence.py"), Path("optimization/retriever/hipporag.py"),
                     Path("experiments/runner.py"), Path("baseline/official.py")):
            archive.write(path, path.name if path.is_absolute() else str(path))
        assert archive.testzip() is None
    if pilot:
        check_graph_interface()
        verify_cached_retrieval(model, endpoint(generator_job))
    ircot.run(model, generator_job, pilot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ircot.MODELS, required=True)
    parser.add_argument("--generator-job", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.model, args.generator_job, args.pilot)
