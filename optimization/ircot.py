"""Batched execution of the published IRCoT participants with frozen graph backends."""

import argparse
import ast
from contextlib import redirect_stdout
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import socket
import subprocess
import sys
import zipfile
from time import perf_counter, sleep
from types import SimpleNamespace
from functools import lru_cache

BASE = Path("/oscar/scratch/zliu328/agent-memory-outputs")
CODE = BASE / "optimization_support_ablation_seed42_20260917/experiment_code"
OFFICIAL = CODE / "ircot"
INPUTS = BASE / "optimization_amem_transfer_seed42_20260918/inputs"
GRAPH = BASE / "optimization_retained_fact_index_seed42_20260914/statement_projection_loop_free_retained_index_rrf_window"
ROOT = BASE / "optimization_ircot_batched_seed42_20260918"
SERVICE_ROOT = ROOT
GRAPH_RETRIEVAL = "graph"
VLLM = Path("/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin")
CONFIG = OFFICIAL / "base_configs/ircot_codex_2wikimultihopqa.jsonnet"
GENERATOR = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODELS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}
VARIANTS = ("bm25", "optimized_graph")
CAPS = (1, 3, 5, 7)
BATCH_SIZE = 16
SEED = 42
RESPONSE_PREFIX = "IRCoT_RESPONSE "
ES_HOME = Path("/oscar/scratch/zliu328/agent-memory-envs/elasticsearch-7.10.2")


def setup():
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import hf_hub_download

    for model, revision in MODELS.items():
        if model.startswith("Qwen/Qwen3.5-"):
            for filename in ("preprocessor_config.json", "video_preprocessor_config.json"):
                hf_hub_download(model, filename, revision=revision)
    subprocess.run([sys.executable, "-m", "pip", "install", "elasticsearch==7.9.1", "matplotlib==3.10.6"], check=True)
    spec = importlib.util.find_spec("elasticsearch")
    serializer = Path(next(iter(spec.submodule_search_locations))) / "serializer.py"
    if "np.float_," in serializer.read_text():
        subprocess.run(["patch", str(serializer), str(Path.cwd() / "requirements/elasticsearch_numpy2.patch")], check=True)
    if not (ES_HOME / "bin/elasticsearch").exists():
        archive = ES_HOME.with_suffix(".tar.gz")
        temporary = archive.with_suffix(".download")
        subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--output", str(temporary),
            "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz"], check=True)
        temporary.replace(archive)
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(ES_HOME.parent)], check=True)
    prepare()


def serve_elasticsearch():
    port = 30000 + int(os.environ["SLURM_JOB_ID"]) % 10000
    host = socket.gethostbyname(socket.gethostname())
    local = Path(os.environ.get("TMPDIR", "/tmp")) / f"ircot-es-{os.environ['SLURM_JOB_ID']}"
    local.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "elasticsearch.json", dict(host=host, port=port,
        job_id=os.environ["SLURM_JOB_ID"], version="7.10.2", local_data=str(local)))
    os.environ["ES_JAVA_OPTS"] = "-Xms2g -Xmx2g -XX:ActiveProcessorCount=1 -Dlog4j2.formatMsgNoLookups=true"
    binary = ES_HOME / "bin/elasticsearch"
    os.execv(binary, [str(binary), "-Ediscovery.type=single-node", f"-Enetwork.host={host}",
        f"-Ehttp.port={port}", f"-Epath.data={local / 'data'}", f"-Epath.logs={local / 'logs'}",
        "-Enode.store.allow_mmap=false", "-Expack.ml.enabled=false"])


def elasticsearch_endpoint():
    from elasticsearch import Elasticsearch
    start = perf_counter()
    while perf_counter() - start < 600:
        path = SERVICE_ROOT / "elasticsearch.json"
        if path.exists():
            saved = json.loads(path.read_text())
            client = Elasticsearch([saved["host"]], scheme="http", port=saved["port"], timeout=30)
            try:
                info = client.info()
                assert info["version"]["number"] == "7.10.2"
                return saved, client
            except Exception:
                client.close()
        sleep(5)
    raise RuntimeError("Elasticsearch 7.10.2 did not become ready")


@lru_cache(maxsize=1)
def wiki_source_fields():
    from dataset_loader.loader import _format_passage
    path = Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset/2wikimultihopqa_corpus.json"
    corpus = json.loads(path.read_text())
    return {_format_passage(item["title"], item["text"]): (item["title"], item["text"]) for item in corpus}


def source_fields(group):
    if group.cases[0].metric == "hipporag":
        corpus = wiki_source_fields()
        return [corpus[text] for text in group.memory_items]
    return [("", text) for text in group.memory_items]


def index_elasticsearch():
    from elasticsearch.helpers import bulk
    from optimization.run_graph import TASKS
    server, client = elasticsearch_endpoint()
    tree = ast.parse((OFFICIAL / "retriever_server/build_index.py").read_text())
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == "paragraphs_index_settings" for target in node.targets)]
    assert len(assignments) == 1
    settings = ast.literal_eval(assignments[0].value)
    entries = []
    try:
        for task_index, task in enumerate(TASKS):
            for group_index, group in enumerate(groups_for(task)):
                name = f"ircot-{task_index}-{group_index}"
                fields = source_fields(group)
                if not client.indices.exists(index=name):
                    client.indices.create(index=name, body=settings)
                    documents = [dict(_op_type="create", _index=name, _id=str(index), _source=dict(
                        id=str(index), title=title, paragraph_text=text, paragraph_index=0,
                        url="", is_abstract=True)) for index, (title, text) in enumerate(fields)]
                    successes, errors = bulk(client, documents, raise_on_error=True)
                    assert successes == len(fields) and not errors
                    client.indices.refresh(index=name)
                assert client.count(index=name)["count"] == len(group.memory_items)
                entries.append(dict(task=task, group_id=group.group_id, index=name, sources=len(fields)))
        write_json(ROOT / "elasticsearch_indices.json", dict(complete=True, server=server,
            indices=entries, index_settings=settings, official_retriever=True,
            source_texts_and_order_unchanged=True, corpus_scope="project six-task frozen corpora"))
    finally:
        client.close()


class ElasticsearchMemory:
    def __init__(self, task, group):
        from retriever_server.elasticsearch_retriever import ElasticsearchRetriever
        manifest = json.loads((SERVICE_ROOT / "elasticsearch_indices.json").read_text())
        assert manifest["complete"]
        entry, = [item for item in manifest["indices"] if item["task"] == task and item["group_id"] == group.group_id]
        self.index, self.sources = entry["index"], group.memory_items
        self.retriever = ElasticsearchRetriever(host=manifest["server"]["host"], port=manifest["server"]["port"])
        assert self.retriever._es.count(index=self.index)["count"] == len(self.sources)

    def retrieve(self, query, top_k):
        from baseline.base import RetrievedItem
        rows = self.retriever.retrieve_paragraphs(corpus_name=self.index, query_text=query,
            query_title_field_too=True, max_buffer_count=100, max_hits_count=top_k)
        return [RetrievedItem(self.sources[int(row["id"])], row["score"]) for row in rows]

    def close(self):
        self.retriever._es.close()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def groups_for(task):
    with (INPUTS / (task.replace(" ", "_") + ".pkl")).open("rb") as stream:
        return pickle.load(stream)


def official_config():
    import _jsonnet
    return json.loads(_jsonnet.evaluate_file(str(CONFIG), ext_vars={
        "RETRIEVER_HOST": "http://127.0.0.1", "RETRIEVER_PORT": "0"}))


def prepare():
    from optimization.run_graph import SOURCE, TASKS
    from experiments.runner import _read_retrieval_records

    tasks = []
    for task in TASKS:
        slug = task.replace(" ", "_")
        groups = groups_for(task)
        original = list(_read_retrieval_records(SOURCE / "hipporag2" / slug / "retrieval.jsonl"))
        assert [(r.group_id, r.case) for r in original] == [(g.group_id, c) for g in groups for c in g.cases]
        for group in groups:
            metadata = json.loads((GRAPH / slug / "memory" / group.group_id / "graph.json").read_text())
            assert Path(metadata["source_graph"]).is_file()
            assert Path(metadata["retained_fact_keys_file"]).is_file()
        tasks.append(dict(task=task, questions=len(original), groups=len(groups)))
    config = official_config()
    assert config["models"][config["start_state"]]["retrieval_count"] == 6
    assert config["models"][config["start_state"]]["global_max_num_paras"] == 15
    revision = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    protocol = dict(seed=SEED, per_request_sampling_seed=SEED, batch_size=BATCH_SIZE,
        caps=list(CAPS), max_rounds=max(CAPS), variants=list(VARIANTS), models=MODELS,
        tasks=tasks, official_revision=revision, config=str(CONFIG), graph_root=str(GRAPH),
        new_extraction=False, reuse_legacy_trajectories=False, reuse_legacy_qa=False,
        prefix_caching=True, original_qa_prompts_and_scoring=True,
        engine="vllm", max_model_len=32768, dtype="bfloat16", bm25_backend="official ElasticsearchRetriever",
        elasticsearch_version="7.10.2", elasticsearch_client_version="7.9.1")
    if GRAPH_RETRIEVAL == "hybrid":
        from optimization.retriever.hybrid_graph import RANK_CONSTANT
        protocol["graph_retrieval"] = dict(method="reciprocal_rank_fusion", rank_constant=RANK_CONSTANT,
            rank_window=config["models"][config["start_state"]]["retrieval_count"],
            lexical_backend="official ElasticsearchRetriever", evidence="original sources without augmentation")
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
    print(json.dumps(dict(phase="prepare", complete=True, questions=sum(t["questions"] for t in tasks))))


def serve():
    port = 40000 + int(os.environ["SLURM_JOB_ID"]) % 10000
    write_json(ROOT / "generator.json", dict(base_url=f"http://{socket.getfqdn()}:{port}/v1",
        model=GENERATOR, job_id=os.environ["SLURM_JOB_ID"]))
    binary = VLLM / "vllm"
    os.environ["PATH"] = str(VLLM) + os.pathsep + os.environ["PATH"]
    os.execv(binary, [str(binary), "serve", GENERATOR, "--host", "0.0.0.0", "--port", str(port),
        "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768", "--seed", str(SEED), "--generation-config", "vllm",
        "--safetensors-load-strategy", "lazy",
        "--api-key", "local-graph-generator"])


def endpoint(job_id):
    import requests
    start = perf_counter()
    while perf_counter() - start < 1800:
        path = SERVICE_ROOT / "generator.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["job_id"] == job_id:
                try:
                    response = requests.get(saved["base_url"] + "/models", timeout=5,
                        headers={"Authorization": "Bearer local-graph-generator"})
                    response.raise_for_status()
                    assert GENERATOR in [item["id"] for item in response.json()["data"]]
                    os.environ["OPENAI_API_KEY"] = "local-graph-generator"
                    return saved["base_url"]
                except requests.RequestException:
                    pass
        sleep(5)
    raise RuntimeError("The requested recognition service did not become ready")


def worker(model):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, GenerationConfig
    from vllm import LLM, SamplingParams

    output = sys.stdout
    with redirect_stdout(sys.stderr):
        snapshot = Path(hf_hub_download(model, "config.json", revision=MODELS[model], local_files_only=True)).parent
        assert snapshot.name == MODELS[model]
        if (snapshot / "generation_config.json").exists():
            defaults = GenerationConfig.from_pretrained(snapshot)
        else:
            defaults = GenerationConfig.from_model_config(AutoConfig.from_pretrained(snapshot))
        # Transformers resolves unspecified generation fields at generate() time.
        for name, value in GenerationConfig._get_default_generation_params().items():
            if getattr(defaults, name, None) is None:
                setattr(defaults, name, value)
        assert defaults.no_repeat_ngram_size == 0 and defaults.num_beams == 1
        llm = LLM(model=str(snapshot), dtype="bfloat16", tensor_parallel_size=1, seed=SEED,
            max_model_len=32768, gpu_memory_utilization=0.85, enable_prefix_caching=True,
            max_num_seqs=BATCH_SIZE, max_num_batched_tokens=8192, language_model_only=True,
            safetensors_load_strategy="lazy")
        tokenizer = llm.get_tokenizer()
        metadata = dict(model=model, revision=MODELS[model], gpu=torch.cuda.get_device_name(),
            vllm=importlib.metadata.version("vllm"), transformers=importlib.metadata.version("transformers"),
            torch=torch.__version__, cuda=torch.version.cuda, seed=SEED, batch_size=BATCH_SIZE,
            prefix_caching=True, max_model_len=32768, language_model_only=True,
            generation_defaults=defaults.to_dict())
    print(RESPONSE_PREFIX + json.dumps(metadata), file=output, flush=True)

    def input_ids(item):
        if item["phase"] == "reason":
            return tokenizer(item["prompt"])["input_ids"]
        return tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True,
            enable_thinking=False, tokenize=True, return_dict=False)

    def generate(items):
        prompts, settings = [], []
        for item in items:
            generation = item["generation"]
            ids = input_ids(item)
            assert len(ids) + generation["max_tokens"] <= 32768
            sampling = dict(temperature=generation["temperature"], max_tokens=generation["max_tokens"],
                seed=SEED, repetition_penalty=defaults.repetition_penalty)
            if generation["temperature"] > 0:
                sampling.update(top_k=generation.get("top_k", defaults.top_k) or -1,
                                top_p=generation.get("top_p", defaults.top_p))
            else:
                sampling.update(top_k=-1, top_p=1.0)
            if item["phase"] == "reason":
                sampling["stop"] = ["\n"]
            prompts.append(dict(prompt_token_ids=ids))
            settings.append(SamplingParams(**sampling))
        start = perf_counter()
        generated = llm.generate(prompts, settings, use_tqdm=False)
        seconds = perf_counter() - start
        assert len(generated) == len(items)
        responses = []
        for item, prompt, response in zip(items, prompts, generated, strict=True):
            assert list(response.prompt_token_ids) == prompt["prompt_token_ids"]
            assert len(response.outputs) == 1
            completion = response.outputs[0]
            answer = completion.text.strip()
            if item["phase"] == "answer":
                assert answer, "Empty final QA completion"
            responses.append(dict(answer=answer, usage=dict(input_tokens=len(response.prompt_token_ids),
                output_tokens=len(completion.token_ids), cached_input_tokens=response.num_cached_tokens),
                generation_settings=item["generation"], finish_reason=completion.finish_reason))
        return dict(responses=responses, batch_seconds=seconds)

    for line in sys.stdin:
        request = json.loads(line)
        if request["phase"] == "close":
            break
        with redirect_stdout(sys.stderr):
            if request["phase"] == "benchmark":
                items = request["items"]
                llm.reset_prefix_cache()
                start = perf_counter()
                serial = [generate([item])["responses"][0] for item in items]
                serial_seconds = perf_counter() - start
                llm.reset_prefix_cache()
                batched = generate(items)
                cold = dict(serial_seconds=serial_seconds, batch_seconds=batched["batch_seconds"])
                llm.reset_prefix_cache()
                start = perf_counter()
                serial = [generate([item])["responses"][0] for item in items]
                serial_seconds = perf_counter() - start
                llm.reset_prefix_cache()
                batched = generate(items)
                result = dict(requests=len(items), serial_seconds=serial_seconds,
                    batch_seconds=batched["batch_seconds"], same_answers=[a["answer"] == b["answer"]
                        for a, b in zip(serial, batched["responses"], strict=True)],
                    batch_output_tokens=sum(r["usage"]["output_tokens"] for r in batched["responses"]),
                    first_pass=cold, warmed_up=True,
                    comparison="same vLLM engine, serial versus batched, caches reset before each")
            elif request["phase"] == "check_context":
                lengths = [len(input_ids(item)) for item in request["items"]]
                totals = [length + item["generation"]["max_tokens"]
                          for length, item in zip(lengths, request["items"], strict=True)]
                result = dict(input_tokens=lengths, max_model_len=32768,
                    overflow=[index for index, total in enumerate(totals) if total > 32768])
            else:
                result = generate(request["items"])
        print(RESPONSE_PREFIX + json.dumps(result), file=output, flush=True)


class Reader:
    def __init__(self, model):
        self.process = subprocess.Popen([str(VLLM / "python"), "-u", str(Path(__file__)),
            "worker", "--model", model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env=dict(os.environ, PATH=str(VLLM) + os.pathsep + os.environ.get("PATH", "")))
        try:
            self.metadata = self.read()
            assert self.metadata["revision"] == MODELS[model]
        except BaseException:
            self.close()
            raise

    def read(self):
        for line in self.process.stdout:
            if line.startswith(RESPONSE_PREFIX):
                return json.loads(line[len(RESPONSE_PREFIX):])
            print(line, end="", file=sys.stderr, flush=True)
        raise RuntimeError(f"vLLM worker exited: {self.process.wait(timeout=30)}")

    def request(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"phase":"close"}\n')
                self.process.stdin.flush()
            self.process.wait(timeout=60)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=30)


class PendingGeneration(Exception):
    pass


class GenerationSlot:
    def __init__(self):
        self.prompt = self.response = None

    def generate_text_sequence(self, prompt):
        from commaqa.inference.prompt_reader import fit_prompt_into_given_limit
        prompt = fit_prompt_into_given_limit(prompt.rstrip(), model_length_limit=8000,
            estimated_generation_length=300, demonstration_delimiter="\n\n\n", shuffle=False,
            remove_method="first", tokenizer_model_name="gpt2", last_is_test_example=True)
        if self.response is None:
            self.prompt = prompt
            raise PendingGeneration()
        assert prompt == self.prompt
        return [(self.response["answer"], 0.0)]


def load_backend(variant, config, task, group, directory):
    from optimization.retriever.hipporag import load_optimized_memory
    if variant == "bm25":
        return ElasticsearchMemory(task, group)
    slug = task.replace(" ", "_")
    frozen = GRAPH / slug / "memory" / group.group_id
    metadata = json.loads((frozen / "graph.json").read_text())
    runtime = directory / "runtime" / slug / group.group_id
    index = directory / "graph_indices" / slug / group.group_id
    index.mkdir(parents=True, exist_ok=True)
    graph = {key: value for key, value in metadata.items() if key not in ("rank_fusion", "compiled_source_file")}
    write_json(index / "graph.json", graph)
    weights = index / "edge_weights.npy"
    if not weights.exists():
        weights.symlink_to(frozen / "edge_weights.npy")
    assert weights.resolve() == (frozen / "edge_weights.npy").resolve()
    memory = load_optimized_memory(config, index, runtime)
    if GRAPH_RETRIEVAL == "hybrid":
        from optimization.retriever.hybrid_graph import HybridGraphMemory
        ordered = json.loads((frozen / "lexical_source_keys.json").read_text())
        controller = official_config()
        window = controller["models"][controller["start_state"]]["retrieval_count"]
        memory = HybridGraphMemory(memory, ordered, rank_window=window, lexical=ElasticsearchMemory(task, group))
    return memory


class Pipeline:
    def __init__(self, memory, group):
        from commaqa.inference import ircot
        from commaqa.inference.data_instances import StructuredDataInstance
        from commaqa.inference.model_search import ModelController
        config = official_config()
        classes = dict(retrieve_and_reset_paragraphs=ircot.RetrieveAndResetParagraphsParticipant,
            step_by_step_cot_gen=ircot.StepByStepCOTGenParticipant,
            step_by_step_exit_controller=ircot.StepByStepExitControllerParticipant)
        self.participants = {}
        models = dict(start_state=config["start_state"], end_state=config["end_state"])
        for name, raw in config["models"].items():
            arguments = dict(raw)
            kind = arguments.pop("name")
            if kind != "retrieve_and_reset_paragraphs":
                arguments["max_num_sentences"] = max(CAPS)
            if "prompt_file" in arguments:
                arguments["prompt_file"] = str(OFFICIAL / arguments["prompt_file"])
            participant = classes[kind](**arguments)
            self.participants[name] = participant
            models[name] = participant.query
        self.controller = ModelController(models, StructuredDataInstance)
        self.memory, self.sources = memory, group.memory_items
        self.positions = {text: index for index, text in enumerate(self.sources)}
        assert len(self.positions) == len(self.sources)
        fields = source_fields(group)
        self.titles = [title or f"Source {index}" for index, (title, _) in enumerate(fields)]
        self.paragraphs = [text for _, text in fields]
        self.source_positions = {(title, text): index for index, (title, text) in
                                 enumerate(zip(self.titles, self.paragraphs, strict=True))}
        assert len(self.source_positions) == len(self.sources)
        self.corpus = config["models"][config["start_state"]]["source_corpus_name"]
        self.calls = []

    def transport(self, url, params):
        assert params["corpus_name"] == self.corpus
        assert params["retrieval_method"] == "retrieve_from_elasticsearch"
        items = self.memory.retrieve(params["query_text"], params["max_hits_count"])
        self.calls.append(dict(query=params["query_text"], requested=params["max_hits_count"],
            source_indices=[self.positions[item.text] for item in items]))
        rows = [dict(title=self.titles[self.positions[item.text]], paragraph_text=self.paragraphs[self.positions[item.text]],
            corpus_name=self.corpus, id=str(self.positions[item.text]), score=item.score) for item in items]
        return SimpleNamespace(ok=True, json=lambda: dict(retrieval=rows))

    def batch(self, cases, reader, variant, group_id, pilot=False):
        from commaqa.inference import ircot
        from commaqa.inference.model_search import SearchState
        from baseline.graph_usage import GraphUsageRecorder
        from optimization.retriever.hipporag import GenerationFailureGuard
        guard = GenerationFailureGuard(self.memory._generator) if variant != "bm25" else None
        original_call = self.memory._generator.openai_client.chat.completions.create if guard else None
        states = [SearchState(self.controller.init_data(dict(qid=case.case_id, question=case.question,
            titles=[], paras=[], metadata={})), self.controller.start_state) for case in cases]
        traces = [dict(case_id=case.case_id, question=case.question, rounds=[]) for case in cases]
        batch_timings, prompts = [], []
        for round_number in range(1, max(CAPS) + 1):
            active = [index for index, state in enumerate(states) if state.next != self.controller.end_state]
            if not active:
                break
            slots, requests, notes = [], [], []
            for index in active:
                state = states[index]
                assert state.next == self.controller.start_state
                usage = io.StringIO()
                if guard:
                    self.memory._generator.openai_client.chat.completions.create = GraphUsageRecorder(usage).wrap(
                        original_call, method=variant, stage=f"ircot_round_{round_number}",
                        group_id=group_id, case_id=cases[index].case_id)
                self.calls = []
                old_transport = ircot.safe_post_request
                start = perf_counter()
                ircot.safe_post_request = self.transport
                try:
                    advanced = self.controller.execute(state)
                finally:
                    ircot.safe_post_request = old_transport
                assert len(advanced) == 1
                if guard:
                    guard.check()
                states[index] = advanced[0]
                slot = GenerationSlot()
                participant = self.participants[states[index].next]
                assert isinstance(participant, ircot.StepByStepCOTGenParticipant)
                participant.generator = slot
                try:
                    participant.query(states[index])
                except PendingGeneration:
                    pass
                else:
                    raise AssertionError("Expected exactly one pending reasoning request")
                assert slot.prompt is not None
                slots.append(slot)
                requests.append(dict(phase="reason", prompt=slot.prompt,
                    generation=dict(temperature=0.0, max_tokens=300)))
                notes.append(dict(retrievals=list(self.calls), retrieval_seconds=perf_counter() - start,
                    recognition=[json.loads(line) for line in usage.getvalue().splitlines()]))
            generated = reader.request(phase="generate", items=requests)
            batch_timings.append(dict(round=round_number, requests=len(requests), seconds=generated["batch_seconds"]))
            prompts.extend(requests)
            for index, slot, response, note in zip(active, slots, generated["responses"], notes, strict=True):
                slot.response = response
                self.participants[states[index].next].generator = slot
                generated_states = self.controller.execute(states[index])
                assert len(generated_states) == 1
                exited = self.controller.execute(generated_states[0])
                assert len(exited) == 1
                state = states[index] = exited[0]
                assert len(state.data["generated_sentences"]) == round_number
                assert len(state.data["paras"]) <= 15
                assert state.next in (self.controller.start_state, self.controller.end_state)
                traces[index]["rounds"].append(dict(note, round=round_number, reasoning=response,
                    generated_sentence=state.data["generated_sentences"][-1],
                    selected_sources=[self.source_positions[pair] for pair in
                        zip(state.data["titles"], state.data["paras"], strict=True)],
                    stopped=state.next == self.controller.end_state))
                if pilot:
                    traces[index]["rounds"][-1]["reasoning_prompt"] = slot.prompt
        assert all(state.next == self.controller.end_state for state in states)
        if guard:
            self.memory._generator.openai_client.chat.completions.create = original_call
        if pilot:
            self.verify_native_controller(cases, traces)
        return traces, batch_timings, prompts

    def verify_native_controller(self, cases, traces):
        from baseline.base import RetrievedItem
        from commaqa.inference import ircot
        from commaqa.inference.model_search import BestFirstDecomposer
        original_memory = self.memory
        try:
            for case, trace in zip(cases, traces, strict=True):
                for cap in CAPS:
                    calls = [call for step in trace["rounds"] for call in step["retrievals"]]
                    position = [0, 0]

                    def retrieve(query, count):
                        saved = calls[position[0]]
                        position[0] += 1
                        assert (query, count) == (saved["query"], saved["requested"])
                        return [RetrievedItem(self.sources[index]) for index in saved["source_indices"]]

                    def generate(prompt):
                        saved = trace["rounds"][position[1]]
                        position[1] += 1
                        slot = GenerationSlot()
                        slot.prompt, slot.response = saved["reasoning_prompt"], saved["reasoning"]
                        return slot.generate_text_sequence(prompt)

                    self.memory = SimpleNamespace(retrieve=retrieve)
                    for participant in self.participants.values():
                        if hasattr(participant, "max_num_sentences"):
                            participant.max_num_sentences = cap
                        if hasattr(participant, "generator"):
                            participant.generator = SimpleNamespace(generate_text_sequence=generate)
                    original_transport = ircot.safe_post_request
                    ircot.safe_post_request = self.transport
                    try:
                        state, _ = BestFirstDecomposer(self.controller).find_answer_decomp(dict(
                            qid=case.case_id, question=case.question, titles=[], paras=[], metadata={}))
                    finally:
                        ircot.safe_post_request = original_transport
                    rounds = min(cap, len(trace["rounds"]))
                    assert position[1] == rounds
                    assert position[0] == sum(len(step["retrievals"]) for step in trace["rounds"][:rounds])
                    selected = [self.source_positions[pair] for pair in
                        zip(state.data["titles"], state.data["paras"], strict=True)]
                    assert selected == trace["rounds"][rounds - 1]["selected_sources"]
                    assert state.data["generated_sentences"] == [step["generated_sentence"] for step in trace["rounds"][:rounds]]
        finally:
            self.memory = original_memory
            for participant in self.participants.values():
                if hasattr(participant, "max_num_sentences"):
                    participant.max_num_sentences = max(CAPS)


def evaluate_task(directory, task, reader, model, pilot):
    from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation, _score, evaluate_retrieval
    from optimization.report_results import TASK_METRICS, audited_score
    rows = list(_read_retrieval_records(directory / "retrieval.jsonl"))
    randomizer = random.Random(SEED)
    requests = [dict(phase="answer", messages=_answer_prompt(row.case, row.retrieved, randomizer)[0],
        generation=_official_generation(row.case)) for row in rows]
    answers, timings = [], []
    for start in range(0, len(requests), BATCH_SIZE):
        result = reader.request(phase="generate", items=requests[start:start + BATCH_SIZE])
        answers.extend(result["responses"])
        timings.append(dict(requests=len(result["responses"]), seconds=result["batch_seconds"]))
    output = directory / "evaluations" / model.replace("/", "_")
    output.mkdir(parents=True, exist_ok=True)

    class SavedAnswers:
        def __init__(self):
            self.index = 0

        def answer(self, messages, **generation):
            expected = requests[self.index]
            assert messages == expected["messages"] and generation == expected["generation"]
            answer = answers[self.index]
            self.index += 1
            return answer["answer"]

    adapter = SavedAnswers()
    summary = evaluate_retrieval(directory / "retrieval.jsonl", answer_model=adapter, output_dir=output, seed=SEED)
    assert adapter.index == len(rows) == len(answers)
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").open()]
    for row, prediction in zip(rows, predictions, strict=True):
        assert (row.group_id, row.case.case_id) == (prediction["group_id"], prediction["case_id"])
        assert prediction["metrics"] == _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
        assert [item.text for item in row.retrieved] == [item["text"] for item in prediction["retrieved"]]
    if not pilot:
        assert len(rows) == TASK_METRICS[task][0]
        score = audited_score(output, task, {(row.group_id, row.case.case_id) for row in rows})
    else:
        score = None
    with (output / "qa_usage.jsonl").open("w") as stream:
        for row, answer in zip(rows, answers, strict=True):
            stream.write(json.dumps(dict(answer["usage"], case_id=row.case.case_id,
                generation_settings=answer["generation_settings"], seed=SEED)) + "\n")
    write_json(directory / "qa_complete.json", dict(complete=True, questions=len(rows), pilot=pilot,
        score=score, summary=summary, native_evaluator=True, scores_recomputed=True, batch_timings=timings,
        input_tokens=sum(a["usage"]["input_tokens"] for a in answers),
        output_tokens=sum(a["usage"]["output_tokens"] for a in answers)))


def run(model, generator_job, pilot, render_context=None):
    from argparse import Namespace
    from optimization.run_graph import TASKS, SOURCE, retrieval_config
    from experiments.runner import RetrievedCase, _read_retrieval_records, _retrieval_record
    from baseline.base import RetrievedItem
    from utils.models import release_accelerator_memory
    prepare()
    service = endpoint(generator_job)
    directory = ROOT / ("pilot" if pilot else "main") / model.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    code_path = directory / "execution_code.zip"
    if code_path.exists():
        code_path = directory / f"execution_code_{os.environ['SLURM_JOB_ID']}.zip"
    with zipfile.ZipFile(code_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, "ircot.py")
    if not pilot:
        verified = json.loads((ROOT / "pilot" / model.replace("/", "_") / "complete.json").read_text())
        assert verified["complete"] and verified["pilot"] and verified["model"] == model
        assert verified["native_controller_equivalence_checked"]
        assert verified["seed"] == SEED and verified["caps"] == list(CAPS)
        assert verified["variants"] == list(VARIANTS) and verified["tasks"] == list(TASKS)
    reader = Reader(model)
    try:
        metadata_path = directory / "reader.json"
        if metadata_path.exists():
            metadata_path = directory / f"reader_{os.environ['SLURM_JOB_ID']}.json"
        write_json(metadata_path, reader.metadata)
        benchmark_prompts = []
        if not pilot and not (directory / "batch_benchmark.json").exists():
            pilot_directory = ROOT / "pilot" / model.replace("/", "_")
            for task in TASKS:
                for variant in VARIANTS:
                    group = groups_for(task)[0]
                    saved = pilot_directory / "traces" / variant / task.replace(" ", "_") / (group.group_id + ".json")
                    for trace in json.loads(saved.read_text())["traces"]:
                        for step in trace["rounds"]:
                            if len(benchmark_prompts) < BATCH_SIZE:
                                benchmark_prompts.append(dict(phase="reason", prompt=step["reasoning_prompt"],
                                    generation=dict(temperature=0.0, max_tokens=300)))
            assert len(benchmark_prompts) == BATCH_SIZE
            write_json(directory / "batch_benchmark.json", reader.request(phase="benchmark", items=benchmark_prompts))
        for task in TASKS:
            slug = task.replace(" ", "_")
            groups = groups_for(task)
            selected_groups = groups[:1] if pilot else groups
            config, _ = retrieval_config(Namespace(task=task, output_root=ROOT, generator_base_url=service,
                path="/oscar/scratch/zliu328/agent-memory-data/locomo/locomo10.json",
                data_root=str(Path.cwd() / "baseline_algorithms/HippoRAG/reproduce/dataset")))
            for variant in VARIANTS:
                collected = []
                for group in selected_groups:
                    cases = group.cases[:2] if pilot else group.cases
                    saved = directory / "traces" / variant / slug / (group.group_id + ".json")
                    if saved.exists():
                        existing = json.loads(saved.read_text())
                        assert existing["complete"] and existing["pilot"] == pilot
                        assert existing["case_ids"] == [case.case_id for case in cases]
                        traces = existing["traces"]
                    else:
                        memory = load_backend(variant, config, task, group, directory)
                        pipeline = None
                        traces, timings = [], []
                        try:
                            if variant != "bm25":
                                passages = memory._memory.chunk_embedding_store.get_all_id_to_rows()
                                assert {row["content"] for row in passages.values()} == set(group.memory_items)
                            pipeline = Pipeline(memory, group)
                            for start in range(0, len(cases), BATCH_SIZE):
                                part, timing, prompts = pipeline.batch(cases[start:start + BATCH_SIZE], reader,
                                    variant, f"{task}/{group.group_id}", pilot=pilot)
                                traces.extend(part)
                                timings.extend(timing)
                        finally:
                            try:
                                memory.close()
                            finally:
                                pipeline = memory = None
                                release_accelerator_memory()
                                import torch
                                print(json.dumps(dict(group=group.group_id, variant=variant,
                                    cuda_allocated_after_close=torch.cuda.memory_allocated())), flush=True)
                        write_json(saved, dict(complete=True, pilot=pilot, model=model, variant=variant,
                            case_ids=[case.case_id for case in cases], traces=traces, batch_timings=timings))
                    if pilot:
                        for trace in traces:
                            for step in trace["rounds"]:
                                if len(benchmark_prompts) < BATCH_SIZE:
                                    benchmark_prompts.append(dict(phase="reason", prompt=step["reasoning_prompt"],
                                        generation=dict(temperature=0.0, max_tokens=300)))
                    collected.extend((group, case, trace) for case, trace in zip(cases, traces, strict=True))
                for cap in CAPS:
                    target = directory / f"cap_{cap}" / variant / slug
                    target.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for group, case, trace in collected:
                        assert trace["case_id"] == case.case_id and trace["question"] == case.question
                        assert 1 <= len(trace["rounds"]) <= max(CAPS) and trace["rounds"][-1]["stopped"]
                        selected = trace["rounds"][min(cap, len(trace["rounds"])) - 1]["selected_sources"]
                        items = tuple(RetrievedItem(group.memory_items[i]) for i in selected)
                        if render_context is not None:
                            items = render_context(task, group, items)
                        rows.append(RetrievedCase(group.group_id, case, items, 15))
                    if not pilot:
                        expected = list(_read_retrieval_records(SOURCE / "hipporag2" / slug / "retrieval.jsonl"))
                        assert [(row.group_id, row.case) for row in rows] == [(row.group_id, row.case) for row in expected]
                    with (target / "retrieval.jsonl").open("w") as stream:
                        for row in rows:
                            stream.write(json.dumps(_retrieval_record(row)) + "\n")
                    if not (target / "qa_complete.json").exists():
                        evaluate_task(target, slug, reader, model, pilot)
                    print(json.dumps(dict(model=model, task=task, variant=variant, cap=cap,
                        questions=len(rows), pilot=pilot, qa_complete=True)), flush=True)
        if pilot:
            assert len(benchmark_prompts) == BATCH_SIZE
            write_json(directory / "batch_benchmark.json", reader.request(phase="benchmark", items=benchmark_prompts))
        write_json(directory / "complete.json", dict(complete=True, model=model, pilot=pilot,
            seed=SEED, caps=list(CAPS), variants=list(VARIANTS), tasks=list(TASKS),
            native_controller_equivalence_checked=pilot))
    finally:
        reader.close()


def report():
    prepare()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.runner import _read_retrieval_records, _score
    from optimization.report_results import TASK_METRICS, audited_score
    from optimization.run_graph import SOURCE, TASKS

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    scores = {}
    lines = ["# IRCoT round-cap results", "", "Seed 42; full six-task data; early stopping enabled.",
        "Each task uses its native metric. Scores are percentages, not averaged across tasks.",
        f"Graph artifacts: `{GRAPH}`.", ""]
    for model, revision in MODELS.items():
        slug = model.replace("/", "_")
        directory = ROOT / "main" / slug
        complete = json.loads((directory / "complete.json").read_text())
        metadata = json.loads((directory / "reader.json").read_text())
        assert complete["complete"] and not complete["pilot"] and complete["model"] == model
        assert complete["seed"] == SEED and complete["tasks"] == list(TASKS)
        assert complete["caps"] == list(CAPS) and complete["variants"] == list(VARIANTS)
        assert metadata["revision"] == revision
        scores[model] = {}
        for task in TASKS:
            task_slug = task.replace(" ", "_")
            count, metric = TASK_METRICS[task_slug]
            expected = list(_read_retrieval_records(SOURCE / "hipporag2" / task_slug / "retrieval.jsonl"))
            expected_keys = {(row.group_id, row.case.case_id) for row in expected}
            assert len(expected) == len(expected_keys) == count
            scores[model][task] = {}
            for variant in VARIANTS:
                curve = []
                for cap in CAPS:
                    target = directory / f"cap_{cap}" / variant / task_slug
                    marker = json.loads((target / "qa_complete.json").read_text())
                    assert marker["complete"] and not marker["pilot"] and marker["questions"] == count
                    rows = list(_read_retrieval_records(target / "retrieval.jsonl"))
                    assert [(r.group_id, r.case) for r in rows] == [(r.group_id, r.case) for r in expected]
                    output = target / "evaluations" / slug
                    with (output / "predictions.jsonl").open() as stream:
                        predictions = [json.loads(line) for line in stream]
                    for row, prediction in zip(rows, predictions, strict=True):
                        assert (row.group_id, row.case.case_id) == (prediction["group_id"], prediction["case_id"])
                        assert prediction["metrics"] == _score(row.case, prediction["prediction"], row.retrieved, row.top_k)
                        assert [item.text for item in row.retrieved] == [item["text"] for item in prediction["retrieved"]]
                    score = audited_score(output, task_slug, expected_keys)
                    assert score is not None and score == marker["score"]
                    curve.append(dict(cap=cap, score=score, metric=metric, questions=count))
                scores[model][task][variant] = curve
        lines += [f"## {model}", "", "| Task | Backend | Cap 1 | Cap 3 | Cap 5 | Cap 7 |",
                  "|---|---|---:|---:|---:|---:|"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for ax, task in zip(axes.flat, TASKS, strict=True):
            graph_label = "IRCoT + graph and BM25" if GRAPH_RETRIEVAL == "hybrid" else "IRCoT + graph and index"
            for variant, label, color in (("bm25", "IRCoT + BM25", "#3268a8"),
                    ("optimized_graph", graph_label, "#c15242")):
                curve = scores[model][task][variant]
                values = [100 * point["score"] for point in curve]
                ax.plot(CAPS, values, marker="o", label=label, color=color)
                lines.append("| " + task + " | " + label + " | " +
                    " | ".join(f"{value:.2f}" for value in values) + " |")
            ax.set_title(task)
            ax.set_xlabel("Maximum retrieval rounds")
            ax.set_ylabel(TASK_METRICS[task.replace(" ", "_")][1] + " (%)")
            ax.set_xticks(CAPS)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(model.split("/")[-1])
        fig.savefig(figures / f"{slug}.pdf")
        fig.savefig(figures / f"{slug}.png", dpi=160)
        plt.close(fig)
        lines.append("")
    write_json(ROOT / "comparison.json", dict(complete=True, seed=SEED, scores=scores,
        graph_root=str(GRAPH), graph_retrieval=GRAPH_RETRIEVAL,
        test_set_used_as_development_set=True, scores_recomputed=True))
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("setup", "prepare", "serve", "serve-elasticsearch", "index-elasticsearch", "worker", "pilot", "run", "report"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--generator-job-id")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--service-root", type=Path, default=SERVICE_ROOT)
    parser.add_argument("--graph-root", type=Path, default=GRAPH,
                        help="Existing graph/index artifacts; use a separate output root for ablations")
    parser.add_argument("--graph-retrieval", choices=("graph", "hybrid"), default="graph")
    args = parser.parse_args()
    ROOT, SERVICE_ROOT, GRAPH_RETRIEVAL = args.output_root, args.service_root, args.graph_retrieval
    if ROOT == SERVICE_ROOT and (GRAPH_RETRIEVAL == "hybrid" or args.graph_root != GRAPH):
        parser.error("Use a separate --output-root when changing graph artifacts or retrieval")
    GRAPH = args.graph_root
    if args.phase == "setup":
        setup()
    elif args.phase == "prepare":
        prepare()
    elif args.phase == "serve-elasticsearch":
        serve_elasticsearch()
    elif args.phase == "index-elasticsearch":
        index_elasticsearch()
    elif args.phase == "serve":
        serve()
    elif args.phase == "worker":
        worker(args.model)
    elif args.phase == "report":
        report()
    else:
        assert args.model and args.generator_job_id
        run(args.model, args.generator_job_id, pilot=args.phase == "pilot")
