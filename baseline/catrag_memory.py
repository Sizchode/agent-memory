"""CatRAG native graph construction and retrieval with separate cost records."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

from baseline.base import RetrievedItem
from baseline.graph_usage import GraphUsageRecorder


class CatRAGMemory:
    def __init__(self, memory_dir, usage_stream, *, group_id, generator_model,
                 generator_base_url, embedding_model, seed=42, load_existing=False):
        source = Path(__file__).resolve().parents[1] / "baseline_algorithms/CatRAG/src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from catrag import CatRAG
        from catrag.utils.config_utils import BaseConfig

        self.memory_dir = Path(memory_dir)
        self.group_id = group_id
        self.usage = GraphUsageRecorder(usage_stream)
        self.load_existing = load_existing
        if load_existing:
            # Load the saved effective config; do not silently relabel old memory.
            saved = json.loads((self.memory_dir / "adapter_settings.json").read_text())
            expected = dict(generator_model=generator_model, embedding_model=embedding_model, seed=seed)
            if any(saved[key] != value for key, value in expected.items()):
                raise ValueError("Requested backbone/seed does not match saved CatRAG memory")
            config = BaseConfig(**saved["native_config"])
            config.save_dir = str(self.memory_dir)
            config.cache_dir = str(self.memory_dir / "node_summaries")
            config.llm_base_url = generator_base_url
            graph_dir = self.memory_dir / f"{generator_model.replace('/', '_')}_{config.embedding_model_name.replace('/', '_')}"
            if not (graph_dir / "directed_graph.pickle").is_file():
                raise FileNotFoundError("No completed CatRAG graph to load")
        else:
            self.memory_dir.mkdir(parents=True, exist_ok=False)
            config = BaseConfig(save_dir=str(self.memory_dir), llm_name=generator_model,
                                llm_base_url=generator_base_url, seed=seed,
                                cache_dir=str(self.memory_dir / "node_summaries"),
                                embedding_model_name="Transformers/" + embedding_model)
            saved = dict(generator_model=generator_model, embedding_model=embedding_model,
                         seed=seed, native_config=asdict(config))
            (self.memory_dir / "adapter_settings.json").write_text(json.dumps(saved, indent=2) + "\n")
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        self.native = CatRAG(global_config=config)
        if load_existing:
            summary_file = Path(config.cache_dir) / f"node_summaries_final_{config.dataset}.json"
            # Native constructor loads the graph but leaves node_abs_store=None.
            # Read its completed summaries directly; never regenerate them here.
            self.native.node_abs_store = json.loads(summary_file.read_text())
        # H100 co-location OOM at native batch=64; batching only, no truncation.
        self.native.embedding_model.batch_size = 8
        self.usage.record(dict(event="runtime_setting", method="catrag", group_id=group_id,
                               embedding_batch_size=8, reason="H100 embedding OOM at batch 64"))
        # Serving compatibility only. All native token limits and prompts survive.
        self.native.llm_model.llm_config.generate_params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}}
        self.usage.record(dict(event="stage", method="catrag", stage="initialize",
                               group_id=group_id, elapsed_seconds=perf_counter() - start,
                               loaded_existing=load_existing))

    @contextmanager
    def _measure(self, stage, case_id=None):
        llm = self.native.llm_model
        clients = [llm.openai_client, llm.async_openai_client]
        originals = [(client.chat.completions, client.chat.completions.create) for client in clients]
        for completions, call in originals:
            completions.create = self.usage.wrap(call, method="catrag", stage=stage,
                                               group_id=self.group_id, case_id=case_id)
        start = perf_counter()
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            for completions, call in originals:
                completions.create = call
            self.usage.record(dict(event="stage", method="catrag", stage=stage,
                                   group_id=self.group_id, case_id=case_id, status=status,
                                   elapsed_seconds=perf_counter() - start))

    def build(self, memory_items):
        if self.load_existing:
            raise RuntimeError("Existing-memory mode does not permit rebuilding")
        with self._measure("build"):
            self.native.index(list(memory_items))

    def retrieve(self, query, top_k, *, case_id=None):
        with self._measure("access", case_id):
            result = self.native.retrieve([query], num_to_retrieve=top_k)[0]
        return [RetrievedItem(text=text, score=float(score))
                for text, score in zip(result.docs, result.doc_scores, strict=True)]

    def close(self):
        # The caller owns the usage stream and the embedding model lifetime.
        self.native.llm_model.openai_client.close()
