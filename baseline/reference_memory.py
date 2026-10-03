"""Construction adapters for the released AnchorMem and StructMem implementations.

These do not choose a new cross-method retrieval budget or run an LLM judge.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from time import perf_counter

from baseline.generation_usage import GenerationUsageTracker
from baseline.official import LightMemBaseline, _prepend, _OFFICIAL_ALGORITHMS


class AnchorMemBuilder:
    def __init__(self, *, output_dir, generator_model, generator_base_url,
                 embedding_model, seed=42, cache_only=False):
        _prepend(_OFFICIAL_ALGORITHMS / "AnchorMem")
        from AnchorMem import AnchorMem
        from src.utils.config_utils import BaseConfig

        # Settings from AnchorMem/main.py's released LoCoMo entry point, not
        # the different defaults of its generic BaseConfig constructor.
        config = BaseConfig(
            save_dir=str(output_dir), llm_name=generator_model,
            llm_base_url=generator_base_url,
            embedding_model_name=f"Transformers/{embedding_model}",
            dataset="locomo", seed=seed, temperature=0.0,
            fact_sim_threshold=0.85, related_fact_top_k=3,
            linking_top_k=5, retrieval_top_k=10, qa_top_k=5,
            embedding_batch_size=8, max_qa_steps=3, max_new_tokens=None,
        )
        self.memory = AnchorMem(global_config=config)
        if cache_only:
            def reject_new_generation(*args, **kwargs):
                raise RuntimeError("Cache-only recovery encountered an uncached generation request")
            self.memory.llm_model.openai_client.chat.completions.create = reject_new_generation
        self.usage = GenerationUsageTracker()
        self.usage.install(self.memory.llm_model.openai_client)

    def build_conversation(self, sample):
        from src.datasets.locomo10_loader import make_docs_from_locomo10_conversations

        # The official formatter supports all turns. Do not consult QA gold
        # evidence to decide which conversations may enter memory.
        docs, _ = make_docs_from_locomo10_conversations(
            [{"sample_id": sample["sample_id"], "conversation": sample["conversation"]}],
            per_session=True, only_referenced=False, chunk_size=3, overlap=1,
        )
        return self.build_documents(docs)

    def build_documents(self, docs):
        """Use the released index(List[str]) API without synthetic dialogue roles."""
        self.memory.global_config.corpus_len = len(docs)
        started = perf_counter()
        self.memory.index(docs)
        result = json.loads(Path(self.memory.fact_results_path).read_text())
        return {"index_seconds": perf_counter() - started, "input_chunks": len(docs),
                "stored_chunks": len(result["docs"]),
                "stored_facts": sum(len(doc["extracted_facts"]) for doc in result["docs"]),
                "stored_events": len(result["events"])}

    def efficiency_metrics(self):
        # Upstream cost_stats can include cache hits or attempted calls.
        # Our counter measures actual successful client responses separately.
        return {"generation_client": self.usage.snapshot(),
                "official_cost_statistics": self.memory.cost_stats}

    def close(self):
        self.memory.llm_model.openai_client.close()


class StructMemBuilder(LightMemBaseline):
    def __init__(self, official_config, *, group, output_dir):
        if not group.memory_timestamps or not group.dialogue_turns:
            raise ValueError("StructMem document ingestion requires an approved time-binding protocol")
        _prepend(_OFFICIAL_ALGORITHMS / "LightMem" / "src")
        from lightmem.memory.prompts import (
            LoCoMo_Event_Binding_factual, LoCoMo_Event_Binding_relational,
        )

        config = deepcopy(official_config)
        self.output_dir = Path(output_dir)
        self.details_path = self.output_dir / "details"
        self.summary_path = self.output_dir / "summaries"
        collection = group.group_id
        config["extraction_mode"] = "event"
        config["embedding_retriever"]["configs"].update(
            path=str(self.details_path), collection_name=collection, on_disk=True,
        )
        config["summary_retriever"] = deepcopy(config["embedding_retriever"])
        config["summary_retriever"]["configs"].update(
            path=str(self.summary_path), collection_name=collection + "_summary",
        )
        self.summary_config = deepcopy(config)
        self.summary_statistics = None
        super().__init__(
            config, memory_timestamps=group.memory_timestamps,
            dialogue_turns=group.dialogue_turns,
            extraction_prompt={"factual": LoCoMo_Event_Binding_factual,
                               "relational": LoCoMo_Event_Binding_relational},
        )

    def build(self, memory_items):
        started = perf_counter()
        super().build(memory_items)
        extracted = perf_counter()

        # The released script summarizes a copy of pre-update memories, then
        # updates the original. Keep that ordering and separate summary flags.
        summary_input = self.output_dir / "summary_input"
        shutil.copytree(self.details_path, summary_input)
        self._memory.summary_retriever.client.close()
        config = deepcopy(self.summary_config)
        config["embedding_retriever"]["configs"]["path"] = str(summary_input)
        summary_memory = LightMemBaseline(config)
        try:
            # Upstream uses a module-level cursor. A new conversation must
            # start from its own first entry, as in the released subprocesses.
            import lightmem.memory.lightmem as native
            native.GLOBAL_LAST_SUMMARY_TIME = None
            summary_started = perf_counter()
            summary_memory._memory.summarize(
                retrieval_scope="global", time_window=3600,
                top_k_seeds=15, process_all=True,
            )
            summary_seconds = perf_counter() - summary_started
            self.summary_statistics = summary_memory.efficiency_metrics()
        finally:
            summary_memory._memory.summary_retriever.client.close()
            summary_memory.close()

        update_times = self.consolidate()
        return {"extraction_seconds": extracted - started,
                "summary_seconds": summary_seconds, **update_times,
                "build_seconds": perf_counter() - started}

    def efficiency_metrics(self):
        return {**super().efficiency_metrics(),
                "summary_generation_statistics": self.summary_statistics}

    def close(self):
        self._memory.summary_retriever.client.close()
        super().close()
