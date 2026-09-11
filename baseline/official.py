"""Thin adapters over pinned official baseline source trees."""

from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

from .base import RetrievedItem
from .generation_usage import GenerationUsageTracker

_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_ALGORITHMS = _ROOT / "baseline_algorithms"


def _prepend(path: Path) -> None:
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class LightMemBaseline:
    def __init__(
        self,
        official_config: dict[str, Any],
        *,
        memory_timestamps: Sequence[str | None] = (),
        dialogue_turns: Sequence[Any] = (),
        extraction_prompt: str | None = None,
    ) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "LightMem" / "src")
        from lightmem.memory.lightmem import LightMemory
        self._memory = LightMemory.from_config(official_config)
        self._generation_usage = GenerationUsageTracker()
        self._generation_usage.install(self._memory.manager.client)
        self._memory_timestamps = tuple(memory_timestamps)
        self._dialogue_turns = tuple(dialogue_turns)
        self._extraction_prompt = extraction_prompt

    def build(self, memory_items: Sequence[str]) -> None:
        from datetime import datetime

        if self._memory_timestamps and len(self._memory_timestamps) != len(memory_items):
            raise ValueError("LightMem timestamps must align one-to-one with memory items")
        if self._dialogue_turns and len(self._dialogue_turns) != len(memory_items):
            raise ValueError("LightMem dialogue turns must align one-to-one with memory items")
        # LoCoMo supplies released session timestamps. Document-only datasets
        # do not; LightMem's required metadata then records their common
        # ingestion session, while the benchmark text remains unchanged.
        ingestion_time = datetime.now().astimezone().isoformat(timespec="seconds")
        timestamps = self._memory_timestamps or (ingestion_time,) * len(memory_items)
        for index, item in enumerate(memory_items):
            is_last = index == len(memory_items) - 1
            turn = self._dialogue_turns[index] if self._dialogue_turns else None
            content = item
            message_metadata: dict[str, str] = {}
            if turn is not None:
                content = turn.text
                if turn.blip_caption:
                    content += f" (image description: {turn.blip_caption})"
                message_metadata = {
                    "speaker_id": turn.speaker_id,
                    "speaker_name": turn.speaker_name,
                }
            # LightMem's released LoCoMo ingestion represents every spoken
            # turn as a user message followed by an empty assistant message.
            # Its source-id converter consequently maps displayed ids back to
            # even sequence numbers.  Keep that representation for every
            # input; messages_use=user_only ensures the empty message is not
            # extracted as document content.
            self._memory.add_memory(
                [
                    {"role": "user", "content": content, "time_stamp": timestamps[index], **message_metadata},
                    {"role": "assistant", "content": "", "time_stamp": timestamps[index], **message_metadata},
                ],
                METADATA_GENERATE_PROMPT=self._extraction_prompt,
                force_segment=is_last,
                force_extract=is_last,
            )
        if memory_items and not self._memory.embedding_retriever.get_all(with_vectors=False):
            raise RuntimeError("LightMem extraction produced no memory entries")

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem(text) for text in self._memory.retrieve(query, limit=top_k)]

    def efficiency_metrics(self) -> dict[str, Any]:
        return {
            "generation_client": self._generation_usage.snapshot(),
            "lightmem_token_statistics": self._memory.get_token_statistics(),
        }

    def close(self) -> None:
        close = getattr(self._memory, "close", None)
        if callable(close):
            close()


class HippoRAG2Baseline:
    def __init__(
        self,
        official_config: dict[str, Any],
        save_dir: str | Path,
    ) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "HippoRAG" / "src")
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig
        config = BaseConfig(**official_config)
        config.save_dir = str(save_dir)
        from hipporag.llm.openai_gpt import CacheOpenAI
        self._generator = CacheOpenAI.from_experiment_config(config)
        # The client has already captured its local vLLM URL. Do not persist an
        # ephemeral Slurm port as part of the reusable memory identity.
        config.llm_base_url = None
        self._generation_usage = GenerationUsageTracker()
        self._generation_usage.install(self._generator.openai_client)
        self._memory = HippoRAG(
            global_config=config,
            extraction_llm=self._generator,
            qa_llm=self._generator,
            index_identity=f"{config.llm_name}:non-thinking",
        )

    def build(self, memory_items: Sequence[str]) -> None:
        self._memory.index(list(memory_items))

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        solution = self._memory.retrieve([query], num_to_retrieve=top_k)[0]
        scores = solution.doc_scores.tolist() if solution.doc_scores is not None else [None] * len(solution.docs)
        return [RetrievedItem(text, score) for text, score in zip(solution.docs, scores)]

    def efficiency_metrics(self) -> dict[str, Any]:
        return {"generation_client": self._generation_usage.snapshot()}

    def close(self) -> None:
        try:
            self._memory.close()
        finally:
            self._generator.close()


class Mem0Baseline:
    def __init__(
        self,
        official_config: dict[str, Any],
        namespace: str,
        *,
        dialogue_roles: Sequence[str] = (),
    ) -> None:
        _prepend(_OFFICIAL_ALGORITHMS / "mem0")
        from mem0 import Memory
        self._memory = Memory.from_config(official_config)
        self._generation_usage = GenerationUsageTracker()
        self._generation_usage.install(self._memory.llm.client)
        self._filters = {"user_id": namespace}
        self._dialogue_roles = tuple(dialogue_roles)

    def build(self, memory_items: Sequence[str]) -> None:
        if self._dialogue_roles and len(self._dialogue_roles) != len(memory_items):
            raise ValueError("Mem0 dialogue roles must align one-to-one with memory items")
        roles = self._dialogue_roles or ("user",) * len(memory_items)
        for item, role in zip(memory_items, roles, strict=True):
            self._memory.add([{"role": role, "content": item}], user_id=self._filters["user_id"], infer=True)
        if memory_items and not self._memory.get_all(filters=self._filters, top_k=1)["results"]:
            raise RuntimeError("Mem0 extraction produced no memory entries")

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        response = self._memory.search(query, top_k=top_k, filters=self._filters)
        return [RetrievedItem(str(item["memory"]), float(item["score"]), item) for item in response["results"]]

    def efficiency_metrics(self) -> dict[str, Any]:
        return {"generation_client": self._generation_usage.snapshot()}

    def close(self) -> None:
        return None
