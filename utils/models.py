"""OpenAI-compatible model clients shared by every controlled experiment."""

from collections.abc import Mapping, Sequence
import gc
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEndpoint:
    """Generator model coordinates and the environment holding its credential."""

    model: str
    base_url: str | None
    api_key_env: str

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"environment variable {self.api_key_env!r} is required for model {self.model!r}")
        return value


class HuggingFaceChatModel:
    """Single-process local Hugging Face inference for the evaluation backbone."""

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        seed: int = 42,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.device_map = device_map
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def answer(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> str:
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = {name: value.to(self._model.get_input_embeddings().weight.device) for name, value in inputs.items()}
        generation = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens or self.max_tokens,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        if top_k is not None:
            generation["top_k"] = top_k
        if top_p is not None:
            generation["top_p"] = top_p
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError(f"local evaluation model {self.model_id!r} returned an empty completion")
        return text

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install torch, transformers, and accelerate for local Hugging Face evaluation") from exc
        dtype = "auto" if self.dtype == "auto" else getattr(torch, self.dtype, None)
        if dtype is None:
            raise ValueError("evaluation dtype must be auto, float16, bfloat16, or float32")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, device_map=self.device_map)
        self._model.eval()


class HuggingFaceEmbedder:
    """Local Hugging Face embedding callable shared by retrieval baselines."""

    def __init__(self, model_id: str, *, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("install sentence-transformers for local Hugging Face embeddings") from exc
            self._model = SentenceTransformer(self.model_id, device=self.device, trust_remote_code=True)
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def release_accelerator_memory() -> None:
    """Release model objects collected between retrieval and answer phases."""

    gc.collect()
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
