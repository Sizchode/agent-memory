"""OpenAI-compatible model clients shared by every controlled experiment."""

from collections.abc import Sequence
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEndpoint:
    """A model served through the OpenAI chat/embeddings API schema."""

    model: str
    base_url: str
    api_key_env: str

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"environment variable {self.api_key_env!r} is required for model {self.model!r}")
        return value


class HuggingFaceChatModel:
    """Single-process local Hugging Face inference for the evaluation backbone."""

    def __init__(self, model_id: str, *, max_tokens: int, dtype: str = "bfloat16", device_map: str = "auto") -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self._model.get_input_embeddings().weight.device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_tokens)
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
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, device_map=self.device_map)
        self._model.eval()


class OpenAIEmbedder:
    """Batch embedding callable compatible with the dense-retrieval baseline."""

    def __init__(self, endpoint: ModelEndpoint, *, dimensions: int | None = None) -> None:
        self.endpoint = endpoint
        self.dimensions = dimensions

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client()
        request = {"model": self.endpoint.model, "input": list(texts)}
        if self.dimensions is not None:
            request["dimensions"] = self.dimensions
        response = client.embeddings.create(**request)
        vectors = [list(item.embedding) for item in response.data]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding endpoint returned a different number of vectors")
        if self.dimensions is not None and any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(f"embedding endpoint did not return the requested {self.dimensions}-dimensional vectors")
        return vectors

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai to use an OpenAI-compatible embedding endpoint") from exc
        return OpenAI(api_key=self.endpoint.api_key(), base_url=self.endpoint.base_url)
