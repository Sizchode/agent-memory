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


class OpenAIChatModel:
    """Deterministic chat-completion adapter used only for generation and QA."""

    def __init__(self, endpoint: ModelEndpoint, *, max_tokens: int, temperature: float = 0.0) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        client = self._client()
        response = client.chat.completions.create(
            model=self.endpoint.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"model {self.endpoint.model!r} returned an empty completion")
        return content.strip()

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai to use an OpenAI-compatible model endpoint") from exc
        return OpenAI(api_key=self.endpoint.api_key(), base_url=self.endpoint.base_url)


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
