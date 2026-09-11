"""Record token usage returned by OpenAI-compatible generation clients."""

from functools import wraps
from threading import Lock
from typing import Any


class GenerationUsageTracker:
    """Count completed client calls without changing their inputs or outputs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls_started = 0
        self._calls_succeeded = 0
        self._calls_failed = 0
        self._responses_without_usage = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    def install(self, client: Any) -> None:
        completions = client.chat.completions
        original_create = completions.create

        @wraps(original_create)
        def measured_create(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self._calls_started += 1
            try:
                response = original_create(*args, **kwargs)
            except BaseException:
                with self._lock:
                    self._calls_failed += 1
                raise

            usage = getattr(response, "usage", None)
            with self._lock:
                self._calls_succeeded += 1
                if usage is None:
                    self._responses_without_usage += 1
                else:
                    self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                    self._total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            return response

        completions.create = measured_create

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "client_calls_started": self._calls_started,
                "client_calls_succeeded": self._calls_succeeded,
                "client_calls_failed": self._calls_failed,
                "responses_without_usage": self._responses_without_usage,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
            }
