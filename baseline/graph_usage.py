"""Per-call JSONL costs for new graph adapters, without changing requests."""

from functools import wraps
import inspect
import json
from threading import Lock
from time import perf_counter
from typing import Mapping


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


class GraphUsageRecorder:
    """Wrap actual provider calls below caches; caller owns the output stream.

    Create a wrapper per stage/group rather than mutating shared context during
    concurrent calls. Cache hits bypass this recorder and need separate records.
    Missing usage stays null, never zero or an estimated token count.
    """

    def __init__(self, stream):
        self.stream = stream
        self._lock = Lock()

    def record(self, record):
        with self._lock:
            self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.stream.flush()

    def wrap(self, call, *, method, stage, group_id, case_id=None):
        identity = dict(method=method, stage=stage, group_id=group_id, case_id=case_id)

        def finish(start, kwargs, response=None, error=None):
            usage = _field(response, "usage") if response is not None else None
            # Do not record prompts, responses, credentials or exception messages.
            record = dict(identity, event="provider_call", elapsed_seconds=perf_counter() - start,
                          status="failed" if error else "completed",
                          error_type=type(error).__name__ if error else None,
                          model=kwargs.get("model"), usage_available=usage is not None,
                          prompt_tokens=_field(usage, "prompt_tokens"),
                          completion_tokens=_field(usage, "completion_tokens"),
                          total_tokens=_field(usage, "total_tokens"))
            record["request_settings"] = {key: kwargs[key] for key in
                ("max_tokens", "max_completion_tokens", "temperature", "seed", "stream")
                if key in kwargs}
            self.record(record)

        if inspect.iscoroutinefunction(call):
            @wraps(call)
            async def measured_async(*args, **kwargs):
                start = perf_counter()
                try:
                    response = await call(*args, **kwargs)
                except BaseException as error:
                    finish(start, kwargs, error=error)
                    raise
                finish(start, kwargs, response=response)
                return response
            return measured_async

        @wraps(call)
        def measured_sync(*args, **kwargs):
            start = perf_counter()
            try:
                response = call(*args, **kwargs)
            except BaseException as error:
                finish(start, kwargs, error=error)
                raise
            # SDK decorators may hide coroutine functions from inspect. Await
            # their result before reading usage, without changing the request.
            if inspect.isawaitable(response):
                async def complete():
                    try:
                        result = await response
                    except BaseException as error:
                        finish(start, kwargs, error=error)
                        raise
                    finish(start, kwargs, response=result)
                    return result
                return complete()
            finish(start, kwargs, response=response)
            return response
        return measured_sync
