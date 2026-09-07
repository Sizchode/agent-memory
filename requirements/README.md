# UV environment layout

This repository deliberately uses separate environments. The pinned official
implementations impose incompatible requirements: for example, LightMem pins
Torch 2.8/OpenAI 2.3, while HippoRAG pins Torch 2.5/OpenAI 3.x. A single solved
environment would change at least one official method.

Nothing in this directory installs packages or creates environments. On HPC,
create exactly four baseline environments:

```bash
uv venv .venv/lightmem --python 3.11
uv pip install --python .venv/lightmem/bin/python -r requirements/lightmem/requirements.txt

uv venv .venv/magma --python 3.11
uv pip install --python .venv/magma/bin/python -r requirements/magma/requirements.txt

uv venv .venv/hipporag --python 3.11
uv pip install --python .venv/hipporag/bin/python -r requirements/hipporag/requirements.txt

uv venv .venv/mem0 --python 3.11
uv pip install --python .venv/mem0/bin/python -r requirements/mem0/requirements.txt
```

The common local LLM and embedding endpoint is an external service consumed via
its OpenAI-compatible URL. It is deliberately not represented as a fifth
baseline environment.

Use `uv pip compile` on HPC to materialize platform-specific lock files only
after selecting the CUDA/PyTorch wheel index and GPU architecture. Commit the
resulting lock files and record `uv pip freeze` in each experiment artifact.
