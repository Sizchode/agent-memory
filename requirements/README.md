# UV environment layout

This repository deliberately uses separate environments. The pinned official
implementations impose incompatible requirements: for example, LightMem pins
Torch 2.8/OpenAI 2.3, while HippoRAG pins Torch 2.5/OpenAI 3.x. A single solved
environment would change at least one official method.

Nothing in this directory installs packages or creates environments. On HPC,
keep the repository in home and create these four environments in scratch:

```bash
ENV_ROOT=/oscar/scratch/zliu328/agent-memory-envs

uv venv "$ENV_ROOT/runner" --python 3.11
uv pip install --python "$ENV_ROOT/runner/bin/python" -r requirements/common.txt

uv venv "$ENV_ROOT/lightmem" --python 3.11
uv pip install --python "$ENV_ROOT/lightmem/bin/python" -r requirements/lightmem/requirements.txt

uv venv "$ENV_ROOT/hipporag" --python 3.11
uv pip install --python "$ENV_ROOT/hipporag/bin/python" -r requirements/hipporag/requirements.txt

uv venv "$ENV_ROOT/mem0" --python 3.11
uv pip install --python "$ENV_ROOT/mem0/bin/python" -r requirements/mem0/requirements.txt
```

The `runner` environment runs BM25/Dense and hosts local Hugging Face evaluator
inference. The official baseline environments also include the common runner
dependencies. The evaluator never uses an API endpoint; only DeepSeek
generation and embeddings may use OpenAI-compatible services.

Use `uv pip compile` on HPC to materialize platform-specific lock files only
after selecting the CUDA/PyTorch wheel index and GPU architecture. Commit the
resulting lock files and record `uv pip freeze` in each experiment artifact.
