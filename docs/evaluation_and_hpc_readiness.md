# Evaluation-model survey and HPC readiness

## Decision for the primary table

The primary paper table should use **no LLM-as-a-judge**.  The selected
benchmarks already provide deterministic metrics:

| Benchmark family | Primary metric | LLM judge |
| --- | --- | --- |
| MemoryAgentBench: SH-Doc QA, MH-Doc QA, EventQA, FactConsolidation-SH/MH | official Substring Exact Match | no |
| LoCoMo | official token F1 and BLEU-1 | no |
| MuSiQue, 2WikiMultiHopQA, HotpotQA | gold-passage Recall@5 and answer F1 | no |

MemoryAgentBench explicitly assigns `substring_exact_match` to the Accurate
Retrieval and Conflict Resolution datasets. [Official repository metric
mapping](https://github.com/HUST-AI-HYZ/MemoryAgentBench#-clarification-on-evaluation-metrics)
documents this. HippoRAG's own evaluator implements normalized EM/F1 rather
than an LLM judge: [official `qa_eval.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/evaluation/qa_eval.py).

LoCoMo has an optional semantic judge in some downstream method repositories,
but it is not needed for our pre-registered main metric. It may be reported
only as a clearly labelled appendix diagnostic, evaluated once per prediction
with a held-fixed judge and never used to choose an answer.

## What the official baseline code actually uses

| Method | LLM role during memory / answering | Evaluation-time LLM role | Default or reported model |
| --- | --- | --- | --- |
| LightMem | memory extraction/consolidation and QA answerer | optional binary semantic judge: question + gold + prediction -> `CORRECT`/`WRONG`, JSON, temperature 0 | reported answerers: `gpt-4o-mini` and `qwen3-30b-a3b-instruct-2507`; reported judges: `gpt-4o-mini` and `qwen2.5-32b-instruct` |
| MAGMA | event/relation construction and answerer | optional continuous semantic scorer (0--1), `gpt-4o-mini` hard-coded in the judge module | answerer CLI defaults to `gpt-4o-mini`; README also lists `gpt-4.1-mini` and `gpt-4o` |
| HippoRAG 2 | online OpenIE (NER + triples) and QA generation | none; retrieval recall and QA EM/F1 are deterministic | `gpt-4o-mini` default LLM; `nvidia/NV-Embed-v2` default embedding |
| Mem0 | fact extraction and ADD/UPDATE/DELETE memory policy; benchmark answerer is external to memory store | the current public memory-benchmarks suite requires an answerer/judge LLM | OSS suite defaults to `gpt-4o-mini` for fact extraction and `text-embedding-3-small` for embeddings |

Evidence:

- LightMem's released commands and result tables name its answer and judge
  models: [official LoCoMo README](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/readme.md).
  Its judge sees no retrieved context; it grades only question, reference, and
  generated answer, then outputs a binary JSON label:
  [official judge implementation](https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/llm_judge.py).
- MAGMA exposes `gpt-4o-mini` as its evaluation CLI default and defaults to
  three generations with `llm_judge` selection:
  [official evaluation entry point](https://github.com/FredJiang0324/MAGMA/blob/467cb70b67ac337b22fdb42194d37c04ad701b62/test_fixed_memory.py).
  Its judge is an answer semantic scorer, not part of retrieval:
  [official judge implementation](https://github.com/FredJiang0324/MAGMA/blob/467cb70b67ac337b22fdb42194d37c04ad701b62/memory/llm_judge.py).
- HippoRAG's configuration defaults to `gpt-4o-mini`, `NV-Embed-v2`,
  online OpenIE, and a 2,048-token LLM output cap:
  [official config](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/utils/config_utils.py).
- Mem0's maintained benchmark suite documents the OSS defaults:
  [official memory-benchmarks README](https://github.com/mem0ai/memory-benchmarks/blob/main/README.md).

## Fair comparison policy

Use one `generation_model` for every LLM call that a method makes: memory
extraction, summary/consolidation, OpenIE, relation construction, decision or
reasoning, and final answer generation. Hold temperature, output caps, endpoint
implementation, input chunk stream, and final evidence budget fixed.

Do not replace method-private operations. OpenIE remains in HippoRAG;
LightMem/Mem0 summary and update policy remain; MAGMA retains its native graph
and LoCoMo session/timestamp representation. Those modules receive the common
generation model, but are not added to BM25 or Dense retrieval.

For a local primary setting, choose one generator after a small structured
output pilot; do not silently mix cloud answering with local extraction. The
candidate set is intentionally not Qwen-only:

| Candidate | Scale / deployment reason | Caution |
| --- | --- | --- |
| `Qwen2.5-14B-Instruct` | lowest-cost serious pilot; fits the requested ten-billion scale | use only if the pilot shows reliable multi-hop and JSON extraction |
| `Qwen2.5-32B-Instruct` | conservative 32B instruction-following primary candidate | higher GPU memory and latency |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 24B, Apache-2.0, 128K context, and official vLLM/function-calling guidance | needs the same benchmark-specific JSON pilot |
| `google/gemma-3-27b-it` | 27B with 128K context; a credible non-Qwen comparison | Gemma 3 is the current official Gemma family; no official Gemma 4 release was found |
| `gpt-oss-20b` | 21B total / 3.6B active, Apache-2.0, local reasoning and structured-output support | reasoning must be disabled or held fixed to avoid variable hidden test-time compute |
| `meta-llama/Llama-3.3-70B-Instruct` | strong 128K reference model | 70B is outside the intended low-cost tier and has a custom licence |

For the first HPC pilot, compare `Qwen2.5-14B-Instruct`,
`Mistral-Small-3.1-24B-Instruct-2503`, `Gemma-3-27B-IT`, and `gpt-oss-20b` on
the same fixed 100-question development slice. Measure JSON validity, OpenIE
parse validity, deterministic-metric score, GPU memory, latency, and token
throughput. Select one winner before the full run, then freeze it for every
method and benchmark. `Qwen3-30B-A3B-Instruct-2507` remains a candidate only
after it passes the same pilot; its vLLM structured-output path has a reported
non-termination issue.

Primary sources for this shortlist: [Mistral Small 3.1 model
card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503),
[Gemma 3 model card](https://huggingface.co/google/gemma-3-27b-pt),
[gpt-oss-20b documentation](https://developers.openai.com/api/docs/models/gpt-oss-20b),
[Llama 3.3 model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
and [Qwen3 vLLM deployment notes](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md).

## Required before the first HPC experiment

The repository is **not yet ready** for an end-to-end HPC run. The following
are blocking implementation tasks, in order.

1. **One real experiment runner.** `main.py` currently loads data and runs a
   BM25 inspection only. Implement selection and execution of all six methods,
   final answer generation, the benchmark-specific metric call, and one common
   prediction/result schema.
2. **Complete the adapters.** Smoke-test each official algorithm with one
   context and one query in its own environment. MAGMA needs a native structured
   LoCoMo path; it must not receive synthetic timestamps for document datasets.
   The other document/conversation ingestion paths need explicit benchmark
   adapters rather than treating all data as interchangeable strings.
3. **Disable evaluation leakage.** Never use MAGMA's default `best-of-3` and
   `llm_judge` answer selection. Generate exactly one temperature-0 answer per
   question, then score it with the predetermined deterministic metric.
4. **Lock environments per method.** The root `requirements.txt` contains only
   loader dependencies. Add reproducible lock files or containers for LightMem,
   MAGMA, HippoRAG, and Mem0. Do not force their incompatible official
   dependency stacks into one environment; communicate through a JSONL
   prediction contract.
5. **Data manifest and fetch step.** Automate/record the exact HF revision for
   MemoryAgentBench, the LoCoMo file revision, and the HippoRAG2 released
   1,000-query files. Store paths, checksums, and dataset revision in each run
   artifact; do not commit benchmark data.
6. **Experiment config and provenance.** Add a checked-in config containing
   model IDs, embedding dimension, chunk policy, top-k, temperature, token
   caps, seed, and endpoint name. Each run must save that resolved config,
   source commit, submodule commits, package versions, predictions, retrieved
   evidence IDs, token/API-call counts, elapsed time, and failures.
7. **HPC launch and recovery.** Add Slurm launchers with a per-run output
   directory, environment activation, secret handling through environment
   variables, rate/concurrency limits for API methods, retry policy, and
   resume-by-completed-query semantics.
8. **Pilot acceptance gate.** Before the full suite, run one context or one
   conversation per benchmark-method pair; verify non-empty retrieval, one
   prediction per question, deterministic scoring, resume behavior, and that
   every output can be aggregated without manual edits.

Only after items 1--8 are complete is cloning the repository on HPC expected
to lead directly to experiment runs rather than integration debugging.
