# Evaluation protocol and HPC readiness

## Deterministic evaluation metrics

The experiment uses no LLM judge. Metrics are selected from each benchmark's
released deterministic evaluator, not from a baseline-specific judge.

| Benchmark family | Metric |
| --- | --- |
| MemoryAgentBench main variants | official Substring Exact Match |
| LoCoMo | official category-specific stemmed token F1 |
| 2WikiMultiHopQA | gold-passage Recall@k, Precision@k, and normalized answer F1 |

MemoryAgentBench uses the exact main configurations from its released
`rag_agents.txt`, rather than pooling length ablations. LoCoMo applies its
released category rules: multi-answer matching for category 1, the reference
before the semicolon for category 3, direct F1 for categories 2 and 4, and the
released unanswerable rule for category 5. The 2Wiki supporting passages are
read from HippoRAG's released files. LoCoMo evaluates all
1,986 questions in the released ten-conversation file; 2WikiMultiHopQA uses its
released 1,000-query file.

Primary references:

- [MemoryAgentBench evaluation mapping](https://github.com/HUST-AI-HYZ/MemoryAgentBench#-clarification-on-evaluation-metrics)
- [LoCoMo released evaluator](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)
- [HippoRAG released evaluator](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/evaluation/qa_eval.py)

This differs from some baseline-paper headline results. HippoRAG 2 reports
passage Recall@5 and token F1, matching this runner. LightMem's paper uses
GPT-4o-mini judge accuracy as its main effectiveness metric. The Mem0 paper
reports deterministic F1 and BLEU-1 alongside an LLM judge. We do not reproduce
or report either paper's judge score, and we do not mix judge outputs with the
benchmark metrics. For LoCoMo, every method is instead scored by the same
released category-specific deterministic evaluator.

## Model-role terminology

**Generator Backbone** reads original articles or conversations and performs
method-native memory generation, extraction, compression, consolidation, and
updates. BM25 and dense retrieval do not use a Generator Backbone.

**Evaluation Backbone** consumes a method's retrieved evidence or memory and
performs downstream QA to produce the final answer. It is not a judge and never
receives the ground-truth answer.

The runner evaluates each frozen retrieval artifact with three Evaluation
Backbones: `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`, and `Qwen/Qwen3.5-2B`.
Chat templates receive `enable_thinking=False`; evaluation never rebuilds
method memories. The optional Llama checkpoint is gated on the
Hugging Face Hub and therefore requires prior account approval and an
`HF_TOKEN` on the compute node; the launcher includes it only when
`INCLUDE_GATED_LLAMA=1` is set.

Evaluation generation follows the released task settings. MemoryAgentBench
uses 50 tokens for SH-Doc/MH-Doc, 40 for EventQA, and 10 for both
FactConsolidation tasks, with temperature 0.7 from its released RAG configs.
LoCoMo's released Hugging Face path uses 50 tokens, temperature 0.4, top-k 10,
and top-p 0.9. HippoRAG 2 uses its released four-turn QA template, greedy
decoding, and a 2,048-token ceiling; only text after `Answer:` is scored.

Shared experimental controls are:

- Generator Backbone: local `Qwen/Qwen3-30B-A3B-Instruct-2507`, served in
  BF16; this checkpoint is instruction-tuned and non-thinking only;
- embedding: `Qwen/Qwen3-Embedding-0.6B`, 1,024 dimensions;
- final retrieval budget: top 5 for every retained benchmark, providing one
  controlled evidence budget across methods;
- MemoryAgentBench task chunking: 512 tokens for SH/MH document QA and
  FactConsolidation, and 4,096 tokens for EventQA;
- answer generation: one non-thinking completion per question and Evaluation
  Backbone.

Algorithm-private prompts and operations are not replaced or added to other
methods.

Generator output limits retain method-specific official settings rather than a
single cross-method cap: LightMem uses 16,000 tokens, Mem0 uses 2,000 tokens,
and HippoRAG 2 uses 512 tokens for NER and 2,048 tokens for triple extraction.
These limits apply to generated memory artifacts, not to the input context
window and not to downstream QA answers.

The local vLLM server uses a 32,768-token context window. This is serving
capacity; it does not change any method's completion budget.

For document-only tasks, the Mem0 custom instruction asks for every supported
fact exactly once in the shortest self-contained wording that retains names,
numbers, dates, relations, and qualifications. This is the disclosed document
input adapter: it leaves Mem0's released extraction and memory-management
pipeline intact while preventing repeated elaboration from exhausting the
official 2,000-token response budget.

## HPC status

The five isolated environments, official dataset conversion, deterministic
metrics, non-thinking controls, retrieval/evaluation process split, and Slurm
array launcher are implemented. The conversions and metrics were checked
directly against the official source and released files.

Before submission, set:

```bash
export LOCOMO_PATH='<official locomo10.json>'
bash experiments/run_experiments.sh
```

After a generated-memory task has completed, retrieval can be changed and run
again without rebuilding memory:

```bash
python main.py \
  --phase retrieve-existing \
  --task 'SH-Doc QA' \
  --baseline mem0 \
  --memory-input-dir /path/to/completed/mem0/SH-Doc_QA \
  --output-dir /path/to/new/retrieval-run/mem0/SH-Doc_QA \
  --official-config experiments/configs/mem0.json \
  --generator-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --generator-base-url http://127.0.0.1:1/v1 \
  --generator-api-key-env VLLM_API_KEY \
  --top-k 5 \
  --seed 42
```

For Mem0 and LightMem, `VLLM_API_KEY` may be any nonempty local placeholder in
this phase: opening and searching their completed stores does not call the
Generator Backbone. HippoRAG 2 also skips indexing, but its released online
retrieval procedure uses the configured LLM to filter candidate facts before
graph search. Therefore a compatible Generator Backbone endpoint must still be
available when rerunning HippoRAG 2 retrieval. The memory input and retrieval
output directories must differ. BM25 and dense retrieval have no generated
memory store, so this phase intentionally rejects them.

To rerun retrieval for every generated-memory baseline and task through Slurm,
reuse the directory name of the completed generation experiment:

```bash
MEMORY_INPUT_EXPERIMENT_ID=qwen3_30b_a3b_instruct_2507_seed42_20260908T \
EXPERIMENT_ID=retrieval_variant_seed42 \
bash experiments/run_experiments.sh --retrieve-existing
```

This submits only LightMem, HippoRAG 2, and Mem0 retrieval workers. It does not
rebuild their stored memory. LightMem and Mem0 do not launch a Generator
Backbone in this mode; HippoRAG 2 launches one for its released online fact
filtering step.

## Per-question failure analysis

After all Evaluation Backbones finish, create the deterministic error report
directly from `retrieval.jsonl` and `predictions.jsonl`:

```bash
python experiments/analyze_failures.py \
  --results-root /path/to/experiment \
  --output /path/to/experiment/failure_analysis.json \
  --expected-evaluators 3
```

For each evaluator, the report retains the questions with an official answer
score of zero and separates zero, partial, and full scores. It also lists the
intersection that scored zero for every evaluator. When the released data
provides question categories, it reports per-category scores; LoCoMo category
IDs are labelled with the released multi-hop, temporal, open-domain,
single-hop, and adversarial types. Only 2WikiMultiHopQA is divided by none,
partial, or complete gold passage recall because only that dataset provides the required gold passage
annotations. For that dataset it also reports downstream answer scores
inside the none, partial, and complete-recall groups, separating missing
evidence from reader errors without introducing a new benchmark metric. The
report does not infer retrieval causes for the other tasks.

No API key is required for the local Generator Backbone. Query-level resume,
a separate provenance subsystem, generated lock files, data checksums, and
artifact hashes are intentionally outside this research runner.
