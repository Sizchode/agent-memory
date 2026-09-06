# agent-memory

MemoryAgentBench experiment layout for dataset loading and baseline evaluation.

```text
dataset_loader/   MemoryAgentBench HF loading, task mapping, and chunking
baseline/         Baseline algorithms
experiments/      Experiment launcher scripts
utils/            Metrics and shared helpers
main.py           CLI entry point
```

Run the baseline with:

```bash
./experiments/run_baseline.sh 'EventQA' --max-contexts 1
```

The loader uses the repository's four HF splits and identifies the requested
five tasks from `metadata["source"]`: `ruler_qa1`, `ruler_qa2`, `eventqa_*`,
`factconsolidation_sh_*`, and `factconsolidation_mh_*`. Each returned sample
contains one context, its sentence-preserving token chunks, and all of its QA
pairs. A runner should build memory from `sample.chunks` once, then iterate
over `sample.question_answers`.

The official benchmark uses a 4096-token default chunk size and
the prompt templates in `utils/prompts.py`. Its accuracy shorthand maps to
substring-exact-match for Accurate Retrieval and Conflict Resolution.
