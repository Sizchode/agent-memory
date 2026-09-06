# agent-memory

MemoryAgentBench experiment layout for dataset loading and baseline evaluation.

```text
dataset_loader/   One loader.py for MemoryAgentBench, LoCoMo, and HippoRAG 2 data
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

The combined evaluation suite also supports two official-protocol groups:

| Benchmark group | Loader input | Metrics |
| --- | --- | --- |
| LoCoMo | `data/locomo10.json` from the official repository | token F1; BLEU-1 |
| MuSiQue / 2WikiMultiHopQA / HotpotQA | HippoRAG 2 `reproduce/dataset` JSON files | gold-passage Recall@5; answer F1 |

The HippoRAG 2 loader defaults to the released 1,000-query samples and keeps
the corpus, gold passages, answer aliases, and query IDs intact. It does not
create data or labels.

Download the official files from
[HippoRAG2Official/reproduce/dataset](https://github.com/WeijianQ/HippoRAG2Official/tree/main/reproduce/dataset)
and pass that directory as `--data-root`. The scoring helpers in
`utils/hipporag_metrics.py` implement the official set-based gold-passage
Recall@5 and normalized answer F1.

```bash
./experiments/run_baseline.sh 'LoCoMo' --path /path/to/locomo10.json
./experiments/run_baseline.sh 'MuSiQue' --data-root /path/to/reproduce/dataset
```

The six baseline implementations and controlled-variable policy are documented
in [`docs/baseline_integration.md`](docs/baseline_integration.md). Clone this
repository with `--recurse-submodules` to fetch the pinned official sources.
