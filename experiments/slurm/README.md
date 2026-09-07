# Slurm array launcher

`run_benchmark.sbatch` launches one baseline at a time. Its 36 array workers
are the Cartesian product of the nine benchmark tasks and four local Hugging
Face evaluation backbones. A worker loads exactly one evaluator, processes one
task, and writes to a unique output directory.

| Array indices | Tasks |
| --- | --- |
| `0-3` | SH-Doc QA × four evaluators |
| `4-7` | MH-Doc QA × four evaluators |
| `8-11` | EventQA × four evaluators |
| `12-15` | FactConsolidation-SH × four evaluators |
| `16-19` | FactConsolidation-MH × four evaluators |
| `20-23` | LoCoMo × four evaluators |
| `24-27` | MuSiQue × four evaluators |
| `28-31` | 2WikiMultiHopQA × four evaluators |
| `32-35` | HotpotQA × four evaluators |

For a first BM25 smoke run, submit only one task/model pair:

```bash
cd /oscar/home/zliu328/agent-memory
export HUGGINGFACE_HUB_TOKEN='set this in your shell; never commit it'
export HF_HOME=/oscar/data/sbach/zliu328/hf_output
sbatch --array=1 --export=ALL,BASELINE=bm25,MAX_CONTEXTS=1 \
  experiments/slurm/run_benchmark.sbatch
```

Index `1` selects `Qwen/Qwen3.5-27B`, a safer first single-H100 smoke model
than loading the larger 35B-A3B checkpoint at index `0`.

To run all four evaluation backbones on the five MemoryAgentBench tasks, use
`--array=0-19`. LoCoMo and the three multi-hop tasks additionally require the
respective `LOCOMO_PATH` and `HIPPORAG_DATA_ROOT` environment variables.

Submit a different baseline as a separate Slurm array, because it selects a
different UV environment. LightMem, HippoRAG 2, and Mem0 also require
`OFFICIAL_CONFIG`; Dense and the official memory baselines require
`EMBEDDING_BASE_URL`.
