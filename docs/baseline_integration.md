# Baseline integration and controlled variables

The repository pins official source trees in `baseline_algorithms/` as Git submodules. Adapters call
their public implementation entry points; algorithm-specific prompts,
summarization, extraction, graph construction, and update rules remain in the
official source.

| Method | Pinned source | Algorithm preserved | Experiment controls applied in `main.py` |
| --- | --- | --- | --- |
| BM25 | local | Okapi BM25 | chunks, top-k |
| Dense retrieval | local | cosine retrieval | embedding function, chunks, top-k |
| LightMem | `zjunlp/LightMem@8449d574` | segmentation, extraction, summaries, offline update, retrieval | LLM, embedding, dimensions, generation budget |
| HippoRAG 2 | `OSU-NLP-Group/HippoRAG@1438aba3` | OpenIE, graph, fact linking, PPR and official prompts | LLM, embedding, retrieval/QA top-k, output budget |
| Mem0 | `mem0ai/mem0@dae67f74` | fact extraction and ADD/UPDATE/DELETE policy with official prompts | LLM, embedding, generation budget |

Primary sources:

- LightMem configuration and implementation: <https://github.com/zjunlp/LightMem>
- HippoRAG 2 configuration and prompts: <https://github.com/OSU-NLP-Group/HippoRAG>
- Mem0 OSS implementation and configuration: <https://github.com/mem0ai/mem0>

## Policy

Internal prompts that define an algorithm are not replaced. This includes
LightMem summary/update prompts, HippoRAG OpenIE/linking prompts, and Mem0 fact/update prompts. The final QA
prompt and answer model are benchmark-runner concerns and must be shared.

`main.py` owns `ExperimentConfig`, the only location for common model choices,
input chunking, final evidence budget, output budget, and answer protocol.
Baseline adapters retain only algorithm-private behavior.

Pass `ExperimentConfig.hipporag_config`, `.lightmem_config`, or `.mem0_config`
to the corresponding adapter.

## Decisions required before HPC runs

1. One answer/internal LLM model and endpoint. The primary controlled table
   uses it for every baseline operation that invokes an LLM.
2. One embedding model, endpoint, and exact vector dimension.
3. One input chunk size and whether each algorithm receives benchmark chunks
   or its official internal segmentation input. These are not equivalent.
4. Retrieval budget: the same final top-k is required; decide whether large
   internal candidate pools (for example HippoRAG's graph candidate pool) stay
   algorithm-native.

Vector-store backends, OpenIE, summary/update policy, and graph construction
are method-private. Keep each official implementation's choice and report it.

## Checkout

```bash
git clone --recurse-submodules https://github.com/Sizchode/agent-memory.git
git submodule update --init --recursive
```
