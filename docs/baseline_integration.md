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

Internal prompts that define an algorithm are retained. The only cross-domain
adaptation is the disclosed document-fact extraction instruction for LightMem
and Mem0 on document benchmarks; LoCoMo uses their native dialogue behavior.
The final QA prompt and answer model are benchmark-runner concerns and are
shared.

`main.py` owns `ExperimentConfig`, the only location for common model choices,
input chunking, final evidence budget, output budget, and answer protocol.
Baseline adapters retain only algorithm-private behavior.

Pass `ExperimentConfig.hipporag_config`, `.lightmem_config`, or `.mem0_config`
to the corresponding adapter.

## Frozen experiment decisions

1. Generator Backbone: local
   `Qwen/Qwen3-30B-A3B-Instruct-2507`, BF16, non-thinking.
2. Embedding: `Qwen/Qwen3-Embedding-0.6B`, 1,024 dimensions.
3. Benchmark input: official MemoryAgentBench main variants and chunk sizes;
   LightMem then applies its official internal pre-compression and topic
   segmentation.
4. Final retrieval budget: top 5 for every method. Internal candidate pools
   remain algorithm-native.

Vector-store backends, OpenIE, summary/update policy, and graph construction
are method-private. Keep each official implementation's choice and report it.

## Adapter audit for the final run

| Scope | Retained adaptation | Reason |
| --- | --- | --- |
| All methods | shared Qwen Generator Backbone where required, Qwen embedding, top 5, seed 42 | controlled comparison |
| Benchmark | exact released task variants, splits, conversions, QA prompts, and deterministic metrics | benchmark protocol |
| LightMem | document-fact prompt on document tasks; released roles and timestamps on LoCoMo | input-domain adapter |
| LightMem | pass its existing `response_format` argument to vLLM; preserve the corrected source ID when constructing an entry | two implementation bugs, no method change |
| Mem0 | document custom instruction on document tasks; released roles on LoCoMo | input-domain adapter |
| HippoRAG 2 | request the JSON object already demonstrated by its OpenIE prompt | local structured-output adapter |
| Measurement | build time, per-query retrieval time, and returned generation-token counts | efficiency reporting only |

LightMem retains official pre-compression, topic segmentation, summarization,
offline updating, and its 16,000-token generation allowance. Mem0 retains its
official memory extraction/update pipeline with a 2,000-token allowance.
HippoRAG 2 retains its graph pipeline with 512-token NER and 2,048-token triple
extraction allowances. The repository has no DeepSeek or Google SDK execution
path and no unused experimental memory algorithm in the final grid.

## Checkout

```bash
git clone --recurse-submodules https://github.com/Sizchode/agent-memory.git
git submodule update --init --recursive
```

The environment setup applies the exact, versioned compatibility patches in
`baseline_patches/` before installing the two affected editable submodules.
