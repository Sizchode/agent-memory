# Baseline integration and controlled variables

The repository pins official source trees as Git submodules. Adapters call
their public implementation entry points; algorithm-specific prompts,
summarization, extraction, graph construction, and update rules remain in the
official source.

| Method | Pinned source | Algorithm preserved | Shared controls |
| --- | --- | --- | --- |
| BM25 | local | Okapi BM25 | chunks, top-k |
| Dense retrieval | local | cosine retrieval | embedding function, chunks, top-k |
| LightMem | `zjunlp/LightMem@8449d574` | segmentation, extraction, summaries, offline update, retrieval | LLM, embedding, dimensions, generation budget |
| MAGMA | `FredJiang0324/MAGMA@467cb70b` | event extraction, semantic/temporal/causal/entity graphs, adaptive traversal | LLM and embedding selected in `MemoryBuilder`; retrieval budget |
| HippoRAG 2 | `OSU-NLP-Group/HippoRAG@1438aba3` | OpenIE, graph, fact linking, PPR and official prompts | LLM, embedding, retrieval/QA top-k, output budget |
| Mem0 | `mem0ai/mem0@dae67f74` | fact extraction and ADD/UPDATE/DELETE policy with official prompts | LLM, embedding, vector store, search top-k |

Primary sources:

- LightMem configuration and implementation: <https://github.com/zjunlp/LightMem>
- MAGMA implementation: <https://github.com/FredJiang0324/MAGMA>
- HippoRAG 2 configuration and prompts: <https://github.com/OSU-NLP-Group/HippoRAG>
- Mem0 OSS implementation and configuration: <https://github.com/mem0ai/mem0>

## Policy

Internal prompts that define an algorithm are not replaced. This includes
LightMem summary/update prompts, MAGMA event and relation extraction prompts,
HippoRAG OpenIE/linking prompts, and Mem0 fact/update prompts. The final QA
prompt and answer model are benchmark-runner concerns and must be shared.

The control helpers in `baseline/config.py` apply common model identifiers and
budgets without disabling algorithm-specific operations.

## Decisions required before HPC runs

1. One answer/internal LLM model and endpoint. Decide whether every internal
   extraction call uses this same model; recommended for the primary controlled
   table.
2. One embedding model, endpoint, and exact vector dimension.
3. One input chunk size and whether each algorithm receives benchmark chunks
   or its official internal segmentation input. These are not equivalent.
4. Retrieval budget: same final top-k is required; decide whether large
   internal candidate pools (for example HippoRAG's graph candidate pool) stay
   algorithm-native.
5. Vector store backend. Storage should not change ranking semantics; local
   Qdrant or FAISS are practical choices, but not every official implementation
   supports both through the same interface.
6. MAGMA mapping for document-only benchmarks. LoCoMo has native session/turn
   structure; document benchmarks need a declared mapping of chunk to event and
   timestamp. This mapping affects its temporal graph and must be reported.

## Checkout

```bash
git clone --recurse-submodules https://github.com/Sizchode/agent-memory.git
git submodule update --init --recursive
```
