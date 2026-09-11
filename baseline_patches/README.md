# Official baseline patches

The project pins HippoRAG and LightMem as upstream Git submodules. The two
patches in this directory record the minimal local compatibility fixes used by
the final experiment without publishing commits to repositories owned by the
upstream authors.

- `hipporag_local_models.patch` declares the released Qwen embedding runtime
  dependencies and aligns the OpenIE instruction with the JSON object already
  demonstrated by HippoRAG's prompt template.
- `lightmem_vllm_and_source_id.patch` forwards LightMem's existing
  `response_format` argument to its vLLM client and preserves the resolved
  source identifier when a memory entry is created.

`requirements/apply_baseline_patches.sh` applies both exact patches and fails
if the pinned upstream source no longer matches. Environment creation invokes
that script before installing the editable baseline packages.
