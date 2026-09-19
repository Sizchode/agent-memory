"""Qwen's published yes/no reranker, with batched scoring and no task rules."""

import math
from time import perf_counter


MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ranking(scores):
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Reranker scores must be finite")
    return sorted(range(len(scores)), key=lambda index: -scores[index])


class QwenReranker:
    """Follow the pinned model card's Transformers prompt and binary score."""

    def __init__(self, batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
            local_files_only=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no, self.yes = (self.tokenizer.convert_tokens_to_ids(token) for token in ("no", "yes"))
        assert self.no != self.yes
        self.usage = dict(pairs=0, input_tokens=0, seconds=0.0)

    def score(self, pairs):
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                texts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
                         for query, document in batch]
                ids = self.tokenizer(texts, padding=False, truncation=False,
                                     return_attention_mask=False)["input_ids"]
                ids = [self.prefix + row + self.suffix for row in ids]
                # The control must score the complete candidate, not silently clip it.
                if any(len(row) > 8192 for row in ids):
                    raise ValueError("Candidate exceeds the model-card 8192-token scoring limit")
                inputs = self.tokenizer.pad(dict(input_ids=ids), padding=True, return_tensors="pt")
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                self.torch.cuda.synchronize()
                begin = perf_counter()
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                probabilities = logits[:, [self.no, self.yes]].float().softmax(dim=-1)[:, 1]
                scores.extend(probabilities.cpu().tolist())
                self.usage["seconds"] += perf_counter() - begin
                self.usage["pairs"] += len(batch)
                self.usage["input_tokens"] += sum(map(len, ids))
        ranking(scores)
        return scores

    def close(self):
        del self.model
        from utils.models import release_accelerator_memory
        release_accelerator_memory()
