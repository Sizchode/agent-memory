import unittest

from baseline import (
    BM25Baseline,
    ControlledConfig,
    DenseRetrievalBaseline,
    control_hipporag,
    control_lightmem,
    control_mem0,
)


class RetrievalBaselineTests(unittest.TestCase):
    def test_bm25_ranks_matching_document_first(self) -> None:
        baseline = BM25Baseline()
        baseline.build(["alpha alpha beta", "gamma delta"])
        self.assertEqual(baseline.retrieve("alpha", 1)[0].text, "alpha alpha beta")

    def test_dense_uses_injected_embedder(self) -> None:
        def embed(texts):
            return [[float("alpha" in text), float("gamma" in text)] for text in texts]

        baseline = DenseRetrievalBaseline(embed)
        baseline.build(["alpha beta", "gamma delta"])
        self.assertEqual(baseline.retrieve("gamma", 1)[0].text, "gamma delta")


class ControlledConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shared = ControlledConfig("shared-llm", "shared-embedder", 5, 512, max_output_tokens=128)

    def test_hipporag_override(self) -> None:
        self.assertEqual(control_hipporag({}, self.shared)["qa_top_k"], 5)

    def test_mem0_override(self) -> None:
        config = control_mem0({"llm": {"config": {}}, "embedder": {"config": {}}}, self.shared)
        self.assertEqual(config["embedder"]["config"]["model"], "shared-embedder")

    def test_lightmem_override(self) -> None:
        config = control_lightmem({"memory_manager": {"configs": {}}, "text_embedder": {"configs": {}}}, self.shared, 768)
        self.assertEqual(config["text_embedder"]["configs"]["embedding_dims"], 768)


if __name__ == "__main__":
    unittest.main()
