"""Focused tests for frozen graph transforms and cache-failure handling."""

import unittest
from types import SimpleNamespace

import igraph as ig
import numpy as np

from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard
from optimization.graph_construction.source_consolidation import latest_relation_weights, retained_statements
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_window import attach_source_windows
from optimization.retriever.hybrid_graph import fuse_rankings, HybridGraphMemory
from optimization.retriever.compiled_sources import CompiledSourceMemory, compiled_item
from baseline.base import RetrievedItem


class GraphConstructionTests(unittest.TestCase):


    def test_reciprocal_rank_fusion_rewards_shared_sources_and_preserves_stable_ties(self):
        graph = [RetrievedItem("a", 0.9), RetrievedItem("b", 0.1)]
        lexical = [RetrievedItem("b", 20.0), RetrievedItem("c", 10.0)]
        result = fuse_rankings(graph, lexical, 3)
        self.assertEqual([item.text for item in result], ["b", "a", "c"])
        self.assertAlmostEqual(result[0].score, 1 / 61 + 1 / 62)
        self.assertEqual(result[0].metadata["fusion_ranks"], {"graph": 2, "lexical": 1})
        self.assertEqual([item.text for item in fuse_rankings(graph, [RetrievedItem("c", 0.0)], 2)], ["a", "b"])

    def test_reciprocal_rank_fusion_rejects_duplicate_ranking_entries(self):
        with self.assertRaises(ValueError):
            fuse_rankings([RetrievedItem("a"), RetrievedItem("a")], [], 2)

    def test_hybrid_graph_interface_uses_same_source_corpus_for_new_queries(self):
        seen = []
        rows = {"p0": {"content": "alpha"}, "p1": {"content": "beta"}}
        base = SimpleNamespace(_memory=SimpleNamespace(chunk_embedding_store=SimpleNamespace(
            get_all_id_to_rows=lambda: rows)), _generator=None,
            retrieve=lambda q, k: seen.append((q, k)) or [RetrievedItem("alpha", 1), RetrievedItem("beta", 0.5)],
            close=lambda: None, efficiency_metrics=lambda: {})
        memory = HybridGraphMemory(base, ["p0", "p1"])
        self.assertEqual(memory.retrieve("beta", 1)[0].text, "beta")
        self.assertEqual(seen, [("beta", 5)])
        with self.assertRaises(ValueError):
            HybridGraphMemory(base, ["p0"])

    def test_source_window_respects_contiguous_timestamp_boundaries(self):
        contents = {str(i): dict(text=f"node{i}", original_source_text=f"record{i}", timestamp=timestamp)
                    for i, timestamp in enumerate(["A", "A", "B", "A", None])}
        result = attach_source_windows(contents, list(contents), 3)
        self.assertEqual(result["0"]["window_sources"], ["1"])
        self.assertEqual(result["1"]["window_sources"], ["0"])
        self.assertEqual(result["3"]["window_sources"], [])
        self.assertEqual(result["4"]["text"], "node4")
        self.assertEqual(contents["0"]["text"], "node0")

    def test_source_window_retains_complete_neighbors_in_source_order(self):
        contents = {str(i): dict(text=f"node{i}", original_source_text=f"record{i}", timestamp="A")
                    for i in range(5)}
        result = attach_source_windows(contents, list(contents), 1)
        self.assertEqual(result["2"]["window_sources"], ["1", "3"])
        self.assertLess(result["2"]["text"].index("record1"), result["2"]["text"].index("record3"))
        self.assertNotIn("record0", result["2"]["text"])


    def test_cache_miss_cannot_be_reported_as_success(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
        guard = CacheMissGuard(SimpleNamespace(openai_client=client))
        guard.check()
        with self.assertRaises(RuntimeError):
            client.chat.completions.create()
        with self.assertRaises(RuntimeError):
            guard.check()

    def test_swallowed_live_provider_failure_is_still_fatal(self):
        def failed_call():
            raise ConnectionError("test provider failure")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failed_call)))
        guard = GenerationFailureGuard(SimpleNamespace(openai_client=client))
        try:
            client.chat.completions.create()
        except ConnectionError:
            pass
        with self.assertRaises(RuntimeError):
            guard.check()

    def test_successful_live_provider_response_is_unchanged(self):
        response = object()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda: response)))
        guard = GenerationFailureGuard(SimpleNamespace(openai_client=client))
        self.assertIs(client.chat.completions.create(), response)
        guard.check()


class SourceConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = ig.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (1, 3), (0, 4), (2, 4)])
        self.graph.vs["name"] = ["s", "old", "new", "p0", "p1"]
        self.graph.es["passage_source"] = [None, None, "p0", "p0", "p1", "p1"]
        self.documents = [
            {"idx": "p1", "extracted_triples": [["s", "r", "new"]]},
            {"idx": "p0", "extracted_triples": [["s", "r", "old"]]},
        ]

    def construct(self, order):
        return latest_relation_weights(self.graph, self.documents, order,
                                       {"s": "s", "old": "old", "new": "new"}, lambda x: x)

    def test_loader_order_not_export_order_controls_retained_support(self):
        weights, stats = self.construct(["p0", "p1"])
        np.testing.assert_array_equal(weights, [0, 1, 0, 0, 1, 1])
        self.assertEqual(stats["retained_statement_support"], 1)
        self.assertEqual(stats["original_passages_preserved"], 2)

    def test_compiled_context_preserves_surface_form_and_source_metadata(self):
        documents = [{"idx": "p", "passage": "Original passage", "extracted_triples":
                      [["Alice", "lives in", "Rome"], ["Alice", "lives in", "Paris"]]}]
        contents = compile_sources(documents, ["p"], [("p", ("alice", "lives in", "paris"))],
                                   {"p": "2026-01-01"}, lambda xs: [x.lower() for x in xs])
        self.assertIn('["Alice", "lives in", "Paris"]', contents["p"]["text"])
        self.assertNotIn("Rome", contents["p"]["text"])
        self.assertIn("Source timestamp: 2026-01-01", contents["p"]["text"])
        self.assertEqual(contents["p"]["original_source_text"], "Original passage")

    def test_source_context_keeps_verbatim_record_and_separates_selected_facts(self):
        original = "Alice lived in Rome.\nAlice now lives in Paris."
        documents = [{"idx": "p", "passage": original, "extracted_triples":
                      [["Alice", "lives in", "Rome"], ["Alice", "lives in", "Paris"]]}]
        contents = compile_sources(documents, ["p"], [("p", ("alice", "lives in", "paris"))],
                                   {}, lambda xs: [x.lower() for x in xs])
        text = contents["p"]["text"]
        self.assertIn(original, text)
        self.assertIn('["Alice", "lives in", "Paris"]', text)
        self.assertNotIn('["Alice", "lives in", "Rome"]', text)
        item = compiled_item(RetrievedItem(original, 0.2), "p", contents["p"])
        self.assertEqual(item.metadata["context_representation"], "original_source_with_retained_triples")

    def test_source_context_keeps_record_even_without_extracted_facts(self):
        contents = compile_sources([{"idx": "p", "passage": "Unextracted detail", "extracted_triples": []}],
                                   ["p"], [], {}, lambda x: x)
        self.assertIn("Unextracted detail", contents["p"]["text"])
        self.assertTrue(contents["p"]["text"].endswith("Facts: []"))

    def test_live_compiled_readout_preserves_source_rank_and_handles_new_queries(self):
        content = dict(text="Facts: [[1]]", original_source_text="raw", source_position=0, timestamp=None)
        seen = []
        base = SimpleNamespace(_memory=SimpleNamespace(chunk_embedding_store=SimpleNamespace(
            get_all_id_to_rows=lambda: {"p": {"content": "raw"}})), _generator=None,
            retrieve=lambda q, k: seen.append((q, k)) or [RetrievedItem("raw", 0.2)],
            close=lambda: None, efficiency_metrics=lambda: {})
        memory = CompiledSourceMemory(base, {"p": content})
        items = memory.retrieve("An unseen question", 5)
        self.assertEqual(seen, [("An unseen question", 5)])
        self.assertEqual(items[0].text, content["text"])
        self.assertEqual(items[0].score, 0.2)
        self.assertEqual(items[0].metadata["original_source_text"], "raw")

    def test_compiled_readout_rejects_wrong_source_identity(self):
        with self.assertRaises(ValueError):
            compiled_item(RetrievedItem("different"), "p", dict(original_source_text="raw"))

    def test_different_relations_do_not_overwrite_each_other(self):
        self.documents[1]["extracted_triples"][0][1] = "other_relation"
        weights, stats = self.construct(["p0", "p1"])
        np.testing.assert_array_equal(weights, np.ones(6))
        self.assertEqual(stats["retained_statement_support"], 2)

    def test_reversing_source_order_changes_only_the_selected_support(self):
        weights, _ = self.construct(["p1", "p0"])
        np.testing.assert_array_equal(weights, [1, 0, 1, 1, 0, 0])

    def test_canonical_schema_merges_relation_aliases(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        self.documents[1]["extracted_triples"][0][1] = "alias"
        schema = {label: {"canonical": "canonical"}
                  for label in ("r", "alias")}
        weights, _ = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema)
        np.testing.assert_array_equal(weights, [0, 1, 0, 0, 1, 1])

    def test_latest_policy_does_not_use_schema_categories(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        schema = {"r": {"canonical": "r", "cardinality": "multiple", "role": "discourse"}}
        weights, _ = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema)
        np.testing.assert_array_equal(weights, [0, 1, 0, 0, 1, 1])
        self.assertEqual(schema["r"]["cardinality"], "multiple")


if __name__ == "__main__":
    unittest.main()
