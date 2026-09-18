"""Focused tests for frozen graph transforms and cache-failure handling."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig
import numpy as np

from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard, load_optimized_memory
from optimization.graph_construction.source_consolidation import latest_relation_weights, retained_statements
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_window import attach_source_windows
from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph
from optimization.retriever.hybrid_graph import fuse_rankings, HybridGraphMemory
from optimization.retriever.compiled_sources import CompiledSourceMemory, compiled_item
from baseline.base import RetrievedItem
from optimization.run_graph import completed_group_prefix


class GraphConstructionTests(unittest.TestCase):

    def test_default_graph_and_query_configuration_agree(self):
        from optimization import build_graph, run_graph

        common = ["--task", "SH-Doc QA", "--output-root", "/unused"]
        with patch("sys.argv", ["build_graph", *common]), patch.object(build_graph, "build") as build:
            build_graph.main()
            args = build.call_args.args[0]
            self.assertEqual(args.construction, "statement_projection_loop_free")
            self.assertFalse(hasattr(args, "retained_fact_index"))
        with patch("sys.argv", ["run_graph", *common, "--phase", "verify"]), \
                patch.object(run_graph, "_seed_everything"), patch.object(run_graph, "verify") as verify:
            run_graph.main()
            self.assertEqual(verify.call_args.args[0].variants,
                             ["statement_projection_loop_free_raw_relations_refined_rrf_window"])

    def test_retrieval_resume_requires_complete_original_groups(self):
        groups = [SimpleNamespace(group_id="g", cases=[SimpleNamespace(case_id="a", question="q1"),
                                                       SimpleNamespace(case_id="b", question="q2")])]
        rows = [{"group_id": "g", "case": {"case_id": "a", "question": "q1"}},
                {"group_id": "g", "case": {"case_id": "b", "question": "q2"}}]
        self.assertEqual(completed_group_prefix([], groups), set())
        self.assertEqual(completed_group_prefix(rows, groups), {"g"})
        with self.assertRaises(ValueError):
            completed_group_prefix(rows[:1], groups)
        with self.assertRaises(ValueError):
            completed_group_prefix(rows[::-1], groups)

    def test_retired_candidate_filter_is_not_silently_ignored(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "graph.json").write_text(json.dumps({"retained_fact_keys_file": "unused"}))
            with self.assertRaisesRegex(ValueError, "archived implementation"):
                load_optimized_memory(None, directory, directory / "runtime")

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


class StatementIncidenceTests(unittest.TestCase):
    def test_loop_free_projection_preserves_fact_degrees_and_residual_edges(self):
        graph = ig.Graph(n=7, edges=[(4, 0), (4, 1), (4, 2), (5, 0), (5, 3), (1, 2)])
        graph.vs["name"] = ["a", "b", "p", "q", "statement-0", "statement-1", "statement-2"]
        graph.es["weight"] = [1, 1, 1, 1, 1, 0.7]
        projected = project_statement_graph(graph, 4)
        self.assertFalse(any(projected.is_loop()))
        np.testing.assert_allclose(projected.strength(weights="weight"), [2, 1.7, 1.7, 1])
        self.assertAlmostEqual(projected.es[projected.get_eid(0, 1)]["weight"], 0.5)
        self.assertAlmostEqual(projected.es[projected.get_eid(0, 3)]["weight"], 1)
        self.assertAlmostEqual(projected.es[projected.get_eid(1, 2)]["weight"], 1.2)
        self.assertEqual(graph.vcount(), 7)

    def test_loop_free_projection_rejects_singleton_facts(self):
        graph = ig.Graph(n=2, edges=[(0, 1)])
        graph.vs["name"] = ["a", "statement-0"]
        graph.es["weight"] = [1]
        with self.assertRaises(ValueError):
            project_statement_graph(graph, 1)

    def test_projection_implements_complete_membership_transition(self):
        graph = ig.Graph(n=4, edges=[(3, 0), (3, 1), (3, 2)])
        graph.vs["name"] = ["a", "b", "source", "statement-0"]
        graph.es["weight"] = [1, 1, 1]
        projected = project_statement_graph(graph, 3)
        self.assertEqual(projected.vs["name"], ["a", "b", "source"])
        np.testing.assert_allclose(projected.strength(weights="weight"), [1, 1, 1])
        scores = projected.personalized_pagerank(reset=[1, 0, 0], damping=0.5, weights="weight")
        np.testing.assert_allclose(scores, [3 / 5, 1 / 5, 1 / 5])

    def test_projection_keeps_residual_edges_and_ignores_unsupported_statements(self):
        graph = ig.Graph(n=5, edges=[(0, 1), (3, 0), (3, 1), (3, 2)])
        graph.vs["name"] = ["a", "b", "source", "statement-0", "statement-1"]
        graph.es["weight"] = [0.7, 1, 1, 1]
        projected = project_statement_graph(graph, 3)
        np.testing.assert_allclose(projected.strength(weights="weight"), [1.7, 1.7, 1])
        self.assertAlmostEqual(projected.es[projected.get_eid(0, 1)]["weight"], 0.7 + 1 / 2)
        self.assertEqual(graph.vcount(), 5)

    def test_statement_identity_preserves_relations_and_shared_sources(self):
        graph = ig.Graph(n=4, edges=[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
        graph.vs["name"] = ["a", "b", "p0", "p1"]
        graph.es["weight"] = [3, 1, 1, 1, 1]
        graph.es["passage_source"] = [None, "p0", "p0", "p1", "p1"]
        graph.es["synonym_score"] = [0.7, 0, 0, 0, 0]
        documents = [{"idx": "p0", "extracted_triples": [["a", "r", "b"], ["a", "s", "b"]]},
                     {"idx": "p1", "extracted_triples": [["a", "r", "b"]]}]
        selected = [(d["idx"], tuple(t)) for d in documents for t in d["extracted_triples"]]
        result = statement_incidence_graph(graph, documents, selected, {"a": "a", "b": "b"},
                                           lambda x: x, refined=False)
        self.assertEqual(result.vs["name"][:4], graph.vs["name"])
        self.assertEqual(result.vcount(), 6)
        self.assertEqual(set(result.neighbors(4)), {0, 1, 2, 3})
        self.assertEqual(set(result.neighbors(5)), {0, 1, 2})
        self.assertEqual(result.es[result.get_eid(0, 1)]["weight"], 0.7)
        self.assertEqual(graph.es["weight"], [3, 1, 1, 1, 1])
        graph.es["weight"] = [1, 0, 0, 1, 1]
        refined = statement_incidence_graph(graph, documents, [("p1", ("a", "r", "b"))],
                                            {"a": "a", "b": "b"}, lambda x: x, refined=True)
        self.assertEqual(refined.vs["name"], result.vs["name"])
        self.assertEqual(set(refined.neighbors(4)), {0, 1, 3})
        self.assertEqual(refined.degree(5), 0)
        self.assertEqual(refined.degree(2), 0)
        self.assertEqual(refined.get_eid(0, 1, error=False), -1)

    def test_duplicate_and_self_statements_use_binary_membership(self):
        graph = ig.Graph(n=2, edges=[(0, 1)])
        graph.vs["name"] = ["a", "p"]
        graph.es["weight"], graph.es["passage_source"], graph.es["synonym_score"] = [1], ["p"], [0]
        documents = [{"idx": "p", "extracted_triples": [["a", "r", "a"], ["a", "r", "a"]]}]
        result = statement_incidence_graph(graph, documents, [("p", ("a", "r", "a"))] * 2,
                                           {"a": "a"}, lambda x: x, refined=False)
        self.assertEqual(result.vcount(), 3)
        self.assertEqual(set(result.neighbors(2)), {0, 1})
        self.assertTrue(result.is_simple())


if __name__ == "__main__":
    unittest.main()
