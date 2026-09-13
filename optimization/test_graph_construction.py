"""Focused tests for frozen graph transforms and cache-failure handling."""

import unittest
from types import SimpleNamespace

import igraph as ig
import numpy as np

from optimization.graph_construction.context_weights import VARIANTS, construct_weights
from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard
from optimization.graph_construction.source_consolidation import latest_relation_weights
from optimization.graph_construction.fact_index import construct_fact_index, apply_fact_index
from optimization.run_fact_index import completed_group_prefix
from optimization.graph_construction.weight_grid import CONFIGS, calibrated_weights
from optimization.graph_construction.provenance_weights import provenance_weights
from optimization.graph_construction.adaptive_synonyms import entity_compatibility, adaptive_weights
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_window import attach_source_windows
from optimization.retriever.hybrid_graph import fuse_rankings, HybridGraphMemory
from optimization.retriever.compiled_sources import CompiledSourceMemory, compiled_item, pack_source_items
from baseline.base import RetrievedItem


class GraphConstructionTests(unittest.TestCase):
    def test_source_packing_preserves_all_unique_records_and_reserves_central_sources(self):
        contents = {str(i): dict(text=f"record{i}", original_source_text=f"record{i}",
                                source_position=i, timestamp="A") for i in range(4)}
        contents = attach_source_windows(contents, list(contents), 3)
        items = [RetrievedItem("record0", 0.8), RetrievedItem("record2", 0.6)]
        packed = pack_source_items(items, {f"record{i}": str(i) for i in range(4)}, contents)
        combined = "\n".join(item.text for item in packed)
        for i in range(4):
            self.assertEqual(combined.count(f"record{i}"), 1)
        self.assertEqual(packed[0].metadata["window_sources"], ["1", "3"])
        self.assertEqual(packed[1].metadata["window_sources"], [])
        self.assertEqual([item.score for item in packed], [0.8, 0.6])

    def test_source_packing_without_neighbors_does_not_change_text(self):
        contents = attach_source_windows({"p": dict(text="base", original_source_text="raw",
                                                    source_position=0, timestamp=None)}, ["p"])
        packed = pack_source_items([RetrievedItem("raw")], {"raw": "p"}, contents)
        self.assertEqual(packed[0].text, "base")

    def test_sentence_serialization_preserves_original_triple_fields(self):
        triples = [["Alice", "did not visit", "Rome"], ["Alice", "visited", "Paris"]]
        contents = compile_sources([dict(idx="p", passage="Source", extracted_triples=triples)],
                                   ["p"], [("p", tuple(triple)) for triple in triples], {}, lambda x: x,
                                   with_source=True, facts_format="sentences")
        self.assertEqual(contents["p"]["retained_triples"], triples)
        self.assertIn("Alice did not visit Rome.\nAlice visited Paris.", contents["p"]["text"])

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

    def setUp(self):
        self.graph = ig.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (1, 4)])
        self.graph.vs["name"] = ["e0", "e1", "e2", "p0", "p1"]
        self.graph.es["weight"] = [0.9, 2.0, 1.0, 1.0]
        self.graph.es["synonym_score"] = [0.9, 0.8, 0.0, 0.0]
        self.graph.es["fact_source_counts"] = [{}, {"p0": 2}, {}, {}]
        self.graph.es["passage_source"] = [None, None, "p0", "p1"]
        self.sources = {"e0": {"p0"}, "e1": {"p1"}, "e2": {"p0"}}

    def weights(self):
        return construct_weights(self.graph, self.sources, ["p0", "p1"], [[1, 0], [0, 1]])

    def test_context_disagreement_removes_only_unsupported_synonym(self):
        np.testing.assert_array_equal(self.weights()["contextual"], [0, 2, 1, 1])

    def test_source_graph_and_inputs_remain_unchanged(self):
        before = self.graph.es["weight"][:]
        self.weights()
        self.assertEqual(before, self.graph.es["weight"])
        self.assertEqual(self.sources["e0"], {"p0"})

    def test_all_variants_have_finite_nonnegative_aligned_weights(self):
        weights = self.weights()
        self.assertEqual(set(weights), set(VARIANTS))
        for values in weights.values():
            self.assertEqual(values.shape, (self.graph.ecount(),))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue((values >= 0).all())

    def test_missing_context_preserves_source_supported_edge(self):
        self.sources.clear()
        np.testing.assert_array_equal(self.weights()["contextual"], [0, 2, 1, 1])

    def test_node_order_does_not_change_transformation(self):
        expected = self.weights()
        reordered = self.graph.permute_vertices([4, 2, 0, 3, 1])
        actual = construct_weights(reordered, self.sources, ["p0", "p1"], [[1, 0], [0, 1]])
        for variant in VARIANTS:
            np.testing.assert_allclose(expected[variant], actual[variant])

    def test_weight_grid_scales_only_synonym_support_before_normalization(self):
        source = np.array([0., 2., 1., 1.])
        variants = calibrated_weights(self.graph, source)
        self.assertEqual(set(variants), set(CONFIGS))
        np.testing.assert_allclose(variants["canonical_syn005"], [0.045, 2, 1, 1])
        np.testing.assert_allclose(variants["canonical_syn020"], [0.18, 2, 1, 1])
        np.testing.assert_array_equal(source, [0, 2, 1, 1])
        for weights in variants.values():
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue((weights >= 0).all())

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

    def test_resume_accepts_only_complete_original_source_groups(self):
        groups = [SimpleNamespace(group_id="a", cases=[SimpleNamespace(case_id="1", question="q1")]),
                  SimpleNamespace(group_id="b", cases=[SimpleNamespace(case_id="2", question="q2"),
                                                       SimpleNamespace(case_id="3", question="q3")])]
        rows = [{"group_id": "a", "case": {"case_id": "1", "question": "q1"}}]
        self.assertEqual(completed_group_prefix(rows, groups), {"a"})
        with self.assertRaises(ValueError):
            completed_group_prefix(rows + [{"group_id": "b", "case": {"case_id": "2", "question": "q2"}}], groups)
        with self.assertRaises(ValueError):
            completed_group_prefix([{"group_id": "a", "case": {"case_id": "1", "question": "changed"}}], groups)


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
        np.testing.assert_array_equal(weights["latest_relation"], [0, 1, 0, 0, 1, 1])
        self.assertEqual(stats["retained_statement_support"], 1)
        self.assertEqual(stats["original_passages_preserved"], 2)

    def test_provenance_fraction_penalizes_mixed_old_and_retained_support(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        self.documents[1]["extracted_triples"].append(["s", "other", "new"])
        base = np.array([0., 2., 1., 0., 1., 1.])
        retained = [("p0", ("s", "other", "new")), ("p1", ("s", "r", "new"))]
        weights = provenance_weights(self.graph, base, self.documents, retained,
                                     {"s": "s", "old": "old", "new": "new"}, lambda x: x)
        np.testing.assert_array_equal(weights["canonical_provenance"], [0, 2, 0.5, 0, 1, 1])
        np.testing.assert_array_equal(base, [0, 2, 1, 0, 1, 1])

    def test_provenance_rejects_support_not_present_in_sources(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        with self.assertRaises(ValueError):
            provenance_weights(self.graph, np.ones(6), self.documents,
                [("missing", ("s", "r", "new"))], {"s": "s", "old": "old", "new": "new"}, lambda x: x)

    def test_compatibility_does_not_treat_duplicate_current_facts_as_conflicts(self):
        self.documents[1]["extracted_triples"] = [["s", "alias", "new"]]
        schema = {label: {"canonical": "r"} for label in ("r", "alias")}
        score = entity_compatibility(self.documents, [("p1", ("s", "r", "new"))],
                                     {"s": "s", "new": "new"}, lambda x: x, schema)
        self.assertEqual(score, {"s": 1.0, "new": 1.0})

    def test_overwritten_assertions_reduce_entity_compatibility(self):
        schema = {"r": {"canonical": "r"}}
        score = entity_compatibility(self.documents, [("p1", ("s", "r", "new"))],
                                     {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema)
        self.assertEqual(score, {"s": 0.5, "new": 1.0, "old": 0.0})

    def test_adaptive_synonyms_do_not_restore_overwritten_entity_links(self):
        self.graph.es["synonym_score"] = [0.9, 0.9, 0, 0, 0, 0]
        base = np.array([0., 1., 0., 0., 1., 1.])
        weights = adaptive_weights(self.graph, base, self.documents, [("p1", ("s", "r", "new"))],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, {"r": {"canonical": "r"}})
        np.testing.assert_array_equal(weights["adaptive_syn020"], base)

    def test_compiled_context_preserves_surface_form_and_source_metadata(self):
        documents = [{"idx": "p", "passage": "Original passage", "extracted_triples":
                      [["Alice", "lives in", "Rome"], ["Alice", "lives in", "Paris"]]}]
        contents = compile_sources(documents, ["p"], [("p", ("alice", "lives in", "paris"))],
                                   {"p": "2026-01-01"}, lambda xs: [x.lower() for x in xs])
        self.assertIn('["Alice", "lives in", "Paris"]', contents["p"]["text"])
        self.assertNotIn("Rome", contents["p"]["text"])
        self.assertIn("Source timestamp: 2026-01-01", contents["p"]["text"])
        self.assertEqual(contents["p"]["original_source_text"], "Original passage")

    def test_compiled_empty_source_is_preserved_without_fabricated_facts(self):
        contents = compile_sources([{"idx": "p", "passage": "old", "extracted_triples": []}],
                                   ["p"], [], {}, lambda x: x)
        self.assertEqual(contents["p"]["text"], "Source position: 0\nFacts: []")

    def test_source_context_keeps_verbatim_record_and_separates_selected_facts(self):
        original = "Alice lived in Rome.\nAlice now lives in Paris."
        documents = [{"idx": "p", "passage": original, "extracted_triples":
                      [["Alice", "lives in", "Rome"], ["Alice", "lives in", "Paris"]]}]
        contents = compile_sources(documents, ["p"], [("p", ("alice", "lives in", "paris"))],
                                   {}, lambda xs: [x.lower() for x in xs], with_source=True)
        text = contents["p"]["text"]
        self.assertIn(original, text)
        self.assertIn('["Alice", "lives in", "Paris"]', text)
        self.assertNotIn('["Alice", "lives in", "Rome"]', text)
        item = compiled_item(RetrievedItem(original, 0.2), "p", contents["p"])
        self.assertEqual(item.metadata["context_representation"], "original_source_with_retained_triples")

    def test_source_context_keeps_record_even_without_extracted_facts(self):
        contents = compile_sources([{"idx": "p", "passage": "Unextracted detail", "extracted_triples": []}],
                                   ["p"], [], {}, lambda x: x, with_source=True)
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
        np.testing.assert_array_equal(weights["latest_relation"], np.ones(6))
        self.assertEqual(stats["retained_statement_support"], 2)

    def test_reversing_source_order_changes_only_the_selected_support(self):
        weights, _ = self.construct(["p1", "p0"])
        np.testing.assert_array_equal(weights["latest_relation"], [1, 0, 1, 1, 0, 0])

    def test_schema_preserves_multivalued_history(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        schema = {"r": {"canonical": "r", "cardinality": "multiple", "role": "content"}}
        weights, _ = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema)
        np.testing.assert_array_equal(weights["schema_latest"], np.ones(6))

    def test_schema_merges_single_valued_relation_aliases(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        self.documents[1]["extracted_triples"][0][1] = "alias"
        schema = {label: {"canonical": "canonical", "cardinality": "single", "role": "content"}
                  for label in ("r", "alias")}
        weights, _ = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema)
        np.testing.assert_array_equal(weights["schema_latest"], [0, 1, 0, 0, 1, 1])

    def test_explicit_canonical_latest_policy_consolidates_multiple_values(self):
        self.graph.es["synonym_score"] = [0.0] * 6
        schema = {"r": {"canonical": "r", "cardinality": "multiple", "role": "content"}}
        weights, _ = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema,
            consolidate_multiple=True)
        np.testing.assert_array_equal(weights["canonical_latest"], [0, 1, 0, 0, 1, 1])
        self.assertEqual(schema["r"]["cardinality"], "multiple")

    def test_schema_removes_only_classified_discourse_support(self):
        self.graph.es["synonym_score"] = [0.2, 0, 0, 0, 0, 0]
        schema = {"r": {"canonical": "r", "cardinality": "multiple", "role": "discourse"}}
        weights, stats = latest_relation_weights(self.graph, self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, lambda x: x, schema=schema)
        np.testing.assert_array_equal(weights["schema_latest"], np.zeros(6))
        np.testing.assert_array_equal(weights["schema_latest_synonyms"], [0.2, 0, 0, 0, 0, 0])
        self.assertEqual(stats["discourse_statement_support_removed"], 2)

    def test_fact_index_tracks_retained_support_without_rewriting_facts(self):
        rows = {"f0": {"content": "('s', 'r', 'old')"}, "f1": {"content": "('s', 'r', 'new')"}}
        result = construct_fact_index(self.documents, ["p0", "p1"],
            {"s": "s", "old": "old", "new": "new"}, rows, lambda x: x)
        self.assertEqual(result["fact_ids"], ["f1"])
        self.assertEqual(result["entity_sources"], {"s": ["p1"], "new": ["p1"]})
        self.assertEqual(result["fact_sources"], {"('s', 'r', 'new')": ["p1"]})
        self.assertEqual(len(rows), 2)

    def test_retained_fact_must_exist_in_source_index(self):
        with self.assertRaises(ValueError):
            construct_fact_index(self.documents, ["p0", "p1"],
                {"s": "s", "old": "old", "new": "new"}, {}, lambda x: x)

    def test_apply_index_selects_existing_embeddings_and_preserves_store(self):
        vectors = {"f0": np.array([1., 0.]), "f1": np.array([0., 1.])}
        store = SimpleNamespace(get_all_ids=lambda: list(vectors),
                                get_embeddings=lambda keys: [vectors[key] for key in keys])
        hippo = SimpleNamespace(fact_embedding_store=store, entity_node_keys=["s", "old", "new"],
                                passage_node_keys=["p0", "p1"])
        index = dict(fact_ids=["f1"], entity_sources={"new": ["p1"]},
                     fact_sources={"('s', 'r', 'new')": ["p1"]})
        apply_fact_index(hippo, index)
        np.testing.assert_array_equal(hippo.fact_embeddings, [[0, 1]])
        self.assertEqual(hippo.fact_node_keys, ["f1"])
        self.assertEqual(store.get_all_ids(), ["f0", "f1"])

    def test_apply_index_rejects_unknown_provenance(self):
        hippo = SimpleNamespace(fact_embedding_store=SimpleNamespace(get_all_ids=lambda: ["f0"]),
                                entity_node_keys=["s"], passage_node_keys=["p0"])
        with self.assertRaises(ValueError):
            apply_fact_index(hippo, dict(fact_ids=["f0"], entity_sources={"s": ["p1"]}, fact_sources={}))


if __name__ == "__main__":
    unittest.main()
