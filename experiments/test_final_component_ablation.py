"""Component-seam tests; scientific result checks run on every actual question."""
from types import SimpleNamespace
import unittest
import numpy as np
import igraph as ig

from experiments.final_component_ablation import KeepCandidateFacts, retrieval_intervention, check_reset, VARIANTS
from experiments.recommendation_findings import source_edge_weights


class ComponentTests(unittest.TestCase):
    def test_identity_preserves_native_candidate_ids_and_order(self):
        recognize = KeepCandidateFacts()
        facts = [("a", "r", "b"), ("c", "r", "d")]
        ids, actual, _ = recognize("q", facts, [9, 2], 5)
        self.assertEqual(ids, [9, 2])
        self.assertEqual(actual, facts)
        self.assertEqual(recognize.calls[0]["indices"], ids)

    def test_identity_rejects_misaligned_candidates(self):
        with self.assertRaises(ValueError):
            KeepCandidateFacts()("q", [("a", "r", "b")], [], 5)

    def test_uniform_weights_preserve_topology_and_restore_on_error(self):
        graph = ig.Graph(n=4, edges=[(0, 2), (1, 2), (1, 3)])
        graph.vs["name"] = ["e1", "e2", "p1", "p2"]
        graph.es["weight"] = [1.5, 2.0, 1.25]
        engine = SimpleNamespace(graph=graph, rerank_filter=KeepCandidateFacts())
        memory = SimpleNamespace(_memory=engine, base=object())
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            with retrieval_intervention(memory, "uniform_weights"):
                self.assertEqual(engine.graph.get_edgelist(), graph.get_edgelist())
                self.assertEqual(engine.graph.vs["name"], graph.vs["name"])
                self.assertEqual(engine.graph.es["weight"], [1, 1, 1])
                raise RuntimeError("test interruption")
        self.assertIs(engine.graph, graph)
        self.assertEqual(graph.es["weight"], [1.5, 2.0, 1.25])

    def test_recognizer_replacement_restores_native_filter(self):
        native = object()
        engine = SimpleNamespace(graph=object(), rerank_filter=native)
        memory = SimpleNamespace(_memory=engine, base=object())
        with retrieval_intervention(memory, "without_recognizer") as identity:
            self.assertIs(engine.rerank_filter, identity)
        self.assertIs(engine.rerank_filter, native)

    def test_fixed_initialization_rejects_changed_scores_or_fallback(self):
        check_reset(np.array([.1, .9]), np.array([.1, .9]))
        check_reset(None, None)
        with self.assertRaises(ValueError):
            check_reset(None, np.array([.1, .9]))
        with self.assertRaises(AssertionError):
            check_reset(np.array([.1, .9]), np.array([.2, .8]))

    def test_membership_equation_deduplicates_source_occurrences(self):
        facts = [("p1", ("a", "r", "b")), ("p2", ("a", "r", "b"))]
        expected = {("a", "p1"): 1 + 1/3, ("b", "p1"): 1 + 1/3,
                    ("a", "p2"): 1 + 1/3, ("b", "p2"): 1 + 1/3}
        self.assertEqual(source_edge_weights(facts + facts, {"a": "a", "b": "b"}), expected)

    def test_all_four_authorized_components_are_declared(self):
        self.assertEqual(set(VARIANTS), {"without_bm25_fusion", "uniform_weights", "without_recognizer", "raw_relations"})


if __name__ == "__main__":
    unittest.main()
