import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments.encoder_robustness import (json_save, validate_bundle, install_bundle,
                                           validate_tasks, protocol_tasks, TASKS, SUPPORTED_TASKS)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.source = {name + "_ids": [name + "0", name + "1"] for name in ("passage", "fact", "query")}
        self.bundle = {key: np.asarray(value) for key, value in self.source.items()}
        self.bundle.update({name: np.eye(2) for name in ("passage", "fact", "query_triple", "query_passage")})

    def test_valid(self):
        validate_bundle(self.source, self.bundle)

    def test_reject_invalid_bundle(self):
        for field, value in (("fact_ids", np.array(["fact1", "fact0"])),
                             ("fact", np.eye(3)), ("passage", np.full((2, 2), np.nan)),
                             ("query_triple", np.zeros((2, 2)))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_bundle(self.source, dict(self.bundle, **{field: value}))

    def test_install_replaces_all_online_vectors(self):
        self.source.update(graph_names=["v0", "v1"], graph_edges=[[0, 1]], graph_weights=[2.])
        graph = SimpleNamespace(vs={"name": ["v0", "v1"]}, es={"weight": [2.]}, get_edgelist=lambda: [(0, 1)])
        engine = SimpleNamespace(graph=graph, fact_node_keys=self.source["fact_ids"],
            passage_node_keys=self.source["passage_ids"], embedding_model=SimpleNamespace(),
            query_to_embedding={"triple": {"old": np.ones(4)}})
        install_bundle(engine, self.source, self.bundle)
        self.assertEqual(set(engine.query_to_embedding["triple"]), set(self.source["query_ids"]))
        self.assertEqual(engine.fact_embeddings.shape, (2, 2))
        self.assertEqual(engine.passage_embeddings.shape, (2, 2))
        with self.assertRaises(RuntimeError):
            engine.embedding_model.batch_encode(["unregistered question"])
        graph.es["weight"] = [3.]
        with self.assertRaises(AssertionError):
            install_bundle(engine, self.source, self.bundle)

    def test_immutable_json_normalizes_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            json_save(path, {"edges": [(0, 1)]})
            json_save(path, {"edges": [(0, 1)]})
            with self.assertRaises(ValueError):
                json_save(path, {"edges": [(1, 0)]})

    def test_concurrent_identical_protocol_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: json_save(path, {"seed": 42}), range(32)))


class TaskScopeTests(unittest.TestCase):
    def test_legacy_and_expanded_scopes(self):
        self.assertEqual(validate_tasks(TASKS), ("2WikiMultiHopQA", "LoCoMo"))
        extension = ("SH-Doc_QA", "MH-Doc_QA", "FactConsolidation-SH", "FactConsolidation-MH")
        self.assertEqual(validate_tasks(extension), extension)
        self.assertEqual(set(validate_tasks(SUPPORTED_TASKS)), set(TASKS + extension))

    def test_reject_missing_duplicate_unknown_tasks(self):
        for tasks in ((), ("LoCoMo", "LoCoMo"), ("unapproved",)):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_tasks(tasks)

    def test_frozen_scope_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            json_save(path, {"tasks": ["FactConsolidation-SH", "FactConsolidation-MH"]})
            self.assertEqual(protocol_tasks(directory), ("FactConsolidation-SH", "FactConsolidation-MH"))
            with self.assertRaises(ValueError):
                json_save(path, {"tasks": ["LoCoMo"]})


if __name__ == "__main__":
    unittest.main()
