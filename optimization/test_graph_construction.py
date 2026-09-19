"""Focused tests for frozen graph transforms and cache-failure handling."""

import unittest
import json
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig
import numpy as np

from optimization.retriever.hipporag import CacheMissGuard, GenerationFailureGuard, restrict_fact_index
from optimization.graph_construction.source_consolidation import latest_relation_weights, retained_statements, statement_weights
from optimization.graph_construction.compiled_sources import compile_sources
from optimization.graph_construction.source_window import attach_source_windows
from optimization.graph_construction.statement_incidence import statement_incidence_graph, project_statement_graph
from optimization.retriever.hybrid_graph import fuse_rankings, HybridGraphMemory
from optimization.retriever.compiled_sources import CompiledSourceMemory, compiled_item
from baseline.base import RetrievedItem
from optimization.run_graph import completed_group_prefix
from optimization.retriever.reranker import ranking
from optimization.graph_construction.fact_index import Atom, FactIndex
from optimization.retriever.query_pattern import messages, parse_pattern
from optimization.retriever.query_pattern import PatternAtom
from optimization.retriever.fact_join import FactJoinSearch


class QueryPatternTests(unittest.TestCase):
    def test_question_only_messages_and_shared_variables(self):
        question = "Where does A's parent live?"
        self.assertEqual(messages(question)[1], dict(role="user", content=question))
        content = json.dumps(dict(atoms=[
            dict(subject="A", relation="has parent", object="?person", query="Who is A's parent?"),
            dict(subject="?person", relation="lives in", object="?place", query="Where does the parent live?")]))
        atoms = parse_pattern(content, question)
        self.assertEqual(atoms[0].object, atoms[1].subject)
        self.assertEqual(parse_pattern('{"atoms": []}', question), ())

    def test_rejects_invented_constants_and_wrong_schema(self):
        atom = dict(subject="A", relation="parent", object="Invented name", query="parent of A")
        for content in (json.dumps(dict(atoms=[atom])), '{"atoms": "bad"}',
                        '{"atoms": [], "answer": "secret"}', 'not JSON'):
            with self.assertRaises(ValueError):
                parse_pattern(content, "Who is A's parent?")


class FactJoinSearchTests(unittest.TestCase):
    def test_scoring_validation_direction_and_incomplete_cover(self):
        documents = [dict(idx=source, passage=source, extracted_triples=[triple]) for source, triple in (
            ("a", ["A", "parent", "B"]), ("b", ["B", "lives", "C"]),
            ("other", ["A", "parent", "D"]), ("reverse", ["B", "parent", "A"]))]
        with TemporaryDirectory() as directory:
            index = FactIndex.build(Path(directory) / "facts.sqlite", documents)
            try:
                search = FactJoinSearch(index, str.casefold)
                patterns = [PatternAtom("A", "parent", "?person", "parent"),
                            PatternAtom("?person", "lives", "C", "where")]
                matches = search.search(patterns, lambda text, ids: [1.0] * len(ids))
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].bindings, {"?person": "B"})
                self.assertEqual(search.source_cover(matches[0], 1), ())
                self.assertEqual(search.source_cover(matches[0], 2), ("a", "b"))
                for scorer in (lambda text, ids: [], lambda text, ids: [float("nan")] * len(ids)):
                    with self.assertRaises(ValueError):
                        search.search(patterns, scorer)
                with self.assertRaises(ValueError):
                    search.search(patterns, lambda text, ids: [], beam_width=0)
            finally:
                index.close()

    def test_bound_entity_limits_scoring_and_source_cover_is_complete(self):
        documents = [dict(idx=source, passage=source, extracted_triples=triples) for source, triples in (
            ("a", [["A", "parent", "B"]]), ("b", [["B", "lives", "C"]]),
            ("distractor", [["D", "lives", "E"]]),
            ("together", [["A", "parent", "B"], ["B", "lives", "C"]]))]
        with TemporaryDirectory() as directory:
            index = FactIndex.build(Path(directory) / "facts.sqlite", documents)
            try:
                search = FactJoinSearch(index, str.casefold)
                calls = []

                def score(text, facts):
                    calls.append((text, facts))
                    return [1.0] * len(facts)

                patterns = [PatternAtom("?person", "lives", "?place", "where"),
                            PatternAtom("a", "parent", "?person", "parent")]
                matches = search.search(patterns, score)
                self.assertEqual(len(matches), 1)
                self.assertEqual(calls, [("a parent", [1]), ("B lives", [2])])
                self.assertEqual(matches[0].bindings, {"?person": "B", "?place": "C"})
                self.assertEqual(search.source_cover(matches[0], 1), ("together",))
                self.assertEqual(search.search([PatternAtom("missing", "parent", "?x", "q")], score), [])
            finally:
                index.close()


class FactIndexTests(unittest.TestCase):
    def test_roundtrip_preserves_roles_sources_and_repeated_occurrences(self):
        documents = [dict(idx="a", passage="first", extracted_triples=[
            ["A", "parent", "B"], ["A", "parent", "B"], ["A", "parent", "C"]]),
            dict(idx="b", passage="second", extracted_triples=[["B", "parent", "A"]]),
            dict(idx="empty", passage="without triples", extracted_triples=[])]
        with TemporaryDirectory() as directory:
            index = FactIndex.build(Path(directory) / "facts.sqlite", documents)
            try:
                self.assertEqual(list(index.documents()), documents)
                self.assertEqual(index.connection.execute("SELECT count(*) FROM facts").fetchone()[0], 3)
                with self.assertRaises(FileExistsError):
                    FactIndex.build(Path(directory) / "facts.sqlite", documents)
            finally:
                index.close()

    def test_join_keeps_alternative_sources_separate_from_joint_facts(self):
        documents = [dict(idx=source, passage=source, extracted_triples=[triple]) for source, triple in (
            ("a", ["A", "parent", "B"]), ("b", ["B", "lives", "C"]),
            ("c", ["A", "parent", "B"]), ("d", ["D", "lives", "C"]))]
        with TemporaryDirectory() as directory:
            index = FactIndex.build(Path(directory) / "facts.sqlite", documents)
            try:
                rows = list(index.joins([Atom("A", "?person", (1,)), Atom("?person", "?place", (2, 3))]))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].bindings, (("?person", "B"), ("?place", "C")))
                self.assertEqual(rows[0].fact_ids, (1, 2))
                self.assertEqual(rows[0].sources, (("a", "c"), ("b",)))
                self.assertEqual(list(index.joins([Atom("B", "A", (1,))])), [])
                self.assertEqual(list(index.joins([Atom("?x", "?x", (1, 2))])), [])
                self.assertEqual(list(index.joins([Atom("?x", "?y", ())])), [])
                with self.assertRaises(ValueError):
                    list(index.joins([Atom("?x", "?y", (999,))]))
            finally:
                index.close()


class RerankerTests(unittest.TestCase):
    def test_stable_descending_order(self):
        self.assertEqual(ranking([0.2, 0.8, 0.8, 0.1]), [1, 2, 0, 3])
        self.assertEqual(ranking([]), [])

    def test_nonfinite_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                ranking([0.5, value])


@unittest.skipUnless(os.environ.get("MODULE_ABLATION_ROOT"), "Set MODULE_ABLATION_ROOT after building all six tasks")
class ModuleArtifactTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CHECK_MODULE_CONTROLS"), "Run after all module retrievals")
    def test_module_retrieval_preserves_original_cases_and_configuration(self):
        from experiments.runner import _read_retrieval_records
        from optimization.run_graph import TASKS, MODULES
        from optimization.report_results import TASK_METRICS

        root = Path(os.environ["MODULE_ABLATION_ROOT"])
        main = "statement_projection_loop_free_retained_index_rrf_window"
        for task in TASKS:
            slug = task.replace(" ", "_")
            original = list(_read_retrieval_records(root / main / slug / "retrieval.jsonl"))
            self.assertEqual(len(original), TASK_METRICS[slug][0])
            config = json.loads((root / main / slug / "settings.json").read_text())["config"]
            for module in MODULES:
                directory = root / f"without_{module}" / slug
                rows = list(_read_retrieval_records(directory / "retrieval.jsonl"))
                self.assertEqual(len(rows), len(original))
                self.assertEqual(json.loads((directory / "retrieval_complete.json").read_text())["questions"], len(rows))
                settings = json.loads((directory / "settings.json").read_text())
                self.assertEqual(settings["config"], config)
                self.assertEqual(settings["additional_generator_calls"], 0)
                for old, new in zip(original, rows, strict=True):
                    self.assertEqual((old.group_id, old.case, old.top_k), (new.group_id, new.case, new.top_k))

    @unittest.skipUnless(os.environ.get("CHECK_MODULE_RETRIEVAL"), "Run after full main-method retrieval")
    def test_main_retrieval_preserves_frozen_qa_inputs(self):
        from experiments.runner import _read_retrieval_records, _answer_prompt, _official_generation
        from optimization.run_graph import BASE, TASKS
        from optimization.report_results import TASK_METRICS

        root = Path(os.environ["MODULE_ABLATION_ROOT"])
        main = "statement_projection_loop_free_retained_index_rrf_window"
        frozen = BASE / "optimization_retained_fact_index_seed42_20260914" / main
        for task in TASKS:
            slug = task.replace(" ", "_")
            original = list(_read_retrieval_records(frozen / slug / "retrieval.jsonl"))
            rebuilt = list(_read_retrieval_records(root / main / slug / "retrieval.jsonl"))
            self.assertEqual(len(original), TASK_METRICS[slug][0])
            self.assertEqual(len(rebuilt), len(original))
            old_rng, new_rng = random.Random(42), random.Random(42)
            for old, new in zip(original, rebuilt, strict=True):
                self.assertEqual((old.group_id, old.case, old.top_k), (new.group_id, new.case, new.top_k))
                self.assertEqual(_answer_prompt(old.case, old.retrieved, old_rng),
                                 _answer_prompt(new.case, new.retrieved, new_rng))
                self.assertEqual(_official_generation(old.case), _official_generation(new.case))
            settings = json.loads((root / main / slug / "settings.json").read_text())
            self.assertEqual(settings["additional_generator_calls"], 0)

    def test_full_reconstruction_and_module_boundaries(self):
        from optimization.run_graph import BASE, TASKS, MODULES

        root = Path(os.environ["MODULE_ABLATION_ROOT"])
        main = "statement_projection_loop_free_retained_index_rrf_window"
        frozen = BASE / "optimization_retained_fact_index_seed42_20260914" / main

        def read(path):
            return json.loads(path.read_text())

        def same_graph(first, second):
            self.assertEqual(first.vs["name"], second.vs["name"])
            self.assertEqual(first.get_edgelist(), second.get_edgelist())
            np.testing.assert_array_equal(first.es["weight"], second.es["weight"])

        groups_checked = 0
        for task in TASKS:
            slug = task.replace(" ", "_")
            expected = read(frozen / slug / "build_complete.json")
            self.assertEqual(read(root / main / slug / "build_complete.json"), expected)
            for module in MODULES:
                self.assertEqual(read(root / f"without_{module}" / slug / "build_complete.json"), expected)
            for group in expected["groups"]:
                directory = root / main / slug / "memory" / group
                metadata = read(directory / "graph.json")
                original_metadata = read(frozen / slug / "memory" / group / "graph.json")
                graph = ig.Graph.Read_Pickle(metadata["constructed_graph_file"])
                same_graph(graph, ig.Graph.Read_Pickle(original_metadata["constructed_graph_file"]))
                np.testing.assert_array_equal(np.load(directory / "edge_weights.npy"),
                    np.load(frozen / slug / "memory" / group / "edge_weights.npy"))
                candidates = read(Path(metadata["retained_fact_keys_file"]))
                self.assertEqual(candidates, read(Path(original_metadata["retained_fact_keys_file"])))
                contents = read(Path(metadata["compiled_source_file"]))
                self.assertEqual(contents, read(Path(original_metadata["compiled_source_file"])))
                for module in MODULES:
                    variant = root / f"without_{module}" / slug
                    control = variant / "memory" / group
                    info = read(control / "graph.json")
                    self.assertEqual(info["source_graph"], metadata["source_graph"])
                    if module == "graph":
                        self.assertNotIn("constructed_graph_file", info)
                        source = ig.Graph.Read_Pickle(info["source_graph"])
                        np.testing.assert_array_equal(np.load(control / "edge_weights.npy"), source.es["weight"])
                    elif module != "fact_consolidation":
                        same_graph(graph, ig.Graph.Read_Pickle(info["constructed_graph_file"]))
                        np.testing.assert_array_equal(np.load(control / "edge_weights.npy"),
                                                      np.load(directory / "edge_weights.npy"))
                    if module in ("candidate_index", "fact_consolidation"):
                        self.assertNotIn("retained_fact_keys_file", info)
                    else:
                        self.assertEqual(read(Path(info["retained_fact_keys_file"])), candidates)
                    if module == "evidence_context":
                        self.assertNotIn("rank_fusion", info)
                        self.assertNotIn("compiled_source_file", info)
                    else:
                        self.assertEqual(info["rank_fusion"], metadata["rank_fusion"])
                        if module != "fact_consolidation":
                            self.assertEqual(read(Path(info["compiled_source_file"])), contents)
                    if module == "fact_consolidation":
                        self.assertIsNone(read(variant / "settings.json")["source_schema"])
                        stats_path = Path(info["compiled_source_file"]).parent / "construction.json"
                        stats = read(stats_path)
                        self.assertEqual(stats["source_statements"], stats["retained_statement_support"])
                        self.assertIsNone(stats["source_schema"])
                groups_checked += 1
        self.assertEqual(groups_checked, 15)


class IRCoTGraphTests(unittest.TestCase):
    def test_graph_artifact_selection_preserves_candidates_and_fusion(self):
        from optimization import ircot

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = root / "retained_fact_keys.json"
            keys.write_text(json.dumps(["fact"]))
            rows = {"source": {"content": "Original source text"}}
            memory = SimpleNamespace(_memory=SimpleNamespace(chunk_embedding_store=SimpleNamespace(
                get_all_id_to_rows=lambda: rows)), _generator=object())
            lexical, config = object(), object()
            group = SimpleNamespace(group_id="group")
            for name in ("main", "without_graph"):
                artifact = root / name / "SH-Doc_QA" / "memory" / group.group_id
                artifact.mkdir(parents=True)
                metadata = dict(source_graph="original.pickle", retained_fact_keys_file=str(keys),
                    rank_fusion=dict(rank_constant=60, rank_window=5), compiled_source_file="unused.json")
                if name == "main":
                    metadata["constructed_graph_file"] = "constructed.pickle"
                (artifact / "graph.json").write_text(json.dumps(metadata))
                (artifact / "lexical_source_keys.json").write_text(json.dumps(list(rows)))
                np.save(artifact / "edge_weights.npy", np.array([1.0]))
                for mode in ("graph", "hybrid"):
                    directory = root / f"output_{name}_{mode}"
                    with patch.object(ircot, "GRAPH", root / name), \
                            patch.object(ircot, "GRAPH_RETRIEVAL", mode), \
                            patch("optimization.retriever.hipporag.load_optimized_memory", return_value=memory) as load, \
                            patch.object(ircot, "ElasticsearchMemory", return_value=lexical) as es, \
                            patch.object(ircot, "official_config", return_value=dict(
                                start_state="retrieve", models={"retrieve": {"retrieval_count": 6}})):
                        result = ircot.load_backend("optimized_graph", config, "SH-Doc QA", group, directory)
                    _, index, runtime = load.call_args.args
                    self.assertIs(load.call_args.args[0], config)
                    self.assertEqual(runtime, directory / "runtime/SH-Doc_QA/group")
                    self.assertEqual(json.loads((index / "graph.json").read_text()),
                                     {k: v for k, v in metadata.items()
                                      if k not in ("rank_fusion", "compiled_source_file")})
                    self.assertEqual((index / "edge_weights.npy").resolve(), artifact / "edge_weights.npy")
                    if mode == "graph":
                        self.assertIs(result, memory)
                        es.assert_not_called()
                    else:
                        self.assertIs(result.base, memory)
                        self.assertIs(result.lexical, lexical)
                        self.assertEqual((result.rank_constant, result.rank_window), (60, 6))
                        es.assert_called_once_with("SH-Doc QA", group)

    def test_alternative_graph_requires_separate_output(self):
        import subprocess
        import sys

        result = subprocess.run([sys.executable, "-m", "optimization.ircot", "prepare",
            "--graph-root", "/unused/without_graph"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Use a separate --output-root", result.stderr)


class GraphConstructionTests(unittest.TestCase):

    def test_recognition_cache_reuse_rejects_changed_configuration(self):
        from optimization import run_graph

        with TemporaryDirectory() as temporary:
            task = Path(temporary) / "SH-Doc_QA"
            task.mkdir()
            (task / "settings.json").write_text(json.dumps({"config": {"seed": 42}}))
            args = SimpleNamespace(task="SH-Doc QA", recognition_cache_root=Path(temporary))
            with patch.object(run_graph, "retrieval_config", return_value=({"seed": 43}, None)), \
                    patch.object(run_graph, "_load_groups", return_value=[]):
                with self.assertRaisesRegex(ValueError, "same original model"):
                    run_graph.retrieve(args)

    def test_default_graph_and_query_configuration_agree(self):
        from optimization import build_graph, run_graph

        common = ["--task", "SH-Doc QA", "--output-root", "/unused"]
        for options, retained in (([], True), (["--no-retained-fact-index"], False)):
            with patch("sys.argv", ["build_graph", *common, *options]), patch.object(build_graph, "build") as build:
                build_graph.main()
                args = build.call_args.args[0]
                self.assertEqual(args.construction, "statement_projection_loop_free")
                self.assertEqual(args.retained_fact_index, retained)
                self.assertIsNone(args.without_module)
        with patch("sys.argv", ["run_graph", *common, "--phase", "verify"]), \
                patch.object(run_graph, "_seed_everything"), patch.object(run_graph, "verify") as verify:
            run_graph.main()
            self.assertEqual(verify.call_args.args[0].variants,
                             ["statement_projection_loop_free_retained_index_rrf_window"])

    def test_module_ablation_names_match_build_and_retrieval(self):
        from optimization import build_graph, run_graph

        common = ["--task", "SH-Doc QA", "--output-root", "/unused"]
        for module in run_graph.MODULES:
            with patch("sys.argv", ["build_graph", *common, "--without-module", module]), \
                    patch.object(build_graph, "build") as build:
                build_graph.main()
                args = build.call_args.args[0]
                self.assertEqual(args.without_module, module)
                self.assertTrue(args.retained_fact_index)
                self.assertEqual(args.construction, "statement_projection_loop_free")
            with patch("sys.argv", ["run_graph", *common, "--phase", "verify", "--variants", f"without_{module}"]), \
                    patch.object(run_graph, "_seed_everything"), patch.object(run_graph, "verify") as verify:
                run_graph.main()
                self.assertEqual(verify.call_args.args[0].variants, [f"without_{module}"])

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

    def test_fact_candidate_selection_preserves_vectors_and_provenance(self):
        vectors = np.arange(6).reshape(3, 2)
        provenance = {"entity": {"source"}}
        hippo = SimpleNamespace(fact_node_keys=["a", "b", "c"], fact_embeddings=vectors,
                                ent_node_to_chunk_ids=provenance, proc_triples_to_docs=provenance)
        restrict_fact_index(hippo, ["c", "a"])
        self.assertEqual(hippo.fact_node_keys, ["c", "a"])
        np.testing.assert_array_equal(hippo.fact_embeddings, vectors[[2, 0]])
        self.assertIs(hippo.ent_node_to_chunk_ids, provenance)
        self.assertIs(hippo.proc_triples_to_docs, provenance)
        with self.assertRaises(ValueError):
            restrict_fact_index(hippo, ["a", "a"])
        with self.assertRaises(ValueError):
            restrict_fact_index(hippo, ["missing"])

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
        lexical = SimpleNamespace(retrieve=lambda q, k: seen.append(("lexical", q, k)) or
                                  [RetrievedItem("beta", 1)], close=lambda: None)
        injected = HybridGraphMemory(base, ["p0", "p1"], rank_window=6, lexical=lexical)
        self.assertIs(injected.lexical, lexical)
        self.assertEqual(injected.retrieve("beta", 6)[0].text, "beta")
        self.assertEqual(seen[-2:], [("beta", 6), ("lexical", "beta", 6)])
        injected.close()

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

    def test_all_support_uses_the_same_weighting_without_source_selection(self):
        selected = [(doc["idx"], tuple(triple)) for doc in self.documents for triple in doc["extracted_triples"]]
        keys = {"s": "s", "old": "old", "new": "new"}
        np.testing.assert_array_equal(statement_weights(self.graph, selected, keys), [1, 1, 1, 1, 1, 1])
        retained, _ = retained_statements(self.documents, ["p0", "p1"], lambda x: x)
        np.testing.assert_array_equal(statement_weights(self.graph, retained, keys), self.construct(["p0", "p1"])[0])

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
