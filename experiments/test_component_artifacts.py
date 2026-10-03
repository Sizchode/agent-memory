import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.component_artifacts import home_quota_guard, publish_tree, restore_tree, verify_archive, safe_relative


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "qa").mkdir(parents=True)
        (self.source / "qa/results.jsonl").write_text('{"answer":"London"}\n' * 100)
        (self.source / "protocol.json").write_text('{"seed":42}\n')
        self.archive = self.root / "archives"

    def publish(self, **kwargs):
        return publish_tree(self.source, ["qa", "protocol.json"], self.archive, "stage",
            budget_bytes=kwargs.get("budget", 1024**2), quota_check=kwargs.get("guard", lambda n: None))

    def test_complete_byte_identical_round_trip_and_overlap(self):
        published = self.publish()
        target = self.root / "restored"
        restore_tree(self.archive, "stage", target)
        restore_tree(self.archive, "stage", target)
        self.assertEqual((target / "qa/results.jsonl").read_bytes(), (self.source / "qa/results.jsonl").read_bytes())
        self.assertEqual(len(published["files"]), 2)

    def test_storage_limit_prevents_publication(self):
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            self.publish(budget=1)
        self.assertFalse((self.archive / "stage.tar.gz").exists())

    def test_quota_denial_prevents_publication(self):
        def deny(n):
            raise RuntimeError("Quota denied")
        with self.assertRaisesRegex(RuntimeError, "Quota denied"):
            self.publish(guard=deny)
        self.assertFalse((self.archive / "stage.tar.gz").exists())

    def test_corrupt_archive_rejected(self):
        self.publish()
        with (self.archive / "stage.tar.gz").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaises(ValueError):
            verify_archive(self.archive, "stage")

    def test_no_overwrite_of_different_results(self):
        self.publish()
        (self.source / "qa/results.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "different completed"):
            self.publish()

    def test_existing_same_results_reused(self):
        self.assertEqual(self.publish(), self.publish())

    def test_restore_conflict_rejected(self):
        self.publish()
        target = self.root / "restored"
        target.mkdir()
        (target / "protocol.json").write_text("different")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            restore_tree(self.archive, "stage", target)

    def test_symlink_not_archived(self):
        (self.source / "qa/link").symlink_to(self.source / "protocol.json")
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.publish()

    def test_path_traversal_rejected(self):
        for value in ("../outside", "/absolute", "a/../outside", "./alias"):
            with self.assertRaises(ValueError):
                safe_relative(value)

    def test_quota_parsing_and_reserve(self):
        output = "Name Path Used(G) (%) Used SLIMIT(G) H-LIMIT(G)\nu /oscar/home 96.97 96.97 100 125 1000\n"
        self.assertEqual(home_quota_guard(output, 1024**3)["soft_limit_g"], "100")
        with self.assertRaises(RuntimeError):
            home_quota_guard(output, 3 * 1024**3)
        with self.assertRaises(ValueError):
            home_quota_guard("unexpected", 1)


if __name__ == "__main__":
    unittest.main()
