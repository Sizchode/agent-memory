import unittest

from dataset_loader import TaskName
from dataset_loader.loader import _matches_source


class LoaderTests(unittest.TestCase):
    def test_single_loader_exports_memory_agent_tasks(self) -> None:
        self.assertEqual(TaskName.EVENT_QA.value, "EventQA")

    def test_source_family_match_is_explicit(self) -> None:
        self.assertTrue(_matches_source("factconsolidation_mh_64k", ("factconsolidation_mh_",)))
        self.assertFalse(_matches_source("factconsolidation_sh_64k", ("factconsolidation_mh_",)))


if __name__ == "__main__":
    unittest.main()
