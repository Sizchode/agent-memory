import tempfile
from pathlib import Path
import unittest

from experiments.backup_manifest import inventory


class BackupManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        (self.base / 'old').mkdir()
        (self.base / 'old/empty').mkdir()
        (self.base / 'old/data').write_bytes(b'original')

    def test_empty_directories_and_files_preserved(self):
        result = inventory(self.base, ['old'])
        self.assertEqual(result['regular_file_bytes'], 8)
        self.assertEqual(result['entries']['old/empty'], {'type': 'directory'})

    def test_symlinks_recorded_without_following_cycles(self):
        (self.base / 'old/cycle').symlink_to(self.base / 'old', target_is_directory=True)
        (self.base / 'old/missing').symlink_to('/a/nonexistent/reference')
        entries = inventory(self.base, ['old'])['entries']
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries['old/missing']['target'], '/a/nonexistent/reference')

    def test_equal_length_content_change_detected(self):
        before = inventory(self.base, ['old'])
        (self.base / 'old/data').write_bytes(b'changed!')
        self.assertNotEqual(before, inventory(self.base, ['old']))

    def test_nonlocal_and_repeated_roots_rejected(self):
        for roots in ([], ['old', 'old'], ['../old'], ['/old'], ['.']):
            with self.assertRaises(ValueError):
                inventory(self.base, roots)

    def test_root_symlink_rejected(self):
        (self.base / 'alias').symlink_to(self.base / 'old', target_is_directory=True)
        with self.assertRaises(ValueError):
            inventory(self.base, ['alias'])


if __name__ == '__main__':
    unittest.main()
