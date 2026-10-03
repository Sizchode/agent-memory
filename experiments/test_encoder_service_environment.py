"""Exercise subprocess tool lookup through the actual service env builder."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.encoder_robustness_service import server_environment


class ServiceEnvironmentTests(unittest.TestCase):
    def test_selected_runtime_tools_are_available_to_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime_bin = Path(temporary) / "runtime" / "bin"
            runtime_bin.mkdir(parents=True)
            tool = runtime_bin / "ninja"
            tool.write_text("#!/bin/sh\nprintf 'selected-runtime-tool\\n'\n")
            tool.chmod(0o755)
            inherited = {"PATH": "/nonexistent-parent-path", "CUDA_VISIBLE_DEVICES": "0"}
            environment = server_environment(runtime_bin / "vllm", inherited)
            result = subprocess.run(
                [sys.executable, "-c", "import subprocess; subprocess.run(['ninja', '--version'], check=True)"],
                env=environment, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), "selected-runtime-tool")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(inherited["PATH"], "/nonexistent-parent-path")

    def test_no_inherited_path_does_not_add_current_directory(self):
        environment = server_environment(Path("/runtime/bin/vllm"), {})
        self.assertEqual(environment["PATH"], "/runtime/bin")

    def test_environment_directory_is_not_resolved_through_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "vllm"
            executable.symlink_to(sys.executable)
            environment = server_environment(executable, {"PATH": os.defpath})
            self.assertEqual(environment["PATH"].split(os.pathsep)[0], temporary)


if __name__ == "__main__":
    unittest.main()
