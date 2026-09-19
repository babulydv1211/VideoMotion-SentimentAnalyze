import tempfile
import unittest
from pathlib import Path

from scene_motion_llm.main_inference import resolve_checkpoint_path


class InferenceCheckpointTest(unittest.TestCase):
    def test_resolve_checkpoint_path_prefers_package_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "scene_motion_llm" / "checkpoints"
            package_dir.mkdir(parents=True)
            expected = package_dir / "best_model.pt"
            expected.touch()

            resolved = resolve_checkpoint_path(str(root), None)

            self.assertEqual(resolved, str(expected))


if __name__ == "__main__":
    unittest.main()
