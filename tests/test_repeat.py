"""Exercise repeat jobs using CPU mock generation and real video encoding."""
import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zimage_app.repeat import repeat_generate
from zimage_app.service import GenerationRequest
from zimage_app.settings import Settings
from zimage_app.ui import build_app
from zimage_app.diagnostics import plot_data
from test_app import MockService, fake_generate


class RepeatTests(unittest.TestCase):
    def setUp(self):
        # Load plot dependencies before patch.dict restores sys.modules after each test.
        plot_data([])
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = replace(Settings(), output_dir=root / "outputs", step_dir=root / "outputs2")
        self.service = MockService(self.settings)
        self.generation = patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=fake_generate)})
        self.generation.start()

    def tearDown(self):
        self.generation.stop()
        self.tmp.cleanup()

    def test_increment_paths_zip_video_and_settings_preserved(self):
        updates = list(repeat_generate(self.service, GenerationRequest("same prompt", steps=2, save_steps=True), 3))
        result = updates[-1]
        root = Path(result["directory"])
        self.assertEqual(root.parent, self.settings.step_dir)
        self.assertEqual(self.service.settings, self.settings)
        self.assertEqual(self.service.loads, 1)
        self.assertFalse(self.settings.output_dir.exists())
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["seeds"], [42, 43, 44])
        self.assertEqual(manifest["completed_count"], 3)
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(Path(result["video"]).is_file())
        for index, seed in enumerate([42, 43, 44], 1):
            case = root / "cases" / f"case_{index:04d}"
            self.assertTrue((case / "final.png").is_file())
            self.assertTrue((case / "steps" / "step_001.png").is_file())
            data = json.loads((case / "metadata.json").read_text())
            self.assertEqual(data["seed"], seed)
            self.assertEqual(data["prompt"], "same prompt")
            self.assertEqual(data["extra"]["case_index"], index)
        with zipfile.ZipFile(result["archive"]) as archive:
            names = archive.namelist()
            self.assertIn(f"{root.name}/cases.mp4", names)
            self.assertIn(f"{root.name}/cases/case_0003/final.png", names)
            self.assertIn(f"{root.name}/manifest.json", names)
            self.assertFalse(any(name.endswith(".zip") for name in names))
        self.assertEqual([len(u["gallery"]) for u in updates], [0, 1, 2, 3, 3])

    def test_random_unique_seeds_persisted_and_jobs_isolated(self):
        with patch("zimage_app.repeat.secrets.randbelow", side_effect=[9, 9, 100, 2]):
            result = list(repeat_generate(self.service, GenerationRequest("scene"), 3, "random"))[-1]
        root = Path(result["directory"])
        self.assertEqual(json.loads((root / "manifest.json").read_text())["seeds"], [9, 100, 2])
        second = list(repeat_generate(self.service, GenerationRequest("scene"), 1))[-1]
        self.assertNotEqual(second["directory"], result["directory"])
        self.assertFalse((root / "cases" / "case_0001" / "steps").exists())

    def test_failed_case_retains_partial_downloads(self):
        calls = 0

        def fail_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("mock OOM")
            return fake_generate(**kwargs)

        with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=fail_second)}):
            result = list(repeat_generate(self.service, GenerationRequest("scene"), 3))[-1]
        data = json.loads((Path(result["directory"]) / "manifest.json").read_text())
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["failed_case"], 2)
        self.assertEqual(data["completed_count"], 1)
        self.assertTrue(Path(result["archive"]).is_file())
        self.assertTrue(Path(result["video"]).is_file())
        self.assertIn("mock OOM", result["status"])

    def test_video_failure_still_exports_cases(self):
        with patch("zimage_app.repeat.create_sequence_video", side_effect=RuntimeError("no encoder")):
            result = list(repeat_generate(self.service, GenerationRequest("scene"), 1))[-1]
        self.assertIsNone(result["video"])
        self.assertTrue(Path(result["archive"]).is_file())
        self.assertIn("no encoder", result["status"])

    def test_invalid_options_create_no_task(self):
        cases = [dict(count=0), dict(count=1.5), dict(seed_mode="bad"), dict(fps=0),
                 dict(fps=float("nan")), dict(count=2, request=GenerationRequest("scene", seed=2**63 - 1))]
        for options in cases:
            request = options.pop("request", GenerationRequest("scene"))
            with self.assertRaises(ValueError):
                list(repeat_generate(self.service, request, **options))
        self.assertFalse(self.settings.step_dir.exists())

    def test_gradio_repeat_output_binding(self):
        app = build_app(self.settings, self.service)
        binding = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == "repeat")
        updates = list(binding.fn("same prompt", "", 2, "increment", 2, 720, 1280, 2, 0, 42, False))
        for update in updates:
            self.assertEqual(len(update), len(binding.outputs))
            for component, value in zip(binding.outputs, update):
                component.postprocess(value)
        self.assertTrue(updates[-1][1].endswith("cases.mp4"))
        self.assertTrue(updates[-1][2].endswith(".zip"))


if __name__ == "__main__":
    unittest.main()
