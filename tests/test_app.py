"""CPU-only checks for shared generation and the workspace adapters."""
import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from zimage_app.inputs import load_records
from zimage_app.service import GenerationRequest, GenerationService
from zimage_app.settings import Settings
from zimage_app.workflows import batch_generate


class InputTests(unittest.TestCase):
    def test_supported_formats_and_selection(self):
        cases = [
            ({"scenarios": [{"id": "s1", "reference_image": "scene"}]}, "s1"),
            ({"samples": [{"source_sample_id": "MSR-001", "dataset_key": "MSR", "image_prompt": "scene"}]}, "MSR-001"),
            ({"prompts": ["scene"]}, "1"),
            ([{"id": "x", "prompt": "scene"}], "x"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            for data, expected in cases:
                path.write_text(json.dumps(data))
                self.assertEqual(load_records(path)[0]["id"], expected)
            path.write_text(json.dumps(cases[1][0]))
            self.assertEqual(len(load_records(path, datasets=["MSR"], ids="MSR-001")), 1)
            with self.assertRaises(ValueError):
                load_records(path, ids="MSR-999")
        self.assertEqual(len(load_records(text="a\n\nb", limit=1)), 1)

    def test_invalid_request(self):
        for request in [GenerationRequest(""), GenerationRequest("x", height=721),
                        GenerationRequest("x", steps=0), GenerationRequest("x", seed=-1)]:
            with self.assertRaises(ValueError):
                request.validate()


class MockVae:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1, shift_factor=0)

    def decode(self, latents, **kwargs):
        return (latents,)


class MockService(GenerationService):
    loads = 0

    def load(self):
        if self.components is None:
            self.loads += 1
            self.components = {"vae": MockVae()}
            self.device = "cpu"


def fake_generate(**kwargs):
    latents = torch.randn((1, 3, 4, 4), generator=kwargs["generator"])
    for i in range(kwargs["num_inference_steps"]):
        kwargs["callback_on_step_end"](i, torch.tensor(900. - i), latents)
    return [Image.new("RGB", (4, 4), "red")]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.service = MockService(replace(Settings(), output_dir=root / "outputs", step_dir=root / "outputs2"))
        self.patch = patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=fake_generate)})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_save_steps_metadata_reuse_and_unique_runs(self):
        request = GenerationRequest("scene", steps=3, save_steps=True)
        first = self.service.generate(request)
        second = self.service.generate(request)
        self.assertEqual(self.service.loads, 1)
        self.assertNotEqual(first["image"], second["image"])
        self.assertEqual(len(first["steps"]), 3)
        self.assertTrue(all(Path(p).is_file() for p in first["steps"]))
        self.assertTrue(Path(first["step_video"]).is_file())
        metadata = json.loads(Path(first["metadata"]).read_text())
        self.assertEqual(metadata["step_video"], first["step_video"])
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(len(self.service.history()[0]), 2)
        self.service.unload()
        self.service.generate(replace(request, save_steps=False))
        self.assertEqual(self.service.loads, 2)

    def test_batch_increments_seed_preserves_ids(self):
        results = list(batch_generate(self.service, GenerationRequest("batch"), None, "a\nb", [], "", 0, False))
        self.assertEqual(len(results[-1][0]), 2)
        files = results[-1][1]
        metadata = [json.loads(Path(p).read_text()) for p in files if p.endswith("metadata.json")]
        self.assertEqual([m["seed"] for m in metadata], [42, 43])
        self.assertEqual([m["extra"]["id"] for m in metadata], ["1", "2"])

    def test_stream_delivers_preview_before_sampling_finishes(self):
        release = threading.Event()
        finished = threading.Event()

        def paused_generate(**kwargs):
            latents = torch.zeros((1, 3, 4, 4))
            kwargs["callback_on_step_end"](0, torch.tensor(900.), latents)
            if not release.wait(5):
                raise RuntimeError("test timed out waiting for preview consumption")
            kwargs["callback_on_step_end"](1, torch.tensor(800.), latents)
            finished.set()
            return [Image.new("RGB", (4, 4), "red")]

        with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=paused_generate)}):
            stream = self.service.generate_stream(GenerationRequest("scene", steps=2, save_steps=True))
            try:
                first = next(stream)
                self.assertFalse(finished.is_set())
                self.assertTrue(Path(first["image"]).is_file())
                self.assertEqual(len(first["steps"]), 1)
                self.assertEqual(json.loads(Path(first["metadata"]).read_text())["status"], "running")
                release.set()
                remaining = list(stream)
            finally:
                release.set()
                stream.close()
        self.assertEqual(len(first["steps"]), 1)
        self.assertEqual(len(remaining[0]["steps"]), 2)
        self.assertTrue(remaining[-1]["image"].endswith("final.png"))
        self.assertEqual(json.loads(Path(remaining[-1]["metadata"]).read_text())["status"], "completed")

    def test_stream_failure_preserves_preview_and_reports_error(self):
        def fail_after_step(**kwargs):
            kwargs["callback_on_step_end"](0, torch.tensor(900.), torch.zeros((1, 3, 4, 4)))
            raise RuntimeError("sampling failed")

        with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=fail_after_step)}):
            stream = self.service.generate_stream(GenerationRequest("scene", save_steps=True))
            preview = next(stream)
            self.assertTrue(Path(preview["image"]).is_file())
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                next(stream)
        metadata = json.loads(Path(preview["metadata"]).read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["step_images"], preview["steps"])

    def test_stream_without_steps_returns_only_final_image(self):
        results = list(self.service.generate_stream(GenerationRequest("scene", steps=2)))
        self.assertEqual(results[-1]["steps"], [])
        self.assertIsNone(results[-1]["step_video"])
        self.assertTrue(results[-1]["image"].endswith("final.png"))
        self.assertFalse(self.service.settings.step_dir.exists())

    def test_video_failure_preserves_successful_generation(self):
        with patch("zimage_app.step_video.create_step_video", side_effect=RuntimeError("encoder unavailable")):
            result = self.service.generate(GenerationRequest("scene", steps=2, save_steps=True))
        self.assertTrue(Path(result["image"]).is_file())
        self.assertEqual(len(result["steps"]), 2)
        self.assertIsNone(result["step_video"])
        self.assertIn("encoder unavailable", result["status"])
        metadata = json.loads(Path(result["metadata"]).read_text())
        self.assertEqual(metadata["status"], "completed")
        self.assertIn("encoder unavailable", metadata["step_video_error"])

    def test_failed_run_is_recorded(self):
        def fail(**kwargs):
            raise RuntimeError("mock failure")
        with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=fail)}):
            with self.assertRaisesRegex(RuntimeError, "mock failure"):
                self.service.generate(GenerationRequest("scene"))
        metadata = next(self.service.settings.output_dir.glob("*/*/metadata.json"))
        self.assertEqual(json.loads(metadata.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
