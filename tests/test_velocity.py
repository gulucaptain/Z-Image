"""CPU checks for velocity semantics, streamed diagnostics and UI bindings."""
import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage.pipeline import generate
from zimage.scheduler import FlowMatchEulerDiscreteScheduler
from zimage_app.diagnostics import VelocityDiagnostics, plot_data
from zimage_app.service import GenerationRequest
from zimage_app.settings import Settings
from test_app import MockService


def event(step, velocity, sigma=1., sigma_next=.5):
    return {"step": step, "skipped": False, "timestep": sigma * 1000,
            "model_time": 1 - sigma, "sigma": sigma, "sigma_next": sigma_next,
            "velocity": velocity, "latents": torch.ones_like(velocity), "guidance_scale": 0.}


class DiagnosticsTests(unittest.TestCase):
    def test_metrics_scale_exports_and_undefined_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = VelocityDiagnostics(tmp)
            v = torch.full((1, 3, 2, 2), 2.)
            collector.record(event(1, v))
            collector.record(event(2, -2 * v))
            collector.record(event(3, torch.zeros_like(v)))
            collector.record({"step": 4, "timestep": 0., "skipped": True})
            rows = collector.rows
            self.assertEqual(rows[0]["velocity_rms"], 2.)
            self.assertEqual(rows[0]["update_rms"], 1.)
            self.assertEqual(rows[0]["relative_update"], 1.)
            self.assertIsNone(rows[0]["direction_cosine"])
            self.assertAlmostEqual(rows[1]["direction_cosine"], -1.)
            self.assertIsNone(rows[2]["direction_cosine"])
            self.assertIsNone(rows[3]["velocity_rms"])
            self.assertEqual(collector.heatmap_max, 2.)
            self.assertEqual(rows[1]["heatmap_saturated_fraction"], 1.)
            self.assertEqual(len(collector.maps), 3)
            raw = np.load(Path(tmp) / "velocity_002.npy")
            np.testing.assert_allclose(raw, 4.)
            saved = json.loads((Path(tmp) / "velocity.json").read_text())
            self.assertEqual(saved["rows"], rows)
            with (Path(tmp) / "velocity.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            strength, direction = plot_data(rows)
            self.assertEqual(len(strength), 6)
            self.assertEqual(len(direction), 1)
            self.assertTrue(all(Path(p).is_file() for p in collector.snapshot()["files"]))

    def test_nonfinite_velocity_is_flagged_and_json_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = VelocityDiagnostics(tmp)
            collector.record(event(1, torch.full((1, 3, 2, 2), float("nan"))))
            self.assertEqual(collector.rows[0]["finite_fraction"], 0.)
            self.assertIsNone(collector.rows[0]["velocity_rms"])
            self.assertIsNone(collector.rows[0]["update_rms"])
            collector.record(event(2, torch.ones((1, 3, 2, 2))))
            self.assertEqual(collector.heatmap_max, 1.)
            self.assertIsNone(collector.rows[1]["direction_cosine"])
            json.loads((Path(tmp) / "velocity.json").read_text())


class Transformer(torch.nn.Module):
    in_channels = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32

    def forward(self, latents, timesteps, embeds):
        # Conditional prediction = +2, unconditional = +1 in model-time convention.
        return ([torch.full_like(value, 2. if i == 0 else 1.) for i, value in enumerate(latents)],)


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, prompts, **kwargs):
        return SimpleNamespace(input_ids=torch.ones((len(prompts), 2), dtype=torch.long),
                               attention_mask=torch.ones((len(prompts), 2), dtype=torch.long))


class PipelineTests(unittest.TestCase):
    def test_skipped_zero_timestep_has_no_velocity(self):
        class ZeroScheduler(FlowMatchEulerDiscreteScheduler):
            def set_timesteps(self, *args, **kwargs):
                super().set_timesteps(*args, **kwargs)
                self.timesteps[-1] = 0.
                self.sigmas[-2] = 0.

        events = []
        encoder = lambda **kwargs: SimpleNamespace(hidden_states=[torch.ones((1, 2, 3))] * 3)
        generate(Transformer(), SimpleNamespace(), encoder, Tokenizer(), ZeroScheduler(),
                 "scene", height=16, width=16, num_inference_steps=2, output_type="latent",
                 callback_on_velocity=events.append)
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]["skipped"])
        self.assertTrue(events[1]["skipped"])
        self.assertNotIn("velocity", events[1])

    def test_callback_matches_actual_euler_update_and_cfg_sign(self):
        for guidance, expected in [(0., -2.), (2., -4.)]:
            scheduler = FlowMatchEulerDiscreteScheduler()
            events, after = [], []
            encoder = lambda **kwargs: SimpleNamespace(hidden_states=[torch.ones((1, 2, 3))] * 3)
            result = generate(Transformer(), SimpleNamespace(), encoder, Tokenizer(), scheduler,
                              "scene", height=16, width=16, num_inference_steps=2,
                              guidance_scale=guidance, cfg_truncation=1., output_type="latent",
                              callback_on_velocity=events.append,
                              callback_on_step_end=lambda i, t, z: after.append(z.clone()))
            self.assertEqual(len(events), 2)
            for i, update in enumerate(events):
                torch.testing.assert_close(update["velocity"], torch.full_like(update["velocity"], expected))
                dt = update["sigma_next"] - update["sigma"]
                self.assertLess(dt, 0)
                torch.testing.assert_close(after[i], update["latents"] + dt * update["velocity"])
                self.assertAlmostEqual(update["model_time"], 1 - update["timestep"] / 1000)
                if guidance:
                    self.assertEqual(update["conditional_velocity"].mean().item(), -2.)
                    self.assertEqual(update["unconditional_velocity"].mean().item(), -1.)
            torch.testing.assert_close(result, after[-1])


def analyzed_generate(**kwargs):
    velocity = torch.ones((1, 3, 4, 4))
    for i in range(kwargs["num_inference_steps"]):
        if kwargs.get("callback_on_velocity"):
            kwargs["callback_on_velocity"](event(i + 1, velocity * (i + 1)))
        kwargs["callback_on_step_end"](i, torch.tensor(900.), velocity)
    return [Image.new("RGB", (4, 4), "red")]


class IntegrationTests(unittest.TestCase):
    def test_stream_analysis_without_step_decoding_and_disabled_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings(), output_dir=Path(tmp) / "outputs", step_dir=Path(tmp) / "steps")
            service = MockService(settings)
            with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=analyzed_generate)}):
                updates = list(service.generate_stream(GenerationRequest("scene", steps=2, analyze_velocity=True)))
                plain = service.generate(GenerationRequest("scene", steps=2))
            self.assertEqual([len(u["diagnostics"]["rows"]) for u in updates], [1, 2, 2])
            self.assertFalse(settings.step_dir.exists())
            self.assertEqual(updates[-1]["steps"], [])
            metadata = json.loads(Path(updates[-1]["metadata"]).read_text())
            self.assertEqual(len(metadata["velocity_analysis"]["rows"]), 2)
            self.assertIsNone(plain["diagnostics"])
            self.assertFalse((Path(plain["image"]).parent / "velocity").exists())

    def test_gradio_handler_streams_all_analysis_outputs(self):
        from zimage_app.ui import build_app

        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings(), output_dir=Path(tmp) / "outputs", step_dir=Path(tmp) / "steps")
            demo = build_app(settings, MockService(settings))
            binding = next(f for f in demo.fns.values() if f.fn and f.fn.__name__ == "generate_one")
            with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=analyzed_generate)}):
                outputs = list(binding.fn("scene", "", 720, 1280, 2, 0, 42, True, True))
            self.assertEqual([len(update[7]) for update in outputs], [0, 1, 2, 2])
            self.assertEqual([len(update[1]) for update in outputs], [0, 1, 2, 2])
            for update in outputs:
                self.assertEqual(len(update), len(binding.outputs))
                for component, value in zip(binding.outputs, update):
                    component.postprocess(value)
            self.assertTrue(outputs[-1][0].endswith("final.png"))
            self.assertTrue(any(path.endswith("velocity.csv") for path in outputs[-1][2]))
            self.assertEqual(len(outputs[-1][8]), 2)
            with patch.dict(sys.modules, {"zimage": SimpleNamespace(generate=analyzed_generate)}):
                disabled = list(binding.fn("scene", "", 720, 1280, 2, 0, 42, False, False))
            self.assertIsNone(disabled[-1][4])
            self.assertTrue(disabled[-1][5].empty)
            self.assertEqual(disabled[-1][7], [])
            self.assertEqual(disabled[-1][8], [])


if __name__ == "__main__":
    unittest.main()
