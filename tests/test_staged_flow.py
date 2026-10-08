"""Check stage masks, analytical copying, time rescaling and UI artifacts."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.dual_noise import DualNoiseConfig
from zimage.pipeline import generate
from zimage.staged_flow import build_staged_fm_targets
from zimage_app.diagnostics import plot_data
from zimage_app.service import GenerationRequest, GenerationService
from zimage_app.settings import Settings
from zimage_app.ui import build_app
from test_dual_noise import components, noise


class StagedFlowTests(unittest.TestCase):
    def sample(self, config, total=7, seed=42):
        model = components()
        states, trace, updates = [], [], []
        result = generate(**model, prompt="scene", height=16, width=16,
                          num_inference_steps=total, generator=torch.Generator().manual_seed(seed),
                          output_type="latent", dual_noise=config,
                          callback_on_dual_state=lambda i, x, y, row: states.append((x.clone(), y.clone())),
                          callback_on_dual_step=trace.append,
                          callback_on_step_end=lambda i, t, z: updates.append((i, z.clone())))
        return result, states, trace, updates, model

    def test_freeze_stages_and_seed_y_terminal_invariance(self):
        reference_model = components()
        baseline = generate(**reference_model, prompt="scene", height=16, width=16,
                            num_inference_steps=3, output_type="latent",
                            generator=torch.Generator().manual_seed(42))
        for seed_y in (43, 44, 45):
            result, states, trace, updates, model = self.sample(DualNoiseConfig(seed_b=seed_y, stage1_steps=3))
            self.assertEqual(len(states), 7)
            self.assertEqual([i for i, _ in updates], list(range(7)))
            for _, y in states[:3]:
                torch.testing.assert_close(y, noise(seed_y), rtol=0, atol=0)
            for x, _ in states[3:]:
                torch.testing.assert_close(x, states[2][0], rtol=0, atol=0)
            torch.testing.assert_close(result, baseline, rtol=0, atol=0)
            torch.testing.assert_close(states[-1][0], states[-1][1], rtol=0, atol=0)
            self.assertEqual(model["transformer"].calls, 3)
            self.assertEqual([r["stage"] for r in trace], [1] * 3 + [2] * 4)
            self.assertEqual(trace[2]["joint_time_next"], 0.5)
            self.assertEqual(trace[-1]["joint_time_next"], 1)
            self.assertEqual(trace[-1]["xy_distance_rms"], 0)
            self.assertTrue(all(r["inactive_update_rms"] == 0 for r in trace))
            self.assertTrue(all(r["model_evaluations"] == 3 for r in trace[3:]))

    def test_epsilon_matches_analytic_residual_and_has_finite_velocities(self):
        config = DualNoiseConfig(stage1_steps=3, stage_split=0.3, terminal_epsilon=0.1)
        result, states, trace, _, _ = self.sample(config, total=8)
        x = states[2][0]
        expected = x + 0.1 / 0.7 * (noise(43) - x)
        torch.testing.assert_close(result, expected)
        self.assertAlmostEqual(trace[-1]["joint_time_next"], 0.9)
        self.assertFalse(trace[-1]["terminal_copy"])
        self.assertTrue(all(torch.isfinite(torch.tensor(r["joint_velocity_rms"])) for r in trace))

    def test_training_path_boundaries_masks_and_factor_two(self):
        q = torch.full((5, 1), 2.)
        x0, y0 = torch.full_like(q, -1.), torch.full_like(q, -3.)
        (x, y), (ux, uy) = build_staged_fm_targets(q, x0, y0, torch.tensor([0., 0.25, 0.5, 0.75, 1.]))
        torch.testing.assert_close(x.flatten(), torch.tensor([-1., 0.5, 2., 2., 2.]))
        torch.testing.assert_close(y.flatten(), torch.tensor([-3., -3., -3., -0.5, 2.]))
        torch.testing.assert_close(ux.flatten(), torch.tensor([6., 6., 0., 0., 0.]))
        torch.testing.assert_close(uy.flatten(), torch.tensor([0., 0., 10., 10., 10.]))

    def test_complementary_training_targets_support_distinct_condition_shape(self):
        q, c = torch.ones((2, 3)), torch.full((2, 1), 0.2)
        (x, y), (ux, uy) = build_staged_fm_targets(q, torch.zeros_like(c), torch.zeros_like(q),
                                                 torch.tensor([0.2, 0.7]), condition=c, split=0.4)
        torch.testing.assert_close(x, torch.tensor([[0.1], [0.2]]))
        torch.testing.assert_close(y[0], torch.zeros(3))
        torch.testing.assert_close(y[1], torch.full((3,), 0.5))
        torch.testing.assert_close(ux, torch.tensor([[0.5], [0.]]))
        torch.testing.assert_close(uy[0], torch.zeros(3))
        torch.testing.assert_close(uy[1], torch.full((3,), 1 / 0.6))

    def test_invalid_stage_options_and_missing_conditional_model(self):
        for config, steps in [(DualNoiseConfig(), 1), (DualNoiseConfig(stage1_steps=4), 4),
                              (DualNoiseConfig(stage1_steps=0), 4), (DualNoiseConfig(stage_split=1), 4),
                              (DualNoiseConfig(terminal_epsilon=0.5), 4)]:
            with self.assertRaises(ValueError):
                config.validate(steps)
        with self.assertRaisesRegex(ValueError, "条件速度模型"):
            DualNoiseConfig("complementary").validate(8)

    def test_gradio_joint_preview_video_and_stage_metadata(self):
        plot_data([])
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings(), output_dir=Path(tmp) / "outputs", step_dir=Path(tmp) / "steps")
            service = GenerationService(settings)
            service.components, service.device = components(), "cpu"
            app = build_app(settings, service)
            binding = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == "dual")
            updates = list(binding.fn("scene", "", 43, 2, 0.5, 0, True, 16, 16, 4, 0, 42, True))
            for update in updates:
                self.assertEqual(len(update), len(binding.outputs))
                for component, value in zip(binding.outputs, update):
                    component.postprocess(value)
            final = updates[-1]
            self.assertTrue(final[3].endswith("steps.mp4"))
            self.assertTrue(final[1].endswith("stage1_X.png"))
            self.assertEqual([row[1] for row in final[6]], [1, 1, 2, 2])
            metadata = json.loads(Path(next(p for p in final[4] if p.endswith("metadata.json"))).read_text())
            self.assertFalse(metadata["joint_model_trained"])
            self.assertEqual(metadata["stage1_steps"], 2)
            self.assertEqual(metadata["stage2_steps"], 2)
            self.assertEqual(Image.open(metadata["stage1_image"]).tobytes(), Image.open(metadata["final_image"]).tobytes())
            with Image.open(final[2][0][0]) as preview:
                self.assertEqual(preview.width, 4)  # Two 2px latent decodes side by side.

    def test_without_step_saving_keeps_stage_endpoint_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(Settings(), output_dir=Path(tmp) / "outputs", step_dir=Path(tmp) / "steps")
            service = GenerationService(settings)
            service.components, service.device = components(), "cpu"
            result = service.generate(GenerationRequest("scene", height=16, width=16, steps=4),
                                      dual_noise=DualNoiseConfig(), kind="dual_noise")
            self.assertEqual(result["steps"], [])
            self.assertTrue(Path(result["stage1_image"]).is_file())
            self.assertTrue(Path(result["joint_final_image"]).is_file())
            self.assertFalse(settings.step_dir.exists())


if __name__ == "__main__":
    unittest.main()
