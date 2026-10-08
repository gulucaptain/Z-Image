"""Verify experimental trajectories against explicit Euler reference updates."""
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.dual_noise import DualNoiseConfig
from zimage.dual_noise import DualNoiseState
from zimage.pipeline import generate
from zimage.scheduler import FlowMatchEulerDiscreteScheduler
from zimage_app.service import GenerationService, GenerationRequest
from zimage_app.settings import Settings
from zimage_app.ui import build_app
from zimage_app.diagnostics import plot_data
from test_app import MockVae
from test_velocity import Tokenizer


class NonlinearTransformer(torch.nn.Module):
    in_channels = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.calls = 0

    def forward(self, latents, timesteps, embeds):
        self.calls += 1
        return ([0.1 * z.square() + (0.2 if i == 0 else 0.1) for i, z in enumerate(latents)],)


def encoder(**kwargs):
    return SimpleNamespace(hidden_states=[torch.ones((1, 2, 3))] * 3)


def components():
    return dict(transformer=NonlinearTransformer(), vae=MockVae(), text_encoder=encoder,
                tokenizer=Tokenizer(), scheduler=FlowMatchEulerDiscreteScheduler())


def noise(seed):
    return torch.randn((1, 3, 2, 2), generator=torch.Generator().manual_seed(seed))


def velocity(z, guidance=0):
    return -0.1 * z.square() - (0.2 + 0.1 * guidance)


class DualNoiseTests(unittest.TestCase):
    def run_pipeline(self, config=None, seed=42, guidance=0):
        model = components()
        events, outputs, trace = [], [], []
        result = generate(**model, prompt="scene", height=16, width=16, num_inference_steps=4,
                          generator=torch.Generator().manual_seed(seed), guidance_scale=guidance,
                          cfg_truncation=1, output_type="latent", dual_noise=config,
                          callback_on_velocity=events.append, callback_on_dual_step=trace.append,
                          callback_on_step_end=lambda i, t, z: outputs.append(z.clone()))
        self.assertEqual(model["scheduler"]._step_index, 4)
        for event, after in zip(events, outputs):
            dt = event["sigma_next"] - event["sigma"]
            torch.testing.assert_close(after, event["latents"] + dt * event["velocity"])
        return result, outputs, trace, model["transformer"].calls

    def test_initial_mix_and_variance_compensation(self):
        for normalized in (False, True):
            config = DualNoiseConfig("initial", weight_b=0.3, normalize_initial=normalized)
            expected = 0.7 * noise(42) + 0.3 * noise(43)
            if normalized:
                expected /= math.sqrt(0.7**2 + 0.3**2)
            for _ in range(4):
                expected = expected - 0.25 * velocity(expected)
            result, _, _, calls = self.run_pipeline(config)
            torch.testing.assert_close(result, expected)
            self.assertEqual(calls, 4)
        generator = torch.Generator().manual_seed(7)
        a, b = [torch.randn((100000,), generator=generator) for _ in range(2)]
        mixed = DualNoiseState(DualNoiseConfig("initial"), a, b).initial
        self.assertAlmostEqual(mixed.var().item(), 1, delta=0.02)

    def test_latent_mix_occurs_after_k_updates_then_single_trajectory(self):
        a, b = noise(42), noise(43)
        for _ in range(2):
            a, b = a - 0.25 * velocity(a), b - 0.25 * velocity(b)
        mixed = 0.6 * a + 0.4 * b
        expected = mixed
        for _ in range(2):
            expected = expected - 0.25 * velocity(expected)
        result, outputs, trace, calls = self.run_pipeline(DualNoiseConfig("latent", weight_b=0.4, mix_step=2))
        torch.testing.assert_close(outputs[1], mixed)
        torch.testing.assert_close(result, expected)
        self.assertEqual([row["merged_now"] for row in trace], [False, True, False, False])
        self.assertGreater(trace[1]["merge_jump_rms"], 0)
        self.assertEqual(calls, 6)

    def test_velocity_rules_and_cfg_match_manual_reference(self):
        results = []
        for rule in ("independent", "coupled"):
            a, b = noise(42), noise(43)
            expected = 0.6 * a + 0.4 * b
            for _ in range(4):
                va, vb = velocity(a, 2), velocity(b, 2)
                combined = 0.6 * va + 0.4 * vb
                expected = expected - 0.25 * combined
                a = a - 0.25 * (combined if rule == "coupled" else va)
                b = b - 0.25 * (combined if rule == "coupled" else vb)
            result, _, trace, calls = self.run_pipeline(
                DualNoiseConfig("velocity", weight_b=0.4, velocity_rule=rule), guidance=2,
            )
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(result, 0.6 * a + 0.4 * b)
            self.assertEqual(calls, 8)
            if rule == "coupled":
                self.assertAlmostEqual(trace[0]["branch_distance_rms"], trace[-1]["branch_distance_rms"], places=6)
            results.append(result)
        self.assertFalse(torch.allclose(*results))

    def test_weight_endpoints_and_identical_seeds_reproduce_baselines(self):
        baseline_a = self.run_pipeline()[0]
        baseline_b = self.run_pipeline(seed=43)[0]
        for mode in ("initial", "latent", "velocity"):
            for weight, baseline in ((0., baseline_a), (1., baseline_b)):
                result = self.run_pipeline(DualNoiseConfig(mode, weight_b=weight, mix_step=2))[0]
                torch.testing.assert_close(result, baseline)
            result = self.run_pipeline(DualNoiseConfig(mode, seed_b=42, mix_step=2))[0]
            torch.testing.assert_close(result, baseline_a)

    def test_invalid_configuration(self):
        for config in (DualNoiseConfig("bad"), DualNoiseConfig(seed_b=-1),
                       DualNoiseConfig(weight_b=float("nan")), DualNoiseConfig(weight_b=1.1),
                       DualNoiseConfig("latent", mix_step=0), DualNoiseConfig("latent", mix_step=5),
                       DualNoiseConfig(velocity_rule="bad")):
            with self.assertRaises(ValueError):
                config.validate(4)

    def test_legacy_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = replace(Settings(), output_dir=root / "outputs", step_dir=root / "steps")
            service = GenerationService(settings)
            service.components, service.device = components(), "cpu"
            result = service.generate(GenerationRequest("scene", height=16, width=16, steps=4),
                                      kind="dual_noise", dual_noise=DualNoiseConfig("latent", mix_step=2))
            data = json.loads(Path(result["metadata"]).read_text())
            self.assertEqual(data["dual_noise"]["mode"], "latent")
            self.assertTrue(data["dual_noise_trace"][1]["merged_now"])


if __name__ == "__main__":
    unittest.main()
