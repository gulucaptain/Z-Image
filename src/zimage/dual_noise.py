"""Two-noise state transitions using the pipeline's post-CFG dz/dsigma."""
import math

import torch


class DualNoiseState:
    def __init__(self, config, noise_a, noise_b, same_seed=False):
        self.config = config
        self.a, self.b = noise_a, noise_b
        self.merged = False
        self.evaluations = 0
        if config.mode == "latent":
            self.initial = noise_a
        else:
            self.initial = self.blend(noise_a, noise_b)
            if config.mode == "initial" and config.normalize_initial and not same_seed:
                w = config.weight_b
                self.initial = self.initial / math.sqrt((1 - w)**2 + w**2)

    def blend(self, a, b):
        return (1 - self.config.weight_b) * a + self.config.weight_b * b

    def velocity(self, current, predict, dt, step):
        config = self.config
        merged_now = False
        jump_rms = 0.0
        if config.mode == "initial" or self.merged:
            prediction = predict(current)
            self.evaluations += 1
        else:
            prediction_a, prediction_b = predict(self.a), predict(self.b)
            va, vb = prediction_a["velocity"], prediction_b["velocity"]
            self.evaluations += 2
            if config.mode == "latent":
                self.a = self.a + dt * va
                self.b = self.b + dt * vb
                prediction = prediction_a
                if step == config.mix_step:
                    target = self.blend(self.a, self.b)
                    jump_rms = (target - self.a).square().mean().sqrt().item()
                    # Include the deliberate jump in the effective update diagnostic.
                    prediction = {**prediction_a, "velocity": (target - current) / dt,
                                  "conditional_velocity": None, "unconditional_velocity": None}
                    self.merged = merged_now = True
            else:
                velocity = self.blend(va, vb)
                if config.velocity_rule == "coupled":
                    self.a = self.a + dt * velocity
                    self.b = self.b + dt * velocity
                else:
                    self.a = self.a + dt * va
                    self.b = self.b + dt * vb
                prediction = {**prediction_a, "velocity": velocity,
                              "conditional_velocity": self.blend(prediction_a["conditional_velocity"], prediction_b["conditional_velocity"])
                              if prediction_a["conditional_velocity"] is not None else None,
                              "unconditional_velocity": self.blend(prediction_a["unconditional_velocity"], prediction_b["unconditional_velocity"])
                              if prediction_a["unconditional_velocity"] is not None else None}
        updated = current + dt * prediction["velocity"]
        trace = {"step": step, "mode": config.mode, "merged_now": merged_now,
                 "merge_jump_rms": jump_rms, "model_evaluations": self.evaluations,
                 "delta_sigma": float(dt.item()),
                 "output_latent_rms": updated.square().mean().sqrt().item(),
                 "effective_velocity_rms": prediction["velocity"].square().mean().sqrt().item()}
        if config.mode != "initial":
            trace["branch_distance_rms"] = (self.a - self.b).square().mean().sqrt().item()
        return prediction, trace
