"""Statistics of the effective velocity along one latent sampling trajectory."""
import csv
import json
import math
from pathlib import Path


FIELDS = ["step", "skipped", "timestep", "model_time", "sigma", "sigma_next", "delta_sigma",
          "velocity_rms", "velocity_mean", "velocity_std", "abs_p50", "abs_p95", "abs_p99",
          "latent_rms", "update_rms", "relative_update", "direction_cosine",
          "conditional_rms", "unconditional_rms", "guidance_scale", "finite_fraction",
          "heatmap_saturated_fraction"]


class VelocityDiagnostics:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.maps = []
        self.previous = None
        self.heatmap_max = None
        self.files = []

    def record(self, event):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw

        row = dict.fromkeys(FIELDS)
        row.update(step=event["step"], timestep=event["timestep"], skipped=event["skipped"])
        if not event["skipped"]:
            # Keep CPU copies only; never retain the GPU tensors from a callback.
            velocity = event["velocity"].detach().float().cpu()
            latent = event["latents"].detach().float().cpu()
            finite = torch.isfinite(velocity)
            row["finite_fraction"] = finite.float().mean().item()
            dt = event["sigma_next"] - event["sigma"]
            row.update(model_time=event["model_time"], sigma=event["sigma"],
                       sigma_next=event["sigma_next"], delta_sigma=dt,
                       guidance_scale=event["guidance_scale"])
            rms = lambda tensor: tensor.square().mean().sqrt().item()
            if finite.all():
                magnitude = velocity.abs().flatten()
                quantiles = torch.quantile(magnitude, torch.tensor([0.5, 0.95, 0.99]))
                row.update(velocity_rms=rms(velocity), velocity_mean=velocity.mean().item(),
                           velocity_std=velocity.std(unbiased=False).item(),
                           abs_p50=quantiles[0].item(), abs_p95=quantiles[1].item(), abs_p99=quantiles[2].item())
                norm = torch.linalg.vector_norm(velocity)
                if self.previous is not None:
                    denominator = norm * torch.linalg.vector_norm(self.previous)
                    if denominator > 0:
                        row["direction_cosine"] = (torch.sum(velocity * self.previous) / denominator).clamp(-1, 1).item()
                self.previous = velocity
            else:
                self.previous = None
            row["latent_rms"] = rms(latent)
            row["update_rms"] = rms(dt * velocity)
            if row["latent_rms"] > 0:
                row["relative_update"] = row["update_rms"] / row["latent_rms"]
            for name in ("conditional", "unconditional"):
                value = event.get(name + "_velocity")
                if value is not None:
                    row[name + "_rms"] = rms(value.detach().float().cpu())

            # Batch size is one in the Gradio single-image workflow.
            heat = velocity[0].square().mean(dim=0).sqrt().numpy()
            finite_heat = heat[np.isfinite(heat)]
            if self.heatmap_max is None and finite_heat.size:
                self.heatmap_max = max(float(finite_heat.max()), 1e-12)
            upper = self.heatmap_max or 1.0
            row["heatmap_saturated_fraction"] = float(np.mean(np.isfinite(heat) & (heat > upper)))
            normalized = np.clip(np.nan_to_num(heat / upper, nan=0, posinf=1, neginf=0), 0, 1)
            # Fixed sequential dark-blue -> teal -> yellow scale, shared by all steps.
            anchors = np.array([[24, 32, 72], [20, 112, 128], [96, 182, 112], [250, 224, 74]])
            colors = np.stack([np.interp(normalized, np.linspace(0, 1, len(anchors)), anchors[:, c])
                               for c in range(3)], axis=-1).astype(np.uint8)
            colors[~np.isfinite(heat)] = [255, 0, 255]
            # Preserve latent aspect ratio and include the scale in the exported image.
            height = max(1, round(480 * heat.shape[0] / heat.shape[1]))
            preview = Image.fromarray(colors).resize((480, height), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (480, height + 44), "white")
            canvas.paste(preview, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, height + 5), f"Step {event['step']} | channel RMS | fixed scale: 0 .. {upper:.5g}", fill="black")
            draw.text((8, height + 23), f"Above scale: {row['heatmap_saturated_fraction']:.1%} | nonfinite: magenta", fill="black")
            png = self.directory / f"velocity_{event['step']:03d}.png"
            raw = png.with_suffix(".npy")
            canvas.save(png)
            np.save(raw, heat)
            self.maps.append((str(png), f"Step {event['step']}"))
            self.files.extend([str(png), str(raw)])

        # JSON null denotes undefined / nonfinite values, never silently a zero.
        row = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in row.items()}
        self.rows.append(row)
        self.persist()

    def persist(self):
        description = {"velocity_convention": "dz/dsigma, after sign conversion and CFG; sigma decreases",
                       "heatmap": "sqrt(mean_channels(velocity**2)), first sample, latent grid",
                       "heatmap_max": self.heatmap_max,
                       "heatmap_scale": "fixed to first finite step maximum; larger values saturate",
                       "direction_cosine": "adjacent evaluated velocities along this trajectory; zero norm is undefined",
                       "rows": self.rows}
        json_path = self.directory / "velocity.json"
        csv_path = self.directory / "velocity.csv"
        for path, content in [(json_path, json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))]:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        temporary = csv_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(csv_path)

    def snapshot(self):
        return {"rows": [dict(row) for row in self.rows], "maps": list(self.maps),
                "files": [str(self.directory / "velocity.csv"), str(self.directory / "velocity.json"), *self.files]}


def plot_data(rows):
    """Long-form tables for Gradio's interactive native plots."""
    import pandas as pd

    strength, direction = [], []
    for row in rows:
        for field, label in [("velocity_rms", "速度 RMS"), ("update_rms", "更新 RMS"),
                             ("conditional_rms", "条件预测 RMS"), ("unconditional_rms", "无条件预测 RMS")]:
            if row.get(field) is not None:
                strength.append({"step": row["step"], "value": row[field], "metric": label})
        if row.get("direction_cosine") is not None:
            direction.append({"step": row["step"], "cosine": row["direction_cosine"]})
    return (pd.DataFrame(strength, columns=["step", "value", "metric"]),
            pd.DataFrame(direction, columns=["step", "cosine"]))
