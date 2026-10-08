"""Inference baseline for (X0,Y0) -> (Qhat,Y0) -> (Qhat,Qhat).

Stage one reuses an existing single-image velocity field with time rescaling.
Stage two uses the markdown's exact copying field; no joint model is trained.
"""
import torch


def build_staged_fm_targets(q, noise_x, noise_y, times, condition=None, split=0.5):
    """Construct joint states and masked training velocities from the proposal.

    condition=None uses (Q,Q); condition=C uses the complementary (C,Q) path.
    This constructs training batches, not a trained conditional velocity model.
    """
    c = q if condition is None else condition
    if not 0 < split < 1:
        raise ValueError("Stage split must be in (0,1).")
    if noise_x.shape != c.shape or noise_y.shape != q.shape or c.shape[0] != q.shape[0]:
        raise ValueError("X noise must match C, Y noise must match Q, and batch sizes must match.")
    times = torch.as_tensor(times, device=q.device, dtype=q.dtype)
    if times.ndim == 0:
        times = times.expand(q.shape[0])
    if times.shape != (q.shape[0],) or not torch.isfinite(times).all() or not ((times >= 0) & (times <= 1)).all():
        raise ValueError("Provide one time in [0,1] per sample.")
    tx = times.reshape((-1,) + (1,) * (c.ndim - 1))
    ty = times.reshape((-1,) + (1,) * (q.ndim - 1))
    active_x, active_y = tx < split, ty >= split
    x = torch.where(active_x, (1 - tx / split) * noise_x + tx / split * c, c)
    y = torch.where(active_y, (1 - (ty - split) / (1 - split)) * noise_y + (ty - split) / (1 - split) * q, noise_y)
    ux = torch.where(active_x, (c - noise_x) / split, torch.zeros_like(c))
    uy = torch.where(active_y, (q - noise_y) / (1 - split), torch.zeros_like(q))
    return (x, y), (ux, uy)


@torch.no_grad()
def generate_staged(base_generate, config, **kwargs):
    total = kwargs.pop("num_inference_steps")
    first_steps, second_steps = config.stage_steps(total)
    split, epsilon = config.stage_split, config.terminal_epsilon
    on_step = kwargs.pop("callback_on_step_end", None)
    on_velocity = kwargs.pop("callback_on_velocity", None)
    on_trace = kwargs.pop("callback_on_dual_step", None)
    on_state = kwargs.pop("callback_on_dual_state", None)
    output_type = kwargs.pop("output_type", "pil")
    y = None
    last_event = None
    evaluations = 0

    def rms(value):
        return value.float().square().mean().sqrt().item()

    def initial_y(x):
        nonlocal y
        if y is None:
            generator = torch.Generator(x.device).manual_seed(config.seed_b)
            y = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32)

    def first_velocity(event):
        nonlocal last_event, evaluations
        last_event = event
        if not event["skipped"]:
            initial_y(event["latents"])
            evaluations += 1
        if on_velocity:
            on_velocity(event)

    def first_step(index, timestep, x):
        initial_y(x)
        event = last_event
        if event and not event["skipped"]:
            t_before = split * (1 - event["sigma"])
            t_after = split * (1 - event["sigma_next"])
            speed = -event["velocity"] / split
            trace = {"step": index + 1, "stage": 1, "active": "X", "frozen": "Y",
                     "joint_time": t_before, "joint_time_next": t_after,
                     "joint_velocity_rms": rms(speed), "model_evaluations": evaluations,
                     "x_rms": rms(x), "y_rms": rms(y), "xy_distance_rms": rms(x - y),
                     "inactive_update_rms": 0.0, "skipped": False}
        else:
            trace = {"step": index + 1, "stage": 1, "active": "X", "frozen": "Y",
                     "skipped": True, "model_evaluations": evaluations, "inactive_update_rms": 0.0}
        if on_trace:
            on_trace(trace)
        if on_state:
            on_state(index, x, y, trace)
        if on_step:
            on_step(index, timestep, x)

    x = base_generate(**kwargs, num_inference_steps=first_steps, output_type="latent",
                      callback_on_velocity=first_velocity, callback_on_step_end=first_step)
    initial_y(x)
    end = 1 - epsilon
    for index in range(second_steps):
        t = split + (end - split) * index / second_steps
        t_next = split + (end - split) * (index + 1) / second_steps
        before = y
        velocity_t = (x - y) / (1 - t)
        y = y + (t_next - t) * velocity_t
        # Exact terminal copy without ever evaluating the singular field at t=1.
        if index == second_steps - 1 and epsilon == 0:
            y = x.clone()
        step_index = first_steps + index
        sigma = (1 - t) / (1 - split)
        sigma_next = (1 - t_next) / (1 - split)
        velocity_sigma = -(1 - split) * velocity_t
        trace = {"step": step_index + 1, "stage": 2, "active": "Y", "frozen": "X",
                 "joint_time": t, "joint_time_next": t_next,
                 "joint_velocity_rms": rms(velocity_t), "model_evaluations": evaluations,
                 "x_rms": rms(x), "y_rms": rms(y), "xy_distance_rms": rms(x - y),
                 "inactive_update_rms": 0.0, "skipped": False,
                 "terminal_copy": index == second_steps - 1 and epsilon == 0}
        if on_trace:
            on_trace(trace)
        if on_velocity:
            on_velocity({"step": step_index + 1, "timestep": sigma * 1000,
                         "model_time": 1 - sigma, "sigma": sigma, "sigma_next": sigma_next,
                         "latents": before, "velocity": velocity_sigma,
                         "conditional_velocity": None, "unconditional_velocity": None,
                         "guidance_scale": 0.0, "skipped": False})
        if on_state:
            on_state(step_index, x, y, trace)
        if on_step:
            on_step(step_index, torch.tensor(sigma * 1000, device=y.device), y)

    if output_type == "latent":
        return y
    vae = kwargs["vae"]
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    decoded = vae.decode(y.to(vae.dtype) / vae.config.scaling_factor + shift, return_dict=False)[0]
    if output_type == "pil":
        from PIL import Image
        pixels = ((decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        return [Image.fromarray(item) for item in pixels]
    return decoded
