"""Remaining-budget feature and per-environment horizon lookup."""

import copy
import warnings

import torch

TIME_FEAT_DIM = 2


def attach_time_feature(data, steps_taken, horizon, time_scale):
    """Return a shallow copy of ``data`` with a ``[1, 2]`` remaining-budget feature.

    The columns are ``(horizon - steps_taken) / horizon`` and
    ``(horizon - steps_taken) / time_scale``. The input is not mutated: staged
    observations from earlier steps share it.
    """
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    scale = float(time_scale)
    if scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")
    remaining = horizon - int(steps_taken)
    featured = copy.copy(data)
    featured.time_feat = torch.tensor(
        [[remaining / horizon, remaining / scale]],
        dtype=torch.float32,
    )
    return featured


def resolve_max_steps(request, num_envs):
    """Return one horizon per environment.

    ``getEnvMaxSteps`` is absent on older jars and then answers ``None``. In that
    case every environment uses ``getMaxSteps`` (the longest horizon).
    """
    raw = request("getEnvMaxSteps")
    if raw is None:
        warnings.warn(
            "getEnvMaxSteps is unavailable; using getMaxSteps for every environment. "
            "Rebuild the environment jar to expose per-environment horizons.",
            stacklevel=2,
        )
        fallback = int(request("getMaxSteps"))
        return [fallback] * num_envs

    steps = [int(value) for value in raw]
    if len(steps) != num_envs:
        raise ValueError(
            f"getEnvMaxSteps returned {len(steps)} horizons for {num_envs} environments"
        )
    if any(step <= 0 for step in steps):
        raise ValueError(f"horizons must be positive, got {steps}")
    return steps


def fetch_instance_names(request, num_envs):
    """Return the model name bound to each environment, or ``env-{i}`` if unknown."""
    names = []
    for env_index in range(num_envs):
        name = request("getEnvModelName", env_index)
        if name is None:
            names.append(f"env-{env_index}")
        else:
            names.append(str(name))
    return names


def instance_index_by_name(names):
    """Map names to dense instance ids, preserving first-seen order."""
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    instance_ids = [unique_names.index(name) for name in names]
    return instance_ids, unique_names


def warn_horizon_mismatch(already_warned, env_index, steps_taken, horizon):
    """Warn once per environment when a finished episode length differs from its horizon."""
    if steps_taken == horizon or env_index in already_warned:
        return
    already_warned.add(env_index)
    warnings.warn(
        f"Environment {env_index} ended after {steps_taken} stored steps; "
        f"horizon is {horizon}.",
        stacklevel=2,
    )
