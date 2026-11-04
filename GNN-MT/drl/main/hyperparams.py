"""PPO and network hyperparameters: defaults, parsing and checks."""

from dataclasses import dataclass, fields
from typing import Optional, Union


def as_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _parse_literal(value):
    """Command-line values arrive as numbers or strings; these strings mean Python constants."""
    return {"True": True, "False": False, "None": None}.get(value, value) if isinstance(value, str) else value


@dataclass
class Hyperparams:
    encoder: str = "shared"

    # Rollout. None means "not set": n_steps="auto" then uses the longest horizon.
    timesteps_per_batch: Optional[int] = None
    n_steps: Union[int, str] = "auto"

    # Update
    mini_batch_size: int = 100
    n_updates_per_iteration: int = 4
    lr: float = 3e-4
    gamma: float = 0.95
    clip: float = 0.2
    gae_lambda: float = 0.95
    # Once rewards are normalized to about unit scale, 0.01 is a larger share of the loss.
    entropy_coef: float = 0.01
    entropy_coef_rule: Optional[float] = None  # None: entropy_coef
    entropy_coef_node: Optional[float] = None  # None: entropy_coef
    entropy_coef_val: Optional[float] = None  # None: entropy_coef
    entropy_normalize: bool = True
    target_kl: float = 0.02

    # Network
    rnn_type: str = "gru"
    masking: bool = True
    encoder_residual: bool = True
    node_global_context: bool = True
    slot_embedding: bool = True
    value_head: str = "bins"
    value_bins: int = 11
    value_ordinal: bool = True
    node_decoder: str = "rnn"
    attn_heads: int = 8
    attn_tanh_clip: float = 10.0
    graph_encoder: str = "sage"
    encoder_heads: int = 4
    encoder_layers: int = 3
    encoder_ff: bool = False
    encoder_reverse_edges: bool = False

    # Reward normalization and time feature
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    rew_norm_freeze_after: int = -1
    time_feature: bool = True
    time_scale: Optional[float] = None

    # Convergence: patience counts ticks, one tick per es_tick_episodes finished episodes.
    es_patience: int = 10
    es_min_delta: float = 0.01
    es_min_delta_rel: float = 0.01
    es_window: int = 5
    es_warmup_episodes: int = 5
    es_tick_episodes: Optional[int] = None  # None: the number of envs
    es_min_timesteps: int = 100_000

    @classmethod
    def from_dict(cls, values):
        """Build from key=value pairs. An unknown key stops the run instead of being ignored."""
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown hyperparameters {unknown}. Known: {sorted(known)}")
        return cls(**{key: _parse_literal(value) for key, value in values.items()})

    def __post_init__(self):
        if self.entropy_coef_rule is None:
            self.entropy_coef_rule = self.entropy_coef
        if self.entropy_coef_node is None:
            self.entropy_coef_node = self.entropy_coef
        if self.entropy_coef_val is None:
            self.entropy_coef_val = self.entropy_coef
        self.entropy_normalize = as_bool(self.entropy_normalize, True)
        self.masking = as_bool(self.masking, True)
        self.encoder_residual = as_bool(self.encoder_residual, True)
        self.node_global_context = as_bool(self.node_global_context, True)
        self.slot_embedding = as_bool(self.slot_embedding, True)
        self.value_ordinal = as_bool(self.value_ordinal, True)
        self.encoder_ff = as_bool(self.encoder_ff, False)
        self.encoder_reverse_edges = as_bool(self.encoder_reverse_edges, False)
        self.normalize_rewards = as_bool(self.normalize_rewards, True)
        self.time_feature = as_bool(self.time_feature, True)

        self._check_choice("encoder", ("shared", "separate"))
        self._check_choice("value_head", ("bins", "gaussian"))
        self._check_choice("node_decoder", ("rnn", "attention"))
        self._check_choice("graph_encoder", ("sage", "gatv2", "transformer"))

        self.value_bins = int(self.value_bins)
        self.attn_heads = int(self.attn_heads)
        self.attn_tanh_clip = float(self.attn_tanh_clip)
        self.encoder_heads = int(self.encoder_heads)
        self.encoder_layers = int(self.encoder_layers)
        self.target_kl = 0.0 if self.target_kl is None else float(self.target_kl)
        self.reward_clip = float(self.reward_clip)
        self.rew_norm_freeze_after = int(self.rew_norm_freeze_after)
        self.es_window = int(self.es_window)
        self.es_warmup_episodes = int(self.es_warmup_episodes)
        self.es_min_delta_rel = float(self.es_min_delta_rel)
        if isinstance(self.n_steps, str) and self.n_steps.strip().lower() == "auto":
            self.n_steps = "auto"
        else:
            self.n_steps = int(self.n_steps)
        if self.timesteps_per_batch is not None:
            self.timesteps_per_batch = int(self.timesteps_per_batch)
        if self.time_scale is not None:
            self.time_scale = float(self.time_scale)
        if self.es_tick_episodes is not None:
            self.es_tick_episodes = int(self.es_tick_episodes)

    def _check_choice(self, name, choices):
        value = str(getattr(self, name))
        if value not in choices:
            raise ValueError(f"Unsupported {name} '{value}'. Use one of {choices}.")
        setattr(self, name, value)
