"""The actor-critic network: how it is built or loaded, and how to query it."""

import json
import math
import os
from collections import namedtuple

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch

from drl.main.hyperparams import as_bool
from drl.neural_model.gnn_predictor import (
    LOGIT_MASK,
    SeparateEncoderAC,
    SharedEncoderAC,
    disable_relation_masks,
)

TFAction = namedtuple("TForceAction", ["action", "nodes", "nodes_dim_mask", "values"])

eps_log_clip = 1e-6
MODEL_CONFIG_FILE = "model_config.json"
ENCODERS = {"shared": SharedEncoderAC, "separate": SeparateEncoderAC}


def model_config(hp, hidden_dim, time_feat_dim, time_scale):
    """What it takes to rebuild the network, stored next to every checkpoint."""
    return {
        "encoder_residual": hp.encoder_residual,
        "node_global_context": hp.node_global_context,
        "slot_embedding": hp.slot_embedding,
        "rnn_type": hp.rnn_type,
        "hidden_dim": hidden_dim,
        "encoder": hp.encoder,
        "entropy_coef": hp.entropy_coef,
        "entropy_coef_rule": hp.entropy_coef_rule,
        "entropy_coef_node": hp.entropy_coef_node,
        "entropy_coef_val": hp.entropy_coef_val,
        "entropy_normalize": hp.entropy_normalize,
        "target_kl": hp.target_kl,
        "value_head": hp.value_head,
        "value_bins": int(hp.value_bins),
        "value_ordinal": hp.value_ordinal,
        "node_decoder": hp.node_decoder,
        "attn_heads": int(hp.attn_heads),
        "attn_tanh_clip": float(hp.attn_tanh_clip),
        "graph_encoder": hp.graph_encoder,
        "encoder_heads": int(hp.encoder_heads),
        "encoder_layers": int(hp.encoder_layers),
        "encoder_ff": hp.encoder_ff,
        "encoder_reverse_edges": hp.encoder_reverse_edges,
        "time_feat_dim": time_feat_dim,
        "time_scale": time_scale,
    }


def read_model_config(directory):
    with open(os.path.join(directory, MODEL_CONFIG_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def write_model_config(directory, config):
    with open(os.path.join(directory, MODEL_CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def factor_stats(
    action_dist,
    action,
    action_mask=None,
    nodes_dist=None,
    nodes=None,
    seq_len_mask=None,
    values_dist_tt=None,
    values_dist=None,
    values_raw=None,
    logp_only=False,
):
    """Per-factor log-probs and, unless ``logp_only``, entropies.

    The rollout asks for log-probs only and returns before the entropy,
    valid-count and normalization work, including the action-mask copy.
    """
    B = action.shape[0]
    device = action.device
    zeros = torch.zeros(B, device=device, dtype=torch.float32)

    skip_rule = (
        len(action_dist.param_shape) == 2 and action_dist.param_shape[1] == 1
    )
    if skip_rule:
        logp_rule = zeros
    else:
        logp_rule = action_dist.log_prob(action)

    logp_node = zeros.clone()
    ent_node = zeros.clone()
    ent_node_norm = zeros.clone()
    n_valid_node = zeros.clone()
    if nodes_dist is not None and nodes is not None:
        if seq_len_mask is None:
            seq_len_mask = torch.ones(
                nodes.shape[0], nodes.shape[-1], dtype=torch.bool, device=device
            )
        else:
            seq_len_mask = seq_len_mask.to(device=device)
            if seq_len_mask.dtype != torch.bool:
                seq_len_mask = seq_len_mask.bool()
        L = min(nodes.shape[-1], seq_len_mask.shape[-1], nodes_dist.logits.shape[1])
        nodes = nodes[..., :L]
        seq_len_mask = seq_len_mask[..., :L]
        valid_mask = (nodes >= 0) & seq_len_mask
        nodes_safe = torch.clamp(nodes, min=0)
        logits = nodes_dist.logits[:, :L]
        node_dist_l = Categorical(logits=logits)
        node_log_probs = node_dist_l.log_prob(nodes_safe)
        logp_node = (node_log_probs * valid_mask).sum(dim=1)
        if not logp_only:
            slot_ent = node_dist_l.entropy()
            seq_f = seq_len_mask.float()
            ent_node = (slot_ent * seq_f).sum(dim=1)
            n_valid_t = (logits > (LOGIT_MASK / 2)).sum(dim=-1).float()
            slot_ent_norm = slot_ent / torch.log(n_valid_t.clamp(min=2.0))
            n_slots = seq_f.sum(dim=1).clamp(min=1.0)
            ent_node_norm = (slot_ent_norm * seq_f).sum(dim=1) / n_slots
            n_valid_node = (n_valid_t * seq_f).sum(dim=1) / n_slots

    logp_val = zeros.clone()
    if values_dist_tt is not None and values_raw is not None:
        if isinstance(values_dist_tt, Categorical):
            idx = values_raw.long()
            logits = values_dist_tt.logits
            if idx.dim() == 1:
                idx = idx.unsqueeze(-1)
            L = min(idx.shape[-1], logits.shape[1])
            idx = idx[..., :L]
            val_dist = Categorical(logits=logits[:, :L])
            logp_val = val_dist.log_prob(idx).sum(dim=1)
        else:
            values_raw = values_raw.clamp(
                min=-1 + eps_log_clip, max=1 - eps_log_clip
            )
            logp_val = values_dist_tt.log_prob(values_raw).sum(dim=1)

    logp = torch.stack([logp_rule, logp_node, logp_val], dim=-1)
    if logp_only:
        return {"logp": logp}

    if skip_rule:
        ent_rule = zeros
        n_valid_rule = torch.ones(B, device=device, dtype=torch.float32)
    else:
        ent_rule = action_dist.entropy()
        if action_mask is not None:
            action_mask = action_mask.to(device=device)
            n_valid_rule = action_mask.reshape(B, -1).sum(dim=-1).float()
        else:
            n_valid_rule = torch.full(
                (B,),
                float(action_dist.logits.shape[-1]),
                device=device,
                dtype=torch.float32,
            )
    ent_rule_norm = ent_rule / torch.log(n_valid_rule.clamp(min=2.0))

    ent_val = zeros.clone()
    ent_val_norm = None
    if values_dist_tt is not None and values_raw is not None:
        if isinstance(values_dist_tt, Categorical):
            ent_src = values_dist if values_dist is not None else values_dist_tt
            ent_val = ent_src.entropy()
            if ent_val.dim() > 1:
                ent_val = ent_val[:, :L].sum(dim=1)
            K = logits.shape[-1]
            ent_val_norm = ent_val / (
                float(max(L, 1)) * math.log(max(float(K), 2.0))
            )
        elif values_dist is not None:
            ent_val = values_dist.entropy().sum(dim=1)

    return {
        "logp": logp,
        "logp_rule": logp_rule,
        "logp_node": logp_node,
        "logp_val": logp_val,
        "ent_rule": ent_rule,
        "ent_node": ent_node,
        "ent_val": ent_val,
        "ent_rule_norm": ent_rule_norm,
        "ent_node_norm": ent_node_norm,
        "ent_val_norm": ent_val_norm,
        "n_valid_rule": n_valid_rule,
        "n_valid_node": n_valid_node,
    }


class Policy:
    """The actor-critic network, and sampling, value and log-probability queries on it."""

    def __init__(self, ac, device):
        self.ac = ac
        self.device = device

    @classmethod
    def create(cls, config, pallete, device, masking=True):
        """A new network as described by a model config."""
        ac = ENCODERS[config["encoder"]](
            pallete,
            config["hidden_dim"],
            len(pallete.units),
            rnn_layer=config["rnn_type"],
            encoder_residual=as_bool(config["encoder_residual"]),
            node_global_context=as_bool(config["node_global_context"]),
            slot_embedding=as_bool(config["slot_embedding"]),
            value_head=config["value_head"],
            num_bins=int(config["value_bins"]),
            ordinal=as_bool(config["value_ordinal"]),
            node_decoder=config["node_decoder"],
            attn_heads=int(config["attn_heads"]),
            attn_tanh_clip=float(config["attn_tanh_clip"]),
            graph_encoder=config["graph_encoder"],
            encoder_heads=int(config["encoder_heads"]),
            encoder_layers=int(config["encoder_layers"]),
            encoder_ff=as_bool(config["encoder_ff"], False),
            encoder_reverse_edges=as_bool(config["encoder_reverse_edges"], False),
            time_feat_dim=int(config["time_feat_dim"]),
        ).to(device)
        if not masking:
            disable_relation_masks(ac)
        return cls(ac, device)

    @classmethod
    def from_checkpoint(cls, path, pallete, device, masking=True):
        """The network of a checkpoint, built from the model config stored next to it."""
        policy = cls.create(read_model_config(os.path.dirname(path)), pallete, device, masking)
        policy.load_weights(path)
        return policy

    def load_weights(self, path):
        """Load ``ac_*.pth`` (a state dict) or ``best_model.pth`` (with optimizer state)."""
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state.get("model_state_dict", state))

    def eval(self):
        self.ac.eval()

    def train(self):
        self.ac.train()

    def set_greedy(self, greedy):
        self.ac.set_greedy(greedy)

    @property
    def value_head_log_std(self):
        """The learned log std of a Gaussian value head; 0 for other heads."""
        decoder = self.ac.actor.value_decoder
        return decoder.log_std.item() if hasattr(decoder, "log_std") else 0.0

    @torch.no_grad()
    def get_action_and_value(self, obs, action_only=False):
        """Sample actions for a list of graphs.

        ``action_only`` skips the critic, log-probabilities and the node-mask
        copy. The training rollout still returns those, without entropy stats.
        """
        batch = Batch.from_data_list(obs)
        model_inputs = (
            batch.x.to(self.device),
            batch.edge_index.to(self.device),
            torch.squeeze(batch.edge_attr, dim=1).to(self.device),
            batch.batch.to(self.device),
            batch.nodes.to(self.device),
            batch.action_mask.to(self.device),
            batch.action_node_mask.to(self.device),
        )
        time_feat = self._batch_time_feat(batch)
        if action_only:
            (
                action,
                _action_dist,
                nodes,
                nodes_dist,
                _seq_len_mask,
                node_mask,
                values,
                values_dist_tt,
                _values_dist,
            ) = self.ac.act(*model_inputs, time_feat=time_feat)
            total_log_prob = None
            V = None
        else:
            (
                action,
                action_dist,
                nodes,
                nodes_dist,
                seq_len_mask,
                node_mask,
                values,
                values_dist_tt,
                values_dist,
            ), V = self.ac(*model_inputs, time_feat=time_feat)
            stats = factor_stats(
                action_dist,
                action,
                nodes_dist=nodes_dist,
                nodes=nodes,
                seq_len_mask=seq_len_mask,
                values_dist_tt=values_dist_tt,
                values_dist=values_dist,
                values_raw=values[1] if values_dist_tt is not None else None,
                logp_only=True,
            )
            total_log_prob = stats["logp"].detach()

        actions_cpu = action.detach().cpu()
        nodes_cpu = nodes.detach().cpu() if nodes_dist is not None else None
        values_real_cpu = values[0].detach().cpu() if values_dist_tt is not None else None
        values_raw_cpu = values[1].detach().cpu() if values_dist_tt is not None else None
        batch_size = actions_cpu.shape[0]
        batch_action = []

        for i in range(batch_size):
            sample = (
                actions_cpu[i],
                nodes_cpu[i].unsqueeze(0) if nodes_cpu is not None else None,
                (
                    (values_real_cpu[i].unsqueeze(0), values_raw_cpu[i].unsqueeze(0))
                    if values_raw_cpu is not None
                    else None
                ),
            )
            batch_action.append(sample)

        if action_only:
            return (batch_action, None, None), None

        if node_mask is not None:
            batch_node_masks = list(node_mask.detach().cpu().unbind(dim=0))
        else:
            batch_node_masks = [None] * batch_size
        lp_list = total_log_prob.detach().cpu().tolist()
        return (
            batch_action,
            lp_list,
            batch_node_masks,
        ), V.detach().cpu().numpy()

    def get_value(self, obs):
        """
        Queries the critic network for a value V(s) for each observation in obs.

        Parameters:
            obs - the observations at the current timestep

        Return:
            V - the predicted values of obs
        """
        # Query the critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        with torch.no_grad():
            V = self.ac.evaluate(
                obs.x.to(self.device),
                obs.edge_index.to(self.device),
                torch.squeeze(obs.edge_attr, dim=1).to(self.device),
                obs.batch.to(self.device),
                time_feat=self._batch_time_feat(obs),
            ).squeeze()
        V = V.unsqueeze(0) if V.dim() == 0 else V
        return V

    def evaluate(self, batch, batch_act, batch_nodes=None, batch_vals_raw=None):
        """
        Estimate V(s) and log pi(a|s) for a mini-batch already on device.
        """
        force_nodes = batch.force_nodes if "force_nodes" in batch else None
        force_mask = batch.force_nodes_mask if "force_nodes_mask" in batch else None
        tf_action = TFAction(batch.force_act, force_nodes, force_mask, None)

        (
            _,
            action_dist,
            _,
            nodes_dist,
            seq_len_mask,
            _,
            _,
            values_dist_tt,
            values_dist,
        ), V = self.ac(
            batch.x,
            batch.edge_index,
            torch.squeeze(batch.edge_attr, dim=1),
            batch.batch,
            batch.nodes,
            batch.action_mask,
            batch.action_node_mask,
            tf_action,
            time_feat=self._batch_time_feat(batch),
        )


        stats = factor_stats(
            action_dist,
            batch_act,
            action_mask=batch.action_mask,
            nodes_dist=nodes_dist,
            nodes=batch_nodes,
            seq_len_mask=seq_len_mask,
            values_dist_tt=values_dist_tt,
            values_dist=values_dist,
            values_raw=batch_vals_raw,
        )
        return V.squeeze(), stats

    def _batch_time_feat(self, batch):
        time_feat = getattr(batch, "time_feat", None)
        if time_feat is None:
            return None
        return time_feat.to(self.device)
