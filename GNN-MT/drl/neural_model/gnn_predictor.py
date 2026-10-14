import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from drl.utils.nn_util import init_layer
from torch_geometric.nn import GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation
from itertools import chain
from typing import NamedTuple
import math

DEBUG_NAN = False
LOGIT_MASK = -1e10
VALUE_COND_NONE = 0
VALUE_COND_LE = 1
VALUE_COND_GE = 2
_VALUE_COND_OPS = {"le": VALUE_COND_LE, "ge": VALUE_COND_GE}


def build_relation_tables(pallete, num_actions, max_len):
    """Edge-type ids for pred_relation_dependency / restriction, or -1 if unused."""
    n_act = max(num_actions, (max(pallete.unitIdToSpec.keys()) + 1) if pallete.unitIdToSpec else 0)
    dep_rel = torch.full((n_act, max_len), -1, dtype=torch.long)
    restr_rel = torch.full((n_act, max_len), -1, dtype=torch.long)
    for action in pallete.unitIdToSpec:
        nparams = pallete.get_num_node_params_for_action(action)
        for pos in range(nparams):
            if pallete.has_relation_dependency(action, pos):
                dep_rel[action, pos] = pallete.get_relation_dependency(action, pos)
            if pallete.has_relation_restriction(action, pos):
                restr_rel[action, pos] = pallete.get_relation_restriction(action, pos)
    return dep_rel, restr_rel


def build_num_node_params(pallete, num_actions):
    n_act = max(num_actions, (max(pallete.unitIdToSpec.keys()) + 1) if pallete.unitIdToSpec else 0)
    num_node_params = torch.zeros(n_act, dtype=torch.long)
    for action in pallete.unitIdToSpec:
        num_node_params[action] = pallete.get_num_node_params_for_action(action)
    return num_node_params


def build_slot_type_table(pallete, num_actions, max_len):
    """Node-type ids expected at each (action, slot), or pad_id if unspecified."""
    n_act = max(
        num_actions,
        (max(pallete.unitIdToSpec.keys()) + 1) if pallete.unitIdToSpec else 0,
    )
    pad_id = len(pallete.dic_nodes)
    slot_type = torch.full((n_act, max(max_len, 0)), pad_id, dtype=torch.long)
    for action in pallete.unitIdToSpec:
        nparams = pallete.get_num_node_params_for_action(action)
        for pos in range(nparams):
            if pallete.has_valid_types(action, pos):
                type_name = pallete.unitIdToSpec[action]["nodeParams"][pos]["validType"]
                slot_type[action, pos] = pallete.dic_nodes[type_name]
    return slot_type, pad_id


def _action_table_rows(pallete, num_actions):
    """Rows in a per-action table, including any gap below the highest unit id."""
    if not pallete.unitIdToSpec:
        return num_actions
    return max(num_actions, max(pallete.unitIdToSpec.keys()) + 1)


def build_value_scale_tables(pallete, num_actions, max_val_len):
    """Range and node-attribute bounds for each action and value slot.

    Returns ``val_min`` and ``val_max`` as float ``[A, Dv]``, and
    ``val_cond_pos``, ``val_cond_attr``, ``val_cond_op`` as long ``[A, Dv, C]``.
    Op 0 is padding, 1 is ``le`` (upper bound), 2 is ``ge`` (lower bound).
    ``val_cond_attr`` is the column in the node-feature tensor.

    Raises ValueError when a unit is missing ``valueParams[d]`` or its range
    for some ``d < max_val_len``. An empty ``max_val_len`` returns empty tables.
    """
    num_rows = _action_table_rows(pallete, num_actions)
    if max_val_len <= 0:
        value_bounds = torch.zeros((num_rows, 0), dtype=torch.float32)
        value_conditions = torch.zeros((num_rows, 0, 0), dtype=torch.long)
        return (
            value_bounds,
            value_bounds.clone(),
            value_conditions,
            value_conditions.clone(),
            value_conditions.clone(),
        )

    range_min = torch.zeros((num_rows, max_val_len), dtype=torch.float32)
    range_max = torch.zeros((num_rows, max_val_len), dtype=torch.float32)
    conditions_by_slot = {}
    widest = 0
    for action_id, spec in pallete.unitIdToSpec.items():
        value_params = spec["valueParams"]
        if len(value_params) < max_val_len:
            raise ValueError(
                f"Action {action_id} ({spec.get('name', '?')}) has "
                f"{len(value_params)} value parameters; expected at least {max_val_len}"
            )
        for slot in range(max_val_len):
            value_param = value_params[slot]
            if "range" not in value_param:
                raise ValueError(
                    f"Action {action_id} ({spec.get('name', '?')}) valueParams[{slot}] "
                    "has no range; min and max are required"
                )
            # Same order as the old ``dict.values()`` read: first is the lower bound.
            range_values = list(value_param["range"].values())
            if len(range_values) < 2:
                raise ValueError(
                    f"Action {action_id} valueParams[{slot}] range needs two bounds, "
                    f"got {range_values}"
                )
            range_min[action_id, slot] = float(range_values[0])
            range_max[action_id, slot] = float(range_values[1])
            flat_conditions = []
            for node_param_pos, cond in pallete.get_node_attr_conditions(action_id, slot).items():
                for attr_name, op_name in cond.items():
                    if op_name not in _VALUE_COND_OPS:
                        raise ValueError(
                            f"Unsupported node attribute condition {op_name!r}; "
                            "expected 'le' or 'ge'"
                        )
                    attribute_column = pallete.additional_node_atts.index(attr_name) + 1
                    flat_conditions.append(
                        (int(node_param_pos), attribute_column, _VALUE_COND_OPS[op_name])
                    )
            conditions_by_slot[(action_id, slot)] = flat_conditions
            widest = max(widest, len(flat_conditions))

    val_cond_pos = torch.zeros((num_rows, max_val_len, widest), dtype=torch.long)
    val_cond_attr = torch.zeros((num_rows, max_val_len, widest), dtype=torch.long)
    val_cond_op = torch.full(
        (num_rows, max_val_len, widest), VALUE_COND_NONE, dtype=torch.long
    )
    for (action_id, slot), flat_conditions in conditions_by_slot.items():
        for cond_index, (position, attribute_column, op_id) in enumerate(flat_conditions):
            val_cond_pos[action_id, slot, cond_index] = position
            val_cond_attr[action_id, slot, cond_index] = attribute_column
            val_cond_op[action_id, slot, cond_index] = op_id
    return range_min, range_max, val_cond_pos, val_cond_attr, val_cond_op


def relation_slots_in_use(relation_ids):
    """True for each node slot where any rule stores a relation id."""
    in_use = []
    for slot in range(relation_ids.shape[1]):
        in_use.append(bool((relation_ids[:, slot] >= 0).any().item()))
    return in_use


class BatchLayout(NamedTuple):
    """Gather indices for one concatenated batch. Built once per decoder forward."""

    ptr: torch.Tensor
    node_position: torch.Tensor
    dense_index: torch.Tensor
    dense_valid: torch.Tensor
    bs_e: torch.Tensor


def build_batch_layout(bs, nodes_bs, edge_index, max_nodes):
    """Index helpers for a PyG batch whose graphs are concatenated in order.

    ``dense_index`` is clamped so the later gather stays in range. Padded
    columns are dropped with ``dense_valid``, so the clamp never becomes a value.
    """
    N = bs.shape[0]
    ptr = torch.cumsum(nodes_bs, dim=0) - nodes_bs
    node_ids = torch.arange(N, device=bs.device)
    node_position = node_ids - ptr[bs]
    columns = torch.arange(max_nodes, device=bs.device)
    last_node = max(N - 1, 0)
    dense_index = (ptr[:, None] + columns).clamp(max=last_node)
    dense_valid = columns < nodes_bs[:, None]
    bs_e = bs[edge_index[0]]
    return BatchLayout(ptr, node_position, dense_index, dense_valid, bs_e)


def to_dense(values, layout, fill):
    """Gather node values into ``[B, max_nodes, ...]`` and fill padded columns.

    ``fill`` stays a Python scalar: a device tensor built from it is a blocking copy.
    """
    gathered = values[layout.dense_index]
    valid = layout.dense_valid
    extra_dims = gathered.dim() - valid.dim()
    if extra_dims > 0:
        valid = valid.reshape(*valid.shape, *([1] * extra_dims))
    return torch.where(valid, gathered, fill)


def resolve_node_sequence_limits(len_seq, nodes_bs, device, teacher_forcing):
    """Device lengths, CPU lengths, max nodes, and the longest node-slot count L.

    Sampling syncs once via ``tolist``. Teacher forcing syncs once via ``cpu``
    because ``pack_padded_sequence`` needs host lengths. ``len_seq`` on CPU is
    ``None`` while sampling.
    """
    if torch.is_tensor(len_seq):
        len_seq = len_seq.to(device)
    else:
        len_seq = torch.as_tensor(len_seq, device=device)
    if teacher_forcing:
        limits = torch.cat((len_seq.detach(), nodes_bs.max().detach().reshape(1))).cpu()
        len_seq_cpu = limits[:-1].to(dtype=torch.long)
        max_nodes = int(limits[-1])
        L = int(len_seq_cpu.max()) if len_seq_cpu.numel() else 0
        return len_seq, len_seq_cpu, max_nodes, L
    max_nodes, L = torch.stack((nodes_bs.max(), len_seq.max())).tolist()
    return len_seq, None, int(max_nodes), int(L)


def graph_offsets(nodes_bs):
    """Segment boundaries ``[B+1]`` of a batch whose graphs are concatenated in order."""
    return F.pad(torch.cumsum(nodes_bs, dim=0), (1, 0))


def segment_sum(src, ptr, reduce="sum"):
    """Reduce each contiguous block ``src[ptr[i]:ptr[i+1]]`` along dim 0.

    Sums each block front to back, so no atomics and, with deterministic
    algorithms on, no sort-based ``scatter_add_``. ``ptr`` is built on device,
    so the input checks (which sync) are skipped.
    """
    return torch.segment_reduce(src, reduce, offsets=ptr, axis=0, unsafe=True)


def segment_softmax(src, ptr, index):
    """Softmax of ``src`` over each block ``ptr[i]:ptr[i+1]``; ``index`` is the block of each row."""
    src_max = segment_sum(src.detach(), ptr, reduce="amax")
    out = (src - src_max.index_select(0, index)).exp()
    out_sum = segment_sum(out, ptr) + 1e-16
    return out / out_sum.index_select(0, index)


def scatter_graphs(src, nodes_bs):
    """Sum ``src`` into one row per graph."""
    return segment_sum(src, graph_offsets(nodes_bs))


def scatter_edge_hits(hit, edge_dst, N):
    """True at targets of edges whose hit flag is set.

    Uses the native ``scatter_reduce_``: the installed torch_scatter has no CUDA
    kernel for ``max``. ``amax`` does not depend on edge order, so the mask
    stays deterministic, and it does not sync.
    """
    hit_score = torch.zeros(N, dtype=torch.float32, device=hit.device)
    hit_score.scatter_reduce_(0, edge_dst, hit.float(), reduce="amax", include_self=True)
    return hit_score > 0


def draw(dist, greedy):
    """Most likely category when ``greedy``, otherwise a random sample."""
    if greedy:
        return dist.logits.argmax(dim=-1)
    return dist.sample()


def set_greedy_decoding(module, greedy):
    """Switch every decoder below ``module`` between sampling and argmax."""
    for submodule in module.modules():
        if hasattr(submodule, "greedy"):
            submodule.greedy = bool(greedy)


def apply_node_sample_overrides(sampled_node, seq_mask_t, gw_mask):
    """Zero steps past the rule length, and mark a step with no legal node as -1."""
    sampled_node = torch.where(seq_mask_t, sampled_node, 0)
    invalid_mask = ~gw_mask.any(dim=1) & seq_mask_t
    return torch.where(invalid_mask, -1, sampled_node)


def selected_node_indicator(layout, sampled_node, bs, valid_write, dtype):
    """1 at the chosen node of each graph that has a legal choice, else 0."""
    chosen = (layout.node_position == sampled_node[bs]) & valid_write[bs]
    return chosen.to(dtype)


def selected_node_embedding(layout, node_embeddings, sampled_node, valid_write):
    """Embedding of each graph's chosen node, zero where nothing was chosen.

    A gather rather than a per-graph scatter sum: with deterministic algorithms
    on, a CUDA ``scatter_add_`` over rows syncs.
    """
    last_node = max(node_embeddings.shape[0] - 1, 0)
    rows = (layout.ptr + sampled_node.clamp(min=0)).clamp(max=last_node)
    return node_embeddings[rows] * valid_write.unsqueeze(1).to(node_embeddings.dtype)


def compute_node_mask(
    node_dim,
    seq_lens,
    bs,
    edge_index,
    edge_attr,
    selected_action,
    action_node_mask,
    dep_rel,
    restr_rel,
    layout,
    slot_has_dependency,
    slot_has_restriction,
    prev_node=None,
):
    """Vectorized per-timestep validity mask.

    ``slot_has_dependency`` and ``slot_has_restriction`` are Python flags for this
    slot, computed once from the palette. Rules without a dependency are left
    unchanged by the ``where`` below, so the slot does not need a device ``any``.

    Returns
    -------
    mask : BoolTensor [N]
        Per-node validity in sparse (concatenated) order.
    gw_mask : BoolTensor [B, max_nodes]
        Dense per-graph mask; padding columns are False.
    """
    N = bs.shape[0]
    device = bs.device
    if edge_attr.dim() > 1:
        edge_attr = edge_attr.view(-1)

    graph_action = selected_action[bs]
    mask = action_node_mask[torch.arange(N, device=device), graph_action, node_dim]
    mask = mask & (seq_lens[bs] > node_dim)

    if node_dim > 0 and prev_node is not None:
        ptr = layout.ptr
        prev_ok = prev_node >= 0
        prev_global = torch.where(prev_ok, prev_node + ptr, torch.full_like(prev_node, -1))

        bs_e = layout.bs_e
        src_is_prev = edge_index[0] == prev_global[bs_e]

        if slot_has_dependency:
            dep_t = dep_rel[selected_action, node_dim]
            has_dep = dep_t >= 0
            hit_dep = src_is_prev & (edge_attr == dep_t[bs_e]) & has_dep[bs_e]
            dep_mask = scatter_edge_hits(hit_dep, edge_index[1], N)
            mask = torch.where(has_dep[bs], mask & dep_mask, mask)

        if slot_has_restriction:
            restr_t = restr_rel[selected_action, node_dim]
            has_restr = restr_t >= 0
            hit_restr = src_is_prev & (edge_attr == restr_t[bs_e]) & has_restr[bs_e]
            restr_mask = scatter_edge_hits(hit_restr, edge_index[1], N)
            mask = torch.where(has_restr[bs], mask & ~restr_mask, mask)

    gw_mask = to_dense(mask, layout, False)
    return mask, gw_mask


def node_masking(
    nodes,
    node_dim,
    seq_lens,
    bs,
    edge_index,
    edge_attr,
    selected_action,
    action_node_mask,
    dep_rel,
    restr_rel,
    layout,
    slot_has_dependency,
    slot_has_restriction,
    prev_node=None,
):
    """Vectorized per-timestep node mask. `nodes` is [N] logits; returns [B, max_nodes]."""
    mask, gw_mask = compute_node_mask(
        node_dim,
        seq_lens,
        bs,
        edge_index,
        edge_attr,
        selected_action,
        action_node_mask,
        dep_rel,
        restr_rel,
        layout,
        slot_has_dependency,
        slot_has_restriction,
        prev_node=prev_node,
    )
    masked_logits = nodes.masked_fill(~mask, LOGIT_MASK)
    logits = to_dense(masked_logits, layout, LOGIT_MASK)
    return logits, gw_mask


class SegmentGraphNorm(GraphNorm):
    """``GraphNorm`` with per-graph means from ``segment_sum`` instead of scatter.

    Same parameters and arithmetic as PyG's ``GraphNorm``, so checkpoints load
    and results match. ``bs``/``ptr`` of ``None`` normalizes over all rows.
    """

    def forward(self, x, bs=None, ptr=None):
        if ptr is None:
            bs = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            # Not torch.tensor([0, N], device=...): that is a blocking copy.
            ptr = torch.arange(2, device=x.device) * x.shape[0]
        count = (ptr[1:] - ptr[:-1]).clamp(min=1).to(x.dtype).unsqueeze(-1)
        mean = segment_sum(x, ptr) / count
        out = x - mean.index_select(0, bs) * self.mean_scale
        var = segment_sum(out.pow(2), ptr) / count
        std = (var + self.eps).sqrt().index_select(0, bs)
        return self.weight * out / std + self.bias


class SegmentAttentionalAggregation(AttentionalAggregation):
    """``AttentionalAggregation`` pooling with segment reductions over ``ptr``."""

    def pool(self, x, bs, ptr):
        gate = segment_softmax(self.gate_nn(x), ptr, bs)
        return segment_sum(gate * x, ptr)


# The convs below get edges sorted by destination node and, within a node, by
# relation (``GraphEncoder._prepare_edges``): ``ptr`` delimits each node's
# incoming edges, ``ptr_nr`` each (node, relation) pair. Aggregations are then
# segment reductions, which need no atomics and, with deterministic algorithms
# on, no sort-based ``scatter_add_``.


class RelSAGEConv(nn.Module):
    """Relational SAGE: ``W_root x_i + sum_r W_r mean_{j in N_r(i)} x_j``.

    The per-relation means are laid out as ``[N, R*H]``, so the sum over
    relations is one matmul.
    """

    def __init__(self, hidden_dim, num_rel):
        super().__init__()
        self.lin_root = nn.Linear(hidden_dim, hidden_dim)
        self.lin_rel = nn.Linear(num_rel * hidden_dim, hidden_dim, bias=False)
        # Init each W_r like its own nn.Linear(hidden_dim, hidden_dim).
        bound = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.lin_rel.weight, -bound, bound)

    def forward(self, x, src, dst, rel, ptr, ptr_nr):
        count = (ptr_nr[1:] - ptr_nr[:-1]).clamp(min=1).to(x.dtype).unsqueeze(-1)
        m = segment_sum(x.index_select(0, src), ptr_nr) / count
        return self.lin_root(x) + self.lin_rel(m.view(x.shape[0], -1))


class GATv2Layer(nn.Module):
    """GATv2 attention with a learned embedding per relation added to the keys."""

    def __init__(self, hidden_dim, heads, num_rel):
        super().__init__()
        self.heads = heads
        self.lin_l = nn.Linear(hidden_dim, hidden_dim)
        self.lin_r = nn.Linear(hidden_dim, hidden_dim)
        self.rel_emb = nn.Embedding(num_rel, hidden_dim)
        self.att = nn.Parameter(torch.empty(heads, hidden_dim // heads))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, src, dst, rel, ptr, ptr_nr):
        N, H = x.shape
        x_l = self.lin_l(x).view(N, self.heads, -1)
        x_r = self.lin_r(x).view(N, self.heads, -1)
        x_j = x_l.index_select(0, src)
        e = x_j + x_r.index_select(0, dst) + self.rel_emb(rel).view(-1, self.heads, H // self.heads)
        alpha = (F.leaky_relu(e, 0.2) * self.att).sum(-1)
        alpha = segment_softmax(alpha, ptr, dst)
        out = segment_sum(x_j * alpha.unsqueeze(-1), ptr)
        return out.view(N, H) + self.bias


class TransformerLayer(nn.Module):
    """Graph transformer attention (Shi et al.) with relation embeddings on keys
    and values, and a root weight."""

    def __init__(self, hidden_dim, heads, num_rel):
        super().__init__()
        self.heads = heads
        self.lin_q = nn.Linear(hidden_dim, hidden_dim)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)
        self.lin_skip = nn.Linear(hidden_dim, hidden_dim)
        self.rel_emb = nn.Embedding(num_rel, hidden_dim)

    def forward(self, x, src, dst, rel, ptr, ptr_nr):
        N, H = x.shape
        C = H // self.heads
        q = self.lin_q(x).view(N, self.heads, C)
        k = self.lin_k(x).view(N, self.heads, C)
        v = self.lin_v(x).view(N, self.heads, C)
        e = self.rel_emb(rel).view(-1, self.heads, C)
        k_j = k.index_select(0, src) + e
        alpha = (q.index_select(0, dst) * k_j).sum(-1) / math.sqrt(C)
        alpha = segment_softmax(alpha, ptr, dst)
        out = segment_sum((v.index_select(0, src) + e) * alpha.unsqueeze(-1), ptr)
        return out.view(N, H) + self.lin_skip(x)


class GraphEncoder(nn.Module):
    """Encodes graph nodes with ``num_layers`` relational message-passing layers.

    ``conv_type`` picks the layer: ``sage`` (``RelSAGEConv``), ``gatv2`` (with a
    self loop per node as its own relation) or ``transformer``. Relations are
    the edge types, plus their reverses with ``encoder_reverse_edges``.
    """

    def __init__(
        self,
        len_vocab_nodes,
        num_node_attr_features,
        num_edge_types,
        hidden_dim,
        attention=True,
        num_layers=3,
        residual=True,
        conv_type="sage",
        heads=4,
        encoder_ff=False,
        encoder_reverse_edges=False,
    ):
        super(GraphEncoder, self).__init__()

        self.num_node_attr_features = num_node_attr_features
        self.len_vocab_nodes = len_vocab_nodes
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.conv_type = conv_type
        self.heads = heads
        self.encoder_ff = bool(encoder_ff)
        self.encoder_reverse_edges = bool(encoder_reverse_edges)

        if conv_type not in ("sage", "gatv2", "transformer"):
            raise ValueError(
                f"Unsupported conv_type '{conv_type}'. Use 'sage', 'gatv2', or 'transformer'."
            )
        if conv_type in ("gatv2", "transformer") and hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )

        # Edge types, their reverses, and for GATv2 the self loop.
        R = num_edge_types * (2 if self.encoder_reverse_edges else 1)
        self.num_rel = max(R + (conv_type == "gatv2"), 1)

        self.emb_nodes = nn.Embedding(len_vocab_nodes, hidden_dim)
        if num_node_attr_features > 0:
            # One affine map of the attributes per node type, as one matmul over
            # ``one_hot(type) x [attr, 1]``; each type is initialized like its
            # own nn.Linear.
            n_in = num_node_attr_features + 1
            self.attr_proj = nn.Linear(len_vocab_nodes * n_in, hidden_dim, bias=False)
            bound = 1.0 / math.sqrt(num_node_attr_features)
            nn.init.uniform_(self.attr_proj.weight, -bound, bound)
            self.concat_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffs = nn.ModuleList() if self.encoder_ff else None
        for _ in range(num_layers):
            if conv_type == "sage":
                conv = RelSAGEConv(hidden_dim, self.num_rel)
            elif conv_type == "gatv2":
                conv = GATv2Layer(hidden_dim, heads, self.num_rel)
            else:
                conv = TransformerLayer(hidden_dim, heads, self.num_rel)
            self.convs.append(conv)
            self.norms.append(SegmentGraphNorm(hidden_dim))
            if self.ffs is not None:
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, 2 * hidden_dim),
                        nn.ReLU(),
                        nn.Linear(2 * hidden_dim, hidden_dim),
                    )
                )

        self.globalattention = attention
        if self.globalattention:
            self.globalAttentionFinish = SegmentAttentionalAggregation(
                gate_nn=nn.Linear(hidden_dim, 1)
            )

    def _embed_nodes(self, nodeTypes):
        """Type embedding + optional per-type attributes, original node order."""
        node_type_ids = nodeTypes[:, 0].long()
        x = self.emb_nodes(node_type_ids)
        if self.num_node_attr_features <= 0:
            return x
        attr = nodeTypes[:, 1:].to(x.dtype)
        attr = torch.cat([attr, torch.ones_like(attr[:, :1])], dim=-1)
        onehot = F.one_hot(node_type_ids, self.len_vocab_nodes).to(x.dtype)
        attr_emb = self.attr_proj((onehot.unsqueeze(-1) * attr.unsqueeze(1)).flatten(1))
        return self.concat_proj(torch.cat([x, attr_emb], dim=-1))

    def _prepare_edges(self, edge_index, edge_attr, num_nodes):
        """Relation ids and edges sorted by (destination, relation), once per forward.

        Returns ``src, dst, rel, ptr, ptr_nr`` as the convs take them.
        """
        src, dst = edge_index[0], edge_index[1]
        rel = edge_attr.view(-1).long()
        if self.encoder_reverse_edges:
            src, dst = torch.cat([src, dst]), torch.cat([dst, src])
            rel = torch.cat([rel, rel + self.num_edge_types])
        if self.conv_type == "gatv2":
            loops = torch.arange(num_nodes, device=src.device)
            src = torch.cat([src, loops])
            dst = torch.cat([dst, loops])
            rel = torch.cat([rel, torch.full_like(loops, self.num_rel - 1)])
        key = dst * self.num_rel + rel
        order = torch.argsort(key, stable=True)
        ids = torch.arange(num_nodes * self.num_rel + 1, device=src.device)
        ptr_nr = torch.searchsorted(key[order], ids)
        ptr = ptr_nr[:: self.num_rel].contiguous()
        return src[order], dst[order], rel[order], ptr, ptr_nr

    def _mp_layers(self, x, edges, bs, ptr, debug_nan=False):
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, *edges)
            if (DEBUG_NAN or debug_nan) and torch.isnan(h).any():
                print(f"[GraphEncoder] Layer {layer_idx} - NaN after conv")
            h = norm(h, bs, ptr)
            h = F.relu(h)
            x = x + h if self.residual else h
            if self.ffs is not None:
                x = x + self.ffs[layer_idx](x)
        return x

    def forward(self, nodeTypes, edge_index, edge_attr, bs, nodes_bs=None, debug_nan=False):
        if nodes_bs is None:
            nodes_bs = torch.bincount(bs)  # syncs; callers with the node counts pass them
        ptr = graph_offsets(nodes_bs)
        nodeEmbeddings = self._embed_nodes(nodeTypes)
        edges = self._prepare_edges(edge_index, edge_attr, nodeEmbeddings.shape[0])
        nodeEmbeddings = self._mp_layers(nodeEmbeddings, edges, bs, ptr, debug_nan)

        if self.globalattention:
            h_G = self.globalAttentionFinish.pool(nodeEmbeddings, bs, ptr)
        else:
            h_G = segment_sum(nodeEmbeddings, ptr)

        if DEBUG_NAN and torch.isnan(h_G).all():
            print("Warning: h_G contains only NaN values")

        return h_G, nodeEmbeddings


class ActionDecoder(nn.Module):
    """Predicts an action given the encoded graph"""

    def __init__(self, hidden_dim, num_actions):
        super(ActionDecoder, self).__init__()
        self.greedy = False
        self.fcAction = nn.Linear(hidden_dim, hidden_dim)
        self.fcAction2 = nn.Linear(hidden_dim, num_actions)

        init_layer(self.fcAction, gain=math.sqrt(2))
        init_layer(self.fcAction2, gain=0.01)

    # @profiler.time_function("ActionDecoder.forward")
    def forward(self, h_G, action_mask=None, **kwargs):

        action = torch.relu(self.fcAction(h_G))
        #  action = self.bn1(action)
        action_logits = self.fcAction2(action)
        # action_logits = self.bn2(action_logits)
        if DEBUG_NAN and torch.isnan(action_logits).all():
            print("Warning: action_logits contains only NaN values")

        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                ~action_mask.reshape(h_G.shape[0], -1), LOGIT_MASK
            )
        action_distribution = Categorical(logits=action_logits)

        sampled_action = draw(action_distribution, self.greedy)

        return sampled_action, action_distribution, action_logits


class ARNodeDecoder(nn.Module):
    """Predicts a sequence of nodes conditioned on the action"""

    def __init__(self, pallete, hidden_dim, rnn_layer="lstm", slot_embedding=True):
        super(ARNodeDecoder, self).__init__()
        self.greedy = False
        self.pallete = pallete
        self.slot_embedding = slot_embedding
        if rnn_layer == "lstm":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        elif rnn_layer == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type. Use 'lstm' or 'gru'.")

        self.fcNodes = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fcNodes2 = nn.Linear(hidden_dim, 1)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, 1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        init_layer(self.fcNodes, gain=math.sqrt(2))
        init_layer(self.fcNodes2, gain=0.01)

        num_actions = len(pallete.unitIdToSpec)
        dep_rel, restr_rel = build_relation_tables(
            pallete, num_actions, pallete.max_len
        )
        self.register_buffer("dep_rel", dep_rel)
        self.register_buffer("restr_rel", restr_rel)
        self.slot_has_dependency = relation_slots_in_use(dep_rel)
        self.slot_has_restriction = relation_slots_in_use(restr_rel)

        if slot_embedding:
            slot_type, pad_id = build_slot_type_table(
                pallete, num_actions, pallete.max_len
            )
            self.register_buffer("slot_type", slot_type)
            self.slot_type_pad_id = pad_id
            self.emb_slot_pos = nn.Embedding(pallete.max_len + 1, hidden_dim)
            self.emb_slot_type = nn.Embedding(
                len(pallete.dic_nodes) + 1, hidden_dim, padding_idx=pad_id
            )

    def forward(
        self,
        action,
        emb_action,
        sequence_input,
        h_G,
        node_embeddings,
        nodeTypes,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        len_seq,
        action_node_mask,
        tf_nodes=None,
        tf_node_mask=None,
        **kwargs,
    ):
        N = bs.shape[0]
        B = nodes_bs.shape[0]
        H = node_embeddings.shape[1]
        len_seq, len_seq_cpu, max_nodes, L = resolve_node_sequence_limits(
            len_seq, nodes_bs, action.device, teacher_forcing=tf_nodes is not None
        )
        layout = build_batch_layout(bs, nodes_bs, edge_index, max_nodes)

        sos = torch.unsqueeze(emb_action, dim=1)

        if type(self.rnn) is nn.LSTM:
            h_t = torch.unsqueeze(h_G, dim=0)
            c_t = torch.zeros_like(h_t)
            hidden = (h_t, c_t)
        elif type(self.rnn) is nn.GRU:
            hidden = torch.unsqueeze(h_G, dim=0)

        range_tensor = torch.arange(L, device=action.device).unsqueeze(0)
        seq_mask = range_tensor < len_seq.unsqueeze(1)

        if tf_nodes is None:
            input_lstm = sos
            selected_nodes, selected_nodes_oneh, node_mask = [], [], []
            final_logits = torch.full(
                (B, L, max_nodes), LOGIT_MASK, device=node_embeddings.device
            )
            prev_sampled_node = None

            for t in range(L):
                if self.slot_embedding:
                    slot_ids = torch.full(
                        (B,), t, dtype=torch.long, device=action.device
                    )
                    slot_emb = self.emb_slot_pos(slot_ids) + self.emb_slot_type(
                        self.slot_type[action, t]
                    )
                    input_lstm = input_lstm + slot_emb.unsqueeze(1)
                output, hidden = self.rnn(input_lstm, hidden)

                assert output.shape[0] == B
                assert output.shape[2] == H
                # Copy each graph's decoder state onto its nodes.
                out_lstm = output[:, -1, :][bs]

                concats = torch.cat((out_lstm, node_embeddings), dim=1)
                concats = torch.relu(self.fcNodes(concats))
                logits = torch.squeeze(self.fcNodes2(concats), dim=1)

                masked_logits, gw_mask = node_masking(
                    logits,
                    t,
                    len_seq,
                    bs,
                    edge_index,
                    edge_attr,
                    action,
                    action_node_mask,
                    self.dep_rel,
                    self.restr_rel,
                    layout,
                    self.slot_has_dependency[t],
                    self.slot_has_restriction[t],
                    prev_node=prev_sampled_node,
                )
                assert (masked_logits.shape[0], gw_mask.shape[0]) == (B, B)
                assert (masked_logits.shape[1], gw_mask.shape[1]) == (
                    max_nodes,
                    max_nodes,
                )

                node_mask.append(gw_mask)
                #    print("masked_logits", masked_logits[0])
                n_dist = Categorical(logits=masked_logits)

                sampled_node = draw(n_dist, self.greedy)

                # Steps past the rule length are ignored. -1 marks a required step
                # whose mask is empty, which the environment rejects.
                sampled_node = apply_node_sample_overrides(
                    sampled_node, seq_mask[:, t], gw_mask
                )

                selected_nodes.append(sampled_node)

                prev_sampled_node = sampled_node

                final_logits[:, t, :] = masked_logits

                valid_write = seq_mask[:, t] & (sampled_node >= 0)
                all_selected_nodes = selected_node_indicator(
                    layout, sampled_node, bs, valid_write, node_embeddings.dtype
                )

                selected_nodes_oneh.append(all_selected_nodes)

                if t < L - 1:
                    input_lstm = selected_node_embedding(
                        layout, node_embeddings, sampled_node, valid_write
                    ).unsqueeze(1)

            return (
                torch.stack(selected_nodes, dim=1),
                Categorical(logits=final_logits),
                seq_mask,
                torch.stack(selected_nodes_oneh, dim=1),
                torch.stack(node_mask, dim=1) if len(node_mask) > 0 else None,
                hidden,
            )
        else:

            emb_seq = torch.unsqueeze(tf_nodes, 2) * torch.unsqueeze(
                node_embeddings, dim=1
            )
            assert emb_seq.shape[0] == N
            assert emb_seq.shape[1] == self.pallete.max_len
            assert emb_seq.shape[2] == H

            out = scatter_graphs(emb_seq, nodes_bs)
            assert out.shape[0] == B
            assert out.shape[1] == self.pallete.max_len
            assert out.shape[2] == H

            input_lstm = torch.cat((sos, out), dim=1)
            assert input_lstm.shape[0] == B
            assert input_lstm.shape[1] == self.pallete.max_len + 1
            assert input_lstm.shape[2] == H

            if self.slot_embedding:
                T_in = input_lstm.shape[1]
                pos_ids = torch.arange(T_in, device=action.device)
                pos_emb = self.emb_slot_pos(pos_ids).unsqueeze(0)
                type_ids = F.pad(
                    self.slot_type[action],
                    (0, 1),
                    value=self.slot_type_pad_id,
                )
                input_lstm = input_lstm + pos_emb + self.emb_slot_type(type_ids)

            pack = pack_padded_sequence(
                input_lstm, len_seq_cpu, batch_first=True, enforce_sorted=False
            )

            output, l_hidden = self.rnn(pack, hidden)
            seq_unpacked, _ = pad_packed_sequence(output, batch_first=True)
            assert seq_unpacked.shape[0] == B
            assert seq_unpacked.shape[1] == L
            assert seq_unpacked.shape[2] == H

            out_lstm = seq_unpacked[bs]
            assert out_lstm.shape[0] == N
            assert out_lstm.shape[1] == L
            assert out_lstm.shape[2] == H

            nodeEmb_unsq = torch.unsqueeze(node_embeddings, dim=1).repeat(
                1, out_lstm.shape[1], 1
            )
            assert nodeEmb_unsq.shape[0] == N
            assert nodeEmb_unsq.shape[1] == L
            assert nodeEmb_unsq.shape[2] == H

            concats = torch.cat((out_lstm, nodeEmb_unsq), dim=2)
            assert concats.shape[0] == N
            assert concats.shape[1] == L
            assert concats.shape[2] == (2 * H)

            concats = torch.relu(self.fcNodes(concats))
            assert concats.shape[2] == H

            nodes_final = torch.squeeze(self.fcNodes2(concats), dim=2)
            # n x l
            assert nodes_final.shape[0] == N
            assert nodes_final.shape[1] == L

            # If tf_node_mask shape in dimension 1 is more than L, shorten it to L
            if tf_node_mask is not None and tf_node_mask.shape[1] > L:
                tf_node_mask = tf_node_mask[:, :L]

            nodes_final = nodes_final.masked_fill(~tf_node_mask, LOGIT_MASK)

            nodes_final = to_dense(nodes_final, layout, LOGIT_MASK)
            nodes_final = nodes_final.transpose(1, 2)

            # If nodes_final shape in dimension 1 is less than pallete.max_len, extend it
            if nodes_final.shape[1] < self.pallete.max_len:
                nodes_final = F.pad(
                    nodes_final,
                    (0, 0, 0, self.pallete.max_len - nodes_final.shape[1]),
                    value=LOGIT_MASK,
                )
                seq_mask = F.pad(
                    seq_mask, (0, self.pallete.max_len - seq_mask.shape[1]), value=False
                )

            return None, Categorical(logits=nodes_final), seq_mask, None, None, l_hidden


class AttnNodeDecoder(nn.Module):
    """Kool et al. attention-model decoder: MHA glimpse + tanh-clipped pointer.

    Same ``forward`` contract as ``ARNodeDecoder``. Context replaces the RNN
    state: ``q_c = W_c [h_G ; emb_action ; h_last ; h_selsum] + slot_emb``.
    """

    def __init__(
        self,
        pallete,
        hidden_dim,
        n_heads=8,
        tanh_clip=10.0,
    ):
        super(AttnNodeDecoder, self).__init__()
        self.greedy = False
        self.pallete = pallete
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.tanh_clip = float(tanh_clip)
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.head_dim = hidden_dim // n_heads

        self.W_c = nn.Linear(4 * hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_q_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(hidden_dim, hidden_dim)
        self.W_q_ptr = nn.Linear(hidden_dim, hidden_dim)

        self.v_last0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.v_sel0 = nn.Parameter(torch.zeros(1, hidden_dim))

        init_layer(self.W_c, gain=math.sqrt(2))
        init_layer(self.W_k, gain=math.sqrt(2))
        init_layer(self.W_v, gain=math.sqrt(2))
        init_layer(self.W_q_g, gain=math.sqrt(2))
        init_layer(self.W_o, gain=math.sqrt(2))
        init_layer(self.W_p, gain=math.sqrt(2))
        init_layer(self.W_q_ptr, gain=0.01)

        num_actions = len(pallete.unitIdToSpec)
        dep_rel, restr_rel = build_relation_tables(
            pallete, num_actions, pallete.max_len
        )
        self.register_buffer("dep_rel", dep_rel)
        self.register_buffer("restr_rel", restr_rel)
        self.slot_has_dependency = relation_slots_in_use(dep_rel)
        self.slot_has_restriction = relation_slots_in_use(restr_rel)

        slot_type, pad_id = build_slot_type_table(
            pallete, num_actions, pallete.max_len
        )
        self.register_buffer("slot_type", slot_type)
        self.slot_type_pad_id = pad_id
        self.emb_slot_pos = nn.Embedding(pallete.max_len + 1, hidden_dim)
        self.emb_slot_type = nn.Embedding(
            len(pallete.dic_nodes) + 1, hidden_dim, padding_idx=pad_id
        )

    def _split_heads(self, x):
        *lead, _H = x.shape
        return x.view(*lead, self.n_heads, self.head_dim)

    def _slot_emb(self, action, t):
        if torch.is_tensor(t):
            pos_ids = t
            type_ids = self.slot_type[
                action, t.clamp(max=max(self.slot_type.shape[1] - 1, 0))
            ]
        else:
            pos_ids = torch.full(
                (action.shape[0],), t, dtype=torch.long, device=action.device
            )
            type_ids = self.slot_type[action, t]
        return self.emb_slot_pos(pos_ids) + self.emb_slot_type(type_ids)

    def _glimpse(self, q, K, V, valid_mask):
        """Masked multi-head attention. q: [B,H] or [B,L,H]; K,V: [B,N,H]."""
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        qh = self._split_heads(self.W_q_g(q))
        scale = math.sqrt(self.head_dim)
        if q.dim() == 2:
            scores = torch.einsum("bhd,bnhd->bhn", qh, Kh) / scale
            mask_h = valid_mask.unsqueeze(1)
            has_valid = valid_mask.any(dim=-1).unsqueeze(1).unsqueeze(2)
        else:
            scores = torch.einsum("blhd,bnhd->blhn", qh, Kh) / scale
            mask_h = valid_mask.unsqueeze(2)
            has_valid = valid_mask.any(dim=-1).unsqueeze(2).unsqueeze(3)
        scores = scores.masked_fill(~mask_h, LOGIT_MASK)
        scores = torch.where(has_valid, scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=-1)
        attn = attn * has_valid.to(attn.dtype)
        if q.dim() == 2:
            out = torch.einsum("bhn,bnhd->bhd", attn, Vh)
            return self.W_o(out.reshape(q.shape[0], self.hidden_dim))
        out = torch.einsum("blhn,bnhd->blhd", attn, Vh)
        return self.W_o(out.reshape(q.shape[0], q.shape[1], self.hidden_dim))

    def _pointer_logits(self, q_g, K_p, valid_mask):
        q_p = self.W_q_ptr(q_g)
        scale = math.sqrt(self.hidden_dim)
        if q_p.dim() == 2:
            u = self.tanh_clip * torch.tanh(
                torch.einsum("bh,bnh->bn", q_p, K_p) / scale
            )
        else:
            u = self.tanh_clip * torch.tanh(
                torch.einsum("blh,bnh->bln", q_p, K_p) / scale
            )
        return u.masked_fill(~valid_mask, LOGIT_MASK)

    def _context(self, h_G, emb_action, h_last, h_selsum, slot_emb):
        return self.W_c(torch.cat([h_G, emb_action, h_last, h_selsum], dim=-1)) + slot_emb

    def forward(
        self,
        action,
        emb_action,
        sequence_input,
        h_G,
        node_embeddings,
        nodeTypes,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        len_seq,
        action_node_mask,
        tf_nodes=None,
        tf_node_mask=None,
        **kwargs,
    ):
        N = bs.shape[0]
        B = nodes_bs.shape[0]
        H = node_embeddings.shape[1]
        device = action.device
        len_seq, _len_seq_cpu, max_nodes, L = resolve_node_sequence_limits(
            len_seq, nodes_bs, device, teacher_forcing=tf_nodes is not None
        )
        layout = build_batch_layout(bs, nodes_bs, edge_index, max_nodes)

        range_tensor = torch.arange(max(L, 1), device=device).unsqueeze(0)
        if L > 0:
            seq_mask = range_tensor[:, :L] < len_seq.unsqueeze(1)
        else:
            seq_mask = len_seq.new_zeros((B, 0), dtype=torch.bool)

        X = to_dense(node_embeddings, layout, 0)
        K_g = self.W_k(X)
        V_g = self.W_v(X)
        K_p = self.W_p(X)

        if tf_nodes is None:
            selected_nodes, selected_nodes_oneh, node_mask = [], [], []
            final_logits = torch.full(
                (B, L, max_nodes), LOGIT_MASK, device=node_embeddings.device
            )
            prev_sampled_node = None

            h_last = self.v_last0.expand(B, -1)
            h_selsum_acc = torch.zeros(B, H, device=device, dtype=h_G.dtype)
            has_sel = torch.zeros(B, dtype=torch.bool, device=device)

            for t in range(L):
                slot_emb = self._slot_emb(action, t)
                h_selsum = torch.where(
                    has_sel.unsqueeze(-1), h_selsum_acc, self.v_sel0.expand(B, -1)
                )
                q_c = self._context(h_G, emb_action, h_last, h_selsum, slot_emb)

                _, gw_mask = compute_node_mask(
                    t,
                    len_seq,
                    bs,
                    edge_index,
                    edge_attr,
                    action,
                    action_node_mask,
                    self.dep_rel,
                    self.restr_rel,
                    layout,
                    self.slot_has_dependency[t],
                    self.slot_has_restriction[t],
                    prev_node=prev_sampled_node,
                )
                q_g = self._glimpse(q_c, K_g, V_g, gw_mask)
                masked_logits = self._pointer_logits(q_g, K_p, gw_mask)

                node_mask.append(gw_mask)
                n_dist = Categorical(logits=masked_logits)

                sampled_node = draw(n_dist, self.greedy)

                sampled_node = apply_node_sample_overrides(
                    sampled_node, seq_mask[:, t], gw_mask
                )

                selected_nodes.append(sampled_node)
                prev_sampled_node = sampled_node
                final_logits[:, t, :] = masked_logits

                valid_write = seq_mask[:, t] & (sampled_node >= 0)
                all_selected_nodes = selected_node_indicator(
                    layout, sampled_node, bs, valid_write, node_embeddings.dtype
                )
                selected_nodes_oneh.append(all_selected_nodes)

                sel_emb = selected_node_embedding(
                    layout, node_embeddings, sampled_node, valid_write
                )
                vw = valid_write.unsqueeze(-1)
                h_last = torch.where(vw, sel_emb, self.v_last0.expand(B, -1))
                h_selsum_acc = h_selsum_acc + torch.where(
                    vw, sel_emb, torch.zeros_like(sel_emb)
                )
                has_sel = has_sel | valid_write

            return (
                torch.stack(selected_nodes, dim=1) if L > 0 else torch.zeros(B, 0, dtype=torch.long, device=device),
                Categorical(logits=final_logits),
                seq_mask,
                torch.stack(selected_nodes_oneh, dim=1) if L > 0 else torch.zeros(N, 0, device=device),
                torch.stack(node_mask, dim=1) if len(node_mask) > 0 else None,
                None,
            )

        T = tf_nodes.shape[1]
        emb_seq = tf_nodes.unsqueeze(-1) * node_embeddings.unsqueeze(1)
        sel = scatter_graphs(emb_seq, nodes_bs)
        selected_any = scatter_graphs(tf_nodes, nodes_bs)

        zeros_bh = torch.zeros(B, 1, H, device=device, dtype=sel.dtype)
        zeros_b = torch.zeros(B, 1, device=device, dtype=selected_any.dtype)
        sel_shift = torch.cat([zeros_bh, sel[:, :-1]], dim=1)
        prev_ok = torch.cat([zeros_b, selected_any[:, :-1]], dim=1)
        h_last = torch.where(
            prev_ok.unsqueeze(-1) > 0,
            sel_shift,
            self.v_last0.expand(B, T, -1),
        )
        cumsum = torch.cumsum(sel, dim=1)
        sel_sum_shift = torch.cat([zeros_bh, cumsum[:, :-1]], dim=1)
        any_prev = torch.cat(
            [zeros_b, torch.cumsum(selected_any, dim=1)[:, :-1]], dim=1
        )
        h_selsum = torch.where(
            any_prev.unsqueeze(-1) > 0,
            sel_sum_shift,
            self.v_sel0.expand(B, T, -1),
        )

        pos_ids = torch.arange(T, device=device)
        if self.slot_type.shape[1] >= T:
            type_ids = self.slot_type[action, :T]
        else:
            type_ids = F.pad(
                self.slot_type[action],
                (0, T - self.slot_type.shape[1]),
                value=self.slot_type_pad_id,
            )
        slot_emb = self.emb_slot_pos(pos_ids).unsqueeze(0) + self.emb_slot_type(type_ids)

        h_G_t = h_G.unsqueeze(1).expand(-1, T, -1)
        emb_action_t = emb_action.unsqueeze(1).expand(-1, T, -1)
        q_c = self.W_c(
            torch.cat([h_G_t, emb_action_t, h_last, h_selsum], dim=-1)
        ) + slot_emb

        if tf_node_mask is not None and tf_node_mask.shape[1] > T:
            tf_node_mask = tf_node_mask[:, :T]
        if tf_node_mask is None:
            valid = layout.dense_valid.unsqueeze(1).expand(-1, T, -1)
        elif tf_node_mask.dim() == 2:
            dense_mask = to_dense(tf_node_mask.bool(), layout, False)
            valid = dense_mask.transpose(1, 2)
        else:
            valid = tf_node_mask
            if valid.shape[-1] != max_nodes:
                if valid.shape[-1] < max_nodes:
                    valid = F.pad(valid, (0, max_nodes - valid.shape[-1]), value=False)
                else:
                    valid = valid[..., :max_nodes]

        valid = valid.bool()

        if valid.shape[1] < T:
            valid = F.pad(valid, (0, 0, 0, T - valid.shape[1]), value=False)
        elif valid.shape[1] > T:
            valid = valid[:, :T]

        q_g = self._glimpse(q_c, K_g, V_g, valid)
        logits = self._pointer_logits(q_g, K_p, valid)

        if logits.shape[1] < self.pallete.max_len:
            pad_t = self.pallete.max_len - logits.shape[1]
            logits = F.pad(logits, (0, 0, 0, pad_t), value=LOGIT_MASK)
        if seq_mask.shape[1] < logits.shape[1]:
            seq_mask = F.pad(
                seq_mask, (0, logits.shape[1] - seq_mask.shape[1]), value=False
            )

        return None, Categorical(logits=logits), seq_mask, None, None, None


class ARValueDecoder(nn.Module):
    """Predicts a sequence of value parameters conditioned on the action and nodes.

    ``value_head="bins"`` (default) uses a categorical over K uniformly spaced
    bins on [-1, 1] (Tang & Agrawal, AAAI 2020), optionally with their ordinal
    parameterization. ``value_head="gaussian"`` keeps the tanh-squashed Gaussian.
    """

    def __init__(
        self,
        pallete,
        hidden_dim,
        rnn_layer="lstm",
        value_head="bins",
        num_bins=51,
        ordinal=True,
    ):
        super(ARValueDecoder, self).__init__()
        self.greedy = False
        self.pallete = pallete
        self.value_head = value_head
        if value_head not in ("bins", "gaussian"):
            raise ValueError(
                f"Unsupported value_head '{value_head}'. Use 'bins' or 'gaussian'."
            )
        self.num_bins = int(num_bins)
        self.ordinal = bool(ordinal)

        if value_head == "gaussian":
            self.transform = TanhTransform(cache_size=1)

        if rnn_layer == "lstm":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
            self.context_encoder = nn.LSTM(
                hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False
            )
            self.fc_context_h = nn.Linear(2 * hidden_dim, hidden_dim)
            self.fc_context_c = nn.Linear(2 * hidden_dim, hidden_dim)
            init_layer(self.fc_context_c, gain=math.sqrt(2))

        elif rnn_layer == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
            self.context_encoder = nn.GRU(
                hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False
            )
            self.fc_context_h = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            raise ValueError("Unsupported RNN type. Use 'lstm' or 'gru'.")

        self.fc_init_input = nn.Linear(
            2 * hidden_dim, hidden_dim
        )  # Initial input for RNN

        self.fcValues = nn.Linear(hidden_dim, hidden_dim)
        if value_head == "bins":
            if self.num_bins < 2:
                raise ValueError("num_bins must be >= 2")
            self.fc_logits = nn.Linear(hidden_dim, self.num_bins)
            self.register_buffer(
                "bin_centers", torch.linspace(-1.0, 1.0, self.num_bins)
            )
        else:
            self.fc_mu = nn.Linear(hidden_dim, 1)  # Predict μ (mean)
            self.log_std = nn.Parameter(torch.ones(1) * -0.5)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, 1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        init_layer(self.fc_context_h, gain=math.sqrt(2))
        init_layer(self.fc_init_input, gain=math.sqrt(2))
        init_layer(self.fcValues, gain=math.sqrt(2))
        if value_head == "bins":
            init_layer(self.fc_logits, gain=0.01)
        else:
            init_layer(self.fc_mu, gain=0.01)

        num_actions = len(pallete.unitIdToSpec)
        (
            val_min,
            val_max,
            val_cond_pos,
            val_cond_attr,
            val_cond_op,
        ) = build_value_scale_tables(pallete, num_actions, pallete.max_val_len)
        # Not saved in checkpoints: rebuilt from the palette, so old weights still load.
        self.register_buffer("val_min", val_min, persistent=False)
        self.register_buffer("val_max", val_max, persistent=False)
        self.register_buffer("val_cond_pos", val_cond_pos, persistent=False)
        self.register_buffer("val_cond_attr", val_cond_attr, persistent=False)
        self.register_buffer("val_cond_op", val_cond_op, persistent=False)

    def _value_logits(self, feat):
        """Map decoder features to categorical logits over bins, shape [..., K]."""
        raw = self.fc_logits(feat)
        if not self.ordinal:
            return raw
        s = F.logsigmoid(raw)
        ns = F.logsigmoid(-raw)
        return torch.cumsum(s, dim=-1) + (
            ns.sum(dim=-1, keepdim=True) - torch.cumsum(ns, dim=-1)
        )

    def _bin_dist(self, feat):
        return Categorical(logits=self._value_logits(feat))

    def _init_hidden(self, h_G):
        if type(self.rnn) is nn.LSTM:
            t_hidden_h = torch.unsqueeze(h_G, dim=0)  # [1, B, H]
            t_hidden_c = torch.zeros_like(t_hidden_h)
            return (t_hidden_h, t_hidden_c)
        return torch.unsqueeze(h_G, dim=0)  # [1, B, H]

    def _selected_node_embeddings(self, node_embeddings, node_indicator, nodes_bs):
        selected_node_embeddings = node_embeddings * torch.sum(
            node_indicator, dim=1, keepdim=True
        )
        return scatter_graphs(selected_node_embeddings, nodes_bs)  # [B, H]

    def _scale_values(self, values, selected_action, value_dim, nodes, node_types, ptr, nodes_bs):
        """Map one slot of tanh-space samples into that action's value range.

        Node-attribute conditions tighten the range in palette order. A stored
        node index of -1 addresses the last node of that graph, matching the
        previous per-graph slice index.
        """
        min_val = self.val_min[selected_action, value_dim]
        max_val = self.val_max[selected_action, value_dim]
        for c in range(self.val_cond_op.shape[-1]):
            op = self.val_cond_op[selected_action, value_dim, c]
            node_param_pos = self.val_cond_pos[selected_action, value_dim, c]
            node_type_attr_idx = self.val_cond_attr[selected_action, value_dim, c]
            node_idx = nodes.gather(1, node_param_pos.unsqueeze(1)).squeeze(1)
            node_idx = torch.where(node_idx < 0, node_idx + nodes_bs, node_idx)
            node_attr_val = node_types[ptr + node_idx, node_type_attr_idx]
            max_val = torch.where(
                op == VALUE_COND_LE, torch.minimum(max_val, node_attr_val), max_val
            )
            min_val = torch.where(
                op == VALUE_COND_GE, torch.maximum(min_val, node_attr_val), min_val
            )
        return (values + 1) * 0.5 * (max_val - min_val) + min_val

    def forward(
        self,
        action,
        emb_action,
        nodes,
        nodes_oneh,
        node_types,
        h_G,
        node_hidden,
        node_embeddings,
        bs,
        nodes_bs,
        len_seq,
        tf_nodes=None,
    ):
        L = len_seq
        B = nodes_bs.shape[0]
        H = node_embeddings.shape[1]
        t_hidden = self._init_hidden(h_G)

        if tf_nodes is None:
            ptr = torch.cumsum(nodes_bs, dim=0) - nodes_bs
            selected_node_embeddings = self._selected_node_embeddings(
                node_embeddings, nodes_oneh, nodes_bs
            )
            input_decoder = self.fc_init_input(
                torch.cat(
                    (emb_action.unsqueeze(1), selected_node_embeddings.unsqueeze(1)),
                    dim=-1,
                )
            )  # [B, 1, H]
            selected_values, selected_raw_values = [], []
            step_logits = []
            final_mus, final_sigmas = [], []
            for t in range(L):
                rnn_out, t_hidden = self.rnn(input_decoder, t_hidden)
                last_out = rnn_out[:, -1, :]

                assert last_out.shape[0] == B
                assert last_out.shape[1] == H

                fc_out = torch.relu(self.fcValues(last_out))

                if self.value_head == "bins":
                    logits_t = self._value_logits(fc_out)
                    step_logits.append(logits_t)
                    dist_t = Categorical(logits=logits_t)
                    idx = draw(dist_t, self.greedy)
                    sampled_raw_values = self.bin_centers[idx]
                    selected_raw_values.append(idx)
                else:
                    mu = self.fc_mu(fc_out).squeeze(-1)
                    log_std = self.log_std.expand_as(mu)
                    sigma = torch.exp(log_std)
                    final_mus.append(mu)
                    final_sigmas.append(sigma)
                    t_dist = TransformedDistribution(
                        Normal(loc=mu, scale=sigma), [self.transform]
                    )
                    if self.greedy:
                        # Median of the tanh-squashed Gaussian, in the same [-1, 1] space as samples.
                        sampled_raw_values = torch.tanh(mu)
                    else:
                        sampled_raw_values = t_dist.rsample()
                    selected_raw_values.append(sampled_raw_values)

                sampled_values = self._scale_values(
                    sampled_raw_values,
                    action,
                    t,
                    nodes,
                    node_types,
                    ptr,
                    nodes_bs,
                )
                selected_values.append(sampled_values)

            if self.value_head == "bins":
                cat_dist = Categorical(logits=torch.stack(step_logits, dim=1))
                return (
                    (
                        torch.stack(selected_values, dim=1),
                        torch.stack(selected_raw_values, dim=1),
                    ),
                    cat_dist,
                    cat_dist,
                )

            m_mu = torch.stack(final_mus, dim=1)
            m_sigma = torch.stack(final_sigmas, dim=1)
            value_dist = Normal(loc=m_mu, scale=m_sigma)
            final_dist = TransformedDistribution(
                Normal(loc=m_mu, scale=m_sigma), [self.transform]
            )
            return (
                (
                    torch.stack(selected_values, dim=1),
                    torch.stack(selected_raw_values, dim=1),
                ),
                final_dist,
                value_dist,
            )

        tf_node_seq = tf_nodes
        selected_node_embeddings = self._selected_node_embeddings(
            node_embeddings, tf_node_seq, nodes_bs
        )
        node_action_concat = self.fc_init_input(
            torch.cat(
                (emb_action.unsqueeze(1), selected_node_embeddings.unsqueeze(1)),
                dim=-1,
            )
        )  # [B, 1, H]
        input_decoder = node_action_concat.repeat(1, L, 1)

        pack = pack_padded_sequence(
            input_decoder,
            torch.full((nodes_bs.shape[0],), L),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.rnn(pack, t_hidden)
        seq_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        assert seq_unpacked.shape[0] == B
        assert seq_unpacked.shape[1] == L
        assert seq_unpacked.shape[2] == H

        concats = torch.relu(self.fcValues(seq_unpacked))
        assert concats.shape[0] == B
        assert concats.shape[1] == L
        assert concats.shape[2] == H

        if self.value_head == "bins":
            cat_dist = self._bin_dist(concats)
            return None, cat_dist, cat_dist

        mu = self.fc_mu(concats).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        sigma = torch.exp(log_std)
        value_dist = Normal(loc=mu, scale=sigma)
        final_dist = TransformedDistribution(
            Normal(loc=mu, scale=sigma), [self.transform]
        )
        return None, final_dist, value_dist


class CriticHead(nn.Module):
    def __init__(self, hidden_dim):
        super(CriticHead, self).__init__()

        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.lin_final = nn.Linear(hidden_dim, 1)

        init_layer(self.lin, gain=math.sqrt(2))
        init_layer(self.lin_final, gain=1.0)

    def forward(self, h_G):

        h_1 = torch.relu(self.lin(h_G))
        output = self.lin_final(h_1)

        return output.squeeze(-1)


def build_graph_encoder(
    pallete,
    hidden_dim,
    residual=True,
    conv_type="sage",
    heads=4,
    num_layers=3,
    encoder_ff=False,
    encoder_reverse_edges=False,
):
    return GraphEncoder(
        len(pallete.dic_nodes),
        len(pallete.additional_node_atts),
        len(pallete.dic_edges),
        hidden_dim,
        attention=True,
        num_layers=num_layers,
        residual=residual,
        conv_type=conv_type,
        heads=heads,
        encoder_ff=encoder_ff,
        encoder_reverse_edges=encoder_reverse_edges,
    )


def fuse_time_feature(graph_embedding, time_feat, projection):
    """Add a projected time feature to the pooled graph embedding.

    Skipped when the model has no time projection or the batch carries no feature.
    Consumers of the embedding start with a linear layer, so this matches
    concatenating the feature without changing downstream shapes.
    """
    if projection is None or time_feat is None:
        return graph_embedding
    if time_feat.dim() == 3 and time_feat.shape[1] == 1:
        time_feat = time_feat.squeeze(1)
    time_feat = time_feat.to(device=graph_embedding.device, dtype=graph_embedding.dtype)
    return graph_embedding + projection(time_feat)


class SharedEncoderAC(nn.Module):
    def __init__(
        self,
        pallete,
        hidden_dim,
        num_actions,
        rnn_layer="lstm",
        encoder_residual=True,
        node_global_context=True,
        slot_embedding=True,
        value_head="bins",
        num_bins=51,
        ordinal=True,
        node_decoder="rnn",
        attn_heads=8,
        attn_tanh_clip=10.0,
        graph_encoder="sage",
        encoder_heads=4,
        encoder_layers=3,
            encoder_ff=False,
        encoder_reverse_edges=False,
        time_feat_dim=0,
    ):
        super().__init__()

        self.pallete = pallete
        self.time_feat_dim = int(time_feat_dim)
        self.time_proj = None
        if self.time_feat_dim > 0:
            self.time_proj = nn.Linear(self.time_feat_dim, hidden_dim)
            init_layer(self.time_proj, gain=0.01)
        self.graph_encoder = build_graph_encoder(
            pallete,
            hidden_dim,
            residual=encoder_residual,
            conv_type=graph_encoder,
            heads=encoder_heads,
            num_layers=encoder_layers,
            encoder_ff=encoder_ff,
            encoder_reverse_edges=encoder_reverse_edges,
        )
        self.actor = ActorHead(
            pallete,
            hidden_dim,
            num_actions,
            rnn_layer,
            node_global_context=node_global_context,
            slot_embedding=slot_embedding,
            value_head=value_head,
            num_bins=num_bins,
            ordinal=ordinal,
            node_decoder=node_decoder,
            attn_heads=attn_heads,
            attn_tanh_clip=attn_tanh_clip,
        )
        self.critic = CriticHead(hidden_dim)

    def policy_parameters(self):
        parts = [self.graph_encoder.parameters(), self.actor.parameters()]
        if self.time_proj is not None:
            parts.append(self.time_proj.parameters())
        return chain(*parts)

    def value_parameters(self):
        parts = [self.graph_encoder.parameters(), self.critic.parameters()]
        if self.time_proj is not None:
            parts.append(self.time_proj.parameters())
        return chain(*parts)

    def encode(self, node_types, edge_index, edge_attr, bs, time_feat=None, nodes_bs=None):
        h_G, node_embeddings = self.graph_encoder(
            node_types, edge_index, edge_attr, bs, nodes_bs
        )
        return fuse_time_feature(h_G, time_feat, self.time_proj), node_embeddings

    def act(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
        time_feat=None,
    ):
        h_G, node_embeddings = self.encode(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat, nodes_bs=nodes_bs
        )
        return self.actor(
            h_G,
            node_embeddings,
            node_types,
            edge_index,
            edge_attr,
            bs,
            nodes_bs,
            action_mask,
            action_node_mask,
        )

    def set_greedy(self, greedy):
        """Argmax decoding for maximum-likelihood inference; training samples."""
        set_greedy_decoding(self.actor, greedy)

    def evaluate(self, node_types, edge_index, edge_attr, bs, time_feat=None):
        h_G, _ = self.encode(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat
        )
        return self.critic(h_G)

    def forward(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
        tf_action=None,
        time_feat=None,
    ):
        # Optional combined method
        h_G, node_embeddings = self.encode(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat, nodes_bs=nodes_bs
        )
        return self.actor(
            h_G,
            node_embeddings,
            node_types,
            edge_index,
            edge_attr,
            bs,
            nodes_bs,
            action_mask,
            action_node_mask,
            tf_action=tf_action,
        ), self.critic(h_G)


class SeparateEncoderAC(nn.Module):
    def __init__(
        self,
        pallete,
        hidden_dim,
        num_actions,
        rnn_layer="lstm",
        encoder_residual=True,
        node_global_context=True,
        slot_embedding=True,
        value_head="bins",
        num_bins=51,
        ordinal=True,
        node_decoder="rnn",
        attn_heads=8,
        attn_tanh_clip=10.0,
        graph_encoder="sage",
        encoder_heads=4,
        encoder_layers=3,
            encoder_ff=False,
        encoder_reverse_edges=False,
        time_feat_dim=0,
    ):
        super().__init__()

        self.pallete = pallete
        self.time_feat_dim = int(time_feat_dim)
        self.actor_time_proj = None
        self.critic_time_proj = None
        if self.time_feat_dim > 0:
            self.actor_time_proj = nn.Linear(self.time_feat_dim, hidden_dim)
            self.critic_time_proj = nn.Linear(self.time_feat_dim, hidden_dim)
            init_layer(self.actor_time_proj, gain=0.01)
            init_layer(self.critic_time_proj, gain=0.01)
        enc_kwargs = dict(
            residual=encoder_residual,
            conv_type=graph_encoder,
            heads=encoder_heads,
            num_layers=encoder_layers,
            encoder_ff=encoder_ff,
            encoder_reverse_edges=encoder_reverse_edges,
        )
        self.actor_enc = build_graph_encoder(pallete, hidden_dim, **enc_kwargs)
        self.critic_enc = build_graph_encoder(pallete, hidden_dim, **enc_kwargs)
        self.actor = ActorHead(
            pallete,
            hidden_dim,
            num_actions,
            rnn_layer,
            node_global_context=node_global_context,
            slot_embedding=slot_embedding,
            value_head=value_head,
            num_bins=num_bins,
            ordinal=ordinal,
            node_decoder=node_decoder,
            attn_heads=attn_heads,
            attn_tanh_clip=attn_tanh_clip,
        )
        self.critic = CriticHead(hidden_dim)

    def policy_parameters(self):
        parts = [self.actor_enc.parameters(), self.actor.parameters()]
        if self.actor_time_proj is not None:
            parts.append(self.actor_time_proj.parameters())
        return chain(*parts)

    def value_parameters(self):
        parts = [self.critic_enc.parameters(), self.critic.parameters()]
        if self.critic_time_proj is not None:
            parts.append(self.critic_time_proj.parameters())
        return chain(*parts)

    def encode_actor(self, node_types, edge_index, edge_attr, bs, time_feat=None, nodes_bs=None):
        h_G, node_embeddings = self.actor_enc(node_types, edge_index, edge_attr, bs, nodes_bs)
        return fuse_time_feature(h_G, time_feat, self.actor_time_proj), node_embeddings

    def encode_critic(self, node_types, edge_index, edge_attr, bs, time_feat=None, nodes_bs=None):
        h_G, node_embeddings = self.critic_enc(node_types, edge_index, edge_attr, bs, nodes_bs)
        return fuse_time_feature(h_G, time_feat, self.critic_time_proj), node_embeddings

    def act(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
        time_feat=None,
    ):
        h_G, node_embeddings = self.encode_actor(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat, nodes_bs=nodes_bs
        )
        return self.actor(
            h_G,
            node_embeddings,
            node_types,
            edge_index,
            edge_attr,
            bs,
            nodes_bs,
            action_mask,
            action_node_mask,
        )

    def set_greedy(self, greedy):
        """Argmax decoding for maximum-likelihood inference; training samples."""
        set_greedy_decoding(self.actor, greedy)

    def evaluate(self, node_types, edge_index, edge_attr, bs, time_feat=None):
        h_G, _ = self.encode_critic(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat
        )
        return self.critic(h_G)

    def forward(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
        tf_action=None,
        time_feat=None,
    ):
        # Optional combined method
        ae_h_G, ae_node_embeddings = self.encode_actor(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat, nodes_bs=nodes_bs
        )
        ce_h_G, _ = self.encode_critic(
            node_types, edge_index, edge_attr, bs, time_feat=time_feat, nodes_bs=nodes_bs
        )

        return self.actor(
            ae_h_G,
            ae_node_embeddings,
            node_types,
            edge_index,
            edge_attr,
            bs,
            nodes_bs,
            action_mask,
            action_node_mask,
            tf_action=tf_action,
        ), self.critic(ce_h_G)


class ActorHead(nn.Module):
    """Full model integrating action, node, and value decoders"""

    def __init__(
        self,
        pallete,
        hidden_dim,
        num_actions,
        rnn_layer="lstm",
        node_global_context=True,
        slot_embedding=True,
        value_head="bins",
        num_bins=51,
        ordinal=True,
        node_decoder="rnn",
        attn_heads=8,
        attn_tanh_clip=10.0,
    ):
        super(ActorHead, self).__init__()
        self.pallete = pallete
        self.node_global_context = node_global_context
        self.node_decoder_type = node_decoder

        self.action_decoder = ActionDecoder(hidden_dim, num_actions)

        if node_decoder == "attention":
            self.node_decoder = AttnNodeDecoder(
                pallete,
                hidden_dim,
                n_heads=attn_heads,
                tanh_clip=attn_tanh_clip,
            )
        elif node_decoder == "rnn":
            self.node_decoder = ARNodeDecoder(
                pallete, hidden_dim, rnn_layer, slot_embedding=slot_embedding
            )
        else:
            raise ValueError(
                f"Unsupported node_decoder '{node_decoder}'. Use 'rnn' or 'attention'."
            )
        self.value_decoder = ARValueDecoder(
            pallete,
            hidden_dim,
            rnn_layer,
            value_head=value_head,
            num_bins=num_bins,
            ordinal=ordinal,
        )

        self.emb_actions = nn.Embedding(num_actions, hidden_dim)
        self.register_buffer(
            "num_node_params", build_num_node_params(pallete, num_actions)
        )
        if node_global_context:
            self.node_ctx = nn.Linear(2 * hidden_dim, hidden_dim)
            init_layer(self.node_ctx, gain=math.sqrt(2))

    def forward(
        self,
        h_G,
        node_embeddings,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
        tf_action=None,
    ):

        # Step 1: Predict action
        action, action_dist, _ = self.action_decoder(h_G, action_mask)  # Predict action

        if tf_action:
            action = tf_action.action
            ## TODO CHECK gradient flow
        emb_action = self.emb_actions(action)

        if self.node_global_context:
            node_embeddings = torch.relu(
                self.node_ctx(torch.cat([node_embeddings, h_G[bs]], dim=-1))
            )

        # Step 2: Predict node sequence
        nodes, nodes_dist, node_mask = None, None, None
        seq_len_mask = None
        if self.pallete.max_len > 0:
            seq_len = self.num_node_params[action]
            nodes, nodes_dist, seq_len_mask, nodes_oneh, node_mask, node_hidden = (
                self.node_decoder(
                    action,
                    emb_action,
                    None,
                    h_G,
                    node_embeddings,
                    node_types,
                    edge_index,
                    edge_attr,
                    bs,
                    nodes_bs,
                    seq_len,
                    action_node_mask,
                    with_gt=False,
                    tf_nodes=tf_action.nodes if tf_action else None,
                    tf_node_mask=tf_action.nodes_dim_mask if tf_action else None,
                )
            )

        #  if tf_action:
        #      nodes_oneh = tf_action.nodes
        # Step 3: Predict value sequence
        values, values_dist, values_dist_tt = None, None, None
        if self.pallete.max_val_len > 0:
            values, values_dist_tt, values_dist = self.value_decoder(
                action,
                emb_action,
                nodes,
                nodes_oneh,
                node_types,
                h_G,
                node_hidden,
                node_embeddings,
                bs,
                nodes_bs,
                self.pallete.max_val_len,
                tf_nodes=tf_action.nodes if tf_action else None,
            )

        return (
            action,
            action_dist,
            nodes,
            nodes_dist,
            seq_len_mask,
            node_mask,
            values,
            values_dist_tt,
            values_dist,
        )
