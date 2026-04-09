import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from drl.utils.nn_util import init_layer
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation
import math

INFERENCE_MODE = False


def get_potential_targets_by_relation(edge_index, edge_attr, node, relation_type):
    """
    Get all relations of a given node with a specific relation type.

    Parameters:
    edge_index (torch.Tensor): A 2 x N tensor representing edges in a graph.
    edge_attr (torch.Tensor): A 1D tensor of size N representing edge relation types.
    node (int): The node whose relations we want to find.
    relation_type (int): The relation type to filter by.

    Returns:
    torch.Tensor: A tensor of nodes connected from the given node via the specified relation type.
    """
    # Find indices where the source node matches the given node
    mask_source = edge_index[0] == node

    # Find indices where the relation type matches
    mask_relation = edge_attr == relation_type

    # Combine masks
    mask = mask_source & mask_relation

    # Extract the destination nodes that satisfy the condition
    related_nodes = edge_index[1][mask]

    return related_nodes


def node_masking(
    pallete,
    nodes,
    node_dim,
    seq_lens,
    bs,
    max_nodes,
    nodeTypes,
    edge_index,
    edge_attr,
    selected_action,
    action_node_mask,
    prev_node=None,
):

    B = len(selected_action)  # Number of graphs ## OP len(selected_action)

    # batches nodes mask
    logits = torch.full((B, max_nodes), -1e10, device=nodes.device)

    # graph wise mask
    gw_mask = torch.zeros(
        (B, max_nodes), dtype=torch.bool, device=nodes.device
    )  # Mask for teacher forcing updates

    # node_cnt = 0
    for graph_idx in range(B):
        if node_dim >= seq_lens[graph_idx]:  ## OP rem item()
            continue
        # Extract node indices belonging to current graph from batched representation
        mask_node_type_indices = (bs == graph_idx).nonzero(as_tuple=True)[
            0
        ]  # Indices of nodes in this graph
        num_nodes = len(mask_node_type_indices)  # Number of valid nodes

        # Extract logits for nodes in current graph from the flattened node tensor
        graph_nodes = nodes[mask_node_type_indices]

        # Define the range of global node IDs for this graph using local contiguous span
        start_node_id = mask_node_type_indices[0]
        end_node_id = start_node_id + num_nodes

        # Find edges where both source and target belong to current graph
        source_in_graph = (edge_index[0] >= start_node_id) & (
            edge_index[0] < end_node_id
        )
        #  target_in_graph = (edge_index[1] >= start_node_id) & (edge_index[1] < end_node_id)
        graph_edge_mask = source_in_graph

        # Extract edges belonging to this graph
        graph_edge_indices = torch.nonzero(graph_edge_mask, as_tuple=False).squeeze(-1)

        # Extract subgraph edges and convert to local node indexing (0-based for current graph)
        graph_edge_index = edge_index[:, graph_edge_indices] - start_node_id
        # Extract corresponding edge attributes/labels for current graph
        graph_edge_attr = edge_attr[graph_edge_indices]

        selected_action_idx = selected_action[graph_idx].detach().item()
        mask = action_node_mask[
            mask_node_type_indices, selected_action_idx, node_dim
        ].clone()

        if node_dim > 0:
            if pallete.has_relation_dependency(selected_action_idx, node_dim):
                node_rel_dependency = pallete.get_relation_dependency(
                    selected_action_idx, node_dim
                )
                valid_node_indices = get_potential_targets_by_relation(
                    graph_edge_index,
                    graph_edge_attr,
                    prev_node[graph_idx],
                    node_rel_dependency,
                )
                node_rel_dependency_mask = torch.zeros(
                    num_nodes, dtype=torch.bool, device=mask.device
                )  # Initialize all False
                node_rel_dependency_mask[valid_node_indices] = True

                mask &= node_rel_dependency_mask

            if pallete.has_relation_restriction(selected_action_idx, node_dim):
                node_rel_restriction = pallete.get_relation_restriction(
                    selected_action_idx, node_dim
                )
                invalid_node_indices = get_potential_targets_by_relation(
                    graph_edge_index,
                    graph_edge_attr,
                    prev_node[graph_idx],
                    node_rel_restriction,
                )
                node_rel_restriction_mask = torch.ones(
                    num_nodes, dtype=torch.bool, device=mask.device
                )  # Initialize all False

                # if invalid_node_indices contains indices out of range, print a warning and ignore those indices
                if (invalid_node_indices >= num_nodes).any():

                    print("Invalid node indices:", invalid_node_indices)
                    print("Graph edge index:", graph_edge_index)
                    print("Graph edge attr:", graph_edge_attr)
                    print("Node relation restriction mask:", node_rel_restriction_mask)
                    print("Node rel restriction:", node_rel_restriction)
                node_rel_restriction_mask[invalid_node_indices] = False

                mask &= node_rel_restriction_mask

        graph_nodes[~mask] = -1e10

        if len(mask) < max_nodes:
            mask = F.pad(mask, (0, max_nodes - len(mask)), value=False)

        gw_mask[graph_idx] = mask

        logits[graph_idx, :num_nodes] = graph_nodes  # Assign logits

    return logits, gw_mask


def scale_action(pallete, values, selected_action, value_dim, nodes, node_types, bs):
    is_1d = values.dim() == 1
    if is_1d:
        values = values.unsqueeze(1)

    num_samples = selected_action.shape[0]

    scaled_vals = values.clone()
    additional_node_atts = pallete.additional_node_atts

    for graph_idx in range(num_samples):
        selected_action_idx = selected_action[graph_idx].detach().item()
        if pallete.has_range(selected_action_idx, value_dim):
            act_range = torch.tensor(
                list(
                    pallete.get_val_range_for_action(
                        selected_action_idx, value_dim
                    ).values()
                ),
                dtype=torch.int,
            )
            max_val = act_range[1]
            min_val = act_range[0]

            node_attr_conds = pallete.get_node_attr_conditions(
                selected_action_idx, value_dim
            )
            if node_attr_conds:
                mask_node_type_indices = (bs == graph_idx).nonzero(as_tuple=True)[
                    0
                ]  # Indices of nodes in this graph
                for node_param_pos, cond in node_attr_conds.items():
                    for attr_name, op in cond.items():
                        node_type_attr_idx = additional_node_atts.index(attr_name) + 1
                        node_attr_val = node_types[mask_node_type_indices][
                            nodes[graph_idx][int(node_param_pos)], node_type_attr_idx
                        ]
                        if op == "le":
                            if max_val > node_attr_val:
                                max_val = node_attr_val
                        elif op == "ge":
                            if min_val < node_attr_val:
                                min_val = node_attr_val

            scaled_vals[graph_idx, value_dim] = (
                values[graph_idx, value_dim] + 1
            ) * 0.5 * (max_val - min_val) + min_val
        else:
            raise ("No value range defined for action!")

    if is_1d:
        return scaled_vals.squeeze(1)
    return scaled_vals


class GraphEncoder(nn.Module):
    """Encodes graph nodes using HeteroConv for heterogeneous graphs"""

    def __init__(
        self,
        len_vocab_nodes,
        num_node_attr_features,
        num_edge_types,
        hidden_dim,
        attention=True,
        num_layers=3,
    ):
        super(GraphEncoder, self).__init__()

        self.num_node_attr_features = num_node_attr_features
        self.len_vocab_nodes = len_vocab_nodes
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layers for all node types
        self.emb_nodes = nn.Embedding(len_vocab_nodes, hidden_dim)

        # Projections for each node type
        if num_node_attr_features > 0:
            # Project attributes to their own dimension, then concatenate with node embeddings
            self.attr_proj = nn.ModuleDict(
                {
                    str(i): nn.Linear(num_node_attr_features, hidden_dim)
                    for i in range(len_vocab_nodes)
                }
            )
            self.concat_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in range(num_edge_types):
                for src_node_type in range(len_vocab_nodes):
                    for dst_node_type in range(len_vocab_nodes):
                        edge_key = (
                            str(src_node_type),
                            str(edge_type),
                            str(dst_node_type),
                        )
                        conv_dict[edge_key] = pyg_nn.SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim, aggr="mean"
                        )

            self.convs.append(pyg_nn.HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(
                nn.ModuleDict(
                    {str(i): GraphNorm(hidden_dim) for i in range(len_vocab_nodes)}
                )
            )

        self.globalattention = attention
        if self.globalattention:
            self.globalAttentionFinish = AttentionalAggregation(
                gate_nn=nn.Linear(hidden_dim, 1)
            )

    def _convert_to_hetero_data(self, nodeTypes, edge_index, edge_attr, bs):
        """Convert homogeneous graph data to heterogeneous format"""
        device = nodeTypes.device
        num_nodes = nodeTypes.shape[0]

        node_type_ids = nodeTypes[:, 0].long()
        node_attr_vals = (
            nodeTypes[:, 1:].float() if self.num_node_attr_features > 0 else None
        )

        # Embed all node types using single embedding layer
        node_type_emb = self.emb_nodes(node_type_ids)

        # Create x_dict: node features per node type
        x_dict = {}
        node_to_hetero_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)

        for node_type in range(self.len_vocab_nodes):
            mask = node_type_ids == node_type
            if mask.any():
                # Get node features for this type
                type_node_indices = mask.nonzero(as_tuple=True)[0]

                # Get embeddings for this node type
                emb = node_type_emb[mask]

                # Concatenate attribute features if present
                if self.num_node_attr_features > 0 and node_attr_vals is not None:
                    attr_emb = self.attr_proj[str(node_type)](node_attr_vals[mask])
                    emb = self.concat_proj(torch.cat([emb, attr_emb], dim=-1))

                x_dict[str(node_type)] = emb

                # Store mapping from original index to heterogeneous index
                node_to_hetero_idx[type_node_indices] = torch.arange(
                    type_node_indices.shape[0], device=device
                )

        edge_index_dict = {}

        for edge_type in range(self.num_edge_types):
            edge_mask = edge_attr == edge_type
            if edge_mask.any():
                type_edges = edge_index[:, edge_mask]

                src_nodes = type_edges[0]
                dst_nodes = type_edges[1]
                src_types = node_type_ids[src_nodes]
                dst_types = node_type_ids[dst_nodes]

                for src_type in src_types.unique():
                    for dst_type in dst_types.unique():
                        mask = (src_types == src_type) & (dst_types == dst_type)
                        if mask.any():
                            edge_key = (
                                str(src_type.item()),
                                str(edge_type),
                                str(dst_type.item()),
                            )

                            # Get edges and convert to heterogeneous indices
                            edges_subset = type_edges[:, mask]
                            hetero_edges = torch.stack(
                                [
                                    node_to_hetero_idx[edges_subset[0]],
                                    node_to_hetero_idx[edges_subset[1]],
                                ],
                                dim=0,
                            )

                            edge_index_dict[edge_key] = hetero_edges
        if torch.isnan(x_dict["0"]).all():
            print("x_dict:", x_dict)
            print("edge_index_dict:", edge_index_dict)
            print("node_type_ids:", node_type_ids)
            print("nodeTypes:", nodeTypes)
            print("edge_index:", edge_index)
            print("edge_attr:", edge_attr)
            print("bs:", bs)
            raise ValueError("x_dict contains None for node type '0'")
        return x_dict, edge_index_dict, node_type_ids

    def _convert_from_hetero_data(self, x_dict, node_type_ids):
        """Convert heterogeneous node embeddings back to homogeneous format"""
        device = node_type_ids.device
        num_nodes = node_type_ids.shape[0]

        nodeEmbeddings = torch.zeros(num_nodes, self.hidden_dim, device=device)

        for node_type in range(self.len_vocab_nodes):
            type_key = str(node_type)
            if type_key in x_dict:
                mask = node_type_ids == node_type
                nodeEmbeddings[mask] = x_dict[type_key]

        return nodeEmbeddings

    def forward(self, nodeTypes, edge_index, edge_attr, bs, debug_nan=False):
        # Convert to heterogeneous format
        x_dict, edge_index_dict, node_type_ids = self._convert_to_hetero_data(
            nodeTypes, edge_index, edge_attr, bs
        )

        if debug_nan:
            for node_type, emb in x_dict.items():
                if torch.isnan(emb).any():
                    print(
                        f"[GraphEncoder] NaN in input embeddings for node type {node_type}: {torch.isnan(emb).sum()}/{emb.numel()}"
                    )

        # Apply HeteroConv layers
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_dict_new = conv(x_dict, edge_index_dict)

            if debug_nan:
                for node_type, emb in x_dict_new.items():
                    if torch.isnan(emb).any():
                        print(
                            f"[GraphEncoder] Layer {layer_idx} - NaN after conv (before norm) for node type {node_type}: {torch.isnan(emb).sum()}/{emb.numel()}"
                        )
                        # Check if input to this layer was valid
                        if (
                            node_type in x_dict
                            and not torch.isnan(x_dict[node_type]).any()
                        ):
                            print(f"  -> Input was valid, conv layer introduced NaN")

            for node_type_str in x_dict_new.keys():
                x_dict_new[node_type_str] = norm[node_type_str](
                    x_dict_new[node_type_str]
                )

                if debug_nan and torch.isnan(x_dict_new[node_type_str]).any():
                    print(
                        f"[GraphEncoder] Layer {layer_idx} - NaN after norm for node type {node_type_str}"
                    )

                x_dict_new[node_type_str] = F.relu(x_dict_new[node_type_str])

            x_dict = x_dict_new

        nodeEmbeddings = self._convert_from_hetero_data(x_dict, node_type_ids)
        H = nodeEmbeddings.shape[1]

        h_G = None
        if self.globalattention:
            h_G = self.globalAttentionFinish(nodeEmbeddings, bs)
        else:
            h_G = pyg_nn.global_add_pool(nodeEmbeddings, bs)
        assert h_G.shape[1] == H

        # If h_G is only nan values, print a warning
        if torch.isnan(h_G).all():
            for key, value in x_dict.items():
                print(f"x_dict[{key}]:", value)
            print("node_type_ids:", node_type_ids)
            print("nodeTypes:", nodeTypes)
            print("edge_index:", edge_index)
            print("edge_attr:", edge_attr)

            print("Warning: h_G contains only NaN values")
            print("nodeEmbeddings:", nodeEmbeddings)
            print("bs:", bs)

        return h_G, nodeEmbeddings


class ActionDecoder(nn.Module):
    """Predicts an action given the encoded graph"""

    def __init__(self, hidden_dim, num_actions):
        super(ActionDecoder, self).__init__()
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
        # if action_logits is only nan values, print a warning
        if torch.isnan(action_logits).all():
            print("Warning: action_logits contains only NaN values")
            print("h_G:", h_G)
            print("action:", action)
            print("action_logits:", action_logits)

        if action_mask is not None and not action_mask.all():
            action_logits[~action_mask.reshape(h_G.shape[0], -1)] = -1e10
        action_distribution = Categorical(logits=action_logits)

        if INFERENCE_MODE:
            sampled_action = torch.argmax(action_distribution.probs, dim=-1)
        else:
            sampled_action = action_distribution.sample()

        return sampled_action, action_distribution, action_logits


class ARNodeDecoder(nn.Module):
    """Predicts a sequence of nodes conditioned on the action"""

    def __init__(self, pallete, hidden_dim, rnn_layer="lstm"):
        super(ARNodeDecoder, self).__init__()
        self.pallete = pallete
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
        L = torch.max(len_seq).item()
        B = nodes_bs.shape[0]
        H = node_embeddings.shape[1]
        max_nodes = nodes_bs.max().item()  # Max nodes across graphs

        sos = torch.unsqueeze(emb_action, dim=1)

        if type(self.rnn) is nn.LSTM:
            h_t = torch.unsqueeze(h_G, dim=0)
            c_t = torch.zeros_like(h_t)
            hidden = (h_t, c_t)
        elif type(self.rnn) is nn.GRU:
            hidden = torch.unsqueeze(h_G, dim=0)

        range_tensor = torch.arange(L, device=action.device).unsqueeze(0)
        seq_mask = range_tensor < len_seq.to(range_tensor.device).unsqueeze(1)

        if tf_nodes is None:
            input_lstm = sos
            selected_nodes, selected_nodes_oneh, node_mask = [], [], []
            mask = torch.zeros((N,), device=node_embeddings.device)
            final_logits = torch.full(
                (B, L, max_nodes), -1e10, device=node_embeddings.device
            )
            prev_sampled_node = None

            for t in range(L):
                output, hidden = self.rnn(input_lstm, hidden)

                assert output.shape[0] == B
                assert output.shape[2] == H
                out_lstm = torch.repeat_interleave(output[:, -1, :], nodes_bs, dim=0)

                concats = torch.cat((out_lstm, node_embeddings), dim=1)
                concats = torch.relu(self.fcNodes(concats))
                logits = torch.squeeze(self.fcNodes2(concats), dim=1)

                # print("Action node mask:", action_node_mask)
                masked_logits, gw_mask = node_masking(
                    self.pallete,
                    logits,
                    t,
                    len_seq,
                    bs,
                    max_nodes,
                    nodeTypes[:, 0],
                    edge_index,
                    edge_attr,
                    action,
                    action_node_mask,
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

                if INFERENCE_MODE:
                    sampled_node = torch.argmax(n_dist.probs, dim=-1)
                else:
                    sampled_node = n_dist.sample()

                # Ensure sampled_node is 0 at timesteps beyond sequence length; these will be ignored in loss computation
                sampled_node[~seq_mask[:, t]] = 0

                # Handle cases where no valid nodes are available at timestep where is needed
                invalid_mask = ~gw_mask.any(dim=1) & seq_mask[:, t]

                if invalid_mask.any():
                    sampled_node[invalid_mask] = -1

                selected_nodes.append(sampled_node)

                prev_sampled_node = sampled_node

                final_logits[:, t, :] = masked_logits

                all_selected_nodes = mask.clone()
                if len(nodes_bs) > 1:
                    bs_selection_idx = torch.cat(
                        [
                            torch.tensor([0], device=nodes_bs.device),
                            torch.cumsum(nodes_bs[:-1], dim=0),
                        ]
                    )
                else:  # Single graph case (batch size 1)
                    bs_selection_idx = torch.tensor([0], device=nodes_bs.device)

                bs_selection_idx = bs_selection_idx + sampled_node
                all_selected_nodes[bs_selection_idx] = (
                    1.0  # Mark the selected node as "1" for this timestep
                )

                selected_nodes_oneh.append(all_selected_nodes)

                if t < L - 1:
                    # Update input_lstm for next iteration
                    emb_seq = all_selected_nodes.unsqueeze(1) * node_embeddings
                    out = scatter(emb_seq, bs, dim=0, reduce="sum").unsqueeze(1)
                    input_lstm = out

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

            out = scatter(emb_seq, bs, dim=0, reduce="sum")
            assert out.shape[0] == B
            assert out.shape[1] == self.pallete.max_len
            assert out.shape[2] == H

            input_lstm = torch.cat((sos, out), dim=1)
            assert input_lstm.shape[0] == B
            assert input_lstm.shape[1] == self.pallete.max_len + 1
            assert input_lstm.shape[2] == H

            pack = pack_padded_sequence(
                input_lstm, len_seq, batch_first=True, enforce_sorted=False
            )

            output, l_hidden = self.rnn(pack, hidden)
            seq_unpacked, _ = pad_packed_sequence(output, batch_first=True)
            assert seq_unpacked.shape[0] == B
            assert seq_unpacked.shape[1] == L
            assert seq_unpacked.shape[2] == H

            out_lstm = torch.repeat_interleave(seq_unpacked, nodes_bs, dim=0)
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

            nodes_final[~tf_node_mask] = -1e10  # Mask out invalid nodes

            nodes_final, _ = to_dense_batch(nodes_final, bs)
            nodes_final = nodes_final.transpose(1, 2)

            # If nodes_final shape in dimension 1 is less than pallete.max_len, extend it
            if nodes_final.shape[1] < self.pallete.max_len:
                nodes_final = F.pad(
                    nodes_final,
                    (0, 0, 0, self.pallete.max_len - nodes_final.shape[1]),
                    value=-1e10,
                )
                seq_mask = F.pad(
                    seq_mask, (0, self.pallete.max_len - seq_mask.shape[1]), value=False
                )

            return None, Categorical(logits=nodes_final), seq_mask, None, None, l_hidden


class ARValueDecoder(nn.Module):
    """Predicts a sequence of nodes conditioned on the action"""

    def __init__(self, pallete, hidden_dim, rnn_layer="lstm"):
        super(ARValueDecoder, self).__init__()
        self.pallete = pallete

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
        init_layer(self.fc_mu, gain=0.01)

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

        if type(self.rnn) is nn.LSTM:
            t_hidden_h = torch.unsqueeze(h_G, dim=0)  # [1, B, H]
            t_hidden_c = torch.zeros_like(t_hidden_h)

            t_hidden = (t_hidden_h, t_hidden_c)
        elif type(self.rnn) is nn.GRU:
            t_hidden = torch.unsqueeze(h_G, dim=0)  # [1, B, H]

        selected_values, selected_raw_values = [], []
        final_mus, final_sigmas = [], []

        if tf_nodes is None:
            selected_node_embeddings = node_embeddings * torch.sum(
                nodes_oneh, dim=1, keepdim=True
            )
            selected_node_embeddings = scatter(
                selected_node_embeddings, bs, dim=0, reduce="sum"
            )  # = [B, H]

            input_decoder = self.fc_init_input(
                torch.cat(
                    (emb_action.unsqueeze(1), selected_node_embeddings.unsqueeze(1)),
                    dim=-1,
                )
            )  # [B, 1, H]
            for t in range(L):

                rnn_out, t_hidden = self.rnn(input_decoder, t_hidden)
                last_out = rnn_out[:, -1, :]

                assert last_out.shape[0] == B
                assert last_out.shape[1] == H

                fc_out = torch.relu(self.fcValues(last_out))

                mu = self.fc_mu(fc_out).squeeze(-1)  # Predict μ (mean)
                log_std = self.log_std.expand_as(mu)
                sigma = torch.exp(log_std)

                final_mus.append(mu)
                final_sigmas.append(sigma)

                value_dist = Normal(loc=mu, scale=sigma)
                #    print("mu:", mu)

                t_dist = TransformedDistribution(value_dist, [self.transform])

                if INFERENCE_MODE:
                    # Use mean for inference mode
                    sampled_raw_values = mu
                else:
                    sampled_raw_values = t_dist.rsample()

                sampled_values = scale_action(
                    self.pallete, sampled_raw_values, action, t, nodes, node_types, bs
                )

                selected_values.append(sampled_values)
                selected_raw_values.append(sampled_raw_values)

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
        else:
            tf_node_seq = tf_nodes
            selected_node_embeddings = node_embeddings * torch.sum(
                tf_node_seq, dim=1, keepdim=True
            )  # = [N, H]
            selected_node_embeddings = scatter(
                selected_node_embeddings, bs, dim=0, reduce="sum"
            )  # = [B, H]

            node_action_concat = self.fc_init_input(
                torch.cat(
                    (emb_action.unsqueeze(1), selected_node_embeddings.unsqueeze(1)),
                    dim=-1,
                )
            )  # [B, 1, H]
            input_decoder = node_action_concat.repeat(
                1, L, 1
            )  # Repeat for each timestep

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

            mu = self.fc_mu(concats).squeeze(-1)  # Predict μ (mean)
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


class SharedEncoderAC(nn.Module):
    def __init__(
        self,
        pallete,
        hidden_dim,
        num_actions,
        autoregressive_mode=False,
        rnn_layer="lstm",
    ):
        super().__init__()

        self.pallete = pallete
        self.autoregressive_mode = autoregressive_mode
        # num_node_types = 1 + len(pallete.additional_node_atts)
        len_vocab_nodes = len(pallete.dic_nodes)
        num_node_attr_features = len(pallete.additional_node_atts)
        num_edge_types = len(pallete.dic_edges)
        self.graph_encoder = GraphEncoder(
            len_vocab_nodes,
            num_node_attr_features,
            num_edge_types,
            hidden_dim,
            attention=True,
            num_layers=3,
        )
        self.actor = ActorHead(
            pallete, hidden_dim, num_actions, autoregressive_mode, rnn_layer
        )
        self.critic = CriticHead(hidden_dim)

    def encode(self, node_types, edge_index, edge_attr, bs):
        return self.graph_encoder(node_types, edge_index, edge_attr, bs)

    def act(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
    ):
        h_G, node_embeddings = self.encode(node_types, edge_index, edge_attr, bs)
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

    def evaluate(self, node_types, edge_index, edge_attr, bs):
        h_G, _ = self.encode(node_types, edge_index, edge_attr, bs)
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
    ):
        # Optional combined method
        h_G, node_embeddings = self.encode(node_types, edge_index, edge_attr, bs)
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
        autoregressive_mode=False,
        rnn_layer="lstm",
    ):
        super().__init__()

        self.pallete = pallete
        self.autoregressive_mode = autoregressive_mode
        num_edge_types = len(pallete.dic_edges)
        len_vocab_nodes = len(pallete.dic_nodes)
        num_node_attr_features = len(pallete.additional_node_atts)

        self.actor_enc = GraphEncoder(
            len_vocab_nodes,
            num_node_attr_features,
            num_edge_types,
            hidden_dim,
            attention=True,
            num_layers=3,
        )
        self.critic_enc = GraphEncoder(
            len_vocab_nodes,
            num_node_attr_features,
            num_edge_types,
            hidden_dim,
            attention=True,
            num_layers=3,
        )
        self.actor = ActorHead(
            pallete, hidden_dim, num_actions, autoregressive_mode, rnn_layer
        )
        self.critic = CriticHead(hidden_dim)

    def act(
        self,
        node_types,
        edge_index,
        edge_attr,
        bs,
        nodes_bs,
        action_mask,
        action_node_mask,
    ):
        h_G, node_embeddings = self.actor_enc(node_types, edge_index, edge_attr, bs)
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

    def evaluate(self, node_types, edge_index, edge_attr, bs):
        h_G, _ = self.critic_enc(node_types, edge_index, edge_attr, bs)
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
    ):
        # Optional combined method
        ae_h_G, ae_node_embeddings = self.actor_enc(
            node_types, edge_index, edge_attr, bs
        )
        ce_h_G, _ = self.critic_enc(node_types, edge_index, edge_attr, bs)

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
        autoregressive_mode=False,
        rnn_layer="lstm",
    ):
        super(ActorHead, self).__init__()
        self.pallete = pallete
        self.autoregressive_mode = autoregressive_mode

        self.action_decoder = ActionDecoder(hidden_dim, num_actions)

        if autoregressive_mode:
            self.node_decoder = ARNodeDecoder(pallete, hidden_dim, rnn_layer)
            self.value_decoder = ARValueDecoder(pallete, hidden_dim, rnn_layer)
        else:
            self.node_decoder = NodeDecoder(pallete, hidden_dim, rnn_layer)
            self.value_decoder = ValueDecoder(pallete, hidden_dim, rnn_layer)

        self.emb_actions = nn.Embedding(num_actions, hidden_dim)

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

        # Step 2: Predict node sequence
        nodes, nodes_dist, node_mask = None, None, None
        if self.pallete.max_len > 0:
            seq_len = torch.tensor(
                [
                    self.pallete.get_num_node_params_for_action(act.item())
                    for act in action
                ]
            )
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
