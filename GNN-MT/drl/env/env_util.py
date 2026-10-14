import numpy as np
import torch
from torch_geometric.data import Data
from drl.utils.pallete import SPECIAL

_MASK_SPEC_CACHE = {}


def node_types_and_childs_in_graph(G, node_types, pallete):
    return {
        item
        for sublist in [
            pallete.class_ancestry_dict[pallete.dic_nodes_inv[n_type]]
            for n_type in node_types
        ]
        for item in sublist
    }


def _get_action_mask_specs(pallete):
    key = id(pallete)
    cached = _MASK_SPEC_CACHE.get(key)
    if cached is not None:
        return cached
    unit_ids = sorted(pallete.unitIdToRequiredNodeTypes.keys())
    specs = []
    for unit_id in unit_ids:
        nparams = pallete.get_num_node_params_for_action(unit_id)
        params = []
        for j in range(nparams):
            params.append(
                {
                    "valid_types": np.asarray(
                        pallete.get_valid_node_types(unit_id, j)
                    ),
                    "req_forb": pallete.get_required_and_forbidden_types(unit_id, j),
                    "attr_conds": pallete.get_attr_conditions(unit_id, j),
                }
            )
        has_cond = pallete.action_has_condition(unit_id)
        specs.append(
            {
                "id": unit_id,
                "required_types": pallete.unitIdToRequiredNodeTypes[unit_id],
                "has_condition": has_cond,
                "conditions": pallete.get_conditions(unit_id) if has_cond else None,
                "nparams": nparams,
                "params": params,
            }
        )
    cached = (unit_ids, specs)
    _MASK_SPEC_CACHE[key] = cached
    return cached


def generate_tensors_from_graph(G, pallete, rule_conditions=None):

    additional_node_atts = pallete.additional_node_atts
    n_nodes = len(G)
    node_types_np = np.zeros(
        (n_nodes, 1 + len(additional_node_atts)), dtype=np.float32
    )

    for n in range(n_nodes):
        node_types_np[n, 0] = pallete.dic_nodes[G.nodes[n]["type"]]
        atts = G.nodes[n].get("atts", {})
        for i_a, a in enumerate(additional_node_atts):
            if a in atts:
                node_types_np[n, i_a + 1] = atts[a]

    edges = []
    edges_lab = []
    for e in sorted(G.edges.data(), key=lambda x: (x[0], x[1], x[2]["type"])):
        source = e[0]
        target = e[1]
        lab = e[2]["type"]
        edges.append([source, target])
        edges_lab.append(pallete.dic_edges[lab])
    if n_nodes == 1 and edges == []:
        edges.append([0, 0])
        edges_lab.append(pallete.dic_edges[SPECIAL])

    edges_np = np.asarray(edges, dtype=np.int64)
    edges_lab_np = np.asarray(edges_lab, dtype=np.int64)
    return generate_tensors_from_arrays(
        node_types_np, edges_np, edges_lab_np, pallete, rule_conditions
    )


def generate_tensors_from_arrays(
    node_types_np, edges_np, edges_lab_np, pallete, rule_conditions=None
):
    n_nodes = int(node_types_np.shape[0])
    additional_node_atts = pallete.additional_node_atts

    node_types_in_graph = sorted(set(node_types_np[:, 0].astype(np.int64).tolist()))
    node_types_w_child_types_in_graph = node_types_and_childs_in_graph(
        None, node_types_in_graph, pallete
    )

    unit_ids, specs = _get_action_mask_specs(pallete)
    n_actions = len(unit_ids)
    action_mask = np.array(
        [
            spec["required_types"].issubset(node_types_w_child_types_in_graph)
            for spec in specs
        ],
        dtype=bool,
    )

    if edges_np.ndim != 2 or edges_np.shape[0] == 0:
        edges_np = np.zeros((0, 2), dtype=np.int64)
        edges_lab_np = np.zeros((0,), dtype=np.int64)
    src = edges_np[:, 0]
    dst = edges_np[:, 1]

    action_node_mask = np.zeros(
        (n_actions, pallete.max_len, n_nodes), dtype=bool
    )

    for i, spec in enumerate(specs):
        if not action_mask[i]:
            continue
        unit_id = spec["id"]
        if spec["has_condition"] or (rule_conditions and unit_id in rule_conditions):
            conds = (
                rule_conditions[unit_id]
                if rule_conditions and unit_id in rule_conditions
                else spec["conditions"]
            )
            if conds and "elements" in conds:
                for n_type in sorted(conds["elements"].keys()):
                    cond_val = conds["elements"][n_type]
                    min_val = cond_val.get("min")
                    max_val = cond_val.get("max")
                    n_type_id = pallete.dic_nodes[n_type]
                    type_cnt = int((node_types_np[:, 0] == n_type_id).sum())
                    if min_val is not None and type_cnt <= min_val:
                        action_mask[i] = False
                        break
                    if max_val is not None and type_cnt >= max_val:
                        action_mask[i] = False
                        break
            if action_mask[i] and conds and "relations" in conds:
                for rel_type in sorted(conds["relations"].keys()):
                    cond_val = conds["relations"][rel_type]
                    min_val = cond_val.get("min")
                    max_val = cond_val.get("max")
                    rel_type_id = pallete.dic_edges[rel_type]
                    type_cnt = int((edges_lab_np == rel_type_id).sum())
                    if min_val is not None and type_cnt <= min_val:
                        action_mask[i] = False
                        break
                    if max_val is not None and type_cnt >= max_val:
                        action_mask[i] = False
                        break

        if not action_mask[i]:
            continue

        node_dim_mask = np.ones((pallete.max_len, n_nodes), dtype=bool)
        for j in range(spec["nparams"]):
            param = spec["params"][j]
            node_dim_mask[j] = np.isin(node_types_np[:, 0], param["valid_types"])

            required_forbidden_types = param["req_forb"]
            if required_forbidden_types:
                rel_mask = get_potential_targets_by_inout_dependency_np(
                    src, dst, edges_lab_np, n_nodes, required_forbidden_types
                )
                node_dim_mask[j] &= rel_mask

            attr_dependencies = param["attr_conds"]
            if attr_dependencies:
                for attr_name in sorted(attr_dependencies.keys()):
                    cond = attr_dependencies[attr_name]
                    a_idx = additional_node_atts.index(attr_name) + 1
                    selected = node_dim_mask[j]
                    attr_values = node_types_np[selected, a_idx]
                    node_dim_mask[j, selected] &= eval_nodes_condition_np(
                        attr_values, cond
                    )

            if not node_dim_mask[j].any():
                action_mask[i] = False
                break

        if action_mask[i]:
            action_node_mask[i] = node_dim_mask

    node_types = torch.from_numpy(node_types_np)
    return Data(
        node_types,
        torch.from_numpy(edges_np.T.copy()),
        torch.from_numpy(edges_lab_np).unsqueeze(1),
        nodes=torch.tensor(n_nodes),
        action_mask=torch.from_numpy(action_mask),
        action_node_mask=torch.from_numpy(action_node_mask).permute(2, 0, 1),
    )


def get_potential_targets_by_inout_dependency_np(
    src, dst, edge_attr, num_nodes, required_forbidden_types
):
    mask = np.ones(num_nodes, dtype=bool)
    required_types = required_forbidden_types.get("required", {})
    forbidden_types = required_forbidden_types.get("forbidden", {})

    if "in" in required_types:
        match = np.isin(edge_attr, required_types["in"])
        valid_nodes = dst[match]
        temp_mask = np.zeros(num_nodes, dtype=bool)
        temp_mask[valid_nodes] = True
        mask &= temp_mask

    if "out" in required_types:
        match = np.isin(edge_attr, required_types["out"])
        valid_nodes = src[match]
        temp_mask = np.zeros(num_nodes, dtype=bool)
        temp_mask[valid_nodes] = True
        mask &= temp_mask

    if "in" in forbidden_types:
        match = np.isin(edge_attr, forbidden_types["in"])
        mask[dst[match]] = False

    if "out" in forbidden_types:
        match = np.isin(edge_attr, forbidden_types["out"])
        mask[src[match]] = False

    return mask


def get_potential_targets_by_inout_dependency(
    edge_index, edge_attr, required_forbidden_types
):
    """Tensor wrapper kept for callers that still pass torch graphs."""
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    attr = edge_attr.view(-1).detach().cpu().numpy()
    num_nodes = int(max(src.max(), dst.max(), 0)) + 1
    mask_np = get_potential_targets_by_inout_dependency_np(
        src, dst, attr, num_nodes, required_forbidden_types
    )
    return torch.from_numpy(mask_np)


def eval_nodes_condition_np(node_attr, condition):
    mask = np.ones(len(node_attr), dtype=bool)
    if "gt" in condition:
        mask &= node_attr > condition["gt"]
    if "lt" in condition:
        mask &= node_attr < condition["lt"]
    if "ge" in condition:
        mask &= node_attr >= condition["ge"]
    if "le" in condition:
        mask &= node_attr <= condition["le"]
    return mask


def eval_nodes_condition(node_attr, condition):
    if torch.is_tensor(node_attr):
        node_attr = node_attr.detach().cpu().numpy()
    return torch.from_numpy(eval_nodes_condition_np(node_attr, condition))
