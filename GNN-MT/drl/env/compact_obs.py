import numpy as np

from drl.env.env_util import generate_tensors_from_arrays
from drl.utils.pallete import SPECIAL


def decode_compact_obs(payload, pallete, conditions=None):
    """Turn a Java compact-graph JSON payload into a PyG Data object and node ids."""
    num_nodes = int(payload["numNodes"])
    node_type_ids = np.asarray(payload["nodeTypes"], dtype=np.float32).reshape(-1)
    additional = pallete.additional_node_atts or []
    node_types_np = np.zeros((num_nodes, 1 + len(additional)), dtype=np.float32)
    if num_nodes > 0:
        node_types_np[:, 0] = node_type_ids[:num_nodes]

    node_attributes = payload.get("nodeAttributes")
    if node_attributes is not None and len(additional) > 0 and num_nodes > 0:
        node_attributes_np = np.asarray(node_attributes, dtype=np.float32)
        if node_attributes_np.ndim == 1:
            node_attributes_np = node_attributes_np.reshape(num_nodes, -1)
        cols = min(len(additional), node_attributes_np.shape[1])
        if cols:
            node_types_np[:, 1 : 1 + cols] = node_attributes_np[:, :cols]

    edge_sources = np.asarray(payload.get("edgeSources") or [], dtype=np.int64).reshape(
        -1
    )
    edge_targets = np.asarray(payload.get("edgeTargets") or [], dtype=np.int64).reshape(
        -1
    )
    edge_type_ids = np.asarray(payload.get("edgeTypes") or [], dtype=np.int64).reshape(
        -1
    )

    num_forward_edge_types = (len(pallete.dic_edges) - 1) // 2
    if edge_sources.size:
        all_edge_sources = np.concatenate([edge_sources, edge_targets])
        all_edge_targets = np.concatenate([edge_targets, edge_sources])
        all_edge_type_ids = np.concatenate(
            [edge_type_ids, edge_type_ids + num_forward_edge_types]
        )
    else:
        all_edge_sources = np.zeros((0,), dtype=np.int64)
        all_edge_targets = np.zeros((0,), dtype=np.int64)
        all_edge_type_ids = np.zeros((0,), dtype=np.int64)

    if num_nodes == 1 and all_edge_sources.size == 0:
        all_edge_sources = np.array([0], dtype=np.int64)
        all_edge_targets = np.array([0], dtype=np.int64)
        all_edge_type_ids = np.array([pallete.dic_edges[SPECIAL]], dtype=np.int64)

    if all_edge_sources.size:
        order = np.lexsort((all_edge_type_ids, all_edge_targets, all_edge_sources))
        edges_np = np.stack(
            [all_edge_sources[order], all_edge_targets[order]], axis=1
        )
        edges_lab_np = all_edge_type_ids[order]
    else:
        edges_np = np.zeros((0, 2), dtype=np.int64)
        edges_lab_np = np.zeros((0,), dtype=np.int64)

    data = generate_tensors_from_arrays(
        node_types_np, edges_np, edges_lab_np, pallete, conditions
    )
    node_ids = payload.get("nodeIds")
    if node_ids is None:
        node_ids = [None] * num_nodes
    return data, list(node_ids)
