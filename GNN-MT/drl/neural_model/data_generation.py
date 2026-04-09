import networkx as nx
import torch
from torch_geometric.data import Data

from drl.utils.pallete import SPECIAL


def generate_tensors_from_graph(G, pallete, len_seq, selected_action_id):
    node_types = torch.zeros(
        len(G), 1 + len(pallete.additional_node_atts)
    )  # Assuming each node has two features

    for n in range(len(G)):
        node_type = pallete.dic_nodes[G.nodes[n]["type"]]
        node_types[n][0] = node_type
        for i_a, a in enumerate(pallete.additional_node_atts):
            if a in G.nodes[n]["atts"]:
                node_types[n][i_a + 1] = G.nodes[n]["atts"][a]

    selection_mask = torch.full((len(G), len_seq), True, dtype=torch.bool)

    if selected_action_id is not None:
        for dim in range(len_seq):
            if pallete.has_valid_types(selected_action_id, dim):
                valid_node_type_indices = torch.tensor(
                    pallete.get_valid_node_types(selected_action_id, dim)
                )
                mask = torch.isin(node_types[:, 0], valid_node_type_indices)
                selection_mask[:, dim] = mask

    special_nodes = [[0 for j in range(pallete.max_len)] for i in range(len(G))]

    ### STATIC - TODO is about maximal length of rule with most parameters (pallete.max_len) vs. len_seq (current rule!)
    special_nodes_masked = [
        [0 if j < len_seq else -1 for j in range(pallete.max_len)]
        for i in range(len(G))
    ]

    for n in range(len(G)):
        if "v_id" in G.nodes[n]:

            special_nodes[n][G.nodes[n]["v_id"]] = 1
            special_nodes_masked[n][G.nodes[n]["v_id"]] = 1

    special_nodes_values = []
    # edges
    edges = []
    edges_lab = []
    for e in list(G.edges.data()):
        source = e[0]
        target = e[1]
        lab = e[2]["type"]
        edges.append([source, target])
        lab = pallete.dic_edges[lab]
        edges_lab.append(lab)
    if len(G) == 1 and edges == []:
        edges.append([0, 0])
        edges_lab.append(pallete.dic_edges[SPECIAL])
    return (
        node_types,
        selection_mask,
        torch.tensor(special_nodes),
        torch.tensor(special_nodes_masked),
        torch.transpose(torch.tensor(edges), 0, 1),
        torch.unsqueeze(torch.tensor(edges_lab), dim=1),
    )


# graphs of the sequence must have ids 0,1,2...len(g)
def graph2data_preaction(G, pallete):
    nT, nSM, sN, sNM, edges, edges_lab = generate_tensors_from_graph(
        G, pallete, 1, None
    )
    data = Data(x=nT, edge_index=edges, edge_attr=edges_lab, nodes=torch.tensor(len(G)))
    return data


def graph2data_postaction(G, pallete, len_seq):
    nT, nSM, sN, sNM, edges, edges_lab = generate_tensors_from_graph(
        G, pallete, len_seq, None
    )
    data = Data(
        x=nT,
        edge_index=edges,
        edge_attr=edges_lab,
        nodes=torch.tensor(len(G)),
        sequence=sN,
        sequence_masked=sNM,
    )
    return data


def sample2data(sample, pallete):

    g = sample[0]
    rule_application = sample[1]
    is_last = sample[2]
    model_path = sample[3]

    len_seq = len(pallete.units[rule_application["unitName"]]["nodeParams"])
    len_seq_val = len(pallete.units[rule_application["unitName"]]["valueParams"])

    nT, nSM, sN, sNM, edges, edges_lab = generate_tensors_from_graph(
        g, pallete, len_seq, rule_application["unitId"]
    )
    # len(pallete.get_special_nodes(id_edit)))

    action = rule_application["unitId"]

    if is_last:
        finished = torch.tensor(1)
    else:
        finished = torch.tensor(0)

    param_names_val = [
        vp["name"] for vp in pallete.units[rule_application["unitName"]]["valueParams"]
    ]
    sequence_val = [
        rule_application["paramValues"][vp_name] if i < pallete.max_val_len else 0
        for i, vp_name in enumerate(param_names_val)
    ]

    val_ranges = [
        tuple(vp["range"].values())
        for vp in pallete.units[rule_application["unitName"]]["valueParams"]
    ]
    data = Data(
        x=nT,  # node types [0,1,1,..]
        selection_mask=nSM,
        edge_index=edges,  # edge ids [[0,1],[0,2],..]
        edge_attr=edges_lab,  # edge label ids
        action=torch.tensor(action),
        nodes=torch.tensor(len(g)),
        sequence=sN,  # edit op connection points
        sequence_masked=sNM,
        sequence_val=torch.tensor(sequence_val).unsqueeze(0),
        val_ranges=torch.tensor(val_ranges),
        len_seq=torch.tensor(len_seq),  # num (node) params of applied rule,
        len_seq_val=torch.tensor(
            len_seq_val
        ),  # num (node) params of applied rule, ### STATIC
        # amount=torch.tensor(rule_application.amount, dtype=torch.float32),
        finished=finished,
        modelPath=model_path,
    )
    return data


def sequence2data(sequence, pallete):
    result = []
    for j, (G, id_edit) in enumerate(sequence):
        nT, nSM, sN, sNM, edges, edges_lab = generate_tensors_from_graph(
            G, pallete, len(pallete.get_special_nodes(id_edit))
        )
        if j == 0:
            finished = torch.tensor(1)
        else:
            finished = torch.tensor(0)

        data = Data(
            x=nT,  # node types [0,1,1,..]
            edge_index=edges,  # edge ids [[0,1],[0,2],..]
            edge_attr=edges_lab,  # edge label ids
            action=torch.tensor(id_edit),
            nodes=torch.tensor(len(G)),
            sequence=sN,  # edit op connection points
            sequence_masked=sNM,
            len_seq=torch.tensor(len(pallete.get_special_nodes(id_edit))),
            finished=finished,
        )
        result.append(data)

    return result


def data2graph(data, pallete):
    G = nx.MultiDiGraph()
    vocab_nodes_2 = {y: x for x, y in pallete.dic_nodes.items()}
    vocab_edges_2 = {y: x for x, y in pallete.dic_edges.items()}
    for n, t in enumerate(data.x.numpy()):
        G.add_node(n, type=vocab_nodes_2[t])
    edges = torch.transpose(data.edge_index, 0, 1)
    for n, e in enumerate(edges.numpy()):
        s = e[0]
        t = e[1]
        n = vocab_edges_2[torch.squeeze(data.edge_attr[n]).item()]
        G.add_edge(s, t, type=n)
    return G
