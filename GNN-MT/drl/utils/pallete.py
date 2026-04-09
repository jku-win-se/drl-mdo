import json

import networkx as nx
import numpy as np
from drl.utils.edit_operation_generation import get_edge_node_dic

SEP_INV = "_inv"
SPECIAL = "<special>"


def node_match(n1, n2):
    return n1["type"] == n2["type"]


def node_match_list(n1, n2):
    ty1 = n1["type"]
    ty2 = n2["type"]
    if isinstance(ty1, list) and isinstance(ty2, list):
        return len([t for t in ty1 if t in ty2]) > 0
    elif isinstance(ty1, list):
        return ty2 in ty1
    elif isinstance(ty2, list):
        return ty1 in ty2
    else:
        return ty1 == ty2


def add_inv_edges(G):
    G_new = nx.MultiDiGraph(G)
    for e in list(G.edges.data()):
        G_new.add_edge(e[1], e[0], type=e[2]["type"] + SEP_INV)
    return G_new


def remove_inv_edges(G):
    G_new = nx.MultiDiGraph(G)
    remove = []
    for e in G.edges:
        if G[e[0]][e[1]][e[2]]["type"].endswith(SEP_INV):
            remove.append((e[0], e[1], e[2]))
    for s, t, k in remove:
        G_new.remove_edge(s, t, k)
    return G_new


def get_initial_graph(root_element):
    initial_graph = nx.MultiDiGraph()
    initial_graph.add_node(0, type=root_element)
    return initial_graph


class Pallete:

    def __init__(self, path_metamodel, root_element, conf):
        self.root_element = root_element
        self.path_metamodel = path_metamodel

        dic_edges, dic_nodes, class_hierarch_dict, class_ancestry_dict = (
            get_edge_node_dic(path_metamodel, root_element)
        )
        self.dic_nodes = dic_nodes
        self.dic_nodes_inv = {v: k for k, v in dic_nodes.items()}
        self.dic_edges = dic_edges
        self.class_hierarch_dict = class_hierarch_dict
        self.class_ancestry_dict = class_ancestry_dict

        self.max_len = conf.maxNodeParams
        self.max_val_len = conf.maxValueParams
        self.additional_node_atts = conf.additionalNodeEncodingAttributes
        self.units = conf.units
        self.unit_prefix = conf.unitPrefix
        self.node_id_attr_name = conf.nodeIdentifierAttributeName
        self.unitIdToSpec = {}
        self.unitIdToParamPosToName = {}
        self.unitIdToRequiredNodeTypes = {}

        # Process units in sorted order for deterministic behavior
        for unit_name in sorted(self.units.keys()):
            spec = self.units[unit_name]
            self.unitIdToSpec[int(spec["id"])] = spec
            self.unitIdToSpec[int(spec["id"])]["name"] = unit_name
            ## needed for action masking
            self.unitIdToRequiredNodeTypes[int(spec["id"])] = set(
                [
                    dic_nodes[node_param["validType"]]
                    for node_param in spec["nodeParams"]
                    if "validType" in node_param
                ]
            )

            paramPosToName = {}
            # Process params in deterministic order
            for p_spec in sorted(
                spec["nodeParams"] + spec["valueParams"], key=lambda x: x["position"]
            ):
                paramPosToName[p_spec["position"]] = p_spec["name"]
            self.unitIdToParamPosToName[int(spec["id"])] = paramPosToName

    def get_unitname_by_id(self, id):
        return self.unit_prefix + self.unitIdToSpec[id]["name"]

    def has_valid_types(self, action, pos):
        return "validType" in self.unitIdToSpec[action]["nodeParams"][pos]

    def has_range(self, action, pos):
        return "range" in self.unitIdToSpec[action]["valueParams"][pos]

    def has_attr_condition(self, action, pos):
        return "attr_condition" in self.unitIdToSpec[action]["nodeParams"][pos]

    def get_attr_conditions(self, action, pos):
        return (
            self.unitIdToSpec[action]["nodeParams"][pos]["attr_condition"]
            if self.has_attr_condition(action, pos)
            else {}
        )

    def has_relation_dependency(self, action, pos):
        return (
            "pred_relation_dependency" in self.unitIdToSpec[action]["nodeParams"][pos]
        )

    def has_relation_restriction(self, action, pos):
        return (
            "pred_relation_restriction" in self.unitIdToSpec[action]["nodeParams"][pos]
        )

    def get_relation_dependency(self, action, pos):
        return self.dic_edges[
            self.unitIdToSpec[action]["nodeParams"][pos]["pred_relation_dependency"]
        ]

    def get_relation_restriction(self, action, pos):
        return self.dic_edges[
            self.unitIdToSpec[action]["nodeParams"][pos]["pred_relation_restriction"]
        ]

    def get_required_and_forbidden_types(self, action, pos):
        req_forbidden_types = {}
        if "relRequired" in self.unitIdToSpec[action]["nodeParams"][pos]:
            req_forbidden_types["required"] = {}
            if "in" in self.unitIdToSpec[action]["nodeParams"][pos]["relRequired"]:
                req_forbidden_types["required"]["in"] = [
                    self.dic_edges[type_name]
                    for type_name in self.unitIdToSpec[action]["nodeParams"][pos][
                        "relRequired"
                    ]["in"]
                ]
            if "out" in self.unitIdToSpec[action]["nodeParams"][pos]["relRequired"]:
                req_forbidden_types["required"]["out"] = [
                    self.dic_edges[type_name]
                    for type_name in self.unitIdToSpec[action]["nodeParams"][pos][
                        "relRequired"
                    ]["out"]
                ]
        if "relForbidden" in self.unitIdToSpec[action]["nodeParams"][pos]:
            req_forbidden_types["forbidden"] = {}
            if "in" in self.unitIdToSpec[action]["nodeParams"][pos]["relForbidden"]:
                req_forbidden_types["forbidden"]["in"] = [
                    self.dic_edges[type_name]
                    for type_name in self.unitIdToSpec[action]["nodeParams"][pos][
                        "relForbidden"
                    ]["in"]
                ]
            if "out" in self.unitIdToSpec[action]["nodeParams"][pos]["relForbidden"]:
                req_forbidden_types["forbidden"]["out"] = [
                    self.dic_edges[type_name]
                    for type_name in self.unitIdToSpec[action]["nodeParams"][pos][
                        "relForbidden"
                    ]["out"]
                ]

        return req_forbidden_types

    def action_has_condition(self, action):
        return "conditions" in self.unitIdToSpec[action]

    def get_conditions(self, action):
        return {
            "elements": self.unitIdToSpec[action]["conditions"]["elements"],
            "relations": self.unitIdToSpec[action]["conditions"]["relations"],
        }

    def get_valid_node_types(self, action, pos):
        valid_node_type = self.unitIdToSpec[action]["nodeParams"][pos]["validType"]
        return self.class_hierarch_dict[valid_node_type]

    def get_num_params_for_action(self, discrete_values):
        # Convert dictionary to array-based mapping with sorted keys for determinism
        unique_vals = np.array(sorted(list(self.unitIdToSpec.keys())))
        seq_node_lengths = np.array(
            [len(self.unitIdToSpec[key]["nodeParams"]) for key in unique_vals]
        )
        seq_val_lengths = np.array(
            [len(self.unitIdToSpec[key]["valueParams"]) for key in unique_vals]
        )

        # Vectorized lookup
        mapped_node_lengths = seq_node_lengths[
            np.searchsorted(unique_vals, discrete_values)
        ]
        mapped_val_length = seq_val_lengths[
            np.searchsorted(unique_vals, discrete_values)
        ]

        return np.max(mapped_node_lengths), np.max(mapped_val_length)

    def get_num_value_params_for_action(self, action):
        return len(self.unitIdToSpec[action]["valueParams"])

    def get_num_node_params_for_action(self, action):
        return len(self.unitIdToSpec[action]["nodeParams"])

    def has_node_attr_conditions(self, action, pos):
        return "node_attr_condition" in self.unitIdToSpec[action]["valueParams"][pos]

    def get_node_attr_conditions(self, action, pos):
        return (
            self.unitIdToSpec[action]["valueParams"][pos]["node_attr_condition"]
            if self.has_node_attr_conditions(action, pos)
            else {}
        )

    def get_val_range_for_action(self, action, pos):
        return self.unitIdToSpec[action]["valueParams"][pos]["range"]

    def get_name_by_param_pos(self, action, pos):
        return self.unitIdToParamPosToName[action][pos]

    def get_node_id_attr_name(self):
        return self.node_id_attr_name

    def add_application_info_to_graph(self, G, application):
        G_new = nx.MultiDiGraph(G)

        for node_param in self.units[application["unitName"]]["nodeParams"]:
            for n in G_new:
                if (
                    G.nodes[n]["atts"][self.get_node_id_attr_name()]
                    == application["paramValues"][node_param["name"]]
                ):
                    G_new.nodes[n]["v_id"] = node_param["position"]

        return G_new

    def remove_out_of_scope(self, G, pattern=False):
        """Iterates over the nodes of G, and those nodes neighbours, and removes
        the corresponding edges if they are not part of the dictionary
        """
        G_new = nx.MultiDiGraph(G)

        edges_delete = []
        # Process nodes in sorted order for deterministic behavior
        for n in sorted(G_new.nodes()):
            for m in sorted(G_new[n].keys()):
                for e in sorted(G_new[n][m].keys()):
                    ty = G_new[n][m][e]["type"]
                    if ty not in self.dic_edges:
                        edges_delete.append((n, m, e))
        for a, b, e in edges_delete:
            G_new.remove_edge(a, b, e)

        if not pattern:
            nodes_delete = []
            for n in sorted(G_new.nodes()):
                if G_new.nodes[n]["type"] not in self.dic_nodes:
                    nodes_delete.append(n)
            for n in nodes_delete:
                G_new.remove_node(n)

        # relabel - ensure deterministic mapping
        new_map = {}
        j = 0
        for n in sorted(G_new.nodes()):
            new_map[n] = j
            j = j + 1
        G_new = nx.relabel_nodes(G_new, new_map)
        return G_new

    def from_json(self, dic):
        self.root_element = dic["root_element"]
        self.path_metamodel = dic["path_metamodel"]

        self.dic_nodes = dic["dic_nodes"]
        self.dic_nodes_inv = {v: k for k, v in self.dic_nodes.items()}
        self.dic_edges = dic["dic_edges"]
        self.class_hierarch_dict = dic["class_hierarch_dict"]
        self.class_ancestry_dict = dic["class_ancestry_dict"]

        self.max_len = dic["max_len"]
        self.max_val_len = dic["max_val_len"]

        self.additional_node_atts = dic["additional_node_atts"]
        self.unit_prefix = dic["unit_prefix"]
        self.node_id_attr_name = dic["node_id_attr_name"]
        self.units = dic["units"]

        self.unitIdToSpec = dic["unitIdToSpec"]
        self.unitIdToParamPosToName = dic["unitIdToParamPosToName"]
        # Make sure values are sets and cast key to int
        self.unitIdToRequiredNodeTypes = {
            int(k): set(v) for k, v in dic["unitIdToRequiredNodeTypes"].items()
        }
        self.unitIdToSpec = {int(k): v for k, v in self.unitIdToSpec.items()}
        # Cast key of unitIdToParamPosToName to int in nested fashion
        self.unitIdToParamPosToName = {
            int(k): {int(k2): v2 for k2, v2 in v.items()}
            for k, v in self.unitIdToParamPosToName.items()
        }


class PalleteEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Pallete):  # Your custom class

            return {
                "root_element": o.root_element,
                "path_metamodel": o.path_metamodel,
                "dic_nodes": o.dic_nodes,
                "dic_nodes_inv": o.dic_nodes_inv,
                "dic_edges": o.dic_edges,
                "max_len": o.max_len,
                "max_val_len": o.max_val_len,
                "additional_node_atts": o.additional_node_atts,
                "unit_prefix": o.unit_prefix,
                "node_id_attr_name": o.node_id_attr_name,
                "units": o.units,
                "unitIdToSpec": o.unitIdToSpec,
                "unitIdToParamPosToName": o.unitIdToParamPosToName,
                "unitIdToRequiredNodeTypes": o.unitIdToRequiredNodeTypes,
                "class_hierarch_dict": o.class_hierarch_dict,
                "class_ancestry_dict": o.class_ancestry_dict,
            }
        elif isinstance(o, set):  # Convert sets to lists
            return list(o)
        return super().default(o)


def is_in(G, pat):
    s_pat, t_pat, _ = list(pat.edges(data=True))[0]
    for s, t, d in G.edges(data=True):
        if node_match_list(G.nodes[s], pat.nodes[s_pat]) and node_match_list(
            G.nodes[t], pat.nodes[t_pat]
        ):
            return True
    return False
