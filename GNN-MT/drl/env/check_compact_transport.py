#!/usr/bin/env python3
"""Compare legacy XMI observations with compact resetAll/stepAll payloads.

Usage (with the CRA env server already listening on --port):

    python -m drl.env.check_compact_transport \\
        --metamodel data/cra.ecore --root_object ClassModel \\
        --case_config datasets/cra/config.json --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import Counter
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import procbridge as pb

from drl.conversion import ConverterUtility
from drl.env.compact_obs import decode_compact_obs
from drl.env.env_util import generate_tensors_from_graph
from drl.env.pb_client import PersistentClient
from drl.model2graph.model2graph import get_graph_from_resource
from drl.utils.pallete import Pallete, SEP_INV, SPECIAL, add_inv_edges


def load_pallete(args):
    with open(args.case_config) as f:
        conf = json.load(f)
    conf_ns = SimpleNamespace(
        unitPrefix=conf["unitPrefix"],
        nodeIdentifierAttributeName=conf["nodeIdentifierAttributeName"],
        maxNodeParams=conf["maxNodeParams"],
        maxValueParams=conf["maxValueParams"],
        units=conf["units"],
        additionalNodeEncodingAttributes=conf.get(
            "additionalNodeEncodingAttributes", []
        ),
    )
    return Pallete(args.metamodel, args.root_object, conf_ns)


def build_observation_spec(pallete):
    edge_types = {
        name: idx
        for name, idx in pallete.dic_edges.items()
        if not name.endswith(SEP_INV) and name != SPECIAL
    }
    return {
        "nodeTypes": dict(pallete.dic_nodes),
        "edgeTypes": edge_types,
        "idAttribute": pallete.get_node_id_attr_name(),
        "numericAttributes": list(pallete.additional_node_atts or []),
    }


def parse_rule_conditions(cond):
    if not cond:
        return None
    inner = cond.get("conditions", cond) if isinstance(cond, dict) else cond
    if not isinstance(inner, dict):
        return None
    return {int(k) if str(k).isdigit() else k: v for k, v in inner.items()}


def xmi_to_graph_data(conv, pallete, xmi, env_idx, conditions):
    resource = conv.resource_from_xmi_str("checkuri" + str(env_idx), xmi)
    graph = add_inv_edges(pallete.remove_out_of_scope(get_graph_from_resource(resource)))
    data = generate_tensors_from_graph(graph, pallete, conditions)
    return graph, data


def summarize_observation(data):
    x = data.x[:, 0].detach().cpu().numpy().astype(np.int64)
    ei = data.edge_index.detach().cpu().numpy()
    et = data.edge_attr.view(-1).detach().cpu().numpy().astype(np.int64)
    src_types = x[ei[0]]
    dst_types = x[ei[1]]
    triples = Counter(zip(src_types.tolist(), dst_types.tolist(), et.tolist()))
    return {
        "n": int(data.nodes.item()) if hasattr(data, "nodes") else int(x.shape[0]),
        "type_hist": Counter(x.tolist()),
        "triples": triples,
        "action_mask_sum": int(data.action_mask.sum().item()),
        "action_node_mask_sum": int(data.action_node_mask.sum().item()),
    }


def compare_observations(label, legacy, compact):
    ok = True
    for key in ("n", "type_hist", "triples", "action_mask_sum", "action_node_mask_sum"):
        if legacy[key] != compact[key]:
            print(f"MISMATCH {label} {key}:")
            print(f"  legacy : {legacy[key]}")
            print(f"  compact: {compact[key]}")
            ok = False
    if ok:
        print(f"OK {label}: n={legacy['n']} types={dict(legacy['type_hist'])} "
              f"edges={sum(legacy['triples'].values())} "
              f"mask={legacy['action_mask_sum']}/{legacy['action_node_mask_sum']}")
    return ok


def port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metamodel", default="data/cra.ecore")
    parser.add_argument("--root_object", default="ClassModel")
    parser.add_argument("--case_config", default="datasets/cra/config.json")
    parser.add_argument("--env_id", type=int, default=0)
    args = parser.parse_args()

    if not port_is_open(args.host, args.port):
        print(f"Server not listening on {args.host}:{args.port}; skip live checks.")
        return 0

    pallete = load_pallete(args)
    conv = ConverterUtility(args.metamodel)
    oneshot = pb.Client(args.host, args.port)
    echo = oneshot.request("echo", 1)
    if echo != 1:
        print(f"FAIL echo: expected 1 got {echo}")
        return 1
    print("OK one-shot echo still works against the looping server")

    client = PersistentClient(args.host, args.port)
    client.request("configureObservation", build_observation_spec(pallete))

    reset_xmi = oneshot.request("reset", args.env_id)
    conditions = parse_rule_conditions(reset_xmi[1])
    _, legacy_data = xmi_to_graph_data(
        conv, pallete, reset_xmi[0], args.env_id, conditions
    )

    reset_compact = client.request("resetAll")
    row = reset_compact[args.env_id]
    compact_data, _nids = decode_compact_obs(
        row[0], pallete, parse_rule_conditions(row[1])
    )

    ok = compare_observations(
        "reset", summarize_observation(legacy_data), summarize_observation(compact_data)
    )

    # Failed application
    fail = client.request(
        "stepAll",
        {
            "_autoReset": False,
            "_actions": [
                {
                    "_envId": args.env_id,
                    "_ruleName": pallete.get_unitname_by_id(0) + "__invalid",
                    "_paramValues": {},
                }
            ],
        },
    )
    fail_row = fail[0]
    if fail_row[0] is not False:
        print(f"FAIL invalid rule obs: expected False got {type(fail_row[0])}")
        ok = False
    else:
        print(f"OK invalid rule -> obs=False reward={fail_row[1]} done={fail_row[2]}")

    skip = client.request(
        "stepAll",
        {
            "_autoReset": False,
            "_actions": [{"_envId": args.env_id, "_skip": True}],
        },
    )
    skip_row = skip[0]
    if skip_row[0] is not None or skip_row[1] != 0 or skip_row[2]:
        print(f"FAIL skip row: {skip_row}")
        ok = False
    else:
        print("OK skip -> [null, 0, false]")

    client.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
