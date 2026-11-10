"""obd_geom.py -- per-window geometry rows for the OBD read (CPU, zero training, 2 torch threads).  Needs torch and the OTBAND read tree.

The lead join, the manoeuvre labels and the per-arm band rows are theory/otband/read/read_otband.py's OWN code: this module parses that file, takes the top-level
nodes `sha`, `join`, `labels`, `arm_arrays` (functions) and the lead-block constants / loads, and executes exactly those nodes in a namespace.  No line is copied, so the
read cannot drift from the OTBAND read; test_obd_read.py asserts the extraction finds exactly this node set and smoke_obd_read.py reproduces READ_FULL.json from it.
read_lib is the OTBAND read's (band_lib.py pointed at the OTBAND export F:/TanitAD-scratch/otband_read/tree, which holds the same stack/tanitad and taniteval as the
code of record cdbff971: README_READ.md, "Tree claim").
"""
import ast
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(os.path.dirname(HERE))
OTB = os.path.join(THEORY, "otband", "read")
SRC_PATH = os.path.join(OTB, "read_otband.py")
sys.path.insert(0, OTB)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import read_lib as BL  # noqa: E402
from read_lib import CI, LM, FF, DT  # noqa: E402

torch.set_num_threads(2)
NB, SEED = 2000, 0
FUNCS = ("sha", "join", "labels", "arm_arrays")
CONSTS = ("ART", "LEAD", "LEAD_SHA", "RAW", "blk", "idx", "clip_of")
ASSERT_MARKS = ("lead block sha mismatch", "duplicate lead rows")
BANDS = {k: v for k, v in BL.BANDS.items() if k != "full_0_6s"}          # the card's two bands: operative knots 1-10, tactical knots 11-30
METRICS = tuple(BL.METRICS) + ("dk_hw_abs_err_m", "dk_tg_abs_err_s")


def extract_nodes(src):
    """-> (kept top-level nodes of read_otband.py in source order, names found).  Pure ast; no execution."""
    tree = ast.parse(src)
    keep, found = [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            keep.append(node), found.append(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTS:
            keep.append(node), found.append(node.targets[0].id)
        elif isinstance(node, ast.Assert) and any(m in (ast.get_source_segment(src, node) or "") for m in ASSERT_MARKS):
            keep.append(node), found.append("assert:" + next(m for m in ASSERT_MARKS if m in ast.get_source_segment(src, node)))
    return keep, found


def build():
    src = open(SRC_PATH, encoding="utf-8").read()
    keep, found = extract_nodes(src)
    ns = {"np": np, "torch": torch, "BL": BL, "LM": LM, "FF": FF, "DT": DT, "CI": CI, "NB": NB, "SEED": SEED, "__name__": "obd_geom_extract"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC_PATH, "exec"), ns)
    g = types.SimpleNamespace(join=ns["join"], labels=ns["labels"], arm_arrays=ns["arm_arrays"], found=found, ns=ns, sha=ns["sha"])
    return g
