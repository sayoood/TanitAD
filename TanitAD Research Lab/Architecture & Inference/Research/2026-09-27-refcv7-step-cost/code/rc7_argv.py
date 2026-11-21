#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 step-cost harness -- the ARGV side.

Builds the argv one ablation rung runs with, from the CANONICAL launch argv
(`stack/ops/runs.d/refcv7-r101-s0.argv.json`, 154 tokens) in the clean tree:

  1. every data path is mapped to this host through a path map (the launch gate's own
     `parse_path_map` / `map_path` / `flag_pairs`, imported from the tree -- one spelling);
  2. the rung's edits are applied (each edit switches ONE refcv7 addition off, or moves one
     lever), expressed as flag operations on the canonical argv, never as a hand-typed argv;
  3. the DEV-BOX DEPARTURES are applied last and returned separately so every run record
     states them: --batch, --steps, --out, --workers, --log-every, --eval-every,
     --save-every, and `--trunk-compile` DROPPED (no Triton on this Windows box).

`R6` is derived from the refcv7 argv by REVERSING `changes_vs_refcv6` (every entry: a flag
refcv6 did not pass is dropped, a changed value restored, and the no-flag R4 lever -- 300
queries -- becomes `--agent-queries 100`). `check_r6_against(config)` compares the result with
the refcv6 run's own `config.json` argv, so the reversal is verified, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

# the refcv7 additions, grouped as the brief lists them. Each rung = a list of edits:
#   ("set", flag, [values])  -> exactly one `flag values` (appended when absent)
#   ("drop", flag)           -> flag removed
MAP_HIRES_FLAGS = ("--map-hires", "--w-map-hires", "--map-hires-class-weights",
                   "--map-hires-decision-rule", "--map-hires-x-max-m",
                   "--map-hires-y-half-m", "--map-hires-grad-ckpt",
                   "--map-hires-near-lift-m", "--map-hires-near-refine-blocks",
                   "--bev-source", "--bev-planner-crop-m")
PRIOR_FLAGS = ("--residual-prior", "--graft-tac8-prior", "--graft-nav-compliance",
               "--nav-compliance-tau-rad", "--nav-compliance-tau-file",
               "--speed-ceiling-filter")

BOX_V6 = [("set", "--agent-queries", ["100"]),
          ("drop", "--slot-deep-supervision"),
          ("set", "--slot-presence-loss", ["bce"]), ("drop", "--slot-presence-prior"),
          ("drop", "--slot-vis1"), ("drop", "--vis1-sidecar"),
          ("drop", "--slot-query-select")]
MAP_V6 = ([("drop", f) for f in MAP_HIRES_FLAGS]
          + [("set", "--w-map", ["1.0"]),
             ("set", "--map-gt-root", ["/home/nvidia/data/sam3_corpus"])])

RUNGS: dict[str, list] = {
    # the launch argv itself
    "R7": [],
    # ---- levers (numerics-preserving candidates) ----
    "ckpt_off": [("set", "--map-hires-grad-ckpt", ["off"])],
    "conflict_off": [("set", "--conflict-detector", ["off"])],
    # ---- one refcv7 addition off at a time (attribution) ----
    "q100": [("set", "--agent-queries", ["100"])],
    "no_deepsup": [("drop", "--slot-deep-supervision")],
    "bce": [("set", "--slot-presence-loss", ["bce"]), ("drop", "--slot-presence-prior")],
    "no_vis1": [("drop", "--slot-vis1"), ("drop", "--vis1-sidecar")],
    "learned": [("drop", "--slot-query-select")],
    "box_v6": BOX_V6,
    "no_near": [("drop", "--map-hires-near-lift-m"),
                ("drop", "--map-hires-near-refine-blocks")],
    "no_priors": [("drop", f) for f in PRIOR_FLAGS],
    "map_v6": MAP_V6,
    # refcv6 = every change reversed (built by `r6_edits`, see below)
    "R6": None,
}


def _load_tree_gate(tree: str):
    import sys
    p = str(Path(tree) / "stack" / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import launch_gate as LG           # noqa: E402  (the gate's argv helpers, one spelling)
    return LG


def canonical(tree: str) -> dict:
    return json.loads((Path(tree) / "stack" / "ops" / "runs.d" /
                       "refcv7-r101-s0.argv.json").read_text(encoding="utf-8"))


def r6_edits(doc: dict) -> list:
    """`changes_vs_refcv6` reversed -> edits that turn the refcv7 argv into refcv6's."""
    ed = []
    for c in doc["changes_vs_refcv6"]:
        f = c.get("flag")
        if f is None:                                  # R4: the code default 300 -> 100
            ed.append(("set", "--agent-queries", [str(c["refcv6"])]))
            continue
        if f == "--out":
            continue
        if c.get("refcv6") is None:
            ed.append(("drop", f))
        else:
            ed.append(("set", f, [str(v) for v in c["refcv6"]]))
    return ed


def apply_edits(LG, argv: list, edits: list) -> list:
    out = list(argv)
    for e in edits:
        if e[0] == "set":
            out = LG.set_flag(out, e[1], list(e[2]))
        elif e[0] == "drop":
            out = LG.set_flag(out, e[1], None)
        else:
            raise ValueError(e)
    return out


def read_pathmap(path: str) -> list:
    lines = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l and not l.startswith("#"):
            lines.append(l)
    return lines


def map_argv(LG, argv: list, pmap: list) -> list:
    out = []
    for f, vals in LG.flag_pairs(argv):
        out.append(f)
        out += [LG.map_path(v, pmap) if (f != "--out" and LG._looks_like_path(v)) else v
                for v in vals]
    return out


def devbox_departures(batch: int, steps: int, out: str, workers: int, log_every: int,
                      eval_every: int, save_every: int, prefetch: int | None = None,
                      keep_compile: bool = False) -> list:
    ed = [("set", "--batch", [str(batch)]), ("set", "--steps", [str(steps)]),
          ("set", "--out", [out]), ("set", "--workers", [str(workers)]),
          ("set", "--log-every", [str(log_every)]), ("set", "--eval-every", [str(eval_every)]),
          ("set", "--save-every", [str(save_every)])]
    if not keep_compile:
        # no Triton on this Windows box -> Inductor cannot run; the trunk is compiled on
        # Thor in BOTH refcv6 and refcv7, so the delta attribution does not need it.
        ed.append(("drop", "--trunk-compile"))
    if prefetch is not None:
        ed.append(("set", "--prefetch-factor", [str(prefetch)]))
    return ed


def build(tree: str, rung: str, pathmap: str, *, batch: int, steps: int, out: str,
          workers: int, log_every: int, eval_every: int, save_every: int,
          extra_edits: list | None = None, prefetch: int | None = None,
          keep_compile: bool = False) -> dict:
    LG = _load_tree_gate(tree)
    doc = canonical(tree)
    base = list(doc["argv"])
    if len(base) != 154:
        raise SystemExit(f"canonical argv has {len(base)} tokens, expected 154")
    edits = r6_edits(doc) if rung == "R6" else list(RUNGS[rung])
    edits += list(extra_edits or [])
    rung_argv = apply_edits(LG, base, edits)
    dep = devbox_departures(batch, steps, out, workers, log_every, eval_every, save_every,
                            prefetch, keep_compile)
    final = apply_edits(LG, rung_argv, dep)
    pmap = LG.parse_path_map(read_pathmap(pathmap))
    mapped = map_argv(LG, final, pmap)
    return {"rung": rung, "edits": edits, "devbox_departures": dep,
            "argv_thor_form": final, "argv": mapped, "n_tokens_canonical": len(base),
            "pathmap": [list(x) for x in pmap]}


def check_r6_against(tree: str, config_json: str) -> dict:
    """Compare the derived R6 argv with the refcv6 run's own config.json argv (flag multiset)."""
    LG = _load_tree_gate(tree)
    doc = canonical(tree)
    r6 = apply_edits(LG, list(doc["argv"]), r6_edits(doc))
    ref = json.loads(Path(config_json).read_text(encoding="utf-8"))["argv"]
    a = {f: v for f, v in LG.flag_pairs(r6) if f != "--out"}
    b = {f: v for f, v in LG.flag_pairs(ref) if f != "--out"}
    extra_in_r6 = {f: a[f] for f in a if f not in b}
    missing_in_r6 = {f: b[f] for f in b if f not in a}
    diff_vals = {f: (a[f], b[f]) for f in a if f in b and a[f] != b[f]}
    return {"n_r6": len(r6), "n_ref": len(ref), "extra_in_r6": extra_in_r6,
            "missing_in_r6": missing_in_r6, "value_diffs": diff_vals,
            "order_equal_ignoring_out": [t for t in r6 if True] == [t for t in ref if True]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--check-r6", default=None, help="refcv6 config.json to verify R6 against")
    a = ap.parse_args()
    if a.check_r6:
        print(json.dumps(check_r6_against(a.tree, a.check_r6), indent=1))
