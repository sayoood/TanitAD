"""POSITIVE CONTROLS for the coupling-(1) measurement.

⛔ A hook that fires ZERO times is indistinguishable from a hook that was never
installed. These three arms discriminate:

  C1  direct call of `layer.bev_wp(q, wp, bev)`  -> the hook MUST fire (proves
      the instrument works at all).
  C2  decoder forward with `sel.anchor_prefilter=True` -> refc.py:2866 is the
      ONE `_decode` site that forwards `bev`, so the hook MUST fire there.
  C3  decoder forward on the default path       -> the measurement under test.

Plus an argv/AST reachability audit of `bev_coupling` and of `bev=`.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import torch

from diag_decoder_couplings import (D_BEV, GRID, build_decoder, make_inputs,
                                    Recorder)
from tanitad.models import refcv6_diffusion as rv6
from tanitad.refs import refc

ROOT = Path("D:/Projects/TanitAD")


def fire_count(dec, fn):
    rec = Recorder()
    rec.attach(dec)
    try:
        fn()
    finally:
        rows = [r for r in rec.rows if r["kind"] == "bev_wp"]
        rec.remove()
    return len(rows), rows


def main():
    out = {}
    flags = rv6.DiffusionFlags(f1_random_t=True, f2_dd_step=True,
                               f3_per_layer=True, f4_adaln=True,
                               f5_emitting_conf=True)
    dec, params = build_decoder(flags=flags)
    inp = make_inputs(dec)
    dec.eval()
    out["attached_bev_params"] = params["bev_coupling_params"]
    out["n_layers_with_bev_wp"] = sum(
        1 for ly in dec.layers if ly.bev_wp is not None)

    # ---- C1: direct call --------------------------------------------------
    q = torch.randn(inp["fmap"].shape[0], 6, dec.cfg.d)
    wp = torch.randn(inp["fmap"].shape[0], 6, 4, 2)
    n, rows = fire_count(dec, lambda: dec.layers[0].bev_wp(q, wp, inp["bev"]))
    out["C1_direct_call_fires"] = n
    out["C1_rows"] = rows

    # ---- C2: prefilter branch (the one site that forwards bev) ------------
    dec.sel.anchor_prefilter = True
    n2, rows2 = fire_count(dec, lambda: dec(
        inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"],
        agent_tokens=inp["agent_tokens"], agent_pad=inp["agent_pad"],
        agent_pos=inp["agent_pos"], bev=inp["bev"]))
    out["C2_prefilter_fires"] = n2
    out["C2_rows"] = rows2[:4]
    dec.sel.anchor_prefilter = False

    # ---- C3: the default path (the measurement under test) ----------------
    n3, _ = fire_count(dec, lambda: dec(
        inp["fmap"], inp["m"], steps=2, v_ms=inp["v_ms"],
        agent_tokens=inp["agent_tokens"], agent_pad=inp["agent_pad"],
        agent_pos=inp["agent_pos"], bev=inp["bev"]))
    out["C3_default_sampler_fires"] = n3
    n4, _ = fire_count(dec, lambda: dec(
        inp["fmap"], inp["m"], steps=0, v_ms=inp["v_ms"],
        agent_tokens=inp["agent_tokens"], agent_pad=inp["agent_pad"],
        agent_pos=inp["agent_pos"], bev=inp["bev"]))
    out["C3_default_classifier_fires"] = n4

    # ---- AST: which `_decode`/`_sample` call sites carry `bev` ------------
    src = (ROOT / "stack/tanitad/refs/refc.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr not in ("_decode",
                                                              "_decode_ctrl",
                                                              "_sample"):
            continue
        names = [a.id for a in node.args if isinstance(a, ast.Name)]
        kw = [k.arg for k in node.keywords]
        sites.append({"line": node.lineno, "callee": f.attr,
                      "positional_names": names, "kwargs": kw,
                      "n_positional": len(node.args),
                      "carries_bev": ("bev" in names or "bev" in kw)})
    out["decode_call_sites"] = sorted(sites, key=lambda r: r["line"])
    out["n_sites_carrying_bev"] = sum(1 for s in sites if s["carries_bev"])
    out["n_sites_total"] = len(sites)

    # ---- AST: does ANY caller pass `bev=` into RefCModel.forward? ---------
    hits = []
    for p in sorted(ROOT.glob("stack/**/*.py")):
        if ".claude" in str(p):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Call) and any(
                    k.arg == "bev" for k in node.keywords):
                hits.append({"file": str(p.relative_to(ROOT)),
                             "line": node.lineno,
                             "callee": ast.unparse(node.func)[:60]})
    out["callers_passing_bev_kwarg"] = hits

    # ---- AST: does anything assign cfg...bev_coupling = ? -----------------
    assigns = []
    for p in sorted(ROOT.glob("**/*.py")):
        if ".claude" in str(p) or ".git" in str(p):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "bev_coupling" not in txt:
            continue
        try:
            t = ast.parse(txt)
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and tgt.attr == "bev_coupling"):
                        assigns.append({"file": str(p.relative_to(ROOT)),
                                        "line": node.lineno,
                                        "target": ast.unparse(tgt)})
    out["bev_coupling_assignments"] = assigns

    txt = json.dumps(out, indent=2, default=str)
    print(txt)
    (Path(__file__).resolve().parents[1] / "raw"
     / "bev_coupling_controls.json").write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
