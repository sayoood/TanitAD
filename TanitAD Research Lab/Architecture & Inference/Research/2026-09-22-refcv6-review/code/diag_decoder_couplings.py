"""refcv6 decoder review — MEASURE the three couplings of SPEC_REFCV6_V2 §1.

Advisory class C: *"put a forward hook on every cross-attention and PRINT THE
CONTEXT SHAPE. If two modules are meant to read different tensors, assert the
storage pointers differ. Do not infer the wiring from the code; measure it."*

This builds a FULLY WIRED `AnchoredDiffusionDecoder` — agent cross-attention on,
WP-B waypoint index attached, refcv6 BEV coupling attached — and runs it in both
modes (classifier `steps=0`, sampler `--sampler ddim`, `steps>0`). Every
`nn.MultiheadAttention` and every `BEVWaypointSampler` is hooked; the hook
records the CONTEXT tensor's shape and `data_ptr()`.

Run:
  PYTHONPATH=D:/Projects/TanitAD/stack python diag_decoder_couplings.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from torch import nn

from tanitad.data.bev_raster import GRID_DEFAULT as _GRID
from tanitad.models import refc_bev_coupling as bevc
from tanitad.models import refcv6_diffusion as rv6
from tanitad.refs import refc
from tanitad.refs import refc_wp_index as wpi

B, N_ANCHOR, FEAT, DMEAS, DCTX = 2, 6, 16, 8, 4
HORIZONS = (5, 10, 15, 20)
S = len(HORIZONS)
GRID = _GRID
D_BEV = 12


def build_decoder(*, flags=None, sampler="ddim", bev=True, wpb=True,
                  agents=True, seed=0, layers=2, d=32):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=d, n_heads=4, layers=layers, ff_mult=2,
                             sampler=sampler, refcv6=flags,
                             cross_agent=bool(agents))
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=FEAT, n_steps=S, d_meas=DMEAS, d_ctx=DCTX, tac_latent_dim=4,
        anchors=torch.randn(N_ANCHOR, S, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=HORIZONS, v0_conditioned=True,
        control_units="alat")
    ctrl = torch.stack([torch.linspace(-2.0, 2.0, N_ANCHOR),
                        torch.linspace(-1.5, 1.5, N_ANCHOR)], dim=-1)
    dec.anchor_controls.copy_(ctrl)
    n_wpb = n_bev = 0
    if wpb and agents:
        n_wpb = dec.attach_wp_index(wpi.WaypointIndexConfig(enable=True))
    if bev:
        n_bev = dec.attach_bev_coupling(
            bevc.BEVCouplingConfig(enable=True, d_model=d, d_bev=D_BEV,
                                   n_points=S), D_BEV)
    return dec, {"wp_index_params": n_wpb, "bev_coupling_params": n_bev}


def make_inputs(dec, *, grid_px=4):
    torch.manual_seed(11)
    fmap = torch.randn(B, FEAT, grid_px, grid_px)
    m = torch.randn(B, DMEAS)
    agent_tokens = torch.randn(B, 5, dec.cfg.d)
    agent_pad = torch.zeros(B, 5, dtype=torch.bool)
    agent_pos = torch.randn(B, 5, 2) * 5.0
    bev = torch.randn(B, D_BEV, *GRID.shape)
    v_ms = torch.full((B,), 8.0)
    return dict(fmap=fmap, m=m, agent_tokens=agent_tokens,
                agent_pad=agent_pad, agent_pos=agent_pos, bev=bev, v_ms=v_ms)


class Recorder:
    def __init__(self):
        self.rows: list[dict] = []
        self.handles: list = []

    def attach(self, dec):
        for li, layer in enumerate(dec.layers):
            self._mha(f"layer{li}.cross[IMAGE]", layer, "cross", li)
            if layer.cross_agent is not None:
                self._mha(f"layer{li}.cross_agent[AGENT]", layer,
                          "cross_agent", li)
            if layer.bev_wp is not None:
                self._bev(f"layer{li}.bev_wp[BEV]", layer.bev_wp, li)

    def _mha(self, name, layer, attr, li):
        mod = getattr(layer, attr)

        def hook(_m, args, kwargs, _out, _name=name, _li=li, _kind=attr):
            q = args[0] if len(args) > 0 else kwargs.get("query")
            k = args[1] if len(args) > 1 else kwargs.get("key")
            v = args[2] if len(args) > 2 else kwargs.get("value")
            am = kwargs.get("attn_mask")
            self.rows.append({
                "site": _name, "layer": _li, "kind": _kind,
                "q_shape": list(q.shape), "ctx_shape": list(k.shape),
                "ctx_tokens": int(k.shape[1]),
                "q_ptr": int(q.data_ptr()), "k_ptr": int(k.data_ptr()),
                "v_ptr": int(v.data_ptr()),
                "k_is_v": bool(k.data_ptr() == v.data_ptr()),
                "attn_mask": None if am is None else list(am.shape),
            })
        self.handles.append(mod.register_forward_hook(hook, with_kwargs=True))

    def _bev(self, name, mod, li):
        def hook(_m, args, _out, _name=name, _li=li):
            q, wp, bev = args[0], args[1], args[2]
            self.rows.append({
                "site": _name, "layer": _li, "kind": "bev_wp",
                "q_shape": list(q.shape),
                "ctx_shape": None if bev is None else list(bev.shape),
                "wp_shape": None if wp is None else list(wp.shape),
                "q_ptr": int(q.data_ptr()),
                "k_ptr": None if bev is None else int(bev.data_ptr()),
                "wp_ptr": None if wp is None else int(wp.data_ptr()),
                "bev_is_none": bev is None,
            })
        self.handles.append(mod.register_forward_hook(hook))

    def clear(self):
        self.rows = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run(dec, inp, *, steps, pass_bev):
    rec = Recorder()
    rec.attach(dec)
    dec.eval()
    kw = dict(steps=steps, v_ms=inp["v_ms"],
              agent_tokens=inp["agent_tokens"], agent_pad=inp["agent_pad"],
              agent_pos=inp["agent_pos"])
    if pass_bev:
        kw["bev"] = inp["bev"]
    with torch.no_grad():
        out = dec(inp["fmap"], inp["m"], **kw)
    rows = list(rec.rows)
    rec.remove()
    return rows, out


def summarise(rows):
    agg = {}
    for r in rows:
        k = r["site"]
        a = agg.setdefault(k, {"calls": 0, "ctx_shapes": set(),
                               "q_shapes": set(), "kind": r["kind"],
                               "k_ptrs": set(), "bev_none": 0})
        a["calls"] += 1
        a["ctx_shapes"].add(tuple(r["ctx_shape"]) if r["ctx_shape"] else None)
        a["q_shapes"].add(tuple(r["q_shape"]))
        if r.get("k_ptr") is not None:
            a["k_ptrs"].add(r["k_ptr"])
        if r.get("bev_is_none"):
            a["bev_none"] += 1
    return {k: {"calls": v["calls"], "kind": v["kind"],
                "ctx_shapes": sorted(str(s) for s in v["ctx_shapes"]),
                "q_shapes": sorted(str(s) for s in v["q_shapes"]),
                "n_distinct_ctx_ptrs": len(v["k_ptrs"]),
                "bev_none_calls": v["bev_none"]}
            for k, v in agg.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report: dict = {}

    flags = rv6.DiffusionFlags(
        f1_random_t=True, f2_dd_step=True, f3_per_layer=True, f4_adaln=True,
        f5_emitting_conf=True, f5_focal=True, f6_w_u0_zero=True,
        f8_flat_waypoint_noise=False, f9_assert_vocab=False)

    for tag, kw in (("all_couplings_attached", {}),):
        dec, params = build_decoder(flags=flags, **kw)
        inp = make_inputs(dec)
        report["params"] = params
        # --- ARM 1: caller PASSES bev (the design intent) ------------------- #
        rows_s, out_s = run(dec, inp, steps=2, pass_bev=True)
        report["sampler_bev_passed"] = summarise(rows_s)
        rows_c, out_c = run(dec, inp, steps=0, pass_bev=True)
        report["classifier_bev_passed"] = summarise(rows_c)
        # --- ARM 2: caller does NOT pass bev (what refc_v3.py actually does)  #
        rows_s0, _ = run(dec, inp, steps=2, pass_bev=False)
        report["sampler_bev_absent"] = summarise(rows_s0)
        # --- pointer separation check (class C) ----------------------------- #
        ptr = {}
        for r in rows_s:
            ptr.setdefault(r["kind"], set()).add(r.get("k_ptr"))
        report["context_ptrs_by_kind"] = {k: sorted(str(x) for x in v)
                                          for k, v in ptr.items()}
        report["ptr_disjoint_image_vs_agent"] = bool(
            not (ptr.get("cross", set()) & ptr.get("cross_agent", set())))
        report["ptr_disjoint_image_vs_bev"] = bool(
            not (ptr.get("cross", set()) & ptr.get("bev_wp", set())))
        report["out_keys"] = sorted(out_s.keys())
        report["sampler_tele"] = {
            k: v for k, v in (out_s.get("sel_tele") or {}).items()
            if not torch.is_tensor(v)} if isinstance(
                out_s.get("sel_tele"), dict) else None
    txt = json.dumps(report, indent=2, default=str)
    print(txt)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
