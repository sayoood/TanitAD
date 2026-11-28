"""THE MISSING GUARD, written as a test and RUN — it goes RED on HEAD.

⛔ Every existing coupling-(1) test (`stack/tests/test_refcv6_perception.py`
:394-518) calls `BEVWaypointSampler` STANDALONE. None runs it through
`AnchoredDiffusionDecoder.forward`, which is where the three wiring breaks are.
This is the CONSUMER-side assertion that was missing. It is written so that it
is RED on HEAD and would go GREEN once `bev` is forwarded to `_sample` and to
the default `_decode` sites.

Arms:
  A  the sampler's denoising passes must see the BEV map      -> EXPECTED RED
  B  the default classifier pass must see the BEV map         -> EXPECTED RED
  C  CONTROL: a direct call fires the hook                    -> must be GREEN
  D  CONTROL: the prefilter branch fires the hook             -> must be GREEN
     (D is what proves A/B are about the WIRING, not the probe.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diag_decoder_couplings import build_decoder, make_inputs, Recorder
from tanitad.models import refcv6_diffusion as rv6

def count(dec, fn):
    rec = Recorder(); rec.attach(dec)
    try:
        with torch.no_grad():
            fn()
    finally:
        rows = [r for r in rec.rows if r["kind"] == "bev_wp"]; rec.remove()
    return len(rows)

def main():
    res = {}
    flags = rv6.DiffusionFlags(f1_random_t=True, f2_dd_step=True,
                               f3_per_layer=True, f4_adaln=True,
                               f5_emitting_conf=True)
    dec, params = build_decoder(flags=flags); dec.eval()
    inp = make_inputs(dec)
    kw = dict(steps=2, v_ms=inp["v_ms"], agent_tokens=inp["agent_tokens"],
              agent_pad=inp["agent_pad"], agent_pos=inp["agent_pos"],
              bev=inp["bev"])
    n_layers = len(dec.layers)
    n_passes = 2                                   # the [10, 0] ladder

    a = count(dec, lambda: dec(inp["fmap"], inp["m"], **kw))
    res["A_sampler_passes"] = {"expected_min": n_layers * n_passes,
                               "measured": a, "GREEN": a >= n_layers * n_passes}
    kw0 = dict(kw); kw0["steps"] = 0
    b = count(dec, lambda: dec(inp["fmap"], inp["m"], **kw0))
    res["B_classifier_pass"] = {"expected_min": n_layers, "measured": b,
                                "GREEN": b >= n_layers}
    q = torch.randn(inp["fmap"].shape[0], 6, dec.cfg.d)
    wp = torch.randn(inp["fmap"].shape[0], 6, 4, 2)
    c = count(dec, lambda: dec.layers[0].bev_wp(q, wp, inp["bev"]))
    res["C_direct_call_CONTROL"] = {"expected": 1, "measured": c, "GREEN": c == 1}
    with torch.no_grad():
        dec.anchor_controls[:, 0] = torch.linspace(-40, 40, dec.anchors.shape[0])
    dec.sel.anchor_prefilter = True
    kwd = dict(kw); kwd["v_ms"] = torch.full_like(inp["v_ms"], 0.05)
    d = count(dec, lambda: dec(inp["fmap"], inp["m"], **kwd))
    dec.sel.anchor_prefilter = False
    res["D_prefilter_branch_CONTROL"] = {"expected_min": n_layers,
                                         "measured": d, "GREEN": d >= n_layers}
    res["bev_coupling_params_built"] = params["bev_coupling_params"]
    res["VERDICT"] = ("GUARD IS RED ON HEAD (the defect is real)"
                      if not (res["A_sampler_passes"]["GREEN"]
                              and res["B_classifier_pass"]["GREEN"])
                      and res["C_direct_call_CONTROL"]["GREEN"]
                      and res["D_prefilter_branch_CONTROL"]["GREEN"]
                      else "INCONCLUSIVE — read the arms")
    txt = json.dumps(res, indent=2); print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "guard_coupling1.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
