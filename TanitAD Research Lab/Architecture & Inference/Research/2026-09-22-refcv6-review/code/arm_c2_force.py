"""C2 sharpened: FORCE the anchor-prefilter branch (refc.py:2866) to be taken,
so the positive control lives INSIDE `AnchoredDiffusionDecoder.forward`."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diag_decoder_couplings import build_decoder, make_inputs, Recorder
from tanitad.models import refcv6_diffusion as rv6

def fires(dec, **kw):
    rec = Recorder(); rec.attach(dec)
    with torch.no_grad():
        dec(**kw)
    rows = [r for r in rec.rows if r["kind"] == "bev_wp"]
    rec.remove()
    return len(rows), rows

def main():
    flags = rv6.DiffusionFlags(f1_random_t=True, f2_dd_step=True,
                               f3_per_layer=True, f4_adaln=True)
    dec, _ = build_decoder(flags=flags); dec.eval()
    inp = make_inputs(dec)
    out = {}
    # widen the anchor controls so most are UNREACHABLE at a low speed
    with torch.no_grad():
        dec.anchor_controls[:, 0] = torch.linspace(-40.0, 40.0, dec.anchors.shape[0])
    dec.sel.anchor_prefilter = True
    for v in (0.05, 1.0, 8.0):
        kw = dict(fmap=inp["fmap"], m=inp["m"], steps=2,
                  v_ms=torch.full((inp["fmap"].shape[0],), v),
                  agent_tokens=inp["agent_tokens"], agent_pad=inp["agent_pad"],
                  agent_pos=inp["agent_pos"], bev=inp["bev"])
        n, rows = fires(dec, **kw)
        out[f"prefilter_v{v}"] = {"bev_wp_fires": n,
                                  "ctx_shape": rows[0]["ctx_shape"] if rows else None}
    dec.sel.anchor_prefilter = False
    kw["v_ms"] = torch.full((inp["fmap"].shape[0],), 0.05)
    n, _ = fires(dec, **kw)
    out["prefilter_OFF_same_inputs"] = {"bev_wp_fires": n}
    txt = json.dumps(out, indent=2)
    print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "bev_c2_forced.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
