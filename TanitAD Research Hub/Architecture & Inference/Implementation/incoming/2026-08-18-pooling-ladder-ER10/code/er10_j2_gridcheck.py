"""J2 — the PRODUCER→CONSUMER pin for the token grid (POOLING_BOTTLENECK §11).

C94's root-cause class is *a fixture that models the CONSUMER'S EXPECTATION
instead of the PRODUCER'S OUTPUT*. The ladder's whole claim rests on one
unstated belief: that `tokens[n]` in the banked cache sits at grid position
`(n // 40, n % 40)` and that `AvgPool2d((4,10))` over that reshape is the
DEPLOYED readout. A silent transpose would leave every number in the report
looking perfectly reasonable and mean something else.

TWO CHECKS, and neither is a hand-written literal:

  A. ⭐ THE REAL OPERATOR, END TO END. Take the checkpoint's OWN
     `readout.proj` weights and evaluate `proj(AvgPool2d((4,10))(tokens))` on
     BANKED tokens, then compare to the BANKED `cells` the producer wrote in
     the same forward pass. If the ordering or the kernel were wrong these do
     not agree. (Residual is REPORTED, not asserted to zero: the producer ran
     under bf16 autocast and stored fp16, so the two paths differ by rounding.)

  B. THE SPATIAL PIN, model-free. Plant a unit impulse at token (r, c) and
     confirm the pooled unit that lights up is exactly (r // kh, c // kw) for
     EVERY arm's kernel. This is what makes "the 40:1 cell is a 64x160 px
     image patch" a measured fact rather than an inherited sentence.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  # isort: skip

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er10_pool_ladder import POOL_ARMS, pool_tokens          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-rows", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"_evidence_class": "MEASURED (ours; banked cache + the checkpoint's "
                              "own readout weights)",
           "eval_tier": "T0-DIAGNOSTIC"}

    # ---- B: the model-free spatial pin (runs first — it needs nothing) ------
    th, tw = 16, 40
    pin = {}
    for arm, (kh, kw) in POOL_ARMS.items():
        gh, gw = th // kh, tw // kw
        bad = []
        for (r, c) in ((0, 0), (3, 9), (4, 10), (7, 19), (15, 39), (6, 18)):
            t = torch.zeros(1, th * tw, 1)
            t[0, r * tw + c, 0] = 1.0
            p = pool_tokens(t, (kh, kw), th, tw).reshape(gh, gw)
            hit = torch.nonzero(p)
            want = (r // kh, c // kw)
            got = tuple(int(x) for x in hit[0]) if len(hit) == 1 else None
            if got != want:
                bad.append({"token": [r, c], "want": list(want), "got": got})
        pin[arm] = {"kernel": [kh, kw], "grid": [gh, gw],
                    "tokens_per_unit": kh * kw,
                    "px_per_unit": [kh * 16, kw * 16],
                    "IMPULSE_PIN_OK": not bad, "failures": bad}
    out["spatial_pin"] = pin
    if any(not v["IMPULSE_PIN_OK"] for v in pin.values()):
        raise SystemExit(f"[j2] ⛔ SPATIAL PIN FAILED: {json.dumps(pin)}")
    print("[j2] B spatial impulse pin: PASS on all "
          f"{len(pin)} kernels; 40:1 unit = "
          f"{pin['p40']['px_per_unit'][0]}x{pin['p40']['px_per_unit'][1]} px "
          f"from {pin['p40']['tokens_per_unit']} tokens", flush=True)

    # ---- A: the real operator against the producer's own cells -------------
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("stack", ck.get("model", ck))
    if not isinstance(sd, dict):
        raise SystemExit("[j2] ⛔ unrecognised checkpoint layout")
    keys = [k for k in sd if "readout" in k and "proj" in k]
    wk = [k for k in keys if k.endswith("weight")]
    bk = [k for k in keys if k.endswith("bias")]
    if not wk:
        raise SystemExit(f"[j2] ⛔ no readout.proj weight in ckpt; saw {keys[:8]}")
    W = sd[wk[0]].float()
    B = sd[bk[0]].float() if bk else torch.zeros(W.shape[0])
    out["readout_proj_key"] = wk[0]
    out["readout_proj_shape"] = list(W.shape)

    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows = blob["rows"][:a.n_rows]
    res = []
    for r in rows:
        tok = r["tokens"].float().unsqueeze(0)             # [1, 640, D]
        flat = pool_tokens(tok, POOL_ARMS["p40"], th, tw)  # [1, 16*D]
        cellsD = flat.reshape(16, -1)                      # [16, D]
        recomputed = cellsD @ W.T + B                      # [16, 128]
        banked = r["cells"].float()
        d = (recomputed - banked).abs()
        res.append({"clip_id": r["clip_id"], "frame_idx": int(r["frame_idx"]),
                    "max_abs_diff": round(float(d.max()), 6),
                    "mean_abs_diff": round(float(d.mean()), 6),
                    "banked_abs_mean": round(float(banked.abs().mean()), 6),
                    "rel_mean": round(float(d.mean()
                                            / banked.abs().mean()), 6),
                    "corr": round(float(np.corrcoef(
                        recomputed.reshape(-1).numpy(),
                        banked.reshape(-1).numpy())[0, 1]), 8)})
    out["operator_check"] = {
        "what": "proj(AvgPool2d((4,10))(banked tokens)) vs banked cells",
        "why_not_exact": "the producer ran under bf16 autocast and stored fp16; "
                         "this path is fp32 from fp16 storage, so the residual "
                         "is ROUNDING and is reported, not asserted to 0",
        "rows": res,
        "worst_rel_mean": round(max(x["rel_mean"] for x in res), 6),
        "min_corr": round(min(x["corr"] for x in res), 8)}
    ok = out["operator_check"]["min_corr"] > 0.999
    out["operator_check"]["AGREES"] = bool(ok)
    print(f"[j2] A operator check: corr min "
          f"{out['operator_check']['min_corr']:.8f}, worst relative mean "
          f"error {out['operator_check']['worst_rel_mean']:.6f} -> "
          f"{'AGREE' if ok else '⛔ DISAGREE'}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")
    if not ok:
        raise SystemExit("[j2] ⛔ the pooled+projected tokens do NOT reproduce "
                         "the banked cells — the ladder's operator is wrong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
