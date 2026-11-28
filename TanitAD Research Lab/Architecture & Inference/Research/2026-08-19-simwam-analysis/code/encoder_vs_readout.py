"""E-DEC-1: is the trunk's information DESTROYED BY THE READOUT, or never there?

THE GOAL THIS SERVES: make the trunk decodable. Two completely different fixes
hang on this answer, and we have been unable to choose between them:

    encoder tokens HIGH, z_op LOW  -> the READOUT is the bottleneck.
                                      Fix: widen the readout / predict on tokens.
    encoder tokens LOW             -> the ENCODER never learned it.
                                      Fix: the objective. O5+O6 do not require
                                      decodable content -- SIGReg asks for
                                      isotropy, which NOISE satisfies perfectly.

`z_op` is the operative latent AFTER the readout bottleneck: 16x40 = 640 patch
tokens pooled to a 4x4 grid (MEMORY: 4 azimuth bins over a 120 deg cylindrical
FOV = 30 deg/bin, 2.1-7.8x too coarse for BEV localisation). DINOv3's column in
every previous comparison was MEAN-POOLED PATCH TOKENS. Comparing those two and
concluding "our encoder is bad" confuses the ENCODER with the READOUT -- so this
gives our own encoder EXACTLY the DINOv3 treatment and puts all five columns on
the same probe, the same clips, the same split.

CONTROLS (2026-08-22 cost four false results without them):
  * CONSTANT control must read EXACTLY 0.0000 -- if it does not, the probe is
    broken and nothing in the table is readable.
  * RAW-PIXEL floor -- a learned representation that loses to pixels added nothing.
  * frozen DINOv3 -- an encoder we did not train, on the same clips.
lambda and the PCA basis are fit on the FIT split only; n and d printed.

T0-DIAGNOSTIC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")

VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
OUT = SP / "e_dec1_encoder_vs_readout.json"
DT = 0.1


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    n_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    clips = sorted(VAL.glob("*.v2ep.pt"))[:n_clips]
    assert clips, f"no val clips under {VAL}"
    frames = 120
    arm = sys.argv[1] if len(sys.argv) > 1 else "champ30k"

    world, step = G.load_arm(arm, dev)
    print(f"\n  E-DEC-1 on {arm}@{step} · {len(clips)} held-out val clips\n", flush=True)

    ZOP, ENC, PIX, POSES = [], [], [], []
    for i, c in enumerate(clips, 1):
        z, act, spd = G.encode_clip(world, c, dev, frames)
        ZOP.append(z.numpy().astype(np.float64))
        ENC.append(np.asarray(P.encode_tokens_meanpooled(world, c, dev, frames),
                              dtype=np.float64))
        d = torch.load(c, map_location="cpu", weights_only=False)
        POSES.append(d["poses"].numpy().astype(np.float64)[:frames])
        PIX.append(P.pooled_frames(c, frames))
        print(f"    [{i}/{len(clips)}] {c.name[:10]} zop={ZOP[-1].shape} enc={ENC[-1].shape}",
              flush=True)
    DN = P.dinov3_encode(clips, frames, dev)
    del world
    torch.cuda.empty_cache()

    T = {"speed": [p[:, 3:4] for p in POSES],
         "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))])
                   for p in POSES]}
    F = {"z_op (AFTER readout)": ZOP,
         "enc tokens (mean-pooled)": ENC,
         "frozen DINOv3": DN,
         "pixel (floor)": PIX,
         "constant (control)": [np.ones((len(z), 1)) for z in ZOP]}

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": arm, "step": int(step),
           "n_clips": len(clips), "hypothesis": "E-DEC-1 encoder vs readout",
           "targets": {}}
    print(f"\n  {'target':<9}{'features':<26}{'R2':>9}  {'CI95':<22}{'n':>6}{'d':>5}")
    print("  " + "-" * 82)
    for tn, Y in T.items():
        rep["targets"][tn] = {}
        for fn, X in F.items():
            Xa = [np.asarray(x)[:len(y)] for x, y in zip(X, Y)]
            Ya = [y[:len(x)] for x, y in zip(Xa, Y)]
            r2, lo, hi, lam, n, d = P.probe(Xa, Ya, 128)
            rep["targets"][tn][fn] = {"r2": round(r2, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                      "n": n, "d": d}
            print(f"  {tn:<9}{fn:<26}{r2:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]{n:>6}{d:>5}")
        print()

    # ---- control audit FIRST: a broken control voids the table ---------------
    bad = [tn for tn in T if abs(rep["targets"][tn]["constant (control)"]["r2"]) > 1e-6]
    rep["constant_control_violations"] = bad
    if bad:
        rep["verdict"] = (f"⛔ VOID — the constant control did not read 0.0000 on {bad}. "
                          f"Nothing in this table is readable.")
        print(f"  {rep['verdict']}")
        OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return 1

    sp = rep["targets"]["speed"]
    zop, enc = sp["z_op (AFTER readout)"], sp["enc tokens (mean-pooled)"]
    dino, pix = sp["frozen DINOv3"], sp["pixel (floor)"]
    enc_beats_zop = enc["ci95"][0] > zop["r2"]
    enc_beats_pix = enc["ci95"][0] > max(0.0, pix["r2"])
    rep["point_estimates"] = {
        "enc_over_zop": round(enc["r2"] / max(zop["r2"], 1e-9), 3),
        "enc_over_pixel": round(enc["r2"] / max(pix["r2"], 1e-9), 3),
        "enc_over_dinov3": round(enc["r2"] / max(dino["r2"], 1e-9), 3),
        "note": ("a CI that straddles zero at this n is an UNDERPOWERED statement, not a null "
                 "result -- report both the ratio and the CI, never the CI alone")}
    if enc_beats_zop and enc_beats_pix:
        v = (f"⭐ THE READOUT IS THE BOTTLENECK — our ENCODER tokens decode speed at "
             f"{enc['r2']:+.4f} (CI lower {enc['ci95'][0]:+.4f}) while z_op AFTER the 4x4 readout "
             f"reads {zop['r2']:+.4f}. The information IS in the trunk and the pooling destroys "
             f"it. Fix the readout, not the objective.")
    elif not enc_beats_pix:
        v = (f"THE ENCODER NEVER LEARNED IT — encoder tokens {enc['r2']:+.4f} "
             f"CI[{enc['ci95'][0]:+.4f}, {enc['ci95'][1]:+.4f}] do not beat the pixel floor "
             f"{pix['r2']:+.4f}, so the readout is exonerated and the OBJECTIVE is the defect: "
             f"O5+O6 never required decodable content (SIGReg's isotropy is satisfied by noise). "
             f"Frozen DINOv3 reads {dino['r2']:+.4f} on the same clips.")
    else:
        v = (f"MIXED — encoder {enc['r2']:+.4f} vs z_op {zop['r2']:+.4f} vs pixels "
             f"{pix['r2']:+.4f}: encoder beats pixels but not decisively past the readout. "
             f"Both levers are live.")
    rep["verdict"] = v
    print(f"  VERDICT: {v}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
