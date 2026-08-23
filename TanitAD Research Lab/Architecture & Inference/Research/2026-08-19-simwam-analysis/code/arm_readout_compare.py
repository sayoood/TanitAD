"""E-DEC-2 read-out: do TRAINED wider-azimuth readouts decode better — on EGO
AND ENVIRONMENT together?

E-DEC-3 measured the azimuth curve by RE-POOLING a frozen encoder. These arms
TRAINED with the wider readout, which is the question that matters:

    lewm   grid 4 x  4 x 128 = 2048   4 azimuth bins, 30.0 deg/bin  (incumbent)
    rdw8   grid 4 x  8 x  64 = 2048   8 bins, 15.0 deg/bin  -- SAME d_op
    rdw20  grid 4 x 20 x  32 = 2560  20 bins,  6.0 deg/bin

⛔ EGO TARGETS ALONE ARE NOT ENOUGH (PI, 2026-08-23, and E-DEC-3b proved it):
widening helped ego speed (+0.143) but HURT `lead_gap_m` (-0.051) and
`nearest_bearing` (-0.068). Ego motion is partly handed to the model by optical
flow in the 3-frame stack; the environment is the real test. So both run here,
and a geometry only "wins" if it does not lose the environment.

  EGO targets  -> physicalai-val clips, genuinely HELD OUT
  ENV targets  -> the 130-clip corpus (the only one with 3D agent cuboids), which
                  these arms TRAINED on. ⚠️ in-sample; the ARM-vs-ARM contrast is
                  the readable part since all three share the same exposure.

Controls: constant must read exactly 0.0000; raw-pixel floor; frozen DINOv3.
LOEO, paired against `lewm`. T0-DIAGNOSTIC.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
OUT = SP / "e_dec2_arm_compare.json"
VAL = SP / "sp2/cache/physicalai-val-w120-256x640cyl"
LEAD = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
LABELS = SP / "sp2/lead130_agents.jsonl"
ARMS = ["lewm", "rdw8", "rdw20"]
N_ENV = 24
F = 100


def env_targets(ag_by_frame, m):
    lead, ncnt = [], []
    for i in range(m):
        ag = ag_by_frame.get(i, [])
        inl = [a["cx"] for a in ag if abs(a.get("cy", 9e9)) < 1.8 and a.get("cx", -1) > 0]
        lead.append(min(inl) if inl else 80.0)
        ncnt.append(float(len(ag)))
    return {"lead_gap_m": np.array(lead)[:, None], "n_agents": np.array(ncnt)[:, None]}


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    LAB = {}
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "arms": {a: {} for a in ARMS}, "blocks": {}}

    for block, cache, n_clips, held in (("EGO (held-out val)", VAL, 12, True),
                                        ("ENV (train corpus, in-sample)", LEAD, N_ENV, False)):
        clips = sorted(cache.glob("*.v2ep.pt"))[:n_clips]
        COLS, TG, PIXA, POA, used = {}, {}, [], [], []
        for arm in ARMS:
            if not (SP / f"v7tiny_{arm}" / "ckpt.pt").is_file():
                print(f"  {arm}: MISSING ckpt — reported, not skipped")
                continue
            world, st = G.load_arm(arm, dev)
            col = []
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                if not held and d["clip_id"] not in LAB:
                    continue
                z, _, _ = G.encode_clip(world, c, dev, F)
                col.append(z.numpy().astype(np.float64))
                if arm == ARMS[0]:
                    m = len(col[-1])
                    POA.append(d["poses"].numpy().astype(np.float64)[:m])
                    PIXA.append(P.pooled_frames(c, m))
                    if held:
                        pass
                    else:
                        for k, v in env_targets(LAB[d["clip_id"]], m).items():
                            TG.setdefault(k, []).append(v)
                    used.append(d["clip_id"])
            COLS[f"z_op {arm}"] = col
            rep["arms"][arm]["d_op"] = int(col[0].shape[1]) if col else None
            del world
            torch.cuda.empty_cache()
        if held:
            TG = {"speed": [p[:, 3:4] for p in POA],
                  "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))])
                            for p in POA]}
        COLS["pixel (floor)"] = PIXA
        COLS["frozen DINOv3"] = P.dinov3_encode(clips[:len(used)], F, dev)
        COLS["constant (control)"] = [np.ones((len(p), 1)) for p in POA]
        n = len(POA)

        def loeo(X, Y):
            out = []
            for e in range(n):
                idx = [i for i in range(n) if i != e]
                Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
                Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
                Xs = [np.asarray(X[e])[:len(Y[e])]]
                Ys = [Y[e][:len(np.asarray(X[e]))]]
                fn = getattr(P, "probe_fit_score", None)
                return_ = fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0]
                out.append(return_)
            return np.array(out, dtype=np.float64)

        print(f"\n  === {block} · {n} clips ===")
        print(f"  {'target':<12}{'column':<22}{'R2':>9}{'vs lewm':>10}{'t':>7}{'favours':>9}")
        print("  " + "-" * 72, flush=True)
        rep["blocks"][block] = {}
        for tn, Y in TG.items():
            R = {k: loeo(v, Y) for k, v in COLS.items()}
            base = R[f"z_op {ARMS[0]}"]
            rep["blocks"][block][tn] = {}
            for k in COLS:
                r = R[k]
                dd = r - base
                mm = float(dd.mean())
                se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0
                t = mm / max(se, 1e-12)
                rep["blocks"][block][tn][k] = {"r2": round(float(r.mean()), 4),
                                               "delta_vs_lewm": round(mm, 4), "t": round(t, 2),
                                               "n_favouring": int((dd > 0).sum()), "n": len(dd)}
                print(f"  {tn:<12}{k:<22}{float(r.mean()):>+9.4f}{mm:>+10.4f}{t:>7.2f}"
                      f"{int((dd > 0).sum()):>6}/{len(dd)}")
            print()

    # verdict: a geometry only wins if it does NOT lose the environment
    def gain(block, tgt, arm):
        b = rep["blocks"].get(block, {}).get(tgt, {}).get(f"z_op {arm}")
        return (b["delta_vs_lewm"], b["t"]) if b else (0.0, 0.0)

    ego_b = "EGO (held-out val)"
    env_b = "ENV (train corpus, in-sample)"
    lines = []
    for arm in ARMS[1:]:
        e = [gain(ego_b, t, arm) for t in ("speed", "d_ego")]
        v = [gain(env_b, t, arm) for t in ("lead_gap_m", "n_agents")]
        ego_up = sum(1 for d, t in e if t > 2.2)
        env_dn = sum(1 for d, t in v if t < -2.2)
        lines.append(f"{arm}: ego wins {ego_up}/2, env REGRESSIONS {env_dn}/2 "
                     f"[ego {['%+.4f' % d for d, _ in e]}, env {['%+.4f' % d for d, _ in v]}]")
    rep["verdict"] = " | ".join(lines) + (
        "  ⇒ a wider readout may only be shipped if it wins ego WITHOUT regressing environment "
        "(E-DEC-3b: the frozen-encoder curve reversed on lead_gap and bearing).")
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
