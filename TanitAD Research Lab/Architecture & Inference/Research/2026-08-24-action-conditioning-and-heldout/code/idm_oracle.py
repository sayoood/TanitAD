"""E-DEC-39 — THE INVERSE-DYNAMICS ORACLE: is the action even RECOVERABLE from
the latent transition? (the bound on what O11 can possibly achieve)

⛔⛔ WHY THIS MUST EXIST BEFORE O11 IS SCORED. O11 asks the predictor to make
ẑ_{t+k} depend on the action. But **if the action's effect on the latent over k
ticks is below the latent's own noise floor, NO objective can succeed** — the
information is not there to learn. O11 would then read REFUTED for a reason that
has nothing to do with the objective, and the pre-registration's REFUTED branch
("the deficit is the injection SITE") would send the next GPU-day in the wrong
direction. This is risk (3) in `PLAN_TO_THE_GOAL_2026-08-24.md`, stated there in
advance: *"the 0.6 s horizon may be too short for actions to matter at all."*

⭐ THE ORACLE. Use only ENCODED latents — no predictor anywhere. Ask a probe to
recover the ACTION from the latent transition:

    z_t  alone            -> a_t     the BASELINE
    [z_t, z_{t+k}]        -> a_t     the QUESTION
    [z_t, z_{t+k} - z_t]  -> a_t     the same information, delta-parameterised

⭐⭐ **THE READABLE QUANTITY IS THE DELTA, NOT THE RAW R².** A latent that encodes
ego speed makes a_t partly recoverable from `z_t` ALONE — through autocorrelation
of the driver's own behaviour, not through any knowledge of consequences. So the
question is strictly: **does adding the FUTURE latent improve action recovery?**
If it does, the transition carries the action and O11 has something real to learn.
If it does not, the action leaves no recoverable trace in 0.4 s of latent motion,
and the ceiling on O11 is zero for reasons no loss can change.

CONTROLS, fixed in advance:
  constant           reads the no-information value EXACTLY (R² = 0 by construction)
  TIME-SHUFFLED      the target action drawn from a random other time in the same
                     clip. ⛔ MUST read ~0. Whatever it reads is what the probe
                     gets from clip-level statistics alone.
  pixels (floor)     raw input at t and t+k, same treatment
  ⚠️ the probe is RFF+ridge (CONVEX, closed form) with λ on a CLIP-DISJOINT inner
     split — both lessons from tonight: an MLP diverged (shuffled control read
     −14) and a random-ROW λ split let clip-level memorisation validate perfectly.

⚠️ SCOPE. T0-DIAGNOSTIC. Held-out, lead-matched. WITHIN-CLIP R² is reported
alongside cross-clip because between-clip offset dominates the cross-clip metric.
A negative here bounds RECOVERABILITY BY THIS FUNCTION CLASS, not in principle.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
sys.path.insert(0, str(SP))
LEAD = Path(os.environ.get("SPD_CORPUS",
                           str(SP / "sp2/cache/physicalai-val130-heldout")))
LABELS = Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
OUT = Path(os.environ.get("SPD_OUT", str(SP / "idm_oracle.json")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,splitp30k").split(",")
MIN_LEAD, N_CLIPS, F, K = 20, 20, 100, 4


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold          # the convex, clip-disjoint probe

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def n_lead(cid):
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in LAB.get(cid, {}).get(i, [])))

    clips = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
             if torch.load(c, map_location="cpu",
                           weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-39 · INVERSE-DYNAMICS ORACLE — is the action RECOVERABLE?"
          f"\n  arms {present} · {len(clips)} lead-matched held-out clips · k={K}"
          f"\n  ⭐ the readable quantity is [z_t,z_t+k] MINUS z_t alone\n", flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "k": K, "function_class": "RFF+ridge, convex, clip-disjoint lambda",
           "arms": {}}

    PIXT, PIXF = [], []
    for arm in present:
        w, st = G.load_arm(arm, dev)
        COL = {"z_t alone": [], "[z_t, z_t+k]": [], "[z_t, dz]": []}
        ACT, need_pix = [], not PIXT
        for c in clips:
            z, act, _ = G.encode_clip(w, c, dev, F)
            zt = z.float().numpy().astype(np.float64)
            a = act.float().numpy().astype(np.float64)
            m = min(len(zt) - K, len(a) - K)
            if m < 25:
                continue
            i0 = np.arange(m)
            COL["z_t alone"].append(zt[i0])
            COL["[z_t, z_t+k]"].append(np.concatenate([zt[i0], zt[i0 + K]], 1))
            COL["[z_t, dz]"].append(np.concatenate([zt[i0], zt[i0 + K] - zt[i0]], 1))
            ACT.append(a[i0])
            if need_pix:
                d, raw, off, n_all, _ = G.frames_of(c)
                im = [torch.from_numpy(np.asarray(
                    Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")).copy())
                    .permute(2, 0, 1).float() / 255.0 for i in range(min(n_all, F))]
                px = torch.nn.functional.adaptive_avg_pool2d(
                    torch.stack(im)[:, -3:], (8, 8)).reshape(len(im), -1).numpy()
                PIXT.append(np.concatenate([px[i0], px[i0 + K]], 1).astype(np.float64))
        COL["pixels (floor)"] = PIXT
        COL["constant (control)"] = [np.ones((len(a), 1)) for a in ACT]
        del w
        torch.cuda.empty_cache()

        nr = [len(a) for a in ACT]
        for k_, v in COL.items():
            if [len(x) for x in v] != nr:
                raise SystemExit(f"[FATAL] {k_} rows {[len(x) for x in v]} vs {nr}")
        print(f"  === {arm} (step {st}) · {len(nr)} clips · {sum(nr)} rows ===")
        print(f"  {'channel':<8}{'column':<20}{'TRUE_w':>9}{'SHUF_w':>9}"
              f"{'vs z_t alone':>14}{'t':>7}")
        rng = np.random.default_rng(0)
        rep["arms"][arm] = {"step": int(st), "channels": {}}
        for ch, name in ((0, "steer"), (1, "accel")):
            Y = [a[:, ch:ch + 1] for a in ACT]
            Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
            got = {}
            for cname, X in COL.items():
                tw, sw = [], []
                for i in range(len(X)):
                    Xtr = [X[j] for j in range(len(X)) if j != i]
                    for Yv, sink in ((Y, tw), (Ysh, sw)):
                        ytr = [Yv[j] for j in range(len(Yv)) if j != i]
                        pred, ym = rff_fold(Xtr, ytr, X[i])
                        yte = Yv[i].ravel()
                        sink.append(1.0 - float(((yte - pred) ** 2).sum())
                                    / max(float(((yte - yte.mean()) ** 2).sum()), 1e-12))
                got[cname] = (np.array(tw), np.array(sw))
            base = got["z_t alone"][0]
            rep["arms"][arm]["channels"].setdefault(name, {})
            for cname, (tw, sw) in got.items():
                d = tw - base
                t = (float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
                     if cname != "z_t alone" else 0.0)
                rep["arms"][arm]["channels"][name][cname] = {
                    "true_within": round(float(tw.mean()), 4),
                    "shuffled_within": round(float(sw.mean()), 4),
                    "delta_vs_z_t_alone": round(float(d.mean()), 4), "t": round(t, 2)}
                print(f"  {name:<8}{cname:<20}{tw.mean():>+9.4f}{sw.mean():>+9.4f}"
                      f"{d.mean():>+14.4f}{t:>7.2f}", flush=True)
            print()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
