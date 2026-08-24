"""E-DEC-30 — WHICH ACTION CHANNEL CONDITIONS THE PREDICTOR?

The predictor's action tensor is 3-dimensional and its channels are NOT
interchangeable (`flagship_v15.py:101`):

    action_dim = 3  ->  [steer, accel, v0/10]

E-DEC-28b/28c shuffled only `aa` = [steer, accel] and held `vv` = v0 at its TRUE
value, then found the prediction moved 0.77 % and nrmse was unchanged to four
decimals. I read that as "the predictor is action-blind". ⛔ IT IS NOT THAT CLAIM.
It is "the predictor ignores steer/accel WHEN SPEED IS TRUE" — two of three
channels, with the one the programme MEASURED to be worth 3.73 -> 0.83 m fwd_ade
left intact. C137 already retracted a programme-wide action-blind claim once, for
a scope defect of exactly this family; this panel is built so it cannot recur.

CONDITIONS — identical windows, identical rollout, ONLY the action tensor differs:

    true          the reference
    shuffle_sa    steer+accel from a random other time in the clip, v0 TRUE
    shuffle_v     v0 from a random other time, steer+accel TRUE      <- the gap
    shuffle_all   all three resampled together                       <- the gap
    zero_sa       steer = accel = 0, v0 true
    zero_v        v0 = 0, steer/accel true
    negate_sa     steer, accel *= -1        (a hard left becomes a hard right)
    scale100_sa   steer, accel *= 100       (the C137 probe form, for continuity)

TWO NUMBERS PER CONDITION, and the first is what makes the second readable:

    d_in  = ||a' - a|| / ||a||                    how different the INPUT actually is
    d_out = ||zhat' - zhat|| / ||zhat - z_last||  the C137-CORRECTED response,
                                                  normalised by the predictor's OWN
                                                  delta, never by an arm property

⚠️ A near-zero `d_out` is evidence of insensitivity ONLY IF `d_in` is large. If
steer and accel sit near zero through a cruising clip, a shuffle is a WEAK
perturbation and the null is a property of the action DISTRIBUTION, not of the
predictor. That is why `d_in` is printed beside every cell.

⭐ POSITIVE CONTROL, without which the whole panel is unreadable: perturb the
LATENT WINDOW by 10 % Gaussian, keep the action true, and report the same `d_out`.
A predictor that responds to the latent but not to a channel is genuinely
insensitive to that channel; one that responds to NEITHER is dead, and then the
panel says nothing about actions at all.

T0-DIAGNOSTIC. MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "actchan.json")))
# ⭐ arms and output are env-overridable so the O11 arm is scored by the
# IDENTICAL instrument on the IDENTICAL windows/clips/seed as its baseline,
# rather than by a forked copy that could drift from it.
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,splitp30k,scale1").split(",")
N_CLIPS, F, STRIDE = 12, 80, 2


def main() -> int:
    import v7tiny_g2 as G
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print(f"\n  E-DEC-30 · WHICH ACTION CHANNEL CONDITIONS THE PREDICTOR?"
          f"\n  action = [steer, accel, v0/10] · arms {present} · {len(clips)} clips"
          f"\n  d_in = how different the INPUT is; d_out = C137-corrected response\n",
          flush=True)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "action_layout": "[steer, accel, v0/10] (flagship_v15.py:101)",
           "method": "identical windows and rollout; only the action tensor differs. "
                     "d_out normalised by the predictor's OWN delta (the C137 "
                     "correction), never by an arm property.",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        acc = {}
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for c in clips:
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                if len(zt) < W + 2:
                    continue
                for i in range(0, len(zt) - W - 1, STRIDE):
                    win = zt[i:i + W][None].to(dev).clone()
                    base = zt[i + W - 1].to(dev).reshape(-1)
                    aa = act[i:i + W][None].to(dev).float()
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    r0 = int(rng.integers(0, max(1, len(act) - W)))
                    r1 = int(rng.integers(0, max(1, len(spd) - W)))
                    aa2 = act[r0:r0 + W][None].to(dev).float()
                    vv2 = (spd[r1] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    if aa2.shape[1] != W:
                        continue
                    ref = torch.cat([aa, vv], -1)

                    def roll(a, wn=win):
                        o = w.predictor(wn, a)[1]
                        return o.reshape(-1)[:zt.shape[1]]

                    z0 = roll(ref)
                    den = float((z0 - base).norm())
                    if den < 1e-9:
                        continue
                    conds = {
                        "shuffle_sa": torch.cat([aa2, vv], -1),
                        "shuffle_v": torch.cat([aa, vv2], -1),
                        "shuffle_all": torch.cat([aa2, vv2], -1),
                        "zero_sa": torch.cat([torch.zeros_like(aa), vv], -1),
                        "zero_v": torch.cat([aa, torch.zeros_like(vv)], -1),
                        "negate_sa": torch.cat([-aa, vv], -1),
                        "scale100_sa": torch.cat([aa * 100.0, vv], -1),
                    }
                    an = float(ref.norm())
                    for k, a in conds.items():
                        zp = roll(a)
                        d = acc.setdefault(k, [[], []])
                        d[0].append(float((a - ref).norm()) / max(an, 1e-9))
                        d[1].append(float((zp - z0).norm()) / den)
                    # ⭐ POSITIVE CONTROL: move the LATENT, keep the action true.
                    pert = win + 0.10 * win.std() * torch.randn_like(win)
                    zp = roll(ref, pert)
                    d = acc.setdefault("[ctrl] latent +10% noise", [[], []])
                    d[0].append(0.10)
                    d[1].append(float((zp - z0).norm()) / den)
                del z, act, spd, zt
            del w
        torch.cuda.empty_cache()

        n = len(acc["shuffle_sa"][0])
        print(f"  === {arm} (step {st}) · {n} windows ===")
        print(f"  {'condition':<26}{'d_in':>10}{'d_out':>10}{'reading':>34}")
        rep["arms"][arm] = {"step": int(st), "n_windows": n, "conditions": {}}
        for k, (di, do) in acc.items():
            mi, mo = float(np.mean(di)), float(np.mean(do))
            if k.startswith("[ctrl]"):
                rd = ("POSITIVE CONTROL - predictor is alive" if mo > 0.05
                      else "DEAD - ignores its own latent too")
            elif mi < 0.02:
                rd = "input barely changed - UNINFORMATIVE"
            elif mo < 0.05:
                rd = "INSENSITIVE to this channel"
            elif mo < 0.25:
                rd = "weakly conditioned"
            else:
                rd = "CONDITIONED on this channel"
            rep["arms"][arm]["conditions"][k] = {
                "d_in": round(mi, 4), "d_out": round(mo, 4),
                "d_out_median": round(float(np.median(do)), 4), "reading": rd}
            print(f"  {k:<26}{mi:>10.4f}{mo:>10.4f}{rd:>34}")
        print()

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
