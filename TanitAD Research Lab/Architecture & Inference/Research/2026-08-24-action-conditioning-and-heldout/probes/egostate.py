"""E-DEC-50 — DOES OUR LATENT CARRY **EGO STATE** AT ALL?

⭐⭐⭐ THE QUESTION THAT REFRAMES MANDATE 3, AND IT CAME FROM A CONTROL, NOT A
HYPOTHESIS. E-DEC-49 measured that the action predicts the ego's own speed change
at r +0.3386 (t 2.65) — REAL, not an echo (`echocheck.py`: the corpus's accel is
the dataset's OWN measured `ax` from a separate sensor channel, r +0.326 against
the realised 1-tick dv — nowhere near the ~1.0 an identity would give; provenance
confirmed at `physicalai.py:604-632`, which states verbatim that accel is NOT a
finite difference of v).

⛔ THE READING I ALMOST PUBLISHED, AND WHY IT WAS WRONG. My draft said *"the
predictor is HANDED the action and throws it away"* — because `zhat` scored
+0.0020. But the panel's own `z_t` column reads **−0.0081**: the TRUE ENCODED
latent, no prediction involved, does not carry the ego's imminent speed change
either. ⇒ THE DEFECT IS NOT IN THE PREDICTOR. A control I had already computed
refuted the interpretation before it was written down — the eighth time in three
days that a control, not a hypothesis, produced the finding.

⭐ SO THE REAL QUESTION IS WHETHER THE LATENT REPRESENTS EGO STATE AT ALL. And it
must be asked at MORE THAN ONE LOCATION, because absence found at one is not
absence: a latent might carry the speed LEVEL and not its CHANGE, which would be a
completely different diagnosis and a completely different fix.

TARGETS, from static to dynamic:
    IDENTITY: accel_t <- action_t   ⭐ MUST READ ~1.0. A rig that fails this
                                    cannot be trusted on anything below it.
    speed_t (LEVEL)                 do we know how fast we are going?
    yawrate_t (LEVEL)               the lateral analogue
    dv over 4 ticks (CHANGE)        the E-DEC-49 target
    dyaw over 4 ticks (CHANGE)      the lateral change

COLUMNS: z_t (encoded) · zhat_t+k (predicted) · action_t (reference) · constant.

⭐ THE READ: if the latent carries the LEVELS but not the CHANGES, the fix is a
dynamics objective on an ego subspace that already exists. If it carries NEITHER,
the latent is a pure scene-appearance representation with no ego state — and
conditioning it on an action asks a variable that does not encode ego motion to
move in response to an ego command, which would explain nine failed objectives,
α* > 1 with zero gain (E-DEC-47), and Q2X below chance, all at once.

CONTROLS: constant at exactly 0.0000; TIME-SHUFFLED every cell; RFF+ridge with
clip-disjoint λ; n printed; verdict gated on the IDENTITY control.
T0-DIAGNOSTIC, held-out.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,splitp30k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "egostate.json")))
N_CLIPS, F, K = 20, 100, 4
TNAMES = ("IDENTITY accel_t", "speed_t (LEVEL)", "yawrate_t (LEVEL)",
          "dv_4tick (CHANGE)", "dyaw_4tick (CHANGE)")


def wrap(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r
    from tanitad.models.flagship_v15 import SPEED_SCALE

    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print("\n  E-DEC-50 - DOES OUR LATENT CARRY EGO STATE AT ALL?")
    print("  levels (speed, yaw-rate) vs changes (dv, dyaw), z_t vs zhat\n", flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K, "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        W = int(w.window)
        ZT, ZH, AC = [], [], []
        T = {k_: [] for k_ in TNAMES}
        with torch.no_grad():
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                a = act.float().numpy().astype(np.float64)
                v = spd.float().numpy().astype(np.float64).ravel()
                zt_, zh_, ac_ = [], [], []
                tt = {k_: [] for k_ in TNAMES}
                for i in range(0, len(zt) - W - K, 1):
                    j = i + W - 1
                    if j + K >= min(len(v), len(yaw)):
                        break
                    aa = act[i:i + W][None].to(dev).float()
                    if aa.shape[1] != W:
                        break
                    win = zt[i:i + W][None].to(dev).clone()
                    vv = (spd[i] / SPEED_SCALE).view(1, 1, 1).expand(1, W, 1).to(dev)
                    o = w.predictor(win, torch.cat([aa, vv], -1))[1]
                    zh_.append(o.reshape(-1)[:zt.shape[1]].cpu().numpy())
                    zt_.append(zt[j].numpy())
                    ac_.append(a[j])
                    tt["IDENTITY accel_t"].append([a[j, 1]])
                    tt["speed_t (LEVEL)"].append([v[j]])
                    tt["yawrate_t (LEVEL)"].append([wrap(yaw[j + 1] - yaw[j])])
                    tt["dv_4tick (CHANGE)"].append([v[j + K] - v[j]])
                    tt["dyaw_4tick (CHANGE)"].append([wrap(yaw[j + K] - yaw[j])])
                if len(zt_) < 25:
                    continue
                ZT.append(np.stack(zt_).astype(np.float64))
                ZH.append(np.stack(zh_).astype(np.float64))
                AC.append(np.stack(ac_).astype(np.float64))
                for k_ in TNAMES:
                    T[k_].append(np.asarray(tt[k_], dtype=np.float64))
        del w
        torch.cuda.empty_cache()
        if len(ZT) < 8:
            continue

        COL = {"z_t (ENCODED)": ZT, "zhat_t+k (PREDICTED)": ZH, "action_t": AC,
               "constant (control)": [np.ones((len(x), 1)) for x in ZT]}
        nrow = sum(len(x) for x in ZT)
        print(f"  === {arm} (step {st}) - {len(ZT)} clips, {nrow} rows, "
              f"d(z)={ZT[0].shape[1]} ===")
        print(f"  {'target':<22}{'column':<24}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
        print("  " + "-" * 82)
        rep["arms"][arm] = {"step": int(st), "n_clips": len(ZT), "n_rows": nrow,
                            "targets": {}}
        rng = np.random.default_rng(0)
        ident_r = None
        for tn in TNAMES:
            Y = T[tn]
            Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
            rep["arms"][arm]["targets"][tn] = {}
            for cn, X in COL.items():
                tr, sh = [], []
                for i in range(len(X)):
                    Xtr = [X[q] for q in range(len(X)) if q != i]
                    for Yv, sink in ((Y, tr), (Ysh, sh)):
                        ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                        pred, _ = rff_fold(Xtr, ytr, X[i])
                        sink.append(within_clip_r(pred, Yv[i].ravel()))
                tr, sh = np.array(tr), np.array(sh)
                dd = tr - sh
                t = float(dd.mean()) / max(
                    float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
                rep["arms"][arm]["targets"][tn][cn] = {
                    "r": round(float(tr.mean()), 4),
                    "shuffled": round(float(sh.mean()), 4),
                    "true_minus_shuffled": round(float(dd.mean()), 4),
                    "t": round(t, 2)}
                flag = ""
                if tn == "IDENTITY accel_t" and cn == "action_t":
                    ident_r = float(tr.mean())
                    flag = "  <== MUST BE ~1.0"
                print(f"  {tn:<22}{cn:<24}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                      f"{dd.mean():>+9.4f}{t:>7.2f}{flag}", flush=True)
            print()

        rep["arms"][arm]["identity_control_r"] = (
            round(float(ident_r), 4) if ident_r is not None else None)
        g = rep["arms"][arm]["targets"]
        if ident_r is None or ident_r < 0.90:
            v = (f"NO VERDICT - the IDENTITY control reads {ident_r}, not ~1.0. The "
                 f"rig cannot be trusted; fix it before reading anything below it.")
        else:
            lv = max(g["speed_t (LEVEL)"]["z_t (ENCODED)"]["t"],
                     g["yawrate_t (LEVEL)"]["z_t (ENCODED)"]["t"])
            ch = max(g["dv_4tick (CHANGE)"]["z_t (ENCODED)"]["t"],
                     g["dyaw_4tick (CHANGE)"]["z_t (ENCODED)"]["t"])
            if lv > 2.0 and ch <= 2.0:
                v = ("LEVELS YES, CHANGES NO - the latent knows the ego's STATE but "
                     "not its DYNAMICS. The fix is a dynamics objective on an ego "
                     "subspace THAT ALREADY EXISTS, and action-conditioning has "
                     "something real to act on.")
            elif lv > 2.0:
                v = ("LEVELS AND CHANGES BOTH PRESENT - ego state is represented; "
                     "action-conditioning's failure is NOT a missing-variable problem.")
            else:
                v = ("NEITHER - the latent is a pure SCENE-APPEARANCE representation "
                     "with NO ego state. Conditioning it on an action asks a variable "
                     "that does not encode ego motion to move in response to an ego "
                     "command. THE LATENT NEEDS AN EXPLICIT EGO COMPONENT.")
        rep["arms"][arm]["verdict"] = v
        if ident_r is not None:
            print(f"  IDENTITY control r = {ident_r:.4f}")
        print(f"  => {v}\n", flush=True)

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
