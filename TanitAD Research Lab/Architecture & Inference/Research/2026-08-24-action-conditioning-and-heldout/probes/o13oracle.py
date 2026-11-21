"""E-DEC-51 — THE O13 ORACLE: does the LATENT ADD ANYTHING to the ACTION on the
ego's own dynamics? Run BEFORE spending ~8 GPU-hours on the O13 training arm.

⭐ WHY THIS IS THE CHEAPEST DISCRIMINATING EXPERIMENT. E-DEC-50 established that
the action predicts the ego's Δspeed (t 2.56) and Δyaw (t 4.57) while `z_t` predicts
neither (−0.13 / 0.98). O13 proposes a supervised head on `(z_t, action_t)` → Δ(speed,
yaw). ⛔ BUT IF THE LATENT ADDS NOTHING TO THE ACTION, THAT HEAD IS DEGENERATE BY
CONSTRUCTION: it will learn to read the 2 action scalars and ignore the 2048-d
latent, the loss will fall, the metric will look excellent, and the WORLD MODEL WILL
HAVE LEARNED NOTHING. That is O11's failure mode (separation without accuracy)
wearing a new target, and it is worth 40 minutes to rule out.

    joint > action   -> the latent contributes; O13 has real headroom.
    joint ~ action   -> ⛔ the head would be an ACTION ECHO. O13 must be
                        redesigned (e.g. predict Δ(ego) from z_t ALONE and use the
                        action only as a modulator) before any GPU is spent.

⚠️ THE DIMENSIONAL CONFOUND, AND WHY THE PANEL CARRIES TWO JOINTS. The raw joint is
2050-d against ~1710 training rows — n < d, which CLAUDE.md records as a trap that
makes validation choose maximal regularisation and everything read +0.0000
"underpowered by construction, not absence". So the panel ALSO runs a BALANCED
joint: PCA-32 of the latent concatenated with the 2 action scalars (34-d), where the
action cannot be swamped. ⭐ THE PCA BASIS IS FIT ON THE TRAINING FOLDS ONLY — fitting
it on all clips would tune on the scored split, which is exactly the C-class defect
that manufactured four results in one afternoon.

CONTROLS: constant reading EXACTLY +0.0000 · time-shuffled every cell · IDENTITY
control (`action_t` -> `accel_t`) that MUST read ~1.0 · clip-disjoint λ · n and d
printed. Per C160 every verdict cell is enumerated and the default is UNCLASSIFIED,
never a substantive claim.
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
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "o13oracle.json")))
N_CLIPS, F, K, NPC = 20, 100, 4, 32


def wrap(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r

    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print("\n  E-DEC-51 - THE O13 ORACLE: does the LATENT ADD to the ACTION?")
    print("  run BEFORE spending ~8 GPU-hours on the O13 training arm\n", flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K,
           "readable": "joint MINUS action = what the latent adds to the action",
           "arms": {}}

    for arm in present:
        w, _st = G.load_arm(arm, dev)
        st = _st
        W = int(w.window)
        ZT, AC, Y_DV, Y_DY, Y_ID = [], [], [], [], []
        with torch.no_grad():
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float()
                a = act.float().numpy().astype(np.float64)
                v = spd.float().numpy().astype(np.float64).ravel()
                zt_, ac_, dv_, dy_, id_ = [], [], [], [], []
                for i in range(0, len(zt) - W - K, 1):
                    j = i + W - 1
                    if j + K >= min(len(v), len(yaw)):
                        break
                    zt_.append(zt[j].numpy())
                    ac_.append(a[j])
                    dv_.append([v[j + K] - v[j]])
                    dy_.append([wrap(yaw[j + K] - yaw[j])])
                    id_.append([a[j, 1]])
                if len(zt_) < 25:
                    continue
                ZT.append(np.stack(zt_).astype(np.float64))
                AC.append(np.stack(ac_).astype(np.float64))
                Y_DV.append(np.asarray(dv_, dtype=np.float64))
                Y_DY.append(np.asarray(dy_, dtype=np.float64))
                Y_ID.append(np.asarray(id_, dtype=np.float64))
        del w
        torch.cuda.empty_cache()
        if len(ZT) < 8:
            continue

        n_rows = sum(len(x) for x in ZT)
        d_z = ZT[0].shape[1]
        print(f"  === {arm} (step {st}) - {len(ZT)} clips, n={n_rows} rows, "
              f"d(z)={d_z}, d(joint_raw)={d_z + 2}, d(joint_pca)={NPC + 2} ===")
        if n_rows < (d_z + 2):
            print(f"  ⚠️ n ({n_rows}) < d(joint_raw) ({d_z + 2}) - the RAW joint is "
                  f"UNDERPOWERED BY CONSTRUCTION; read joint_pca32.")
        print(f"  {'target':<20}{'column':<26}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
        print("  " + "-" * 82)
        rep["arms"][arm] = {"step": int(st), "n_clips": len(ZT), "n_rows": n_rows,
                            "d_z": d_z, "targets": {}}
        rng = np.random.default_rng(0)
        ident_r = None

        for tname, Y in (("IDENTITY accel_t", Y_ID), ("dv_4tick", Y_DV),
                         ("dyaw_4tick", Y_DY)):
            Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
            cells = {}
            names = ["constant (control)", "action_t", "z_t", "joint_raw",
                     "joint_pca32"]
            for cn in names:
                tr, sh = [], []
                for i in range(len(ZT)):
                    tr_idx = [q for q in range(len(ZT)) if q != i]
                    # ⭐ PCA basis fit on the TRAINING folds only.
                    if cn == "joint_pca32":
                        Ztr = np.concatenate([ZT[q] for q in tr_idx])
                        mu = Ztr.mean(0, keepdims=True)
                        _u, _s, Vt = np.linalg.svd(Ztr - mu, full_matrices=False)
                        B = Vt[:NPC].T

                        def proj(k):
                            return np.concatenate([(ZT[k] - mu) @ B, AC[k]], 1)
                        Xtr = [proj(q) for q in tr_idx]
                        Xte = proj(i)
                    else:
                        def mk(k):
                            if cn == "constant (control)":
                                return np.ones((len(ZT[k]), 1))
                            if cn == "action_t":
                                return AC[k]
                            if cn == "z_t":
                                return ZT[k]
                            return np.concatenate([ZT[k], AC[k]], 1)
                        Xtr = [mk(q) for q in tr_idx]
                        Xte = mk(i)
                    for Yv, sink in ((Y, tr), (Ysh, sh)):
                        ytr = [Yv[q] for q in tr_idx]
                        pred, _ = rff_fold(Xtr, ytr, Xte)
                        sink.append(within_clip_r(pred, Yv[i].ravel()))
                tr, sh = np.array(tr), np.array(sh)
                cells[cn] = (tr, sh)
                dd = tr - sh
                t = float(dd.mean()) / max(
                    float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
                flag = ""
                if tname == "IDENTITY accel_t" and cn == "action_t":
                    ident_r = float(tr.mean())
                    flag = "  <== MUST BE ~1.0"
                print(f"  {tname:<20}{cn:<26}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                      f"{dd.mean():>+9.4f}{t:>7.2f}{flag}", flush=True)

            def tt(dd):
                return float(dd.mean()) / max(
                    float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            g_raw = cells["joint_raw"][0] - cells["action_t"][0]
            g_pca = cells["joint_pca32"][0] - cells["action_t"][0]
            rep["arms"][arm]["targets"][tname] = {
                cn: {"r": round(float(v[0].mean()), 4),
                     "shuffled": round(float(v[1].mean()), 4),
                     "t": round(tt(v[0] - v[1]), 2)} for cn, v in cells.items()}
            rep["arms"][arm]["targets"][tname]["latent_adds_raw"] = {
                "delta": round(float(g_raw.mean()), 4), "t": round(tt(g_raw), 2)}
            rep["arms"][arm]["targets"][tname]["latent_adds_pca32"] = {
                "delta": round(float(g_pca.mean()), 4), "t": round(tt(g_pca), 2)}
            print(f"  {'':20}-> latent adds (raw)    {g_raw.mean():+.4f} "
                  f"(t {tt(g_raw):+.2f})")
            print(f"  {'':20}-> latent adds (pca32)  {g_pca.mean():+.4f} "
                  f"(t {tt(g_pca):+.2f})\n", flush=True)

        # ⛔ C160: every cell enumerated; the default is UNCLASSIFIED, not a claim.
        rep["arms"][arm]["identity_control_r"] = (
            round(float(ident_r), 4) if ident_r is not None else None)
        if ident_r is None or ident_r < 0.90:
            v = (f"NO VERDICT - IDENTITY control reads {ident_r}, not ~1.0; the rig "
                 f"cannot be trusted.")
        else:
            g = rep["arms"][arm]["targets"]
            adds = max(g["dv_4tick"]["latent_adds_pca32"]["t"],
                       g["dyaw_4tick"]["latent_adds_pca32"]["t"])
            act = max(g["dv_4tick"]["action_t"]["t"], g["dyaw_4tick"]["action_t"]["t"])
            if act <= 2.0:
                v = ("NO VERDICT - the action itself does not predict the ego's "
                     "dynamics in this panel, so 'the latent adds nothing to it' "
                     "would be meaningless (the C159 rule).")
            elif adds > 2.0:
                v = ("LATENT CONTRIBUTES - O13 has real headroom; the head cannot "
                     "reach this score by reading the action alone.")
            elif adds <= 0.0:
                v = ("ACTION ECHO RISK - the latent adds NOTHING (or hurts). An O13 "
                     "head on (z, action) would learn to read the action and ignore "
                     "the latent. REDESIGN before spending GPU.")
            else:
                v = (f"UNCLASSIFIED - latent adds t {adds:+.2f}, between 0 and 2. "
                     f"Not separable at this n; state it, do not round it to a claim.")
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
