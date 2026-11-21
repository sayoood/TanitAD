"""E-DEC-48 (H2) — DOES THE ACTION ADD ANYTHING **GIVEN THE SCENE**?

⛔ THE HYPOTHESIS THAT WOULD REDIRECT THE PROGRAMME. Drivers brake BECAUSE the
lead brakes. In observational data the action is largely a FUNCTION of the scene,
so a model that reads the scene can predict the future WITHOUT using the action —
and the action's MARGINAL contribution is ~0 even though its CAUSAL effect is
real. This is the mirror of Causal Confusion in Imitation Learning (de Haan et
al., arXiv 1905.11979, banked).

If true it explains BOTH why nine objective terms failed AND why the information
is genuinely absent from Δz: **there is nothing for a marginal predictor to
gain.** And then no loss can help — the fix is INTERVENTIONAL data (counterfactual
actions, simulation), not another objective.

THE MEASUREMENT — three nested predictors of Δz's top directions:

    scene            only the scene features
    action           only the ego's action
    scene + action   both

⭐ THE READABLE QUANTITY IS `(scene+action) − scene`: the action's MARGINAL
contribution GIVEN the scene. If that is ~0 while `action` ALONE is > 0, the
action is REDUNDANT with the scene — confounded, not uninformative. ⚠️ The reverse
gap `(scene+action) − action` distinguishes "confounded" from "the action simply
does nothing".

CONTROLS: a constant reading EXACTLY 0.0000; a TIME-SHUFFLED control on every
cell; NONLINEAR RFF+ridge with clip-disjoint λ; n printed. T0-DIAGNOSTIC,
held-out, lead-matched.
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
LABELS = pathlib.Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k,postrain10k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "confound.json")))
MIN_LEAD, N_CLIPS, F, K, N_DIR = 20, 20, 100, 4, 8


def main() -> int:
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r
    from tanitad.data.psg_targets import PSG_N_COLS, azimuth_column

    dev = torch.device("cuda")
    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def scene(cid, m):
        o = np.zeros((m, 5))
        for i in range(m):
            a = LAB.get(cid, {}).get(i, [])
            cols = np.zeros(PSG_N_COLS)
            for q in a:
                c = azimuth_column(float(q["cx"]), float(q["cy"]))
                if c is not None:
                    cols[c] += 1
            o[i] = [len(a), np.log1p(cols[:3].sum()), np.log1p(cols[3:5].sum()),
                    np.log1p(cols[5:].sum()), (cols == 0).sum()]
        return o

    def n_lead(cid):
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in LAB.get(cid, {}).get(i, [])))

    clips = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
             if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    present = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    print("\n  E-DEC-48 (H2) - DOES THE ACTION ADD ANYTHING GIVEN THE SCENE?\n",
          flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "readable": "(scene+action) MINUS scene = the action's MARGINAL "
                       "contribution given the scene",
           "arms": {}}

    for arm in present:
        w, st = G.load_arm(arm, dev)
        Z, ACT, SCN = [], [], []
        for c in clips:
            d, _r, _o, n_all, _ = G.frames_of(c)
            z, act, spd = G.encode_clip(w, c, dev, F)
            zt = z.float().numpy().astype(np.float64)
            a = act.float().numpy().astype(np.float64)
            v = spd.float().numpy().astype(np.float64).reshape(-1, 1)
            m = min(len(zt) - K, len(a) - K, len(v) - K)
            s = scene(d["clip_id"], min(n_all, F))
            if m < 25 or len(s) < m + K:
                continue
            i0 = np.arange(m)
            Z.append(zt[i0 + K] - zt[i0])
            ACT.append(np.concatenate([a[i0], v[i0]], 1))
            SCN.append(s[i0])
        del w
        torch.cuda.empty_cache()
        if len(Z) < 8:
            continue

        DZ = np.concatenate(Z)
        mu = DZ.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(DZ - mu, full_matrices=False)
        COL = {"scene": SCN, "action": ACT,
               "scene+action": [np.concatenate([s, a], 1) for s, a in zip(SCN, ACT)],
               "constant (control)": [np.ones((len(x), 1)) for x in SCN]}
        rng = np.random.default_rng(0)
        res = {}
        for cn, X in COL.items():
            tr, sh = [], []
            for j in range(N_DIR):
                Y = [(dz - mu) @ Vt[j][:, None] for dz in Z]
                Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
                for i in range(len(X)):
                    Xtr = [X[q] for q in range(len(X)) if q != i]
                    for Yv, sink in ((Y, tr), (Ysh, sh)):
                        ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                        pred, _ = rff_fold(Xtr, ytr, X[i])
                        sink.append(within_clip_r(pred, Yv[i].ravel()))
            res[cn] = (np.array(tr), np.array(sh))

        print(f"  === {arm} (step {st}) - {len(Z)} clips ===")
        print(f"  {'column':<20}{'mean r':>10}{'shuffled':>11}{'true-shuf':>11}{'t':>7}")
        rep["arms"][arm] = {"step": int(st), "n_clips": len(Z), "columns": {}}
        for cn, (tr, sh) in res.items():
            dd = tr - sh
            t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            rep["arms"][arm]["columns"][cn] = {
                "mean_r": round(float(tr.mean()), 4),
                "shuffled": round(float(sh.mean()), 4),
                "true_minus_shuffled": round(float(dd.mean()), 4), "t": round(t, 2)}
            print(f"  {cn:<20}{tr.mean():>+10.4f}{sh.mean():>+11.4f}"
                  f"{dd.mean():>+11.4f}{t:>7.2f}", flush=True)

        sa, s0, a0 = res["scene+action"][0], res["scene"][0], res["action"][0]
        marg = sa - s0
        tm = float(marg.mean()) / max(float(marg.std(ddof=1) / np.sqrt(len(marg))), 1e-12)
        rev = sa - a0
        tr_ = float(rev.mean()) / max(float(rev.std(ddof=1) / np.sqrt(len(rev))), 1e-12)
        rep["arms"][arm]["action_marginal_given_scene"] = {
            "delta": round(float(marg.mean()), 4), "t": round(tm, 2)}
        rep["arms"][arm]["scene_marginal_given_action"] = {
            "delta": round(float(rev.mean()), 4), "t": round(tr_, 2)}
        print()
        print(f"  ACTION's MARGINAL contribution GIVEN the scene: "
              f"{marg.mean():+.4f}  t {tm:+.2f}")
        print(f"  SCENE's  MARGINAL contribution GIVEN the action: "
              f"{rev.mean():+.4f}  t {tr_:+.2f}")
        verdict = ("ACTION IS REDUNDANT WITH THE SCENE - confounded, and no loss "
                   "can help" if tm < 2 else
                   "the action adds real information the scene does not carry")
        print(f"  => {verdict}")
        print()

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
