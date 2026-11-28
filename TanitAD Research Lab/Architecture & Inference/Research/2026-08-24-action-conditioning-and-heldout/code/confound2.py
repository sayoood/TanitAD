"""E-DEC-48b (H2, CORRECTED) — IS THE ACTION REDUNDANT WITH THE SCENE?

⛔⛔ WHY THE FIRST VERSION COULD NOT ANSWER THIS. `confound.py` predicted Δz and
found scene t −0.64, action t 1.88, scene+action t −2.19 — **nothing worked** —
and its verdict line printed *"ACTION IS REDUNDANT WITH THE SCENE"*. **You cannot
conclude REDUNDANCY when NEITHER predictor works.** Redundancy means: the action
works alone, the scene works alone, and together they add nothing beyond the
scene. All-null is a different finding — it is E-DEC-40 restated (Δz's residual is
noise), wearing a confounding label. The verdict fired on `marginal_t < 2`, which
is true whenever nothing works at all. Seventh auto-verdict in two days to fire on
the wrong quantity.

⭐ THE FIX IS THE TARGET, NOT THE STATISTICS. Ask the question about a quantity the
scene DEMONSTRABLY predicts, so "adds nothing on top of it" is meaningful:

    target = the FUTURE SCENE at t+k
             n_agents(t+k) · occ_center(t+k) · n_free_cols(t+k) · lead_closing(t+k)

    scene_t          strong by autocorrelation — THE POSITIVE CONTROL. If this
                     is not significant the panel cannot answer anything and the
                     run ABORTS rather than reporting.
    action_t         the ego's command alone
    scene_t+action_t both

⭐⭐ THE READABLE QUANTITY: `(scene+action) − scene` = the action's MARGINAL
contribution GIVEN the scene, on a target the scene actually explains.

    scene works AND marginal ≈ 0   -> the action is REDUNDANT WITH THE SCENE.
                                      Confounded. NO LOSS CAN HELP; the fix is
                                      INTERVENTIONAL data (counterfactual actions,
                                      simulation). ⇒ a PI-level redirection.
    scene works AND marginal > 0   -> the action carries information the scene
                                      does not. A loss CAN exploit it, and
                                      ActSWM's frozen readout (arXiv 2607.26712)
                                      is the arm to run.
    scene does NOT work            -> ABORT. The panel is uninformative and no
                                      verdict is admissible.

⚠️ THE VERDICT IS PRINTED ONLY WHEN THE POSITIVE CONTROL PASSES. That is the
guard whose absence made the first version report a redirection it had not earned.

CONTROLS: constant reading EXACTLY 0.0000; TIME-SHUFFLED on every cell; NONLINEAR
RFF+ridge with clip-disjoint λ; n printed. T0-DIAGNOSTIC, held-out, lead-matched.
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
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "confound2.json")))
MIN_LEAD, N_CLIPS, F, K = 20, 20, 100, 4


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

    def feats(cid, m):
        """scene descriptors + the lead closing rate, per frame."""
        o = np.zeros((m, 5))
        rng_ = np.full(m, np.nan)
        for i in range(m):
            a = LAB.get(cid, {}).get(i, [])
            cols = np.zeros(PSG_N_COLS)
            for q in a:
                c = azimuth_column(float(q["cx"]), float(q["cy"]))
                if c is not None:
                    cols[c] += 1
            o[i] = [len(a), np.log1p(cols[:3].sum()), np.log1p(cols[3:5].sum()),
                    np.log1p(cols[5:].sum()), (cols == 0).sum()]
            inl = [x["cx"] for x in a
                   if abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0]
            if inl:
                rng_[i] = min(inl)
        clo = np.full(m, np.nan)
        clo[1:] = rng_[1:] - rng_[:-1]
        return o, clo

    def n_lead(cid):
        return sum(1 for i in range(F)
                   if any(abs(x.get("cy", 9e9)) < 1.8 and x.get("cx", -1) > 0
                          for x in LAB.get(cid, {}).get(i, [])))

    clips = [c for c in sorted(LEAD.glob("*.v2ep.pt"))
             if torch.load(c, map_location="cpu", weights_only=False)["clip_id"] in LAB]
    clips = [c for c in clips
             if n_lead(torch.load(c, map_location="cpu",
                                  weights_only=False)["clip_id"]) >= MIN_LEAD][:N_CLIPS]
    arm = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()][0]
    print("\n  E-DEC-48b (H2, CORRECTED) - IS THE ACTION REDUNDANT WITH THE SCENE?")
    print("  target = the FUTURE SCENE at t+k, which the scene at t DOES predict")
    print("  => 'adds nothing on top of it' is a meaningful statement\n", flush=True)

    w, st = G.load_arm(arm, dev)
    SCN, ACT, TGT = [], [], {"n_agents": [], "occ_center": [],
                             "n_free_cols": [], "lead_closing": []}
    MK = []
    for c in clips:
        d, _r, _o, n_all, _ = G.frames_of(c)
        _z, act, spd = G.encode_clip(w, c, dev, F)
        a = act.float().numpy().astype(np.float64)
        v = spd.float().numpy().astype(np.float64).reshape(-1, 1)
        s, clo = feats(d["clip_id"], min(n_all, F))
        m = min(len(s) - K, len(a) - K, len(v) - K)
        if m < 25:
            continue
        i0 = np.arange(m)
        SCN.append(s[i0])
        ACT.append(np.concatenate([a[i0], v[i0]], 1))
        TGT["n_agents"].append(s[i0 + K][:, 0:1])
        TGT["occ_center"].append(s[i0 + K][:, 2:3])
        TGT["n_free_cols"].append(s[i0 + K][:, 4:5])
        TGT["lead_closing"].append(np.nan_to_num(clo[i0 + K])[:, None])
        MK.append(~np.isnan(clo[i0 + K]))
    del w
    torch.cuda.empty_cache()

    COL = {"scene_t (POSITIVE CONTROL)": SCN, "action_t": ACT,
           "scene_t + action_t": [np.concatenate([s, a], 1) for s, a in zip(SCN, ACT)],
           "constant (control)": [np.ones((len(x), 1)) for x in SCN]}
    rng = np.random.default_rng(0)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "arm": arm, "step": int(st), "k": K,
           "readable": "(scene+action) MINUS scene, on a target the scene predicts",
           "targets": {}}
    print(f"  {'target':<14}{'column':<28}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
    print("  " + "-" * 78)
    for tn, Y0 in TGT.items():
        if tn == "lead_closing":
            keep = [i for i in range(len(Y0)) if int(MK[i].sum()) >= 20]
            if len(keep) < 8:
                print(f"  {tn:<14}SKIPPED - only {len(keep)} clips with >=20 rows")
                continue
            Y = [Y0[i][MK[i]] for i in keep]
            C = {k: [v[i][MK[i]] for i in keep] for k, v in COL.items()}
        else:
            Y, C = Y0, COL
        res = {}
        for cn, X in C.items():
            tr, sh = [], []
            Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
            for i in range(len(X)):
                Xtr = [X[q] for q in range(len(X)) if q != i]
                for Yv, sink in ((Y, tr), (Ysh, sh)):
                    ytr = [Yv[q] for q in range(len(Yv)) if q != i]
                    pred, _ = rff_fold(Xtr, ytr, X[i])
                    sink.append(within_clip_r(pred, Yv[i].ravel()))
            res[cn] = (np.array(tr), np.array(sh))
        rep["targets"][tn] = {"n_clips": len(Y), "columns": {}}
        for cn, (tr, sh) in res.items():
            dd = tr - sh
            t = float(dd.mean()) / max(float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            rep["targets"][tn]["columns"][cn] = {
                "r": round(float(tr.mean()), 4), "shuffled": round(float(sh.mean()), 4),
                "true_minus_shuffled": round(float(dd.mean()), 4), "t": round(t, 2)}
            print(f"  {tn:<14}{cn:<28}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>7.2f}")
        sc_, sa_ = res["scene_t (POSITIVE CONTROL)"][0], res["scene_t + action_t"][0]
        dsc = sc_ - res["constant (control)"][0]
        t_ctrl = float(dsc.mean()) / max(float(dsc.std(ddof=1) / np.sqrt(len(dsc))), 1e-12)
        marg = sa_ - sc_
        tm = float(marg.mean()) / max(float(marg.std(ddof=1) / np.sqrt(len(marg))), 1e-12)
        rep["targets"][tn]["positive_control_t"] = round(t_ctrl, 2)
        rep["targets"][tn]["action_marginal_given_scene"] = {
            "delta": round(float(marg.mean()), 4), "t": round(tm, 2)}
        # ⛔ THE VERDICT PRINTS ONLY IF THE POSITIVE CONTROL PASSED.
        if t_ctrl < 2.0:
            v = (f"NO VERDICT - the positive control (scene_t) reads t {t_ctrl:+.2f}; "
                 f"this panel cannot answer for {tn}")
        elif tm < 2.0:
            v = ("ACTION IS REDUNDANT WITH THE SCENE for this target - "
                 "confounded; no loss can exploit what adds nothing")
        else:
            v = ("the ACTION CARRIES INFORMATION THE SCENE DOES NOT - "
                 "a loss can exploit it")
        rep["targets"][tn]["verdict"] = v
        print(f"  {'':14}-> ctrl t {t_ctrl:+.2f} · action marginal {marg.mean():+.4f} "
              f"(t {tm:+.2f})  ::  {v}\n", flush=True)

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
