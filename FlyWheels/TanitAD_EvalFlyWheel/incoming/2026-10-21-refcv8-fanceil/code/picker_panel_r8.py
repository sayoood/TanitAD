#!/usr/bin/env python
"""picker_panel_r8.py -- D-REFCV8-FANCEIL NEXT LEVER (Rule Zero): can a GT-FREE picker over the SAME fan recover the headroom
the fan ceiling exposed?  POST-HOC, EXPLORATORY -- not a registered claim; any adoption needs its own pre-registration.

Pickers (none uses GT, ego beyond what the model already consumed, or an extra network -> inference-admissible as RULES):
  own         the model's final pick out['sel_idx'] (E9: core sel_score + goal-distance graft + drivable critic, e9_rank masks)
  base        the banked pre-graft core selection out['sel_idx_base'] (decisions sidecar, same inference seed)
  argmax_core argmax of the core sel_score (out['sel_score'], captured by the fan roll)
  mbr         fan_mbr_x8.mbr_pick (minimum-Bayes-risk inside the core score's top-K), cell CHOSEN on episode-half A
              (even rank of sorted episode ids) over {anchor_logits, refined_logits, sel_score} x K{4,8,16,32} x T{0.5,1,2},
              REPORTED on half B only (the refcv7 X8 discipline, imported unchanged)
SPEED-CAP CHECK (the PI 10-18 cap is ON): planned max speed of the picked candidate from the 8 model slots
(max segment |dp|/dt over origin->0.5->...->6 s) vs the window's fed ceiling v_max_ms (v_max_valid only); a pick counts as
OVER when it exceeds the ceiling by > 0.5 m/s. A PROXY (slot-level, not the model's 0.1 s rollout) -- read as a comparison
between pickers on the same proxy, not as the model's own compliance number.
Estimator: FULL-SET mean, paired episode-cluster bootstrap (taniteval ci, n_boot 2000, seed 0). CI = EPISODES only.
Tier: the picks are os-type (no GT); their ADE uses GT for scoring only.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

SLOT_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])


def _load(path, name):
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def vmax_plan(fan8):
    """[..., 8, 2] -> planned max speed over the 8 slots (origin first)."""
    pts = np.concatenate([np.zeros(fan8.shape[:-2] + (1, 2)), fan8], -2)
    dt = np.diff(np.concatenate([[0.0], SLOT_T]))
    v = np.linalg.norm(np.diff(pts, axis=-2), axis=-1) / dt
    return v.max(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--bank-dump", required=True)
    ap.add_argument("--bank-s6", required=True)
    ap.add_argument("--x8-dir", required=True, help="refcv7 battery code dir (fan_control_x8.py + fan_mbr_x8.py)")
    ap.add_argument("--taniteval-root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--frozen-cell", default=None,
                    help="score,K,T -- use THIS MBR cell (frozen from another seed) instead of choosing on half A")
    a = ap.parse_args()
    sys.path.insert(0, a.taniteval_root)
    os.environ.setdefault("TANITEVAL_ROOT", a.taniteval_root)
    _load(os.path.join(a.x8_dir, "fan_control_x8.py"), "fan_control_x8")
    MB = _load(os.path.join(a.x8_dir, "fan_mbr_x8.py"), "fan_mbr_x8")
    from taniteval import ci as C
    p = lambda x, y, e, t: dict(C.paired_episode_cluster_bootstrap(np.asarray(x, np.float64), np.asarray(y, np.float64), e,
                                                                   n_boot=2000, seed=0), metric=t)
    bt = lambda v, e, t: dict(C.episode_cluster_bootstrap(np.asarray(v, np.float64), e, n_boot=2000, seed=0), metric=t)
    z = np.load(a.fan_npz, allow_pickle=True)
    keys = [(int(c), int(w)) for c, w in zip(z["clip_index"], z["ws"])]
    fan8 = np.asarray(z["anchor_traj"], np.float64)
    B, N = fan8.shape[:2]
    ar = np.arange(B)
    own = np.asarray(z["sel_idx"]).astype(np.int64)
    bank = {}
    for f in sorted(glob.glob(os.path.join(a.bank_dump, "ep*.npz"))):
        q = np.load(f, allow_pickle=True)
        dq = np.load(os.path.join(a.bank_dump, "decisions", os.path.basename(f)), allow_pickle=True)
        ci = int(np.asarray(q["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(q["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(q["ws"]).ravel()):
            bank[(ci, int(w))] = (q["g"][j].astype(np.float64), eid, int(dq["sel_idx_base"][j]), int(dq["sel_idx_nav_true"][j]))
    side = np.load(os.path.join(a.bank_dump, "refcv6_extras.npz"), allow_pickle=True)
    sk = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(side["clip_index"], side["ws"]))}
    vmax = np.asarray([side["v_max_ms"][sk[k]] for k in keys], np.float64)
    vval = np.asarray([side["v_max_valid"][sk[k]] for k in keys], np.float64) > 0.5
    gt = np.stack([bank[k][0] for k in keys])
    eid = [bank[k][1] for k in keys]
    base = np.asarray([bank[k][2] for k in keys], np.int64)
    own_b = np.asarray([bank[k][3] for k in keys], np.int64)
    de = np.linalg.norm(fan8[:, :, :4] - gt[:, None], axis=-1).mean(-1)
    sc = np.asarray(z["sel_score"], np.float64)
    picks = {"own": own, "base": base, "argmax_core": sc.argmax(1)}
    # ---- MBR, half-A selection / half-B report (fan_mbr_x8 discipline) ----
    ue = sorted(set(eid), key=lambda s: int(s))
    half = {e: (i % 2) for i, e in enumerate(ue)}
    hA = np.asarray([half[e] == 0 for e in eid])
    hB = ~hA
    grid = []
    for sn in ("anchor_logits", "refined_logits", "sel_score"):
        if sn not in z.files:
            continue
        s_ = np.asarray(z[sn], np.float64)
        s_ = np.where(np.isfinite(s_), s_, -1e30)
        for K in (4, 8, 16, 32):
            for T in (0.5, 1.0, 2.0):
                pk = MB.mbr_pick(fan8[:, :, :4], s_, K, T)
                gain = float((de[ar, own] - de[ar, pk])[hA].mean())
                grid.append({"score": sn, "K": K, "T": T, "gain_on_A": gain, "_pk": pk})
    best = max(grid, key=lambda r: r["gain_on_A"])
    if a.frozen_cell:
        fs, fk, ft = a.frozen_cell.split(",")
        best = next(r for r in grid if r["score"] == fs and r["K"] == int(fk) and r["T"] == float(ft))
        best = dict(best, frozen_from_cli=a.frozen_cell)
    picks["mbr_chosenA"] = best["_pk"]
    # ---- the same MBR cell restricted to SPEED-CAP-PROXY-compliant candidates (score of a non-compliant candidate -> -1e30
    #      when the window's ceiling is valid; a window with no compliant candidate keeps the full fan). The cell is NOT
    #      re-chosen: it inherits the half-A choice above, so half B stays held out.
    vplan0 = vmax_plan(fan8)
    ok = ~((vplan0 > vmax[:, None] + 0.5) & vval[:, None])
    ok = np.where(ok.any(1, keepdims=True), ok, True)
    s_b = np.asarray(z[best["score"]], np.float64)
    s_b = np.where(np.isfinite(s_b), s_b, -1e30)
    picks["mbr_chosenA_capproxy"] = MB.mbr_pick(fan8[:, :, :4], np.where(ok, s_b, -1e30), best["K"], best["T"])
    picks["argmax_core_capproxy"] = np.where(ok, sc, -1e30).argmax(1)
    for r in grid:
        r.pop("_pk")
    # ---- S6 windows ----
    s6 = {}
    for f in sorted(glob.glob(os.path.join(a.bank_s6, "ep*.npz"))):
        q = np.load(f, allow_pickle=True)
        ci = int(np.asarray(q["clip_index"]).ravel()[0])
        for j, w in enumerate(np.asarray(q["ws"]).ravel()):
            s6[(ci, int(w))] = q["g"][j].astype(np.float64)
    i6 = np.asarray([i for i, k in enumerate(keys) if k in s6])
    g6 = np.stack([s6[keys[i]] for i in i6])
    e6 = [eid[i] for i in i6]
    de6 = np.linalg.norm(fan8[i6][:, :, [1, 3, 4, 5, 6, 7]] - g6[:, None], axis=-1).mean(-1)
    vplan = vmax_plan(fan8)                                         # [B, N]
    out = {"seed": a.seed, "n_windows": B, "n_candidates": N, "n_s6_windows": int(len(i6)),
           "own_vs_banked_sel_idx_agreement": round(float((own == own_b).mean()), 6),
           "mbr_grid_on_A": grid, "mbr_chosen_on_A": {k: best[k] for k in ("score", "K", "T", "gain_on_A", "frozen_from_cli") if k in best},
           "n_half_A_windows": int(hA.sum()), "n_half_B_windows": int(hB.sum()), "pickers": {}}
    eidB = [e for e, m in zip(eid, hB) if m]
    for nm, pk in picks.items():
        d2, d6 = de[ar, pk], de6[np.arange(len(i6)), pk[i6]]
        over = (vplan[ar, pk] > vmax + 0.5) & vval
        r = {"agree_with_own": round(float((pk == own).mean()), 4),
             "ade_2s": bt(d2, eid, nm + " 2s"), "ade_s6": bt(d6, e6, nm + " S6"),
             "minus_own_2s": p(d2, de[ar, own], eid, nm + " - own (2s)"),
             "minus_own_s6": p(d6, de6[np.arange(len(i6)), own[i6]], e6, nm + " - own (S6)"),
             "minus_own_2s_halfB": p(d2[hB], de[ar, own][hB], eidB, nm + " - own (2s, half B)"),
             "speedcap_proxy_over_frac": round(float(over[vval].mean()), 4), "n_vmax_valid": int(vval.sum())}
        out["pickers"][nm] = r
    out["stamp"] = {"status": "POST-HOC EXPLORATORY (Rule Zero next lever) -- not a registered claim",
                    "estimator": "FULL-SET mean; (paired) episode-cluster bootstrap taniteval ci n_boot 2000 seed 0",
                    "ci_answers": "another draw of EPISODES only; one training seed; one inference seed",
                    "base_source": "banked decisions sel_idx_base of the SAME inference seed (the fan roll reproduces the banked "
                                   "sel_idx on own_vs_banked_sel_idx_agreement of windows)",
                    "script": os.path.abspath(__file__)}
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    for nm, r in out["pickers"].items():
        print("ZZPICKZZ %s agree=%.3f 2s=%.4f d2s=%+.4f [%+.4f,%+.4f] S6=%.4f dS6=%+.4f [%+.4f,%+.4f] over=%.4f" % (
            nm, r["agree_with_own"], r["ade_2s"]["mean"], r["minus_own_2s"]["delta"], r["minus_own_2s"]["lo"],
            r["minus_own_2s"]["hi"], r["ade_s6"]["mean"], r["minus_own_s6"]["delta"], r["minus_own_s6"]["lo"],
            r["minus_own_s6"]["hi"], r["speedcap_proxy_over_frac"]))
    print("mbr chosen on A:", out["mbr_chosen_on_A"])
    for nm, r in out["pickers"].items():
        q = r["minus_own_2s_halfB"]
        print("ZZHALFBZZ %s d2s_halfB=%+.4f [%+.4f,%+.4f] sep=%s" % (nm, q["delta"], q["lo"], q["hi"], q["separated"]))


if __name__ == "__main__":
    main()
