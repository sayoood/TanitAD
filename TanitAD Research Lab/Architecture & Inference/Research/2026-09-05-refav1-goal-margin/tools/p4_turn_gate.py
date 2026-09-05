#!/usr/bin/env python
"""P4 — DOES refav1 EXECUTE A TURN WHERE THE ROAD ACTUALLY TURNS?

⭐ THE QUESTION, KEYED ON THE ROAD AND NOT ON THE DECODE. The predecessor's
`turn_execution.py` asks "on windows where the head DECODED a turn, did the
planner turn?" — the right gate-2 question, and answered. This asks the PI's
question instead: **on windows where the ROAD turns, does the executed plan
carry curvature?** — and its complement, **on straight road, does it stay
straight?** Both are required; recall alone is not a result.

⛔⛔ WHY THIS IS A REWRITE, AND THE BUG IT AVOIDS. The first version joined the
banked `gt_kappa` on `(clip_index, ws)`. **`clip_index` in a dump is the episode's
position in the SET THAT ARM LOADED**, so on an 8-episode panel it runs 0..7
while the 141-episode bank runs 0..140. Indices 0..7 exist in BOTH, so the join
would have SILENTLY SUCCEEDED against the WRONG EPISODES and produced a
plausible table. Caught before the arms landed, by reading a dump rather than
assuming its schema.

⇒ **The GT curvature is now computed from the dump's OWN `g` (the GT waypoints
banked beside every window), by the same expression `gt_kappa.py` uses** —
`ff._seq_geometry(g, 0.2)`, `yaw_rate.mean / speed.mean.clamp_min(0.5)`. No join,
no index, nothing to mismatch. The name-remapped join against the bank is kept as
a **CONTROL** (`--gt-bank`, `--episode-dir`), because two independent routes to
one physical fact is what makes either trustworthy.

⛔ CONTROLS — every "exactly 0.0" needs a live column beside it:
  * `ha0_ext` — the NON-PLANNER integrator banked in the SAME npz, which must
    read NON-ZERO curvature on a materially non-zero fraction of windows. If it
    does not, the curvature column is dead and every zero here is a READ ERROR.
    (Predecessor: 98.9 % non-zero, absmax 0.603.) ⚠️ This is the arm's own
    `hold_ext_controls`, the canonical integrator — decision M11 / D-MM-ADJ-1 —
    never `echo_gate.ha0_ext`; the two disagree by 1.86 m at 15 s.
  * the banked zero-GPU DECODE PREDICTION for this exact window set must be
    reproduced, or the run is not on the windows it claims.
  * the name-remapped bank join must agree with the from-`g` curvature.
  * ⚠️ near-stationary windows are EXCLUDED (`--min-v0`): gt_kappa divides by
    `speed.clamp_min(0.5)`, so below ~1 m/s it is not a curvature (MEASURED:
    13/4786 windows read |kappa|>0.2, 92.3 % of them at v0<1 m/s, max 3.565 =
    R 0.28 m). Excluded counts are printed, never silently dropped.
  * n printed for every cell; n < 10 marked UNDERPOWERED.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--episodes-json", default=None,
                    help="p4_episodes.json carrying the banked decode prediction")
    ap.add_argument("--gt-bank", default=None,
                    help="intent npz — the CONTROL route to gt_kappa")
    ap.add_argument("--cache-dir", default=None,
                    help="the arm's cache dir; its sorted stems ARE the arm's "
                         "clip_index order, which is how the control remaps")
    ap.add_argument("--kappa-turn", type=float, default=0.04)
    ap.add_argument("--exec-eps", type=float, default=1e-3)
    ap.add_argument("--min-v0", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--taniteval", default="C:/Users/Admin/tanitad-wt/taniteval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    sys.path.insert(0, a.taniteval)
    import torch
    from taniteval import four_families as ff
    from tanitad.models.v6 import tactical_lat_actions
    LAT = list(tactical_lat_actions("v7.0"))
    CURV = [i for i, t in enumerate(LAT)
            if t.startswith(("TURN_", "NUDGE_", "LANE_CHANGE_"))]

    CL, GLAT, HAX, SRC, WS, CIX, G, V0 = [], [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.dump, "ep*.npz"))):
        d_f = os.path.join(a.dump, "decisions", os.path.basename(f))
        if not os.path.exists(d_f):
            continue
        with np.load(f) as e, np.load(d_f) as d:
            CL.append(d["cl_controls"]); GLAT.append(d["goal_lat_cl"])
            HAX.append(d["ha0_ext_controls"]); SRC.append(d["plan_source_cl"])
            WS.append(d["ws"]); G.append(e["g"]); V0.append(e["v0"].ravel())
            CIX.append(np.full(len(d["ws"]),
                               int(np.asarray(e["clip_index"]).ravel()[0])))
    if not CL:
        print("NO WINDOWS READ -- INCONCLUSIVE (not a zero)")
        return 2
    CL = np.concatenate(CL); GLAT = np.concatenate(GLAT)
    HAX = np.concatenate(HAX); SRC = np.concatenate(SRC)
    WS = np.concatenate(WS); CIX = np.concatenate(CIX)
    G = np.concatenate(G); V0 = np.concatenate(V0)
    n_all = len(WS)

    # ---- PRIMARY: gt curvature from the dump's OWN waypoints -------------- #
    Gg = ff._seq_geometry(torch.as_tensor(G[..., :2]).float(), a.dt)
    gt_all = (Gg["yaw_rate"].mean(1)
              / Gg["speed"].mean(1).clamp_min(0.5)).numpy().astype(np.float64)

    keep = V0 >= a.min_v0
    n_excl = int((~keep).sum())
    kap = np.abs(CL[:, :, 1]).max(axis=1)
    ksgn = CL[np.arange(n_all), np.abs(CL[:, :, 1]).argmax(axis=1), 1]
    hax_k = np.abs(HAX[:, :, 1]).max(axis=1)

    gt = gt_all[keep]
    kap_k, ksgn_k, glat_k, src_k = kap[keep], ksgn[keep], GLAT[keep], SRC[keep]
    n = len(gt)
    turn = np.abs(gt) > a.kappa_turn
    strt = ~turn

    def blk(m, name, signed=False):
        if m.sum() == 0:
            return {"name": name, "n": 0, "UNDERPOWERED_n_lt_10": True}
        k = kap_k[m]
        nz = k > a.exec_eps
        out = {"name": name, "n": int(m.sum()),
               "UNDERPOWERED_n_lt_10": bool(m.sum() < 10),
               "frac_executed_nonzero_curvature": float(nz.mean()),
               "kappa_absmax_mean": float(k.mean()),
               "kappa_absmax_p50": float(np.percentile(k, 50)),
               "kappa_absmax_max": float(k.max()),
               "n_exactly_zero": int((k == 0.0).sum()),
               "decode_curv_frac": float(np.isin(glat_k[m], CURV).mean()),
               "plan_source_hist": {int(s): int((src_k[m] == s).sum())
                                    for s in np.unique(src_k[m])}}
        if signed:
            ok = nz & (np.sign(ksgn_k[m]) == np.sign(gt[m]))
            out["frac_executed_nonzero_AND_correct_direction"] = float(ok.mean())
        return out

    ctrl = {
        "CONTROL_ha0_ext_curvature_MUST_BE_NONZERO": {
            "frac_nonzero": float((hax_k > 1e-6).mean()),
            "absmax": float(hax_k.max()), "n": int(n_all),
            "passes": bool((hax_k > 1e-6).mean() > 0.5),
            "note": ("the NON-planner integrator banked in the SAME npz. If "
                     "this reads zero the curvature column is dead and every "
                     "zero above is a READ ERROR, not a measurement.")},
        "low_speed_excluded": {"n_excluded": n_excl, "n_kept": int(n),
                               "min_v0": a.min_v0, "passes": True},
    }

    # ---- CONTROL: the name-remapped bank join must agree ------------------ #
    if a.gt_bank and a.cache_dir and os.path.isdir(a.cache_dir):
        try:
            stems = sorted(p.stem for p in pathlib.Path(a.cache_dir).glob("*.pt")
                           if p.stem != "index")
            Z = np.load(a.gt_bank, allow_pickle=False)
            bank_names = [str(x) for x in Z["clip_names"]]
            name_to_bank = {nm: i for i, nm in enumerate(bank_names)}
            # arm clip_index j -> bank clip_index
            remap = {j: name_to_bank.get(s) for j, s in enumerate(stems)}
            key = {(int(c), int(w)): float(k) for c, w, k in
                   zip(Z["clip_index"], Z["ws"], Z["gt_kappa"])}
            ref = np.array([key.get((remap.get(int(c), -1), int(w)), np.nan)
                            for c, w in zip(CIX, WS)])
            ok = np.isfinite(ref)
            d = (float(np.abs(ref[ok] - gt_all[ok]).max()) if ok.sum() else None)
            ctrl["CONTROL_bank_join_agrees_after_name_remap"] = {
                "n_matched": int(ok.sum()), "n_windows": int(n_all),
                "max_abs_diff": d,
                "remap_is_injective": bool(
                    len({v for v in remap.values() if v is not None})
                    == len([v for v in remap.values() if v is not None])),
                "passes": bool(ok.sum() > 0 and d is not None and d < 1e-4),
                "note": ("⛔ the arm's clip_index is SUBSET-relative (0..7 here) "
                         "while the bank's is 0..140; a raw (clip_index, ws) "
                         "join would silently hit the WRONG episodes")}
        except Exception as e:                            # pragma: no cover
            ctrl["CONTROL_bank_join_agrees_after_name_remap"] = {
                "passes": False, "error": repr(e)}

    if a.episodes_json and os.path.exists(a.episodes_json):
        with open(a.episodes_json, encoding="utf-8") as fh:
            pred = json.load(fh).get("prediction")
        if pred and turn.sum():
            got = float(np.isin(glat_k[turn], CURV).mean())
            ctrl["CONTROL_reproduces_banked_decode_prediction"] = {
                "predicted": pred.get("decode_curv_on_real_turns"),
                "measured": got,
                "passes": bool(abs(got - pred["decode_curv_on_real_turns"]) < 0.10)}

    R = {"dump": a.dump, "arm": a.label,
         "n_windows_total": int(n_all), "n_windows_scored": int(n),
         "kappa_turn_threshold": a.kappa_turn, "exec_nonzero_eps": a.exec_eps,
         "gt_source": "computed from the dump's own `g` (no join)",
         "panel_note": ("episodes selected for real-turn DENSITY; these are "
                        "CONDITIONAL rates on a turn-rich panel, NOT corpus "
                        "marginals"),
         "GT_TURN_windows": blk(turn, "GT TURN (|kappa|>thr)", signed=True),
         "GT_STRAIGHT_windows": blk(strt, "GT STRAIGHT"),
         "decoded_lat_hist": {LAT[i]: int((glat_k == i).sum())
                              for i in range(len(LAT)) if (glat_k == i).sum()},
         "controls": ctrl}
    failed = [k for k, v in ctrl.items() if not v.get("passes")]
    R["controls_failed"] = failed
    t = R["GT_TURN_windows"]
    R["verdict"] = (
        "INCONCLUSIVE (control failed: " + ",".join(failed) + ")" if failed
        else ("GATE OPEN: executed curvature non-zero on "
              f"{t.get('frac_executed_nonzero_curvature'):.4f} of GT-turn windows"
              if t.get("frac_executed_nonzero_curvature", 0) > 0
              else "GATE SHUT: executed curvature EXACTLY 0.0 on every GT-turn "
                   "window, with a LIVE control column"))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)

    s = R["GT_STRAIGHT_windows"]
    print(f"[{a.label}] n_total={n_all} scored={n} (excluded {n_excl} v0<{a.min_v0}) "
          f"thr={a.kappa_turn}")
    print(f"  GT TURN     n={t.get('n')}  exec_nonzero="
          f"{t.get('frac_executed_nonzero_curvature')}  correct_dir="
          f"{t.get('frac_executed_nonzero_AND_correct_direction')}  "
          f"|k|max={t.get('kappa_absmax_max')}  |k|p50={t.get('kappa_absmax_p50')}  "
          f"decode_curv={t.get('decode_curv_frac')}")
    print(f"  GT STRAIGHT n={s.get('n')}  exec_nonzero="
          f"{s.get('frac_executed_nonzero_curvature')}  "
          f"|k|max={s.get('kappa_absmax_max')}  decode_curv={s.get('decode_curv_frac')}")
    print("  CONTROLS: " + json.dumps({k: v.get("passes") for k, v in ctrl.items()}))
    hx = ctrl["CONTROL_ha0_ext_curvature_MUST_BE_NONZERO"]
    print(f"  ha0_ext nonzero={hx['frac_nonzero']:.4f} absmax={hx['absmax']:.4f}"
          "   <- the same-breath control that must read NON-ZERO")
    print(f"  VERDICT: {R['verdict']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
