#!/usr/bin/env python3
"""P2 -- THE FEASIBILITY-AWARE DECODE, and the frontier that prices it.

ZERO GPU. Reads the banked fan and the checkpoint's frozen anchor vocabulary, and sweeps
the ONE quantity a PROJECTION has that a SHRINK does not: the friction budget `mu`.

⛔ WHY BOTH FRONTIERS ARE COMPUTED HERE, IN ONE SCRIPT, ON ONE NPZ.
The sibling's displacement frontier sweeps `path(lam) = bank + lam*(fan - bank)` -- an
ISOTROPIC SHRINK of the whole displacement. This one sweeps the friction budget of a
CONTROL-SPACE PROJECTION, which removes only the infeasible COMPONENT of the control
profile and keeps the rest. They are different families, and a verdict from one quoted
against the other is the `df` / Thor `free` / `step_s` scope error in a geometry costume.
The only fair axis is MATCHED `fan_peak_g`, so both are computed on the SAME npz and the
SAME windows, and the lambda column is cross-checked against the sibling's independently
published table (a DIFFERENT script, same object) as a route control.

CONTROLS
  C-OBJECT   : npz md5 + shapes + the checkpoint tensor key that supplied the bank.
  C-KNOWN    : the frozen bank's own envelope/kamm/peak_g must reproduce
               `bank_vs_fan_feasibility.json`'s `bank` block, written by a DIFFERENT
               script on a DIFFERENT window draw. The bank is a FIXED path set, so those
               three are window-independent and must match; `off_reach` is NOT and is not
               asserted.
  C-OFF      : the disabled lever returns the input object (weak by construction -- it
               proves the pipeline, not the arithmetic).
  C-ROUNDTRIP: an already-feasible path (the `ha0` v0-hold floor) must come back to <1e-6.
  C-LAMBDA   : lam=1 must reproduce the emitted fan's metrics exactly.

SCOPE, STATED: the projection covers the 2 s prefix (slots 0-3, uniform 0.5 s), which is
the grid every metric here is computed on. Slots 4-7 are at 1 s spacing and are NOT
touched; a deployed decode needs the same treatment on that grid, and that is a named
work item, not a silent omission.
ASCII-only output.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"
sys.path.insert(0, os.path.join(_REPO, "stack"))
sys.path.insert(0, os.path.join(_REPO, "taniteval"))
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))

from tanitad.refs import feasible_decode as FD      # noqa: E402
import fan_safety as FS                             # noqa: E402
import taniteval.ci as CI                           # noqa: E402

TOOL = "2026-09-05-feasible-decode/raw/mu_frontier.py"
NS = 4                                              # the 2 s prefix, slots 0-3
MUS = [None, 2.0, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3]     # None here = BOX ONLY
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_anchors(sd, n, s):
    hits = {k: tuple(v.shape) for k, v in sd.items()
            if k.endswith("anchors") and hasattr(v, "shape") and v.dim() == 3
            and v.shape[-1] == 2}
    if not hits:
        raise SystemExit("no [N,S,2] '*anchors' tensor in the checkpoint")
    key = sorted(hits, key=lambda k: (0 if "decoder.anchors" in k else 1, k))[0]
    a = sd[key].detach().float()
    if tuple(a.shape) != (n, s, 2):
        raise SystemExit("anchors %s do not match the fan (%d, %d, 2)"
                         % (tuple(a.shape), n, s))
    return key, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bank-vs-fan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.npz)
    npz_md5 = md5(a.npz)
    fan8 = torch.from_numpy(z["fan8"]).float()                   # [W, N, 8, 2]
    W, N = fan8.shape[0], fan8.shape[1]
    v0 = torch.from_numpy(z["v0"]).float()                       # [W]
    gt4 = torch.from_numpy(z["gt4"]).float()                     # [W, 4, 2]
    lead5 = torch.from_numpy(z["lead5"]).float().reshape(W, 1, 5, 2)
    sel = torch.from_numpy(z["sel_idx"]).long()
    eid = z["eid"]
    ar = torch.arange(W)

    print("=== P2: the FEASIBILITY-AWARE DECODE -- the mu frontier vs the lambda frontier ===")
    print("  object under test: fan8 %s  npz md5 %s" % (tuple(fan8.shape), npz_md5))

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    akey, anchors = find_anchors(sd, N, fan8.shape[2])
    bank8 = anchors[None].expand(W, -1, -1, -1).contiguous()
    print("  bank tensor: %s %s" % (akey, tuple(anchors.shape)))

    def score(p8):
        """p8 [W, N, 8, 2] -> (flags dict on the 2 s prefix, ade [W, N])."""
        p5 = FS.with_origin(p8[..., :NS, :])
        sc = FS.score_paths(p5, v0, lead5, lead_len_m=FS.LEAD_LEN_DEFAULT_M)
        ade = (p8[..., :NS, :] - gt4[:, None]).norm(dim=-1).mean(dim=-1)
        return sc, ade

    def row(tag, p8, extra=None):
        sc, ade = score(p8)
        d = (p8[..., :NS, :] - fan8[..., :NS, :]).abs().mean()
        r = {"arm": tag,
             "mean_abs_move_from_emitted_m": float(d),
             "fan_envelope": float(sc["envelope"].float().mean()),
             "fan_kamm_over": float(sc["kamm_over"].float().mean()),
             "fan_off_reach": float(sc["off_reach"].float().mean()),
             "fan_infeasible": float(sc["infeasible"].float().mean()),
             "fan_peak_g": float(sc["peak_g"].float().mean()),
             "oracle_ade_m": float(ade.min(dim=1).values.mean()),
             "sel_ade_m": float(ade[ar, sel].mean()),
             "sel_peak_g": float(sc["peak_g"][ar, sel].float().mean()),
             "sel_envelope": float(sc["envelope"][ar, sel].float().mean()),
             "sel_kamm_over": float(sc["kamm_over"][ar, sel].float().mean()),
             "sel_off_reach": float(sc["off_reach"][ar, sel].float().mean())}
        # the paired, episode-clustered interval on the number that costs us something
        dd = (ade[ar, sel] - ade0_sel).double().numpy()
        b = CI.episode_cluster_bootstrap(dd, eid, n_boot=a.n_boot, seed=a.seed)
        r["d_sel_ade_m"] = {"mean": b["mean"], "lo": b["lo"], "hi": b["hi"],
                            "n_windows": b["n_windows"], "n_episodes": b["n_episodes"],
                            "estimator": "paired episode_cluster_bootstrap",
                            "separated": bool(b["lo"] > 0 or b["hi"] < 0)}
        if extra:
            r.update(extra)
        return r

    sc0, ade0 = score(fan8)
    ade0_sel = ade0[ar, sel]

    # ---------------- the PROJECTION frontier ---------------------------------- #
    proj_rows = [row("emitted (mu=off)", fan8, {"mu": None, "lever": "off"})]
    for mu in MUS:
        p5 = FD.project_feasible(FS.with_origin(fan8[..., :NS, :]), v0, mu=mu)
        p8 = fan8.clone()
        p8[..., :NS, :] = p5[..., 1:, :]
        proj_rows.append(row("proj mu=%s" % ("box" if mu is None else "%.2f" % mu), p8,
                             {"mu": mu, "lever": "projection"}))
    # the entry-clamped variant of the headline arm, reported BESIDE, never merged
    p5e = FD.project_feasible(FS.with_origin(fan8[..., :NS, :]), v0, mu=FD.MU_KAMM,
                              clamp_entry=True)
    p8e = fan8.clone()
    p8e[..., :NS, :] = p5e[..., 1:, :]
    entry_row = row("proj mu=0.70 +entry", p8e, {"mu": 0.7, "lever": "projection+entry"})

    # ---------------- the SHRINK frontier, same object, same windows ------------ #
    offs = fan8 - bank8
    lam_rows = []
    for lam in LAMBDAS:
        lam_rows.append(row("lambda=%.2f" % lam, bank8 + lam * offs,
                            {"lambda": lam, "lever": "shrink"}))

    # ================= CONTROLS ================================================= #
    p5in = FS.with_origin(fan8[..., :NS, :])
    c_off = {"name": "C-OFF disabled lever returns the input OBJECT",
             "is_same_object": bool(FD.project_feasible(p5in, v0, enabled=False) is p5in),
             "weak_because": "it short-circuits, so it proves the pipeline and NOT the "
                             "projection arithmetic; C-ROUNDTRIP is the arithmetic's control"}
    c_off["PASS"] = c_off["is_same_object"]

    ha0 = FS.hold_v0_path(v0.double())
    c_rt = {"name": "C-ROUNDTRIP an already-feasible path is a fixed point",
            "object": "fan_safety.hold_v0_path(v0) -- the ha0 floor, W=%d" % W,
            "max_abs_diff_m": float((FD.project_feasible(ha0, v0.double(),
                                                         mu=FD.MU_KAMM) - ha0).abs().max())}
    c_rt["PASS"] = bool(c_rt["max_abs_diff_m"] < 1e-6)

    with open(a.bank_vs_fan, "r", encoding="utf-8") as fh:
        bvf = json.load(fh)
    r_bank = lam_rows[0]
    c_known = {"name": "C-KNOWN the frozen bank reproduces a DIFFERENT script's artifact",
               "object_under_test": akey,
               "reference_artifact": a.bank_vs_fan,
               "reference_n_windows": bvf["n_windows"], "this_n_windows": W,
               "peak_g": {"ours": r_bank["fan_peak_g"], "ref": bvf["bank"]["peak_g"]},
               "envelope": {"ours": r_bank["fan_envelope"], "ref": bvf["bank"]["envelope"]},
               "kamm_over": {"ours": r_bank["fan_kamm_over"],
                             "ref": bvf["bank"]["kamm_over"]},
               "why_window_independent": "the vocabulary is a FIXED [N,S,2] path set "
                                         "(anchor_v0_cond False), so envelope/kamm/peak_g "
                                         "do not depend on the window draw; off_reach does "
                                         "and is NOT asserted"}
    c_known["PASS"] = bool(abs(c_known["peak_g"]["ours"] - c_known["peak_g"]["ref"]) < 1e-4
                           and abs(c_known["envelope"]["ours"]
                                   - c_known["envelope"]["ref"]) < 1e-6
                           and abs(c_known["kamm_over"]["ours"]
                                   - c_known["kamm_over"]["ref"]) < 1e-6)
    c_lam1 = {"name": "C-LAMBDA lambda=1 reproduces the emitted fan",
              "max_abs_diff_peak_g": abs(lam_rows[-1]["fan_peak_g"]
                                         - float(sc0["peak_g"].float().mean())),
              "blind_by_construction": "lambda=1 is the emitted fan whatever the LEFT "
                                       "operand is -- it cannot see a wrong bank (RET #30); "
                                       "C-KNOWN is the control that can"}
    c_lam1["PASS"] = bool(c_lam1["max_abs_diff_peak_g"] < 1e-9)

    # ---------------- P2 criteria ---------------------------------------------- #
    head = [r for r in proj_rows if r.get("mu") == 0.7][0]
    p2c1 = {"criterion": "at mu=0.7 the fan envelope and kamm_over are 0.0000 EXACTLY",
            "fan_envelope": head["fan_envelope"], "fan_kamm_over": head["fan_kamm_over"]}
    p2c1["PASS"] = bool(head["fan_envelope"] == 0.0 and head["fan_kamm_over"] == 0.0)

    # matched-peak_g comparison: what does the SHRINK cost to reach the projection's peak_g?
    lp = np.array([r["fan_peak_g"] for r in lam_rows])
    lo_ = np.array([r["oracle_ade_m"] for r in lam_rows])
    ls_ = np.array([r["sel_ade_m"] for r in lam_rows])
    o_emit = proj_rows[0]["oracle_ade_m"]
    s_emit = proj_rows[0]["sel_ade_m"]
    matched = []
    for r in proj_rows[1:]:
        g = r["fan_peak_g"]
        if g < lp.min() or g > lp.max():
            matched.append({"arm": r["arm"], "fan_peak_g": g,
                            "note": "outside the lambda frontier's peak_g range"})
            continue
        o_lam = float(np.interp(g, lp, lo_))
        s_lam = float(np.interp(g, lp, ls_))
        matched.append({
            "arm": r["arm"], "fan_peak_g": g,
            "proj_d_oracle_ade_m": r["oracle_ade_m"] - o_emit,
            "shrink_d_oracle_ade_m": o_lam - o_emit,
            "proj_d_sel_ade_m": r["sel_ade_m"] - s_emit,
            "shrink_d_sel_ade_m": s_lam - s_emit,
            "ratio_oracle": ((r["oracle_ade_m"] - o_emit) / (o_lam - o_emit)
                             if abs(o_lam - o_emit) > 1e-9 else None),
            "ratio_sel": ((r["sel_ade_m"] - s_emit) / (s_lam - s_emit)
                          if abs(s_lam - s_emit) > 1e-9 else None)})
    m07 = [m for m in matched if m["arm"].startswith("proj mu=0.70")][0]
    p2c2 = {"criterion": "at MATCHED fan_peak_g, the projection's oracle_ade penalty is "
                         "< 0.5x the shrink's",
            "at": m07}
    p2c2["PASS"] = bool(m07.get("ratio_oracle") is not None
                        and m07["ratio_oracle"] < 0.5)

    d1 = proj_rows[0]["fan_peak_g"] - r_bank["fan_peak_g"]
    p2c3 = {"gap_D1_fan_peak_g": d1,
            "closed_at_mu_0p70": proj_rows[0]["fan_peak_g"] - head["fan_peak_g"],
            "pct_of_D1_closed": 100.0 * (proj_rows[0]["fan_peak_g"] - head["fan_peak_g"]) / d1,
            "residual_peak_g_above_bank": head["fan_peak_g"] - r_bank["fan_peak_g"],
            "ratio_emitted_over_bank_before": proj_rows[0]["fan_peak_g"] / r_bank["fan_peak_g"],
            "ratio_after": head["fan_peak_g"] / r_bank["fan_peak_g"]}

    p2c4 = {"criterion": "off_reach must not rise by more than +0.05 absolute",
            "emitted": proj_rows[0]["fan_off_reach"], "at_mu_0p70": head["fan_off_reach"]}
    p2c4["delta"] = p2c4["at_mu_0p70"] - p2c4["emitted"]
    p2c4["PASS"] = bool(p2c4["delta"] <= 0.05)

    out = {"_tool": TOOL, "_tier": "T0 readout on the emitted fan -- never a driving claim",
           "_evidence_class": "MEASURED (ours)",
           "npz": a.npz, "npz_md5": npz_md5, "ckpt": a.ckpt, "anchors_key": akey,
           "n_windows": W, "n_candidates": N,
           "n_episodes": int(len(set(eid.tolist()))),
           "scope": "the 2 s prefix (slots 0-3, uniform 0.5 s). Slots 4-7 are on a 1 s "
                    "grid and are NOT projected; that is a named work item.",
           "margin": FD.MARGIN,
           "controls": {"C_OFF": c_off, "C_ROUNDTRIP": c_rt, "C_KNOWN": c_known,
                        "C_LAMBDA1": c_lam1},
           "P2_C1": p2c1, "P2_C2": p2c2, "P2_C3": p2c3, "P2_C4": p2c4,
           "matched_peak_g": matched,
           "projection_frontier": proj_rows, "entry_variant": entry_row,
           "shrink_frontier": lam_rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("  C-OFF PASS=%s | C-ROUNDTRIP PASS=%s (%.2e m) | C-KNOWN PASS=%s "
          "(bank peak_g ours %.6f vs ref %.6f) | C-LAMBDA1 PASS=%s"
          % (c_off["PASS"], c_rt["PASS"], c_rt["max_abs_diff_m"], c_known["PASS"],
             c_known["peak_g"]["ours"], c_known["peak_g"]["ref"], c_lam1["PASS"]))
    print("")
    hdr = ("%-22s %8s %8s %8s %8s %9s %9s %9s %9s"
           % ("arm", "fan_env", "fan_kamm", "peak_g", "offrch", "orc_ade",
              "sel_ade", "sel_pkg", "sel_env"))
    print("  " + hdr)
    for r in proj_rows + [entry_row]:
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %9.4f"
              % (r["arm"], r["fan_envelope"], r["fan_kamm_over"], r["fan_peak_g"],
                 r["fan_off_reach"], r["oracle_ade_m"], r["sel_ade_m"],
                 r["sel_peak_g"], r["sel_envelope"]))
    print("")
    print("  --- the SHRINK frontier on the SAME windows (for the matched-peak_g axis) ---")
    for r in lam_rows:
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %9.4f"
              % (r["arm"], r["fan_envelope"], r["fan_kamm_over"], r["fan_peak_g"],
                 r["fan_off_reach"], r["oracle_ade_m"], r["sel_ade_m"],
                 r["sel_peak_g"], r["sel_envelope"]))
    print("")
    print("  --- MATCHED fan_peak_g: what each family costs to reach the same safety ---")
    print("  %-22s %8s %12s %12s %8s" % ("arm", "peak_g", "proj d_orcADE",
                                         "shrink d_orcADE", "ratio"))
    for m in matched:
        if "ratio_oracle" not in m:
            print("  %-22s %8.4f  %s" % (m["arm"], m["fan_peak_g"], m["note"]))
            continue
        print("  %-22s %8.4f %12.4f %12.4f %8s"
              % (m["arm"], m["fan_peak_g"], m["proj_d_oracle_ade_m"],
                 m["shrink_d_oracle_ade_m"],
                 ("%.3f" % m["ratio_oracle"]) if m["ratio_oracle"] is not None else "-"))
    print("")
    print("  P2-C1 structural zero at mu=0.70: envelope %.4f kamm %.4f -> %s"
          % (p2c1["fan_envelope"], p2c1["fan_kamm_over"],
             "PASS" if p2c1["PASS"] else "FAIL"))
    print("  P2-C2 matched-peak_g ADE ratio proj/shrink = %s -> %s"
          % (("%.3f" % m07["ratio_oracle"]) if m07.get("ratio_oracle") else "n/a",
             "PASS" if p2c2["PASS"] else "FAIL"))
    print("  P2-C3 closed %.2f%% of the D1 = %.4f g fan gap; ratio to bank %.2fx -> %.2fx"
          % (p2c3["pct_of_D1_closed"], d1, p2c3["ratio_emitted_over_bank_before"],
             p2c3["ratio_after"]))
    print("  P2-C4 off_reach %.4f -> %.4f (delta %+.4f) -> %s"
          % (p2c4["emitted"], p2c4["at_mu_0p70"], p2c4["delta"],
             "PASS" if p2c4["PASS"] else "FAIL"))
    print("  sel_ade delta at mu=0.70: %+.4f m [%+.4f, %+.4f] separated=%s (paired "
          "episode-cluster bootstrap, %d windows / %d episodes)"
          % (head["d_sel_ade_m"]["mean"], head["d_sel_ade_m"]["lo"],
             head["d_sel_ade_m"]["hi"], head["d_sel_ade_m"]["separated"],
             head["d_sel_ade_m"]["n_windows"], head["d_sel_ade_m"]["n_episodes"]))
    print("[mu] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
