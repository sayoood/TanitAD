"""A7 verdict -- computes EXACTLY what PREREG_REFCV6_DEVBOX_PREPARATION.md A7.3 / A7.4 / A7.6 /
A7.8 committed before any A7 data, and nothing else.

⛔ Written and staged BEFORE the first A7 arm launched (2026-09-19). Every threshold below is
the pre-registered one; where the prereg used words ("orders of magnitude"), the reading is
fixed HERE, before data, and printed with the result.

Evidence class of every number it prints: MEASURED (ours), tier T0 -- held-out halfB,
open-loop, trainer-side read, 2,000 steps x batch 2 (0.38 of one halfA epoch). NOT a
capability claim (A7.7 item 2).

Usage: python a7_analyze.py <panel_out_dir> --out <verdict.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARMS = ("A7-IN-s0", "A7-IN-s1", "A7-RND-s0", "A7-RND-s1")
N_BOOT = 2000
# A7.3(a): "orders of magnitude" (plural) is read as >= 2 decades -- fixed before data.
IDENTITY_MAX_RATIO = 1e-2
# per-family eval keys (A7.7 item 3): LOSSES, not the doctrine's family METRICS.
# sign +1 = lower is better (a loss); -1 = higher is better (an accuracy).
FAMILIES = {
    "LONGITUDINAL": [("eval_lon", 1), ("eval_lon_tac", 1)],
    "LATERAL": [("eval_lat", 1), ("eval_lat_tac", 1)],
    "TACTICAL": [("eval_tac_v6", 1), ("eval_goal_tac", 1), ("eval_anchor_acc", -1)],
    "STRATEGIC": [("eval_route", 1)],
    "PLANNER (headline)": [("eval_traj", 1)],
    "TOTAL": [("eval_loss", 1)],
}


def sha12(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def _load(panel: Path, arm: str):
    d = panel / arm
    chk = json.loads((d / "a7_arm_check.json").read_text(encoding="utf-8"))
    met = [json.loads(x) for x in (d / "run" / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    ev = [r for r in met if "eval_traj" in r][-1]
    rows = [json.loads(x) for x in (d / "eval_windows.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    bnr = json.loads((d / "run" / "bn_recalib.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "run" / "config.json").read_text(encoding="utf-8"))
    import torch
    stats = torch.load(d / "run" / "bn_recalib_stats.pt", weights_only=False)
    return {"check": chk, "eval": ev, "rows": rows, "bnr": bnr, "cfg": cfg, "stats": stats}


def _stat_distance(x: dict, y: dict) -> dict:
    """Max over BN channels of the relative var difference (the prereg's literal measure)
    and of |d mean| / sqrt(var_y) (the scale-free mean measure A7.4 also uses)."""
    vx, vy = x["var"].double().numpy(), y["var"].double().numpy()
    mx, my = x["mean"].double().numpy(), y["mean"].double().numpy()
    vy_c = np.clip(vy, 1e-12, None)
    return {"max_rel_var": float(np.max(np.abs(vx - vy) / vy_c)),
            "max_mean_over_std": float(np.max(np.abs(mx - my) / np.sqrt(vy_c)))}


def _slot_weighted(traj, frac):
    w = float(np.sum(frac))
    return float(np.sum(traj * frac) / w) if w > 0 else 0.0


def _paired_bootstrap(rnd_rows, in_rows):
    """A7.8 item 4: per draw, per arm, the slot-weighted ratio; plain mean beside it.
    delta = RND - IN, so POSITIVE = ImageNet better (lower loss) -- E's sign."""
    from taniteval.ci import paired_episode_cluster_bootstrap
    eid_r = [r["episode_id"] for r in rnd_rows]
    eid_i = [r["episode_id"] for r in in_rows]
    if eid_r != eid_i:
        raise SystemExit("⛔ the two arms' dump windows are not aligned -- no paired read")
    n = len(rnd_rows)
    traj = np.array([r["traj"] for r in rnd_rows] + [r["traj"] for r in in_rows])
    frac = np.array([r["slot_valid_frac"] for r in rnd_rows]
                    + [r["slot_valid_frac"] for r in in_rows])

    def slot_weighted_ratio(idx):
        ii = np.asarray(idx).astype(np.int64)
        return _slot_weighted(traj[ii], frac[ii])
    a = np.arange(n, dtype=np.float64)                 # RND windows
    b = np.arange(n, 2 * n, dtype=np.float64)          # IN windows, same order
    eids = [sha12(e) for e in eid_r]                   # clip identity only as sha12
    ratio = paired_episode_cluster_bootstrap(a, b, eids, n_boot=N_BOOT, seed=0,
                                             reduce=slot_weighted_ratio)
    plain = paired_episode_cluster_bootstrap(
        np.array([r["traj"] for r in rnd_rows]), np.array([r["traj"] for r in in_rows]),
        eids, n_boot=N_BOOT, seed=0, reduce="mean")
    return {"slot_weighted_ratio": ratio, "plain_window_mean_sensitivity": plain,
            "question_answered": "would another draw of EPISODES say this? -- BLIND to "
                                 "training variance, which is why R, not this, is the headline"}


def _e_f_r(vals: dict, sign: int = 1) -> dict:
    """E = mean(RND) - mean(IN) (sign-adjusted so POSITIVE = ImageNet better);
    F = max within-condition seed difference; R = E / F."""
    i0, i1, r0, r1 = (vals[a] for a in ARMS)
    e = sign * (((r0 + r1) / 2.0) - ((i0 + i1) / 2.0))
    f = max(abs(i0 - i1), abs(r0 - r1))
    r = (math.inf if e > 0 else (-math.inf if e < 0 else float("nan"))) if f == 0 else e / f
    return {"E": e, "F": f, "R": r, "values": dict(vals)}


def analyze(panel: Path) -> dict:
    data = {a: _load(panel, a) for a in ARMS}
    out = {"_evidence_class": "MEASURED (ours)", "_tier": "T0 -- held-out halfB, open-loop, "
           "trainer-side read, 2,000 steps x batch 2 (0.38 halfA epoch); NOT a capability claim",
           "arms": {}, "void_reasons": []}

    # ---- validity (A7.6 last row) ----------------------------------------------------
    for a in ARMS:
        st = data[a]["check"]["status"]
        out["arms"][a] = {"check": st, "eval_traj": data[a]["eval"]["eval_traj"],
                          "staleness": data[a]["bnr"].get("staleness"),
                          "freeze_held": data[a]["bnr"].get("freeze_held"),
                          "recal_windows_sha12": data[a]["cfg"]["trunk_bn_recalib"][
                              "windows_sha12"]}
        if st != "VALID":
            out["void_reasons"].append("%s check %s" % (a, st))
    shas = {out["arms"][a]["recal_windows_sha12"] for a in ARMS}
    if len(shas) != 1:
        out["void_reasons"].append("recal windows differ across arms: %s" % sorted(shas))

    # ---- argv audit (A7.5): the ONLY differences are seed / pretrained / out / dump ---
    allowed = {"--seed", "--out", "--eval-window-dump"}
    ref = data["A7-IN-s0"]["cfg"]["argv"]

    def norm(argv):
        toks, skip = [], False
        for i, t in enumerate(argv):
            if skip:
                skip = False
                continue
            if t in allowed:
                skip = True
                continue
            if t in ("--trunk-pretrained", "--no-trunk-pretrained"):
                continue
            toks.append(t)
        return toks
    audit = {a: norm(data[a]["cfg"]["argv"]) == norm(ref) for a in ARMS}
    out["argv_audit_identical_except_allowed"] = audit
    if not all(audit.values()):
        out["void_reasons"].append("argv differs beyond seed/pretrained/out/dump: %s" % audit)

    # ---- A7.3(a) identity control --------------------------------------------------
    s = {a: data[a]["stats"]["start"] for a in ARMS}
    same = _stat_distance(s["A7-IN-s1"], s["A7-IN-s0"])
    between = _stat_distance(s["A7-RND-s0"], s["A7-IN-s0"])
    ratio = {k: (same[k] / between[k] if between[k] > 0 else math.inf) for k in same}
    ident_ok = all(v <= IDENTITY_MAX_RATIO for v in ratio.values())
    out["identity_control_A7_3a"] = {
        "IN_s0_vs_IN_s1": same, "IN_s0_vs_RND_s0": between, "ratio_same_over_between": ratio,
        "threshold": IDENTITY_MAX_RATIO,
        "threshold_reading": "'orders of magnitude' (plural) = >= 2 decades, fixed pre-data",
        "passed": ident_ok}
    if not ident_ok:
        out["void_reasons"].append("A7.3(a) identity control failed: %s" % ratio)

    # ---- A7.6 headline ---------------------------------------------------------------
    head = _e_f_r({a: data[a]["eval"]["eval_traj"] for a in ARMS})
    i0, i1, r0, r1 = (head["values"][a] for a in ARMS)
    all_cross = all(x < y for x in (i0, i1) for y in (r0, r1))
    R = head["R"]
    if out["void_reasons"]:
        verdict = "VOID"
    elif R > 1:
        verdict = "SUPPORTS"
        # a derived IMPLICATION, asserted rather than assumed: R > 1 forces all four
        # cross-pairs to favour ImageNet (E <= F whenever one pair is reversed).
        assert all_cross, "R > 1 without all four cross-pairs -- arithmetic defect"
    elif R < -1:
        verdict = "RANDOM AHEAD"
    else:
        verdict = "UNDERPOWERED (not REFUTED)"
    out["headline_A7_6"] = dict(head, all_four_cross_pairs_favour_IN=all_cross,
                                verdict=verdict)

    # ---- A7.4 staleness --------------------------------------------------------------
    stl = {}
    for key in ("bn_staleness_var", "bn_staleness_mean"):
        v = {a: (data[a]["bnr"].get("staleness") or {}).get(key) for a in ARMS}
        if any(x is None for x in v.values()):
            stl[key] = {"values": v, "note": "missing"}
            continue
        between_d = ((v["A7-RND-s0"] + v["A7-RND-s1"]) - (v["A7-IN-s0"] + v["A7-IN-s1"])) / 2
        within = max(abs(v["A7-IN-s0"] - v["A7-IN-s1"]), abs(v["A7-RND-s0"] - v["A7-RND-s1"]))
        stl[key] = {"values": v, "between_RND_minus_IN": between_d, "within_seed_spread": within,
                    "partly_attributable_to_stale_normalisation": bool(between_d > within)}
    out["staleness_A7_4"] = stl

    # ---- supporting: paired episode-cluster bootstrap per seed-matched pair ---------
    out["bootstrap_supporting_A7_8"] = {
        "s0": _paired_bootstrap(data["A7-RND-s0"]["rows"], data["A7-IN-s0"]["rows"]),
        "s1": _paired_bootstrap(data["A7-RND-s1"]["rows"], data["A7-IN-s1"]["rows"])}

    # ---- families: per family, never pooled (A7.7 item 3) ---------------------------
    fam = {}
    for f, keys in FAMILIES.items():
        fam[f] = {}
        for k, sign in keys:
            vals = {a: data[a]["eval"].get(k) for a in ARMS}
            if any(v is None for v in vals.values()):
                fam[f][k] = {"note": "not emitted by this eval", "values": vals}
            else:
                fam[f][k] = _e_f_r(vals, sign)
    out["families_losses_not_metrics"] = fam
    out["families_note"] = ("family LOSSES from the trainer-side held-out read; the doctrine's "
                            "family METRICS (headway/TTC, curvature/yaw-rate error, manoeuvre "
                            "confusion, route accuracy) are NOT emitted -- a work item, not "
                            "waived (A7.7 item 3; needs the T1 harness)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analyze(Path(a.panel))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    h = res["headline_A7_6"]
    print("A7 VERDICT: %s | R = E/F = %.4g | E = %.5f | F = %.5f" % (
        h["verdict"], h["R"], h["E"], h["F"]))
    for arm in ARMS:
        print("   %-10s eval_traj %.5f  [%s]" % (arm, h["values"][arm],
                                                 res["arms"][arm]["check"]))
    if res["void_reasons"]:
        print("   VOID reasons:", res["void_reasons"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
