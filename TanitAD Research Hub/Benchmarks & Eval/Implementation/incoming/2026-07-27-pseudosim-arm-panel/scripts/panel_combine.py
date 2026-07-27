#!/usr/bin/env python3
"""Combine per-arm pseudo-simulation dumps into THE PANEL. **No GPU. No model.**

This is the arithmetic-only path: every number in `PSEUDOSIM_ARM_PANEL.md` is
produced here from the `pw_<arm>.npz` dumps, so any bar recomputes on a laptop
with no checkpoint and no corpus. Nothing in `taniteval.pseudosim` is
reimplemented — `score_windows`, `discriminative_range`, `composite` and the
bootstrap are imported and called.

WHAT IT ENFORCES
----------------
1. ⭐ **ROW-IDENTITY ACROSS ARMS.** Every arm must have the identical
   `(ep_i, anchor, dlat, dyaw, dlon)` key sequence, in the identical order. This
   is asserted, not assumed — a paired bootstrap over misaligned rows is a
   fabricated number. Any arm that fails is REFUSED, not silently dropped.
2. The discriminative-range gate is recomputed **with `by_arm` over the whole
   panel**, which is the adjudicating form: a component that has range within one
   arm but no spread ACROSS arms cannot rank anything.
2b. ⭐ **THE PANEL GATE — a component enters the composite only if it is
   admissible for EVERY arm.** Stated as a rule before it was applied, and it
   exists because the per-arm gate produces **different weight sets per arm**,
   which makes the arms non-comparable: a paired delta would then mix a metric
   change with a model change. MEASURED on the 2-episode smoke: `comfort` is
   inadmissible for `v4_oracle` (all-zero, range 0) and for `cv_holdv0`
   (all-one, saturated) but ADMISSIBLE for REF-C (0.0125 pass-rate) — so without
   this rule REF-C would be scored on a 3-component composite and v4 on a
   2-component one. The per-arm verdicts are kept in the artifact
   (`composite_per_arm_gate`) so the effect of the rule is auditable, and the
   panel is also reported under the per-arm gate as a sensitivity.
   ⚠️ Note this keeps `comfort` DROPPED, exactly as the published run had it —
   the rule is not retuning a bound after seeing who fails, it is refusing to
   compare two different composites.
3. Paired contrasts use `taniteval.ci.paired_episode_cluster_bootstrap`
   (B=2000, unit = val episode) on rows finite in BOTH arms.
   `overlapping_holdout_se` appears nowhere.
4. ⚠️ The pre-registered **D-NULL** criterion is evaluated mechanically and its
   verdict is written into the artifact, so the "is this instrument even able to
   rank?" question is answered by the data and not by the narrator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

for p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts", "/root/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)

from taniteval import ci as _ci        # noqa: E402
from taniteval import pseudosim as PS  # noqa: E402


def load_pw(path):
    z = np.load(path, allow_pickle=False)
    return {
        "traj": torch.as_tensor(z["traj"]),
        "ref_path": torch.as_tensor(z["ref_path"]),
        "ref_yaw": torch.as_tensor(z["ref_yaw"]),
        "v0": torch.as_tensor(z["v0"]),
        "pt_dlat": torch.as_tensor(z["pt_dlat"]),
        "pt_dyaw": torch.as_tensor(z["pt_dyaw"]),
        "pt_dlon": torch.as_tensor(z["pt_dlon"]),
        "anchor": torch.as_tensor(z["anchor"]),
        "ep_i": torch.as_tensor(z["ep_i"]),
        "eid": [str(x) for x in z["eid"]],
    }


def key_of(pw):
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


def paired(a, b, eid, n_boot=2000):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return None
    out = _ci.paired_episode_cluster_bootstrap(
        a[m], b[m], list(np.asarray(eid)[m]), n_boot=n_boot)
    out["n_rows_paired"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="v4_oracle")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--adversary", default="stand_still",
                    help="comma-separated VALIDATION PROBES: scored and "
                         "reported, but excluded from the panel gate and from "
                         "the ranking. A probe whose whole purpose is to be "
                         "REFUSED must not be able to delete the metric for "
                         "every real arm.")
    a = ap.parse_args()

    in_dir = Path(a.in_dir)
    arms, metas = {}, {}
    for f in sorted(in_dir.glob("pw_*.npz")):
        name = f.name[len("pw_"):-len(".npz")]
        arms[name] = load_pw(f)
        mj = in_dir / f"arm_{name}.json"
        if mj.exists():
            metas[name] = json.loads(mj.read_text()).get("_meta", {})
    assert arms, f"no pw_*.npz in {in_dir}"
    print(f"[panel] {len(arms)} arms: {sorted(arms)}", flush=True)

    # ---------------- 1. ROW IDENTITY -------------------------------------- #
    ref = a.reference if a.reference in arms else sorted(arms)[0]
    kref = key_of(arms[ref])
    row_id, refused = {}, {}
    for n, pw in arms.items():
        k = key_of(pw)
        ok = (k.shape == kref.shape) and bool((k == kref).all())
        row_id[n] = {"n_rows": int(k.shape[0]), "identical_to_reference": ok}
        if not ok:
            refused[n] = (f"row keys differ from {ref}: shape {k.shape} vs "
                          f"{kref.shape}")
    for n in refused:
        arms.pop(n)
    assert arms, f"every arm refused on row identity: {refused}"
    eid = arms[ref]["eid"] if ref in arms else arms[sorted(arms)[0]]["eid"]

    # ---------------- 2. SCORE + PANEL-WIDE RANGE GATE --------------------- #
    scores = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw)
        d = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        d["no_collision"] = None
        d["ttc"] = None
        d["_extra"] = {k: sc[k] for k in
                       ("ego_progress_raw_ratio", "cross_track_end_m",
                        "cross_track_hold_matched_m", "along_track_end_m",
                        "_cross_track_hold_v0_m_DIAGNOSTIC_NOT_USED")}
        scores[n] = d
    by_arm = {n: {k: v for k, v in d.items() if k != "_extra"}
              for n, d in scores.items()}

    res = {
        "_experiment": "PSEUDO-SIMULATION ARM PANEL -- every reachable arm on "
                       "the bounded pre-generated perturbation grid",
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the pw_*.npz dumps)",
        "_protocol": PS.PROTOCOL,
        "_estimator": f"taniteval.ci.episode_cluster_bootstrap / "
                      f"paired_episode_cluster_bootstrap (B={a.n_boot}, "
                      f"unit = val episode), paired on IDENTICAL rows",
        "_refused_estimator": ("overlapping_holdout_se -- it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": PS.TRAFFIC_MODE_NOTE,
        "collision_and_ttc": {"emitted": False,
                              "reason": PS.COLLISION_UNAVAILABLE_REASON},
        "row_identity": row_id, "arms_refused_on_row_identity": refused,
        "reference_arm": ref, "arm_meta": metas,
        "arms": {}, "paired": {},
    }
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        out_p.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    # --- 2b. THE PANEL GATE: admissible for EVERY arm, or not in the composite #
    adversary = {s for s in a.adversary.split(",") if s.strip()}
    gate_arms = [n for n in arms if n not in adversary]
    per_arm_ranges = {n: PS.discriminative_range(by_arm[n], by_arm=by_arm)
                      for n in arms}
    panel_admissible, panel_why = {}, {}
    for comp in PS.COMPONENT_WEIGHTS:
        bad = [n for n in gate_arms
               if not per_arm_ranges[n].get(comp, {}).get("admissible")]
        panel_admissible[comp] = (len(bad) == 0)
        panel_why[comp] = ("admissible for every arm" if not bad else
                           "INADMISSIBLE for " + ", ".join(sorted(bad)) +
                           " -> dropped from EVERY arm so the composite is the "
                           "same object on both sides of every paired delta")
    res["PANEL_GATE"] = {
        "rule": ("a weighted component enters the composite ONLY if it is "
                 "admissible for EVERY arm in the panel. Stated before it was "
                 "applied. Without it the arms carry different weight sets and "
                 "a paired delta mixes a metric change with a model change."),
        "admitted": {k: v for k, v in panel_admissible.items() if v},
        "dropped": {k: panel_why[k] for k, v in panel_admissible.items() if not v},
        "per_arm_admissibility": {
            n: {c: per_arm_ranges[n].get(c, {}).get("admissible")
                for c in PS.COMPONENT_WEIGHTS} for n in sorted(arms)},
        "validation_probes_excluded_from_the_gate": sorted(adversary & set(arms)),
        "why_probes_are_excluded": (
            "`stand_still` is an ADVERSARY, not a candidate: its whole purpose "
            "is to be refused. It is inadmissible on EVERY component by "
            "construction (ego_progress identically 0, recovery 100 % NaN, "
            "comfort saturated at 1.0), so letting it into the intersection "
            "would make the composite vacuous for every real arm -- the probe "
            "would delete the metric it exists to test. It is still scored and "
            "still reported; it just does not vote on the gate."),
    }
    print("[panel] PANEL GATE admitted="
          f"{sorted(res['PANEL_GATE']['admitted'])} "
          f"dropped={sorted(res['PANEL_GATE']['dropped'])}", flush=True)

    comp_val, comp_val_perarm = {}, {}
    for n, pw in arms.items():
        node = PS.emit(pw, arm=n, n_boot=a.n_boot, by_arm_scores=by_arm)
        comp_val_perarm[n] = node.pop("_per_window_composite", None)
        node.pop("_per_window", None)
        # the PER-ARM-gate composite is kept for audit ...
        node["composite_per_arm_gate"] = node.pop("composite", None)
        # ... and the PANEL-gate composite is the one that ranks.
        pr = dict(per_arm_ranges[n])
        for c, ok in panel_admissible.items():
            if not ok and c in pr:
                pr[c] = dict(pr[c], admissible=False,
                             reason="dropped by the PANEL GATE: " + panel_why[c])
        try:
            comp = PS.composite(by_arm[n], pr)
            val = comp.pop("value")
            comp["ci"] = PS._boot(val, arms[n]["eid"], a.n_boot, 0)
            comp["_panel_gated"] = True
            comp_val[n] = val
            node["composite"] = comp
        except PS.VacuousMetric as exc:
            node["composite"] = {"REFUSED_TO_EMIT": str(exc)}
            comp_val[n] = None
        node["_meta"] = metas.get(n, {})
        res["arms"][n] = node
        c = node.get("composite", {})
        ci = (c.get("ci") or {}) if isinstance(c, dict) else {}
        print(f"[panel] {n:28s} PSS={ci.get('mean')} "
              f"[{ci.get('lo')},{ci.get('hi')}]", flush=True)
        bank()

    # per-arm diagnostics that make the panel readable without a re-run
    res["_recovery_defined_fraction"] = {
        n: round(float(np.isfinite(scores[n]["recovery"]).mean()), 6)
        for n in arms}
    res["_ego_progress_mean"] = {
        n: round(float(np.nanmean(scores[n]["ego_progress"])), 6) for n in arms}
    res["_along_track_end_mean_m"] = {
        n: round(float(np.nanmean(scores[n]["_extra"]["along_track_end_m"])), 6)
        for n in arms}
    res["_cross_track_end_mean_m"] = {
        n: round(float(np.nanmean(scores[n]["_extra"]["cross_track_end_m"])), 6)
        for n in arms}

    # ---------------- 3. PAIRED CONTRASTS ---------------------------------- #
    names = sorted(arms)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            blk = {"_estimator": "taniteval.ci.paired_episode_cluster_bootstrap "
                                 f"(B={a.n_boot}, unit = val episode)",
                   "_refused_estimator": "overlapping_holdout_se"}
            for comp in ("ego_progress", "recovery", "comfort"):
                blk[comp] = paired(scores[x][comp], scores[y][comp], eid, a.n_boot)
            if comp_val.get(x) is not None and comp_val.get(y) is not None:
                blk["PSS_recovery_progress"] = paired(comp_val[x], comp_val[y],
                                                      eid, a.n_boot)
            # SENSITIVITY: the same delta under the PER-ARM gate, so the effect
            # of the panel gate on every conclusion is visible, not asserted.
            if (comp_val_perarm.get(x) is not None
                    and comp_val_perarm.get(y) is not None):
                blk["_PSS_under_per_arm_gate_SENSITIVITY"] = paired(
                    comp_val_perarm[x], comp_val_perarm[y], eid, a.n_boot)
            res["paired"][f"{x}__minus__{y}"] = blk
            d = blk.get("PSS_recovery_progress") or {}
            print(f"[paired] {x} - {y}: PSS {d.get('delta')} "
                  f"[{d.get('lo')},{d.get('hi')}] sep={d.get('separated')}",
                  flush=True)
            bank()

    # ---------------- 4. THE PRE-REGISTERED VERDICTS ----------------------- #
    def pss(x, y):
        k, sgn = f"{x}__minus__{y}", 1.0
        if k not in res["paired"]:
            k, sgn = f"{y}__minus__{x}", -1.0
        d = (res["paired"].get(k) or {}).get("PSS_recovery_progress")
        if not d:
            return None
        return {"delta": sgn * d["delta"], "lo": (sgn * d["hi"] if sgn < 0
                                                  else d["lo"]),
                "hi": (sgn * d["lo"] if sgn < 0 else d["hi"]),
                "separated": d["separated"], "_sign_flipped": sgn < 0}

    g1 = pss("v4_oracle", "v4_blind")
    g2p = pss("v4_oracle", "cv_holdv0")
    g2r = None
    k = "cv_holdv0__minus__v4_oracle"
    if k in res["paired"] and res["paired"][k].get("recovery"):
        d = res["paired"][k]["recovery"]
        g2r = {"delta": -d["delta"], "lo": -d["hi"], "hi": -d["lo"],
               "separated": d["separated"]}
    elif "v4_oracle__minus__cv_holdv0" in res["paired"]:
        g2r = res["paired"]["v4_oracle__minus__cv_holdv0"].get("recovery")

    res["PREREGISTERED_GATES"] = {
        "G1_instrument_sensitivity": {
            "question": "does the protocol separate an arm that CAN see the "
                        "perturbation from the IDENTICAL arm that cannot?",
            "v4_oracle_minus_v4_blind_PSS": g1,
            "published_reference": "+0.1882 [+0.1240, +0.2557] SEPARATED",
            "PASS": bool(g1 and g1["separated"] and g1["delta"] > 0),
            "if_fail": "NO ARM SCORE IS ADMISSIBLE; the panel reports the "
                       "failure instead of a ranking."},
        "G2_port_fidelity": {
            "v4_oracle_minus_cv_holdv0_PSS": g2p,
            "v4_oracle_minus_cv_holdv0_recovery": g2r,
            "published_reference_PSS": "-0.0034 [-0.0138, +0.0078] n.s.",
            "published_reference_recovery": "-0.0168 [-0.0332, -0.0008] SEPARATED",
            "PASS_pss_ns": bool(g2p and not g2p["separated"]),
            "PASS_recovery_sep_neg": bool(g2r and g2r.get("separated")
                                          and g2r["delta"] < 0)},
    }

    # G5: the standing-still adversary, if it was run
    if "stand_still" in arms:
        rc = np.asarray(scores["stand_still"]["recovery"], float)
        defined = float(np.isfinite(rc).mean())
        res["PREREGISTERED_GATES"]["G5_standing_still_adversary"] = {
            "recovery_defined_fraction": round(defined, 6),
            "recovery_mean_where_defined": (round(float(np.nanmean(rc)), 6)
                                            if np.isfinite(rc).any() else None),
            "frac_scored_1p0": round(float((rc[np.isfinite(rc)] >= 0.999).mean()),
                                     6) if np.isfinite(rc).any() else None,
            "PASS": bool(defined < 0.02),
            "rule": ("a plan that does not move must be EXCLUDED (NaN), never "
                     "scored 1.0. The pre-fix metric put this shape +0.597 "
                     "ABOVE a sighted arm.")}

    # D-NULL: is the instrument able to rank at all?
    learned = [n for n in names if n != "v4_blind" and n not in adversary]
    contrasts, n_sep = [], 0
    for i, x in enumerate(learned):
        for y in learned[i + 1:]:
            d = pss(x, y)
            if d:
                contrasts.append({"pair": f"{x} - {y}", **d})
                n_sep += int(bool(d["separated"]))
    res["PREREGISTERED_GATES"]["D_DISCRIMINATIVE_POWER"] = {
        "definition": ("D-NULL iff EVERY pairwise paired PSS contrast among the "
                       "non-blind arms straddles zero => the instrument "
                       "separates sighted from blind but CANNOT RANK OUR ARMS, "
                       "and NO ranking is published."),
        "n_contrasts": len(contrasts), "n_separated": n_sep,
        "verdict": ("D-NULL" if (contrasts and n_sep == 0) else
                    "D-RANK" if (contrasts and n_sep == len(contrasts)) else
                    "D-PARTIAL"),
        "contrasts": contrasts}
    bank()
    print(json.dumps(res["PREREGISTERED_GATES"], indent=2, default=str))
    print(f"[panel] wrote {a.out}")
    print("PANEL_COMBINE_DONE", flush=True)


if __name__ == "__main__":
    main()
