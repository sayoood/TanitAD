#!/usr/bin/env python3
"""Recompute THE ARM PANEL under the PUBLISHED and the TWO-SIDED progress term.

**No GPU. No model. No checkpoint. No corpus.** Every number is arithmetic over
the committed ``pw_<arm>.npz`` per-window dumps, so any bar here recomputes on a
laptop. Nothing in ``taniteval.pseudosim`` is reimplemented — ``score_windows``,
``discriminative_range``, ``composite`` and the bootstrap are IMPORTED.

WHAT THIS ANSWERS
-----------------
1. ⭐ **Does ``cv_holdv0`` still rank first once the metric can see over-travel?**
   The previous headline (*"a zero-parameter baseline beats every arm"*) was
   measured on a term that charges nothing for travelling too far, and
   ``cv_holdv0`` is precisely the arm that gets the longitudinal axis right by
   construction. If the ranking changes, that is the week's result.
2. **Does the source stream's intervention now separate?** ``v1_ego_v0`` minus
   ``v1_tactical_follow`` was ``+0.0078 [-0.0110, +0.0260] n.s.`` under
   ``clamp_v1`` despite a 5.65x along-track RMS improvement.
3. **How sensitive is the ranking to the over-travel slope ``w``?** Two-sided
   does not automatically mean symmetric, so ``w`` is swept rather than chosen.

THE GATE
--------
⛔ **PANEL-WIDE**, exactly as ``panel_combine.py`` defines it: a weighted
component enters the composite only if it is admissible for EVERY non-probe arm,
so both sides of every paired delta are the same object. The per-arm gate is NOT
used and is not offered — under it, re-timing a plan at constant speed lifts
``comfort`` 0.0004 -> 0.2882 (720x) and flips the source stream's own primary
verdict from n.s. to SEPARATED. It is computed only as a *refusal record*.

Validation probes (``stand_still``, ``v1_ego_half``, ``v1_ego_double``,
``oracle_lon_straight``) are scored and reported but excluded from the gate and
from the ranking — the panel's own ``stand_still`` rule: a probe whose purpose is
to be refused must not delete the metric for every real arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
for p in (str(_REPO / "stack"), str(_REPO / "stack" / "scripts"),
          str(_REPO / "taniteval"),
          "/root/TanitAD/stack", "/root/taniteval"):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

from taniteval import ci as _ci        # noqa: E402
from taniteval import pseudosim as PS  # noqa: E402

#: probes: scored, reported, but they do not vote on the gate or the ranking
PROBES = ("stand_still", "v1_ego_half", "v1_ego_double", "oracle_lon_straight")
#: the terms recomputed side by side. `clamp_v1` FIRST so the reproduction gate
#: runs before anything new is trusted.
TERMS = ("clamp_v1", "twosided_v2")
#: the over-travel-slope sensitivity grid (w = 0.5 / 1.5 / 2 / 3; w = 1 is
#: `twosided_v2` itself)
SENSITIVITY = ("twosided_asym_w0p5", "twosided_asym_w1p5",
               "twosided_asym_w2", "twosided_asym_w3")

#: PUBLISHED values under `clamp_v1` — the reproduction gate. If any of these
#: fails to reproduce, NOTHING in this artifact is admissible.
PUBLISHED_CLAMP_V1 = {
    "cv_holdv0": 0.5705, "v4_oracle": 0.5622, "refc_xl_produced": 0.5499,
    "v1_tactical_follow": 0.5471, "v1_tactical_oracle": 0.5467,
    "refc_small_produced": 0.5444, "refc_base_produced": 0.5439,
    "nospeed_tactical_oracle": 0.5394, "v4_blind": 0.3749,
    "v1_ego_v0": 0.5608, "v1_ego_oracle_lon": 0.5946, "v1_lat_straight": 0.5460,
    "refc_base_v0on": 0.5439, "refc_base_v0off": 0.4980,
    "refc_xl_v0on": 0.5499, "v1_ego_half": 0.3117,
}


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
    """The row identity a paired bootstrap is only valid over. Same as
    ``panel_combine.key_of`` — asserted, never assumed."""
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


def paired(a, b, eid, n_boot=2000):
    """``taniteval.ci.paired_episode_cluster_bootstrap`` on rows finite in BOTH.

    ⛔ ``overlapping_holdout_se`` appears nowhere: it biases the POINT ESTIMATE
    as well as the interval."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return None
    out = _ci.paired_episode_cluster_bootstrap(a[m], b[m],
                                               list(np.asarray(eid)[m]),
                                               n_boot=n_boot)
    out["n_rows_paired"] = int(m.sum())
    return out


def decompose(pw):
    """Lateral / longitudinal split of the 2 s endpoint — the axis this whole
    finding lives on. ``pseudosim._cross_and_along`` IMPORTED, not reimplemented.
    Sign: along + = the plan travelled FURTHER than the human."""
    x, y, rx, ry = PS._cross_and_along(pw)
    along = (x[:, -1] - rx[:, -1]).numpy()
    cross = (y[:, -1] - ry[:, -1]).numpy()
    return {
        "along_err_mean_m": round(float(np.nanmean(along)), 4),
        "along_abs_err_mean_m": round(float(np.nanmean(np.abs(along))), 4),
        "along_rms_m": round(float(np.sqrt(np.nanmean(along ** 2))), 4),
        "cross_abs_err_mean_m": round(float(np.nanmean(np.abs(cross))), 4),
        "cross_rms_m": round(float(np.sqrt(np.nanmean(cross ** 2))), 4),
        "longitudinal_share_of_sq_err": round(float(
            np.nansum(along ** 2) / max(1e-9, np.nansum(along ** 2)
                                        + np.nansum(cross ** 2))), 4),
    }


def ratio_stats(pw):
    """The UNCLAMPED ratio the published term throws away — the quantity that
    explains the blindness."""
    sc = PS.score_windows(pw, progress_term="clamp_v1")
    r = np.asarray(sc["ego_progress_raw_ratio"], float)
    hu = np.isfinite(np.asarray(sc["ego_progress"], float))   # the scored rows
    r = r[hu]
    if r.size == 0:
        return None
    return {
        "n_scored_rows": int(r.size),
        "frac_under_0p95": round(float((r < 0.95).mean()), 4),
        "frac_within_5pct": round(float(((r >= 0.95) & (r <= 1.05)).mean()), 4),
        "frac_over_1p05": round(float((r > 1.05).mean()), 4),
        "median": round(float(np.median(r)), 4),
        "p05": round(float(np.percentile(r, 5)), 4),
        "p95": round(float(np.percentile(r, 95)), 4),
    }


def score_panel(arms, term, probes, n_boot):
    """Per-arm per-window composite under ONE progress term, PANEL-WIDE gate."""
    scores = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term=term)
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = None
        scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in probes]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores)
               for n in arms}
    admissible, why = {}, {}
    for comp in PS.COMPONENT_WEIGHTS:
        bad = [n for n in gate_arms
               if not per_arm[n].get(comp, {}).get("admissible")]
        admissible[comp] = (len(bad) == 0)
        why[comp] = ("admissible for every arm" if not bad else
                     "INADMISSIBLE for " + ", ".join(sorted(bad)) +
                     " -> dropped from EVERY arm")
    vals, cis, refused = {}, {}, {}
    for n in arms:
        pr = dict(per_arm[n])
        for c, ok in admissible.items():
            if not ok and c in pr:
                pr[c] = dict(pr[c], admissible=False,
                             reason="dropped by the PANEL GATE: " + why[c])
        try:
            comp = PS.composite(scores[n], pr, progress_term=term)
            v = comp.pop("value")
            vals[n] = v
            cis[n] = PS._boot(v, arms[n]["eid"], n_boot, 0)
        except PS.VacuousMetric as exc:
            vals[n] = None
            refused[n] = str(exc)[:200]
    return {
        "term": term, "metric_id": PS.metric_id(term),
        "panel_gate": {"admitted": {k: v for k, v in admissible.items() if v},
                       "dropped": {k: why[k] for k, v in admissible.items()
                                   if not v},
                       "probes_excluded_from_the_gate": sorted(
                           set(probes) & set(arms))},
        "per_arm_range": {n: {c: per_arm[n].get(c, {}).get("admissible")
                              for c in PS.COMPONENT_WEIGHTS} for n in arms},
        "composite_ci": cis, "refused": refused,
        "ego_progress_mean": {
            n: round(float(np.nanmean(scores[n]["ego_progress"])), 6)
            for n in arms},
        "recovery_mean": {
            n: round(float(np.nanmean(scores[n]["recovery"])), 6) for n in arms},
        "recovery_defined_frac": {
            n: round(float(np.isfinite(scores[n]["recovery"]).mean()), 6)
            for n in arms},
        "_vals": vals, "_scores": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True,
                    help="directory of pw_*.npz; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--reference", default="cv_holdv0")
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    arms, seen = {}, {}
    for d in a.in_dir:
        for f in sorted(Path(d).glob("pw_*.npz")):
            name = f.name[len("pw_"):-len(".npz")]
            if name in arms:
                raise SystemExit(f"duplicate arm {name}: {seen[name]} vs {f}")
            arms[name] = load_pw(f)
            seen[name] = str(f)
    print(f"[panel] {len(arms)} arms: {sorted(arms)}", flush=True)

    # ---------------- ROW IDENTITY (a paired delta over misaligned rows is a
    # fabricated number) ---------------------------------------------------- #
    ref = a.reference if a.reference in arms else sorted(arms)[0]
    kref = key_of(arms[ref])
    row_id, refused_rows = {}, {}
    for n, pw in list(arms.items()):
        k = key_of(pw)
        ok = (k.shape == kref.shape) and bool((k == kref).all())
        row_id[n] = {"n_rows": int(k.shape[0]), "identical_to_reference": ok}
        if not ok:
            refused_rows[n] = f"row keys differ from {ref}"
            arms.pop(n)
    assert arms, f"every arm refused on row identity: {refused_rows}"
    eid = arms[ref]["eid"]

    res = {
        "_experiment": ("PSS PROGRESS TERM v1 (one-sided, PUBLISHED) vs v2 "
                        "(two-sided) on the identical rows"),
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the pw_*.npz)",
        "_protocol": PS.PROTOCOL,
        "_estimator": (f"taniteval.ci.episode_cluster_bootstrap / "
                       f"paired_episode_cluster_bootstrap (B={a.n_boot}, "
                       f"unit = val episode), paired on IDENTICAL rows"),
        "_refused_estimator": ("overlapping_holdout_se -- it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "_gate": ("PANEL-WIDE only. The per-arm gate is REFUSED, not offered: "
                  "re-timing smooths a plan, comfort jumps 720x, and the source "
                  "stream's own primary verdict flips n.s. -> SEPARATED."),
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "collision_and_ttc": {"emitted": False,
                              "reason": PS.COLLISION_UNAVAILABLE_REASON},
        "arm_sources": seen,
        "row_identity": row_id, "arms_refused_on_row_identity": refused_rows,
        "reference_arm": ref,
        "probes_excluded": list(PROBES),
        "terms": {}, "paired": {}, "decomposition": {}, "ratio": {},
    }
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        out_p.write_text(json.dumps(res, indent=2, default=str),
                         encoding="utf-8")

    # ---------------- axis + ratio diagnostics (term-independent) ----------- #
    for n, pw in arms.items():
        res["decomposition"][n] = decompose(pw)
        res["ratio"][n] = ratio_stats(pw)
    bank()

    # ---------------- the panels ------------------------------------------- #
    terms = list(TERMS) + (list(SENSITIVITY) if a.sensitivity else [])
    panels = {}
    for term in terms:
        p = score_panel(arms, term, PROBES, a.n_boot)
        panels[term] = p
        node = {k: v for k, v in p.items() if not k.startswith("_")}
        ranked = sorted(((n, c["mean"]) for n, c in p["composite_ci"].items()
                         if c and n not in PROBES),
                        key=lambda t: -t[1])
        node["ranking"] = [{"rank": i + 1, "arm": n, "PSS": v}
                           for i, (n, v) in enumerate(ranked)]
        node["rank1"] = ranked[0][0] if ranked else None
        res["terms"][term] = node
        print(f"[{term}] rank1={node['rank1']} "
              f"gate={sorted(node['panel_gate']['admitted'])}", flush=True)
        for r in node["ranking"]:
            print(f"    {r['rank']:2d}. {r['arm']:28s} {r['PSS']}", flush=True)
        bank()

    # ---------------- REPRODUCTION GATE ------------------------------------ #
    rep, worst = {}, 0.0
    for n, want in PUBLISHED_CLAMP_V1.items():
        got = (panels["clamp_v1"]["composite_ci"].get(n) or {}).get("mean")
        if got is None:
            rep[n] = {"published": want, "got": None, "status": "ARM ABSENT"}
            continue
        d = abs(got - want)
        worst = max(worst, d)
        rep[n] = {"published": want, "got": got, "abs_diff": round(d, 6),
                  "status": "OK" if d <= 5e-4 else "MISMATCH"}
    res["REPRODUCTION_GATE_clamp_v1"] = {
        "rule": ("every PUBLISHED clamp_v1 composite must reproduce to 4 dp "
                 "from these dumps. If it does not, NOTHING here is admissible "
                 "— the recomputation, not the metric, is wrong."),
        "max_abs_diff": round(worst, 6),
        "PASS": bool(worst <= 5e-4),
        "per_arm": rep,
    }
    print(f"[repro] max|diff| vs published clamp_v1 = {worst:.6f} "
          f"PASS={res['REPRODUCTION_GATE_clamp_v1']['PASS']}", flush=True)
    bank()

    # ---------------- paired contrasts ------------------------------------- #
    real = [n for n in sorted(arms) if n not in PROBES]
    #: the pre-registered contrasts + every arm against the incumbent rank-1
    wanted = [(x, ref) for x in real if x != ref] + [
        ("v1_ego_v0", "v1_tactical_follow"),      # the source stream's primary
        ("v1_ego_v0", "v1_tactical_oracle"),
        ("v1_ego_oracle_lon", "v1_tactical_follow"),
        ("v1_ego_oracle_lon", "cv_holdv0"),
        ("refc_xl_v0on", "refc_xl_v0off"),        # Block A, the degradation
        ("refc_base_v0on", "refc_base_v0off"),
        ("v1_lat_straight", "v1_tactical_follow"),
        ("v4_oracle", "v4_blind"),                # G1 instrument sensitivity
        ("v1_ego_half", "v1_tactical_follow"),    # the degradation guard
    ]
    for term in terms:
        p = panels[term]
        blk = {}
        for x, y in wanted:
            if x not in arms or y not in arms:
                continue
            if p["_vals"].get(x) is None or p["_vals"].get(y) is None:
                blk[f"{x}__minus__{y}"] = {"REFUSED": "one arm has no composite"}
                continue
            d = paired(p["_vals"][x], p["_vals"][y], eid, a.n_boot)
            dep = paired(p["_scores"][x]["ego_progress"],
                         p["_scores"][y]["ego_progress"], eid, a.n_boot)
            blk[f"{x}__minus__{y}"] = {"PSS": d, "ego_progress": dep}
            print(f"[{term}] {x} - {y}: PSS {d['delta']} "
                  f"[{d['lo']},{d['hi']}] sep={d['separated']}", flush=True)
        res["paired"][term] = blk
        bank()

    res["_elapsed_s"] = round(time.time() - t0, 1)
    bank()
    print(f"[panel] wrote {a.out} in {res['_elapsed_s']} s")
    print("RECOMPUTE_PANEL_DONE", flush=True)


if __name__ == "__main__":
    main()
