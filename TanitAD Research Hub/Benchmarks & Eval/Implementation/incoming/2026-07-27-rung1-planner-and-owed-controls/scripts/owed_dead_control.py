#!/usr/bin/env python3
"""OWED ROWS 49-51 — the `DEAD-CONTROL` family, re-adjudicated. ZERO GPU.

⚠️ THE FIRST FINDING IS THAT THE FROZEN LIST HAS THIS FAMILY'S DIRECTION INVERTED.
`CONTROL_READJUDICATION.md` §1 describes `DEAD-CONTROL` as *"a perturbation with
no physical content has NO effect => the envelope is real"*, i.e. it treats the
null as the DESIRED verdict. In this codebase the opposite is pre-registered:

    scripts/lowood_ci_yawext.py:54-55
    # The DESTROYED-OBSERVATION controls. Each must land FAR above the 12 deg
    # value or C-ADE has no dynamic range and outcome C is declared.

    P1_REVALIDATION.md:77-79
    "Pre-committed rule: if the destroyed controls are not CI-separated far above
     the 12 deg value, C-ADE has no dynamic range, every P1 envelope number lies
     inside its noise floor, and outcome C is declared."

So these are **dynamic-range controls that must FIRE**, and a null on them is a
failure of the criterion, not a clean bill of health. That inversion changes what
"re-powering" even means, so it is established from the source before any number.

⭐ AND THE PRE-REGISTERED COMPARISON WAS NEVER RUN ON THIS AXIS. The rule is
"far above the **12 deg value**", but every published dead-control interval is
paired against the **Delta=0 baseline**, not against `yaw12`. The contrast the
rule names is computed here for the first time, on all three dead kinds and on
all three metrics, from per-window data already in the repo.

M9 — the whole artifact, not the matching node:
  * row 49 (`yawext_12ep :: dead_noise.paired_along_2s`, n=12) is answered by its
    own 40-episode sibling in the SAME artifact set at 3.3x the clusters;
  * rows 50 and 51 are the SAME contrast at two episode counts;
  * `paired_cross_2s` is a sibling axis that separates where along-track does not;
  * `conditions.yaw[15]` (psi = 90 deg) is a fourth dynamic-range control on the
    same footing, and its along-track row is ALSO not separated -- so three of
    four destroyed-observation controls fail on this axis.

⛔ No estimator is re-implemented: `taniteval.ci.paired_episode_cluster_bootstrap`
is called directly on the committed per-window arrays. Validated in BOTH
directions (`selftest`): it must reproduce the committed intervals AND it must
fire on a planted shift.

Usage:
    python owed_dead_control.py --p1 <p1-envelope-revalidation dir> --out <raw dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from taniteval import ci as _ci                              # noqa: E402

B_BOOT, SEED = 2000, 0
DEAD = ("dead_black", "dead_noise", "dead_shuffle")
#: the pre-registered reference the dead controls must land FAR ABOVE
P1_GRID_END = "yaw12"
METRICS = ("ade", "along", "cross")


def paired(a, b, eid, label, reduce="mean"):
    r = _ci.paired_episode_cluster_bootstrap(np.asarray(a, float),
                                             np.asarray(b, float),
                                             [str(e) for e in eid],
                                             n_boot=B_BOOT, seed=SEED,
                                             reduce=reduce)
    r["contrast"] = label
    r["ci95"] = round(float((r["hi"] - r["lo"]) / 2.0), 6)
    r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
    return r


def power_ceiling(mde: float, effect_to_catch: float, what: str) -> dict:
    """M8 — the MDE stated against the effect the control EXISTS to catch.

    Unlike a bounded metric there is no closed-form maximum here (metres), so
    `can_fire` is expressed against the named effect rather than against a
    physical bound, and that is said rather than implied.
    """
    return {
        "mde": round(float(mde), 6),
        "effect_it_exists_to_catch": round(float(effect_to_catch), 6),
        "effect_it_exists_to_catch_is": what,
        "mde_as_pct_of_that_effect": (None if effect_to_catch == 0 else
                                      round(100.0 * mde / abs(effect_to_catch), 1)),
        "sharp_enough": bool(abs(effect_to_catch) > mde),
        "_read": ("the control is sharper than the effect it guards -- a null "
                  "here carries evidence"
                  if abs(effect_to_catch) > mde else
                  "the control is BLUNTER than the effect it guards -- a null "
                  "here is 'not run', not 'passed'"),
    }


def selftest(z, eid) -> dict:
    t = {}
    a = np.asarray(z["along__dead_shuffle"], float)
    same = paired(a, a, eid, "selftest_identical")
    shift = paired(a + 0.25, a, eid, "selftest_shift_0.25")
    t["identical_not_separated"] = (not same["separated"]) and abs(same["delta"]) < 1e-12
    t["shifted_separated"] = bool(shift["separated"]
                                  and abs(shift["delta"] - 0.25) < 1e-6)
    t["ALL_PASS"] = all(bool(v) for k, v in t.items() if k != "ALL_PASS")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p1 = Path(a.p1).resolve()
    outd = Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    j12 = json.load(open(p1 / "artifacts" / "yawext_12ep.json", encoding="utf-8"))
    j40 = json.load(open(p1 / "artifacts" / "yawext_40ep.json", encoding="utf-8"))
    z = np.load(p1 / "artifacts" / "yawext_40ep_perwindow.npz")
    eid = [str(e) for e in z["eid"]]

    st = selftest(z, eid)
    print("[selftest]", json.dumps(st), flush=True)
    if not st["ALL_PASS"]:
        (outd / "owed_dead_control.json").write_text(
            json.dumps({"selftest": st, "ABORTED": True}, indent=2), encoding="utf-8")
        return 3

    def cond(js, kind):
        return js["conditions"][kind][0]

    out = {
        "block": "owed_controls/DEAD-CONTROL",
        "frozen_list_rows": [49, 50, 51],
        "estimator": {"interval": "paired_episode_cluster_bootstrap",
                      "module": "taniteval.ci", "n_boot": B_BOOT, "seed": SEED,
                      "resampling_unit": "val episode"},
        "selftest": st,
        "DIRECTION_CORRECTION": {
            "frozen_list_says": ("a perturbation with no physical content has NO "
                                 "effect => the envelope is real (null = desired)"),
            "the_source_says": ("each must land FAR ABOVE the 12 deg value or "
                                "C-ADE has no dynamic range and outcome C is "
                                "declared (null = the criterion FAILED)"),
            "source": ["scripts/lowood_ci_yawext.py:54-55",
                       "P1_REVALIDATION.md:77-79"],
            "consequence": ("for this family a NULL is a failing verdict, not a "
                            "passing one; 'not separated' means the criterion "
                            "has no dynamic range on that axis"),
        },
        "n": {"12ep": {"n_windows": cond(j12, "dead_noise")["paired_along_2s"]["n_windows"],
                       "n_episodes": cond(j12, "dead_noise")["paired_along_2s"]["n_episodes"]},
              "40ep": {"n_windows": int(len(eid)), "n_episodes": int(len(set(eid)))}},
        "fidelity_vs_committed": {},
        "rows": {},
        "never_run_contrast_vs_p1_grid_end": {},
        "fourth_dynamic_range_control_yaw90": {},
    }

    # ---- fidelity: the committed intervals must come back out of the npz ----- #
    fid = {}
    for kind in DEAD:
        c = cond(j40, kind)["paired_along_2s"]
        r = paired(z[f"along__{kind}"], z["along__baseline"], eid,
                   f"along__{kind} - baseline")
        fid[f"40ep.{kind}.paired_along_2s"] = {
            "committed": {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                          "separated": c["separated"]},
            "recomputed": {"delta": round(r["delta"], 4), "lo": round(r["lo"], 4),
                           "hi": round(r["hi"], 4), "separated": r["separated"]},
            "matches": bool(abs(r["delta"] - c["delta"]) < 5e-4
                            and abs(r["lo"] - c["lo"]) < 5e-3
                            and abs(r["hi"] - c["hi"]) < 5e-3
                            and r["separated"] == c["separated"]),
        }
    fid["REPRODUCES"] = all(v["matches"] for v in fid.values() if isinstance(v, dict))
    out["fidelity_vs_committed"] = fid
    print(f"[fidelity] REPRODUCES = {fid['REPRODUCES']}", flush=True)

    # ---- ⭐ the contrast the pre-registration NAMES and nobody ran ------------ #
    for kind in DEAD:
        blk = {}
        for m in METRICS:
            blk[m] = paired(z[f"{m}__{kind}"], z[f"{m}__{P1_GRID_END}"], eid,
                            f"{m}__{kind} - {m}__{P1_GRID_END}")
        out["never_run_contrast_vs_p1_grid_end"][kind] = blk
        print(f"[vs {P1_GRID_END}] {kind:12s} "
              + "  ".join(f"{m}={blk[m]['delta']:+.4f} "
                          f"sep={str(blk[m]['separated']):5s}" for m in METRICS),
              flush=True)

    # ---- the fourth control on the same footing (M9: read the whole artifact) - #
    for m in METRICS:
        out["fourth_dynamic_range_control_yaw90"][m] = {
            "vs_baseline": paired(z[f"{m}__yaw90"], z[f"{m}__baseline"], eid,
                                  f"{m}__yaw90 - {m}__baseline"),
            "vs_p1_grid_end": paired(z[f"{m}__yaw90"], z[f"{m}__{P1_GRID_END}"],
                                     eid, f"{m}__yaw90 - {m}__{P1_GRID_END}"),
        }

    # ---- the three frozen rows ---------------------------------------------- #
    rows = [
        (49, "12ep", "dead_noise", cond(j12, "dead_noise")["paired_along_2s"]),
        (50, "40ep", "dead_shuffle", cond(j40, "dead_shuffle")["paired_along_2s"]),
        (51, "12ep", "dead_shuffle", cond(j12, "dead_shuffle")["paired_along_2s"]),
    ]
    for idx, dep, kind, committed in rows:
        sib = cond(j40, kind)["paired_along_2s"]        # the 40-ep sibling
        higher_n = (dep == "12ep")
        # the effect the control exists to catch: separation ABOVE the 12 deg
        # grid end, measured on this axis for the first time
        eff = out["never_run_contrast_vs_p1_grid_end"][kind]["along"]["delta"]
        pc = power_ceiling(committed["ci95"], eff,
                           f"the dead-vs-{P1_GRID_END} along-track gap the "
                           f"pre-registration requires to be separated")
        if higher_n and sib["separated"]:
            verdict = "RESOLVED BY A HIGHER-n SIBLING IN THE SAME ARTIFACT SET"
            why = (f"the same contrast at n=40 clusters ({sib['delta']:+.4f} "
                   f"[{sib['lo']:+.4f}, {sib['hi']:+.4f}]) IS separated; the "
                   f"n=12 null was a power artefact")
        elif higher_n:
            verdict = "OWED — NOT RE-POWERABLE HERE"
            why = (f"the same contrast at n=40 ({sib['delta']:+.4f} "
                   f"[{sib['lo']:+.4f}, {sib['hi']:+.4f}]) is ALSO not separated, "
                   f"so 3.3x the clusters did not resolve it")
        elif not committed["separated"]:
            verdict = "CONTROL FAILS (dynamic-range control did NOT fire)"
            why = ("this is a control that MUST separate; it does not, so the "
                   "along-track component of C-ADE has no demonstrated dynamic "
                   "range for this perturbation")
        else:
            verdict = "HOLDS (POWERED)"
            why = "the dynamic-range control fired as required"
        # ⚠️ SCOPE, stated so the verdict is not over-read: the pre-registered
        # gate (P1_REVALIDATION §1.3) is on the ADE headline, which separates in
        # every dead cell. `paired_along_2s` is the DECOMPOSITION the report
        # added under its own design rule 3 and is consumed by no downstream
        # script (`yaw_edge_analysis.py` reads only `ade__*`). So a failing
        # verdict here scopes to the decomposition, not to the P1 gate.
        scope = {
            "the_preregistered_gate_is_on": "ade (paired_vs_baseline)",
            "ade_gate_separated": bool(
                cond(j40 if dep == "40ep" else j12, kind)
                ["paired_vs_baseline"]["separated"]),
            "this_node_is": "the along-track DECOMPOSITION of that control",
            "consumed_downstream_by": "nothing (yaw_edge_analysis.py reads ade__* only)",
            "so_this_verdict_withdraws": "no published P1 claim",
        }
        # what n would be needed, from the observed half-width scaling
        need = None
        if not committed["separated"] and abs(committed["delta"]) > 0:
            need = int(np.ceil((committed["ci95"] / abs(committed["delta"])) ** 2
                               * committed["n_episodes"]))
        out["rows"][str(idx)] = {
            "deployment": dep, "kind": kind, "node": "paired_along_2s",
            "committed": {k: committed[k] for k in
                          ("delta", "lo", "hi", "ci95", "separated",
                           "n_windows", "n_episodes", "estimator")},
            "higher_n_sibling_40ep": {"delta": sib["delta"], "lo": sib["lo"],
                                      "hi": sib["hi"],
                                      "separated": sib["separated"]},
            "sibling_axis_cross_2s": {
                k: cond(j40 if dep == "40ep" else j12, kind)["paired_cross_2s"][k]
                for k in ("delta", "lo", "hi", "separated")},
            "headline_metric_paired_vs_baseline": {
                k: cond(j40 if dep == "40ep" else j12, kind)["paired_vs_baseline"][k]
                for k in ("delta", "lo", "hi", "separated")},
            "power_ceiling": pc,
            "episodes_needed_for_alpha_0.05": need,
            "scope": scope,
            "VERDICT": verdict, "why": why,
        }
        print(f"[row {idx}] {dep} {kind}: {verdict}", flush=True)

    # ---- reducer robustness, free from the same arrays ------------------------ #
    out["reducer_sensitivity_40ep_dead_shuffle_along"] = {
        red: paired(z["along__dead_shuffle"], z["along__baseline"], eid,
                    f"reduce={red}", reduce=red)
        for red in ("mean", "median")}

    (outd / "owed_dead_control.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[write] {outd / 'owed_dead_control.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
