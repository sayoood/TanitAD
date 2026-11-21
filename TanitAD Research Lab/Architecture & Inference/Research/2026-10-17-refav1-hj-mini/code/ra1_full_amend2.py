"""PREREG_RA1_FULL Amendment 2's reading (IDM at gamma = 200, the committed next lever), committed before its data.

  python ra1_full_amend2.py --root <logs/ra1_full> --S 3000 --out amend2_final.json

  IDM-200 (decides): A1g200 - A0, eval-141 ADE@6s, paired per seed (A0 seed k with A1g200 seed k): PASS iff at BOTH seeds it is < 0,
           separated, and |delta| > 2 x max(F_A1g200, F_A0), F = |s0 - s1| on eval-141 -- Amendment 1's bar with A1g200 in A1's place.
  Beside it, deciding nothing: the holdout difference; A1g200 - A1g50 per seed; A1g200's K row vs P10-yaw-4512; its level-1
           yaw-rate sensitivity (k = 10) vs A0's; and the paired LONGITUDINAL / LATERAL family deltas A1g200 - A0 per component
           (`ra1_hjepa_eval.paired_family_block` over the runs' per-window family arrays).
Refusals (rc 2): a missing run; a run whose BUILT cfg.gamma / cfg.gamma_up is not the cell's (200 / 50 for A1g200, 50 / 50 for A1g50,
None for A0) -- a flag that did not reach the model is how a wired arm silently tests nothing; anything `ra1_full_report.check_full`
refuses (one holdout, one train list, one prior, yaw rate, the budget); windows that do not align.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (another draw of EPISODES); F = |s0 - s1| (training seed)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RUNDIR = {"A0": "A0_b0.08_s{s}", "A1g200": "A1_b0.08g200_s{s}", "A1g50": "A1_b0.08g50_s{s}"}
BUILT = {"A0": (None, 50.0), "A1g200": (200.0, 50.0), "A1g50": (50.0, 50.0)}     # (cfg.gamma, cfg.gamma_up) as the model was BUILT
ARM_OF = {"A0": "A0", "A1g200": "A1", "A1g50": "A1"}
B = 2000


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _paths():
    for p in (HERE.parents[5] / "stack", HERE.parents[5] / "taniteval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def read(a, pairfn) -> dict:
    rep = _mod("ra1_full_report", HERE.parent / "ra1_full_report.py")
    base = os.path.join(a.root, "runs", f"st{a.S}")
    runs = {}
    for cell, tmpl in RUNDIR.items():
        for s in (0, 1):
            r = rep.load_run(os.path.join(base, tmpl.format(s=s)))
            if r is not None:
                runs[(cell, s)] = r
    miss = [f"{RUNDIR[c].format(s=s)}" for c in RUNDIR for s in (0, 1) if (c, s) not in runs]
    if miss:
        raise ValueError(f"Amendment 2 needs every run at st{a.S}; missing {miss}")
    for (cell, s), r in runs.items():
        cfg = r["ev"].get("cfg") or {}
        got = (cfg.get("gamma"), cfg.get("gamma_up"))
        if got != BUILT[cell]:
            raise ValueError(f"{r['dir']}: built (gamma, gamma_up) = {got}, the cell {cell} needs {BUILT[cell]}")
    rep.check_full({a.S: {ARM_OF[c]: {s: r for (cc, s), r in runs.items() if cc == c} for c in ("A0", "A1g200")}})
    rep.check_full({a.S: {ARM_OF[c]: {s: r for (cc, s), r in runs.items() if cc == c} for c in ("A0", "A1g50")}})
    e = {c: [runs[(c, s)]["ev"]["K"]["ade6_arm"]["mean"] for s in (0, 1)] for c in RUNDIR}
    F = {c: abs(v[0] - v[1]) for c, v in e.items()}
    two_F = 2 * max(F["A1g200"], F["A0"])
    rows = []
    for s in (0, 1):
        g, z = runs[("A1g200", s)], runs[("A0", s)]
        d = rep.paired_runs(g, z, "eval141", "ade6", "ade6", pairfn)
        rows.append(dict(seed=s, A1g200_minus_A0=d, below=d["delta"] < 0, separated=bool(d["separated"]), beats_2x_floor=abs(d["delta"]) > two_F,
                         holdout_NON_DECIDING=rep.paired_runs(g, z, "holdout", "hold_ade6", "hold_ade6", pairfn),
                         A1g200_minus_A1g50_DESCRIPTIVE=rep.paired_runs(g, runs[("A1g50", s)], "eval141", "ade6", "ade6", pairfn),
                         K_vs_P10_DESCRIPTIVE=g["ev"]["K"].get("ade6_arm_minus_p10"),
                         sens_yaw_rate_k10=dict(A1g200=float(np.mean(g["pw"]["sens_yaw_rate_k10"])), A0=float(np.mean(z["pw"]["sens_yaw_rate_k10"])),
                                                A1g200_minus_A0=rep.paired_runs(g, z, "eval141", "sens_yaw_rate_k10", "sens_yaw_rate_k10", pairfn)),
                         families_A1g200_minus_A0=families(g, z)))
    ok = all(r["below"] and r["separated"] and r["beats_2x_floor"] for r in rows)
    out = dict(schema="ra1-full-amend2-v1", prereg="PREREG_RA1_FULL.md Amendment 2", S=a.S, cell="b0.08g200",
               tier="T1-class open-loop trajectory (sensitivity is T0)", estimator="paired episode-cluster bootstrap, B=2000 (episodes); F = |s0 - s1|",
               IDM200=dict(bar="A1g200 - A0, eval-141 ADE@6s: < 0, separated at BOTH seeds, |delta| > 2 x max(F_A1g200, F_A0)", PASS=ok,
                           floors_eval141=F, two_F=two_F, arm_means_eval141=e, rows=rows))
    if ok:
        out["next"] = ("IDM-200 PASS: IDM at gamma 200 lowers open-loop 6 s error beyond training-seed noise at full scale (not a closed-loop "
                       "claim); gamma 200 is the IDM weight carried into the RA-2 candidate, confirmed under RA-2's own PREREG.")
    else:
        out["next"] = ("IDM-200 FAIL: IDM's trajectory benefit is NOT established at gamma 50 or 200. IDM is kept for its SENS PASS (the "
                       "action-sensitive latent) and dropped as a trajectory-error lever, with no further gamma arms; any later read is SEEDS "
                       "(A0 / A1 seeds 2, 3) under its own amendment.")
    out["reading"] = (f"IDM-200 {'PASS' if ok else 'FAIL'} (A1g200 - A0 " + " / ".join(
        f"{r['A1g200_minus_A0']['delta']:+.4f}{'' if r['separated'] else ' ns'}" for r in rows) + f", 2F {two_F:.4f}; A1g200 - A1g50 "
        + " / ".join(f"{r['A1g200_minus_A1g50_DESCRIPTIVE']['delta']:+.4f}" for r in rows) + ").")
    return out


def families(g, z):
    """A1g200 - A0 by LON / LAT component, paired over episodes, from the runs' own per-window family arrays (fam_arm_*)."""
    _paths()
    E = _mod("ra1_hjepa_eval", HERE.parents[5] / "stack" / "scripts" / "ra1_hjepa_eval.py")
    pa, pb = E.family_pw_from_npz(_npz(g), "arm"), E.family_pw_from_npz(_npz(z), "arm")
    if not pa or not pb:
        return dict(status="UNAVAILABLE", reason="a run carries no per-window family arrays (it predates 6fbd5ba)")
    if not np.array_equal(np.asarray(g["pw"]["eid"]), np.asarray(z["pw"]["eid"])):
        raise ValueError(f"family windows do not align between {g['dir']} and {z['dir']}")
    return E.paired_family_block(pa, pb, np.asarray(g["pw"]["eid"]), B, 0)


class _npz:
    """`family_pw_from_npz` wants an object with `.files` and item access; a loaded run keeps its arrays as a dict."""
    def __init__(self, r):
        self.d = r["pw"]
        self.files = list(self.d)

    def __getitem__(self, k):
        return self.d[k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap)
    except (OSError, KeyError, ValueError) as e:
        print(f"AMEND2 INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
