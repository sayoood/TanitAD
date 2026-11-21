"""PREREG_V8F_CZ stage 2, committed before its data: does training T to write 0 in ABSENT constraint slots make v8F-mini drive better?

  python v8f_cz_read.py --eval-dir <eval> --runs-dir <runs> --out cz_read.json

Arms: M2b (PREREG_V8F_MINI Amendment 4, banked) and M2b-cz = M2b with ONE lever moved (`--arm M2b-cz`: cons_absent zero), seeds 0, 1,
the same 4,512 clips, the RA-1 holdout withheld, the same step budget. Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows,
141 episodes) -- the programme's standard v8F-mini eval, stamped "T1-class open loop" as in v8f_mini_full_verdict.py.

  PRIMARY  per seed s: delta_s = ADE6(M2b-cz_s) - ADE6(M2b_s), paired over episodes.
           SUCCESS iff delta_s < 0, separated and |delta_s| > 2F at BOTH seeds, F = |mean ADE6 M2b_s0 - M2b_s1| (M2b's seed floor).
  MECHANISM (deciding the reading of a FAIL, not the verdict): the absent-slot oracle's code shift (`--tac absmask`, mean |code -
           code0|) on M2b-cz is at most HALF of M2b's at both seeds -- T now writes ~0 where a constraint is absent.
  Verdicts: SUCCESS | FAIL-MECHANISM-OK (the fix took but did not carry: next lever = present-slot accuracy) | FAIL-NO-MECHANISM (the
           fix did not take: the loss term is too weak or wrong -- diagnose before any other arm).
Reported beside it, deciding nothing: LON / LAT family deltas (paired, on the same windows), each run's eval-json families incl. the
TACTICAL and STRATEGIC blocks (STRATEGIC is un-paused for v8F-mini, PI ruling 2026-10-17), and the residual absmask headroom on M2b-cz.
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts; an
M2b-cz run whose built lever is not cons_absent zero, or an M2b run whose is not mask; evals not on one grid.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES); training-seed variance is answered by F alone."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
N_TRAIN = 4512
ARMS, SEEDS = ("M2b", "M2b-cz"), (0, 1)
WANT_LEVER = {"M2b": "mask", "M2b-cz": "zero"}
B = 2000


def refuse(msg):
    print(f"CZ READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_runs(runs_dir):
    S = {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(runs_dir, f"{arm}_s{s}", "summary.json")
            if not os.path.exists(p):
                refuse(f"{p} is missing (the run did not finish)")
            S[(arm, s)] = json.load(open(p, encoding="utf-8"))
    bad = []
    if len({d["train_sha256"] for d in S.values()}) != 1:
        bad.append("the runs did not train on one clip list")
    if {d.get("excluded_sha256") for d in S.values()} != {HOLDOUT_SHA}:
        bad.append("a run did not withhold the RA-1 holdout")
    if {d["n_clips"] for d in S.values()} != {N_TRAIN}:
        bad.append(f"n_clips {sorted({d['n_clips'] for d in S.values()})} != {N_TRAIN}")
    if not all(d.get("done") is True for d in S.values()):
        bad.append("a run is not done")
    if len({d["steps"] for d in S.values()}) != 1:
        bad.append(f"the runs trained for different step counts: {sorted({d['steps'] for d in S.values()})}")
    for (arm, s), d in S.items():
        if d["arm"] != arm or d["seed"] != s:
            bad.append(f"{arm}_s{s}/summary.json says {d['arm']} seed {d['seed']}")
        got = (d.get("levers") or {}).get("cons_absent", "mask")             # runs built before the lever existed are mask by construction
        if got != WANT_LEVER[arm]:
            bad.append(f"{arm}_s{s} was built with cons_absent {got}, expected {WANT_LEVER[arm]}")
    if bad:
        refuse("; ".join(bad))
    return S


def load_evals(eval_dir):
    Z, J, A, eid = {}, {}, {}, None
    for arm in ARMS:
        for s in SEEDS:
            t = f"{arm}_s{s}"
            pz, pj, pa = (os.path.join(eval_dir, f"eval_{t}.npz"), os.path.join(eval_dir, f"eval_{t}.json"),
                          os.path.join(eval_dir, f"eval_{t}-tacabs.json"))
            for p in (pz, pj, pa):
                if not os.path.exists(p):
                    refuse(f"{os.path.basename(p)} is missing from {eval_dir}")
            z = np.load(pz)
            if eid is None:
                eid = np.asarray(z["eid"])
            elif not np.array_equal(np.asarray(z["eid"]), eid):
                refuse(f"eval_{t}.npz is not on the same eval grid (eid differs)")
            Z[(arm, s)] = {"ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64)}
            J[(arm, s)] = json.load(open(pj, encoding="utf-8")).get("raw", {})
            m = json.load(open(pa, encoding="utf-8")).get("tac_manipulation") or {}
            if m.get("mode") != "absmask":
                refuse(f"eval_{t}-tacabs.json is not an absmask run")
            A[(arm, s)] = m
    return Z, J, A, eid


def read(a, pairfn, fam):
    load_runs(a.runs_dir)
    Z, J, A, eid = load_evals(a.eval_dir)
    F = abs(Z[("M2b", 0)]["ade6"].mean() - Z[("M2b", 1)]["ade6"].mean())
    rows = {}
    for s in SEEDS:
        d = pairfn(Z[("M2b-cz", s)]["ade6"], Z[("M2b", s)]["ade6"], eid, a.n_boot, 0)
        sh = {arm: float(A[(arm, s)]["code_shift_mean"]) for arm in ARMS}
        rows[f"s{s}"] = dict(delta=d, ade6={arm: float(Z[(arm, s)]["ade6"].mean()) for arm in ARMS}, absmask_code_shift=sh,
                             mechanism=bool(sh["M2b-cz"] <= 0.5 * sh["M2b"]),
                             eval_json={arm: {k: J[(arm, s)].get(k) for k in ("traj", "families", "strategic", "tactical") if k in J[(arm, s)]}
                                        for arm in ARMS})
        if fam is not None:
            rows[f"s{s}"]["families"] = fam(Z[("M2b-cz", s)]["pred"], Z[("M2b", s)]["pred"], eid)
    success = all(rows[f"s{s}"]["delta"]["delta"] < 0 and rows[f"s{s}"]["delta"]["separated"]
                  and abs(rows[f"s{s}"]["delta"]["delta"]) > 2 * F for s in SEEDS)
    mech = all(rows[f"s{s}"]["mechanism"] for s in SEEDS)
    v = "SUCCESS" if success else ("FAIL-MECHANISM-OK" if mech else "FAIL-NO-MECHANISM")
    nxt = {"SUCCESS": "SUCCESS: M2b-cz drives better than M2b beyond 2F at both seeds. Next: the T1 closed-loop read, then the remaining "
                      "present-slot headroom (Amendment 2's dose-response) on top of M2b-cz.",
           "FAIL-MECHANISM-OK": "FAIL with the mechanism in place: T now writes ~0 in absent slots but driving did not improve beyond 2F. "
                                "Next lever: T's PRESENT-slot accuracy, under its own PREREG.",
           "FAIL-NO-MECHANISM": "FAIL without the mechanism: T still writes large values in absent slots. Diagnose the loss term (weight, "
                                "masking, the has-row mask) before any other arm."}[v]
    out = dict(schema="v8f-cz-read-v1", prereg="PREREG_V8F_CZ.md stage 2", tier="T1-class open loop (eval-141, 2,399 windows)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |M2b_s0 - M2b_s1| = {F:.4f} m",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), F=F, two_F=2 * F, rows=rows, VERDICT=v, mechanism=mech, next=nxt)
    out["reading"] = (f"PREREG_V8F_CZ stage 2: {v}; M2b-cz - M2b ADE6 " +
                      " / ".join(f"{rows[f's{s}']['delta']['delta']:+.3f}{'' if rows[f's{s}']['delta']['separated'] else ' ns'}" for s in SEEDS) +
                      f" m (2F {2 * F:.4f}); absmask code shift M2b-cz / M2b " +
                      " | ".join(f"{rows[f's{s}']['absmask_code_shift']['M2b-cz']:.3f}/{rows[f's{s}']['absmask_code_shift']['M2b']:.3f}"
                                 for s in SEEDS) + ".")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--no-families", action="store_true", help="skip the descriptive family blocks (tests)")
    ap.add_argument("--eval-grid", default=str(HERE.parents[1] / "raw" / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(HERE.parents[1] / "raw" / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(HERE.parents[1] / "raw" / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt in place of the rebuild")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("v8f_tac_groups_read_for_cz2", HERE.parent / "v8f_tac_groups_read.py")
    R1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R1)
    R1._paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap, None if a.no_families else R1.family_fn(a))
    except (OSError, KeyError, ValueError) as e:
        print(f"CZ READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
