"""PREREG_V8F_WC, committed before its data: does up-weighting T's constraint loss (w_cons 0.5 -> 4.0, ONE lever moved vs M2b) make
T's PRESENT constraint slots more accurate, and v8F-mini drive better?

  python v8f_wc_read.py --stage s0|final --eval-dir <eval> --runs-dir <runs> --out wc_read.json

Arms: M2b (PREREG_V8F_MINI Amendment 4, banked) and M2b-wc = `--arm M2b --w-cons 4.0` (queue stage `trainwc`), the same 4,512 clips,
the RA-1 holdout withheld, M2b's 19,164 steps. Eval: `v8f_mini_eval.py run` on eval-141 (2,399 windows, 141 episodes), stamped
"T1-class open loop" as in v8f_mini_full_verdict.py.

  PRIMARY   per seed s: delta_s = ADE6(M2b-wc_s) - ADE6(M2b_s), paired over episodes. A seed PASSES iff delta_s < 0, separated and
            |delta_s| > 2F, F = |mean ADE6 M2b_s0 - M2b_s1| (M2b's seed floor, 0.1207 m on the banked evals).
  MECHANISM per seed s (deciding the reading of a FAIL, never the verdict): T's present-slot constraint error, the per-window mean of
            (T - GT)^2 in SD units over the slots where the GT constraint is PRESENT (GT code != 0, windows with a v9 row), is LOWER on
            M2b-wc than on M2b: delta < 0 and separated. T's values come from the absmask eval (its present slots are T's own, untouched);
            the GT from the GT-oracle eval. The two arms' GT codes must be identical.
  STAGES    s0: seed 0 only. CONTINUE iff seed 0 PASSES (only then is seed 1 trained); otherwise STOP -- one failing seed already fails a
            criterion that needs both. final: both seeds.
  VERDICTS  SUCCESS (both seeds pass) | FAIL-MECHANISM-OK (T's present slots improved at every seed read, driving did not) |
            FAIL-NO-MECHANISM (the weight did not make T more accurate).
Reported beside it, deciding nothing: LON / LAT family deltas (paired, same windows), each run's eval-json families incl. TACTICAL and
STRATEGIC (un-paused for v8F-mini, PI ruling 2026-10-17), per-group present-slot RMSE (latcons / lonspeed / stoplead).
Refusals (rc 2): a missing file; runs not on one clip list / not withholding the RA-1 holdout / not done / different step counts; an
M2b-wc run whose argv is not `--arm M2b ... --w-cons 4.0` or whose built w_cons is not 4.0 or whose selfcheck is not PASS; an M2b run
with any --w-cons in its argv; evals not on one grid; GT codes that differ between the arms.
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
WC = 4.0
M2B_W_CONS = 0.5
ARMS = ("M2b", "M2b-wc")
C0 = 22 + 5 + 7                     # N_GOAL + N_LAT + N_LON: the constraint slots follow the decisions
N_CONS = 26
# offsets inside the constraint block, as v8f_mini.CODE_GROUPS has them (NOT contiguous: lonspeed is 12-15 + 22-25); pinned by
# test_v8f_wc.py::test_reader_layout_matches_the_model
GROUPS = {"latcons": tuple(range(0, 12)), "lonspeed": (12, 13, 14, 15, 22, 23, 24, 25), "stoplead": (16, 17, 18, 19, 20, 21)}
B = 2000


def refuse(msg):
    print(f"WC READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _flag(argv, name):
    if name not in argv:
        return None
    i = argv.index(name)
    return argv[i + 1] if i + 1 < len(argv) else ""


def load_runs(runs_dir, seeds):
    S, bad = {}, []
    for arm in ARMS:
        for s in seeds:
            d = os.path.join(runs_dir, f"{arm}_s{s}")
            for f in ("summary.json", "argv.json", "selfcheck.json"):
                if not os.path.exists(os.path.join(d, f)):
                    refuse(f"{os.path.join(d, f)} is missing (the run did not finish, or was not pulled)")
            S[(arm, s)] = {f: json.load(open(os.path.join(d, f), encoding="utf-8")) for f in ("summary.json", "argv.json", "selfcheck.json")}
    sm = {k: v["summary.json"] for k, v in S.items()}
    if len({d["train_sha256"] for d in sm.values()}) != 1:
        bad.append("the runs did not train on one clip list")
    if {d.get("excluded_sha256") for d in sm.values()} != {HOLDOUT_SHA}:
        bad.append("a run did not withhold the RA-1 holdout")
    if {d["n_clips"] for d in sm.values()} != {N_TRAIN}:
        bad.append(f"n_clips {sorted({d['n_clips'] for d in sm.values()})} != {N_TRAIN}")
    if not all(d.get("done") is True for d in sm.values()):
        bad.append("a run is not done")
    if len({d["steps"] for d in sm.values()}) != 1:
        bad.append(f"the runs trained for different step counts: {sorted({d['steps'] for d in sm.values()})}")
    for (arm, s), v in S.items():
        d, argv = v["summary.json"], v["argv.json"].get("argv", [])
        if d["arm"] != "M2b" or d["seed"] != s:
            bad.append(f"{arm}_s{s}/summary.json says {d['arm']} seed {d['seed']}, expected M2b seed {s}")
        if v["selfcheck.json"].get("declared_vs_built") != "PASS":
            bad.append(f"{arm}_s{s}: the launch gate's declared_vs_built is not PASS")
        if _flag(argv, "--arm") != "M2b":
            bad.append(f"{arm}_s{s}: argv --arm {_flag(argv, '--arm')}, expected M2b")
        w = ((d.get("levers") or {}).get("w") or {}).get("w_cons")
        if arm == "M2b-wc":
            got = _flag(argv, "--w-cons")
            if got is None or float(got) != WC:
                bad.append(f"M2b-wc_s{s}: argv --w-cons {got}, expected {WC}")
            if w is None or float(w) != WC:
                bad.append(f"M2b-wc_s{s} was built with w_cons {w}, expected {WC}")
        else:
            if _flag(argv, "--w-cons") is not None:
                bad.append(f"M2b_s{s}: argv carries --w-cons {_flag(argv, '--w-cons')}; M2b is the default {M2B_W_CONS}")
            if w is not None and float(w) != M2B_W_CONS:                # runs built before the key existed are 0.5 by construction
                bad.append(f"M2b_s{s} was built with w_cons {w}, expected {M2B_W_CONS}")
    if bad:
        refuse("; ".join(bad))
    return S


def load_evals(eval_dir, seeds):
    Z, J, eid = {}, {}, None
    for arm in ARMS:
        for s in seeds:
            t = f"{arm}_s{s}"
            p = {k: os.path.join(eval_dir, f"eval_{t}{suf}") for k, suf in
                 (("z", ".npz"), ("j", ".json"), ("abs", "-tacabs.npz"), ("orc", "-tacorc.npz"), ("absj", "-tacabs.json"))}
            for q in p.values():
                if not os.path.exists(q):
                    refuse(f"{os.path.basename(q)} is missing from {eval_dir}")
            z, za, zo = np.load(p["z"]), np.load(p["abs"]), np.load(p["orc"])
            for name, zz in (("", z), ("-tacabs", za), ("-tacorc", zo)):
                if eid is None:
                    eid = np.asarray(zz["eid"])
                elif not np.array_equal(np.asarray(zz["eid"]), eid):
                    refuse(f"eval_{t}{name}.npz is not on the same eval grid (eid differs)")
            if (json.load(open(p["absj"], encoding="utf-8")).get("tac_manipulation") or {}).get("mode") != "absmask":
                refuse(f"eval_{t}-tacabs.json is not an absmask run")
            Z[(arm, s)] = {"ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64),
                           "t_cons": np.asarray(za["tac_code"], np.float64)[:, C0:C0 + N_CONS],
                           "g_code": np.asarray(zo["tac_code"], np.float64), "has": ~np.asarray(zo["tac_oracle_fallback"], bool)}
            J[(arm, s)] = json.load(open(p["j"], encoding="utf-8")).get("raw", {})
    return Z, J, eid


def seed_floor(eval_dir, eid):
    """F = |mean ADE6 M2b_s0 - M2b_s1|, always from the two banked M2b evals (stage s0 reads only seed 0 of each arm otherwise)."""
    m = []
    for s in (0, 1):
        p = os.path.join(eval_dir, f"eval_M2b_s{s}.npz")
        if not os.path.exists(p):
            refuse(f"{os.path.basename(p)} is missing from {eval_dir} (M2b's seed floor needs both seeds)")
        z = np.load(p)
        if not np.array_equal(np.asarray(z["eid"]), eid):
            refuse(f"eval_M2b_s{s}.npz is not on the same eval grid (eid differs)")
        m.append(float(np.asarray(z["ade6"], np.float64).mean()))
    return abs(m[0] - m[1])


def present_error(Z, seeds):
    """Per window: the mean squared error of T's PRESENT constraint slots (SD units); NaN where a window has no present slot."""
    ref = Z[("M2b", seeds[0])]
    has, G = ref["has"], ref["g_code"]
    for k, v in Z.items():
        if not np.array_equal(v["has"], has) or not np.array_equal(v["g_code"][has], G[has]):
            refuse(f"{k[0]}_s{k[1]}: the GT code differs from M2b_s{seeds[0]}'s -- not the same labels on the same windows")
    g = G[:, C0:C0 + N_CONS]
    pres = has[:, None] & (g != 0.0)
    out = {}
    for k, v in Z.items():
        e2 = np.where(pres, (v["t_cons"] - g) ** 2, 0.0)
        n = pres.sum(1)
        out[k] = {"mse": np.where(n > 0, e2.sum(1) / np.maximum(n, 1), np.nan),
                  "rmse_group": {gname: float(np.sqrt(e2[:, list(r)][pres[:, list(r)]].mean())) if pres[:, list(r)].any() else None
                                 for gname, r in GROUPS.items()}}
    return out, pres


def read(a, pairfn, fam):
    seeds = (0,) if a.stage == "s0" else (0, 1)
    load_runs(a.runs_dir, seeds)
    Z, J, eid = load_evals(a.eval_dir, seeds)
    F = seed_floor(a.eval_dir, eid)                                      # from BOTH banked M2b evals, at either stage
    PE, pres = present_error(Z, seeds)
    ok = np.isfinite(PE[("M2b", seeds[0])]["mse"])
    rows = {}
    for s in seeds:
        d = pairfn(Z[("M2b-wc", s)]["ade6"], Z[("M2b", s)]["ade6"], eid, a.n_boot, 0)
        dm = pairfn(PE[("M2b-wc", s)]["mse"][ok], PE[("M2b", s)]["mse"][ok], eid[ok], a.n_boot, 0)
        passed = bool(d["delta"] < 0 and d["separated"] and abs(d["delta"]) > 2 * F)
        mech = bool(dm["delta"] < 0 and dm["separated"])
        rows[f"s{s}"] = dict(delta=d, passed=passed, ade6={arm: float(Z[(arm, s)]["ade6"].mean()) for arm in ARMS},
                             present_mse=dm, mechanism=mech,
                             present_rmse={arm: float(np.sqrt(PE[(arm, s)]["mse"][ok].mean())) for arm in ARMS},
                             present_rmse_group={arm: PE[(arm, s)]["rmse_group"] for arm in ARMS},
                             eval_json={arm: {k: J[(arm, s)].get(k) for k in ("traj", "families", "strategic", "tactical")
                                              if k in J[(arm, s)]} for arm in ARMS})
        if fam is not None:
            rows[f"s{s}"]["families"] = fam(Z[("M2b-wc", s)]["pred"], Z[("M2b", s)]["pred"], eid)
    mech_all = all(rows[f"s{s}"]["mechanism"] for s in seeds)
    if a.stage == "s0":
        go = rows["s0"]["passed"]
        decision = "CONTINUE" if go else "STOP"
        verdict = None if go else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")
    else:
        decision = None
        verdict = "SUCCESS" if all(rows[f"s{s}"]["passed"] for s in seeds) else ("FAIL-MECHANISM-OK" if mech_all else "FAIL-NO-MECHANISM")
    nxt = {None: "CONTINUE: seed 0 passes; train seed 1 (the verdict needs both).",
           "SUCCESS": "SUCCESS: M2b-wc drives better than M2b beyond 2F at both seeds. Next: the T1 closed-loop read of M2b-wc, then a "
                      "second dose on top of it (registry row on the PI's review).",
           "FAIL-MECHANISM-OK": "FAIL with the mechanism in place: the weight made T's present slots more accurate, but driving did not "
                                "improve beyond 2F. Next lever: O's use of the constraints (O trained against T's own code), under its own PREREG.",
           "FAIL-NO-MECHANISM": "FAIL without the mechanism: an 8x weight did not make T's present slots more accurate, so the loss weight "
                                "is not what limits T. Next lever: T's INPUT / capacity for the constraints (an architecture change: PI go)."}[verdict]
    out = dict(schema="v8f-wc-read-v1", prereg=f"PREREG_V8F_WC.md stage {a.stage}", tier="T1-class open loop (eval-141, 2,399 windows)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |M2b_s0 - M2b_s1| = {F:.4f} m (both banked evals)",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), n_windows_present=int(ok.sum()),
               present_slot_frac=float(pres[ok].mean()), F=F, two_F=2 * F, w_cons={"M2b": M2B_W_CONS, "M2b-wc": WC},
               rows=rows, stage=a.stage, DECISION=decision, VERDICT=verdict, mechanism=mech_all, next=nxt)
    out["reading"] = (f"PREREG_V8F_WC stage {a.stage}: {decision or verdict}; M2b-wc - M2b ADE6 " +
                      " / ".join(f"{rows[f's{s}']['delta']['delta']:+.3f}{'' if rows[f's{s}']['delta']['separated'] else ' ns'}" for s in seeds) +
                      f" m (2F {2 * F:.4f}); T present-slot RMSE M2b-wc / M2b " +
                      " | ".join(f"{rows[f's{s}']['present_rmse']['M2b-wc']:.3f}/{rows[f's{s}']['present_rmse']['M2b']:.3f}"
                                 f"{'' if rows[f's{s}']['present_mse']['separated'] else ' ns'}" for s in seeds) + " SD." +
                      ("" if verdict is None else f" VERDICT {verdict}."))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("s0", "final"), required=True)
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
    spec = importlib.util.spec_from_file_location("v8f_tac_groups_read_for_wc", HERE.parent / "v8f_tac_groups_read.py")
    R1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R1)
    R1._paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap, None if a.no_families else R1.family_fn(a))
    except (OSError, KeyError, ValueError) as e:
        print(f"WC READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
