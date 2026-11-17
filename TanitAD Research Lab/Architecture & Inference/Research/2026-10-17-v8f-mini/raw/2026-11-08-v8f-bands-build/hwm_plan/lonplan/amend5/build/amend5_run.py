"""AMEND_HWMPLAN_5 driver: plans every RA arm of a stage for one chassis in ONE process (model and priors loaded once; the checkpoint / latents sha256 checked once).
  python amend5_run.py plan --stage train --chassis s0 [--bases A0 P P23]    TRAIN stage (beside, in-sample, never a result); runs now on A0's TRAIN plans
  python amend5_run.py plan --stage score --chassis s0                       SCORE: REFUSED unless BASE_SOLVER_AMEND5.md names the branch and base (amend5_stage.read_base_solver)
Arm lists (section 4): B3 (base A0): RAL1 (committed), MEANL1, NEG, SHUF (source l1), ORACLE, OFF; beside RAL2 on P's banked files (--beside, no outcome).
B1 / B2 (base P or P23): RAL2 (committed), RAL1 (second), MEANL2, MEANL1, NEG, SHUF (source l2), ORACLE, OFF. TRAIN: the same lists for each base whose TRAIN files exist (RAL2 only with l2_prim_raw).
Every arm goes through ra_l1.run_arm (C-L4-RA argv -> built, fresh folder, sha256 checks, stamps). An existing out folder is refused, never overwritten."""
import argparse, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend5_stage as ST
import ra_l1 as R
import numpy as np


def has_l2(stage, base, k):
    p = ST.base_plan(stage, base, k, 0)
    if not os.path.exists(p):
        return False
    with np.load(p, allow_pickle=False) as z:
        return "l2_prim_raw" in z.files


def arm_list(stage, k, solver=None, bases=("A0",)):
    """-> [(arm, base, src_for_neg_shuf, beside)]."""
    out = []
    if stage == "score":
        base, br = solver["base"], solver["branch"]
        if br == "B3":
            out += [("RAL1", "A0", None, False), ("MEANL1", "A0", None, False), ("NEG", "A0", "l1", False), ("SHUF", "A0", "l1", False), ("ORACLE", "A0", None, False), ("OFF", "A0", None, False)]
            if os.path.exists(ST.base_plan("score", "P", k, 0)) and has_l2("score", "P", k):
                out.append(("RAL2", "P", None, True))
        else:
            out += [("RAL2", base, None, False), ("RAL1", base, None, False), ("MEANL2", base, None, False), ("MEANL1", base, None, False), ("NEG", base, "l2", False), ("SHUF", base, "l2", False),
                    ("ORACLE", base, None, False), ("OFF", base, None, False)]
        return out
    for b in bases:
        if not os.path.exists(ST.base_plan("train", b, k, 0)):
            continue
        l2 = b != "A0" and has_l2("train", b, k)
        out += ([("RAL2", b, None, False)] if l2 else []) + [("RAL1", b, None, False)] + ([("MEANL2", b, None, False)] if l2 else []) + [("MEANL1", b, None, False)]
        out += [("NEG", b, "l2" if l2 else "l1", False), ("SHUF", b, "l2" if l2 else "l1", False), ("ORACLE", b, None, False)]
    return out


def run_stage(stage, k, bases=("A0",), root=None, log=print):
    solver = ST.read_base_solver() if stage == "score" else None
    R.verify_inputs(k)
    ctx = R.load_ctx(k)
    import json
    wsha = ST.CH[k]["wsha"] if stage == "score" else json.load(open(f"{ST.L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"]
    done = []
    for arm, base, src, beside in arm_list(stage, k, solver, bases):
        o = ST.out_dir(stage, arm, base, k, root)
        b = ST.build(stage, arm, base, k, out=o, src_for_nl=src)
        argv = b["argv"] + (["--beside"] if beside else [])
        t = time.time()
        if stage == "train" and os.path.exists(os.path.join(o, "DONE")):
            log(f"{stage} {arm} {base} {k} skip (DONE exists: already planned, in-sample record)", flush=True)
            done.append((arm, base, o)); continue
        R.run_arm(argv, b["declared"], ctx=ctx, windows_src=ST.windows_of(stage, k), windows_sha=wsha, solver=solver)
        log(f"{stage} {arm} {base} {k} done {round(time.time() - t, 1)} s", flush=True)
        done.append((arm, base, o))
    return done


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--stage", required=True, choices=list(ST.STAGES)); p.add_argument("--chassis", required=True); p.add_argument("--bases", nargs="+", default=["A0"])
    a = p.parse_args() if False else ap.parse_args()
    d = run_stage(a.stage, a.chassis, tuple(a.bases))
    print("PLANNED", a.stage, a.chassis, [f"{x[0]}_{x[1]}" for x in d], flush=True)


if __name__ == "__main__":
    main()
