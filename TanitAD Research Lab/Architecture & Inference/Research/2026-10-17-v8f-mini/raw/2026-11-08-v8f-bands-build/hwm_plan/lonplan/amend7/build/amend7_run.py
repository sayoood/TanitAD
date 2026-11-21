"""AMEND_HWMPLAN_7 driver: AMEND_5's ra_l1.run_arm on the amend7 folders (amend7_env first), arms RAL1 (readout A), RAL1B (readout B), NEG, SHUF, ORACLE, OFF on SCORE; RAL1, NEG, SHUF, ORACLE on TRAIN (beside).
MEAN-L1 is a reference read from AMEND_6's folders: its plan files are copied here after their sha256 is checked against the folder's SOLVER_RA.json, so AMEND_5's score() / train_stage() find it where they look.
Usage: amend7_run.py plan --stage train|score --chassis s0|s1"""
import amend7_env  # noqa: F401
import argparse, copy, json, os, shutil, sys, time
sys.dont_write_bytecode = True
from amend7_env import ROOT, ST, A6, R, sha_B
import ra7_l1
import lonplan_common as C

SCORE_ARMS = [("RAL1", "RAL1", None), ("RAL1B", "RAL1", None), ("NEG", "NEG", "l1"), ("SHUF", "SHUF", "l1"), ("ORACLE", "ORACLE", None), ("OFF", "OFF", None)]
TRAIN_ARMS = [("RAL1", "RAL1", None), ("NEG", "NEG", "l1"), ("SHUF", "SHUF", "l1"), ("ORACLE", "ORACLE", None)]


def stage_refs(stage, k, log=print):
    """Copy AMEND_6's MEAN-L1 folder (no readout, no world model) into the amend7 tree after checking every plan file and windows.npz against its SOLVER_RA.json. An existing complete copy is kept."""
    src, dst = f"{A6}/{stage}/MEANL1_A0_{k}", ST.out_dir(stage, "MEANL1", "A0", k)
    if os.path.exists(f"{dst}/DONE"):
        return dst
    sol = json.load(open(f"{src}/SOLVER_RA.json", encoding="utf-8"))
    for f, v in sol["plans"].items():
        if C.sha256_file(f"{src}/{f}") != v["sha256"]:
            raise R.RefusedRun(f"{src}/{f} sha256 differs from its SOLVER_RA.json")
    if C.sha256_file(f"{src}/windows.npz") != sol["windows_sha256"]:
        raise R.RefusedRun(f"{src}/windows.npz sha256 differs from its SOLVER_RA.json")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for f, v in sol["plans"].items():
        if C.sha256_file(f"{dst}/{f}") != v["sha256"]:
            raise R.RefusedRun(f"{dst}/{f} copy differs")
    log(f"reference copy {stage} MEANL1 {k}: {len(sol['plans'])} plan files, sha256 verified", flush=True)
    return dst


def run_stage7(stage, k, log=print):
    solver = ST.read_base_solver() if stage == "score" else None
    R.verify_inputs(k)
    ctx = R.load_ctx(k)                                         # readout A (the sha256 in RA7_FIT_<k>.json)
    ctxB = None
    if stage == "score":
        ctxB = copy.copy(ctx)
        ctxB.ra = ra7_l1.load_readout7(f"{ST.READOUT_DIR}/readout7_MLP_R35_{k}_B.pt", k, sha_B(k))
    stage_refs(stage, k, log)
    wsha = ST.CH[k]["wsha"] if stage == "score" else json.load(open(f"{ST.L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"]
    done = []
    for name, arm, src in (SCORE_ARMS if stage == "score" else TRAIN_ARMS):
        o = ST.out_dir(stage, name, "A0", k)
        b = ST.build(stage, arm, "A0", k, out=o, src_for_nl=src)
        decl = dict(b["declared"], readout_sha256=sha_B(k)) if name == "RAL1B" else b["declared"]
        t = time.time()
        R.run_arm(b["argv"], decl, ctx=(ctxB if name == "RAL1B" else ctx), windows_src=ST.windows_of(stage, k), windows_sha=wsha, solver=solver)
        log(f"{stage} {name} A0 {k} done {round(time.time() - t, 1)} s", flush=True)
        done.append((name, "A0", o))
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--stage", required=True, choices=["train", "score"]); ap.add_argument("--chassis", required=True)
    a = ap.parse_args()
    d = run_stage7(a.stage, a.chassis)
    print("PLANNED", a.stage, a.chassis, [f"{x[0]}_{x[1]}" for x in d], flush=True)
