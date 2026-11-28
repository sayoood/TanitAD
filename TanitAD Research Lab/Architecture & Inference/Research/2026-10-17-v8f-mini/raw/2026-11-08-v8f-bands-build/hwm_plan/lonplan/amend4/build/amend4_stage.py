"""AMEND_HWMPLAN_4 stage table and CLI: the ONE place that spells the argv of every planned arm (TRAIN stage and SCORE), so the controls (C-L4-IMG), the launcher and the readout read the same table.
Arms (section 4): P = `--l2-image prim`, P23 = `--l2-image prim23`, RS = `--l2-image prim --l1-sub donor` (seed 0 only). A0 and A1 are LON-PLAN's banked arms (reused; MM ruling M2) and are never re-planned here.
Chassis, checkpoints, TRAIN priors, SCORE windows: LON-PLAN's own (score_chain.py CH table), by sha256. TRAIN windows: LON-PLAN's selection set (sel_s{k}/selection_windows.npz, 564 windows, 141 clips).
AMEND4_ROOT (default: the amend4 folder) is where train/ score/ controls/ readout/ live; the build's smoke sets it elsewhere so the real folders stay new.
usage: python amend4_stage.py plan --stage train|score --arm P|P23|RS --chassis s0|s1      (OMP_NUM_THREADS=2; a fresh out folder is required)
       python amend4_stage.py argv --stage ... --arm ... --chassis ...                      (prints the argv, plans nothing)"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L0 = "F:/TanitAD-scratch/actrsp/lonplan"
ROOT = os.environ.get("AMEND4_ROOT", "F:/TanitAD-scratch/actrsp/lonplan/amend4")
CH = {"s0": dict(tag="R35_s0", win=L0 + "/score/A0_s0/windows.npz", wsha="371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", cksha="d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf",
                 lat="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt", latsha="76a2d4f5482bd98670bab40ebbf4b3c85fde8ebf222596947fa16f0c7e82754c"),
      "s1": dict(tag="R35_s1", win=L0 + "/score/A0_s1/windows.npz", wsha="8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt", cksha="a97e6017cffb13830dacd04e1ea108fa8534bc3ab3007b8f536f01ac0846f881",
                 lat="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s1.pt", latsha="692eb002")}
# the spec text writes hwm_plan.py as 5a072272...; the file and LON-PLAN's SOLVER.json read 5a0722719f7c...: the card has a transcription slip in the 8th hex digit, so 7 digits are compared here and the slip is reported
CODE_SHA_PREFIX = {"hwm_plan.py": "5a07227", "run_hwm.py": "935b3422", "hwm_eval.py": "79392075", "hwm_data.py": "a2d3922e"}
ARMS = {"P": dict(l2="prim", l1="own", seeds={"score": [0, 1], "train": [0]}),
        "P23": dict(l2="prim23", l1="own", seeds={"score": [0, 1], "train": [0]}),
        "RS": dict(l2="prim", l1="donor", seeds={"score": [0], "train": [0]})}
STAGES = ("train", "score")


def sel_windows(k):
    return f"{L0}/sel_{k}/selection_windows.npz"


def sel_windows_sha(k):
    return json.load(open(f"{L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"]


def out_dir(stage, arm, k):
    return f"{ROOT}/{stage}/{arm}_{k}"


def build(stage, arm, k, out=None):
    """-> dict(argv, declare_l2, declare_l1, windows_src, windows_sha, ckpt_sha). The argv names --l2-image (and --l1-sub for RS), --lam=0 and --seed-mode hold."""
    if stage not in STAGES or arm not in ARMS or k not in CH:
        raise ValueError((stage, arm, k))
    c, s = CH[k], ARMS[arm]
    o = out or out_dir(stage, arm, k)
    argv = ["--tag", c["tag"], "--ckpt", c["ck"], "--train-latents", c["lat"], "--train-tag", c["tag"], "--seed-mode", "hold", "--lam=0", "--l2-image", s["l2"]]
    if s["l1"] != "own":
        argv += ["--l1-sub", s["l1"]]
    argv += ["--arms", "hwm_oracle", "--seeds", *map(str, s["seeds"][stage]), "--workers", "1", "--out", o]
    ws, wsha = (c["win"], c["wsha"]) if stage == "score" else (sel_windows(k), sel_windows_sha(k))
    return dict(argv=argv, declare_l2=s["l2"], declare_l1=s["l1"], windows_src=ws, windows_sha=wsha, ckpt_sha=c["cksha"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "argv"]); ap.add_argument("--stage", required=True, choices=list(STAGES)); ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--chassis", required=True, choices=list(CH))
    a = ap.parse_args()
    b = build(a.stage, a.arm, a.chassis)
    if a.cmd == "argv":
        print(json.dumps(b, indent=1)); return
    import amend4_run as W
    sol = W.run_arm(b["argv"], b["declare_l2"], b["windows_src"], b["windows_sha"], b["declare_l1"], b["ckpt_sha"])
    out = b["argv"][b["argv"].index("--out") + 1]
    open(os.path.join(out, "DONE"), "w").write("ok\n")
    print("PLANNED", a.stage, a.arm, a.chassis, sorted(sol["plans"]), flush=True)


if __name__ == "__main__":
    main()
