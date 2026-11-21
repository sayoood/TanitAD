"""Seconds-long pre-check of a pulled arm, BEFORE any rollout: runs the reader's own pulled-file controls (KP1, KR1, KR2, KR3) on the
pulled run dirs, so a selfcheck / summary shape the reader would refuse is found before ~45 min of rollouts. The functions are the
reader's (loaded by file path, no second stack root); nothing here re-implements a rule.
python -I precheck_pulled.py --r-arm RAPR [--recover-dir E:/Projects/TanitAD-artifacts/v8f-recover]   -> rc 0 PASS, rc 3 a control refuses"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("recover_read_arms", os.path.join(HERE, "recover_read.py"))
RR = importlib.util.module_from_spec(spec)
sys.modules["recover_read_arms"] = RR
spec.loader.exec_module(RR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-arm", required=True, choices=RR.ARMS)
    ap.add_argument("--recover-dir", default=RR.RECOVER_DIR)
    a = ap.parse_args()
    rd, arm = a.recover_dir, a.r_arm
    rdir = lambda s, base: f"{rd}/M2b_s{s}_recft_{arm if base == 'R' else base}"   # noqa: E731  (the reader's own mapping)
    J = lambda p: json.load(open(p, encoding="utf-8"))   # noqa: E731
    out = {}
    for s in RR.SEEDS:
        summ = {b: J(f"{rdir(s, b)}/summary.json") for b in ("R", "C0", "C0rep")}
        argv = {b: J(f"{rdir(s, b)}/argv.json")["argv"] for b in ("R", "C0", "C0rep")}
        ref = J(f"{RR.DAGOACC_DIR}/M2b_s{s}_dagoacc/summary.json")
        out[f"KP1_s{s}"] = {b: RR.kp1_check(summ[b], ref) for b in ("R", "C0", "C0rep")}
        out[f"KP1_s{s}"]["PASS"] = all(v["PASS"] for v in out[f"KP1_s{s}"].values())
        out[f"KR1_s{s}"] = RR.kr1_check(summ["R"], summ["C0"], summ["C0rep"], arm)
        r_argv = J(f"{rd}/M2b_s{s}_recft_R/argv.json")["argv"] if arm != "R" else None
        out[f"KR2argv_s{s}"] = RR.argv_audit(argv["R"], argv["C0"], argv["C0rep"], s, arm, r_argv)
        out[f"KR3_s{s}"] = RR.kr3_check(J(f"{rdir(s, 'R')}/gate_selfcheck.json"), s, arm)
    bad = [k for k, v in out.items() if not v.get("PASS")]
    print(json.dumps({"arm": arm, "refused": bad, "PASS": not bad}, indent=1))
    for k in bad:
        print("REFUSED", k, json.dumps(out[k])[:600])
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
