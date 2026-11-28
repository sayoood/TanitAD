"""mk_jobs_acc.py -- card PREREG_V8F_PROOF_ADD153: writes the box-2 job files (CRLF, backslash interpreter path) into rulefc/leadacc/jobs/.

  python -I mk_jobs_acc.py              -> acc_smoke.cmd and the four stage-1 jobs
  python -I mk_jobs_acc.py --stage2 ARM -> acc_s2rep.cmd (ARM at training seed 149) and acc_s2smooth.cmd (ARM with the 2 s d label, seed 148)
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)
PY = BS.join(["D:", "tanitad", "venv", "Scripts", "python.exe"])
DIR = BS.join(["D:", "tanitad", "leadacc"])
REF = "--ref-ckpt D:/tanitad/leadspeed/all/head_30000.pt --ref-sha 83faaa6d355562f7"
STAGE1 = ("K25D", "K25DM", "K10D", "K40D")


def cmd_text(script, args):
    lines = ["@echo off", "set OMP_NUM_THREADS=4", "set MKL_NUM_THREADS=4", "set PYTHONDONTWRITEBYTECODE=1", "set PYTHONIOENCODING=utf-8",
             "set TEMP=" + BS.join(["D:", "tanitad", "tmp"]), "set TMP=" + BS.join(["D:", "tanitad", "tmp"]),
             'start /b /belownormal /wait "" %s %s%s%s %s' % (PY, DIR, BS, script, args)]
    return "\r\n".join(lines) + "\r\n"


def write(name, script, args, out_dir):
    p = os.path.join(out_dir, name + ".cmd")
    with open(p, "w", encoding="ascii", newline="") as f:
        f.write(cmd_text(script, args))
    return p


def stage1(out_dir):
    ps = [write("acc_smoke", "smoke_acc.py", "--out D:/tanitad/leadacc/smoke " + REF, out_dir)]
    for arm in STAGE1:
        ps.append(write("acc_" + arm, "train_acc_box2.py", "--arm %s --out D:/tanitad/leadacc/%s --train-seed 148 %s" % (arm, arm, REF), out_dir))
    return ps


def stage2(arm, out_dir):
    return [write("acc_s2rep", "train_acc_box2.py", "--arm %s --out D:/tanitad/leadacc/%s_s149 --train-seed 149 %s" % (arm, arm, REF), out_dir),
            write("acc_s2smooth", "train_acc_box2.py", "--arm %s --out D:/tanitad/leadacc/%sS --train-seed 148 --label-run 10 %s" % (arm, arm, REF), out_dir)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", default=None, choices=STAGE1)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "jobs"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (stage2(a.stage2, a.out_dir) if a.stage2 else stage1(a.out_dir)):
        print(p)
