"""mk_jobs_am2.py -- card PREREG_V8F_PROOF_ADD153 amendment 2: the box-2 job files (CRLF) of stage 4.

  python -I mk_jobs_am2.py train   -> acc_a2_K25D_s150.cmd, acc_a2_K25DS_s149.cmd
  python -I mk_jobs_am2.py read    -> acc_a2_ens4.cmd, with every head's sha256 taken from its pulled ACC.json (refuses if one is missing)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_jobs_acc as MK  # noqa: E402

OUT = os.path.join(HERE, "jobs")
HEADS = (("K25D", "acc_K25D"), ("K25D_s149", "acc_s2rep"), ("K25DS", "acc_s2smooth"), ("K25DS_s149", "acc_a2_K25DS_s149"), ("K25D_s150", "acc_a2_K25D_s150"))
SETS = "ENS2=0,1;ENS4=0,1,2,3;ENS5=0,1,2,3,4"


def train():
    return [MK.write("acc_a2_K25D_s150", "train_acc_box2.py", "--arm K25D --out D:/tanitad/leadacc/K25D_s150 --train-seed 150 " + MK.REF, OUT),
            MK.write("acc_a2_K25DS_s149", "train_acc_box2.py", "--arm K25D --out D:/tanitad/leadacc/K25DS_s149 --train-seed 149 --label-run 10 " + MK.REF, OUT)]


def read():
    hs = []
    for rd, job in HEADS:
        r = json.load(open(os.path.join(HERE, "box2_" + job, "ACC.json"), encoding="utf-8"))
        if not r.get("controls_all_pass"):
            raise SystemExit("controls of %s did not pass" % job)
        hs.append("D:/tanitad/leadacc/%s/head_30000.pt=%s" % (rd, r["train"]["ckpt_sha256"]["30000"]))
    args = '--out D:/tanitad/leadacc/ens4 --arm K25D --expect-ens D:/tanitad/leadacc/ens/ENS.json --record ENS4 --compare ENS2 --sets "%s" --heads %s' % (SETS, " ".join(hs))
    return [MK.write("acc_a2_ens4", "ens_read_n_box2.py", args, OUT)]


if __name__ == "__main__":
    for p in (train() if sys.argv[1] == "train" else read()):
        print(p)
