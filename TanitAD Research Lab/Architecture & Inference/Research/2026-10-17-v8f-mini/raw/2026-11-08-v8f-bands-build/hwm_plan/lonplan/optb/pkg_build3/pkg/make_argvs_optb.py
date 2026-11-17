"""AMEND_OPTB section 1: the ten Option B argvs plus the pilot (never launched here). Base = actsens_NULL_s2 (chassis 0) / NULL_s3 (chassis 1) argv.txt (R35's recipe), with ONLY: --seed, --out, --init-ckpt (the chassis' R35 ckpt_last.pt on box 2),
--steps 4000, `--wm-init-from-ckpt-wm on --wm-freeze-proj on` in ALL ten, and on the RSP arms `--wm-rsp-w 1.0 --wm-rsp-readout fixed_k --wm-rsp-probe <box-2 file> --wm-rsp-probe-sha256 <hex>`.
usage: python make_argvs_optb.py <outdir>   -> argvs_optb.json (+ argvs_optb.txt)"""
import hashlib
import json
import os
import sys

SRC = {0: "C:/Users/Admin/t2v8f_pulls/actsens_NULL_s2/argv.txt", 1: "C:/Users/Admin/t2v8f_pulls/actsens_NULL_s3/argv.txt"}
PROBE_LOCAL = {k: f"F:/TanitAD-scratch/actrsp/optb/readout_K10_R35_s{k}.pt" for k in (0, 1)}
INPUTS = "D:\\tanitad\\optb\\inputs\\"
OUT = "D:\\tanitad\\runs\\v8f_optb\\"
PILOT_OUT = "D:\\tanitad\\optb\\pilot_out\\FTRSP_s0"
STEPS = "4000"
#: (tag, chassis, seed, lever)   AMEND_OPTB section 1 order
ARMS = [("FTRSP_s0", 0, 0, 1), ("FTNULL_s0", 0, 0, 0), ("FTRSP_s2", 0, 2, 1), ("FTNULL_s2", 0, 2, 0), ("FTNULL_s4", 0, 4, 0),
        ("FTRSP_s1", 1, 1, 1), ("FTNULL_s1", 1, 1, 0), ("FTRSP_s3", 1, 3, 1), ("FTNULL_s3", 1, 3, 0), ("FTNULL_s5", 1, 5, 0)]


def build():
    base = {k: open(SRC[k], encoding="utf-8").read().split() for k in (0, 1)}
    sha = {k: hashlib.sha256(open(PROBE_LOCAL[k], "rb").read()).hexdigest() for k in (0, 1)}
    rows = []
    for tag, k, seed, rsp in ARMS:
        t = list(base[k])
        for f in ("--wm-rsp-w", "--wm-rsp-readout", "--wm-rsp-probe", "--wm-init-from-ckpt-wm", "--wm-freeze-proj", "--wm-idm-gamma", "--wm-sampler"):
            assert f not in t, f
        t[t.index("--seed") + 1] = str(seed)
        t[t.index("--out") + 1] = OUT + tag
        t[t.index("--init-ckpt") + 1] = INPUTS + f"ladder_R35_s{k}.pt"
        t[t.index("--steps") + 1] = STEPS
        t += ["--wm-init-from-ckpt-wm", "on", "--wm-freeze-proj", "on"]
        if rsp:
            t += ["--wm-rsp-w", "1.0", "--wm-rsp-readout", "fixed_k", "--wm-rsp-probe", INPUTS + f"readout_K10_R35_s{k}.pt", "--wm-rsp-probe-sha256", sha[k]]
        rows.append((tag, t))
    pt = list(rows[0][1])
    pt[pt.index("--out") + 1] = PILOT_OUT
    pt[pt.index("--steps") + 1] = "200"
    rows.insert(0, ("PILOT_FTRSP_s0", pt))
    return rows


if __name__ == "__main__":
    d = sys.argv[1]
    rows = build()
    json.dump({n: {"argv": t} for n, t in rows}, open(os.path.join(d, "argvs_optb.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    with open(os.path.join(d, "argvs_optb.txt"), "w", encoding="utf-8", newline="\n") as f:
        for n, t in rows:
            f.write(f"{n}\t{' '.join(t)}\n")
    for n, t in rows:
        print(n, len(t), t.count("--wm-rsp-w"), t[t.index("--seed") + 1])
