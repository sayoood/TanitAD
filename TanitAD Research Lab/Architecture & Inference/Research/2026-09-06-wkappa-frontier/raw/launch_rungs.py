#!/usr/bin/env python3
"""W_KAPPA FRONTIER -- launch by-goal turn-weight rungs on the p4 40-window panel.

ONE variable: `w_turn`, the curvature weight applied ONLY to windows whose
DECODED tactical lateral goal is a TURN. `w_lane_keep` = `w_shift` = 15.11245
are PINNED at the wk15 value on every rung, so the arm differs from `gkappa`
(banked, w_turn = 0.0) in exactly one number.

Reconstructed token-for-token from `dump_gkappa/manifest.json` (corpus/model/
grid/plan_cfg/cost blocks). ASCII only: the dev box is cp1252 and a non-ASCII
print() is fatal.
"""
import os
import subprocess
import sys
import time

CTG = r"C:\Users\Admin\tanitad-ctg"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
OUT = r"C:\Users\Admin\wkfront\out"
TOOL = os.path.join(CTG, "taniteval", "tools", "refav1_arm.py")

CKPT = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt"
CFG = "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json"
CACHE = "C:/Users/Admin/refav1_margin/p4/fp8"
EPS = "C:/Users/Admin/refav1_margin/p4/eps"
LABELS = "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"

W_LANE = "15.11245"
COST_W = "0.0,15.11245,64.29715042415070"   # W_JERK, W_KAPPA(no-goal fallback), W_VEND


CORPORA = {
    # the banked p4 panel every earlier arm was run on -- turn-ENRICHED
    # (selected at kappa_thr 0.04 for 17 real turns in 40 windows)
    "p4": ("C:/Users/Admin/refav1_margin/p4/fp8",
           "C:/Users/Admin/refav1_margin/p4/eps"),
    # ⭐ EPISODE-DISJOINT draws for the curvature claim's episode replication --
    # NOT turn-enriched, so a fairer curvature test and a weaker recall test.
    "dA": ("C:/Users/Admin/wkfront/dA/fp8", "C:/Users/Admin/wkfront/dA/eps"),
    "dB": ("C:/Users/Admin/wkfront/dB/fp8", "C:/Users/Admin/wkfront/dB/eps"),
}


def argv_for(tag, w_turn, seed, corpus="p4"):
    # MODE spelling:
    #   "<float>"  -> by-goal arm, --w-kappa-by-goal W_LANE,<float>
    #   "S<float>" -> plain SCALAR arm at that W_KAPPA, flag OMITTED.
    # ⛔ The scalar spelling is the tool's own instruction, not a shortcut: it
    # REFUSES an all-equal --w-kappa-by-goal map ("INERT BY CONSTRUCTION ... To
    # run the scalar control, OMIT this flag").
    scalar = w_turn.startswith("S")
    cost_w = ("0.0,%s,64.29715042415070" % w_turn[1:]) if scalar else COST_W
    a = [
        PY, TOOL,
        "--ckpt", CKPT,
        "--config", CFG,
        "--cache", CORPORA[corpus][0],
        "--episodes", CORPORA[corpus][1],
        "--labels", LABELS,
        "--nav", LABELS,
        "--arm", "refav1-21109-wkf-%s" % tag,
        "--out", "%s/rec_%s.json" % (OUT.replace("\\", "/"), tag),
        "--dump-dir", "%s/dump_%s" % (OUT.replace("\\", "/"), tag),
        "--device", "cuda",
        "--window-stride", "16",
        "--horizon-k", "10",
        "--wm-k", "30",
        "--cost-metric", "ccos",
        "--cost-weights", cost_w,
        "--plan-n-samples", "300",
        "--plan-n-iters", "30",
        "--plan-n-elites", "30",
        "--plan-seed", str(seed),
        "--n-boot", "2000",
        "--seed", "0",
        "--no-lead-block",
        # ⛔ MATCHES `gkappa` / `wk15` EXACTLY: those arms carry
        # arms=['cl','ha','ha0','ha0_ext','ol'] with NO `cl_navshuf`, i.e. they
        # were run with this flag. Omitting it adds a SECOND planning arm --
        # 2x the wall-clock -- and breaks config parity with the two endpoints
        # this package's whole comparison rests on. No claim here is
        # nav-conditioned (curvature, turn recall, friction circle), which is
        # the only thing the flag's own warning makes inadmissible.
        "--no-navshuf",
    ]
    if not scalar:
        a += ["--w-kappa-by-goal", "%s,%s" % (W_LANE, w_turn)]
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        tag, w_turn, seed = parts[0], parts[1], int(parts[2])
        corpus = parts[3] if len(parts) > 3 else "p4"
        jobs.append((tag, w_turn, seed, corpus))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(CTG, "stack")
    env["OMP_NUM_THREADS"] = "6"
    env["PYTHONIOENCODING"] = "utf-8"
    procs = []
    for tag, w_turn, seed, corpus in jobs:
        log = open(os.path.join(OUT, "%s.log" % tag), "w", encoding="utf-8")
        a = argv_for(tag, w_turn, seed, corpus)
        print("[launch] %s w_turn=%s seed=%d corpus=%s" % (tag, w_turn, seed, corpus))
        print("[argv  ] %s" % " ".join(a))
        p = subprocess.Popen(a, cwd=CTG, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        procs.append((tag, p, log))
        time.sleep(3)
    print("[launched] %d arms, pids=%s" % (len(procs), [p.pid for _, p, _ in procs]))
    rc = 0
    for tag, p, log in procs:
        r = p.wait()
        log.close()
        print("[done] %s rc=%d" % (tag, r))
        rc |= r
    print("ZZLAUNCHDONE-%d-%dZZ" % (len(procs), rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
