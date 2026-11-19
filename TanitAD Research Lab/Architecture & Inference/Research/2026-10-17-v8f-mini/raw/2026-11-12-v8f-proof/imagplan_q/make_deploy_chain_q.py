#!/usr/bin/env python
"""make_deploy_chain_q.py -- builds deploy_q1.sh (from a copy of ../imagplan/deploy_k6.sh; chain_q1.sh is written by hand) for note PREREG_V8F_PROOF_ADD40_N7.
Every edit is asserted to match the stated number of times. The originals are not changed."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "imagplan")


def rd(n):
    return open(os.path.join(SRC, n), encoding="utf-8", newline="").read().replace("\r\n", "\n")


def apply(s, edits):
    for old, new, cnt in edits:
        n = s.count(old)
        if n != cnt:
            sys.exit("edit matches %d times (need %d): %r" % (n, cnt, old[:80]))
        s = s.replace(old, new)
    return s


DEPLOY = [
    ("# deploy_k6.sh <NAME>", "# deploy_q1.sh (note PREREG_V8F_PROOF_ADD40_N7, form of deploy_k6.sh) <NAME>", 1),
    ("H=/f/TanitAD-scratch/v8proof/imagplan; W=/workspace/v8proof; D=$W/imagplan_k;", "H=/f/TanitAD-scratch/v8proof/imagplan_q; W=/workspace/v8proof; D=$W/imagplan_q;", 1),
    ("for f in imag_plan.py v8f_t1_eval_imagplan.py pod_imagplan_run.sh pod_imagplan_seq6.sh; do", "for f in imag_plan.py v8f_t1_eval_imagplan_q.py qhead_model.py pod_imagplan_run_q.sh pod_imagplan_seq_q.sh; do", 1),
    ("bash -n pod_imagplan_seq6.sh && bash -n pod_imagplan_run.sh &&", "bash -n pod_imagplan_seq_q.sh && bash -n pod_imagplan_run_q.sh &&", 1),
    ("nohup setsid bash pod_imagplan_seq6.sh $NAME", "nohup setsid bash pod_imagplan_seq_q.sh $NAME", 1),
    ("ZZ DEPLOY-K", "ZZ DEPLOY-Q", 5),
    ("Form of deploy_k.sh for the sequence with the memory guard (pod_imagplan_seq6.sh)", "Sequence with the memory guard (pod_imagplan_seq_q.sh)", 1),
]

def main():
    d = apply(rd("deploy_k6.sh"), DEPLOY)
    open(os.path.join(HERE, "deploy_q1.sh"), "w", encoding="utf-8", newline="\n").write(d)
    print("wrote deploy_q1.sh")


if __name__ == "__main__":
    main()
