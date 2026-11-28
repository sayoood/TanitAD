#!/usr/bin/env python
"""make_pod_scripts_q.py -- builds pod_imagplan_run_q.sh and pod_imagplan_seq_q.sh (note PREREG_V8F_PROOF_ADD40_N7) from COPIES of ../imagplan/pod_imagplan_run.sh and pod_imagplan_seq6.sh.
Every edit is asserted to match exactly once (replace-all edits state their count). The originals are not changed."""
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


RUN = [
    ("# pod_imagplan_run.sh <NEWARM>", "# pod_imagplan_run_q.sh (note PREREG_V8F_PROOF_ADD40_N7: form of pod_imagplan_run.sh with the mode cand-cvq) <NEWARM>", 1),
    ("MODE: off | cand-imag | cand-cv | cand-label.", "MODE: off | cand-imag | cand-cv | cand-label | cand-cvq (env IP_QHEAD_FILE = the head file, required; IP_QHEAD_CONST = metres replacing the head output: controls q1 1e9, q2 0).", 1),
    ("off|cand-imag|cand-cv|cand-label|cand-imagkin) ;;", "off|cand-imag|cand-cv|cand-label|cand-imagkin|cand-cvq) ;;", 1),
    ("MODE must be off|cand-imag|cand-cv|cand-label|cand-imagkin (got", "MODE must be off|cand-imag|cand-cv|cand-label|cand-imagkin|cand-cvq (got", 1),
    ('usage: pod_imagplan_run.sh', 'usage: pod_imagplan_run_q.sh', 1),
    ("v8f_t1_eval_imagplan.py", "v8f_t1_eval_imagplan_q.py", 7),
    ("for f in imag_plan.py v8f_t1_eval_imagplan_q.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py;", "for f in imag_plan.py v8f_t1_eval_imagplan_q.py qhead_model.py dist_rule.py v8f_t1_eval_rule.py imag_dump.py;", 1),
    ("LIBPATH=$(for f in imag_plan.py dist_rule.py", "LIBPATH=$(for f in imag_plan.py qhead_model.py dist_rule.py", 1),
    ('# 4. the command: the chain\'s dump argv + the wrapper flags\n', '''# 3b. note N7: the quantile head variables belong to the mode cand-cvq only; for it the head file is required
if [ "$MODE" = cand-cvq ]; then
  [ -n "${IP_QHEAD_FILE:-}" ] || die "cand-cvq needs IP_QHEAD_FILE"
  [ -f "$IP_QHEAD_FILE" ] || die "no head file $IP_QHEAD_FILE"
else
  [ -z "${IP_QHEAD_FILE:-}" ] && [ -z "${IP_QHEAD_CONST:-}" ] || die "IP_QHEAD_FILE / IP_QHEAD_CONST with MODE $MODE would change nothing"
fi
# 4. the command: the chain's dump argv + the wrapper flags
''', 1),
]

SEQ = [
    ("# pod_imagplan_seq6.sh <NAME>", "# pod_imagplan_seq_q.sh (note PREREG_V8F_PROOF_ADD40_N7; form of pod_imagplan_seq6.sh with the tags IPo IPc IPl IPcq IPcqid IPcq0 only) <NAME>", 1),
    ("D=${IP_D:-$W/imagplan_k}", "D=${IP_D:-$W/imagplan_q}", 1),
    ("export IP_FAMILY=decel IP_LIBDIRS=$W/distrule:$W/probe", """export IP_FAMILY=decel IP_LIBDIRS=$W/distrule:$W/probe
# the banked head of the checkpoint: LS_* in qhead_dose_ls/heads, LSD_* in qhead_dose/heads (override the folder with IP_QHEAD_DIR)
case "$BASE" in LS_s*) QDIR=${IP_QHEAD_DIR:-$W/qhead_dose_ls/heads};; LSD_s*) QDIR=${IP_QHEAD_DIR:-$W/qhead_dose/heads};; *) QDIR=${IP_QHEAD_DIR:-/nonexistent};; esac
QFILE=$QDIR/$BASE.qhead.npz""", 1),
    ('pat="v8f_t1_eval_imagplan.py .*--out-dir', 'pat="v8f_t1_eval_imagplan_q.py .*--out-dir', 1),
    ('$(sha256sum "$D/v8f_t1_eval_imagplan.py" | cut -c1-16)', '$(sha256sum "$D/v8f_t1_eval_imagplan_q.py" | cut -c1-16) qhead_model $(sha256sum "$D/qhead_model.py" | cut -c1-16)', 1),
    ("# Everything else as seq4: tool folder IP_D (default /workspace/v8proof/imagplan_k), IP_FAMILY=decel, serial runs through pod_imagplan_run.sh, tags",
     "# Tool folder IP_D (default /workspace/v8proof/imagplan_q), IP_FAMILY=decel, serial runs through pod_imagplan_run_q.sh, tags (this form: IPo IPc IPl IPcq IPcqid IPcq0 only; IPcq = cand-cvq, IPcqid = cand-cvq with the constant 1e9, IPcq0 = with 0):", 1),
    ('env "$@" bash "$D/pod_imagplan_run.sh"', 'env "$@" bash "$D/pod_imagplan_run_q.sh"', 1),
    ("""    IPi)    one IPi cand-imag $NEED IP_P45=1 ;;
    IPid1)  one IPid1 cand-imag $NEED IP_CAND_SET=0 IP_FALLBACK=none ;;
    IPid3)  one IPid3 cand-imag $NEED IP_CONST_GAP=1000000000 ;;
    IPn)    one IPn cand-imag $NEED_NOISE IP_NOISE=0 ;;
    IPs)    one IPs cand-imag $NEED IP_SHIFT_M=-20 ;;
""", "", 1),
    ("""    IPk)    one IPk cand-imagkin $NEED IP_P45=1 ;;
    IPkid1) one IPkid1 cand-imagkin $NEED IP_CAND_SET=0 IP_FALLBACK=none ;;
    IPkid3) one IPkid3 cand-imagkin $NEED IP_CONST_GAP=1000000000 ;;
    IPkn)   one IPkn cand-imagkin $NEED_NOISE IP_NOISE=0 ;;
    IPks)   one IPks cand-imagkin $NEED IP_SHIFT_M=-20 ;;
""", """    IPcq)   one IPcq cand-cvq $NEED IP_QHEAD_FILE=$QFILE ;;
    IPcqid) one IPcqid cand-cvq $NEED IP_QHEAD_FILE=$QFILE IP_QHEAD_CONST=1000000000 ;;
    IPcq0)  one IPcq0 cand-cvq $NEED IP_QHEAD_FILE=$QFILE IP_QHEAD_CONST=0 ;;
""", 1),
    ("""cmpd IPid1 IPo "p1 first part, must be equal"
cmpd IPid3 IPo "p1b, must be equal"
cmpd IPn IPi "p3, must be equal"
cmpd IPi IPo "firing check, expected to differ"
cmpd IPs IPi "firing check, expected to differ"
cmpd IPkid1 IPo "p1 first part of cand-imagkin, must be equal"
cmpd IPkid3 IPo "p1b of cand-imagkin, must be equal"
cmpd IPkn IPk "p3 of cand-imagkin, must be equal"
cmpd IPk IPo "firing check, expected to differ"
cmpd IPks IPk "firing check, expected to differ"
cmpd IPk IPi "cand-imagkin against cand-imag, expected to differ"
""", """cmpd IPcqid IPc "q1, must be equal"
cmpd IPcq0 IPc "q2 firing check, expected to differ"
cmpd IPcq IPc "cand-cvq against cand-cv, expected to differ"
cmpd IPc IPo "firing check, expected to differ"
cmpd IPl IPo "firing check, expected to differ"
""", 1),
]


def main():
    r = apply(rd("pod_imagplan_run.sh"), RUN)
    s = apply(rd("pod_imagplan_seq6.sh"), SEQ)
    for n, t in (("pod_imagplan_run_q.sh", r), ("pod_imagplan_seq_q.sh", s)):
        with open(os.path.join(HERE, n), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
        print("wrote", n, len(t))


if __name__ == "__main__":
    main()
