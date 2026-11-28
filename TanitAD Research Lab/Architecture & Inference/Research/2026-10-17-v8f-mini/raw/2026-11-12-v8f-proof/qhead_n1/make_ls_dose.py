#!/usr/bin/env python
"""make_ls_dose.py -- writes the tools of the LS dose arms LSQF_s0, LSQF_s1 (note PREREG_V8F_PROOF_ADD42_N3) from the tools of the LSD dose arms by asserted text
edits (each old text must occur the stated number of times; -1 = at least once, replaced everywhere):

  dose/pod_qhead_dose_fits_ls.sh   from dose/pod_qhead_dose_fits.sh   (the two fits are LS_s0 seed 0, LS_s1 seed 1)
  ops/wait_qhead_dose_ls.sh        from ops/wait_qhead_dose.sh        (pod folder qhead_dose_ls, tools from qhead_dose/tools, pulled folder dose/pulled_ls;
                                                                       the input job is started by the lead with QD_ARMS, QD_RUNQ, QD_OUT, QD_TOOLS)
  qhead_names_qfls.py              from the text of qhead_names_qf.py (one arm type: LS, arms LSQF)
  qhead_read_qfls.sh               from qhead_read_qf.sh
  ops/wait_qhead_qfls.sh           from ops/wait_qhead_qf.sh
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_qf_tools as MQ  # noqa: E402

PODT = "/workspace/v8proof/qhead_dose/tools"

FITS = [
    ('ARMLINES="LSD_s0 0\nLSD_s1 1"\n', 'ARMLINES="LS_s0 0\nLS_s1 1"\n', 1),
    ('# pod_qhead_dose_fits.sh : ', '# pod_qhead_dose_fits_ls.sh : the LS checkpoints (note ADD42_N3), made by make_ls_dose.py from pod_qhead_dose_fits.sh; run with QD_OUT and QD_TOOLS. First form:\n# pod_qhead_dose_fits.sh : ', 1),
]
CHAIN = [
    ('O=/workspace/v8proof/qhead_dose\n', 'O=/workspace/v8proof/qhead_dose_ls\nT=%s\n' % PODT, 1),
    ('DD=$H/qhead_n1/dose; L=$DD/pulled; LW=$W/qhead_n1/dose/pulled\n', 'DD=$H/qhead_n1/dose; L=$DD/pulled_ls; LW=$W/qhead_n1/dose/pulled_ls\n', 1),
    ('for ARM in LSD_s0 LSD_s1; do', 'for ARM in LS_s0 LS_s1; do', 1),
    ('for b in LSD_s0 LSD_s1; do', 'for b in LS_s0 LS_s1; do', 1),
    ('$O/tools/dose_d2_state_check.py', '$T/dose_d2_state_check.py', 1),
    ('pod 30 "cd $O && nohup setsid bash tools/pod_qhead_dose_fits.sh > $O/start_fits.out 2>&1 < /dev/null &" > /dev/null',
     'pod 30 "cd $O && QD_OUT=$O QD_TOOLS=$T nohup setsid bash $T/pod_qhead_dose_fits_ls.sh > $O/start_fits.out 2>&1 < /dev/null &" > /dev/null', 1),
    ('# wait_qhead_dose.sh -- dev box, note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step, LSD type).',
     '# wait_qhead_dose_ls.sh -- the LS checkpoints (note PREREG_V8F_PROOF_ADD42_N3), made by make_ls_dose.py from wait_qhead_dose.sh. First form:\n'
     '# wait_qhead_dose.sh -- dev box, note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step, LSD type).', 1),
]
NAMES = [
    ('TYPES = {"LSD": dict(qh="LSDQF", none="LSD", rh="LSDRH", rl="LSDRL", mc="LSDMC", qc="LSDQC", base="LSD")}\n',
     'TYPES = {"LS": dict(qh="LSQF", none="LS", rh="LSRH", rl="LSRL", mc="LSMC", qc="LSQC", base="LS")}\n', 1),
    ('"card": "PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step: head fitted on all 4,572 training clips; arms LSDQF)"',
     '"card": "PREREG_V8F_PROOF_ADD42_N3 (dose step of card section 8 for the LS type: head fitted on all 4,572 training clips; arms LSQF)"', 1),
]
READ = [
    ('LSDQF', 'LSQF', -1),
    ('LSD', 'LS', -1),
    ('$N1/dose/pulled}', '$N1/dose/pulled_ls}', 1),
    ('$N1/dose/read}', '$N1/dose/read_ls}', 1),
    ('qhead_names_qf.py', 'qhead_names_qfls.py', -1),
    ('qhead_read_qf.sh', 'qhead_read_qfls.sh', -1),
]
WAIT = [
    ('LSDQF', 'LSQF', -1),
    ('RUNF=/workspace/v8proof/qhead_dose/run_roll', 'RUNF=/workspace/v8proof/qhead_dose_ls/run_roll', 1),
    ('ROUT=/workspace/v8proof/qhead_dose/rollouts.out', 'ROUT=/workspace/v8proof/qhead_dose_ls/rollouts.out', 1),
    ('qhead_n1/dose/pulled/rollouts.out', 'qhead_n1/dose/pulled_ls/rollouts.out', -1),
    ('qhead_read_qf', 'qhead_read_qfls', -1),
    ('wait_qhead_qf.sh', 'wait_qhead_qfls.sh', -1),
]


def apply(t, edits):
    for i, (old, new, n) in enumerate(edits):
        c = t.count(old)
        assert (c >= 1) if n == -1 else (c == n), "edit %d (%r): the old text occurs %d times, expected %s" % (i, old[:40], c, n)
        t = t.replace(old, new)
    return t


def rd(rel):
    t = io.open(os.path.join(HERE, rel), encoding="utf-8", newline="").read()
    assert "\r" not in t, rel
    return t


def names_qfls_text():
    return apply(MQ.names_qf_text(), NAMES)


def main():
    out = {"dose/pod_qhead_dose_fits_ls.sh": apply(rd("dose/pod_qhead_dose_fits.sh"), FITS),
           "ops/wait_qhead_dose_ls.sh": apply(rd("ops/wait_qhead_dose.sh"), CHAIN),
           "qhead_names_qfls.py": names_qfls_text(),
           "qhead_read_qfls.sh": apply(rd("qhead_read_qf.sh"), READ),
           "ops/wait_qhead_qfls.sh": apply(rd("ops/wait_qhead_qf.sh"), WAIT)}
    for rel, t in out.items():
        assert "LSD" not in t.replace("LSD type", "").replace("LSD dose", "") or rel.endswith(("fits_ls.sh", "dose_ls.sh", "qfls.py")), "an LSD name is left in " + rel
        p = os.path.join(HERE, rel)
        io.open(p, "w", encoding="utf-8", newline="\n").write(t)
        print(hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
