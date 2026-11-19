#!/usr/bin/env python
"""make_qf_tools.py -- writes the tools of the dose arms LSDQF_s0, LSDQF_s1 (note PREREG_V8F_PROOF_ADD42_N2 section 3) from the tools of note N1 by asserted text
edits (each old text must occur the stated number of times):

  qhead_names_qf.py            from qhead_names_n2.py (itself built from qhead_names_n1.py): one arm type (LSD, arms LSDQF), names with the suffix -QF,
                               and two more gating controls read from files: d1 (label pack) and d2 (states, with the planted shift that must fail).
  ops/pod_qhead_qx_rollouts.sh from ops/pod_qhead_n1_rollouts.sh: run folder and heads folder by env (QN_RUNF, QN_HEADS), and an optional wait for the end marker
                               of another rollout queue (QN_WAIT_A, QN_WAIT_B: goes on when either file exists), so that two queues never take one slot.

  python make_qf_tools.py      (writes both files and prints the sha256)
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_names_n2 as MK  # noqa: E402

NAMES_EDITS = [
    ('"""qhead_names_n2.py -- ',
     '"""qhead_names_qf.py -- note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step): the arms LSDQF_s0, LSDQF_s1 (the LSD type only), names with the suffix -QF, and the\n'
     'controls d1 (the label pack of all clips equals the banked pack on the probe clips) and d2 (the new states equal the banked states; the planted shift fails) gate\n'
     'every name beside h1 to h8. Made from qhead_names_n2.py by make_qf_tools.py. The text of the earlier forms follows.\n\nqhead_names_n2.py -- ', 1),
    ('TYPES = {"LS": dict(qh="LSQE", none="LS", rh="LSRH", rl="LSRL", mc="LSMC", qc="LSQC", base="LS"),\n'
     '         "LSD": dict(qh="LSDQE", none="LSD", rh="LSDRH", rl="LSDRL", mc="LSDMC", qc="LSDQC", base="LSD")}\n',
     'TYPES = {"LSD": dict(qh="LSDQF", none="LSD", rh="LSDRH", rl="LSDRL", mc="LSDMC", qc="LSDQC", base="LSD")}\n', 1),
    ('-QE"', '-QF"', 10),
    ('    return C, hw, q3\n',
     '    d1 = jl(getattr(a, "d1", ""))\n'
     '    C["d1_label_pack_equals_banked_pack"] = {"PASS": bool(d1 and d1.get("pass") is True), "why": None if d1 else "no d1 record"}\n'
     '    d2 = {}\n'
     '    for ty in TYPES:\n'
     '        for s in SEEDS:\n'
     '            base = "%s_s%d" % (TYPES[ty]["base"], s)\n'
     '            v = jl(os.path.join(a.d2_dir, "d2_%s.json" % base)) if getattr(a, "d2_dir", "") else None\n'
     '            ps = (v or {}).get("planted_shift") or {}\n'
     '            d2[base] = {"PASS": bool(v and v.get("pass") is True and ps.get("fails") is True), "pass": (v or {}).get("pass"), "planted_shift_fails": ps.get("fails")}\n'
     '    C["d2_new_states_equal_banked_states"] = {"PASS": bool(all(v["PASS"] for v in d2.values())), "per_base": d2}\n'
     '    return C, hw, q3\n', 1),
    ('    for k in ("hw", "q3", "fit-dir", "h1-text", "tool-file", "rule-log-dir", "h3", "h6"):\n',
     '    for k in ("hw", "q3", "fit-dir", "h1-text", "tool-file", "rule-log-dir", "h3", "h6", "d1", "d2-dir"):\n', 1),
    ('res = {"card": "PREREG_V8F_PROOF_ADD42_N1", "reading_rule"',
     'res = {"card": "PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step: head fitted on all 4,572 training clips; arms LSDQF)", "reading_rule"', 1),
    ('print("== controls of note ADD42_N1 (printed first)")', 'print("== controls of note ADD42_N2 section 3 (printed first)")', 1),
]

ROLL_EDITS = [
    ('RUNF=$W/qhead_run_n1; HEADS=$RUNF/heads; RUN=$D/pod_qhead_run.sh\n',
     'RUNF=${QN_RUNF:-$W/qhead_run_n1}; HEADS=${QN_HEADS:-$RUNF/heads}; RUN=$D/pod_qhead_run.sh\nmkdir -p "$RUNF" || exit 9\n', 1),
    ('say "queue start; run script sha256',
     'if [ -n "${QN_WAIT_A:-}" ]; then j=0; while ! { [ -e "$QN_WAIT_A" ] || [ -e "${QN_WAIT_B:-$QN_WAIT_A}" ]; }; do j=$((j+1)); [ $j -gt 480 ] && stop "the other rollout queue has not ended in 8 h"; sleep 60; done; say "the other rollout queue has ended"; fi\n'
     'say "queue start; run script sha256', 1),
    ('# pod_qhead_n1_rollouts.sh -- runs ON THE POD (nohup). Notes PREREG_V8F_PROOF_ADD42_N1 and _N2:',
     '# pod_qhead_qx_rollouts.sh -- pod_qhead_n1_rollouts.sh with the run folder and the heads folder by env (QN_RUNF, QN_HEADS) and an optional wait for the end marker\n'
     '# of another rollout queue (QN_WAIT_A, QN_WAIT_B). Made by make_qf_tools.py. Used for the dose arms of note N2 section 3 (LSDQF). The text of the first form follows.\n'
     '# pod_qhead_n1_rollouts.sh -- runs ON THE POD (nohup). Notes PREREG_V8F_PROOF_ADD42_N1 and _N2:', 1),
]


def apply(t, edits):
    for i, (old, new, n) in enumerate(edits):
        assert t.count(old) == n, "edit %d: the old text occurs %d times, expected %d" % (i, t.count(old), n)
        t = t.replace(old, new)
    return t


def names_qf_text():
    n1 = io.open(os.path.join(HERE, "qhead_names_n1.py"), encoding="utf-8", newline="").read()
    return apply(MK.build(n1), NAMES_EDITS)


def main():
    out = []
    t = names_qf_text()
    assert "QE" not in t.split('"""', 2)[2], "a QE name is left in the code"
    p = os.path.join(HERE, "qhead_names_qf.py")
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
    out.append(p)
    r = io.open(os.path.join(HERE, "ops", "pod_qhead_n1_rollouts.sh"), encoding="utf-8", newline="").read()
    assert "\r" not in r
    p = os.path.join(HERE, "ops", "pod_qhead_qx_rollouts.sh")
    io.open(p, "w", encoding="utf-8", newline="\n").write(apply(r, ROLL_EDITS))
    out.append(p)
    for p in out:
        print(hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], os.path.relpath(p, HERE).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
