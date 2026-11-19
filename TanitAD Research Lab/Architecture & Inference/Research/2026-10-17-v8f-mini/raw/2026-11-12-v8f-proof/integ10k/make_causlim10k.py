"""make_causlim10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K: the causal readout of card ADD55 together with the chord form of the acceleration limit
of card ADD56, no rule, no training) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rz.py   -> v8f_t1_eval_reader_rule_rzc.py   (adds --alim-form path|chord; the chord function is the text of make_feasc10k.py, checked against
                                                                        v8f_t1_eval_reader_rule_ac.py; path (default) = the wrapper rz unchanged)
  test_accel_limit_chord.py       -> test_accel_limit_chord_rzc.py    (same 24 checks, against the new wrapper; prints the wrapper sha256)
  check_tick_readout_rz.py        -> check_tick_readout_rzc.py        (same check, against the new wrapper, plus rz against rzc bit for bit; prints the wrapper sha256)
  pod_accwin2s10k_queue.sh        -> pod_causlim10k_queue.sh
  chain_accwin2s10k.sh            -> chain_causlim10k.sh
The originals are not changed. Pure text generator (no numpy, no torch). Usage: python make_causlim10k.py (in this folder)"""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_causlim10k.py"


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert ME in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes) sha256 %s" % (name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:70], c, n)
    return text.replace(old, new)


def rep_code(text, old, new, n):
    """replaces in the lines that are not comments; n = the number of lines that must change"""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert k == n, "code anchor %r changed %d lines, expected %d" % (old, k, n)
    return "\n".join(out)


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


def code_lines(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# ---------------------------------------------------------------- the chord text of card ADD56, taken from its generator and checked against its wrapper
mf = load("make_feasc10k.py")
i0 = mf.index("CHORD = '''") + len("CHORD = '''")
i1 = mf.index("'''\nw = rep(w, \"\\n\\n\\nclass AccelLimited:\"")
CHORD = mf[i0:i1]
assert "\\" not in CHORD and CHORD.startswith("\n\ndef _sample_rows(") and CHORD.endswith("\n"), "the chord text of make_feasc10k.py was not cut cleanly"
ac = load("v8f_t1_eval_reader_rule_ac.py")
assert ac.count(CHORD) == 1, "the chord text of make_feasc10k.py is not in v8f_t1_eval_reader_rule_ac.py exactly once"

# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_rz.py")
OLD_HEAD = 'v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.\n'
w = rep(w, '"""' + OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rzc.py -- written by make_causlim10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K): the readouts of card ADD55\n'
        'together with the chord form of the acceleration limit of card ADD56: one more parameter, --alim-form path|chord. path (default) is the wrapper rz unchanged: the limit\n'
        'acts on the arc speed along the plan path. chord: the limit is held on the chord speeds between the knots of the limited plan (the function accel_limit_chord of\n'
        'v8f_t1_eval_reader_rule_ac.py, same text). The limit acts on every committed plan of the model and of the kinematic prior alone, and the readout acts on every committed\n'
        'plan, so a run with both has both on the model rollout and on the prior rollout. Text of the source below.\n' + OLD_HEAD)
w = rep(w, '"alim": None, "m": 10, "alog": {},', '"alim": None, "aform": "path", "m": 10, "alog": {},')
w = rep(w, "\n\n\nclass AccelLimited:", CHORD + "\n\nclass AccelLimited:")
w = rep(w, 'out, info = accel_limit(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])',
        'out, info = (accel_limit_chord if STATE["aform"] == "chord" else accel_limit)(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])')
w = rep(w, "form=np.asarray(form), replan_steps=", 'form=np.asarray(form), alim_form=np.asarray(STATE["aform"]), replan_steps=')
w = rep(w, '    ap.add_argument("--alim-log", default="")\n', '    ap.add_argument("--alim-log", default="")\n    ap.add_argument("--alim-form", choices=("path", "chord"), default="path")\n')
w = rep(w, '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n',
        '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n'
        '    if mine.alim_form != "path" and STATE["alim"] is None:\n'
        '        raise SystemExit("REFUSED: --alim-form chord without the two limits would change nothing")\n'
        '    STATE["aform"] = mine.alim_form\n')
compile(w, "v8f_t1_eval_reader_rule_rzc.py", "exec")
save("v8f_t1_eval_reader_rule_rzc.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the unit test of the chord limit against the new wrapper
t = load("test_accel_limit_chord.py")
t = rep(t, '"""test_accel_limit_chord.py -- pure tests of accel_limit_chord of v8f_t1_eval_reader_rule_ac.py (card PREREG_V8F_PROOF_ADD56), no model, no GPU. Run on the pod:\n',
        '"""test_accel_limit_chord_rzc.py -- written by make_causlim10k.py from test_accel_limit_chord.py for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K): the same 24 checks against\n'
        'v8f_t1_eval_reader_rule_rzc.py (AC below is that wrapper); it prints the sha256 of the wrapper file it ran with. Text of the source below.\n'
        'test_accel_limit_chord.py -- pure tests of accel_limit_chord of v8f_t1_eval_reader_rule_ac.py (card PREREG_V8F_PROOF_ADD56), no model, no GPU. Run on the pod:\n')
t = rep(t, "import glob\nimport os\n", "import glob\nimport hashlib\nimport os\n")
t = rep(t, "import v8f_t1_eval_reader_rule_ac as AC  # noqa: E402", "import v8f_t1_eval_reader_rule_rzc as AC  # noqa: E402")
t = rep(t, 'print("ZZ TEST-ACCEL-LIMIT-CHORD %s %d/%d" % ("PASS" if all(RES) else "FAIL", sum(RES), len(RES)))',
        'print("wrapper rzc sha256 %s" % hashlib.sha256(open(AC.__file__, "rb").read()).hexdigest())\n'
        'print("ZZ TEST-ACCEL-LIMIT-CHORD %s %d/%d" % ("PASS" if all(RES) else "FAIL", sum(RES), len(RES)))')
compile(t, "test_accel_limit_chord_rzc.py", "exec")
save("test_accel_limit_chord_rzc.py", t)

# ---------------------------------------------------------------- the tick-readout check against the new wrapper
k = load("check_tick_readout_rz.py")
k = rep(k, '"""check_tick_readout_rz.py -- tool check of the two readouts of v8f_t1_eval_reader_rule_rz.py (card PREREG_V8F_PROOF_ADD55), no model, no GPU, no data of any card.\n',
        '"""check_tick_readout_rzc.py -- written by make_causlim10k.py from check_tick_readout_rz.py for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K): the same check through\n'
        'v8f_t1_eval_reader_rule_rzc.py (RZ below is that wrapper), plus the five readouts of rzc against those of rz bit for bit and the default of the limit form. It prints the\n'
        'sha256 of the wrapper file it ran with. Last line: ZZ CHECK-TICK-READOUT-RZC all hold: True/False. Text of the source below.\n'
        'check_tick_readout_rz.py -- tool check of the two readouts of v8f_t1_eval_reader_rule_rz.py (card PREREG_V8F_PROOF_ADD55), no model, no GPU, no data of any card.\n')
k = rep(k, "import os\nimport sys\n\nimport numpy as np\n", "import hashlib\nimport os\nimport sys\n\nimport numpy as np\n")
k = rep(k, "import v8f_t1_eval_reader_rule_rz as RZ  # noqa: E402\n", "import v8f_t1_eval_reader_rule_rz as RZ0  # noqa: E402\nimport v8f_t1_eval_reader_rule_rzc as RZ  # noqa: E402\n")
k = rep(k, 'print("pinned tool %s" % os.path.basename(path))\n',
        'same_rz = True\n'
        'for mode in ("pinned", "backward", "causal", "causal_w", "causal_z"):\n'
        '    a2 = RZ0.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)\n'
        '    b2 = RZ.counted_frames(T.hermite_frames, mode)(k_prev, knots, m, 0.1)\n'
        '    same_rz = same_rz and all(np.array_equal(x1, x2) for x1, x2 in zip(a2, b2))\n'
        'form_ok = RZ.STATE["aform"] == "path" and RZ.STATE["alim"] is None and RZ.STATE["readout"] == "pinned" and callable(RZ.accel_limit_chord)\n'
        'print("the five readouts through the wrapper of card ADD55 and through the wrapper rzc agree bit for bit: %s | default limit form path, no limit, readout pinned: %s" % (same_rz, form_ok))\n'
        'ok_all = ok_all and same_rz and form_ok\n'
        'print("pinned tool %s | wrapper rzc sha256 %s" % (os.path.basename(path), hashlib.sha256(open(RZ.__file__, "rb").read()).hexdigest()))\n')
k = rep(k, 'print("ZZ CHECK-TICK-READOUT-RZ all hold: %s" % ok_all)', 'print("ZZ CHECK-TICK-READOUT-RZC all hold: %s" % ok_all)')
compile(k, "check_tick_readout_rzc.py", "exec")
save("check_tick_readout_rzc.py", k)

# ---------------------------------------------------------------- the pod queue
q = load("pod_accwin2s10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n",
        "#!/usr/bin/env bash\n# pod_causlim10k_queue.sh -- written by make_causlim10k.py from pod_accwin2s10k_queue.sh. Card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K): the tool checks of control c9 first\n"
        "# (the unit test of the chord limit and the tick-readout check, both against v8f_t1_eval_reader_rule_rzc.py, on the pod, no model), then the two controls (R0CEIDc: causal readout,\n"
        "# limits out of reach, chord form, 5 steps, model seed 0, compared with the banked dump of R0C5; R0CEIDe: pinned readout, limits 3.0 / 8.0, chord form, 5 steps, model seed 0,\n"
        "# compared with the banked dump of R0E5), then the causal readout with the chord limit 3.0 / 8.0 m/s2 at 5 steps (R0CE5) and at 10 steps (R0CE10) at two model seeds. No rule,\n"
        "# no training. The queue waits for the end marker of the queue of card ADD58 (variables WAIT_DONE / WAIT_STOP, defaults ACCWIN2S10K_DONE.txt / ACCWIN2S10K_STOP.txt in the\n"
        "# folder $W; WAIT_MAX seconds, default 86400). Markers START_CAUSLIM10K.txt / CAUSLIM10K_STOP.txt / CAUSLIM10K_DONE.txt; step markers cl9check, ce1cmp, ce1ecmp.\n"
        "# The header lines below are those of the earlier queues.\n")
q = rep_code(q, "ACCWIN2S10K", "CAUSLIM10K", 3)
q = rep(q, 'card ADD58: tool check, then five rollouts without a rule (acceleration read at the replan: over the last 1.0 s at 5 steps as control; over the last 2.0 s of executed motion at 5 and at 10 steps); checkpoints of card ADD46"',
        'card ADD61: tool checks, then two controls and four rollouts without a rule (causal readout with the chord limit 3.0 / 8.0 m/s2 at 5 and at 10 steps); checkpoints of card ADD46"')
q = rep(q, 'WAIT_DONE=${WAIT_DONE:-RULE1S10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-RULE1S10K_STOP.txt}',
        'WAIT_DONE=${WAIT_DONE:-ACCWIN2S10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-ACCWIN2S10K_STOP.txt}; WAIT_MAX=${WAIT_MAX:-86400}')
q = rep(q, '[ $(( $(date +%s) - t0 )) -gt 28800 ] && stop "WAIT-PREVIOUS-QUEUE-TIMEOUT"', '[ $(( $(date +%s) - t0 )) -gt "$WAIT_MAX" ] && stop "WAIT-PREVIOUS-QUEUE-TIMEOUT"')
q = rep(q, "the tool check and the rollouts of card ADD58 start", "the tool checks and the rollouts of card ADD61 start")
q = rep(q, '[ -f "$W/t1/I10R0W5_s0/T1_DONE" ] || stop "NO-BANKED-R0W5"', '[ -f "$W/t1/I10R0C5_s0/T1_DONE" ] || stop "NO-BANKED-R0C5"')
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rv.py v8f_t1_eval_reader_rule_rz.py check_tick_readout_rv.py ',
        'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzc.py v8f_t1_eval_reader_rule_rz.py v8f_t1_eval_reader_rule_a.py v8f_t1_eval_reader_rule_ro.py test_accel_limit_chord_rzc.py check_tick_readout_rzc.py ')
q = rep_code(q, "v8f_t1_eval_reader_rule_rv.py", "v8f_t1_eval_reader_rule_rzc.py", 3)
q = rep(q, "for nm in I10R0 I10R0M5 I10R0W5 I10R0Z5 I10R0W10; do", "for nm in I10R0 I10R0M5 I10R0C5 I10R0E5 I10R0E10; do")
q = rep_code(q, '--alim-log "$out/ticks/alim_log_$arm.npz")', '--alim-log "$out/ticks/alim_log_$arm.npz" --alim-form chord)', 1)
q = rep(q, '| cut -c1-64) started $(ts)" > "$out/ticks/run_$arm.txt"',
        '| cut -c1-64) wrapper sha256 $(sha256sum "$B/v8f_t1_eval_reader_rule_rzc.py" | cut -c1-64) started $(ts)" > "$out/ticks/run_$arm.txt"')
q = span(q, '# ---------------------------------------------------------------- control v9 first: the tool check on the pod', '  READOUT=causal_v REPLAN=10 rollout I10R0V10 R0 "$k" "$k"\ndone\n',
         '# ---------------------------------------------------------------- control c9 first: the two tool checks on the pod (no model, no rollout, no GPU, two threads); their output files and the copy of the wrapper they ran with are read by the reader\n'
         'if ! done_step cl9check; then\n'
         '  if [ "$DRY" = 1 ]; then\n'
         '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/test_accel_limit_chord_rzc.py $W/t1/I10R0M5_s0/dump_I10R0M5_s0_m10_causal_heads > $W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt"\n'
         '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_tick_readout_rzc.py $TREE > $W/causlim_test/CHECK_TICK_READOUT_RZC.txt"\n'
         '  else\n'
         '    mkdir -p "$W/causlim_test"\n'
         '    cp -f "$B/v8f_t1_eval_reader_rule_rzc.py" "$W/causlim_test/v8f_t1_eval_reader_rule_rzc.py"\n'
         '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 1800 "$PY" "$B/test_accel_limit_chord_rzc.py" "$W/t1/I10R0M5_s0/dump_I10R0M5_s0_m10_causal_heads" > "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" 2>&1\n'
         '    tail -n 1 "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" | grep -q "^ZZ TEST-ACCEL-LIMIT-CHORD PASS 24/24" || stop "TOOL-TEST-CHORD-RZC-FAILED"\n'
         '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_tick_readout_rzc.py" "$TREE" > "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" 2>&1\n'
         '    tail -n 1 "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" | grep -q "^ZZ CHECK-TICK-READOUT-RZC all hold: True" || stop "TOOL-CHECK-RZC-FAILED"\n'
         '    mark cl9check\n'
         '  fi\n'
         'fi\n'
         'AMAX=1e9 BLIM=1e9 READOUT=causal REPLAN=5 rollout I10R0CEIDc R0 0 0       # control c1: causal readout, limits out of reach, chord form: must give the banked dump of R0C5 (card ADD54)\n'
         'if [ "$DRY" != 1 ] && ! done_step ce1cmp; then\n'
         '  mkdir -p "$W/small_causlim"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0C5_s0/dump_I10R0C5_s0_m10_causal_heads" "$W/t1/I10R0CEIDc_s0/dump_I10R0CEIDc_s0_m10_causal_heads" > "$W/small_causlim/CMP_R0CEIDc.txt" 2>&1 || stop "C1-RERUN-DIFFERS"\n'
         '  mark ce1cmp\n'
         'fi\n'
         'AMAX=3.0 BLIM=8.0 READOUT=pinned REPLAN=5 rollout I10R0CEIDe R0 0 0       # control c2: pinned readout, limits 3.0 / 8.0, chord form: must give the banked dump of R0E5 (card ADD56)\n'
         'if [ "$DRY" != 1 ] && ! done_step ce1ecmp; then\n'
         '  mkdir -p "$W/small_causlim"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0E5_s0/dump_I10R0E5_s0_m10_causal_heads" "$W/t1/I10R0CEIDe_s0/dump_I10R0CEIDe_s0_m10_causal_heads" > "$W/small_causlim/CMP_R0CEIDe.txt" 2>&1 || stop "C2-RERUN-DIFFERS"\n'
         '  mark ce1ecmp\n'
         'fi\n'
         'for k in "${ARMS[@]}"; do\n'
         '  AMAX=3.0 BLIM=8.0 READOUT=causal REPLAN=5 rollout I10R0CE5 R0 "$k" "$k"\n'
         'done\n'
         'for k in "${ARMS[@]}"; do\n'
         '  AMAX=3.0 BLIM=8.0 READOUT=causal REPLAN=10 rollout I10R0CE10 R0 "$k" "$k"\n'
         'done\n')
c = code_lines(q)
for bad in ("HANDOVER10K", "I10R0VID", "I10R0V5", "I10R0V10", "v1cmp", "v9check", "small_accwin", "accwin2s_test", "_rv.py", "causal_v", "causal_w ", "RULE1S10K"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 4 and c.count("CAUSLIM10K") >= 3 and c.count("--alim-form chord") == 1 and "ce1cmp" in c and "ce1ecmp" in c, "queue sanity"
save("pod_causlim10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
ch = load("chain_accwin2s10k.sh")
ch = rep(ch, "#!/usr/bin/env bash\n# chain_accwin2s10k.sh -- written by make_accwin2s10k.py from chain_accwin10k.sh for card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K). The poll runs 24 h (the queue first waits for the\n"
         "# end marker of the previous queue, up to 8 h). The header lines below are those of the parent.\n",
         "#!/usr/bin/env bash\n# chain_causlim10k.sh -- written by make_causlim10k.py from chain_accwin2s10k.sh for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). The poll runs 48 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD58, up to 24 h). The header lines below are those of the parent.\n")
ch = rep(ch, '"$I/pod_accwin2s10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rv.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/check_tick_readout_rv.py"',
         '"$I/pod_causlim10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzc.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/v8f_t1_eval_reader_rule_a.py" "$I/v8f_t1_eval_reader_rule_ro.py" "$I/test_accel_limit_chord_rzc.py" "$I/check_tick_readout_rzc.py"')
ch = rep(ch, "for nm in I10R0VID I10R0V5 I10R0V10; do", "for nm in I10R0CEIDc I10R0CEIDe I10R0CE5 I10R0CE10; do")
ch = rep(ch, "$W/accwin2s_test/CHECK_TICK_READOUT_RV.txt", "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt $W/causlim_test/CHECK_TICK_READOUT_RZC.txt")
ch = rep(ch, '[ "$k" -gt 1440 ] && { st=STOP; say "not ended after 24 h"; break; }', '[ "$k" -gt 2880 ] && { st=STOP; say "not ended after 48 h"; break; }')
ch = rep(ch, "after 24 h (a poll", "after 48 h (a poll")
for old, new in (("pod_accwin2s10k_queue.sh", "pod_causlim10k_queue.sh"), ("ACCWIN2S10K", "CAUSLIM10K"), ("small_accwin2s", "small_causlim"), ("chain_accwin2s10k.log", "chain_causlim10k.log"),
                 ("queue_accwin2s_stdout", "queue_causlim_stdout")):
    assert ch.count(old) >= 1, old
    ch = ch.replace(old, new)
cc = code_lines(ch)
for bad in ("ACCWIN2S", "accwin2s", "I10R0VID", "_rv.py", "check_tick_readout_rv"):
    assert bad not in cc, bad
save("chain_causlim10k.sh", ch)
print("wrapper sha256 %s" % WSHA)
print("ZZ MAKE-CAUSLIM10K DONE")
