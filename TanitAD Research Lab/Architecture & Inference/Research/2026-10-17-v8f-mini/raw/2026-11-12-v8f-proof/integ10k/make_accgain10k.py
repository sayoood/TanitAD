"""make_accgain10k.py -- writes the generated tools of card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K: the acceleration the model reads at a replan tick, as a dose, at the 2.0 s cycle;
no rule, no training) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rz.py   -> v8f_t1_eval_reader_rule_rzg.py   (sha256 of the parent fa7207d72311b5cf, asserted; adds the readout causal_g with --acc-gain and --path-exec)
  pod_causlim10k_queue.sh         -> pod_accgain10k_queue.sh
  chain_causlim10k.sh             -> chain_accgain10k.sh
  podread_causlim.sh              -> podread_accgain.sh
  wait_podread_causlim.sh         -> wait_podread_accgain.sh
The originals are not changed. Pure text generator (no numpy, no torch). Usage: python make_accgain10k.py (in this folder)"""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_accgain10k.py"
PARENT_SHA = "fa7207d72311b5cf"


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


BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "closed" + " loop")


def no_banned(name, text):
    low = text.lower()
    for b in BANNED:
        assert b not in low, "%s contains %r" % (name, b)


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_rz.py")
assert hashlib.sha256(w.encode("utf-8")).hexdigest().startswith(PARENT_SHA), "the parent wrapper is not the one of card ADD55"
OLD_HEAD = 'v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.\n'
w = rep(w, '"""' + OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rzg.py -- written by make_accgain10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K): one more readout, causal_g,\n'
        'with two parameters.\n'
        '  --readout causal_g --acc-gain G : positions and headings as pinned; the speed column from executed steps only: at the knot of executed step i the step speed s[i], at the\n'
        '            half step before it s[i] - 0.5 x G x (s[i] - s[i-1]). The one-frame acceleration read at every executed knot, and so at the tick, is G times the executed one.\n'
        '            G = 1.0 goes through the formula of the readout causal and G = 0.0 through the formula of the readout causal_z, unchanged (so the dumps equal those of the two\n'
        '            readouts array for array); every other G goes through the general formula above.\n'
        '  --path-exec (only with causal_g): positions and headings from the pinned frames function given the executed knots only (as the readout backward does); the speed column as above.\n'
        'The readout log gets three more fields: acc_gain, path_exec and max_abs_acceleration_minus_gain_executed (the largest absolute difference between the one-frame\n'
        'acceleration read at the last executed knot and G times the executed acceleration, over all committed plans). Every other readout of the parent is unchanged (bit for bit).\n'
        'Header of the parent:\n' + OLD_HEAD)
w = rep(w, '"readout": "pinned", "hf":', '"readout": "pinned", "gain": None, "path_exec": False, "hf":')

NEW_FN = '''

def causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec=False, force_general=False):
    """the readout causal_g. Speed column: at the knot of executed step i the step speed s[i], at the half step before it s[i] - 0.5 * gain * (s[i] - s[i-1]); s as in
    causal_frames. gain == 1.0 -> the formula of causal, gain == 0.0 -> the formula of causal_z (both unchanged, unless force_general, used by the tool check only);
    other gains -> the general formula. path_exec: positions and headings from the pinned frames function given the executed knots only (backward_frames)."""
    base = orig
    if path_exec:
        def base(kp, kn, mm, p0):
            return backward_frames(orig, kp, kn, mm, p0)
    g = float(gain)
    if not force_general and g == 1.0:
        return causal_frames(base, k_prev, knots, m, psi0, "step")
    if not force_general and g == 0.0:
        return causal_frames(base, k_prev, knots, m, psi0, "zero")
    xy, psi, v = base(k_prev, knots, m, psi0)
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
    v = np.asarray(v, np.float64).copy()
    for i in range(1, m + 1):
        v[2 * (i - 1)] = s[i] - 0.5 * g * (s[i] - s[i - 1])
        v[2 * (i - 1) + 1] = s[i]
    return xy, psi, v
'''
w = rep(w, "\n\ndef counted_frames(orig, mode):", NEW_FN + "\n\ndef counted_frames(orig, mode, gain=None, path_exec=False):")
w = rep(w, '        if mode in ("causal", "causal_w", "causal_z"):\n',
        '        if mode == "causal_g":\n'
        '            xy, psi, v = causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec)\n'
        '        elif mode in ("causal", "causal_w", "causal_z"):\n')
w = rep(w, '        a0r, n_w = (float(v[-1]) - float(v[-2])) / (DT_S / 2.0), min(m, 5)\n',
        '        a0r, n_w = (float(v[-1]) - float(v[-2])) / (DT_S / 2.0), min(m, 5)\n'
        '        dga = abs(a0r - float(gain) * (float(s[m]) - float(s[m - 1])) / DT_S) if (mode == "causal_g" and gain is not None) else None\n')
w = rep(w, '            if np.isfinite(a0r) and np.isfinite(daw):\n',
        '            if dga is not None and np.isfinite(dga):\n'
        '                h["max_dga"] = max(h.get("max_dga", 0.0), dga)\n'
        '            if np.isfinite(a0r) and np.isfinite(daw):\n')
w = rep(w, '"max_abs_acceleration_minus_window": h.get("max_daw", 0.0)}',
        '"max_abs_acceleration_minus_window": h.get("max_daw", 0.0),\n'
        '           "acc_gain": STATE["gain"], "path_exec": bool(STATE["path_exec"]), "max_abs_acceleration_minus_gain_executed": h.get("max_dga")}')
w = rep(w, '    ap.add_argument("--readout", choices=("pinned", "backward", "causal", "causal_w", "causal_z"), default=None)\n',
        '    ap.add_argument("--readout", choices=("pinned", "backward", "causal", "causal_w", "causal_z", "causal_g"), default=None)\n'
        '    ap.add_argument("--acc-gain", type=float, default=None)\n'
        '    ap.add_argument("--path-exec", action="store_true")\n')
w = rep(w, '    orig_hf = T.hermite_frames\n    if mine.readout is not None:\n',
        '    if (mine.readout == "causal_g") != (mine.acc_gain is not None):\n'
        '        raise SystemExit("REFUSED: --acc-gain goes with --readout causal_g and only with it")\n'
        '    if mine.path_exec and mine.readout != "causal_g":\n'
        '        raise SystemExit("REFUSED: --path-exec goes with --readout causal_g and only with it")\n'
        '    if mine.acc_gain is not None and not np.isfinite(mine.acc_gain):\n'
        '        raise SystemExit("REFUSED: --acc-gain must be finite")\n'
        '    orig_hf = T.hermite_frames\n    if mine.readout is not None:\n')
w = rep(w, '        T.hermite_frames = counted_frames(orig_hf, mine.readout)\n',
        '        STATE["gain"] = None if mine.acc_gain is None else float(mine.acc_gain)\n'
        '        STATE["path_exec"] = bool(mine.path_exec)\n'
        '        T.hermite_frames = counted_frames(orig_hf, mine.readout, STATE["gain"], STATE["path_exec"])\n')
compile(w, "v8f_t1_eval_reader_rule_rzg.py", "exec")
no_banned("rzg", w)
save("v8f_t1_eval_reader_rule_rzg.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the pod queue
q = load("pod_causlim10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n",
        "#!/usr/bin/env bash\n# pod_accgain10k_queue.sh -- written by make_accgain10k.py from pod_causlim10k_queue.sh. Card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K): the tool check of control a4 first\n"
        "# (check_acc_gain.py on the pod, no model), then the two identity controls (R0GID1: gain 1.0, 10 steps, model seed 0, compared with the banked dump of R0C10, marker ga1cmp;\n"
        "# R0GID0: gain 0.0, 5 steps, model seed 0, compared with the banked dump of R0Z5, marker ga0cmp), then eight rollouts without a rule at 10 steps and two model seeds with the\n"
        "# wrapper v8f_t1_eval_reader_rule_rzg.py: R0G0 (gain 0), R0G50 (gain 0.5), R0Gm25 (gain -0.25), R0G0X (gain 0, --path-exec). No limit, no training. The queue waits for the end\n"
        "# marker of the queue of card ADD62 (variables WAIT_DONE / WAIT_STOP, defaults RULEFAST10K_DONE.txt / RULEFAST10K_STOP.txt in the folder $W; WAIT_MAX seconds, default 86400).\n"
        "# Markers START_ACCGAIN10K.txt / ACCGAIN10K_STOP.txt / ACCGAIN10K_DONE.txt; step markers ga4check, ga1cmp, ga0cmp. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "CAUSLIM10K", "ACCGAIN10K", 3)
q = rep(q, 'card ADD61: tool checks, then two controls and four rollouts without a rule (causal readout with the chord limit 3.0 / 8.0 m/s2 at 5 and at 10 steps); checkpoints of card ADD46"',
        'card ADD63: tool check, then two controls and eight rollouts without a rule (readout causal_g at 10 steps: gain 0, 0.5, -0.25, and gain 0 with the executed path only); checkpoints of card ADD46"')
q = rep(q, 'WAIT_DONE=${WAIT_DONE:-ACCWIN2S10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-ACCWIN2S10K_STOP.txt}',
        'WAIT_DONE=${WAIT_DONE:-RULEFAST10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-RULEFAST10K_STOP.txt}')
q = rep(q, "the tool checks and the rollouts of card ADD61 start", "the tool check and the rollouts of card ADD63 start")
q = rep(q, '  [ -f "$W/t1/I10R0C5_s0/T1_DONE" ] || stop "NO-BANKED-R0C5"\n',
        '  [ -f "$W/t1/I10R0C10_s0/T1_DONE" ] || stop "NO-BANKED-R0C10"\n  [ -f "$W/t1/I10R0Z5_s0/T1_DONE" ] || stop "NO-BANKED-R0Z5"\n')
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzc.py v8f_t1_eval_reader_rule_rz.py v8f_t1_eval_reader_rule_a.py v8f_t1_eval_reader_rule_ro.py test_accel_limit_chord_rzc.py check_tick_readout_rzc.py ',
        'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzg.py v8f_t1_eval_reader_rule_rz.py v8f_t1_eval_reader_rule_ro.py check_acc_gain.py ')
q = rep(q, "for nm in I10R0 I10R0M5 I10R0C5 I10R0E5 I10R0E10; do", "for nm in I10R0 I10R0C10 I10R0Bk10; do")
q = rep(q, '  [ -n "${AMAX:-}" ] && extra+=(--a-max "$AMAX" --b-lim "${BLIM:?}" --alim-log "$out/ticks/alim_log_$arm.npz" --alim-form chord)\n',
        '  [ -n "${ACCGAIN:-}" ] && extra+=(--acc-gain "$ACCGAIN")\n  [ "${PATHEXEC:-0}" = 1 ] && extra+=(--path-exec)\n')
NEWBLOCK = (
    '# ---------------------------------------------------------------- control a4 first: the tool check on the pod (no model, no rollout, no GPU, two threads); its output file and the copy of the wrapper it ran with are read by the reader\n'
    'if ! done_step ga4check; then\n'
    '  if [ "$DRY" = 1 ]; then\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_acc_gain.py $TREE > $W/accgain_test/CHECK_ACC_GAIN.txt"\n'
    '  else\n'
    '    mkdir -p "$W/accgain_test"\n'
    '    cp -f "$B/v8f_t1_eval_reader_rule_rzg.py" "$W/accgain_test/v8f_t1_eval_reader_rule_rzg.py"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_acc_gain.py" "$TREE" > "$W/accgain_test/CHECK_ACC_GAIN.txt" 2>&1\n'
    '    tail -n 1 "$W/accgain_test/CHECK_ACC_GAIN.txt" | grep -q "^ZZ CHECK-ACC-GAIN all hold: True" || stop "TOOL-CHECK-ACC-GAIN-FAILED"\n'
    '    mark ga4check\n'
    '  fi\n'
    'fi\n'
    'READOUT=causal_g ACCGAIN=1 REPLAN=10 rollout I10R0GID1 R0 0 0       # control a1: gain 1.0 at 10 steps: must give the banked dump of R0C10 (card ADD54)\n'
    'if [ "$DRY" != 1 ] && ! done_step ga1cmp; then\n'
    '  mkdir -p "$W/small_accgain"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0C10_s0/dump_I10R0C10_s0_m10_causal_heads" "$W/t1/I10R0GID1_s0/dump_I10R0GID1_s0_m10_causal_heads" > "$W/small_accgain/CMP_R0GID1.txt" 2>&1 || stop "A1-RERUN-DIFFERS"\n'
    '  mark ga1cmp\n'
    'fi\n'
    'READOUT=causal_g ACCGAIN=0 REPLAN=5 rollout I10R0GID0 R0 0 0       # control a2: gain 0.0 at 5 steps: must give the banked dump of R0Z5 (card ADD55)\n'
    'if [ "$DRY" != 1 ] && ! done_step ga0cmp; then\n'
    '  mkdir -p "$W/small_accgain"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0Z5_s0/dump_I10R0Z5_s0_m10_causal_heads" "$W/t1/I10R0GID0_s0/dump_I10R0GID0_s0_m10_causal_heads" > "$W/small_accgain/CMP_R0GID0.txt" 2>&1 || stop "A2-RERUN-DIFFERS"\n'
    '  mark ga0cmp\n'
    'fi\n'
    'for k in "${ARMS[@]}"; do\n'
    '  READOUT=causal_g ACCGAIN=0 REPLAN=10 rollout I10R0G0 R0 "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0.5 REPLAN=10 rollout I10R0G50 R0 "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=-0.25 REPLAN=10 rollout I10R0Gm25 R0 "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 REPLAN=10 rollout I10R0G0X R0 "$k" "$k"\n'
    'done\n')
q = span(q, '# ---------------------------------------------------------------- control c9 first: the two tool checks on the pod',
         'AMAX=3.0 BLIM=8.0 READOUT=causal REPLAN=10 rollout I10R0CE10 R0 "$k" "$k"\ndone\n', NEWBLOCK)
q = rep_code(q, "v8f_t1_eval_reader_rule_rzc.py", "v8f_t1_eval_reader_rule_rzg.py", 3)
c = code_lines(q)
for bad in ("CAUSLIM", "AMAX", "BLIM", "alim", "rzc", "chord", "I10R0CE", "cl9check", "ce1cmp", "ACCWIN2S10K"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 6 and c.count("ACCGAIN10K") >= 3 and "ga1cmp" in c and "ga0cmp" in c and "ga4check" in c, "queue sanity"
assert "MINGB=${I10_MIN_FREE_GB:-28}" in q and "MAXGPU=3" in q
no_banned("queue", q)
save("pod_accgain10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
ch = load("chain_causlim10k.sh")
ch = rep(ch, "#!/usr/bin/env bash\n# chain_causlim10k.sh -- written by make_causlim10k.py from chain_accwin2s10k.sh for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). The poll runs 48 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD58, up to 24 h). The header lines below are those of the parent.\n",
         "#!/usr/bin/env bash\n# chain_accgain10k.sh -- written by make_accgain10k.py from chain_causlim10k.sh for card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K). The poll runs 48 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD62, up to 24 h). The header lines below are those of the parent.\n")
ch = rep(ch, '"$I/pod_causlim10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzc.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/v8f_t1_eval_reader_rule_a.py" "$I/v8f_t1_eval_reader_rule_ro.py" "$I/test_accel_limit_chord_rzc.py" "$I/check_tick_readout_rzc.py"',
         '"$I/pod_accgain10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzg.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/v8f_t1_eval_reader_rule_ro.py" "$I/check_acc_gain.py"')
ch = rep(ch, "for nm in I10R0CEIDc I10R0CEIDe I10R0CE5 I10R0CE10; do", "for nm in I10R0GID1 I10R0GID0 I10R0G0 I10R0G50 I10R0Gm25 I10R0G0X; do")
ch = rep(ch, "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt $W/causlim_test/CHECK_TICK_READOUT_RZC.txt", "$W/accgain_test/CHECK_ACC_GAIN.txt")
for old, new in (("pod_causlim10k_queue.sh", "pod_accgain10k_queue.sh"), ("CAUSLIM10K", "ACCGAIN10K"), ("small_causlim", "small_accgain"), ("chain_causlim10k.log", "chain_accgain10k.log"),
                 ("queue_causlim_stdout", "queue_accgain_stdout")):
    assert ch.count(old) >= 1, old
    ch = ch.replace(old, new)
cc = code_lines(ch).lower()
for bad in ("causlim", "rzc", "chord", "ceid"):
    assert bad not in cc, bad
no_banned("chain", ch)
save("chain_accgain10k.sh", ch)

# ---------------------------------------------------------------- the pod read
pr = load("podread_causlim.sh")
pr = rep(pr, "# podread_causlim.sh -- pod-side read of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). Written from podread_accwin2s.sh (card ADD58) with the names, pairs and files of this card.\n",
         "# podread_accgain.sh -- pod-side read of card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K). Written by make_accgain10k.py from podread_causlim.sh (card ADD61) with the names, pairs and files of this card.\n")
pr = rep(pr, "# Same readers and rules as the pod reads of cards ADD47 to ADD58.\n", "# Same readers and rules as the pod reads of cards ADD47 to ADD61.\n")
pr = rep(pr, "only if e1 passes, the read of card ADD61.", "only if e1 passes, the read of card ADD63.")
pr = rep(pr, "# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_causlim10k.py and $P/root/integ10k/CAUSLIM10K_REUSED.json (sha256 of the result files of the\n"
         "# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/causlim_test/, the two compare files into $W/small_causlim/.\n",
         "# Files it needs on the pod (pushed by the lead before it runs): $P/root/integ10k/read_accgain10k.py and $P/root/integ10k/ACCGAIN10K_REUSED.json (sha256 of the result files of the\n"
         "# reused runs, recorded before the queue started); the tool-check output and the wrapper copy are written by the queue into $W/accgain_test/, the two compare files into $W/small_accgain/.\n")
pr = rep(pr, '[ -f "$PL/small_causlim/CMP_R0CEIDc.txt" ] || stop "NO-CMP-FILE-C1"\n[ -f "$PL/small_causlim/CMP_R0CEIDe.txt" ] || stop "NO-CMP-FILE-C2"\n',
         '[ -f "$PL/small_causlim/CMP_R0GID1.txt" ] || stop "NO-CMP-FILE-A1"\n[ -f "$PL/small_causlim/CMP_R0GID0.txt" ] || stop "NO-CMP-FILE-A2"\n')
pr = rep(pr, '[ -f "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" ] || stop "NO-TOOL-TEST-OUTPUT"\n[ -f "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"\n',
         '[ -f "$W/causlim_test/CHECK_ACC_GAIN.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"\n')
pr = rep(pr, "# ---- the read of card ADD61 (pairs of the card: SP, the two PARTS pairs, GC, PR and the pair beside PR)\n",
         "# ---- the read of card ADD63 (pairs of the card: D1, D2, ZP, PX, then the pairs beside)\n")
pr = rep(pr, 'PAIRS="I10R0CE5:I10R0M5,I10R0CE5:I10R0C5,I10R0CE5:I10R0E5,I10R0CE5:I10R0,I10R0CE10:I10R0,I10R0CE10:I10R0E10"\n',
         'PAIRS="I10R0G50:I10R0G0,I10R0C10:I10R0G50,I10R0G0:I10R0,I10R0G0X:I10R0G0,I10R0Gm25:I10R0G0,I10R0Gm25:I10R0Bk10,I10R0G0X:I10R0,I10R0G0:I10R0C10,I10R0G50:I10R0C10,I10R0Gm25:I10R0C10,I10R0G0X:I10R0C10"\n')
pr = rep(pr, '--reused "$P/root/integ10k/CAUSLIM10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzc.py" --test-wrapper "$W/causlim_test/v8f_t1_eval_reader_rule_rzc.py" --test-out-chord "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" --test-out-tick "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt"',
         '--reused "$P/root/integ10k/CAUSLIM10K_REUSED.json" --wrapper "$W/build/v8f_t1_eval_reader_rule_rzg.py" --test-wrapper "$W/causlim_test/v8f_t1_eval_reader_rule_rzg.py" --test-out "$W/causlim_test/CHECK_ACC_GAIN.txt"')
for old, new in (("causlim", "accgain"), ("CAUSLIM10K", "ACCGAIN10K"), ("CAUSLIM", "ACCGAIN"), ("read_accgain10k", "read_accgain10k")):
    pr = pr.replace(old, new)
low = code_lines(pr).lower()
for bad in ("causlim", "rzc", "chord", "ceid", "ce5", "ce10"):
    assert bad not in low, bad
no_banned("podread", pr)
save("podread_accgain.sh", pr)

# ---------------------------------------------------------------- the dev-box waiter
wt = load("wait_podread_causlim.sh")
wt = rep(wt, "# wait_podread_causlim.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K), written from wait_podread_accwin2s.sh (card ADD58), bash only (no Python on the dev box).\n",
         "# wait_podread_accgain.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD63 (ACCGAIN10K), written by make_accgain10k.py from wait_podread_causlim.sh (card ADD61), bash only (no Python on the dev box).\n")
wt = rep(wt, 'echo "ZZ WAIT-PODREAD-CAUSLIM DONE $(ts)"\n',
         'pod "cat $W/accgain_test/CHECK_ACC_GAIN.txt" > "$I/read_accgain/pod/CHECK_ACC_GAIN.txt"\n'
         'a=$(pod "sha256sum $W/accgain_test/CHECK_ACC_GAIN.txt | cut -c1-64"); b=$(sha256sum "$I/read_accgain/pod/CHECK_ACC_GAIN.txt" | cut -c1-64)\n'
         '[ "$a" = "$b" ] && echo "[$(ts)] pulled CHECK_ACC_GAIN.txt sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH CHECK_ACC_GAIN.txt"\n'
         'echo "ZZ WAIT-PODREAD-CAUSLIM DONE $(ts)"\n')
for old, new in (("causlim", "accgain"), ("CAUSLIM10K", "ACCGAIN10K"), ("CAUSLIM", "ACCGAIN")):
    wt = wt.replace(old, new)
assert "causlim" not in wt.lower()
no_banned("waiter", wt)
save("wait_podread_accgain.sh", wt)
print("wrapper sha256 %s" % WSHA)
print("ZZ MAKE-ACCGAIN10K DONE")
