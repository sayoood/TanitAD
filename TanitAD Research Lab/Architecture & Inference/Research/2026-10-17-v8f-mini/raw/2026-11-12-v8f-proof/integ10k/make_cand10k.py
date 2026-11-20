"""make_cand10k.py -- writes the generated tools of card PREREG_V8F_PROOF_ADD65 (CAND10K: the lead rule under the causal readout without own acceleration, with and without the second
look, against the programme's convention; no training) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rzf.py -> v8f_t1_eval_reader_rule_rzgf.py (sha256 of the parent 51b3e783536689f6, asserted; the readout edits of make_accgain10k.py: causal_g, --acc-gain, --path-exec)
  check_rule_every.py            -> check_rule_every_rzgf.py          (the wrapper name only)
  check_acc_gain.py              -> check_acc_gain_rzgf.py            (the wrapper name only)
  pod_rulefast10k_queue.sh       -> pod_cand10k_queue.sh
  chain_rulefast10k.sh           -> chain_cand10k.sh
  podread_rulefast.sh            -> podread_cand.sh
  wait_podread_rulefast.sh       -> wait_podread_cand.sh
The originals are not changed. Pure text generator (no numpy, no torch). Usage: python make_cand10k.py (in this folder)"""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_cand10k.py"
PARENT_SHA = "51b3e783536689f6"


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


def rep_all_code(text, old, new):
    """replaces in the lines that are not comments (at least one line must change)"""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert k >= 1, "code anchor %r not found" % old
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


# ---------------------------------------------------------------- the wrapper: the readout edits of make_accgain10k.py applied to the wrapper of card ADD62
w = load("v8f_t1_eval_reader_rule_rzf.py")
assert hashlib.sha256(w.encode("utf-8")).hexdigest().startswith(PARENT_SHA), "the parent wrapper is not the one of card ADD62"
OLD_HEAD = '"""v8f_t1_eval_reader_rule_rzf.py -- written by make_rulefast10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): one more parameter,\n'
w = rep(w, OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rzgf.py -- written by make_cand10k.py from v8f_t1_eval_reader_rule_rzf.py for card PREREG_V8F_PROOF_ADD65 (CAND10K): the readout causal_g of card ADD63,\n'
        'with two parameters, added to the wrapper of card ADD62 (--rule-every). Nothing else is new.\n'
        '  --readout causal_g --acc-gain G : positions and headings as pinned; the speed column from executed steps only: at the knot of executed step i the step speed s[i], at the\n'
        '            half step before it s[i] - 0.5 x G x (s[i] - s[i-1]). The one-frame acceleration read at every executed knot, and so at the tick, is G times the executed one.\n'
        '            G = 1.0 goes through the formula of the readout causal and G = 0.0 through the formula of the readout causal_z, unchanged (so the dumps equal those of the two\n'
        '            readouts array for array); every other G goes through the general formula above.\n'
        '  --path-exec (only with causal_g): positions and headings from the pinned frames function given the executed knots only (as the readout backward does); the speed column as above.\n'
        'The readout log gets three more fields: acc_gain, path_exec and max_abs_acceleration_minus_gain_executed (the largest absolute difference between the one-frame\n'
        'acceleration read at the last executed knot and G times the executed acceleration, over all committed plans). Every other readout and the parameter --rule-every of the parent\n'
        'are unchanged (bit for bit). Readout K of card ADD65 is --readout causal_g --acc-gain 0 --path-exec; the second look is --rule-every 5.\n'
        'Header of the parent:\n'
        'v8f_t1_eval_reader_rule_rzf.py -- written by make_rulefast10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): one more parameter,\n')
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
compile(w, "v8f_t1_eval_reader_rule_rzgf.py", "exec")
no_banned("rzgf", w)
save("v8f_t1_eval_reader_rule_rzgf.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the two tool checks: copies with the wrapper name changed and nothing else
cr = load("check_rule_every.py")
assert cr.count("rzf") == 5 and "rzgf" not in cr, cr.count("rzf")
cr = cr.replace("rzf", "rzgf")
cr = rep(cr, '"""check_rule_every.py -- ', '"""check_rule_every_rzgf.py (copy of check_rule_every.py, written by make_cand10k.py; the wrapper name is the only change) -- ')
compile(cr, "check_rule_every_rzgf.py", "exec")
no_banned("check_rule_every_rzgf", cr)
save("check_rule_every_rzgf.py", cr)
ca = load("check_acc_gain.py")
assert ca.count("rzg") == 3 and "rzgf" not in ca, ca.count("rzg")
ca = ca.replace("rzg", "rzgf")
ca = rep(ca, '"""check_acc_gain.py -- ', '"""check_acc_gain_rzgf.py (copy of check_acc_gain.py, written by make_cand10k.py; the wrapper name is the only change) -- ')
compile(ca, "check_acc_gain_rzgf.py", "exec")
no_banned("check_acc_gain_rzgf", ca)
save("check_acc_gain_rzgf.py", ca)

# ---------------------------------------------------------------- the pod queue
WR = "v8f_t1_eval_reader_rule_rzgf.py"
q = load("pod_rulefast10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n",
        "#!/usr/bin/env bash\n# pod_cand10k_queue.sh -- written by make_cand10k.py from pod_rulefast10k_queue.sh. Card PREREG_V8F_PROOF_ADD65 (CAND10K): the two tool checks of control c6 first (check_rule_every_rzgf.py\n"
        "# and check_acc_gain_rzgf.py on the pod, no model; step marker ck6check), then the control I10R0KID (no rule, --readout causal_g --acc-gain 0 --path-exec, model seed 0, compared file for\n"
        "# file with the banked dump of I10R0G0X_s0: marker ck1cmp), then the control I10RLcKIDp (label-fed, pinned readout, --rule-every 5, model seed 0, compared with the banked dump of\n"
        "# I10RLcF5_s0: marker ck2cmp), then at two model seeds the eight rollouts of the card table with readout K: I10RLcK, I10RQ50K (rule at the tick), I10RLcKF5, I10RQ50KF5 (rule every 5\n"
        "# steps), reader seed = model seed, all at the 2.0 s replan interval, with the wrapper " + WR + ". No training. The queue waits for the end marker of the queue of card ADD63\n"
        "# (variables WAIT_DONE / WAIT_STOP, defaults ACCGAIN10K_DONE.txt / ACCGAIN10K_STOP.txt in the folder $W; WAIT_MAX seconds, default 86400). Markers START_CAND10K.txt / CAND10K_STOP.txt /\n"
        "# CAND10K_DONE.txt; step markers ck6check, ck1cmp, ck2cmp. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "RULEFAST10K", "CAND10K", 3)
q = rep(q, 'card ADD62: tool check, then the control RLcFID and six rollouts with the lead rule applied again inside the 2.0 s block (every 5 and every 2 steps); checkpoints of card ADD46"',
        'card ADD65: two tool checks, then two controls and eight rollouts with the lead rule under readout K, at the tick and every 5 steps; checkpoints of card ADD46"')
q = rep(q, 'WAIT_DONE=${WAIT_DONE:-CAUSLIM10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-CAUSLIM10K_STOP.txt}', 'WAIT_DONE=${WAIT_DONE:-ACCGAIN10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-ACCGAIN10K_STOP.txt}')
q = rep(q, "the tool check and the rollouts of card ADD62 start", "the tool checks and the rollouts of card ADD65 start")
q = rep(q, '  [ -f "$W/t1/I10RLc_s0/T1_DONE" ] || stop "NO-BANKED-RLC"\n',
        '  [ -f "$W/t1/I10R0G0X_s0/T1_DONE" ] || stop "NO-BANKED-R0G0X"\n  [ -f "$W/t1/I10RLcF5_s0/T1_DONE" ] || stop "NO-BANKED-RLCF5"\n')
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzf.py check_rule_every.py ',
        'for f in cmp_dumps.py ' + WR + ' v8f_t1_eval_reader_rule_rz.py check_rule_every_rzgf.py check_acc_gain_rzgf.py ')
q = rep(q, "for nm in I10R0 I10RLc I10RQ50 I10RLcM5 I10RQ50M5; do", "for nm in I10R0 I10R0G0X I10RLc I10RQ50 I10RLcF5 I10RQ50F5; do")
q = rep(q, '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n',
        '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n  [ -n "${ACCGAIN:-}" ] && extra+=(--acc-gain "$ACCGAIN")\n  [ "${PATHEXEC:-0}" = 1 ] && extra+=(--path-exec)\n')
q = rep(q, 'replan steps ${REPLAN:-10} rule every ${EVERY:-10} model seed $m', 'replan steps ${REPLAN:-10} rule every ${EVERY:-10} readout ${READOUT:-pinned} model seed $m')
NEWBLOCK = (
    '# ---------------------------------------------------------------- control c6 first: the two tool checks on the pod (no model, no rollout, no GPU, two threads); their output files and the copy of the wrapper they ran with are read by the reader\n'
    'if ! done_step ck6check; then\n'
    '  if [ "$DRY" = 1 ]; then\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_rule_every_rzgf.py $TREE > $W/cand_test/CHECK_RULE_EVERY_RZGF.txt"\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_acc_gain_rzgf.py $TREE > $W/cand_test/CHECK_ACC_GAIN_RZGF.txt"\n'
    '  else\n'
    '    mkdir -p "$W/cand_test"\n'
    '    cp -f "$B/' + WR + '" "$W/cand_test/' + WR + '"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 1800 "$PY" "$B/check_rule_every_rzgf.py" "$TREE" > "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" 2>&1\n'
    '    tail -n 1 "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" | grep -q "^ZZ CHECK-RULE-EVERY all hold: True" || stop "TOOL-CHECK-RULE-EVERY-RZGF-FAILED"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_acc_gain_rzgf.py" "$TREE" > "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt" 2>&1\n'
    '    tail -n 1 "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt" | grep -q "^ZZ CHECK-ACC-GAIN all hold: True" || stop "TOOL-CHECK-ACC-GAIN-RZGF-FAILED"\n'
    '    mark ck6check\n'
    '  fi\n'
    'fi\n'
    'READOUT=causal_g ACCGAIN=0 PATHEXEC=1 rollout I10R0KID R0 0 0       # control c1: readout K without a rule must give the banked dump of I10R0G0X at model seed 0\n'
    'if [ "$DRY" != 1 ] && ! done_step ck1cmp; then\n'
    '  mkdir -p "$W/small_cand"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0G0X_s0/dump_I10R0G0X_s0_m10_causal_heads" "$W/t1/I10R0KID_s0/dump_I10R0KID_s0_m10_causal_heads" > "$W/small_cand/CMP_R0KID.txt" 2>&1 || stop "C1-RERUN-DIFFERS"\n'
    '  mark ck1cmp\n'
    'fi\n'
    'EVERY=5 rollout I10RLcKIDp RLc 0 0       # control c2: the pinned readout with the second look must give the banked dump of I10RLcF5 at model seed 0\n'
    'if [ "$DRY" != 1 ] && ! done_step ck2cmp; then\n'
    '  mkdir -p "$W/small_cand"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLcF5_s0/dump_I10RLcF5_s0_m10_causal_heads" "$W/t1/I10RLcKIDp_s0/dump_I10RLcKIDp_s0_m10_causal_heads" > "$W/small_cand/CMP_RLcKIDp.txt" 2>&1 || stop "C2-RERUN-DIFFERS"\n'
    '  mark ck2cmp\n'
    'fi\n'
    'for k in "${ARMS[@]}"; do\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 rollout I10RLcK RLc "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50K RR "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 EVERY=5 rollout I10RLcKF5 RLc "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50KF5 RR "$k" "$k"\n'
    'done\n')
q = span(q, '# ---------------------------------------------------------------- control k5 first: the tool check on the pod', '  EVERY=2 rollout I10RLcF2 RLc "$k" "$k"      # dose, names nothing\ndone\n', NEWBLOCK)
q = rep_code(q, "v8f_t1_eval_reader_rule_rzf.py", WR, 3)
c = code_lines(q)
for bad in ("RULEFAST10K", "RLcFID", "RLcF2", "rf9check", "fr1cmp", "rulefast", "CHECK_RULE_EVERY.txt", "CAUSLIM"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 6 and c.count("CAND10K") >= 3 and "ck1cmp" in c and "ck2cmp" in c and "ck6check" in c and "wrapper sha256" in c, "queue sanity"
assert "ACCGAIN10K_DONE.txt" in c and "ACCGAIN10K_STOP.txt" in c and "MINGB=${I10_MIN_FREE_GB:-28}" in q and "MAXGPU=3" in q
no_banned("queue", q)
save("pod_cand10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
ch = load("chain_rulefast10k.sh")
ch = rep(ch, "#!/usr/bin/env bash\n# chain_rulefast10k.sh -- written by make_rulefast10k.py from chain_causlim10k.sh for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K). The poll runs 72 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD61, up to 24 h). The header lines below are those of the parent.\n",
         "#!/usr/bin/env bash\n# chain_cand10k.sh -- written by make_cand10k.py from chain_rulefast10k.sh for card PREREG_V8F_PROOF_ADD65 (CAND10K). The poll runs 72 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD63, up to 24 h). The header lines below are those of the parent.\n")
ch = rep(ch, '"$I/pod_rulefast10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzf.py" "$I/check_rule_every.py"',
         '"$I/pod_cand10k_queue.sh" "$I/' + WR + '" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/check_rule_every_rzgf.py" "$I/check_acc_gain_rzgf.py"')
ch = rep(ch, "for nm in I10RLcFID I10RLcF5 I10RQ50F5 I10RLcF2; do", "for nm in I10R0KID I10RLcKIDp I10RLcK I10RQ50K I10RLcKF5 I10RQ50KF5; do")
ch = rep(ch, "$W/rulefast_test/CHECK_RULE_EVERY.txt", "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt $W/cand_test/CHECK_ACC_GAIN_RZGF.txt")
ch = rep(ch, "EXITS on CAUSLIM10K_DONE, CAUSLIM10K_STOP or after 72 h", "EXITS on CAND10K_DONE, CAND10K_STOP or after 72 h")
for old, new in (("pod_rulefast10k_queue.sh", "pod_cand10k_queue.sh"), ("small_rulefast", "small_cand"), ("chain_rulefast10k.log", "chain_cand10k.log"), ("queue_rulefast_stdout", "queue_cand_stdout"),
                 ("START_RULEFAST10K", "START_CAND10K"), ("CHAIN-RULEFAST10K", "CHAIN-CAND10K"), ("RULEFAST10K_DONE", "CAND10K_DONE"), ("RULEFAST10K_STOP", "CAND10K_STOP")):
    assert ch.count(old) >= 1, old
    ch = ch.replace(old, new)
cc = code_lines(ch)
for bad in ("RULEFAST", "rulefast", "RLcFID", "RLcF2", "rulefast_test", "CAUSLIM"):
    assert bad not in cc, bad
assert "CAND10K_DONE" in cc and "CAND10K_STOP" in cc and "cand_test" in cc
no_banned("chain", ch)
save("chain_cand10k.sh", ch)

# ---------------------------------------------------------------- the pod read and its waiter
PAIRS = ("I10RLcK:I10R0G0X,I10RQ50K:I10R0G0X,I10RLcKF5:I10RLcK,I10RQ50KF5:I10RQ50K,I10RLcKF5:I10RLc,I10RQ50KF5:I10RQ50,"
         "I10RLcK:I10RLc,I10RQ50K:I10RQ50,I10RLcKF5:I10RLcF5,I10RQ50KF5:I10RQ50F5,I10RQ50KF5:I10RLcKF5,I10RLcKF5:I10R0,I10RQ50KF5:I10R0")
pr = load("podread_rulefast.sh")
pr = rep(pr, "#!/usr/bin/env bash\n# podread_rulefast.sh -- pod-side read of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K). Written by make_rulefast10k.py from podread_causlim.sh (card ADD61) with the names, pairs and files of this card.\n",
         "#!/usr/bin/env bash\n# podread_cand.sh -- pod-side read of card PREREG_V8F_PROOF_ADD65 (CAND10K). Written by make_cand10k.py from podread_rulefast.sh (card ADD62) with the names, pairs and files of this card.\n")
pr = rep(pr, "only if e1 passes, the read of card ADD62.", "only if e1 passes, the read of card ADD65.")
pr = rep(pr, "ZZ PODREAD-RULEFAST DONE or ZZ PODREAD-RULEFAST STOP <why>.", "ZZ PODREAD-CAND DONE or ZZ PODREAD-CAND STOP <why>.")
pr = rep(pr, "$P/root/integ10k/read_rulefast10k.py and $P/root/integ10k/RULEFAST10K_REUSED.json (sha256 of the result files and rule logs of the\n# reused runs, recorded before the queue started); the tool-check output and the wrapper copy are written by the queue into $W/rulefast_test/, the compare file into $W/small_rulefast/.",
         "$P/root/integ10k/read_cand10k.py and $P/root/integ10k/CAND10K_REUSED.json (sha256 of the result files and rule logs of the\n# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/cand_test/, the two compare files into $W/small_cand/.")
pr = rep(pr, '[ -f "$PL/small_rulefast/CMP_RLcFID.txt" ] || stop "NO-CMP-FILE-K1"\n', '[ -f "$PL/small_rulefast/CMP_R0KID.txt" ] || stop "NO-CMP-FILE-C1"\n[ -f "$PL/small_rulefast/CMP_RLcKIDp.txt" ] || stop "NO-CMP-FILE-C2"\n')
pr = rep(pr, '[ -f "$W/rulefast_test/CHECK_RULE_EVERY.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"\n',
         '[ -f "$W/rulefast_test/CHECK_RULE_EVERY_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"\n[ -f "$W/rulefast_test/CHECK_ACC_GAIN_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"\n')
pr = rep(pr, 'PAIRS="I10RLcF5:I10RLc,I10RQ50F5:I10RQ50,I10RLcF2:I10RLc,I10RLcF2:I10RLcF5,I10RLcF5:I10R0,I10RQ50F5:I10R0,I10RLcF2:I10R0,I10RQ50F5:I10RLcF5"', 'PAIRS="%s"' % PAIRS)
pr = rep(pr, "# ---- the read of card ADD62 (pairs of the card: FR per form, the dose pairs, the pairs beside)", "# ---- the read of card ADD65 (pairs of the card: RK, FK and CC per form, the pairs beside)")
pr = rep(pr, '--wrapper "$W/build/v8f_t1_eval_reader_rule_rzf.py" --test-wrapper "$W/rulefast_test/v8f_t1_eval_reader_rule_rzf.py" --test-out "$W/rulefast_test/CHECK_RULE_EVERY.txt"',
         '--wrapper "$W/build/' + WR + '" --test-wrapper "$W/rulefast_test/' + WR + '" --test-out-rule "$W/rulefast_test/CHECK_RULE_EVERY_RZGF.txt" --test-out-gain "$W/rulefast_test/CHECK_ACC_GAIN_RZGF.txt"')
for old, new in (("rulefast_test", "cand_test"), ("small_rulefast", "small_cand"), ("read_rulefast10k", "read_cand10k"), ("READ_RULEFAST10K", "READ_CAND10K"), ("RULEFAST10K_REUSED", "CAND10K_REUSED"),
                 ("RULEFAST10K_DONE", "CAND10K_DONE"), ("PODREAD-RULEFAST", "PODREAD-CAND"), ("read_rulefast", "read_cand"), ("e1rf", "e1cd")):
    pr = rep_all_code(pr, old, new)
assert "rulefast" not in code_lines(pr).lower(), "podread names"
no_banned("podread", pr)
save("podread_cand.sh", pr)

wt = load("wait_podread_rulefast.sh")
wt = rep(wt, "#!/usr/bin/env bash\n# wait_podread_rulefast.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K), written by make_rulefast10k.py from wait_podread_causlim.sh (card ADD61), bash only (no Python on the dev box).\n",
         "#!/usr/bin/env bash\n# wait_podread_cand.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD65 (CAND10K), written by make_cand10k.py from wait_podread_rulefast.sh (card ADD62), bash only (no Python on the dev box).\n")
wt = rep(wt, 'echo "ZZ WAIT-PODREAD-RULEFAST DONE $(ts)"\n',
         'for f in CHECK_RULE_EVERY_RZGF.txt CHECK_ACC_GAIN_RZGF.txt; do\n'
         '  pod "cat $W/cand_test/$f" > "$I/read_cand/pod/$f"\n'
         '  a=$(pod "sha256sum $W/cand_test/$f | cut -c1-64"); b=$(sha256sum "$I/read_cand/pod/$f" | cut -c1-64)\n'
         '  [ "$a" = "$b" ] && echo "[$(ts)] pulled $f sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $f"\n'
         'done\n'
         'echo "ZZ WAIT-PODREAD-RULEFAST DONE $(ts)"\n')
for old, new in (("podread_rulefast", "podread_cand"), ("READ_RULEFAST10K", "READ_CAND10K"), ("read_rulefast", "read_cand"), ("RULEFAST10K", "CAND10K"), ("PODREAD-RULEFAST", "PODREAD-CAND"), ("e1rf", "e1cd")):
    wt = rep_all_code(wt, old, new)
assert "rulefast" not in code_lines(wt).lower(), "wait names"
no_banned("waiter", wt)
save("wait_podread_cand.sh", wt)
print("wrapper sha256 %s" % WSHA)
print("ZZ MAKE-CAND10K DONE")
