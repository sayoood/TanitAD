"""make_headbound10k.py -- writes the generated tools of card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K: a curvature bound on the heading of the kinematic executor, under the pinned readout,
under readout K and in the candidate of card ADD65; no training) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rzgf.py -> v8f_t1_eval_reader_rule_rzgfh.py (sha256 of the parent 969936595512ef80, asserted; one more parameter --kappa-max)
  check_rule_every_rzgf.py        -> check_rule_every_rzgfh.py        (the wrapper name only)
  check_acc_gain_rzgf.py          -> check_acc_gain_rzgfh.py          (the wrapper name only)
  pod_cand10k_queue.sh            -> pod_headbound10k_queue.sh
  chain_cand10k.sh                -> chain_headbound10k.sh
  podread_cand.sh                 -> podread_headbound.sh
  wait_podread_cand.sh            -> wait_podread_headbound.sh
The originals are not changed. Pure text generator (standard library only). Usage: python make_headbound10k.py (in this folder).
check_head_bound.py (tool check h4), read_headbound10k.py, HEADBOUND10K_DEFS.md and HEADBOUND10K_REUSED.json are written by hand, not by this script."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_headbound10k.py"
PARENT_SHA = "969936595512ef80"


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


# ---------------------------------------------------------------- the wrapper: one more parameter, --kappa-max
w = load("v8f_t1_eval_reader_rule_rzgf.py")
assert hashlib.sha256(w.encode("utf-8")).hexdigest().startswith(PARENT_SHA), "the parent wrapper is not the one of card ADD65"
OLD_HEAD = '"""v8f_t1_eval_reader_rule_rzgf.py -- written by make_cand10k.py from v8f_t1_eval_reader_rule_rzf.py for card PREREG_V8F_PROOF_ADD65 (CAND10K): the readout causal_g of card ADD63,\n'
w = rep(w, OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rzgfh.py -- written by make_headbound10k.py from v8f_t1_eval_reader_rule_rzgf.py for card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K): one more parameter,\n'
        '  --kappa-max K (1/m, finite, above 0; needs --readout, which may be pinned): after the chosen readout has returned (xy, psi, v) for the 2m frames of the executed steps of a commit,\n'
        '            the heading column is passed through the bound, frame by frame, starting from psi0 (the heading before the block): with prev the previous output heading (psi0 for the\n'
        '            first frame), xy_prev the previous frame position (knots[0], the current knot, for the first frame), d = wrap(psi[f] - prev) and lim = K x |xy[f] - xy_prev|:\n'
        '            if |d| <= lim the output heading is psi[f] itself, else prev + sign(d) x lim (wrapped into [-pi, pi)). Positions and speeds are not changed. The psi returned here is what\n'
        '            the pinned tool stores as the executed heading (track column 2) and passes back as psi0 at the next commit (v8f_t1_eval.py: commit_plan, pose[2]).\n'
        '            Without the parameter the wrapper is the wrapper of card ADD65 bit for bit. The readout log gets the fields kappa_max, frames_total, frames_bound_acted, max_excess_rad\n'
        '            (largest of |output step| - lim over all frames), max_abs_diff_where_not_acted, max_abs_diff_xy, max_abs_diff_v and the frame counts by the speed of the frame (below 2, 2 to 8,\n'
        '            above 8 m/s). Nothing else is new.\n'
        'Header of the parent:\n'
        'v8f_t1_eval_reader_rule_rzgf.py -- written by make_cand10k.py from v8f_t1_eval_reader_rule_rzf.py for card PREREG_V8F_PROOF_ADD65 (CAND10K): the readout causal_g of card ADD63,\n')

NEW_FN = '''import math  # noqa: E402

STATE.setdefault("kappa", None)
STATE.setdefault("kb", None)
KB_BINS = (("lt2", 0.0, 2.0), ("2to8", 2.0, 8.0), ("gt8", 8.0, float("inf")))


def wrap_pi(a):
    """angle into [-pi, pi)"""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def kb_new():
    """the counters of the heading bound (readout log fields)"""
    return {"frames_total": 0, "frames_bound_acted": 0, "max_excess_rad": None, "max_abs_diff_where_not_acted": 0.0, "max_abs_diff_xy": 0.0, "max_abs_diff_v": 0.0,
            "by_speed": {n: {"frames_total": 0, "frames_bound_acted": 0} for n, _, _ in KB_BINS}}


def bound_heading(xy, psi, v, xy0, psi0, kappa, rec=None):
    """the heading bound: frame by frame from psi0 and the position xy0 of the current knot; a heading step is at most kappa x the distance between the two frame positions.
    The output heading is psi[f] itself when its step from the previous output heading is within the bound, else the previous output heading turned by exactly the bound towards psi[f].
    Returns the new heading column; counters into rec (a dict of kb_new) when given."""
    xy = np.asarray(xy, np.float64)
    psi = np.asarray(psi, np.float64)
    vv = np.asarray(v, np.float64)
    out = np.empty(len(psi), np.float64)
    prev, pxy = float(psi0), np.asarray(xy0, np.float64)
    for f in range(len(psi)):
        lim = float(kappa) * float(np.hypot(xy[f, 0] - pxy[0], xy[f, 1] - pxy[1]))
        d = wrap_pi(float(psi[f]) - prev)
        acted = not (abs(d) <= lim)
        o = wrap_pi(prev + (1.0 if d > 0 else -1.0) * lim) if acted else float(psi[f])
        if rec is not None:
            ex = abs(wrap_pi(o - prev)) - lim
            rec["max_excess_rad"] = ex if rec["max_excess_rad"] is None else max(rec["max_excess_rad"], ex)
            rec["frames_total"] += 1
            rec["frames_bound_acted"] += int(acted)
            if not acted:
                rec["max_abs_diff_where_not_acted"] = max(rec["max_abs_diff_where_not_acted"], abs(o - float(psi[f])))
            for n, lo, hi in KB_BINS:
                if lo <= float(vv[f]) < hi:
                    rec["by_speed"][n]["frames_total"] += 1
                    rec["by_speed"][n]["frames_bound_acted"] += int(acted)
        out[f] = o
        prev, pxy = o, xy[f]
    return out
'''
w = rep(w, "\n\ndef counted_frames(orig, mode, gain=None, path_exec=False):", "\n\n" + NEW_FN + "\n\ndef counted_frames(orig, mode, gain=None, path_exec=False, kappa=None):")
w = rep(w, "        else:\n            xy, psi, v = orig(k_prev, knots, m, psi0)\n        K = np.vstack(",
        "        else:\n            xy, psi, v = orig(k_prev, knots, m, psi0)\n"
        "        if kappa is not None:                                         # the heading bound of card ADD67; positions and speeds pass through\n"
        "            xy_r, v_r = np.array(xy, copy=True), np.array(v, copy=True)\n"
        "            kb = STATE[\"kb\"]\n"
        "            psi = bound_heading(xy, psi, v, np.asarray(knots, np.float64)[0], psi0, kappa, kb)\n"
        "            kb[\"max_abs_diff_xy\"] = max(kb[\"max_abs_diff_xy\"], float(np.max(np.abs(np.asarray(xy) - xy_r))))\n"
        "            kb[\"max_abs_diff_v\"] = max(kb[\"max_abs_diff_v\"], float(np.max(np.abs(np.asarray(v) - v_r))))\n"
        "        K = np.vstack(")
w = rep(w, '"acc_gain": STATE["gain"], "path_exec": bool(STATE["path_exec"]), "max_abs_acceleration_minus_gain_executed": h.get("max_dga")}',
        '"acc_gain": STATE["gain"], "path_exec": bool(STATE["path_exec"]), "max_abs_acceleration_minus_gain_executed": h.get("max_dga"),\n'
        '           "kappa_max": STATE["kappa"]}\n'
        '    if STATE["kb"] is not None:\n'
        '        rec.update(frames_total=STATE["kb"]["frames_total"], frames_bound_acted=STATE["kb"]["frames_bound_acted"], max_excess_rad=STATE["kb"]["max_excess_rad"],\n'
        '                   max_abs_diff_where_not_acted=STATE["kb"]["max_abs_diff_where_not_acted"], max_abs_diff_xy=STATE["kb"]["max_abs_diff_xy"], max_abs_diff_v=STATE["kb"]["max_abs_diff_v"],\n'
        '                   by_speed_bin=STATE["kb"]["by_speed"])')
w = rep(w, '    ap.add_argument("--path-exec", action="store_true")\n', '    ap.add_argument("--path-exec", action="store_true")\n    ap.add_argument("--kappa-max", type=float, default=None)\n')
w = rep(w, "    mine, rest = ap.parse_known_args(argv)\n",
        "    mine, rest = ap.parse_known_args(argv)\n"
        "    if mine.kappa_max is not None and (not math.isfinite(mine.kappa_max) or not mine.kappa_max > 0):\n"
        "        raise SystemExit(\"REFUSED: --kappa-max must be finite and above 0\")\n"
        "    if mine.kappa_max is not None and mine.readout is None:\n"
        "        raise SystemExit(\"REFUSED: --kappa-max goes with --readout (pinned is allowed): the readout log carries the fields of the bound\")\n")
w = rep(w, '        T.hermite_frames = counted_frames(orig_hf, mine.readout, STATE["gain"], STATE["path_exec"])\n',
        '        STATE["kappa"] = None if mine.kappa_max is None else float(mine.kappa_max)\n'
        '        STATE["kb"] = None if mine.kappa_max is None else kb_new()\n'
        '        T.hermite_frames = counted_frames(orig_hf, mine.readout, STATE["gain"], STATE["path_exec"], STATE["kappa"])\n')
compile(w, "v8f_t1_eval_reader_rule_rzgfh.py", "exec")
no_banned("rzgfh", w)
save("v8f_t1_eval_reader_rule_rzgfh.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the two parent tool checks: copies with the wrapper name changed and nothing else
cr = load("check_rule_every_rzgf.py")
assert cr.count("rzgf") == 6 and "rzgfh" not in cr, cr.count("rzgf")
cr = cr.replace("rzgf", "rzgfh")
cr = rep(cr, '"""check_rule_every_rzgfh.py (copy of check_rule_every.py, written by make_cand10k.py; the wrapper name is the only change) -- ',
         '"""check_rule_every_rzgfh.py (copy of check_rule_every_rzgf.py, written by make_headbound10k.py; the wrapper name is the only change) -- ')
compile(cr, "check_rule_every_rzgfh.py", "exec")
no_banned("check_rule_every_rzgfh", cr)
save("check_rule_every_rzgfh.py", cr)
ca = load("check_acc_gain_rzgf.py")
assert ca.count("rzgf") == 4 and "rzgfh" not in ca, ca.count("rzgf")
ca = ca.replace("rzgf", "rzgfh")
ca = rep(ca, '"""check_acc_gain_rzgfh.py (copy of check_acc_gain.py, written by make_cand10k.py; the wrapper name is the only change) -- ',
         '"""check_acc_gain_rzgfh.py (copy of check_acc_gain_rzgf.py, written by make_headbound10k.py; the wrapper name is the only change) -- ')
compile(ca, "check_acc_gain_rzgfh.py", "exec")
no_banned("check_acc_gain_rzgfh", ca)
save("check_acc_gain_rzgfh.py", ca)

# ---------------------------------------------------------------- the pod queue
WR = "v8f_t1_eval_reader_rule_rzgfh.py"
q = load("pod_cand10k_queue.sh")
OLDQ_HEAD = q[:q.index("# pod_rulefast10k_queue.sh -- written by make_rulefast10k.py")]
assert OLDQ_HEAD.startswith("#!/usr/bin/env bash\n# pod_cand10k_queue.sh -- written by make_cand10k.py")
q = q.replace(OLDQ_HEAD,
              "#!/usr/bin/env bash\n# pod_headbound10k_queue.sh -- written by make_headbound10k.py from pod_cand10k_queue.sh. Card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K): the three tool checks of control h4 first\n"
              "# (check_head_bound.py, and check_rule_every_rzgfh.py and check_acc_gain_rzgfh.py, on the pod, no model; step marker hb6check), then the control I10R0HbID (no rule, --readout pinned,\n"
              "# --kappa-max 1e9, model seed 0, compared file for file with the banked dump of I10R0_s0: marker hb1cmp), then the control I10R0KHbID (no rule, readout K = --readout causal_g --acc-gain 0\n"
              "# --path-exec, --kappa-max 1e9, model seed 0, compared with the banked dump of I10R0G0X_s0: marker hb2cmp), then at two model seeds the five rollouts of the card table with the bound:\n"
              "# I10R0Hb20 (no rule, pinned, 0.2), I10R0KHb20 (no rule, K, 0.2), I10R0KHb10 (no rule, K, 0.1), I10RLcKF5Hb20 (label-fed, every 5 steps, K, 0.2), I10RQ50KF5Hb20 (reader-fed, every 5 steps,\n"
              "# K, 0.2), reader seed = model seed, all at the 2.0 s replan interval, with the wrapper " + WR + ". No training. The queue waits for the end marker of the queue of card ADD65\n"
              "# (variables WAIT_DONE / WAIT_STOP, defaults CAND10K_DONE.txt / CAND10K_STOP.txt in the folder $W; WAIT_MAX seconds, default 86400); after it, it takes the sha256 of the result files, rule logs\n"
              "# and tick files of the two runs of card ADD65 that this card reuses (I10RLcKF5, I10RQ50KF5) into $W/small_hb/REUSED_CAND_TAKEN.txt (control h7). Markers START_HEADBOUND10K.txt /\n"
              "# HEADBOUND10K_STOP.txt / HEADBOUND10K_DONE.txt; step markers hb6check, hb1cmp, hb2cmp. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "CAND10K", "HEADBOUND10K", 3)
q = rep(q, 'card ADD65: two tool checks, then two controls and eight rollouts with the lead rule under readout K, at the tick and every 5 steps; checkpoints of card ADD46"',
        'card ADD67: three tool checks, then two controls and ten rollouts with the heading bound; checkpoints of card ADD46"')
q = rep(q, 'WAIT_DONE=${WAIT_DONE:-ACCGAIN10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-ACCGAIN10K_STOP.txt}', 'WAIT_DONE=${WAIT_DONE:-CAND10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-CAND10K_STOP.txt}')
q = rep(q, "the tool checks and the rollouts of card ADD65 start", "the tool checks and the rollouts of card ADD67 start")
q = rep(q, '  [ -f "$W/t1/I10R0G0X_s0/T1_DONE" ] || stop "NO-BANKED-R0G0X"\n  [ -f "$W/t1/I10RLcF5_s0/T1_DONE" ] || stop "NO-BANKED-RLCF5"\n',
        '  [ -f "$W/t1/I10R0_s0/T1_DONE" ] || stop "NO-BANKED-R0"\n  [ -f "$W/t1/I10R0G0X_s0/T1_DONE" ] || stop "NO-BANKED-R0G0X"\n')
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzgf.py v8f_t1_eval_reader_rule_rz.py check_rule_every_rzgf.py check_acc_gain_rzgf.py ',
        'for f in cmp_dumps.py ' + WR + ' v8f_t1_eval_reader_rule_rz.py check_head_bound.py check_rule_every_rzgfh.py check_acc_gain_rzgfh.py ')
q = rep(q, "for nm in I10R0 I10R0G0X I10RLc I10RQ50 I10RLcF5 I10RQ50F5; do", "for nm in I10R0 I10R0G0X I10RLc I10RQ50; do")
q = rep(q, '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n',
        '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n  [ -n "${KAPPA:-}" ] && extra+=(--kappa-max "$KAPPA")\n')
q = rep(q, 'readout ${READOUT:-pinned} model seed $m', 'readout ${READOUT:-pinned} kappa ${KAPPA:-none} model seed $m')
q = rep(q, "# ---------------------------------------------------------------- 5 the twelve rollouts", "# ---------------------------------------------------------------- 5 the twelve rollouts of the card (two controls and ten)")
HB = ("$W/hb_test/CHECK_HEAD_BOUND.txt", "$W/hb_test/CHECK_RULE_EVERY_RZGFH.txt", "$W/hb_test/CHECK_ACC_GAIN_RZGFH.txt")
NEWBLOCK = (
    '# ---------------------------------------------------------------- the reused runs of card ADD65: their hashes are taken after that queue has ended (control h7)\n'
    'if [ "$DRY" = 1 ]; then\n'
    '  echo "CMD: sha256sum the result files, rule logs, tick files of I10RLcKF5 and I10RQ50KF5 (both model seeds), if present, into $W/small_hb/REUSED_CAND_TAKEN.txt"\n'
    'elif [ ! -f "$W/small_hb/REUSED_CAND_TAKEN.txt" ]; then\n'
    '  mkdir -p "$W/small_hb"\n'
    '  ( cd "$W" && for nm in I10RLcKF5 I10RQ50KF5; do for k in 0 1; do d="t1/${nm}_s$k"; [ -f "$d/T1_DONE" ] || { echo "ABSENT $d"; continue; }; sha256sum "$d/t1v8f_${nm}_s${k}_m10_causal_heads.json" "$d"/ticks/rule_log*.npz "$d"/ticks/ticks_${nm}_s${k}_m10_causal_heads.npz "$d"/ticks/readout_log_${nm}_s${k}.json; done; done ) > "$W/small_hb/REUSED_CAND_TAKEN.txt.part" 2>&1 && mv "$W/small_hb/REUSED_CAND_TAKEN.txt.part" "$W/small_hb/REUSED_CAND_TAKEN.txt"\n'
    '  log "hashes of the reused runs of card ADD65 taken: $(grep -c . "$W/small_hb/REUSED_CAND_TAKEN.txt") lines"\n'
    'fi\n'
    '# ---------------------------------------------------------------- control h4 first: the three tool checks on the pod (no model, no rollout, no GPU, two threads); their output files and the copy of the wrapper they ran with are read by the reader\n'
    'if ! done_step hb6check; then\n'
    '  if [ "$DRY" = 1 ]; then\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_head_bound.py $TREE > $W/hb_test/CHECK_HEAD_BOUND.txt"\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_rule_every_rzgfh.py $TREE > $W/hb_test/CHECK_RULE_EVERY_RZGFH.txt"\n'
    '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_acc_gain_rzgfh.py $TREE > $W/hb_test/CHECK_ACC_GAIN_RZGFH.txt"\n'
    '  else\n'
    '    mkdir -p "$W/hb_test"\n'
    '    cp -f "$B/' + WR + '" "$W/hb_test/' + WR + '"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_head_bound.py" "$TREE" > "$W/hb_test/CHECK_HEAD_BOUND.txt" 2>&1\n'
    '    tail -n 1 "$W/hb_test/CHECK_HEAD_BOUND.txt" | grep -q "^ZZ CHECK-HEAD-BOUND all hold: True" || stop "TOOL-CHECK-HEAD-BOUND-FAILED"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 1800 "$PY" "$B/check_rule_every_rzgfh.py" "$TREE" > "$W/hb_test/CHECK_RULE_EVERY_RZGFH.txt" 2>&1\n'
    '    tail -n 1 "$W/hb_test/CHECK_RULE_EVERY_RZGFH.txt" | grep -q "^ZZ CHECK-RULE-EVERY all hold: True" || stop "TOOL-CHECK-RULE-EVERY-RZGFH-FAILED"\n'
    '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_acc_gain_rzgfh.py" "$TREE" > "$W/hb_test/CHECK_ACC_GAIN_RZGFH.txt" 2>&1\n'
    '    tail -n 1 "$W/hb_test/CHECK_ACC_GAIN_RZGFH.txt" | grep -q "^ZZ CHECK-ACC-GAIN all hold: True" || stop "TOOL-CHECK-ACC-GAIN-RZGFH-FAILED"\n'
    '    mark hb6check\n'
    '  fi\n'
    'fi\n'
    'READOUT=pinned KAPPA=1e9 rollout I10R0HbID R0 0 0       # control h1: the bound with a huge K, pinned readout, no rule, must give the banked dump of I10R0 at model seed 0\n'
    'if [ "$DRY" != 1 ] && ! done_step hb1cmp; then\n'
    '  mkdir -p "$W/small_hb"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0_s0/dump_I10R0_s0_m10_causal_heads" "$W/t1/I10R0HbID_s0/dump_I10R0HbID_s0_m10_causal_heads" > "$W/small_hb/CMP_R0HbID.txt" 2>&1 || stop "H1-RERUN-DIFFERS"\n'
    '  mark hb1cmp\n'
    'fi\n'
    'READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=1e9 rollout I10R0KHbID R0 0 0       # control h2: the bound with a huge K under readout K, no rule, must give the banked dump of I10R0G0X at model seed 0\n'
    'if [ "$DRY" != 1 ] && ! done_step hb2cmp; then\n'
    '  mkdir -p "$W/small_hb"\n'
    '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0G0X_s0/dump_I10R0G0X_s0_m10_causal_heads" "$W/t1/I10R0KHbID_s0/dump_I10R0KHbID_s0_m10_causal_heads" > "$W/small_hb/CMP_R0KHbID.txt" 2>&1 || stop "H2-RERUN-DIFFERS"\n'
    '  mark hb2cmp\n'
    'fi\n'
    'for k in "${ARMS[@]}"; do\n'
    '  READOUT=pinned KAPPA=0.2 rollout I10R0Hb20 R0 "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 rollout I10R0KHb20 R0 "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.1 rollout I10R0KHb10 R0 "$k" "$k"       # dose, names nothing\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 rollout I10RLcKF5Hb20 RLc "$k" "$k"\n'
    '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50KF5Hb20 RR "$k" "$k"\n'
    'done\n')
q = span(q, '# ---------------------------------------------------------------- control c6 first: the two tool checks on the pod', '  READOUT=causal_g ACCGAIN=0 PATHEXEC=1 EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50KF5 RR "$k" "$k"\ndone\n', NEWBLOCK)
q = rep_code(q, "v8f_t1_eval_reader_rule_rzgf.py", WR, 3)
q = rep(q, 'echo "$(ts) all steps done" > "$W/HEADBOUND10K_DONE.txt"\nlog "INTEG10K_DONE"', 'echo "$(ts) all steps done" > "$W/HEADBOUND10K_DONE.txt"\nlog "HEADBOUND10K_DONE"')
c = code_lines(q)
for bad in ("CAND10K_STOP.txt\"", "RLcKIDp", "ck1cmp", "ck2cmp", "ck6check", "I10R0KID ", "rzgf.py", "cand_test", "small_cand", "CHECK_ACC_GAIN_RZGF.txt", "REUSED_CAND10K"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 7 and c.count("HEADBOUND10K") >= 3 and "hb1cmp" in c and "hb2cmp" in c and "hb6check" in c and "wrapper sha256" in c, "queue sanity"
assert "CAND10K_DONE.txt" in c and "CAND10K_STOP.txt" in c and "MINGB=${I10_MIN_FREE_GB:-28}" in q and "MAXGPU=3" in q
no_banned("queue", q)
save("pod_headbound10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
ch = load("chain_cand10k.sh")
ch = rep(ch, "#!/usr/bin/env bash\n# chain_cand10k.sh -- written by make_cand10k.py from chain_rulefast10k.sh for card PREREG_V8F_PROOF_ADD65 (CAND10K). The poll runs 72 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD63, up to 24 h). The header lines below are those of the parent.\n",
         "#!/usr/bin/env bash\n# chain_headbound10k.sh -- written by make_headbound10k.py from chain_cand10k.sh for card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K). The poll runs 72 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD65, up to 24 h). The header lines below are those of the parent.\n")
ch = rep(ch, '"$I/pod_cand10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzgf.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/check_rule_every_rzgf.py" "$I/check_acc_gain_rzgf.py"',
         '"$I/pod_headbound10k_queue.sh" "$I/' + WR + '" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/check_head_bound.py" "$I/check_rule_every_rzgfh.py" "$I/check_acc_gain_rzgfh.py"')
ch = rep(ch, "for nm in I10R0KID I10RLcKIDp I10RLcK I10RQ50K I10RLcKF5 I10RQ50KF5; do", "for nm in I10R0HbID I10R0KHbID I10R0Hb20 I10R0KHb20 I10R0KHb10 I10RLcKF5Hb20 I10RQ50KF5Hb20; do")
ch = rep(ch, "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt $W/cand_test/CHECK_ACC_GAIN_RZGF.txt", "$W/hb_test/CHECK_HEAD_BOUND.txt $W/hb_test/CHECK_RULE_EVERY_RZGFH.txt $W/hb_test/CHECK_ACC_GAIN_RZGFH.txt")
for old, new in (("EXITS on CAND10K_DONE, CAND10K_STOP or after 72 h", "EXITS on HEADBOUND10K_DONE, HEADBOUND10K_STOP or after 72 h"),
                 ("pod_cand10k_queue.sh", "pod_headbound10k_queue.sh"), ("small_cand", "small_hb"), ("chain_cand10k.log", "chain_headbound10k.log"), ("queue_cand_stdout", "queue_headbound_stdout"),
                 ("START_CAND10K", "START_HEADBOUND10K"), ("CHAIN-CAND10K", "CHAIN-HEADBOUND10K"), ("CAND10K_DONE", "HEADBOUND10K_DONE"), ("CAND10K_STOP", "HEADBOUND10K_STOP")):
    assert ch.count(old) >= 1, old
    ch = ch.replace(old, new)
cc = code_lines(ch)
for bad in ("CAND10K", "cand_test", "RLcKIDp", "I10R0KID", "small_cand", "rzgf.py"):
    assert bad not in cc, bad
assert "HEADBOUND10K_DONE" in cc and "HEADBOUND10K_STOP" in cc and "hb_test" in cc
no_banned("chain", ch)
save("chain_headbound10k.sh", ch)

# ---------------------------------------------------------------- the pod read and its waiter
PAIRS_ALWAYS = ("I10R0Hb20:I10R0,I10R0KHb20:I10R0G0X,I10R0KHb10:I10R0KHb20,I10R0KHb20:I10R0,I10RLcKF5Hb20:I10RLc,I10RQ50KF5Hb20:I10RQ50")
PAIRS_HC = ("I10RLcKF5Hb20:I10RLcKF5,I10RQ50KF5Hb20:I10RQ50KF5")
pr = load("podread_cand.sh")
pr = rep(pr, "#!/usr/bin/env bash\n# podread_cand.sh -- pod-side read of card PREREG_V8F_PROOF_ADD65 (CAND10K). Written by make_cand10k.py from podread_rulefast.sh (card ADD62) with the names, pairs and files of this card.\n",
         "#!/usr/bin/env bash\n# podread_headbound.sh -- pod-side read of card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K). Written by make_headbound10k.py from podread_cand.sh (card ADD65) with the names, pairs and files of this card.\n")
pr = rep(pr, "only if e1 passes, the read of card ADD65.", "only if e1 passes, the read of card ADD67.")
pr = rep(pr, "ZZ PODREAD-CAND DONE or ZZ PODREAD-CAND STOP <why>.", "ZZ PODREAD-HEADBOUND DONE or ZZ PODREAD-HEADBOUND STOP <why>.")
pr = rep(pr, "$P/root/integ10k/read_cand10k.py and $P/root/integ10k/CAND10K_REUSED.json (sha256 of the result files and rule logs of the\n# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/cand_test/, the two compare files into $W/small_cand/.",
         "$P/root/integ10k/read_headbound10k.py and $P/root/integ10k/HEADBOUND10K_REUSED.json (sha256 of the result files and rule logs of the\n# reused runs, recorded before the queue started; the two runs of card ADD65 are hashed by the queue into $W/small_hb/REUSED_CAND_TAKEN.txt); the three tool-check outputs and the wrapper copy are written by the queue into $W/hb_test/, the two compare files into $W/small_hb/. The two pairs of the form HC are read only if the read of card ADD65 exists with controls_all_pass true.")
pr = rep(pr, '[ -f "$PL/small_cand/CMP_R0KID.txt" ] || stop "NO-CMP-FILE-C1"\n[ -f "$PL/small_cand/CMP_RLcKIDp.txt" ] || stop "NO-CMP-FILE-C2"\n',
         '[ -f "$PL/small_cand/CMP_R0HbID.txt" ] || stop "NO-CMP-FILE-H1"\n[ -f "$PL/small_cand/CMP_R0KHbID.txt" ] || stop "NO-CMP-FILE-H2"\n[ -f "$PL/small_cand/REUSED_CAND_TAKEN.txt" ] || stop "NO-CAND-HASH-FILE"\n')
pr = rep(pr, '[ -f "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"\n[ -f "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"\n',
         '[ -f "$W/cand_test/CHECK_HEAD_BOUND.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-HEAD"\n[ -f "$W/cand_test/CHECK_RULE_EVERY_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-RULE"\n[ -f "$W/cand_test/CHECK_ACC_GAIN_RZGFH.txt" ] || stop "NO-TOOL-CHECK-OUTPUT-GAIN"\n')
pr = rep(pr, 'PAIRS="I10RLcK:I10R0G0X,I10RQ50K:I10R0G0X,I10RLcKF5:I10RLcK,I10RQ50KF5:I10RQ50K,I10RLcKF5:I10RLc,I10RQ50KF5:I10RQ50,I10RLcK:I10RLc,I10RQ50K:I10RQ50,I10RLcKF5:I10RLcF5,I10RQ50KF5:I10RQ50F5,I10RQ50KF5:I10RLcKF5,I10RLcKF5:I10R0,I10RQ50KF5:I10R0"',
         'PAIRS="' + PAIRS_ALWAYS + '"\nCANDJSON=@@CJ@@\nif [ -f "$CANDJSON" ] && grep -q \'"controls_all_pass": true\' "$CANDJSON"; then PAIRS="$PAIRS,' + PAIRS_HC + '"; echo "[$(ts)] card ADD65 read present with controls_all_pass true: the pairs HC are read"; else echo "[$(ts)] card ADD65 read absent or its controls did not pass: the pairs HC are not read"; fi')
pr = rep(pr, "# ---- the read of card ADD65 (pairs of the card: RK, FK and CC per form, the pairs beside)", "# ---- the read of card ADD67 (pairs of the card: HP, HK and HC per form, the pairs beside)")
pr = rep(pr, '--wrapper "$W/build/v8f_t1_eval_reader_rule_rzgf.py" --test-wrapper "$W/cand_test/v8f_t1_eval_reader_rule_rzgf.py" --test-out-rule "$W/cand_test/CHECK_RULE_EVERY_RZGF.txt" --test-out-gain "$W/cand_test/CHECK_ACC_GAIN_RZGF.txt"',
         '--wrapper "$W/build/' + WR + '" --test-wrapper "$W/cand_test/' + WR + '" --test-out-head "$W/cand_test/CHECK_HEAD_BOUND.txt" --test-out-rule "$W/cand_test/CHECK_RULE_EVERY_RZGFH.txt" --test-out-gain "$W/cand_test/CHECK_ACC_GAIN_RZGFH.txt" --taken "$PL/small_cand/REUSED_CAND_TAKEN.txt" --cand-json "$CANDJSON" --pairs-read "$PAIRS" --w "$W" --tree "$W/tree"')
for old, new in (("cand_test", "hb_test"), ("small_cand", "small_hb"), ("read_cand10k", "read_headbound10k"), ("READ_CAND10K.json\"", "READ_HEADBOUND10K.json\""), ("CAND10K_REUSED", "HEADBOUND10K_REUSED"),
                 ("CAND10K_DONE", "HEADBOUND10K_DONE"), ("PODREAD-CAND", "PODREAD-HEADBOUND"), ("read_cand", "read_headbound"), ("e1cd", "e1hb")):
    pr = rep_all_code(pr, old, new)
pr = pr.replace("@@CJ@@", "$P/out/read_cand/READ_CAND10K.json")     # the read of card ADD65 keeps its own place
c = code_lines(pr)
assert "READ_CAND10K.json" in c and "--cand-json" in c and "cand_test" not in c and "small_cand" not in c and "I10RLcK:" not in c and "CAND10K_DONE" not in c, "podread names"
no_banned("podread", pr)
save("podread_headbound.sh", pr)

wt = load("wait_podread_cand.sh")
wt = rep(wt, "#!/usr/bin/env bash\n# wait_podread_cand.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD65 (CAND10K), written by make_cand10k.py from wait_podread_rulefast.sh (card ADD62), bash only (no Python on the dev box).\n"
         "# Waits for the pod marker CAND10K_DONE.txt (or _STOP), then runs the read on the pod (podread_cand.sh, pushed before) and brings the small result files back as text\n"
         "# into read_cand/pod/. Last line: ZZ WAIT-PODREAD-CAND DONE or STOP. (These two comment lines were set by hand at the review; the generator carried the parent's names in them.)\n",
         "#!/usr/bin/env bash\n# wait_podread_headbound.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K), written by make_headbound10k.py from wait_podread_cand.sh (card ADD65), bash only (no Python on the dev box).\n"
         "# Waits for the pod marker HEADBOUND10K_DONE.txt (or _STOP), then runs the read on the pod (podread_headbound.sh, pushed before) and brings the small result files back as text\n"
         "# into read_headbound/pod/. Last line: ZZ WAIT-PODREAD-HEADBOUND DONE or STOP.\n")
wt = rep(wt, "for f in CHECK_RULE_EVERY_RZGF.txt CHECK_ACC_GAIN_RZGF.txt; do", "for f in CHECK_HEAD_BOUND.txt CHECK_RULE_EVERY_RZGFH.txt CHECK_ACC_GAIN_RZGFH.txt; do")
wt = rep(wt, "PODT=2400 pod", "PODT=3600 pod")
for old, new in (("cand_test", "hb_test"), ("podread_cand", "podread_headbound"), ("READ_CAND10K", "READ_HEADBOUND10K"), ("read_cand", "read_headbound"), ("CAND10K", "HEADBOUND10K"), ("PODREAD-CAND", "PODREAD-HEADBOUND"), ("e1cd", "e1hb")):
    wt = rep_all_code(wt, old, new)
assert "ZZ WAIT-PODREAD-CAND" not in wt
c = code_lines(wt).lower()
assert "cand" not in c, "wait names"
no_banned("waiter", wt)
save("wait_podread_headbound.sh", wt)
print("wrapper sha256 %s" % WSHA)
print("ZZ MAKE-HEADBOUND10K DONE")
