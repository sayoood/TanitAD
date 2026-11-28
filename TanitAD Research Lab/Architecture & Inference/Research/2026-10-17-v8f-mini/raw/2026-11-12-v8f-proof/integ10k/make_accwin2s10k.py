"""make_accwin2s10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K: the acceleration the model reads at a replan, taken over the last 2.0 s of executed
motion across replans) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rz.py -> v8f_t1_eval_reader_rule_rv.py  (one more readout: causal_v)
  pod_accwin10k_queue.sh        -> pod_accwin2s10k_queue.sh
  chain_accwin10k.sh            -> chain_accwin2s10k.sh
The originals are not changed. Pure text generator (no numpy, no torch). Usage: python make_accwin2s10k.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_accwin2s10k.py"


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
    print("wrote %s (%d bytes)" % (name, len(text.encode("utf-8"))))


def rep(text, old, new, n=1, at_least=False):
    c = text.count(old)
    assert (c >= n) if at_least else (c == n), "edit anchor %r occurs %d times, expected %s%d" % (old[:70], c, ">= " if at_least else "", n)
    return text.replace(old, new)


def rep_code(text, old, new, n, at_least=False):
    """replaces in the lines that are not comments; n = the number of lines that must change"""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert (k >= n) if at_least else (k == n), "code anchor %r changed %d lines, expected %d" % (old, k, n)
    return "\n".join(out)


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


def code_lines(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_rz.py")
OLD_HEAD = 'v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.\n'
w = rep(w, '"""' + OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rv.py -- written by make_accwin2s10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K): one more readout, causal_v.\n'
        'It keeps the positions and headings of the pinned frames function and the causal speed at every executed knot (the step speed into it). The half-step speed, which sets\n'
        'the one-frame acceleration the model reads at executed step i of a block, is\n'
        '  causal_v  acceleration = (s[i] - s[i-w]) / (0.2 s x w), w = min(10, the number of earlier steps in the rollout\'s own track), at least 1; s runs over the steps of the\n'
        '            track: the executed steps of the current block, before them the executed steps of the earlier blocks, before the first block the logged steps the window\n'
        '            started from (so the reach is 2.0 s and crosses replans). The step speed is the distance between consecutive track knots over 0.2 s.\n'
        'The block does not see the earlier knots: while WindowState.commit_plan runs, STATE["exec"] points at the window state and window2_steps reads the knots 0 .. j-1 of\n'
        'its track (the last of them is k_prev). commit_plan is replaced for this readout only and restored where the frames function is restored.\n'
        'pinned, backward, causal, causal_w, causal_z are unchanged (bit for bit). Header of the parent:\n' + OLD_HEAD)
w = rep(w, '"readout": "pinned", "hf": {"calls": 0,', '"readout": "pinned", "exec": None, "w2log": None, "hf": {"calls": 0,')
w = rep(w, 'def causal_frames(orig, k_prev, knots, m, psi0, acc="step"):\n',
        'W2_LEN = 10                                        # steps (2.0 s): the longest reach of the readout causal_v\n'
        '\n'
        '\n'
        'def commit_plan_with_state(orig_cp):\n'
        '    """WindowState.commit_plan with STATE["exec"] set to the window state while it runs (the frames function then reads the rollout\'s own earlier knots)."""\n'
        '    def commit_plan(self, plan_ego, m):\n'
        '        STATE["exec"] = self\n'
        '        try:\n'
        '            return orig_cp(self, plan_ego, m)\n'
        '        finally:\n'
        '            STATE["exec"] = None\n'
        '    return commit_plan\n'
        '\n'
        '\n'
        'def window2_steps(k_prev, knots, m):\n'
        '    """-> (s_ext, n_pre, e_blk). s_ext = step speeds over the knots of the rollout\'s own track before the block (at most the last 11 finite knots j-11 .. j-1, the last of them\n'
        '    k_prev) followed by the knots of the block knots[0 .. m]; s_ext[n_pre + i] is the step speed into executed knot i of the block (i = 0: the step into the current knot),\n'
        '    n_pre = the number of track steps before s[0]; e_blk = the index of the last executed knot counted from the window start (t0)."""\n'
        '    ex = STATE["exec"]\n'
        '    if ex is None:\n'
        '        raise RuntimeError("readout causal_v: no window state is set (commit_plan was not wrapped)")\n'
        '    j = int(ex.j)\n'
        '    tr = np.asarray(ex.track, np.float64)[0:2 * j:2, :2]                       # knots 0 .. j-1 of the own track (rows 2q)\n'
        '    fin = np.isfinite(tr).all(axis=1)\n'
        '    n_ok = 0\n'
        '    while n_ok < len(fin) and fin[len(fin) - 1 - n_ok]:\n'
        '        n_ok += 1\n'
        '    hist = tr[len(tr) - min(n_ok, W2_LEN + 1):]\n'
        '    if len(hist) == 0 or not np.array_equal(hist[-1], np.asarray(k_prev, np.float64)):\n'
        '        raise RuntimeError("readout causal_v: the last earlier knot of the track is not k_prev")\n'
        '    ext = np.vstack([hist, np.asarray(knots, np.float64)[:m + 1]])\n'
        '    return np.linalg.norm(np.diff(ext, axis=0), axis=-1) / DT_S, len(hist) - 1, j - int(ex.t0) + m\n'
        '\n'
        '\n'
        'def causal_frames(orig, k_prev, knots, m, psi0, acc="step"):\n')
w = rep(w, '    v = np.asarray(v, np.float64).copy()\n    for i in range(1, m + 1):\n        if acc == "zero":\n',
        '    v = np.asarray(v, np.float64).copy()\n'
        '    if acc == "window2":\n'
        '        s_ext, n_pre, e_blk = window2_steps(k_prev, knots, m)\n'
        '    for i in range(1, m + 1):\n'
        '        if acc == "zero":\n')
w = rep(w, '        else:\n            v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])\n',
        '        elif acc == "window2":\n'
        '            n_w = max(1, min(W2_LEN, i + n_pre))\n'
        '            v[2 * (i - 1)] = s[i] - 0.5 * DT_S * (s_ext[n_pre + i] - s_ext[n_pre + i - n_w]) / (DT_S * n_w)\n'
        '        else:\n            v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])\n')
w = rep(w, '        v[2 * (i - 1) + 1] = s[i]\n    return xy, psi, v\n',
        '        v[2 * (i - 1) + 1] = s[i]\n'
        '    if acc == "window2":\n'
        '        n_w = max(1, min(W2_LEN, m + n_pre))\n'
        '        STATE["w2log"] = (n_w, (s_ext[n_pre + m] - s_ext[n_pre + m - n_w]) / (DT_S * n_w), e_blk)\n'
        '    return xy, psi, v\n')
w = rep(w, '        if mode in ("causal", "causal_w", "causal_z"):\n            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0, {"causal": "step", "causal_w": "window", "causal_z": "zero"}[mode])\n',
        '        if mode in ("causal", "causal_w", "causal_z", "causal_v"):\n'
        '            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0, {"causal": "step", "causal_w": "window", "causal_z": "zero", "causal_v": "window2"}[mode])\n')
w = rep(w, '                h["max_a0"], h["max_daw"], h["sum_a0"] = max(h.get("max_a0", 0.0), abs(a0r)), max(h.get("max_daw", 0.0), daw), h.get("sum_a0", 0.0) + abs(a0r)\n',
        '                h["max_a0"], h["max_daw"], h["sum_a0"] = max(h.get("max_a0", 0.0), abs(a0r)), max(h.get("max_daw", 0.0), daw), h.get("sum_a0", 0.0) + abs(a0r)\n'
        '                if mode == "causal_v":                      # window length at the last executed knot, the tick value against the 2.0 s window value, the ticks whose whole reach lies in the dump\n'
        '                    n_w2, ref2, e_blk = STATE["w2log"]\n'
        '                    h["w2_calls"] = h.get("w2_calls", 0) + 1\n'
        '                    h["w2_sum_len"], h["w2_min_len"] = h.get("w2_sum_len", 0) + n_w2, min(h.get("w2_min_len", n_w2), n_w2)\n'
        '                    h["w2_n_short"], h["w2_max_diff"] = h.get("w2_n_short", 0) + int(n_w2 < W2_LEN), max(h.get("w2_max_diff", 0.0), abs(a0r - ref2))\n'
        '                    if e_blk - W2_LEN >= 1:\n'
        '                        h["w2_dump_calls"], h["w2_dump_sum_abs_a0"] = h.get("w2_dump_calls", 0) + 1, h.get("w2_dump_sum_abs_a0", 0.0) + abs(a0r)\n')
w = rep(w, '"max_abs_acceleration_minus_window": h.get("max_daw", 0.0)}\n',
        '"max_abs_acceleration_minus_window": h.get("max_daw", 0.0),\n'
        '           "window2_len_cap": W2_LEN, "window2_calls": int(h.get("w2_calls", 0)), "window2_mean_len": h.get("w2_sum_len", 0) / max(h.get("w2_calls", 0), 1),\n'
        '           "window2_sum_len": int(h.get("w2_sum_len", 0)), "window2_min_len": int(h.get("w2_min_len", 0)), "window2_calls_below_cap": int(h.get("w2_n_short", 0)),\n'
        '           "window2_max_abs_acceleration_minus_window2": h.get("w2_max_diff", 0.0), "window2_dump_calls": int(h.get("w2_dump_calls", 0)),\n'
        '           "window2_dump_sum_abs_acceleration_read": h.get("w2_dump_sum_abs_a0", 0.0)}\n')
w = rep(w, 'choices=("pinned", "backward", "causal", "causal_w", "causal_z")', 'choices=("pinned", "backward", "causal", "causal_w", "causal_z", "causal_v")')
w = rep(w, '    orig_hf = T.hermite_frames\n', '    orig_hf = T.hermite_frames\n    orig_cp = T.WindowState.commit_plan\n')
w = rep(w, '        T.hermite_frames = counted_frames(orig_hf, mine.readout)\n',
        '        T.hermite_frames = counted_frames(orig_hf, mine.readout)\n'
        '        if mine.readout == "causal_v":\n'
        '            T.WindowState.commit_plan = commit_plan_with_state(orig_cp)\n')
w = rep_code(w, "T.hermite_frames = orig_hf", "T.hermite_frames = orig_hf; T.WindowState.commit_plan = orig_cp", 3)
assert w.count("T.WindowState.commit_plan = orig_cp") == 3 and w.count("orig_cp = T.WindowState.commit_plan") == 1
save("v8f_t1_eval_reader_rule_rv.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_accwin10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n",
        "#!/usr/bin/env bash\n# pod_accwin2s10k_queue.sh -- written by make_accwin2s10k.py from pod_accwin10k_queue.sh. Card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K): the tool check of control v9 first\n"
        "# (check_tick_readout_rv.py on the pod, no model), then five rollouts without a rule with the wrapper v8f_t1_eval_reader_rule_rv.py: R0VID (readout causal_w, 5 steps, model seed 0:\n"
        "# must equal the banked dump of R0W5, control v1, comparison marker v1cmp), then the readout causal_v (the acceleration over the last 2.0 s of executed motion) at 5 steps\n"
        "# (R0V5) and at 10 steps (R0V10) at two model seeds. No limit, no training. The queue waits for the end marker of the previous queue (variables WAIT_DONE / WAIT_STOP, defaults\n"
        "# RULE1S10K_DONE.txt / RULE1S10K_STOP.txt in the folder $W). Markers START_ACCWIN2S10K.txt / ACCWIN2S10K_STOP.txt / ACCWIN2S10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "ACCWIN10K", "ACCWIN2S10K", 3)
q = rep(q, 'card ADD55: seven rollouts without a rule (acceleration read at the replan: over the last 1.0 s and held at 0, at 5 steps; over the last 1.0 s at 10 steps); checkpoints of card ADD46"',
        'card ADD58: tool check, then five rollouts without a rule (acceleration read at the replan: over the last 1.0 s at 5 steps as control; over the last 2.0 s of executed motion at 5 and at 10 steps); checkpoints of card ADD46"')
q = span(q, 'until [ -f "$W/HANDOVER10K_DONE.txt" ]', 'stop "NO-BANKED-R0C5"\n',
         'WAIT_DONE=${WAIT_DONE:-RULE1S10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-RULE1S10K_STOP.txt}\n'
         '  until [ -f "$W/$WAIT_DONE" ] || [ -f "$W/$WAIT_STOP" ]; do\n'
         '    [ $(( $(date +%s) - t0 )) -gt 28800 ] && stop "WAIT-PREVIOUS-QUEUE-TIMEOUT"\n'
         '    sleep 30\n'
         '  done\n'
         '  log "the previous queue has ended ($WAIT_DONE or $WAIT_STOP); the tool check and the rollouts of card ADD58 start"\n'
         '  [ -f "$W/t1/I10R0W5_s0/T1_DONE" ] || stop "NO-BANKED-R0W5"\n')
q = rep_code(q, "v8f_t1_eval_reader_rule_rz.py", "v8f_t1_eval_reader_rule_rv.py", 3)
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rv.py ', 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rv.py v8f_t1_eval_reader_rule_rz.py check_tick_readout_rv.py ')
q = rep(q, '[ "$DRY" = 1 ] || [ -f "$W/t1/I10R0M5_s1/T1_DONE" ] || stop "NO-BANKED-R0M5"\n',
        '[ "$DRY" = 1 ] || [ -f "$W/t1/I10R0M5_s1/T1_DONE" ] || stop "NO-BANKED-R0M5"\n'
        'if [ "$DRY" != 1 ]; then                                # the reused runs of the card (control v6 reads their result files)\n'
        '  for k in "${ARMS[@]}"; do\n'
        '    for nm in I10R0 I10R0M5 I10R0W5 I10R0Z5 I10R0W10; do\n'
        '      [ -f "$W/t1/${nm}_s$k/T1_DONE" ] || stop "NO-BANKED-${nm} s$k"\n'
        '    done\n'
        '  done\n'
        'fi\n')
q = span(q, 'READOUT=causal REPLAN=5 rollout I10R0ZID R0 0 0', '  READOUT=causal_w REPLAN=10 rollout I10R0W10 R0 "$k" "$k"\ndone\n',
         '# ---------------------------------------------------------------- control v9 first: the tool check on the pod (no model, no GPU); its output file and the copy of the wrapper it ran with are read by the reader\n'
         'if ! done_step v9check; then\n'
         '  if [ "$DRY" = 1 ]; then\n'
         '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_tick_readout_rv.py $TREE > $W/accwin2s_test/CHECK_TICK_READOUT_RV.txt"\n'
         '  else\n'
         '    mkdir -p "$W/accwin2s_test"\n'
         '    cp -f "$B/v8f_t1_eval_reader_rule_rv.py" "$W/accwin2s_test/v8f_t1_eval_reader_rule_rv.py"\n'
         '    CUDA_VISIBLE_DEVICES=-1 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/check_tick_readout_rv.py" "$TREE" > "$W/accwin2s_test/CHECK_TICK_READOUT_RV.txt" 2>&1\n'
         '    tail -n 1 "$W/accwin2s_test/CHECK_TICK_READOUT_RV.txt" | grep -q "^ZZ CHECK-TICK-READOUT-RV all hold: True" || stop "TOOL-CHECK-RV-FAILED"\n'
         '    mark v9check\n'
         '  fi\n'
         'fi\n'
         'READOUT=causal_w REPLAN=5 rollout I10R0VID R0 0 0       # control v1: the readout causal_w through the new wrapper must give the banked dump of R0W5 (card ADD55)\n'
         'if [ "$DRY" != 1 ] && ! done_step v1cmp; then\n'
         '  mkdir -p "$W/small_accwin2s"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0W5_s0/dump_I10R0W5_s0_m10_causal_heads" "$W/t1/I10R0VID_s0/dump_I10R0VID_s0_m10_causal_heads" > "$W/small_accwin2s/CMP_R0VID.txt" 2>&1 || stop "V1-RERUN-DIFFERS"\n'
         '  mark v1cmp\n'
         'fi\n'
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal_v REPLAN=5 rollout I10R0V5 R0 "$k" "$k"\n'
         'done\n'
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal_v REPLAN=10 rollout I10R0V10 R0 "$k" "$k"\n'
         'done\n')
c = code_lines(q)
for bad in ("HANDOVER10K", "ACCWIN10K", "I10R0ZID", "z1cmp", "small_accwin/", '_rz.py"', "I10R0C5", "FEAS10K"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 3 and c.count("ACCWIN2S10K") >= 3 and "v1cmp" in c
save("pod_accwin2s10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_accwin10k.sh")
c = rep(c, '"$I/pod_accwin10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rz.py"',
        '"$I/pod_accwin2s10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rv.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/check_tick_readout_rv.py"')
c = rep(c, "bash -n $B/pod_accwin10k_queue.sh", "bash -n $B/pod_accwin2s10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_accwin10k_queue.sh > $W/queue_accwin_stdout.txt", "nohup setsid bash $B/pod_accwin2s10k_queue.sh > $W/queue_accwin2s_stdout.txt")
c = rep(c, "LOGF=$I/chain_accwin10k.log", "LOGF=$I/chain_accwin2s10k.log")
c = rep(c, "for nm in I10R0ZID I10R0W5 I10R0Z5 I10R0W10; do", "for nm in I10R0VID I10R0V5 I10R0V10; do")
c = rep(c, "  d=$(pod \"[ -f $W/t1/I10RRx_s$m/T1_DONE ] && echo yes\" | tr -cd 'a-z'); [ \"$d\" = yes ] && pull_dir \"I10RRx_s$m\" \"$W/t1/I10RRx_s$m\" \"$PULL/I10RRx_s$m\"\n", "")
c = rep(c, "small_accwin", "small_accwin2s", 5, at_least=True)
c = rep(c, "DIAG_TAIL3.json $W/small_accwin2s/", "DIAG_TAIL3.json $W/accwin2s_test/CHECK_TICK_READOUT_RV.txt $W/small_accwin2s/")
c = rep(c, "ACCWIN10K", "ACCWIN2S10K", 6, at_least=True)
c = rep(c, '[ "$k" -gt 720 ] && { st=STOP; say "not ended after 12 h"; break; }', '[ "$k" -gt 1440 ] && { st=STOP; say "not ended after 24 h"; break; }')
c = rep(c, "after 12 h (a poll", "after 24 h (a poll")
c = rep(c, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# chain_accwin2s10k.sh -- written by make_accwin2s10k.py from chain_accwin10k.sh for card PREREG_V8F_PROOF_ADD58 (ACCWIN2S10K). The poll runs 24 h (the queue first waits for the\n# end marker of the previous queue, up to 8 h). The header lines below are those of the parent.\n")
cc = code_lines(c)
for bad in ("ACCWIN10K", "accwin10k", "I10R0ZID", "I10RRx", "small_accwin/"):
    assert bad not in cc, bad
save("chain_accwin2s10k.sh", c)
print("ZZ MAKE-ACCWIN2S10K DONE")
