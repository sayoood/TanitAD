"""make_handover10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD54 (HANDOVER10K: what the model reads as its own speed and acceleration at a replan) by asserted
text edits:
  v8f_t1_eval_reader_rule_a.py -> v8f_t1_eval_reader_rule_ro.py  (one more parameter, --readout pinned|backward|causal, with --readout-log)
  pod_feas10k_queue.sh         -> pod_handover10k_queue2.sh
  chain_feas10k.sh             -> chain_handover10k.sh
The originals are not changed. Usage: python make_handover10k.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card (a file name can be taken)
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert "make_handover10k.py" in head or "chain_handover10k" in head, "%s exists and was not written by this script" % name
    with io.open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (name, len(text.encode("utf-8"))))


def rep(text, old, new, n=1, at_least=False):
    c = text.count(old)
    assert (c >= n) if at_least else (c == n), "edit anchor %r occurs %d times, expected %s%d" % (old[:70], c, ">= " if at_least else "", n)
    return text.replace(old, new)


def rep_code(text, old, new, n):
    """replace in lines that are not comments; exactly n lines must change."""
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


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_a.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py from v8f_t1_eval_reader_rule_t.py for card PREREG_V8F_PROOF_ADD53 (FEAS10K): two more parameters,',
        '"""v8f_t1_eval_reader_rule_ro.py -- written by make_handover10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): one more parameter,\n'
        '--readout, which changes what the rebuilt ego state holds as speed (and with it the one-frame acceleration) after executed steps. The pinned tool fills the 10 Hz frames\n'
        'of the executed steps by a spline with central tangents, so the speed at the last executed knot is the mean of the last executed step and the next step of the plan,\n'
        'which is never executed.  pinned   = the tool as it is (the frames function is passed through and only counted);\n'
        '                          backward = the same function given the executed knots only (a backward difference at the last executed knot);\n'
        '                          causal   = positions and headings as pinned, the speed column from executed steps only: at a knot the step speed into it, at the half\n'
        '                                     step the mean of that step and the step before it (so the one-frame acceleration at a tick is the executed one).\n'
        'It acts on every committed plan (model, kinematic prior alone, rule forms). A study arm; the pinned file is not edited. Header of the parent:\n'
        'v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py from v8f_t1_eval_reader_rule_t.py for card PREREG_V8F_PROOF_ADD53 (FEAS10K): two more parameters,')
w = rep(w, '"alim": None, "m": 10, "alog": {},\n', '"alim": None, "m": 10, "alog": {}, "readout": "pinned", "hf": {"calls": 0, "max_dv": 0.0, "max_da": 0.0, "sum_dv": 0.0, "sum_da": 0.0},\n')
w = rep(w, 'class AccelLimited:\n',
        'def backward_frames(orig, k_prev, knots, m, psi0):\n'
        '    """the pinned frames function on the executed knots only: its last tangent is then the backward difference."""\n'
        '    return orig(k_prev, np.asarray(knots, np.float64)[:m + 1], m, psi0)\n\n\n'
        'def causal_frames(orig, k_prev, knots, m, psi0):\n'
        '    """positions and headings of the pinned frames function; the speed column from executed steps only. s[0] = the step into the current knot (the last executed step\n'
        '    before this block, or the logged step before the window), s[i] = executed step i; frame 2i-1 (half step) = (s[i-1] + s[i]) / 2, frame 2i (knot) = s[i]."""\n'
        '    xy, psi, v = orig(k_prev, knots, m, psi0)\n'
        '    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])\n'
        '    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S\n'
        '    v = np.asarray(v, np.float64).copy()\n'
        '    for i in range(1, m + 1):\n'
        '        v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])\n'
        '        v[2 * (i - 1) + 1] = s[i]\n'
        '    return xy, psi, v\n\n\n'
        'def counted_frames(orig, mode):\n'
        '    """the frames function of the chosen readout; counts its calls and, per call, how far the speed and the one-frame acceleration at the last executed knot lie from\n'
        '    the last executed step speed and the executed acceleration (s[m] - s[m-1]) / 0.2 s."""\n'
        '    def frames(k_prev, knots, m, psi0):\n'
        '        if mode == "causal":\n'
        '            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0)\n'
        '        elif mode == "backward":\n'
        '            xy, psi, v = backward_frames(orig, k_prev, knots, m, psi0)\n'
        '        else:\n'
        '            xy, psi, v = orig(k_prev, knots, m, psi0)\n'
        '        K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])\n'
        '        s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S\n'
        '        h = STATE["hf"]\n'
        '        dv, da = abs(float(v[-1]) - float(s[m])), abs((float(v[-1]) - float(v[-2])) / (DT_S / 2.0) - (float(s[m]) - float(s[m - 1])) / DT_S)\n'
        '        if np.isfinite(dv) and np.isfinite(da):\n'
        '            h["calls"] += 1\n'
        '            h["max_dv"], h["max_da"], h["sum_dv"], h["sum_da"] = max(h["max_dv"], dv), max(h["max_da"], da), h["sum_dv"] + dv, h["sum_da"] + da\n'
        '        return xy, psi, v\n'
        '    return frames\n\n\n'
        'def save_rlog(path, form, replan_steps):\n'
        '    if not path:\n'
        '        return\n'
        '    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)\n'
        '    h = STATE["hf"]\n'
        '    rec = {"readout": STATE["readout"], "form": form, "replan_steps": int(replan_steps), "calls": int(h["calls"]), "max_abs_speed_minus_last_step": h["max_dv"],\n'
        '           "max_abs_acceleration_minus_executed": h["max_da"], "mean_abs_speed_minus_last_step": h["sum_dv"] / max(h["calls"], 1),\n'
        '           "mean_abs_acceleration_minus_executed": h["sum_da"] / max(h["calls"], 1)}\n'
        '    with open(path, "w", encoding="utf-8", newline="\\n") as fh:\n'
        '        json.dump(rec, fh, indent=1)\n'
        '    print("wrote", path, flush=True)\n\n\n'
        'class AccelLimited:\n')
w = rep(w, '    ap.add_argument("--alim-log", default="")\n',
        '    ap.add_argument("--alim-log", default="")\n    ap.add_argument("--readout", choices=("pinned", "backward", "causal"), default=None)\n    ap.add_argument("--readout-log", default="")\n')
w = rep(w, '        T.floor_planner = limited_floor\n',
        '        T.floor_planner = limited_floor\n'
        '    if (mine.readout is None) != (not mine.readout_log):\n'
        '        raise SystemExit("REFUSED: --readout and --readout-log come together")\n'
        '    orig_hf = T.hermite_frames\n'
        '    if mine.readout is not None:\n'
        '        STATE["readout"] = mine.readout\n'
        '        T.hermite_frames = counted_frames(orig_hf, mine.readout)\n')
w = rep(w, '        if STATE["alim"] is None:\n            return T.cmd_run(a)\n',
        '        if STATE["alim"] is None:\n'
        '            try:\n'
        '                return T.cmd_run(a)\n'
        '            finally:\n'
        '                T.hermite_frames = orig_hf\n'
        '                save_rlog(mine.readout_log, f, a.replan_steps)\n')
w = rep(w, '            T.floor_planner = orig_fp\n            save_alog(mine.alim_log, f, a.replan_steps)\n',
        '            T.floor_planner = orig_fp\n            T.hermite_frames = orig_hf\n            save_alog(mine.alim_log, f, a.replan_steps)\n            save_rlog(mine.readout_log, f, a.replan_steps)\n')
w = rep(w, '        T.floor_planner = orig_fp\n        save_alog(mine.alim_log, f, a.replan_steps)\n',
        '        T.floor_planner = orig_fp\n        T.hermite_frames = orig_hf\n        save_alog(mine.alim_log, f, a.replan_steps)\n        save_rlog(mine.readout_log, f, a.replan_steps)\n')
save("v8f_t1_eval_reader_rule_ro.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_feas10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_handover10k_queue2.sh -- written by make_handover10k.py from pod_feas10k_queue.sh. Card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): seven rollouts without a rule\n"
        "# with the wrapper v8f_t1_eval_reader_rule_ro.py: R0HID (readout pinned, 10 steps, model seed 0: must equal the banked R0, control h1), then at two model seeds the causal\n"
        "# readout at 5 steps (R0C5), the backward readout at 5 steps (R0B5) and the causal readout at 10 steps (R0C10). No limit, no training, no map fit. The queue waits for\n"
        "# the end of the queue of card ADD53 before its first rollout. Markers START_HANDOVER10K.txt / HANDOVER10K_STOP.txt / HANDOVER10K_DONE.txt.\n"
        "# The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD53: eleven rollouts (acceleration limit 3.0 / 8.0 m/s2 on the executed plan at 5 and at 10 steps, 1.5 / 8.0 at 5 steps without a rule); reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD54: seven rollouts without a rule (speed readout at the replan: causal and backward at 5 steps, causal at 10 steps); checkpoints of card ADD46"\n'
        'if [ "$DRY" != 1 ]; then\n'
        '  t0=$(date +%s)\n'
        '  until [ -f "$W/FEAS10K_DONE.txt" ] || [ -f "$W/FEAS10K_STOP.txt" ]; do\n'
        '    [ $(( $(date +%s) - t0 )) -gt 21600 ] && stop "WAIT-FEAS-TIMEOUT"\n'
        '    sleep 30\n'
        '  done\n'
        '  log "the queue of card ADD53 has ended; the rollouts of card ADD54 start"\n'
        'fi')
n_code = sum(1 for ln in q.split("\n") if not ln.lstrip().startswith("#") and "FEAS10K" in ln)
q = rep_code(q, "START_FEAS10K", "START_HANDOVER10K", 1)
q = rep_code(q, "v8f_t1_eval_reader_rule_a.py", "v8f_t1_eval_reader_rule_ro.py", 3)
q = rep(q, '  [ -n "${AMAX:-}" ] && extra+=(--a-max "$AMAX" --b-lim "${BLIM:?}" --alim-log "$out/ticks/alim_log_$arm.npz")\n',
        '  [ -n "${AMAX:-}" ] && extra+=(--a-max "$AMAX" --b-lim "${BLIM:?}" --alim-log "$out/ticks/alim_log_$arm.npz")\n'
        '  [ -n "${READOUT:-}" ] && extra+=(--readout "$READOUT" --readout-log "$out/ticks/readout_log_$arm.json")\n')
q = span(q, "AMAX=1e9 BLIM=1e9 REPLAN=10 rollout I10R0AID R0 0 0", '  AMAX=1.5 BLIM=8.0 REPLAN=5 rollout I10R0A5L R0 "$k" "$k"      # the dose arm: half the acceleration limit, no rule\ndone\n',
         'READOUT=pinned REPLAN=10 rollout I10R0HID R0 0 0       # control h1 first: the pinned readout through the new wrapper must give the banked R0\n'
         'if [ "$DRY" != 1 ] && ! done_step h1cmp; then\n'
         '  mkdir -p "$W/small_handover"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0_s0/dump_I10R0_s0_m10_causal_heads" "$W/t1/I10R0HID_s0/dump_I10R0HID_s0_m10_causal_heads" > "$W/small_handover/CMP_R0HID.txt" 2>&1 || stop "H1-RERUN-DIFFERS"\n'
         "  mark h1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal REPLAN=5 rollout I10R0C5 R0 "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=backward REPLAN=5 rollout I10R0B5 R0 "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal REPLAN=10 rollout I10R0C10 R0 "$k" "$k"\n'
         "done\n")
# the end markers of this queue (the wait above keeps the two FEAS10K names it reads)
lines, k = [], 0
for ln in q.split("\n"):
    if not ln.lstrip().startswith("#") and ("FEAS10K_DONE" in ln or "FEAS10K_STOP" in ln) and "until [" not in ln:
        ln = ln.replace("FEAS10K_DONE", "HANDOVER10K_DONE").replace("FEAS10K_STOP", "HANDOVER10K_STOP")
        k += 1
    lines.append(ln)
q = "\n".join(lines)
assert k >= 2, k
c = code_lines(q)
assert c.count("FEAS10K") == 2 and "I10R0A5" not in c and "small_feas" not in c and "_a.py" not in c and c.count("rollout I10R") == 4 and c.count("--readout-log") == 1, (c.count("FEAS10K"), c.count("rollout I10R"))
save("pod_handover10k_queue2.sh", q)

c = load("chain_feas10k.sh")
c = rep(c, "LOGF=$I/chain_feas10k.log", "LOGF=$I/chain_handover10k.log")
c = rep(c, '"$I/pod_feas10k_queue.sh" "$I/v8f_t1_eval_reader_rule_a.py"', '"$I/pod_handover10k_queue2.sh" "$I/v8f_t1_eval_reader_rule_ro.py"')
c = rep(c, "bash -n $B/pod_feas10k_queue.sh", "bash -n $B/pod_handover10k_queue2.sh")
c = rep(c, "nohup setsid bash $B/pod_feas10k_queue.sh > $W/queue_feas_stdout.txt", "nohup setsid bash $B/pod_handover10k_queue2.sh > $W/queue_handover_stdout.txt")
c = rep(c, "FEAS10K", "HANDOVER10K", 6, at_least=True)
c = rep(c, "for nm in I10R0AID I10R0A5 I10RLcA5 I10RQ50A5 I10R0A10 I10R0A5L; do", "for nm in I10R0HID I10R0C5 I10R0B5 I10R0C10; do")
c = rep(c, "small_feas", "small_handover", 2, at_least=True)
cc = code_lines(c)
assert "FEAS10K" not in cc and "small_feas" not in cc and "pod_feas10k" not in cc and "_a.py" not in cc
save("chain_handover10k.sh", c)
print("ZZ MAKE-HANDOVER10K DONE")
