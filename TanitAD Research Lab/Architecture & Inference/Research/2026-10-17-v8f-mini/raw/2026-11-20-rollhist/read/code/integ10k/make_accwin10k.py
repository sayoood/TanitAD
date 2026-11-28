"""make_accwin10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD55 (ACCWIN10K: the acceleration the model reads at a replan, over a longer executed window or held
at zero) by asserted text edits:
  v8f_t1_eval_reader_rule_ro.py -> v8f_t1_eval_reader_rule_rz.py  (two more readouts: causal_w, causal_z)
  pod_handover10k_queue2.sh     -> pod_accwin10k_queue.sh
  chain_handover10k.sh          -> chain_accwin10k.sh
The originals are not changed. Usage: python make_accwin10k.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_accwin10k.py"


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card (a file name can be taken)
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert ME in head or "chain_accwin10k" in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (name, len(text.encode("utf-8"))))


def rep(text, old, new, n=1, at_least=False):
    c = text.count(old)
    assert (c >= n) if at_least else (c == n), "edit anchor %r occurs %d times, expected %s%d" % (old[:70], c, ">= " if at_least else "", n)
    return text.replace(old, new)


def rep_code(text, old, new, n):
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
w = load("v8f_t1_eval_reader_rule_ro.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_ro.py -- written by make_handover10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): one more parameter,',
        '"""v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.\n'
        'Both keep the positions and headings of the pinned frames function and the causal speed at a knot (the step speed into it); they differ from `causal` in the half-step\n'
        'speed, which sets the one-frame acceleration the model reads at every executed step:\n'
        '  causal_w  acceleration = (s[i] - s[i-w]) / (0.2 s x w), w = min(i, 5): the executed speed change over the last 1.0 s of the block (i = step in the block, s[0] = the\n'
        '            step into the block)\n'
        '  causal_z  acceleration = 0 at every executed step (the model reads its executed speed and no acceleration from its own rollout)\n'
        '`causal`, `backward`, `pinned` are unchanged (bit for bit). Header of the parent:\n'
        'v8f_t1_eval_reader_rule_ro.py -- written by make_handover10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): one more parameter,')
w = rep(w, 'def causal_frames(orig, k_prev, knots, m, psi0):\n', 'def causal_frames(orig, k_prev, knots, m, psi0, acc="step"):\n')
w = rep(w, '        v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])\n',
        '        if acc == "zero":\n'
        '            v[2 * (i - 1)] = s[i]\n'
        '        elif acc == "window":\n'
        '            n_w = min(i, 5)\n'
        '            v[2 * (i - 1)] = s[i] - 0.5 * DT_S * (s[i] - s[i - n_w]) / (DT_S * n_w)\n'
        '        else:\n'
        '            v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])\n')
w = rep(w, '        if mode == "causal":\n            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0)\n',
        '        if mode in ("causal", "causal_w", "causal_z"):\n'
        '            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0, {"causal": "step", "causal_w": "window", "causal_z": "zero"}[mode])\n')
w = rep(w, '        dv, da = abs(float(v[-1]) - float(s[m])), abs((float(v[-1]) - float(v[-2])) / (DT_S / 2.0) - (float(s[m]) - float(s[m - 1])) / DT_S)\n',
        '        dv, da = abs(float(v[-1]) - float(s[m])), abs((float(v[-1]) - float(v[-2])) / (DT_S / 2.0) - (float(s[m]) - float(s[m - 1])) / DT_S)\n'
        '        a0r, n_w = (float(v[-1]) - float(v[-2])) / (DT_S / 2.0), min(m, 5)\n'
        '        daw = abs(a0r - (float(s[m]) - float(s[m - n_w])) / (DT_S * n_w))\n')
w = rep(w, '            h["max_dv"], h["max_da"], h["sum_dv"], h["sum_da"] = max(h["max_dv"], dv), max(h["max_da"], da), h["sum_dv"] + dv, h["sum_da"] + da\n',
        '            h["max_dv"], h["max_da"], h["sum_dv"], h["sum_da"] = max(h["max_dv"], dv), max(h["max_da"], da), h["sum_dv"] + dv, h["sum_da"] + da\n'
        '            if np.isfinite(a0r) and np.isfinite(daw):\n'
        '                h["max_a0"], h["max_daw"], h["sum_a0"] = max(h.get("max_a0", 0.0), abs(a0r)), max(h.get("max_daw", 0.0), daw), h.get("sum_a0", 0.0) + abs(a0r)\n')
w = rep(w, '           "mean_abs_acceleration_minus_executed": h["sum_da"] / max(h["calls"], 1)}\n',
        '           "mean_abs_acceleration_minus_executed": h["sum_da"] / max(h["calls"], 1), "max_abs_acceleration_read": h.get("max_a0", 0.0),\n'
        '           "mean_abs_acceleration_read": h.get("sum_a0", 0.0) / max(h["calls"], 1), "max_abs_acceleration_minus_window": h.get("max_daw", 0.0)}\n')
w = rep(w, 'choices=("pinned", "backward", "causal")', 'choices=("pinned", "backward", "causal", "causal_w", "causal_z")')
save("v8f_t1_eval_reader_rule_rz.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_handover10k_queue2.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_accwin10k_queue.sh -- written by make_accwin10k.py from pod_handover10k_queue2.sh. Card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): seven rollouts without a rule with the\n"
        "# wrapper v8f_t1_eval_reader_rule_rz.py: R0ZID (readout causal, 5 steps, model seed 0: must equal the dump of R0C5 of card ADD54, control z1), then at two model seeds\n"
        "# the readout causal_w at 5 steps (R0W5), causal_z at 5 steps (R0Z5) and causal_w at 10 steps (R0W10). No limit, no training. The queue waits for the end of the queue\n"
        "# of card ADD54 before its first rollout. Markers START_ACCWIN10K.txt / ACCWIN10K_STOP.txt / ACCWIN10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD54: seven rollouts without a rule (speed readout at the replan: causal and backward at 5 steps, causal at 10 steps); checkpoints of card ADD46"',
        'card ADD55: seven rollouts without a rule (acceleration read at the replan: over the last 1.0 s and held at 0, at 5 steps; over the last 1.0 s at 10 steps); checkpoints of card ADD46"')
q = rep_code(q, 'until [ -f "$W/FEAS10K_DONE.txt" ] || [ -f "$W/FEAS10K_STOP.txt" ]; do', 'until [ -f "$W/HANDOVER10K_DONE.txt" ] || [ -f "$W/HANDOVER10K_STOP.txt" ]; do', 1)
q = rep_code(q, 'stop "WAIT-FEAS-TIMEOUT"', 'stop "WAIT-HANDOVER-TIMEOUT"', 1)
q = rep_code(q, 'log "the queue of card ADD53 has ended; the rollouts of card ADD54 start"', 'log "the queue of card ADD54 has ended; the rollouts of card ADD55 start"\n  [ -f "$W/t1/I10R0C5_s0/T1_DONE" ] || stop "NO-BANKED-R0C5"', 1)
q = rep_code(q, "START_HANDOVER10K", "START_ACCWIN10K", 1)
q = rep_code(q, "v8f_t1_eval_reader_rule_ro.py", "v8f_t1_eval_reader_rule_rz.py", 3)
q = rep_code(q, '> "$W/HANDOVER10K_STOP.txt"', '> "$W/ACCWIN10K_STOP.txt"', 1)
q = rep_code(q, '> "$W/HANDOVER10K_DONE.txt"', '> "$W/ACCWIN10K_DONE.txt"', 1)
q = span(q, "READOUT=pinned REPLAN=10 rollout I10R0HID R0 0 0", '  READOUT=causal REPLAN=10 rollout I10R0C10 R0 "$k" "$k"\ndone\n',
         'READOUT=causal REPLAN=5 rollout I10R0ZID R0 0 0       # control z1 first: the causal readout through the new wrapper must give the dump of R0C5 (card ADD54)\n'
         'if [ "$DRY" != 1 ] && ! done_step z1cmp; then\n'
         '  mkdir -p "$W/small_accwin"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0C5_s0/dump_I10R0C5_s0_m10_causal_heads" "$W/t1/I10R0ZID_s0/dump_I10R0ZID_s0_m10_causal_heads" > "$W/small_accwin/CMP_R0ZID.txt" 2>&1 || stop "Z1-RERUN-DIFFERS"\n'
         "  mark z1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal_w REPLAN=5 rollout I10R0W5 R0 "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal_z REPLAN=5 rollout I10R0Z5 R0 "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  READOUT=causal_w REPLAN=10 rollout I10R0W10 R0 "$k" "$k"\n'
         "done\n")
c = code_lines(q)
assert c.count("HANDOVER10K") == 2 and "FEAS10K" not in c and "I10R0HID" not in c and "small_handover" not in c and "_ro.py" not in c and c.count("rollout I10R") == 4, (c.count("HANDOVER10K"), c.count("rollout I10R"))
save("pod_accwin10k_queue.sh", q)

c = load("chain_handover10k.sh")
c = rep(c, "LOGF=$I/chain_handover10k.log", "LOGF=$I/chain_accwin10k.log")
c = rep(c, '"$I/pod_handover10k_queue2.sh" "$I/v8f_t1_eval_reader_rule_ro.py"', '"$I/pod_accwin10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rz.py"')
c = rep(c, "bash -n $B/pod_handover10k_queue2.sh", "bash -n $B/pod_accwin10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_handover10k_queue2.sh > $W/queue_handover_stdout.txt", "nohup setsid bash $B/pod_accwin10k_queue.sh > $W/queue_accwin_stdout.txt")
c = rep(c, "HANDOVER10K", "ACCWIN10K", 6, at_least=True)
c = rep(c, "for nm in I10R0HID I10R0C5 I10R0B5 I10R0C10; do", "for nm in I10R0ZID I10R0W5 I10R0Z5 I10R0W10; do")
c = rep(c, "small_handover", "small_accwin", 2, at_least=True)
cc = code_lines(c)
assert "HANDOVER10K" not in cc and "small_handover" not in cc and "pod_handover10k" not in cc and "_ro.py" not in cc
save("chain_accwin10k.sh", c)
print("ZZ MAKE-ACCWIN10K DONE")
