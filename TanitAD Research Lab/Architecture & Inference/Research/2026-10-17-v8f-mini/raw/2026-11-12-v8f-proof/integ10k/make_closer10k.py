"""make_closer10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD49 (CLOSER10K) by asserted text edits:
  v8f_t1_eval_reader_rule_c.py -> v8f_t1_eval_reader_rule_h.py  (one more form: RRh = reader presence and gap through the map, closing speed from the reader's own third
                                                                 output, the trained closing-speed output of the token probe; tokens only)
  pod_close10k_queue.sh        -> pod_closer10k_queue.sh        (the rerun RQ50R of RQ50 at model seed 0 and its dump comparison first, then RQ50h at two model seeds)
  chain_close10k.sh            -> chain_closer10k.sh
The originals are not changed. Usage: python make_closer10k.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    with io.open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (name, len(text.encode("utf-8"))))


def rep(text, old, new, n=1, at_least=False):
    c = text.count(old)
    assert (c >= n) if at_least else (c == n), "edit anchor %r occurs %d times, expected %s%d" % (old[:70], c, ">= " if at_least else "", n)
    return text.replace(old, new)


def span(text, start, end, new, keep_end=False):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + (0 if keep_end else len(end))
    assert i < j
    return text[:i] + new + text[j:]


def code_lines(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_c.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_c.py -- written by make_close10k.py from v8f_t1_eval_reader_rule.py for card PREREG_V8F_PROOF_ADD48 (CLOSE10K): two more forms,',
        '"""v8f_t1_eval_reader_rule_h.py -- written by make_closer10k.py from v8f_t1_eval_reader_rule_c.py for card PREREG_V8F_PROOF_ADD49 (CLOSER10K): one more form,\n'
        'RRh (reader presence and gap through the map, closing speed = the third output of the same token reader times 10 m/s; tokens only, no model weight, no label).\n'
        'The seven forms of the parent are unchanged. Header of the parent:\n'
        'v8f_t1_eval_reader_rule_c.py -- written by make_close10k.py from v8f_t1_eval_reader_rule.py for card PREREG_V8F_PROOF_ADD48 (CLOSE10K): two more forms,')
w = rep(w, 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc")', 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh")\nCLOS_SCALE_MS = 10.0                                # = CLOS_SCALE_MS of tok_probe_lr.py (the probe was trained on closing / 10 m/s)')
w = rep(w, 'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "cols": {k: [] for k in ("p", "g_read", "g_used", "ok")}}',
        'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "cols": {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}}')
w = rep(w, '    elif form in ("RR", "RRc"):\n        used = np.minimum(', '    elif form in ("RR", "RRc", "RRh"):\n        used = np.minimum(')
w = rep(w, 'def head_values_reader(model, feats, frame_ok=None, hist=None):',
        'def reader_closing(rd, tokens):\n'
        '    """The third output of the token reader: closing speed (lead speed minus ego speed) in m/s, [B]. Reads nothing but the tokens."""\n'
        '    import tok_reader as TR\n'
        '    if hasattr(tokens, "detach"):\n'
        '        tokens = tokens.detach().float().cpu().numpy()\n'
        '    E = rd.frame_embed(np.asarray(tokens, np.float32))\n'
        '    x = E.reshape(E.shape[0], -1)\n'
        '    h = TR._gelu(x @ rd.w["head.mlp.0.weight"].T + rd.w["head.mlp.0.bias"])\n'
        '    o = h @ rd.w["head.mlp.2.weight"].T + rd.w["head.mlp.2.bias"]\n'
        '    assert o.shape[1] == 3, o.shape\n'
        '    return (o[:, 2].astype(np.float64) * CLOS_SCALE_MS).astype(np.float32)\n\n\n'
        'def head_values_reader(model, feats, frame_ok=None, hist=None):')
w = rep(w, '    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok)):\n',
        '    c = reader_closing(rd, f)                                   # logged in every reader form; fed into the rule only in form RRh\n'
        '    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):\n')
w = rep(w, '    return p, used.astype(np.float32), np.zeros(len(p), np.float32), ok\n',
        '    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else np.zeros(len(p), np.float32)), ok\n')
w = rep(w, '    if f in GAP_PLANT or f in ("RR", "RRc"):', '    if f in GAP_PLANT or f in ("RR", "RRc", "RRh"):')
w = rep(w, '        if f in ("RR", "RRc"):\n', '        if f in ("RR", "RRc", "RRh"):\n')
w = rep(w, '        mix = ("head", "head", "label" if f == "RRc" else "zero")', '        mix = ("head", "head", {"RRc": "label", "RRh": "head"}.get(f, "zero"))')
save("v8f_t1_eval_reader_rule_h.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_close10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_closer10k_queue.sh -- written by make_closer10k.py from pod_close10k_queue.sh. Card PREREG_V8F_PROOF_ADD49 (CLOSER10K): three rollouts with the wrapper\n"
        "# v8f_t1_eval_reader_rule_h.py: RQ50R (rerun of RQ50 at model seed 0, compared with the banked dump: control h1), RQ50h at two model seeds. No training, no map fit.\n"
        "# Markers START_CLOSER10K.txt / CLOSER10K_STOP.txt / CLOSER10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD48 (CLOSE10K): five rollouts with the closing speed; reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD49: three rollouts with the closing speed of the reader; reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "CLOSE10K", "CLOSER10K", 4, at_least=True)
q = rep(q, "v8f_t1_eval_reader_rule_c.py", "v8f_t1_eval_reader_rule_h.py", 3, at_least=True)
q = rep(q, '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RLb_s0/T1_DONE" ] || stop "NO-BANKED-RLB"', '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RQ50_s0/T1_DONE" ] || stop "NO-BANKED-RQ50"')
q = rep(q, "    RR|RRc|RRid|RR0) extra+=(--reader", "    RR|RRc|RRh|RRid|RR0) extra+=(--reader")
q = rep(q, '  case "$form" in RR|RRc) extra+=(--qmap', '  case "$form" in RR|RRc|RRh) extra+=(--qmap')
q = span(q, "rollout I10RLbR RLb 0 0 ", '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50c RRc "$k" "$k"\ndone\n',
         'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50R RR 0 0       # control h1 first: the rerun of RQ50 under the new wrapper\n'
         'if [ "$DRY" != 1 ] && ! done_step h1cmp; then\n'
         '  mkdir -p "$W/small_closer"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RQ50_s0/dump_I10RQ50_s0_m10_causal_heads" "$W/t1/I10RQ50R_s0/dump_I10RQ50R_s0_m10_causal_heads" > "$W/small_closer/CMP_RQ50R.txt" 2>&1 || stop "H1-RERUN-DIFFERS"\n'
         "  mark h1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50h RRh "$k" "$k"\n'
         "done\n")
c = code_lines(q)
assert "CLOSE10K" not in c.replace("CLOSER10K", "") and "I10RLc" not in c and "I10RQ50c" not in c and "small_close/" not in c and "_c.py" not in c
assert c.count("rollout I10R") == 2 and c.count("v8f_t1_eval_reader_rule_h.py") == 3
save("pod_closer10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_close10k.sh")
c = rep(c, "LOGF=$I/chain_close10k.log", "LOGF=$I/chain_closer10k.log")
c = rep(c, '"$I/pod_close10k_queue.sh" "$I/v8f_t1_eval_reader_rule_c.py" "$H/distrule/cmp_dumps.py"', '"$I/pod_closer10k_queue.sh" "$I/v8f_t1_eval_reader_rule_h.py" "$H/distrule/cmp_dumps.py"')
c = rep(c, "bash -n $B/pod_close10k_queue.sh", "bash -n $B/pod_closer10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_close10k_queue.sh > $W/queue_close_stdout.txt", "nohup setsid bash $B/pod_closer10k_queue.sh > $W/queue_closer_stdout.txt")
c = rep(c, "CLOSE10K", "CLOSER10K", 6, at_least=True)
c = rep(c, "for nm in I10RLc I10RQ50c I10RLbR; do", "for nm in I10RQ50h I10RQ50R; do")
c = rep(c, "small_close", "small_closer", 6, at_least=True)
cc = code_lines(c)
assert "CLOSE10K" not in cc.replace("CLOSER10K", "") and "small_close/" not in cc and "pod_close10k" not in cc
save("chain_closer10k.sh", c)
print("ZZ MAKE-CLOSER10K DONE")
