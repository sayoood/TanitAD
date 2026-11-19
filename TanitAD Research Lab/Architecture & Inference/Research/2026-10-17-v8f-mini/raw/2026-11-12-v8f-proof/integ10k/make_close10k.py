"""make_close10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD48 (CLOSE10K) by asserted text edits:
  v8f_t1_eval_reader_rule.py -> v8f_t1_eval_reader_rule_c.py  (two more forms: RLc = labels with the label closing speed, RRc = reader presence and gap with the label closing speed)
  pod_price10k_queue.sh      -> pod_close10k_queue.sh         (no map step; the rerun RLbR and its dump comparison first, then RLc and RQ50c; own markers)
  chain_price10k.sh          -> chain_close10k.sh             (pushes the new files, starts the queue, waits on its markers, pulls the five run folders and the small files)
The originals are not changed. Usage: python make_close10k.py (in this folder)"""
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
w = load("v8f_t1_eval_reader_rule.py")
w = rep(w, '"""v8f_t1_eval_reader_rule.py -- card PREREG_V8F_PROOF_ADD46',
        '"""v8f_t1_eval_reader_rule_c.py -- written by make_close10k.py from v8f_t1_eval_reader_rule.py for card PREREG_V8F_PROOF_ADD48 (CLOSE10K): two more forms,\n'
        'RLc (labels: presence, gap and closing speed) and RRc (reader presence and gap through the map, closing speed from the labels where the label has one, else 0;\n'
        'a diagnostic, not admissible at inference). The five forms of card ADD46 are unchanged. Header of the original:\n'
        'v8f_t1_eval_reader_rule.py -- card PREREG_V8F_PROOF_ADD46')
w = rep(w, 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0")', 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc")')
w = rep(w, '    elif form == "RR":\n        used = np.minimum(', '    elif form in ("RR", "RRc"):\n        used = np.minimum(')
w = rep(w, '    if f in GAP_PLANT or f == "RR":', '    if f in GAP_PLANT or f in ("RR", "RRc"):')
w = rep(w, '        if f == "RR":\n', '        if f in ("RR", "RRc"):\n')
w = rep(w, '        mix = ("head", "head", "zero")', '        mix = ("head", "head", "label" if f == "RRc" else "zero")')
w = rep(w, '    elif f == "RLb":\n        W.head_values = head_values_none\n        mix = ("label", "label", "zero")',
        '    elif f in ("RLb", "RLc"):\n        W.head_values = head_values_none\n        mix = ("label", "label", "label" if f == "RLc" else "zero")')
save("v8f_t1_eval_reader_rule_c.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_price10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_close10k_queue.sh -- written by make_close10k.py from pod_price10k_queue.sh. Card PREREG_V8F_PROOF_ADD48 (CLOSE10K): five rollouts with the wrapper\n"
        "# v8f_t1_eval_reader_rule_c.py: RLbR (rerun of RLb at model seed 0, compared with the banked dump: control c1), RLc and RQ50c at two model seeds. No training, no map fit.\n"
        "# Markers START_CLOSE10K.txt / CLOSE10K_STOP.txt / CLOSE10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, "START_PRICE10K.txt", "START_CLOSE10K.txt", 2, at_least=True)
q = rep(q, "PRICE10K_STOP.txt", "CLOSE10K_STOP.txt", 1, at_least=True)
q = rep(q, "PRICE10K_DONE.txt", "CLOSE10K_DONE.txt", 1, at_least=True)
q = rep(q, 'card ADD47 (PRICE10K): maps of the sweep and six rollouts; reader arms and checkpoints of card ADD46"',
        'card ADD48 (CLOSE10K): five rollouts with the closing speed; reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "for f in qmap_fit_lvl.py price_maps_check.py QMAP_IDENTITY.json LS_REF40_WINDOWS.json", "for f in cmp_dumps.py v8f_t1_eval_reader_rule_c.py LS_REF40_WINDOWS.json")
q = span(q, "# ---------------------------------------------------------------- 1 maps of the sweep (CPU) and their check (control m1)\n",
         "# ---------------------------------------------------------------- 5 the twelve rollouts\n",
         "# ---------------------------------------------------------------- 1 inputs of card ADD48 (no map is fitted here)\n"
         'for k in "${ARMS[@]}"; do\n'
         '  [ "$DRY" = 1 ] || [ -f "$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" ] || stop "NO-MAP-Q50 s$k"\n'
         "done\n"
         '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RLb_s0/T1_DONE" ] || stop "NO-BANKED-RLB"\n\n', keep_end=True)
q = rep(q, "    RR|RRid|RR0) extra+=(--reader", "    RR|RRc|RRid|RR0) extra+=(--reader")
q = rep(q, '  [ "$form" = RR ] && extra+=(--qmap "${QMAP_FILE:?}")          # card ADD47: the map file is given by the caller',
        '  case "$form" in RR|RRc) extra+=(--qmap "${QMAP_FILE:?}") ;; esac          # the map file is given by the caller')
q = rep(q, "$B/v8f_t1_eval_reader_rule.py", "$B/v8f_t1_eval_reader_rule_c.py", 2)
q = span(q, 'for k in "${ARMS[@]}"; do\n  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50 RR "$k" "$k"      # the deciding level first\n',
         '  QMAP_FILE="$B/QMAP_IDENTITY.json" rollout I10RRaw RR "$k" "$k"\ndone\n',
         "rollout I10RLbR RLb 0 0                                   # control c1 first: the rerun of RLb under the new wrapper\n"
         'if [ "$DRY" != 1 ] && ! done_step c1cmp; then\n'
         '  mkdir -p "$W/small_close"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLb_s0/dump_I10RLb_s0_m10_causal_heads" "$W/t1/I10RLbR_s0/dump_I10RLbR_s0_m10_causal_heads" > "$W/small_close/CMP_RLBR.txt" 2>&1 || stop "C1-RERUN-DIFFERS"\n'
         "  mark c1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do rollout I10RLc RLc "$k" "$k"; done\n'
         'for k in "${ARMS[@]}"; do\n'
         '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50c RRc "$k" "$k"\n'
         "done\n")
c = code_lines(q)
assert "qmap_fit_lvl" not in c and "price_maps_check" not in c and "I10RQ35" not in c and "I10RRaw" not in c and "PRICE10K" not in c
assert c.count("v8f_t1_eval_reader_rule_c.py") == 3 and c.count("rollout I10R") == 3
save("pod_close10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_price10k.sh")
c = rep(c, "LOGF=$I/chain_price10k.log", "LOGF=$I/chain_close10k.log")
c = rep(c, '"$I/pod_price10k_queue.sh" "$I/qmap_fit_lvl.py" "$I/price_maps_check.py" "$I/QMAP_IDENTITY.json"', '"$I/pod_close10k_queue.sh" "$I/v8f_t1_eval_reader_rule_c.py" "$H/distrule/cmp_dumps.py"')
c = rep(c, "bash -n $B/pod_price10k_queue.sh", "bash -n $B/pod_close10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_price10k_queue.sh > $W/queue_price_stdout.txt", "nohup setsid bash $B/pod_close10k_queue.sh > $W/queue_close_stdout.txt")
c = rep(c, "START_PRICE10K.txt", "START_CLOSE10K.txt", 3, at_least=True)
c = rep(c, "PRICE10K_STOP", "CLOSE10K_STOP", 2, at_least=True)
c = rep(c, "PRICE10K_DONE", "CLOSE10K_DONE", 1, at_least=True)
c = rep(c, "CHAIN-PRICE10K", "CHAIN-CLOSE10K", 3, at_least=True)
c = rep(c, "for nm in I10RQ50 I10RQ35 I10RRaw; do", "for nm in I10RLc I10RQ50c I10RLbR; do")
c = rep(c, "$W/small_price", "$W/small_close", 5, at_least=True)
c = rep(c, 'pull_dir small_price "$W/small_close" "$PULL/small_price"', 'pull_dir small_close "$W/small_close" "$PULL/small_close"')
cc = code_lines(c)
assert "PRICE10K" not in cc and "small_price" not in cc and "qmap_fit_lvl" not in cc
save("chain_close10k.sh", c)
print("ZZ MAKE-CLOSE10K DONE")
