"""make_taucurve10k.py -- writes the tools of the time-gap curve of the lead rule on the 10k checkpoints (card PREREG_V8F_PROOF_ADD51, TAUCURVE10K) by asserted text edits:
  v8f_t1_eval_reader_rule_d.py -> v8f_t1_eval_reader_rule_t.py  (one more parameter: --tau, the time gap of the rule in seconds; without it the rule keeps its 1.2 s)
  pod_gaprate10k_queue.sh      -> pod_taucurve10k_queue.sh      (a rerun of RLc with --tau 1.2 against the banked dump, then RLc and RQ50 at the time gaps 0.8, 1.0, 1.5, 2.0 s)
  chain_gaprate10k.sh          -> chain_taucurve10k.sh
The originals are not changed. Usage: python make_taucurve10k.py (in this folder)"""
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
w = load("v8f_t1_eval_reader_rule_d.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_d.py -- written by make_gaprate10k.py from v8f_t1_eval_reader_rule_h.py for card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): one more form,',
        '"""v8f_t1_eval_reader_rule_t.py -- written by make_taucurve10k.py from v8f_t1_eval_reader_rule_d.py for card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): one more parameter,\n'
        '--tau SECONDS, the time gap of the distance rule (dist_rule.distance_rule(tau=...)); without it the rule keeps its banked 1.2 s and every form is unchanged.\n'
        'The scoring of the unsafe rate does not read this parameter. Header of the parent:\n'
        'v8f_t1_eval_reader_rule_d.py -- written by make_gaprate10k.py from v8f_t1_eval_reader_rule_h.py for card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): one more form,')
w = rep(w, '    ap.add_argument("--k-diff", type=int, default=0)\n', '    ap.add_argument("--k-diff", type=int, default=0)\n    ap.add_argument("--tau", type=float, default=None)\n')
w = rep(w, '    if f == "R0":\n        if mine.rule_log or mine.reader or mine.qmap:',
        '    if mine.tau is not None and (f == "R0" or not (0.2 <= mine.tau <= 4.0)):\n        raise SystemExit("REFUSED: --tau needs a rule form and a value in [0.2, 4.0] s")\n'
        '    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}\n'
        '    if f == "R0":\n        if mine.rule_log or mine.reader or mine.qmap:')
w = rep(w, '    W.MODE_CFG["mix"] = ("mix", {})\n', '    W.MODE_CFG["mix"] = ("mix", dict(rule_kw))\n')
w = rep(w, '"TAU": R.TAU_S,', '"TAU": float(rule_kw.get("tau", R.TAU_S)),')
w = rep(w, '"mode_kwargs": "{}", "k_diff"', '"mode_kwargs": json.dumps(rule_kw, sort_keys=True), "k_diff"')
save("v8f_t1_eval_reader_rule_t.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_gaprate10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_taucurve10k_queue.sh -- written by make_taucurve10k.py from pod_gaprate10k_queue.sh. Card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): seventeen rollouts with the wrapper\n"
        "# v8f_t1_eval_reader_rule_t.py: RLcT12 (rerun of RLc at model seed 0 with --tau 1.2, compared with the banked dump: control t1), then per time gap 0.8, 1.0, 1.5, 2.0 s and\n"
        "# model seed the label-fed rule with the closing speed (RLcT<tau>) and the reader-fed rule at level 0.50 (RQ50T<tau>). No training, no map fit.\n"
        "# Markers START_TAUCURVE10K.txt / TAUCURVE10K_STOP.txt / TAUCURVE10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD50: a smoke and five rollouts (closing speed from the change of the reader gap; crossed reader seeds at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD51: seventeen rollouts (the time gap of the rule at 0.8, 1.0, 1.5, 2.0 s, label-fed with the closing speed and reader-fed at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "GAPRATE10K", "TAUCURVE10K", 4, at_least=True)
q = rep(q, "v8f_t1_eval_reader_rule_d.py", "v8f_t1_eval_reader_rule_t.py", 3, at_least=True)
q = rep(q, "for f in check_prev_read.py cmp_dumps.py ", "for f in cmp_dumps.py ")
q = rep(q, '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RQ50_s0/T1_DONE" ] || stop "NO-BANKED-RQ50"', '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RLc_s0/T1_DONE" ] || stop "NO-BANKED-RLC"')
q = rep(q, '  [ "$form" = RRd ] && extra+=(--slope "${SLOPE:?}" --k-diff "$KDIFF")\n', '  [ "$form" = RRd ] && extra+=(--slope "${SLOPE:?}" --k-diff "$KDIFF")\n  [ -n "${TAU:-}" ] && extra+=(--tau "$TAU")\n')
q = span(q, "KDIFF=25\n", 'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50x RR 1 0       # model seed 1 with reader seed 0 and that reader\'s map\n',
         'KDIFF=25\n'
         'TAU=1.2 rollout I10RLcT12 RLc 0 0       # control t1 first: RLc with the time gap given as 1.2 s must equal the banked RLc\n'
         'if [ "$DRY" != 1 ] && ! done_step t1cmp; then\n'
         '  mkdir -p "$W/small_taucurve"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLc_s0/dump_I10RLc_s0_m10_causal_heads" "$W/t1/I10RLcT12_s0/dump_I10RLcT12_s0_m10_causal_heads" > "$W/small_taucurve/CMP_RLcT12.txt" 2>&1 || stop "T1-RERUN-DIFFERS"\n'
         "  mark t1cmp\n"
         "fi\n"
         'for tt in 08:0.8 10:1.0 15:1.5 20:2.0; do\n'
         '  tn=${tt%%:*}; tv=${tt##*:}\n'
         '  for k in "${ARMS[@]}"; do\n'
         '    TAU=$tv rollout "I10RLcT$tn" RLc "$k" "$k"\n'
         '    TAU=$tv QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout "I10RQ50T$tn" RR "$k" "$k"\n'
         "  done\n"
         "done\n")
c = code_lines(q)
assert "GAPRATE10K" not in c and "I10RQ50d" not in c and "I10RQ50x" not in c and "small_gaprate" not in c and "_d.py" not in c and "prevcheck" not in c and "smoke" not in c
assert c.count("rollout \"I10R") == 2 and c.count("rollout I10RLcT12") == 1 and c.count("v8f_t1_eval_reader_rule_t.py") == 3
save("pod_taucurve10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_gaprate10k.sh")
c = rep(c, "LOGF=$I/chain_gaprate10k.log", "LOGF=$I/chain_taucurve10k.log")
c = rep(c, '"$I/pod_gaprate10k_queue.sh" "$I/v8f_t1_eval_reader_rule_d.py" "$I/check_prev_read.py" "$H/distrule/cmp_dumps.py"',
        '"$I/pod_taucurve10k_queue.sh" "$I/v8f_t1_eval_reader_rule_t.py" "$H/distrule/cmp_dumps.py"')
c = rep(c, "bash -n $B/pod_gaprate10k_queue.sh", "bash -n $B/pod_taucurve10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_gaprate10k_queue.sh > $W/queue_gaprate_stdout.txt", "nohup setsid bash $B/pod_taucurve10k_queue.sh > $W/queue_taucurve_stdout.txt")
c = rep(c, "GAPRATE10K", "TAUCURVE10K", 6, at_least=True)
c = rep(c, "for nm in I10RQ50d I10RQ50Rd I10RQ50x; do", "for nm in I10RLcT12 I10RLcT08 I10RQ50T08 I10RLcT10 I10RQ50T10 I10RLcT15 I10RQ50T15 I10RLcT20 I10RQ50T20; do")
c = rep(c, "$W/queue_i10k.log $W/smoke_gaprate/PREV_CHECK_smoke.json $W/k8_n2.log", "$W/queue_i10k.log $W/k8_n2.log")
c = rep(c, "small_gaprate", "small_taucurve", 2, at_least=True)
cc = code_lines(c)
assert "GAPRATE10K" not in cc and "small_gaprate" not in cc and "pod_gaprate10k" not in cc and "_d.py" not in cc
save("chain_taucurve10k.sh", c)
print("ZZ MAKE-TAUCURVE10K DONE")
