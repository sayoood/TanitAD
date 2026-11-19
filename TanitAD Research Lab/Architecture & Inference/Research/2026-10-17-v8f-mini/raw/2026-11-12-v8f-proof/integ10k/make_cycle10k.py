"""make_cycle10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD52 (CYCLE10K: the replanning cycle of model and rule at 1.0 s instead of 2.0 s) by asserted text edits:
  pod_taucurve10k_queue.sh -> pod_cycle10k_queue.sh   (the replan interval of a rollout comes from REPLAN, default 10 steps; a rerun of RLc with REPLAN=10 against the
                                                       banked dump, then R0, RLc and RQ50 at REPLAN=5 at two model seeds)
  chain_taucurve10k.sh     -> chain_cycle10k.sh
The wrapper is that of card ADD51 (v8f_t1_eval_reader_rule_t.py), unchanged. The originals are not changed. Usage: python make_cycle10k.py (in this folder)"""
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


q = load("pod_taucurve10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_cycle10k_queue.sh -- written by make_cycle10k.py from pod_taucurve10k_queue.sh. Card PREREG_V8F_PROOF_ADD52 (CYCLE10K): seven rollouts with the wrapper\n"
        "# v8f_t1_eval_reader_rule_t.py: RLcM10 (rerun of RLc at model seed 0 with the replan interval given as 10 steps, compared with the banked dump: control y1), then at a\n"
        "# replan interval of 5 steps (1.0 s) and two model seeds: no rule (R0M5), the label-fed rule with the closing speed (RLcM5), the reader-fed rule at level 0.50 (RQ50M5).\n"
        "# The folder and tag names keep the ending m10_causal_heads because the readers find the dumps by it; the interval of a run is in its run record. No training, no map fit.\n"
        "# Markers START_CYCLE10K.txt / CYCLE10K_STOP.txt / CYCLE10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD51: seventeen rollouts (the time gap of the rule at 0.8, 1.0, 1.5, 2.0 s, label-fed with the closing speed and reader-fed at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD52: seven rollouts (replan interval 5 steps for no rule, the label-fed rule with the closing speed and the reader-fed rule at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "TAUCURVE10K", "CYCLE10K", 4, at_least=True)
q = rep(q, "TMO=${I10_TIMEOUT:-2700}", "TMO=${I10_TIMEOUT:-5400}")
q = rep(q, "--replan-steps 10 ", "--replan-steps ${REPLAN:-10} ", 2)
q = rep(q, '  echo "arm $arm form $form model seed $m reader seed $r ckpt sha256', '  echo "arm $arm form $form replan steps ${REPLAN:-10} model seed $m reader seed $r ckpt sha256')
q = span(q, "TAU=1.2 rollout I10RLcT12 RLc 0 0", '    TAU=$tv QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout "I10RQ50T$tn" RR "$k" "$k"\n  done\ndone\n',
         'REPLAN=10 rollout I10RLcM10 RLc 0 0       # control y1 first: RLc with the interval given as 10 steps must equal the banked RLc\n'
         'if [ "$DRY" != 1 ] && ! done_step y1cmp; then\n'
         '  mkdir -p "$W/small_cycle"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLc_s0/dump_I10RLc_s0_m10_causal_heads" "$W/t1/I10RLcM10_s0/dump_I10RLcM10_s0_m10_causal_heads" > "$W/small_cycle/CMP_RLcM10.txt" 2>&1 || stop "Y1-RERUN-DIFFERS"\n'
         "  mark y1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  REPLAN=5 rollout I10R0M5 R0 "$k" "$k"\n'
         '  REPLAN=5 rollout I10RLcM5 RLc "$k" "$k"\n'
         '  REPLAN=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50M5 RR "$k" "$k"\n'
         "done\n")
c = code_lines(q)
assert "TAUCURVE10K" not in c and "I10RLcT" not in c and "I10RQ50T" not in c and "small_taucurve" not in c and "--replan-steps 10" not in c
assert c.count("rollout I10R") == 4 and c.count("${REPLAN:-10}") == 3 and c.count("v8f_t1_eval_reader_rule_t.py") == 3
save("pod_cycle10k_queue.sh", q)

c = load("chain_taucurve10k.sh")
c = rep(c, "LOGF=$I/chain_taucurve10k.log", "LOGF=$I/chain_cycle10k.log")
c = rep(c, '"$I/pod_taucurve10k_queue.sh" "$I/v8f_t1_eval_reader_rule_t.py"', '"$I/pod_cycle10k_queue.sh" "$I/v8f_t1_eval_reader_rule_t.py"')
c = rep(c, "bash -n $B/pod_taucurve10k_queue.sh", "bash -n $B/pod_cycle10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_taucurve10k_queue.sh > $W/queue_taucurve_stdout.txt", "nohup setsid bash $B/pod_cycle10k_queue.sh > $W/queue_cycle_stdout.txt")
c = rep(c, "TAUCURVE10K", "CYCLE10K", 6, at_least=True)
c = rep(c, "for nm in I10RLcT12 I10RLcT08 I10RQ50T08 I10RLcT10 I10RQ50T10 I10RLcT15 I10RQ50T15 I10RLcT20 I10RQ50T20; do", "for nm in I10RLcM10 I10R0M5 I10RLcM5 I10RQ50M5; do")
c = rep(c, "small_taucurve", "small_cycle", 2, at_least=True)
cc = code_lines(c)
assert "TAUCURVE10K" not in cc and "small_taucurve" not in cc and "pod_taucurve10k" not in cc
save("chain_cycle10k.sh", c)
print("ZZ MAKE-CYCLE10K DONE")
