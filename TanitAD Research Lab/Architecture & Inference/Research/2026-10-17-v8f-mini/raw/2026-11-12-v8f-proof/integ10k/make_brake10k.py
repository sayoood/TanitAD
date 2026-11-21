"""make_brake10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD57 (BRAKE10K: the braking limit of the lead rule, 8.0 and 10.0 m/s2 in place of 6.0, at the programme's
2.0 s cycle) by asserted text edits:
  v8f_t1_eval_reader_rule_a.py -> v8f_t1_eval_reader_rule_bm.py  (one more parameter, --b-max)
  pod_feas10k_queue.sh         -> pod_brake10k_queue.sh          (seven rollouts; starts after the queue of card ADD56)
  chain_feas10k.sh             -> chain_brake10k.sh
  podread_feas.sh              -> podread_brake.sh
  wait_podread_feas.sh         -> wait_podread_brake.sh
The reader read_brake10k.py is written by hand. The originals are not changed. Usage: python make_brake10k.py (in this folder)"""
import hashlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card (a file name can be taken)
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert "make_brake10k.py" in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes) sha256 %s" % (name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:70], c, n)
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


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_a.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py',
        '"""v8f_t1_eval_reader_rule_bm.py -- written by make_brake10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD57 (BRAKE10K): one more parameter,\n'
        '--b-max (m/s2): the braking limit of the lead rule (the hardest braking the cap may ask for), default the rule\'s own 6.0. Rule forms only. Text of the source below.\n'
        'v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py')
w = rep(w, '    ap.add_argument("--alim-log", default="")\n', '    ap.add_argument("--alim-log", default="")\n    ap.add_argument("--b-max", type=float, default=None)\n')
w = rep(w, '    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}\n',
        '    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}\n'
        '    if mine.b_max is not None:\n'
        '        if f == "R0" or not (2.0 <= mine.b_max <= 12.0):\n'
        '            raise SystemExit("REFUSED: --b-max needs a rule form and a value in [2.0, 12.0] m/s2")\n'
        '        rule_kw["b_max"] = float(mine.b_max)\n')
w = rep(w, '"B_MAX": R.B_MAX,', '"B_MAX": float(rule_kw.get("b_max", R.B_MAX)),')
assert w.index("f = mine.form") < w.index('rule_kw["b_max"]')
compile(w, "v8f_t1_eval_reader_rule_bm.py", "exec")
save("v8f_t1_eval_reader_rule_bm.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the pod queue
q = load("pod_feas10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_brake10k_queue.sh -- written by make_brake10k.py from pod_feas10k_queue.sh. Card PREREG_V8F_PROOF_ADD57 (BRAKE10K): seven rollouts at the\n"
        "# 2.0 s cycle with the wrapper v8f_t1_eval_reader_rule_bm.py: RLcB6 (braking limit given as 6.0, model seed 0: must equal the banked RLc, control b1), then the\n"
        "# label-fed and the reader-fed rule with the limit 8.0 (RLcB8, RQ50B8) and the label-fed rule with 10.0 (RLcB10, dose), each at two model seeds. No training.\n"
        "# It waits for the queue of card ADD56 to end. Markers START_BRAKE10K.txt / BRAKE10K_STOP.txt / BRAKE10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "FEAS10K", "BRAKE10K", 3)
q = rep_code(q, "v8f_t1_eval_reader_rule_a.py", "v8f_t1_eval_reader_rule_bm.py", 3)
q = rep(q, '  [ -n "${TAU:-}" ] && extra+=(--tau "$TAU")\n', '  [ -n "${TAU:-}" ] && extra+=(--tau "$TAU")\n  [ -n "${BMAX:-}" ] && extra+=(--b-max "$BMAX")\n')
RUNS = ('BMAX=6.0 rollout I10RLcB6 RLc 0 0       # control b1 first: the braking limit given as 6.0 must give the banked RLc\n'
        'if [ "$DRY" != 1 ] && ! done_step b1cmp; then\n'
        '  mkdir -p "$W/small_brake"\n'
        '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLc_s0/dump_I10RLc_s0_m10_causal_heads" "$W/t1/I10RLcB6_s0/dump_I10RLcB6_s0_m10_causal_heads" > "$W/small_brake/CMP_RLcB6.txt" 2>&1 || stop "B1-RERUN-DIFFERS"\n'
        '  mark b1cmp\n'
        'fi\n'
        'for k in "${ARMS[@]}"; do\n'
        '  BMAX=8.0 rollout I10RLcB8 RLc "$k" "$k"\n'
        '  BMAX=8.0 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50B8 RR "$k" "$k"\n'
        'done\n'
        'for k in "${ARMS[@]}"; do\n'
        '  BMAX=10.0 rollout I10RLcB10 RLc "$k" "$k"      # dose, names nothing\n'
        'done\n')
q = span(q, "AMAX=1e9 BLIM=1e9 REPLAN=10 rollout I10R0AID R0 0 0", "# the dose arm: half the acceleration limit, no rule\ndone\n", RUNS)
assert "AMAX=" not in "\n".join(ln for ln in q.split("\n") if not ln.lstrip().startswith("#") and "extra+=" not in ln and "AMAX:-" not in ln)
q = rep(q, 'log "queue N2 start (DRY=$DRY): card ADD53: eleven rollouts', 'log "queue N2 start (DRY=$DRY): card ADD57 (braking limit of the lead rule 8.0 and 10.0 m/s2, 2.0 s cycle): seven rollouts. Text of the source queue: eleven rollouts')
anchor = "for f in cmp_dumps.py v8f_t1_eval_reader_rule_bm.py LS_REF40_WINDOWS.json"
WAIT = ('if [ "$DRY" != 1 ]; then\n'
        '  t0=$(date +%s)\n'
        '  until [ -f "$W/FEASC10K_DONE.txt" ] || [ -f "$W/FEASC10K_STOP.txt" ]; do\n'
        '    [ $(( $(date +%s) - t0 )) -gt 28800 ] && stop "WAIT-FEASC-TIMEOUT"\n'
        '    sleep 30\n'
        '  done\n'
        '  log "the queue of card ADD56 has ended; the rollouts of card ADD57 start"\n'
        'fi\n')
q = rep(q, anchor, WAIT + anchor)
save("pod_brake10k_queue.sh", q)

# ---------------------------------------------------------------- the chain
c = load("chain_feas10k.sh")
c = rep(c, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# chain_brake10k.sh -- written by make_brake10k.py from chain_feas10k.sh (card PREREG_V8F_PROOF_ADD57). The text below is the original apart from the names.\n")
c = rep(c, "for nm in I10R0AID I10R0A5 I10RLcA5 I10RQ50A5 I10R0A10 I10R0A5L; do", "for nm in I10RLcB6 I10RLcB8 I10RQ50B8 I10RLcB10; do")
for old, new in (("pod_feas10k_queue.sh", "pod_brake10k_queue.sh"), ("v8f_t1_eval_reader_rule_a.py", "v8f_t1_eval_reader_rule_bm.py"), ("FEAS10K", "BRAKE10K"), ("small_feas", "small_brake"),
                 ("chain_feas10k.log", "chain_brake10k.log"), ("queue_feas_stdout", "queue_brake_stdout")):
    assert c.count(old) >= 1, old
    c = c.replace(old, new)
save("chain_brake10k.sh", c)

# ---------------------------------------------------------------- the pod read and its waiter
PAIRS = "I10RLcB8:I10RLc,I10RQ50B8:I10RQ50,I10RLcB10:I10RLc,I10RLcB10:I10RLcB8,I10RLcB8:I10R0,I10RQ50B8:I10R0,I10RQ50B8:I10RLcB8"
for src, dst, head in (("podread_feas.sh", "podread_brake.sh", "# podread_brake.sh -- written by make_brake10k.py from podread_feas.sh: the pod read of card PREREG_V8F_PROOF_ADD57 (BRAKE10K).\n"),
                       ("wait_podread_feas.sh", "wait_podread_brake.sh", "# wait_podread_brake.sh -- written by make_brake10k.py from wait_podread_feas.sh (card PREREG_V8F_PROOF_ADD57).\n")):
    t = load(src)
    if src == "podread_feas.sh":
        t, n1 = re.subn(r'PAIRS="[^"\n]*"', 'PAIRS="%s"' % PAIRS, t)
        t, n2 = re.subn(r'"\$PY" "\$P/root/integ10k/read_feas10k\.py"[^\n]*> "', '"$PY" "$P/root/integ10k/read_brake10k.py" --read-dir "$OUT/read_feas" --pull "$PL" --reused "$P/root/integ10k/TAUCURVE10K_REUSED.json" '
                      '--wrapper "$W/build/v8f_t1_eval_reader_rule_bm.py" --out "$OUT/read_feas/READ_FEAS10K.json" > "', t)
        assert n1 == 1 and n2 == 1, (n1, n2)
        t = rep(t, "CMP_R0AID.txt", "CMP_RLcB6.txt")
    for old, new in (("FEAS10K", "BRAKE10K"), ("PODREAD-FEAS", "PODREAD-BRAKE"), ("read_feas", "read_brake"), ("small_feas", "small_brake"), ("e1f", "e1h"), ("card ADD53", "card ADD57")):
        t = t.replace(old, new)
    t = rep(t, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n" + head)
    assert "feas" not in t.replace("podread_feas.sh", "").replace("wait_podread_feas.sh", "") and "A5" not in t and "TAUCURVE10K_REUSED.json" in t or dst.startswith("wait"), dst
    if dst.startswith("wait"):
        t = rep(t, '[ "$k" -gt 600 ]', '[ "$k" -gt 1200 ]')
        t = rep(t, "not ended after 5 h", "not ended after 10 h")
    save(dst, t)
print("wrapper sha256 %s" % WSHA)
