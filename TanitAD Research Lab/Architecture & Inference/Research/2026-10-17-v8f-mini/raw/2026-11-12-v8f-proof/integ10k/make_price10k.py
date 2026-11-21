"""make_price10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD47 (PRICE10K) from those of card ADD46 by asserted text edits:
  qmap_fit.py               -> qmap_fit_lvl.py        (the level from the environment variable QMAP_LEVEL; the tool's own band becomes 0 .. level + 0.05; card name)
  pod_integ10k_queue_n2.sh  -> pod_price10k_queue.sh  (no reader step; maps at 0.35 and 0.50 and their check; six rollouts with the map file as a parameter; own markers)
  chain_i10k_n2.sh          -> chain_price10k.sh      (pushes the new files, starts the queue, waits on its markers, pulls the six run folders and the small files)
and writes QMAP_IDENTITY.json. The originals are not changed. Usage: python make_price10k.py (in this folder)"""
import io
import json
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


# ---------------------------------------------------------------- the map tool
m = load("qmap_fit.py")
m = rep(m, "Q_LEVEL = 20.0\n", 'Q_LEVEL = float(os.environ["QMAP_LEVEL"])          # card ADD47: the level of the sweep (percent), given by the queue\n')
m = rep(m, "BAND = (0.15, 0.25)\n", "BAND = (0.0, Q_LEVEL / 100.0 + 0.05)                 # card ADD47 control m1: coverage at most the level + 0.05\n")
m = rep(m, '"card": "PREREG_V8F_PROOF_ADD46"', '"card": "PREREG_V8F_PROOF_ADD47"')
save("qmap_fit_lvl.py", m)

with io.open(os.path.join(HERE, "QMAP_IDENTITY.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"arm": "IDENTITY", "card": "PREREG_V8F_PROOF_ADD47", "what": "identity map: g_used = min(g, Q(g)) = g for every g", "xs": [1.0, 150.0], "ys": [1.0, 150.0]}, fh, indent=1, sort_keys=True)
print("wrote QMAP_IDENTITY.json")

# ---------------------------------------------------------------- the pod queue
q = load("pod_integ10k_queue_n2.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_price10k_queue.sh -- written by make_price10k.py from pod_integ10k_queue_n2.sh. Card PREREG_V8F_PROOF_ADD47 (PRICE10K): maps at the levels 0.35 and 0.50\n"
        "# from the banked reader rows, their check, then six rollouts (RQ50, RQ35, RRaw per model seed) with the map file as a parameter. No training, no reader step.\n"
        "# Markers START_PRICE10K.txt / PRICE10K_STOP.txt / PRICE10K_DONE.txt. The header lines below are those of the first queue.\n")
q = rep(q, "START_I10K_N2.txt", "START_PRICE10K.txt", 2, at_least=True)
q = rep(q, "INTEG10K_N2_STOP.txt", "PRICE10K_STOP.txt", 1, at_least=True)
q = rep(q, "INTEG10K_N2_DONE.txt", "PRICE10K_DONE.txt", 1, at_least=True)
q = rep(q, 'mkdir -p "$W/markers" "$W/lr16" "$W/rows" "$W/qmap" "$W/t1"', 'mkdir -p "$W/markers" "$W/lr16" "$W/rows" "$W/qmap" "$W/qmap_sweep" "$W/t1"')
q = rep(q, "for f in check_k8_n2.py ckpt_sha_s0.txt ckpt_sha_s1.txt LS_REF40_WINDOWS.json", "for f in qmap_fit_lvl.py price_maps_check.py QMAP_IDENTITY.json LS_REF40_WINDOWS.json")
q = rep(q, 'continuation of card ADD46 by note N2; the reader arms are those of the first queue"', 'card ADD47 (PRICE10K): maps of the sweep and six rollouts; reader arms and checkpoints of card ADD46"')
q = span(q, "# ---------------------------------------------------------------- 1 reader training (GPU)\n", "# ---------------------------------------------------------------- 5 the twelve rollouts\n",
         "# ---------------------------------------------------------------- 1 maps of the sweep (CPU) and their check (control m1)\n"
         'for k in "${ARMS[@]}"; do\n'
         '  [ "$DRY" = 1 ] || [ -f "$W/rows/rows_LR16_s$k.npz" ] || stop "NO-READER-ROWS s$k"\n'
         "  for L in 35 50; do\n"
         '    if ! done_step "pmap${k}_$L"; then\n'
         '      runlog "$W/qmap_sweep.log" env QMAP_LEVEL=$L timeout 600 "$PY" "$B/qmap_fit_lvl.py" --rows "$W/rows/rows_LR16_s$k.npz" --arm "LR16_s${k}_q$L" --out-dir "$W/qmap_sweep" || stop "QMAP-SWEEP-FAIL s$k q$L"\n'
         '      mark "pmap${k}_$L"\n'
         "    fi\n"
         "  done\n"
         "done\n"
         "if ! done_step pmapcheck; then\n"
         '  runlog "$W/qmap_sweep.log" timeout 600 "$PY" "$B/price_maps_check.py" --q20-dir "$W/qmap" --sweep-dir "$W/qmap_sweep" --identity "$B/QMAP_IDENTITY.json" --out "$W/qmap_sweep/PRICE_MAPS_CHECK.json" || stop "MAPS-CHECK-FAIL"\n'
         "  mark pmapcheck\n"
         "fi\n\n", keep_end=True)
q = rep(q, '[ "$form" = RR ] && extra+=(--qmap "$W/qmap/QMAP_LR16_s$r.json")', '[ "$form" = RR ] && extra+=(--qmap "${QMAP_FILE:?}")          # card ADD47: the map file is given by the caller')
q = span(q, 'for k in "${ARMS[@]}"; do rollout I10R0 R0 "$k" "$k"; done', "rollout I10RRx RR 1 0\n",
         'for k in "${ARMS[@]}"; do\n'
         '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50 RR "$k" "$k"      # the deciding level first\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q35.json" rollout I10RQ35 RR "$k" "$k"\n'
         '  QMAP_FILE="$B/QMAP_IDENTITY.json" rollout I10RRaw RR "$k" "$k"\n'
         "done\n")
code = "\n".join(ln for ln in q.splitlines() if not ln.lstrip().startswith("#"))          # the header comments of the first queues stay; the checks below read code lines only
assert "check_k8_n2" not in code and "tok_reader.py\" extract" not in code and "I10RRx" not in code and "QMAP_LR16_s$r.json" not in code
save("pod_price10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_i10k_n2.sh")
c = rep(c, "LOGF=$I/chain_i10k_n2.log", "LOGF=$I/chain_price10k.log")
c = rep(c, '"$I/pod_integ10k_queue_n2.sh" "$I/check_k8_n2.py" "$I/pull/small/ckpt_sha_s0.txt" "$I/pull/small/ckpt_sha_s1.txt"', '"$I/pod_price10k_queue.sh" "$I/qmap_fit_lvl.py" "$I/price_maps_check.py" "$I/QMAP_IDENTITY.json"')
c = rep(c, "bash -n $B/pod_integ10k_queue_n2.sh", "bash -n $B/pod_price10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_integ10k_queue_n2.sh > $W/queue_n2_stdout.txt", "nohup setsid bash $B/pod_price10k_queue.sh > $W/queue_price_stdout.txt")
c = rep(c, "START_I10K_N2.txt", "START_PRICE10K.txt", 3, at_least=True)
c = rep(c, "INTEG10K_N2_STOP", "PRICE10K_STOP", 2, at_least=True)
c = rep(c, "INTEG10K_N2_DONE", "PRICE10K_DONE", 1, at_least=True)
c = rep(c, "CHAIN-I10K-N2", "CHAIN-PRICE10K", 3, at_least=True)
c = rep(c, "for nm in I10R0 I10RLb I10RR I10RRid I10RR0; do", "for nm in I10RQ50 I10RQ35 I10RRaw; do")
c = rep(c, "cp -f $W/qmap/* ", "cp -f $W/qmap_sweep/* $W/qmap_sweep.log ")
c = rep(c, "$W/small_n2", "$W/small_price", 5, at_least=True)
c = rep(c, 'pull_dir small_n2 "$W/small_price" "$PULL/small_n2"', 'pull_dir small_price "$W/small_price" "$PULL/small_price"')
save("chain_price10k.sh", c)
print("ZZ MAKE-PRICE10K DONE")
