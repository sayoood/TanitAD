"""make_n2.py -- writes the continuation files of card PREREG_V8F_PROOF_ADD46 in the form of its note N2 from the originals, by asserted text edits:
  pod_integ10k_queue.sh -> pod_integ10k_queue_n2.sh   (no reader training: the reader arms of the first queue are checked by sha256; the train-log step is check_k8_n2.py;
                                                       own START / STOP / DONE markers; every other step is the text of the original)
  chain_i10k.sh         -> chain_i10k_n2.sh           (pushes the new files, starts the N2 queue, waits on the N2 markers, pulls the small files into pull/small_n2)
  read_i10k.py          -> read_i10k_n2.py            (small files from pull/small_n2; the train-log part of k8 is K8_N2.json; the tool check is reported beside)
  read_i10k.sh          -> read_i10k_n2.sh            (calls read_i10k_n2.py)
The originals are not changed. Every edit asserts how often its old text occurs. Usage: python make_n2.py   (run in this folder)"""
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


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start, text.count(start), end, text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


# ---------------------------------------------------------------- the pod queue
q = load("pod_integ10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_integ10k_queue_n2.sh -- written by make_n2.py from pod_integ10k_queue.sh. Continuation of card PREREG_V8F_PROOF_ADD46 by its note N2:\n"
        "# the reader arms are those the first queue trained (no training here; checkpoints checked by sha256), the train-log step is check_k8_n2.py, own markers\n"
        "# START_I10K_N2.txt / INTEG10K_N2_STOP.txt / INTEG10K_N2_DONE.txt. The text below is the original apart from these edits.\n")
q = rep(q, "START_I10K.txt", "START_I10K_N2.txt", 2, at_least=True)
q = rep(q, "INTEG10K_STOP.txt", "INTEG10K_N2_STOP.txt", 1, at_least=True)
q = rep(q, "INTEG10K_DONE.txt", "INTEG10K_N2_DONE.txt", 1, at_least=True)
q = rep(q, 'log "queue start (DRY=$DRY)"', 'log "queue N2 start (DRY=$DRY): continuation of card ADD46 by note N2; the reader arms are those of the first queue"')
q = rep(q, "for f in LS_REF40_WINDOWS.json reader_controls.py", "for f in check_k8_n2.py ckpt_sha_s0.txt ckpt_sha_s1.txt LS_REF40_WINDOWS.json reader_controls.py")
q = span(q, "if ! done_step train; then\n", "  mark train\nfi\n",
         "if ! done_step train_n2; then          # note N2: no training; the reader arms of the first queue, checked against the sha256 its pull recorded\n"
         '  [ "$DRY" = 1 ] || [ -f "$W/lr16/TRAIN_DONE.json" ] || stop "TRAIN-NO-DONE-FILE"\n'
         '  for k in "${ARMS[@]}"; do\n'
         '    if [ "$DRY" = 1 ]; then echo "CHECK: sha256 of $W/lr16/TK16_s$k/ckpt_last.pt equals $B/ckpt_sha_s$k.txt"; continue; fi\n'
         "    want=$(tr -cd 'a-f0-9' < \"$B/ckpt_sha_s$k.txt\"); got=$(sha256sum \"$W/lr16/TK16_s$k/ckpt_last.pt\" | cut -c1-64)\n"
         '    [ ${#want} = 64 ] && [ "$want" = "$got" ] || stop "READER-CKPT-SHA s$k"\n'
         "  done\n"
         "  mark train_n2\n"
         "fi\n")
q = span(q, "if ! done_step checklog; then\n", "  mark checklog\nfi\n",
         "if ! done_step k8n2; then              # note N2: the tool check is run and recorded (its rc does not stop the queue); check_k8_n2.py decides\n"
         '  [ "$DRY" = 1 ] || [ -f "$W/diag/out/DIAG_TAIL3.json" ] || stop "NO-DIAG-TAIL3"\n'
         '  runlog "$W/lr16_checklog.txt" timeout 600 "$PY" "$B/tok_probe_lr.py" check-log --run-dir "$W/lr16" --arms TK16_s0,TK16_s1\n'
         '  runlog "$W/k8_n2.log" timeout 600 "$PY" "$B/check_k8_n2.py" --run-dir "$W/lr16" --diag "$W/diag/out/DIAG_TAIL3.json" --out "$W/rows/K8_N2.json" || stop "K8-N2-FAIL"\n'
         "  mark k8n2\n"
         "fi\n")
assert "tok_probe_lr.py\" train " not in q and "done_step train;" not in q and "CHECKLOG-FAIL" not in q
save("pod_integ10k_queue_n2.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_i10k.sh")
c = rep(c, "LOGF=$I/chain_i10k.log", "LOGF=$I/chain_i10k_n2.log")
c = rep(c, '"$I/pod_integ10k_queue.sh"', '"$I/pod_integ10k_queue_n2.sh" "$I/check_k8_n2.py" "$I/pull/small/ckpt_sha_s0.txt" "$I/pull/small/ckpt_sha_s1.txt"')
c = rep(c, "bash -n $B/pod_integ10k_queue.sh", "bash -n $B/pod_integ10k_queue_n2.sh")
c = rep(c, "nohup setsid bash $B/pod_integ10k_queue.sh > $W/queue_stdout.txt", "nohup setsid bash $B/pod_integ10k_queue_n2.sh > $W/queue_n2_stdout.txt")
c = rep(c, "START_I10K.txt", "START_I10K_N2.txt", 3, at_least=True)
c = rep(c, "INTEG10K_STOP", "INTEG10K_N2_STOP", 2, at_least=True)
c = rep(c, "INTEG10K_DONE", "INTEG10K_N2_DONE", 1, at_least=True)
c = rep(c, "CHAIN-I10K", "CHAIN-I10K-N2", 3, at_least=True)
c = rep(c, "$W/queue_i10k.log $W/small/", "$W/queue_i10k.log $W/k8_n2.log $W/diag/out/DIAG_TAIL.json $W/diag/out/DIAG_TAIL2.json $W/diag/out/DIAG_TAIL3.json $W/small/")
c = rep(c, "$W/small", "$W/small_n2", 5, at_least=True)
c = rep(c, 'pull_dir small_n2 "$W/small_n2" "$PULL/small"', 'pull_dir small_n2 "$W/small_n2" "$PULL/small_n2"') if 'pull_dir small_n2 "$W/small_n2" "$PULL/small"' in c else \
    rep(c, 'pull_dir small "$W/small_n2" "$PULL/small"', 'pull_dir small_n2 "$W/small_n2" "$PULL/small_n2"')
assert "pod_integ10k_queue.sh" not in c.replace("pod_integ10k_queue.sh ->", "").split("FILES=(")[1], "the first queue is still named after FILES"
save("chain_i10k_n2.sh", c)

# ---------------------------------------------------------------- the reader
r = load("read_i10k.py")
r = rep(r, 'sm = os.path.join(a.pull, "small")', 'sm = os.path.join(a.pull, "small_n2")')
old = '            inp["train"][k] = {"steps": td.get("steps"), "p5_all_pass": bool(td.get("p5_all_pass")) and bool((td.get("p5") or {}).get("TK16_s%d" % k, {}).get("pass", True))}\n'
r = rep(r, old, old + '            inp["train"][k]["k8_n2_pass"] = bool((jl(os.path.join(sm, "K8_N2.json")) or {}).get("PASS") is True)   # note N2: the train-log part of k8\n')
r = rep(r, 'tr[k].get("p5_all_pass") is True for k in SEEDS', 'tr[k].get("k8_n2_pass") is True for k in SEEDS')
r = rep(r, 'C["k8_counts_and_train_log"] = {"n_windows_total"',
        'C["k8_counts_and_train_log"] = {"train_log_part": "read by note N2 of the card (K8_N2.json, key k8_n2_pass); p5_all_pass is the tool check on every training step: reported, not used", "n_windows_total"')
save("read_i10k_n2.py", r)

s = load("read_i10k.sh")
s = rep(s, "read_i10k.py", "read_i10k_n2.py", 2, at_least=True)
s = rep(s, "READ-I10K", "READ-I10K-N2", 1, at_least=True)
save("read_i10k_n2.sh", s)
print("ZZ MAKE-N2 DONE")
