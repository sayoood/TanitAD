"""make_gaprate10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD50 (GAPRATE10K) by asserted text edits:
  v8f_t1_eval_reader_rule_h.py -> v8f_t1_eval_reader_rule_d.py  (one more form: RRd = reader presence and gap through the map, closing speed = slope times the change
                                                                 of the reader's own gap read over the last K cache steps where the reader is present at both ends, else 0)
  pod_closer10k_queue.sh       -> pod_gaprate10k_queue.sh       (a smoke of the new form, the rerun RQ50Rd of RQ50 at model seed 0 with its dump comparison, RQ50d at two
                                                                 model seeds each followed by the earlier-read check, RQ50x = level 0.50 with crossed reader seeds)
  chain_closer10k.sh           -> chain_gaprate10k.sh
The originals are not changed. Usage: python make_gaprate10k.py (in this folder)"""
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
w = load("v8f_t1_eval_reader_rule_h.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_h.py -- written by make_closer10k.py from v8f_t1_eval_reader_rule_c.py for card PREREG_V8F_PROOF_ADD49 (CLOSER10K): one more form,',
        '"""v8f_t1_eval_reader_rule_d.py -- written by make_gaprate10k.py from v8f_t1_eval_reader_rule_h.py for card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): one more form,\n'
        'RRd (reader presence and gap through the map; closing speed = slope times the change of the reader\'s own gap read over the last K cache steps, where the reader is\n'
        'present at both ends, else exactly 0; the earlier read uses the logged camera tokens of the cache steps j-K-4 .. j-K of the window\'s own clip, frames before the\n'
        'tick only; no model weight, no label). The eight forms of the parent are unchanged. Header of the parent:\n'
        'v8f_t1_eval_reader_rule_h.py -- written by make_closer10k.py from v8f_t1_eval_reader_rule_c.py for card PREREG_V8F_PROOF_ADD49 (CLOSER10K): one more form,')
w = rep(w, 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh")',
        'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh", "RRd")\nDT_S = 0.2                                          # seconds per cache step (the pinned tool: 30 steps = 6 s)')
w = rep(w, 'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "cols": {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}}',
        'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "slope": None, "k": 0, "prev": None, "zero": None,\n'
        '         "cols": {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}}')
w = rep(w, '    elif form in ("RR", "RRc", "RRh"):\n        used = np.minimum(', '    elif form in ("RR", "RRc", "RRh", "RRd"):\n        used = np.minimum(')
w = rep(w, 'def head_values_reader(model, feats, frame_ok=None, hist=None):',
        'def prev_read(states, rd, K):\n'
        '    """The reader at the earlier cache step j-K of each window: the tokens of the cache steps j-K-4 .. j-K of the window\'s own clip (the logged array the batch tokens\n'
        '    are gathered from; frames at or before the tick only). -> p [B], g [B] (NaN where the 5-frame history before j-K is not complete), ok [B]."""\n'
        '    B = len(states)\n'
        '    p, g, ok = np.full(B, np.nan, np.float32), np.full(B, np.nan, np.float32), np.zeros(B, bool)\n'
        '    rows, toks = [], []\n'
        '    for i, s in enumerate(states):\n'
        '        e = int(s.j) - int(K)\n'
        '        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):\n'
        '            rows.append(i)\n'
        '            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))\n'
        '    if rows:\n'
        '        pp, gg = rd.read(np.stack(toks))\n'
        '        p[rows], g[rows], ok[rows] = pp, gg, True\n'
        '    return p, g, ok\n\n\n'
        'def gap_rate_closing(present, g, p_prev, g_prev, ok_prev, slope, K, p_min=P_MIN):\n'
        '    """-> (have [B] bool, c_fed [B] float32, m/s). have: the reader is present now and at the cache step j-K (full history there, probability >= p_min).\n'
        '    c_fed = slope * (g - g_prev) / (DT_S * K) where have (the gap rate = lead speed minus ego speed), else exactly 0."""\n'
        '    g, gp = np.asarray(g, np.float64), np.asarray(g_prev, np.float64)\n'
        '    with np.errstate(invalid="ignore"):\n'
        '        have = np.asarray(present, bool) & np.asarray(ok_prev, bool) & (np.asarray(p_prev, np.float64) >= p_min) & np.isfinite(g) & np.isfinite(gp)\n'
        '        raw = (g - gp) / (DT_S * float(K))\n'
        '    return have, np.where(have, float(slope) * raw, 0.0).astype(np.float32)\n\n\n'
        'def head_values_reader(model, feats, frame_ok=None, hist=None):')
w = rep(w, '    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):\n        STATE["cols"][k].append(np.asarray(v))\n',
        '    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):\n        STATE["cols"][k].append(np.asarray(v))\n'
        '    c_fed = np.zeros(len(p), np.float32)\n'
        '    if STATE["form"] == "RRd":\n'
        '        if STATE["prev"] is None or STATE["zero"] is None or len(STATE["prev"][0]) != len(p) or len(STATE["zero"][0]) != len(p):\n'
        '            raise RuntimeError("the earlier read does not belong to this batch")\n'
        '        pp, gp, okp = STATE["prev"]\n'
        '        p0, g0, ok0 = STATE["zero"]\n'
        '        have, c_fed = gap_rate_closing(present, g, pp, gp, okp, STATE["slope"], STATE["k"])\n'
        '        okb = np.asarray(ok, bool)\n'
        '        with np.errstate(invalid="ignore"):                         # the gather at K = 0 must give the read of the batch tokens: logged per tick (control d4)\n'
        '            d0 = np.maximum(np.abs(np.asarray(g, np.float64) - g0), np.abs(np.asarray(p, np.float64) - p0))\n'
        '        id0 = np.where(okb & ok0, d0, np.where(okb == ok0, 0.0, np.inf))\n'
        '        for k, v in (("p_prev", pp), ("g_prev", gp), ("ok_prev", okp), ("have", have), ("c_fed", c_fed), ("id0", id0)):\n'
        '            STATE["cols"].setdefault(k, []).append(np.asarray(v))\n'
        '        STATE["prev"] = STATE["zero"] = None\n')
w = rep(w, '    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else np.zeros(len(p), np.float32)), ok\n',
        '    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else c_fed), ok\n')
w = rep(w, '        if self.source == "mix":\n            self.label = X.STATE["label"]\n',
        '        if self.source == "mix":\n            self.label = X.STATE["label"]\n\n'
        '    def __call__(self, batch):\n'
        '        if STATE["form"] == "RRd":\n'
        '            if self.states is None:\n'
        '                raise RuntimeError("ReaderPlanner not bound to the states of this batch")\n'
        '            STATE["prev"] = prev_read(self.states, STATE["reader"], STATE["k"])\n'
        '            STATE["zero"] = prev_read(self.states, STATE["reader"], 0)\n'
        '        return X.MixPlanner.__call__(self, batch)\n')
w = rep(w, '    ap.add_argument("--rule-log", default="")\n',
        '    ap.add_argument("--rule-log", default="")\n    ap.add_argument("--slope", type=float, default=None)\n    ap.add_argument("--k-diff", type=int, default=0)\n')
w = rep(w, '    f = mine.form\n    if f == "R0":',
        '    f = mine.form\n    if f != "RRd" and (mine.slope is not None or mine.k_diff):\n        raise SystemExit("REFUSED: --slope / --k-diff belong to form RRd only")\n    if f == "R0":')
w = rep(w, '    if f in GAP_PLANT or f in ("RR", "RRc", "RRh"):', '    if f in GAP_PLANT or f in ("RR", "RRc", "RRh", "RRd"):')
w = rep(w, '        if f in ("RR", "RRc", "RRh"):\n', '        if f in ("RR", "RRc", "RRh", "RRd"):\n')
w = rep(w, '        STATE["reader"] = TR.TokReader.from_ckpt(mine.reader)\n',
        '        STATE["reader"] = TR.TokReader.from_ckpt(mine.reader)\n'
        '        if f == "RRd":\n'
        '            if mine.slope is None or not (0.0 < mine.slope <= 2.0) or mine.k_diff < 1:\n'
        '                raise SystemExit("REFUSED: form RRd needs --slope in (0, 2] and --k-diff >= 1")\n'
        '            STATE["slope"], STATE["k"] = float(mine.slope), int(mine.k_diff)\n')
w = rep(w, '        mix = ("head", "head", {"RRc": "label", "RRh": "head"}.get(f, "zero"))', '        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head"}.get(f, "zero"))')
w = rep(w, '"mode_kwargs": "{}"}', '"mode_kwargs": "{}", "k_diff": int(STATE["k"] or 0), "slope": float(STATE["slope"] or 0.0), "dt_s": DT_S}')
save("v8f_t1_eval_reader_rule_d.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_closer10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_gaprate10k_queue.sh -- written by make_gaprate10k.py from pod_closer10k_queue.sh. Card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): a smoke of the form RRd on 96\n"
        "# windows, then five rollouts with the wrapper v8f_t1_eval_reader_rule_d.py: RQ50Rd (rerun of RQ50 at model seed 0, compared with the banked dump: control d1),\n"
        "# RQ50d at two model seeds (each followed by check_prev_read.py), RQ50x at two model seeds (level 0.50 with the reader and the map of the other seed). No training, no map fit.\n"
        "# Markers START_GAPRATE10K.txt / GAPRATE10K_STOP.txt / GAPRATE10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD49: three rollouts with the closing speed of the reader; reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD50: a smoke and five rollouts (closing speed from the change of the reader gap; crossed reader seeds at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "CLOSER10K", "GAPRATE10K", 4, at_least=True)
q = rep(q, "v8f_t1_eval_reader_rule_h.py", "v8f_t1_eval_reader_rule_d.py", 3, at_least=True)
q = rep(q, "for f in cmp_dumps.py ", "for f in check_prev_read.py cmp_dumps.py ")
q = rep(q, "    RR|RRc|RRh|RRid|RR0) extra+=(--reader", "    RR|RRc|RRh|RRd|RRid|RR0) extra+=(--reader")
q = rep(q, '  case "$form" in RR|RRc|RRh) extra+=(--qmap', '  [ "$form" = RRd ] && extra+=(--slope "${SLOPE:?}" --k-diff "$KDIFF")\n  case "$form" in RR|RRc|RRh|RRd) extra+=(--qmap')
q = span(q, 'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50R RR 0 0', '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50h RRh "$k" "$k"\ndone\n',
         'KDIFF=25\n'
         'SLOPES=(0.7708154241032367 0.7831096226098508)    # slope_used at k = 25 of read_closer/CLOSING_DIFF_T0.json, reader seeds 0 and 1 (fitted on the fit clips of the training corpus)\n'
         'prevcheck() { # RULELOG READERSEED OUTJSON : identities of the run and the earlier reads recomputed from the cache files by the reader\'s own clip path (CPU)\n'
         '  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" "$B/check_prev_read.py" --rule-log "$1" --reader "$W/lr16/TK16_s$2/ckpt_last.pt" --eval-cache "$EVALC" --join "$JOINE" --k "$KDIFF" --build "$B" --out "$3"\n'
         '}\n'
         '# ---- 0 smoke of the new form on a subset of the grid: the queue stops before any rollout of the card if the form does not run or its identities do not hold\n'
         'if [ "$DRY" = 1 ]; then echo "SMOKE: form RRd on 96 windows, then check_prev_read.py"; elif ! done_step smoke_d; then\n'
         '  S=$W/smoke_gaprate; [ -e "$S" ] && stop "SMOKE-FOLDER-EXISTS"\n'
         '  wait_slot "smoke RRd"\n'
         '  mkdir -p "$S/ticks"\n'
         '  DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 900 "$PY" "$B/v8f_t1_eval_reader_rule_d.py" --tree "$TREE" --form RRd --reader "$W/lr16/TK16_s0/ckpt_last.pt" --qmap "$W/qmap_sweep/QMAP_LR16_s0_q50.json" --slope "${SLOPES[0]}" --k-diff "$KDIFF" --rule-log "$S/ticks/rule_log_smoke.npz" run --ckpt "$W/ckpt/UE_s0/A/ckpt_last.pt" --tag smoke_gaprate --tracks "$TRACKS" --eval-cache "$EVALC" --eval-grid "$GRID" --eval-join "$JOINE" --out-dir "$S" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$S/ticks" --dump-heads --device cuda --max-windows 96 > "$S/smoke.log" 2>&1 || stop "SMOKE-RC"\n'
         '  prevcheck "$S/ticks/rule_log_smoke.npz" 0 "$S/PREV_CHECK_smoke.json" > "$S/prevcheck.log" 2>&1 || stop "SMOKE-PREVCHECK"\n'
         '  mark smoke_d\n'
         '  log "smoke of form RRd passed"\n'
         'fi\n'
         'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50Rd RR 0 0       # control d1 first: the rerun of RQ50 under the new wrapper\n'
         'if [ "$DRY" != 1 ] && ! done_step d1cmp; then\n'
         '  mkdir -p "$W/small_gaprate"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RQ50_s0/dump_I10RQ50_s0_m10_causal_heads" "$W/t1/I10RQ50Rd_s0/dump_I10RQ50Rd_s0_m10_causal_heads" > "$W/small_gaprate/CMP_RQ50Rd.txt" 2>&1 || stop "D1-RERUN-DIFFERS"\n'
         "  mark d1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" SLOPE="${SLOPES[$k]}" rollout I10RQ50d RRd "$k" "$k"\n'
         '  if [ "$DRY" != 1 ] && ! done_step "prev_$k"; then\n'
         '    prevcheck "$W/t1/I10RQ50d_s$k/ticks/rule_log_I10RQ50d_s$k.npz" "$k" "$W/small_gaprate/PREV_CHECK_s$k.json" > "$W/small_gaprate/prevcheck_s$k.log" 2>&1 || stop "PREVCHECK s$k"\n'
         '    mark "prev_$k"\n'
         "  fi\n"
         "done\n"
         'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s1_q50.json" rollout I10RQ50x RR 0 1       # level 0.50 with crossed reader seeds: model seed 0 with reader seed 1 and that reader\'s map\n'
         'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s0_q50.json" rollout I10RQ50x RR 1 0       # model seed 1 with reader seed 0 and that reader\'s map\n')
c = code_lines(q)
assert "CLOSER10K" not in c and "I10RQ50h" not in c and "small_closer" not in c and "_h.py" not in c and "RRh \"$k\"" not in c
assert c.count("rollout I10R") == 4 and c.count("v8f_t1_eval_reader_rule_d.py") == 4 and c.count("check_prev_read.py") == 3
save("pod_gaprate10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
c = load("chain_closer10k.sh")
c = rep(c, "LOGF=$I/chain_closer10k.log", "LOGF=$I/chain_gaprate10k.log")
c = rep(c, '"$I/pod_closer10k_queue.sh" "$I/v8f_t1_eval_reader_rule_h.py" "$H/distrule/cmp_dumps.py"',
        '"$I/pod_gaprate10k_queue.sh" "$I/v8f_t1_eval_reader_rule_d.py" "$I/check_prev_read.py" "$H/distrule/cmp_dumps.py"')
c = rep(c, "bash -n $B/pod_closer10k_queue.sh", "bash -n $B/pod_gaprate10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_closer10k_queue.sh > $W/queue_closer_stdout.txt", "nohup setsid bash $B/pod_gaprate10k_queue.sh > $W/queue_gaprate_stdout.txt")
c = rep(c, "CLOSER10K", "GAPRATE10K", 6, at_least=True)
c = rep(c, "for nm in I10RQ50h I10RQ50R; do", "for nm in I10RQ50d I10RQ50Rd I10RQ50x; do")
c = rep(c, "$W/queue_i10k.log $W/k8_n2.log", "$W/queue_i10k.log $W/smoke_gaprate/PREV_CHECK_smoke.json $W/k8_n2.log")
c = rep(c, "small_closer", "small_gaprate", 2, at_least=True)
cc = code_lines(c)
assert "CLOSER10K" not in cc and "small_closer" not in cc and "pod_closer10k" not in cc and "_h.py" not in cc
save("chain_gaprate10k.sh", c)
print("ZZ MAKE-GAPRATE10K DONE")
