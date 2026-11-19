"""make_rulefast10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K: the lead rule applied again inside the 2.0 s block, the model still planning every
2.0 s, no training) by asserted text edits (each old text exactly once, or the counted number of code lines):
  v8f_t1_eval_reader_rule_rz.py -> v8f_t1_eval_reader_rule_rzf.py  (one more parameter, --rule-every h; default 10 = the rule at the tick only = the wrapper rz unchanged)
  pod_causlim10k_queue.sh       -> pod_rulefast10k_queue.sh        (tool check, control RLcFID, then RLcF5, RQ50F5, RLcF2 at two model seeds)
  chain_causlim10k.sh           -> chain_rulefast10k.sh
  podread_causlim.sh            -> podread_rulefast.sh
  wait_podread_causlim.sh       -> wait_podread_rulefast.sh
The originals are not changed. Pure text generator (no numpy, no torch). Usage: python make_rulefast10k.py (in this folder)"""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_rulefast10k.py"


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert ME in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes) sha256 %s" % (name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:70], c, n)
    return text.replace(old, new)


def rep_code(text, old, new, n):
    """replaces in the lines that are not comments; n = the number of lines that must change"""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert k == n, "code anchor %r changed %d lines, expected %d" % (old, k, n)
    return "\n".join(out)


def rep_all_code(text, old, new):
    """replaces in the lines that are not comments (at least one line must change)"""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert k >= 1, "code anchor %r not found" % old
    return "\n".join(out)


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


def code_lines(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# ---------------------------------------------------------------- the wrapper
MID = r'''# ---------------------------------------------------------------- the later looks of the rule inside the block (card PREREG_V8F_PROOF_ADD62, RULEFAST10K)
MID_FIELDS = tuple(W.LOG_FIELDS) + ("q", "excess", "nchg")
STATE["every"] = 10                                    # --rule-every; 10 (or any value at or above the replan interval) = the rule at the tick only
STATE["T"] = None                                      # the pinned tool module (plan_to_world, envelope), set by main()
STATE["mid"] = {"rows": {f: [] for f in MID_FIELDS}, "rd": {k: [] for k in ("p", "g_read", "g_used", "ok")}, "p1": {"ci": [], "t0": [], "plan": []}, "env": {"plans": 0, "hits": 0}}


def _frame_ok(s, e, hist):
    """frame_ok of the last `hist` frames e-hist+1 .. e as the item builder gives it at a tick on frame e: real frames only (idx >= 0) and, for the rows already in the window
    state (idx <= s.j), a finite p10f row (or idx < 1). Rows after s.j are the executed rows a real tick on frame e would have; they are finite by construction."""
    idx = np.arange(e - hist + 1, e + 1)
    past = idx <= int(s.j)
    idp = np.clip(idx, 0, len(s.d["p10f"]) - 1)
    fin = np.isfinite(np.asarray(s.d["p10f"])[idp]).all(axis=1) | (idx < 1)
    return bool(((idx >= 0) & (~past | fin)).all())


def _reader_at(states, rd, e_off):
    """The token reader at the logged frame j + e_off of each window (cache steps e-hist+1 .. e of the window's own clip, the array the batch tokens are gathered from; frames
    at or before e only). -> p [B], g [B] (float64, NaN where there is no complete history in the clip), ok [B] (full history of real frames)."""
    B = len(states)
    p, g, ok = np.full(B, np.nan), np.full(B, np.nan), np.zeros(B, bool)
    rows, toks = [], []
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):
            rows.append(i)
            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))
            ok[i] = _frame_ok(s, e, rd.hist)
    if rows:
        pp, gg = rd.read(np.stack(toks))
        p[rows], g[rows] = pp, gg
    return p, g, ok


def mid_looks(pl, plan):
    """The rule again, for every later step q = h, 2h, ... (< m), on the part of the plan that is still to be driven. `plan` is the plan after the rule at the tick (the base planner
    followed by X.MixPlanner); `pl` is the ReaderPlanner (states, m, kw). The car is at knot q of the plan built so far (the executor follows the plan exactly in this tier); the
    rule inputs are those a tick on frame j + q takes (same code as X.MixPlanner: label read at j + q, or the reader on the logged tokens of j + q; logged speed and logged arc at
    that step; the car's progress at knot q; the car's speed = the chord speed of the last step before q); the remaining knots are taken relative to knot q; the executed-step
    count is m - q. Steps up to q are never changed; an unchanged later knot stays the original float32 value. Returns the new plan (a copy)."""
    st, m, h = pl.states, int(pl.m), int(STATE["every"])
    T = STATE["T"]
    B = len(st)
    out = np.array(plan, copy=True)
    ci = np.array([s.ci for s in st], np.int64)
    t0 = np.array([s.t0 for s in st], np.int64)
    j = np.array([s.j for s in st], np.int64)
    k = j - t0
    ar = np.arange(B)
    form, mix, label, rd, M = STATE["form"], X.MIX, X.STATE["label"], STATE["reader"], STATE["mid"]
    first = k == 0
    if first.any():                                    # the plan after the rule at the first tick: the reference path of control k4 (second part)
        M["p1"]["ci"].append(ci[first].copy())
        M["p1"]["t0"].append(t0[first].copy())
        M["p1"]["plan"].append(np.array(plan, copy=True)[first])
    g_log = np.stack([np.asarray(s.d["gt"][s.t0], np.float64) for s in st])
    for q in range(h, m, h):
        P = out.astype(np.float64)
        kq, jq = k + q, j + q
        pos = np.empty((B, 1, 2))                      # the car at knot q in the t0 ego frame: the arithmetic of floors_kin.gt_waypoints on the world position of the knot
        for b, s in enumerate(st):
            wxy = T.plan_to_world(P[b, q - 1:q], np.asarray(s.d["pose"][int(s.j)], np.float64))[0]
            p0 = np.asarray(s.track[2 * int(s.t0)], np.float64)
            d = wxy - p0[:2]
            c_, s_ = np.cos(-p0[2]), np.sin(-p0[2])
            pos[b, 0] = (d[0] * c_ - d[1] * s_, d[0] * s_ + d[1] * c_)
        a_m = R.progress(g_log, pos)[:, 0]             # k + q > 0 always
        a_l = R.logged_arc(g_log)[ar, np.clip(kq - 1, 0, 29)]
        v_l = np.array([s.logged_v[int(x)] if int(x) < len(s.logged_v) else np.nan for s, x in zip(st, jq)], np.float64)
        prev = P[:, q - 2] if q >= 2 else np.zeros((B, 2))
        v0 = np.linalg.norm(P[:, q - 1] - prev, axis=-1) / DT_S
        rdl = [label.read(int(c), int(x)) for c, x in zip(ci, jq)]
        lp = np.array([r[0] for r in rdl], bool)
        lg = np.array([r[1] for r in rdl], np.float64)
        lc = np.array([r[2] for r in rdl], np.float64)
        if form == "RR":
            p, g, okr = _reader_at(st, rd, q)
            _, used = feed(form, p, g, okr, STATE["xs"], STATE["ys"])
            hg, hc = used.astype(np.float32).astype(np.float64), np.zeros(B, np.float64)
            hp = (p >= W.HEAD_P_MIN) & okr
            p_src = p
            for kk, vv in (("p", p), ("g_read", g), ("g_used", used), ("ok", okr)):
                M["rd"][kk].append(np.asarray(vv))
        else:                                          # head_values_none of the wrapper: the label forms read no token
            p_src, hg, hc = np.zeros(B, np.float64), np.full(B, np.nan), np.zeros(B, np.float64)
            hp = np.zeros(B, bool)
        present = lp.copy() if mix["presence"] == "label" else hp.copy()
        lab_g, lab_c, head_g, head_c = np.isfinite(lg), np.isfinite(lc), np.isfinite(hg), np.isfinite(hc)
        if mix["gap"] == "label":
            g_src = np.where(present & ~lab_g & head_g, hg, lg)
        else:
            g_src = np.where(present & ~head_g & lab_g, lg, hg)
        if mix["closing"] == "label":
            c_src = np.where(present & ~lab_c & head_c, hc, lc)
        elif mix["closing"] == "head":
            c_src = np.where(present & ~head_c & lab_c, lc, hc)
        else:
            c_src = np.zeros(B, np.float64)
        with np.errstate(invalid="ignore"):
            src_bad = present & ~(np.isfinite(g_src) & np.isfinite(c_src) & np.isfinite(v_l))
            present = present & ~src_bad
            v_L = np.maximum(0.0, v_l + c_src)
            g0 = g_src + a_l - a_m
        rel = P[:, q:] - P[:, q - 1:q]                 # the remaining knots relative to knot q (the rule works on arc lengths along the path: a translation changes nothing)
        res = R.distance_rule(rel, v0, present, g_src, g0, v_L, m=m - q, **pl.kw)
        excess, nchg = np.zeros(B), np.zeros(B, np.int64)
        if res.changed.any():
            new_rel = np.asarray(res.plan, np.float64)
            ch = np.nonzero(res.changed)[0]
            prog = R.progress(rel[ch], new_rel[ch])    # arc coordinate of the new knots on the path before the look (float64, before the committed plan is rounded to float32)
            for n_, b in enumerate(ch):
                excess[b] = float(np.max(prog[n_] - R.plan_path(rel[b])[1][1:]))
                same = (new_rel[b] == rel[b]).all(axis=-1)
                nchg[b] = int((~same).sum())
                out[b, q:] = np.where(same[:, None], out[b, q:], (new_rel[b] + P[b, q - 1]).astype(out.dtype))
        row = dict(ci=ci, t0=t0, j=jq, k=kq, tick=k // max(m, 1), active=res.active, present=present, src_bad=src_bad, sat=res.n_sat, sat_exec=res.n_sat_exec, lift=res.n_lift,
                   changed=res.changed, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0, v_l=v_l, v_L=v_L, a_l=a_l, a_m=a_m, g0=g0, q=np.full(B, q, np.int64), excess=excess, nchg=nchg)
        for f in MID_FIELDS:
            M["rows"][f].append(np.asarray(row[f]))
    for b in range(B):                                 # how often the pinned tool's step envelope would reshape the committed plan (reported, not acted on)
        M["env"]["plans"] += 1
        M["env"]["hits"] += int(T.envelope(out[b].astype(np.float64))[1])
    return out


def save_mid(rule_log, meta, m):
    """Writes ticks/rule_log_mid_<run>.npz (one row per window, tick and later look: the columns of the tick log plus q, excess, nchg), ticks/rule_log_mid_<run>_rd.npz (reader
    reads of the later looks, reader forms) and ticks/p1_first_<run>.npz (the plan after the rule at the first tick). Nothing is written when the rule acts at the tick only."""
    if int(STATE["every"]) >= int(m):
        return
    d, base = os.path.split(os.path.abspath(rule_log))
    assert base.startswith("rule_log_") and base.endswith(".npz"), base
    run = base[len("rule_log_"):]
    M = STATE["mid"]
    cols = {f: (np.concatenate(M["rows"][f]) if M["rows"][f] else np.zeros(0)) for f in MID_FIELDS}
    extra = {"rule_every": np.asarray(int(STATE["every"])), "mid_envelope_plans": np.asarray(M["env"]["plans"]), "mid_envelope_hits": np.asarray(M["env"]["hits"])}
    mid = os.path.join(d, "rule_log_mid_" + run)
    with open(mid, "wb") as fh:
        np.savez(fh, **cols, **{k: np.asarray(v) for k, v in meta.items()}, **extra)
    print("wrote", mid, flush=True)
    if M["rd"]["p"]:
        side = mid[:-4] + "_rd.npz"
        with open(side, "wb") as fh:
            np.savez(fh, **{k: np.concatenate(v) for k, v in M["rd"].items()}, form=np.asarray(STATE["form"]))
        print("wrote", side, flush=True)
    p1 = os.path.join(d, "p1_first_" + run)
    with open(p1, "wb") as fh:
        np.savez(fh, ci=np.concatenate(M["p1"]["ci"]), t0=np.concatenate(M["p1"]["t0"]), plan=np.concatenate(M["p1"]["plan"]))
    print("wrote", p1, flush=True)


'''

w = load("v8f_t1_eval_reader_rule_rz.py")
OLD_HEAD = 'v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.\n'
w = rep(w, '"""' + OLD_HEAD,
        '"""v8f_t1_eval_reader_rule_rzf.py -- written by make_rulefast10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): one more parameter,\n'
        '--rule-every h (executed steps). The lead rule acts at the tick as before and then again for every later step q = h, 2h, ... below the replan interval, on the part of the plan\n'
        'that is still to be driven, with the rule inputs a tick on frame j + q would take (label read or reader read of the logged frame j + q, logged speed and arc at that step,\n'
        'the car progress at knot q, the car speed = the chord speed of the last step before q). One commit of the replan interval as before. Rule forms RLb, RLc and RR only. The later\n'
        'looks go to ticks/rule_log_mid_<run>.npz (the tick log keeps one row per window and tick); the plan after the rule at the first tick goes to ticks/p1_first_<run>.npz.\n'
        'Default h = 10 (or any h at or above the replan interval): nothing new runs and nothing new is written. Text of the source below.\n' + OLD_HEAD)
w = rep(w, "\n\n\nclass ReaderPlanner(X.MixPlanner):", "\n\n" + MID + "\nclass ReaderPlanner(X.MixPlanner):")
w = rep(w, "        return X.MixPlanner.__call__(self, batch)\n",
        "        plan = X.MixPlanner.__call__(self, batch)\n"
        "        if int(STATE[\"every\"]) < int(self.m):\n"
        "            plan = mid_looks(self, plan)\n"
        "        return plan\n")
w = rep(w, '    ap.add_argument("--readout-log", default="")\n', '    ap.add_argument("--readout-log", default="")\n    ap.add_argument("--rule-every", type=int, default=10)\n')
w = rep(w, "    T, path = W.load_pinned(mine.tree)\n", "    T, path = W.load_pinned(mine.tree)\n    STATE[\"T\"] = T\n")
w = rep(w, '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n',
        '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n'
        '    STATE["every"] = int(mine.rule_every)\n'
        '    if STATE["every"] < 1:\n'
        '        raise SystemExit("REFUSED: --rule-every needs a value of at least 1 step")\n'
        '    if STATE["every"] < int(a.replan_steps):\n'
        '        if f not in ("RLb", "RLc", "RR"):\n'
        '            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc or RR")\n'
        '        if int(a.replan_steps) % STATE["every"]:\n'
        '            raise SystemExit("REFUSED: --rule-every must divide the replan interval")\n'
        '        if not mine.rule_log:\n'
        '            raise SystemExit("REFUSED: --rule-every below the replan interval needs --rule-log (the later looks are written beside it)")\n'
        '        if STATE["alim"] is not None:\n'
        '            raise SystemExit("REFUSED: --rule-every below the replan interval with an acceleration limit is not a tested combination")\n')
w = rep(w, '            log.save(mine.rule_log, meta)\n            print("wrote", mine.rule_log, flush=True)\n',
        '            log.save(mine.rule_log, meta)\n            print("wrote", mine.rule_log, flush=True)\n            save_mid(mine.rule_log, meta, a.replan_steps)\n')
compile(w, "v8f_t1_eval_reader_rule_rzf.py", "exec")
save("v8f_t1_eval_reader_rule_rzf.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the pod queue
q = load("pod_causlim10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n",
        "#!/usr/bin/env bash\n# pod_rulefast10k_queue.sh -- written by make_rulefast10k.py from pod_causlim10k_queue.sh. Card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): the tool check first (check_rule_every.py\n"
        "# on the pod, no model), then the control RLcFID (label-fed rule, --rule-every given as 10, model seed 0, compared file for file with the banked dump of RLc: marker fr1cmp), then\n"
        "# at two model seeds the label-fed rule every 5 steps (RLcF5), the reader-fed rule at level 0.50 every 5 steps (RQ50F5) and the label-fed rule every 2 steps (RLcF2), all at the\n"
        "# 2.0 s replan interval of the model with the wrapper v8f_t1_eval_reader_rule_rzf.py. No training. The queue waits for the end marker of the queue of card ADD61 (variables\n"
        "# WAIT_DONE / WAIT_STOP, defaults CAUSLIM10K_DONE.txt / CAUSLIM10K_STOP.txt in the folder $W; WAIT_MAX seconds, default 86400). Markers START_RULEFAST10K.txt /\n"
        "# RULEFAST10K_STOP.txt / RULEFAST10K_DONE.txt; step markers rf9check, fr1cmp. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "CAUSLIM10K", "RULEFAST10K", 3)
q = rep(q, 'card ADD61: tool checks, then two controls and four rollouts without a rule (causal readout with the chord limit 3.0 / 8.0 m/s2 at 5 and at 10 steps); checkpoints of card ADD46"',
        'card ADD62: tool check, then the control RLcFID and six rollouts with the lead rule applied again inside the 2.0 s block (every 5 and every 2 steps); checkpoints of card ADD46"')
q = rep(q, 'WAIT_DONE=${WAIT_DONE:-ACCWIN2S10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-ACCWIN2S10K_STOP.txt}', 'WAIT_DONE=${WAIT_DONE:-CAUSLIM10K_DONE.txt}; WAIT_STOP=${WAIT_STOP:-CAUSLIM10K_STOP.txt}')
q = rep(q, "the tool checks and the rollouts of card ADD61 start", "the tool check and the rollouts of card ADD62 start")
q = rep(q, '  [ -f "$W/t1/I10R0C5_s0/T1_DONE" ] || stop "NO-BANKED-R0C5"\n', '  [ -f "$W/t1/I10RLc_s0/T1_DONE" ] || stop "NO-BANKED-RLC"\n')
q = rep(q, 'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzc.py v8f_t1_eval_reader_rule_rz.py v8f_t1_eval_reader_rule_a.py v8f_t1_eval_reader_rule_ro.py test_accel_limit_chord_rzc.py check_tick_readout_rzc.py ',
        'for f in cmp_dumps.py v8f_t1_eval_reader_rule_rzf.py check_rule_every.py ')
q = rep(q, '[ "$DRY" = 1 ] || [ -f "$W/t1/I10R0M5_s1/T1_DONE" ] || stop "NO-BANKED-R0M5"\n', '')
q = rep(q, "for nm in I10R0 I10R0M5 I10R0C5 I10R0E5 I10R0E10; do", "for nm in I10R0 I10RLc I10RQ50 I10RLcM5 I10RQ50M5; do")
q = rep(q, '  [ -n "${AMAX:-}" ] && extra+=(--a-max "$AMAX" --b-lim "${BLIM:?}" --alim-log "$out/ticks/alim_log_$arm.npz" --alim-form chord)\n',
        '  [ -n "${EVERY:-}" ] && extra+=(--rule-every "$EVERY")\n')
q = rep(q, 'replan steps ${REPLAN:-10} model seed $m', 'replan steps ${REPLAN:-10} rule every ${EVERY:-10} model seed $m')
q = span(q, '# ---------------------------------------------------------------- control c9 first: the two tool checks on the pod', '  AMAX=3.0 BLIM=8.0 READOUT=causal REPLAN=10 rollout I10R0CE10 R0 "$k" "$k"\ndone\n',
         '# ---------------------------------------------------------------- control k5 first: the tool check on the pod (no model, no rollout, no GPU, three threads); its output file and the copy of the wrapper it ran with are read by the reader\n'
         'if ! done_step rf9check; then\n'
         '  if [ "$DRY" = 1 ]; then\n'
         '    echo "CMD: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 DISTRULE_DIR=$B PYTHONPATH=$TREE/stack $PY $B/check_rule_every.py $TREE > $W/rulefast_test/CHECK_RULE_EVERY.txt"\n'
         '  else\n'
         '    mkdir -p "$W/rulefast_test"\n'
         '    cp -f "$B/v8f_t1_eval_reader_rule_rzf.py" "$W/rulefast_test/v8f_t1_eval_reader_rule_rzf.py"\n'
         '    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 DISTRULE_DIR=$B PYTHONPATH="$TREE/stack" timeout 1800 "$PY" "$B/check_rule_every.py" "$TREE" > "$W/rulefast_test/CHECK_RULE_EVERY.txt" 2>&1\n'
         '    tail -n 1 "$W/rulefast_test/CHECK_RULE_EVERY.txt" | grep -q "^ZZ CHECK-RULE-EVERY all hold: True" || stop "TOOL-CHECK-RULE-EVERY-FAILED"\n'
         '    mark rf9check\n'
         '  fi\n'
         'fi\n'
         'EVERY=10 rollout I10RLcFID RLc 0 0       # control k1: the rule-every parameter given as 10 must give the banked dump of RLc at model seed 0\n'
         'if [ "$DRY" != 1 ] && ! done_step fr1cmp; then\n'
         '  mkdir -p "$W/small_rulefast"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10RLc_s0/dump_I10RLc_s0_m10_causal_heads" "$W/t1/I10RLcFID_s0/dump_I10RLcFID_s0_m10_causal_heads" > "$W/small_rulefast/CMP_RLcFID.txt" 2>&1 || stop "K1-RERUN-DIFFERS"\n'
         '  mark fr1cmp\n'
         'fi\n'
         'for k in "${ARMS[@]}"; do\n'
         '  EVERY=5 rollout I10RLcF5 RLc "$k" "$k"\n'
         '  EVERY=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50F5 RR "$k" "$k"\n'
         'done\n'
         'for k in "${ARMS[@]}"; do\n'
         '  EVERY=2 rollout I10RLcF2 RLc "$k" "$k"      # dose, names nothing\n'
         'done\n')
q = rep_code(q, "v8f_t1_eval_reader_rule_rzc.py", "v8f_t1_eval_reader_rule_rzf.py", 3)
c = code_lines(q)
for bad in ("CAUSLIM10K_START", "AMAX", "alim-form", "cl9check", "ce1cmp", "ce1ecmp", "_rzc.py", "I10R0CE", "small_causlim", "causlim_test", "ACCWIN2S10K"):
    assert bad not in c, bad
assert c.count("rollout I10R") == 4 and c.count("RULEFAST10K") >= 3 and "fr1cmp" in c and "rf9check" in c and 'wrapper sha256' in c, "queue sanity"
assert "CAUSLIM10K_DONE.txt" in c and "RULEFAST10K_DONE.txt" in c
save("pod_rulefast10k_queue.sh", q)

# ---------------------------------------------------------------- the dev-box chain
ch = load("chain_causlim10k.sh")
ch = rep(ch, "#!/usr/bin/env bash\n# chain_causlim10k.sh -- written by make_causlim10k.py from chain_accwin2s10k.sh for card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). The poll runs 48 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD58, up to 24 h). The header lines below are those of the parent.\n",
         "#!/usr/bin/env bash\n# chain_rulefast10k.sh -- written by make_rulefast10k.py from chain_causlim10k.sh for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K). The poll runs 72 h (the queue first waits for the\n"
         "# end marker of the queue of card ADD61, up to 24 h). The header lines below are those of the parent.\n")
ch = rep(ch, '"$I/pod_causlim10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzc.py" "$I/v8f_t1_eval_reader_rule_rz.py" "$I/v8f_t1_eval_reader_rule_a.py" "$I/v8f_t1_eval_reader_rule_ro.py" "$I/test_accel_limit_chord_rzc.py" "$I/check_tick_readout_rzc.py"',
         '"$I/pod_causlim10k_queue.sh" "$I/v8f_t1_eval_reader_rule_rzf.py" "$I/check_rule_every.py"')
ch = rep(ch, "for nm in I10R0CEIDc I10R0CEIDe I10R0CE5 I10R0CE10; do", "for nm in I10RLcFID I10RLcF5 I10RQ50F5 I10RLcF2; do")
ch = rep(ch, "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt $W/causlim_test/CHECK_TICK_READOUT_RZC.txt", "$W/rulefast_test/CHECK_RULE_EVERY.txt")
ch = rep(ch, '[ "$k" -gt 2880 ] && { st=STOP; say "not ended after 48 h"; break; }', '[ "$k" -gt 4320 ] && { st=STOP; say "not ended after 72 h"; break; }')
ch = rep(ch, "EXITS on CAUSLIM10K_DONE, CAUSLIM10K_STOP or after 48 h", "EXITS on CAUSLIM10K_DONE, CAUSLIM10K_STOP or after 72 h")
for old, new in (("pod_causlim10k_queue.sh", "pod_rulefast10k_queue.sh"), ("small_causlim", "small_rulefast"), ("chain_causlim10k.log", "chain_rulefast10k.log"), ("queue_causlim_stdout", "queue_rulefast_stdout"),
                 ("START_CAUSLIM10K", "START_RULEFAST10K"), ("CHAIN-CAUSLIM10K", "CHAIN-RULEFAST10K")):
    assert ch.count(old) >= 1, old
    ch = ch.replace(old, new)
# the markers the chain polls for are those of THIS queue; the queue itself waits for CAUSLIM10K_DONE.txt
ch = ch.replace("$W/CAUSLIM10K_DONE.txt", "$W/RULEFAST10K_DONE.txt").replace("$W/CAUSLIM10K_STOP.txt", "$W/RULEFAST10K_STOP.txt")
cc = code_lines(ch)
for bad in ("causlim", "I10R0CE", "_rzc.py", "TEST_ACCEL", "CHECK_TICK"):
    assert bad not in cc.replace("pod_rulefast10k_queue.sh", ""), bad
assert "RULEFAST10K_DONE.txt" in cc and "RULEFAST10K_STOP.txt" in cc and "CAUSLIM10K_DONE" not in cc.replace("# ", ""), "chain markers"
save("chain_rulefast10k.sh", ch)

# ---------------------------------------------------------------- the pod read and its waiter
PAIRS = "I10RLcF5:I10RLc,I10RQ50F5:I10RQ50,I10RLcF2:I10RLc,I10RLcF2:I10RLcF5,I10RLcF5:I10R0,I10RQ50F5:I10R0,I10RLcF2:I10R0,I10RQ50F5:I10RLcF5"
pr = load("podread_causlim.sh")
pr = rep(pr, "#!/usr/bin/env bash\n# podread_causlim.sh -- pod-side read of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K). Written from podread_accwin2s.sh (card ADD58) with the names, pairs and files of this card.\n",
         "#!/usr/bin/env bash\n# podread_rulefast.sh -- pod-side read of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K). Written by make_rulefast10k.py from podread_causlim.sh (card ADD61) with the names, pairs and files of this card.\n")
pr = rep(pr, "# Order: control e1 first", "# Order: control e1 first")
pr = rep(pr, "only if e1 passes, the read of card ADD61.", "only if e1 passes, the read of card ADD62.")
pr = rep(pr, "ZZ PODREAD-CAUSLIM DONE or ZZ PODREAD-CAUSLIM STOP <why>.", "ZZ PODREAD-RULEFAST DONE or ZZ PODREAD-RULEFAST STOP <why>.")
pr = rep(pr, "$P/root/integ10k/read_causlim10k.py and $P/root/integ10k/CAUSLIM10K_REUSED.json (sha256 of the result files of the\n# reused runs, recorded before the queue started); the two tool-check outputs and the wrapper copy are written by the queue into $W/causlim_test/, the two compare files into $W/small_causlim/.",
         "$P/root/integ10k/read_rulefast10k.py and $P/root/integ10k/RULEFAST10K_REUSED.json (sha256 of the result files and rule logs of the\n# reused runs, recorded before the queue started); the tool-check output and the wrapper copy are written by the queue into $W/rulefast_test/, the compare file into $W/small_rulefast/.")
pr = rep(pr, '[ -f "$PL/small_causlim/CMP_R0CEIDc.txt" ] || stop "NO-CMP-FILE-C1"\n[ -f "$PL/small_causlim/CMP_R0CEIDe.txt" ] || stop "NO-CMP-FILE-C2"\n', '[ -f "$PL/small_causlim/CMP_RLcFID.txt" ] || stop "NO-CMP-FILE-K1"\n')
pr = rep(pr, '[ -f "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" ] || stop "NO-TOOL-TEST-OUTPUT"\n[ -f "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"\n',
         '[ -f "$W/causlim_test/CHECK_RULE_EVERY.txt" ] || stop "NO-TOOL-CHECK-OUTPUT"\n')
pr = rep(pr, 'PAIRS="I10R0CE5:I10R0M5,I10R0CE5:I10R0C5,I10R0CE5:I10R0E5,I10R0CE5:I10R0,I10R0CE10:I10R0,I10R0CE10:I10R0E10"', 'PAIRS="%s"' % PAIRS)
pr = rep(pr, "# ---- the read of card ADD61 (pairs of the card: SP, the two PARTS pairs, GC, PR and the pair beside PR)", "# ---- the read of card ADD62 (pairs of the card: FR per form, the dose pairs, the pairs beside)")
pr = rep(pr, '--wrapper "$W/build/v8f_t1_eval_reader_rule_rzc.py" --test-wrapper "$W/causlim_test/v8f_t1_eval_reader_rule_rzc.py" --test-out-chord "$W/causlim_test/TEST_ACCEL_LIMIT_CHORD_RZC.txt" --test-out-tick "$W/causlim_test/CHECK_TICK_READOUT_RZC.txt"',
         '--wrapper "$W/build/v8f_t1_eval_reader_rule_rzf.py" --test-wrapper "$W/causlim_test/v8f_t1_eval_reader_rule_rzf.py" --test-out "$W/causlim_test/CHECK_RULE_EVERY.txt"')
for old, new in (("causlim_test", "rulefast_test"), ("small_causlim", "small_rulefast"), ("read_causlim10k", "read_rulefast10k"), ("READ_CAUSLIM10K", "READ_RULEFAST10K"), ("CAUSLIM10K_REUSED", "RULEFAST10K_REUSED"),
                 ("CAUSLIM10K_DONE", "RULEFAST10K_DONE"), ("PODREAD-CAUSLIM", "PODREAD-RULEFAST"), ("read_causlim", "read_rulefast"), ("e1cl", "e1rf")):
    pr = rep_all_code(pr, old, new)
assert "causlim" not in code_lines(pr).lower(), "podread names"
save("podread_rulefast.sh", pr)

wt = load("wait_podread_causlim.sh")
wt = rep(wt, "#!/usr/bin/env bash\n# wait_podread_causlim.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD61 (CAUSLIM10K), written from wait_podread_accwin2s.sh (card ADD58), bash only (no Python on the dev box).\n",
         "#!/usr/bin/env bash\n# wait_podread_rulefast.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD62 (RULEFAST10K), written by make_rulefast10k.py from wait_podread_causlim.sh (card ADD61), bash only (no Python on the dev box).\n")
for old, new in (("podread_causlim", "podread_rulefast"), ("READ_CAUSLIM10K", "READ_RULEFAST10K"), ("read_causlim", "read_rulefast"), ("CAUSLIM10K", "RULEFAST10K"), ("PODREAD-CAUSLIM", "PODREAD-RULEFAST"), ("e1cl", "e1rf")):
    wt = rep_all_code(wt, old, new)
wt = rep(wt, '[ "$k" -gt 5760 ]', '[ "$k" -gt 8640 ]')
wt = rep(wt, "not ended after 48 h", "not ended after 72 h")
assert "causlim" not in code_lines(wt).lower(), "wait names"
save("wait_podread_rulefast.sh", wt)
print("wrapper sha256 %s" % WSHA)
print("ZZ MAKE-RULEFAST10K DONE")
