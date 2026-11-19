"""make_feas10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD53 (FEAS10K: the 1.0 s cycle with an acceleration limit on the executed plan) by asserted text edits:
  v8f_t1_eval_reader_rule_t.py -> v8f_t1_eval_reader_rule_a.py  (two more parameters, --a-max and --b-lim in m/s2: every plan is retimed along its own path so that the
                                                                 step speeds obey the two limits, starting from the ego speed at the tick; before the lead rule if one is on)
  pod_cycle10k_queue.sh        -> pod_feas10k_queue.sh
  chain_cycle10k.sh            -> chain_feas10k.sh
The originals are not changed. Usage: python make_feas10k.py (in this folder)"""
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
w = load("v8f_t1_eval_reader_rule_t.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_t.py -- written by make_taucurve10k.py from v8f_t1_eval_reader_rule_d.py for card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): one more parameter,',
        '"""v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py from v8f_t1_eval_reader_rule_t.py for card PREREG_V8F_PROOF_ADD53 (FEAS10K): two more parameters,\n'
        '--a-max and --b-lim (m/s2). With them every plan of the model is retimed along its own path before anything else: the step speeds may rise by at most a-max * 0.2 s\n'
        'and fall by at most b-lim * 0.2 s per step, starting from the ego speed at the tick (the speed feature the model itself reads). It is a study arm, no vehicle model:\n'
        'no jerk limit, no lateral limit. Works with every form, R0 included; a lead rule, if on, acts on the limited plan. The rollout of the kinematic prior alone (the\n'
        'floor arrays of the dump) is limited the same way and has its own log columns. Without the two parameters nothing changes.\n'
        'Header of the parent:\n'
        'v8f_t1_eval_reader_rule_t.py -- written by make_taucurve10k.py from v8f_t1_eval_reader_rule_d.py for card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): one more parameter,')
w = rep(w, 'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "slope": None, "k": 0, "prev": None, "zero": None,\n',
        'STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "slope": None, "k": 0, "prev": None, "zero": None, "alim": None, "m": 10, "alog": {},\n')
w = rep(w, 'class ReaderPlanner(X.MixPlanner):',
        'def accel_limit(plan, v0, a_max, b_lim, m, dt=DT_S):\n'
        '    """Retimes each plan along its own path (the lag form of the distance rule): the arc advance a\'(i) of step i is clipped to\n'
        '    [a\'(i-1) + max(0, v\'(i-1) - b_lim dt) dt, a\'(i-1) + (v\'(i-1) + a_max dt) dt] with v\'(0) = v0; a step that is not clipped keeps the plan\'s own advance.\n'
        '    -> (plan\', info); info per row: n_up / n_dn = executed steps (i <= m) clipped by a_max / b_lim, req = the largest acceleration the plan asked for on an executed step."""\n'
        '    p = np.asarray(plan)\n'
        '    B, N = p.shape[0], p.shape[1]\n'
        '    p64 = p.astype(np.float64)\n'
        '    v0 = np.asarray(v0, np.float64)\n'
        '    ok = np.isfinite(v0)\n'
        '    aP = np.zeros((B, N + 1))\n'
        '    PP, EXT = [], []\n'
        '    for b in range(B):\n'
        '        P, L, ext = R.plan_path(p64[b])\n'
        '        aP[b] = L\n'
        '        PP.append(P)\n'
        '        EXT.append(ext)\n'
        '    lag, a_prev, v_prev = np.zeros(B), np.zeros(B), np.where(ok, v0, 0.0)\n'
        '    a_new = np.zeros((B, N))\n'
        '    n_up, n_dn, req = np.zeros(B, np.int64), np.zeros(B, np.int64), np.full(B, -np.inf)\n'
        '    for i in range(1, N + 1):\n'
        '        a_w = aP[:, i] - lag\n'
        '        hi = a_prev + (v_prev + a_max * dt) * dt\n'
        '        lo = a_prev + np.maximum(0.0, v_prev - b_lim * dt) * dt\n'
        '        a_i = np.where(ok, np.minimum(np.maximum(a_w, lo), hi), aP[:, i])\n'
        '        if i <= m:\n'
        '            n_up += (ok & (a_w > hi)).astype(np.int64)\n'
        '            n_dn += (ok & (a_w < lo)).astype(np.int64)\n'
        '            req = np.maximum(req, np.where(ok, ((a_w - a_prev) / dt - v_prev) / dt, -np.inf))\n'
        '        lag = aP[:, i] - a_i\n'
        '        v_prev = (a_i - a_prev) / dt\n'
        '        a_prev = a_i\n'
        '        a_new[:, i - 1] = a_i\n'
        '    out = p.copy()\n'
        '    changed = np.zeros(B, bool)\n'
        '    for b in range(B):\n'
        '        keep = a_new[b] == aP[b, 1:]                       # an unchanged step is the original knot (bit for bit)\n'
        '        if keep.all():\n'
        '            continue\n'
        '        pts = R.sample_path(PP[b], aP[b], EXT[b], a_new[b])\n'
        '        pts[keep] = p64[b][keep]\n'
        '        new = pts.astype(p.dtype)\n'
        '        changed[b] = bool((new != p[b]).any())\n'
        '        out[b] = new\n'
        '    v_exec = np.diff(np.concatenate([np.zeros((B, 1)), a_new[:, :m]], 1), axis=1) / dt     # arc speeds of the executed steps of the limited plan\n'
        '    return (out if changed.any() else plan), {"n_up": n_up, "n_dn": n_dn, "req": req, "changed": changed, "v0": v0, "v_exec": v_exec, "_a_new": a_new}\n\n\n'
        'class AccelLimited:\n'
        '    """The model planner followed by the acceleration limit. Reads the plan and the ego speed at the tick, nothing else; keeps keys_read, aux, heads of the base."""\n\n'
        '    def __init__(self, base, tag="m"):\n'
        '        self.base, self.tag = base, tag\n\n'
        '    keys_read = property(lambda self: self.base.keys_read)\n'
        '    aux = property(lambda self: self.base.aux)\n'
        '    heads = property(lambda self: self.base.heads)\n\n'
        '    def __call__(self, batch):\n'
        '        plan = self.base(batch)\n'
        '        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()\n'
        '        out, info = accel_limit(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])\n'
        '        info["nb"] = np.asarray([len(v0)])                       # one value per call: the rows of this call\n'
        '        for k, v in info.items():\n'
        '            if not k.startswith("_"):\n'
        '                STATE["alog"].setdefault(self.tag + "_" + k, []).append(np.asarray(v))\n'
        '        return out\n\n\n'
        'def save_alog(path, form, replan_steps):\n'
        '    if not path or STATE["alim"] is None:\n'
        '        return\n'
        '    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)\n'
        '    cols = {k: np.concatenate(v) for k, v in STATE["alog"].items()}\n'
        '    with open(path, "wb") as fh:\n'
        '        np.savez(fh, **cols, a_max=np.asarray(STATE["alim"][0]), b_lim=np.asarray(STATE["alim"][1]), form=np.asarray(form), replan_steps=np.asarray(int(replan_steps)))\n'
        '    print("wrote", path, flush=True)\n\n\n'
        'class ReaderPlanner(X.MixPlanner):')
w = rep(w, '        if self.source == "mix":\n            self.label = X.STATE["label"]\n',
        '        if self.source == "mix":\n            self.label = X.STATE["label"]\n'
        '        if STATE["alim"] is not None:\n            self.base = AccelLimited(self.base)        # the lead rule then acts on the limited plan\n')
w = rep(w, '    ap.add_argument("--tau", type=float, default=None)\n',
        '    ap.add_argument("--tau", type=float, default=None)\n    ap.add_argument("--a-max", type=float, default=None)\n    ap.add_argument("--b-lim", type=float, default=None)\n'
        '    ap.add_argument("--alim-log", default="")\n')
w = rep(w, '    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}\n',
        '    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}\n'
        '    if (mine.a_max is None) != (mine.b_lim is None) or (mine.a_max is not None and (not mine.a_max > 0 or not mine.b_lim > 0 or not mine.alim_log)):\n'
        '        raise SystemExit("REFUSED: --a-max and --b-lim come together, both above 0, with --alim-log")\n'
        '    if mine.a_max is None and mine.alim_log:\n'
        '        raise SystemExit("REFUSED: --alim-log without the two limits would write nothing")\n'
        '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n'
        '    STATE["m"] = int(a.replan_steps)\n'
        '    orig_fp = T.floor_planner\n'
        '    if STATE["alim"] is not None:                      # the rollout of the kinematic prior alone is limited the same way (its own log columns f_*)\n'
        '        def limited_floor(model, device):\n'
        '            return AccelLimited(orig_fp(model, device), "f")\n\n'
        '        T.floor_planner = limited_floor\n')
w = rep(w, '            raise SystemExit("REFUSED: R0 installs nothing; --rule-log/--reader/--qmap would write or read nothing")\n        return T.cmd_run(a)\n',
        '            raise SystemExit("REFUSED: R0 installs nothing; --rule-log/--reader/--qmap would write or read nothing")\n'
        '        if STATE["alim"] is None:\n'
        '            return T.cmd_run(a)\n'
        '        orig_mp = T.model_planner\n\n'
        '        def limited_planner(model, device, keep_aux=False, keep_heads=False):\n'
        '            return AccelLimited(orig_mp(model, device, keep_aux=keep_aux, keep_heads=keep_heads))\n\n'
        '        T.model_planner = limited_planner\n'
        '        try:\n'
        '            return T.cmd_run(a)\n'
        '        finally:\n'
        '            T.model_planner = orig_mp\n'
        '            T.floor_planner = orig_fp\n'
        '            save_alog(mine.alim_log, f, a.replan_steps)\n')
w = rep(w, '        restore()\n        if mine.rule_log:\n', '        restore()\n        T.floor_planner = orig_fp\n        save_alog(mine.alim_log, f, a.replan_steps)\n        if mine.rule_log:\n')
w = rep(w, '"dt_s": DT_S}', '"dt_s": DT_S, "a_max": float(STATE["alim"][0]) if STATE["alim"] else float("nan"), "b_lim": float(STATE["alim"][1]) if STATE["alim"] else float("nan")}')
save("v8f_t1_eval_reader_rule_a.py", w)

# ---------------------------------------------------------------- the pod queue
q = load("pod_cycle10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_feas10k_queue.sh -- written by make_feas10k.py from pod_cycle10k_queue.sh. Card PREREG_V8F_PROOF_ADD53 (FEAS10K): eleven rollouts with the wrapper\n"
        "# v8f_t1_eval_reader_rule_a.py: R0AID (no rule, limits out of reach, 10 steps, model seed 0: must equal the banked R0, control f1), then with the limits 3.0 / 8.0 m/s2 at\n"
        "# 5 steps and two model seeds: no rule (R0A5), the label-fed rule with the closing speed (RLcA5), the reader-fed rule at level 0.50 (RQ50A5); then no rule with the\n"
        "# limits at 10 steps (R0A10) and no rule with the limits 1.5 / 8.0 at 5 steps (R0A5L), each at two model seeds. No training, no map fit.\n"
        "# Markers START_FEAS10K.txt / FEAS10K_STOP.txt / FEAS10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep(q, 'card ADD52: seven rollouts (replan interval 5 steps for no rule, the label-fed rule with the closing speed and the reader-fed rule at level 0.50); reader arms, maps and checkpoints of cards ADD46/ADD47"',
        'card ADD53: eleven rollouts (acceleration limit 3.0 / 8.0 m/s2 on the executed plan at 5 and at 10 steps, 1.5 / 8.0 at 5 steps without a rule); reader arms, maps and checkpoints of cards ADD46/ADD47"')
q = rep(q, "CYCLE10K", "FEAS10K", 4, at_least=True)
q = rep(q, "v8f_t1_eval_reader_rule_t.py", "v8f_t1_eval_reader_rule_a.py", 3, at_least=True)
q = rep(q, '  [ -n "${TAU:-}" ] && extra+=(--tau "$TAU")\n', '  [ -n "${TAU:-}" ] && extra+=(--tau "$TAU")\n  [ -n "${AMAX:-}" ] && extra+=(--a-max "$AMAX" --b-lim "${BLIM:?}" --alim-log "$out/ticks/alim_log_$arm.npz")\n')
q = rep(q, '[ "$DRY" = 1 ] || [ -f "$W/t1/I10RLc_s0/T1_DONE" ] || stop "NO-BANKED-RLC"', '[ "$DRY" = 1 ] || [ -f "$W/t1/I10R0M5_s1/T1_DONE" ] || stop "NO-BANKED-R0M5"')
q = span(q, "REPLAN=10 rollout I10RLcM10 RLc 0 0", '  REPLAN=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50M5 RR "$k" "$k"\ndone\n',
         'AMAX=1e9 BLIM=1e9 REPLAN=10 rollout I10R0AID R0 0 0       # control f1 first: limits out of reach must give the banked R0\n'
         'if [ "$DRY" != 1 ] && ! done_step f1cmp; then\n'
         '  mkdir -p "$W/small_feas"\n'
         '  "$PY" "$B/cmp_dumps.py" "$W/t1/I10R0_s0/dump_I10R0_s0_m10_causal_heads" "$W/t1/I10R0AID_s0/dump_I10R0AID_s0_m10_causal_heads" > "$W/small_feas/CMP_R0AID.txt" 2>&1 || stop "F1-RERUN-DIFFERS"\n'
         "  mark f1cmp\n"
         "fi\n"
         'for k in "${ARMS[@]}"; do\n'
         '  AMAX=3.0 BLIM=8.0 REPLAN=5 rollout I10R0A5 R0 "$k" "$k"\n'
         '  AMAX=3.0 BLIM=8.0 REPLAN=5 rollout I10RLcA5 RLc "$k" "$k"\n'
         '  AMAX=3.0 BLIM=8.0 REPLAN=5 QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json" rollout I10RQ50A5 RR "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  AMAX=3.0 BLIM=8.0 REPLAN=10 rollout I10R0A10 R0 "$k" "$k"\n'
         "done\n"
         'for k in "${ARMS[@]}"; do\n'
         '  AMAX=1.5 BLIM=8.0 REPLAN=5 rollout I10R0A5L R0 "$k" "$k"      # the dose arm: half the acceleration limit, no rule\n'
         "done\n")
c = code_lines(q)
assert "CYCLE10K" not in c and "I10RLcM10" not in c and "small_cycle" not in c and "_t.py" not in c
assert c.count("rollout I10R") == 6 and c.count("v8f_t1_eval_reader_rule_a.py") == 3 and c.count("--alim-log") == 1
save("pod_feas10k_queue.sh", q)

c = load("chain_cycle10k.sh")
c = rep(c, "LOGF=$I/chain_cycle10k.log", "LOGF=$I/chain_feas10k.log")
c = rep(c, '"$I/pod_cycle10k_queue.sh" "$I/v8f_t1_eval_reader_rule_t.py"', '"$I/pod_feas10k_queue.sh" "$I/v8f_t1_eval_reader_rule_a.py"')
c = rep(c, "bash -n $B/pod_cycle10k_queue.sh", "bash -n $B/pod_feas10k_queue.sh")
c = rep(c, "nohup setsid bash $B/pod_cycle10k_queue.sh > $W/queue_cycle_stdout.txt", "nohup setsid bash $B/pod_feas10k_queue.sh > $W/queue_feas_stdout.txt")
c = rep(c, "CYCLE10K", "FEAS10K", 6, at_least=True)
c = rep(c, "for nm in I10RLcM10 I10R0M5 I10RLcM5 I10RQ50M5; do", "for nm in I10R0AID I10R0A5 I10RLcA5 I10RQ50A5 I10R0A10 I10R0A5L; do")
c = rep(c, "small_cycle", "small_feas", 2, at_least=True)
cc = code_lines(c)
assert "CYCLE10K" not in cc and "small_cycle" not in cc and "pod_cycle10k" not in cc and "_t.py" not in cc
save("chain_feas10k.sh", c)
print("ZZ MAKE-FEAS10K DONE")
