"""make_feasc10k.py -- writes the tools of card PREREG_V8F_PROOF_ADD56 (the repeat of card ADD53 after its control f5 did not pass: the acceleration limit held on the
executed knots themselves) by asserted text edits:
  v8f_t1_eval_reader_rule_a.py -> v8f_t1_eval_reader_rule_ac.py  (one more parameter, --alim-form path|chord; path = the wrapper of card ADD53 unchanged)
  pod_feas10k_queue.sh         -> pod_feasc10k_queue.sh          (run names E in place of A, --alim-form chord, starts after the queue of card ADD55)
  chain_feas10k.sh             -> chain_feasc10k.sh
  read_feas10k.py              -> read_feasc10k.py
  podread_feas.sh              -> podread_feasc.sh
  wait_podread_feas.sh         -> wait_podread_feasc.sh
The originals are not changed. Usage: python make_feasc10k.py (in this folder)"""
import hashlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = re.compile(r"I10(R0|RLc|RQ50)A(5L|5|10|ID)")


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):                              # never write over a file of another card (a file name can be taken)
        head = io.open(p, encoding="utf-8", errors="replace").read(3000)
        assert "make_feasc10k.py" in head, "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes) sha256 %s" % (name, len(text.encode("utf-8")), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:70], c, n)
    return text.replace(old, new)


def rep_code(text, old, new, n):
    """replace in lines that are not comments; exactly n lines must change."""
    out, k = [], 0
    for ln in text.split("\n"):
        if not ln.lstrip().startswith("#") and old in ln:
            ln = ln.replace(old, new)
            k += 1
        out.append(ln)
    assert k == n, "code anchor %r changed %d lines, expected %d" % (old, k, n)
    return "\n".join(out)


def runs(text, n=None):
    """run names of card ADD53 (..A5, ..A10, ..A5L, ..AID) -> the names of the repeat (..E5, ..E10, ..E5L, ..EID); n = number of names that must change."""
    out, c = RUN.subn(r"I10\1E\2", text)
    assert (c > 0) if n is None else (c == n), "run names changed: %d, expected %s" % (c, n)
    return out


# ---------------------------------------------------------------- the wrapper
w = load("v8f_t1_eval_reader_rule_a.py")
w = rep(w, '"""v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py',
        '"""v8f_t1_eval_reader_rule_ac.py -- written by make_feasc10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD56 (the repeat of card ADD53):\n'
        'one more parameter, --alim-form path|chord. path (default) is the wrapper of card ADD53 unchanged: the limit acts on the arc speed along the plan path. chord: the\n'
        'limit is held on the chord speeds between the knots of the limited plan, which is what the executed track and the state rebuild show. Text of the source below.\n'
        'v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py')
w = rep(w, '"alim": None, "m": 10, "alog": {},', '"alim": None, "aform": "path", "m": 10, "alog": {},')
CHORD = '''

def _sample_rows(P, L, EXT, a):
    """Points of B plan paths at one arc length each: the batched form of dist_rule.sample_path (same segment choice, same extension ray)."""
    a = np.maximum(np.asarray(a, np.float64), 0.0)
    r = np.arange(len(a))
    j = np.clip((L <= a[:, None]).sum(1) - 1, 0, L.shape[1] - 2)
    wgt = (a - L[r, j]) / np.maximum(L[r, j + 1] - L[r, j], 1e-12)
    pts = P[r, j] + wgt[:, None] * (P[r, j + 1] - P[r, j])
    far = a >= L[:, -1]
    if far.any():
        pts[far] = P[far, -1] + EXT[far] * (a[far] - L[far, -1])[:, None]
    return pts


def accel_limit_chord(plan, v0, a_max, b_lim, m, dt=DT_S, iters=44):
    """The limit of accel_limit held on the knots themselves. Step i of the limited plan lies on the plan's own path; its chord length from the knot before, c'(i), is kept in
    [max(0, v'(i-1) - b_lim dt) dt, (v'(i-1) + a_max dt) dt] with v'(i) = c'(i) / dt and v'(0) = v0. Lag form as accel_limit: a step that needs no clip keeps the plan's own
    arc advance. On a straight path this is accel_limit; on a bent path the chord is shorter than the arc, and the chord is what the executed track and the state rebuild
    show. -> (plan', info); info as accel_limit with v_exec = the chord speeds of the executed steps, plus n_fail = executed steps whose lower bound was not reached on the
    path within the search (a path that turns back on itself)."""
    p = np.asarray(plan)
    B, N = p.shape[0], p.shape[1]
    p64 = p.astype(np.float64)
    v0 = np.asarray(v0, np.float64)
    ok = np.isfinite(v0)
    PP, aP, EXT = np.zeros((B, N + 1, 2)), np.zeros((B, N + 1)), np.zeros((B, 2))
    for b in range(B):
        PP[b], aP[b], EXT[b] = R.plan_path(p64[b])

    def chord(rows, arc, q):
        d = _sample_rows(PP[rows], aP[rows], EXT[rows], arc) - q
        return np.hypot(d[:, 0], d[:, 1])

    allr = np.arange(B)
    lag, a_prev, q_prev, v_prev = np.zeros(B), np.zeros(B), np.zeros((B, 2)), np.where(ok, v0, 0.0)
    a_new, c_new = np.zeros((B, N)), np.zeros((B, N))
    n_up, n_dn, n_fail, req = np.zeros(B, np.int64), np.zeros(B, np.int64), np.zeros(B, np.int64), np.full(B, -np.inf)
    for i in range(1, N + 1):
        a_w = aP[:, i] - lag
        c_w = chord(allr, a_w, q_prev)
        hi = (v_prev + a_max * dt) * dt
        lo = np.maximum(0.0, v_prev - b_lim * dt) * dt
        up, dn = ok & (c_w > hi), ok & (c_w < lo)
        a_i = a_w.copy()
        fail = np.zeros(B, bool)
        ru = np.nonzero(up)[0]
        if len(ru):                                        # the chord is 0 at a_prev and above hi at a_w: halve the arc interval; the lower end keeps chord <= hi
            x0, x1 = a_prev[ru].copy(), a_w[ru].copy()
            for _ in range(iters):
                xm = 0.5 * (x0 + x1)
                big = chord(ru, xm, q_prev[ru]) > hi[ru]
                x1, x0 = np.where(big, xm, x1), np.where(big, x0, xm)
            a_i[ru] = x0
        rd = np.nonzero(dn)[0]
        if len(rd):                                        # the chord is below lo at a_w: go on along the path until it reaches lo, then halve; the upper end keeps chord >= lo
            x0 = a_w[rd].copy()
            x1 = np.maximum(a_w[rd], a_prev[rd] + lo[rd])
            for _ in range(200):
                c1 = chord(rd, x1, q_prev[rd])
                short = c1 < lo[rd]
                if not short.any():
                    break
                x1 = np.where(short, x1 + np.maximum(np.maximum(lo[rd] - c1, 0.25 * lo[rd]), 0.05), x1)
            fail[rd] = chord(rd, x1, q_prev[rd]) < lo[rd]
            for _ in range(iters):
                xm = 0.5 * (x0 + x1)
                short = chord(rd, xm, q_prev[rd]) < lo[rd]
                x0, x1 = np.where(short, xm, x0), np.where(short, x1, xm)
            a_i[rd] = x1
        q_i = _sample_rows(PP, aP, EXT, a_i)
        c_i = np.hypot(q_i[:, 0] - q_prev[:, 0], q_i[:, 1] - q_prev[:, 1])
        if i <= m:
            n_up += up.astype(np.int64)
            n_dn += dn.astype(np.int64)
            n_fail += fail.astype(np.int64)
            req = np.maximum(req, np.where(ok, (c_w / dt - v_prev) / dt, -np.inf))
        lag = aP[:, i] - a_i
        v_prev = c_i / dt
        a_prev, q_prev = a_i, q_i
        a_new[:, i - 1] = a_i
        c_new[:, i - 1] = c_i / dt
    out = p.copy()
    changed = np.zeros(B, bool)
    for b in range(B):
        keep = a_new[b] == aP[b, 1:]                       # an unchanged step is the original knot (bit for bit)
        if keep.all():
            continue
        pts = R.sample_path(PP[b], aP[b], EXT[b], a_new[b])
        pts[keep] = p64[b][keep]
        new = pts.astype(p.dtype)
        changed[b] = bool((new != p[b]).any())
        out[b] = new
    return (out if changed.any() else plan), {"n_up": n_up, "n_dn": n_dn, "req": req, "changed": changed, "v0": v0, "v_exec": c_new[:, :m], "n_fail": n_fail, "_a_new": a_new}
'''
w = rep(w, "\n\n\nclass AccelLimited:", CHORD + "\n\nclass AccelLimited:")
w = rep(w, 'out, info = accel_limit(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])',
        'out, info = (accel_limit_chord if STATE["aform"] == "chord" else accel_limit)(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])')
w = rep(w, "form=np.asarray(form), replan_steps=", 'form=np.asarray(form), alim_form=np.asarray(STATE["aform"]), replan_steps=')
w = rep(w, '    ap.add_argument("--alim-log", default="")\n', '    ap.add_argument("--alim-log", default="")\n    ap.add_argument("--alim-form", choices=("path", "chord"), default="path")\n')
w = rep(w, '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n',
        '    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))\n'
        '    if mine.alim_form != "path" and STATE["alim"] is None:\n'
        '        raise SystemExit("REFUSED: --alim-form chord without the two limits would change nothing")\n'
        '    STATE["aform"] = mine.alim_form\n')
compile(w, "v8f_t1_eval_reader_rule_ac.py", "exec")
save("v8f_t1_eval_reader_rule_ac.py", w)
WSHA = hashlib.sha256(w.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------- the pod queue
q = load("pod_feas10k_queue.sh")
q = rep(q, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# pod_feasc10k_queue.sh -- written by make_feasc10k.py from pod_feas10k_queue.sh. Card PREREG_V8F_PROOF_ADD56: the eleven rollouts of card ADD53 again\n"
        "# under the run names ..E5, ..E10, ..E5L, ..EID with the wrapper v8f_t1_eval_reader_rule_ac.py and --alim-form chord (the limit held on the executed knots). It waits for\n"
        "# the queue of card ADD55 to end. Markers START_FEASC10K.txt / FEASC10K_STOP.txt / FEASC10K_DONE.txt. The header lines below are those of the earlier queues.\n")
q = rep_code(q, "FEAS10K", "FEASC10K", 3)
q = rep_code(q, "v8f_t1_eval_reader_rule_a.py", "v8f_t1_eval_reader_rule_ac.py", 3)
q = rep_code(q, "small_feas", "small_feasc", 2)
q = rep_code(q, "f1cmp", "f1ecmp", 2)
q = rep_code(q, '--alim-log "$out/ticks/alim_log_$arm.npz")', '--alim-log "$out/ticks/alim_log_$arm.npz" --alim-form chord)', 1)
q = runs(q, 8)
q = rep_code(q, "CMP_R0AID.txt", "CMP_R0EID.txt", 1)
q = rep(q, 'log "queue N2 start (DRY=$DRY): card ADD53: eleven rollouts', 'log "queue N2 start (DRY=$DRY): card ADD56 (repeat of card ADD53, limit on the executed knots): eleven rollouts')
anchor = "for f in cmp_dumps.py v8f_t1_eval_reader_rule_ac.py LS_REF40_WINDOWS.json"
WAIT = ('if [ "$DRY" != 1 ]; then\n'
        '  t0=$(date +%s)\n'
        '  until [ -f "$W/ACCWIN10K_DONE.txt" ] || [ -f "$W/ACCWIN10K_STOP.txt" ]; do\n'
        '    [ $(( $(date +%s) - t0 )) -gt 21600 ] && stop "WAIT-ACCWIN-TIMEOUT"\n'
        '    sleep 30\n'
        '  done\n'
        '  log "the queue of card ADD55 has ended; the rollouts of card ADD56 start"\n'
        'fi\n')
q = rep(q, anchor, WAIT + anchor)
save("pod_feasc10k_queue.sh", q)

# ---------------------------------------------------------------- the chain
c = load("chain_feas10k.sh")
c = rep(c, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# chain_feasc10k.sh -- written by make_feasc10k.py from chain_feas10k.sh (card PREREG_V8F_PROOF_ADD56). The text below is the original apart from the names.\n")
for old, new in (("pod_feas10k_queue.sh", "pod_feasc10k_queue.sh"), ("v8f_t1_eval_reader_rule_a.py", "v8f_t1_eval_reader_rule_ac.py"), ("FEAS10K", "FEASC10K"), ("small_feas", "small_feasc"),
                 ("chain_feas10k.log", "chain_feasc10k.log"), ("queue_feas_stdout", "queue_feasc_stdout")):
    assert c.count(old) >= 1, old
    c = c.replace(old, new)
if RUN.search(c):
    c = runs(c)
save("chain_feasc10k.sh", c)

# ---------------------------------------------------------------- the reader
r = load("read_feas10k.py")
r = rep(r, '"""read_feas10k.py -- card PREREG_V8F_PROOF_ADD53 (FEAS10K) with its Amendment 1',
        '"""read_feasc10k.py -- written by make_feasc10k.py from read_feas10k.py for card PREREG_V8F_PROOF_ADD56: the repeat of card ADD53 with the limit held on the executed\n'
        'knots (run names E in place of A; names, bars, reads and controls of card ADD53; f2 also asks for the form chord in every limit log; r3 reports the steps whose\n'
        'lower bound was not reached on the path). Text of the source below.\n'
        'read_feas10k.py -- card PREREG_V8F_PROOF_ADD53 (FEAS10K) with its Amendment 1')
n_before = len(re.findall(r"A5|A10|AID", r))
r = re.sub(r"A5", "E5", r)
r = re.sub(r"A10", "E10", r)
r = re.sub(r"AID", "EID", r)
assert n_before > 30 and not re.search(r"I10(R0|RLc|RQ50)A", r), n_before
r = rep(r, '"small_feas"', '"small_feasc"')
r = rep(r, 'WRAPPER_SHA_PREFIX = "bf19862738459c84"', 'WRAPPER_SHA_PREFIX = "%s"' % WSHA[:16])
r = rep(r, '"result_file_steps": replan_of_result(a.pull, arm, k)}', '"result_file_steps": replan_of_result(a.pull, arm, k), "alim_form": (str(L["alim_form"]) if "alim_form" in L.files else None)}')
r = rep(r, 'f2[key]["result_file_steps"] == m_steps)', 'f2[key]["result_file_steps"] == m_steps and f2[key]["alim_form"] == "chord")')
r = rep(r, '        worst, rows_ok, first_ok, first_max = 0.0, True, True, 0.0\n',
        '        worst, rows_ok, first_ok, first_max = 0.0, True, True, 0.0\n'
        '        r3[key] = {tag: (int(np.asarray(L[tag + "_n_fail"]).sum()) if tag + "_n_fail" in L.files else None) for tag in ("m", "f")}\n')
r = rep(r, "    for arm, k in NEW_RUNS:\n        key = \"%s_s%d\" % (arm, k)\n        a_max, b_lim, m_steps = LIMITS[arm]\n",
        "    r3 = {}\n    for arm, k in NEW_RUNS:\n        key = \"%s_s%d\" % (arm, k)\n        a_max, b_lim, m_steps = LIMITS[arm]\n")
r = rep(r, '    R["r2_rule_forms_block_share"] = r2\n', '    R["r2_rule_forms_block_share"] = r2\n    R["r3_lower_bound_not_reached_steps"] = r3\n')
r = rep(r, '"ZZ TEST-ACCEL-LIMIT PASS 18/18" in t_txt', 'bool(re.search(r"ZZ TEST-ACCEL-LIMIT-CHORD PASS (\\d+)/\\1(?!\\d)", t_txt))')
compile(r, "read_feasc10k.py", "exec")
save("read_feasc10k.py", r)

# ---------------------------------------------------------------- the pod read and its waiter
for src, dst, head in (("podread_feas.sh", "podread_feasc.sh", "# podread_feasc.sh -- written by make_feasc10k.py from podread_feas.sh: the pod read of card PREREG_V8F_PROOF_ADD56 (repeat of card ADD53).\n"),
                       ("wait_podread_feas.sh", "wait_podread_feasc.sh", "# wait_podread_feasc.sh -- written by make_feasc10k.py from wait_podread_feas.sh (card PREREG_V8F_PROOF_ADD56).\n")):
    t = load(src)
    t = t.replace("FEAS10K_REUSED.json", "@@REUSED@@")
    for old, new in (("FEAS10K", "FEASC10K"), ("PODREAD-FEAS", "PODREAD-FEASC"), ("read_feas", "read_feasc"), ("small_feas", "small_feasc"), ("e1f", "e1g"),
                     ("feas_test", "feasc_test"), ("rule_a.py", "rule_ac.py"), ("TEST_ACCEL_LIMIT.txt", "TEST_ACCEL_LIMIT_CHORD.txt")):     # read_feas also covers podread_feas
        assert old in t or old in ("feas_test", "rule_a.py", "TEST_ACCEL_LIMIT.txt", "small_feas", "e1f"), (src, old)
        t = t.replace(old, new)
    t = t.replace("@@REUSED@@", "FEAS10K_REUSED.json")
    t = re.sub(r"A5", "E5", re.sub(r"A10", "E10", re.sub(r"AID", "EID", t)))
    t = rep(t, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n" + head)
    assert "podread_feasc.sh" in t and "feascc" not in t, dst
    assert not re.search(r"I10(R0|RLc|RQ50)A", t) and "rule_a.py" not in t
    save(dst, t)
print("wrapper sha256 %s" % WSHA)
