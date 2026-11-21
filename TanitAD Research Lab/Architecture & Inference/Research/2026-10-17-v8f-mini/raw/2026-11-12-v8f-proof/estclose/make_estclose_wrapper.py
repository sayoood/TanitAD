"""make_estclose_wrapper.py -- writes v8f_t1_eval_reader_rule_rzgfhqe.py from v8f_t1_eval_reader_rule_rzgfhqc.py by asserted text edits (each old text exactly once).
Card PREREG_V8F_PROOF_ADD79 (ESTCLOSE10K): the form RRe, the lead rule of form RR (reader presence, gap through the q50 map) whose closing-speed input is an ESTIMATE computed from the
token reader's own per-frame reads over the last H frames (estimator file of closeread/closest.py, new flag --closing-est). Design: ESTCLOSE_DESIGN.md.
The parent, the pinned tool, the distrule files and closest.py are not changed. Standard library only.
Usage: python make_estclose_wrapper.py [--parent PATH] [--out PATH]   (defaults: ../integ10k/v8f_t1_eval_reader_rule_rzgfhqc.py -> ./v8f_t1_eval_reader_rule_rzgfhqe.py)
Refuses when the parent hash differs, refuses to overwrite the output (set EC_OVERWRITE_OWN=1 to rewrite it). Prints, per edit, the parent line and the line of the new text."""
import argparse
import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "integ10k", "v8f_t1_eval_reader_rule_rzgfhqc.py"))
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqe.py")
PARENT_SHA16 = "8732329b4476347d"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhqc.py -- written by make_midlook_closing.py from v8f_t1_eval_reader_rule_rzgfhq.py (study "midlook-closing"): the later looks of the rule for RRc and RRh.\n'

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqe.py -- written by make_estclose_wrapper.py from v8f_t1_eval_reader_rule_rzgfhqc.py (card ADD79, study "estclose"): the form RRe.
  RRe  the lead rule of form RR (presence = reader probability >= 0.5 and a full five-frame history; gap = min(g_reader, Q(g_reader)), Q the q50 map) with the closing-speed input of the
       rule ESTIMATED from the token reader's own reads over the last H frames. New flag --closing-est NPZ (required for RRe, refused for every other form): an estimator file of
       closest.py (ClosingEstimator; arm L = ridge or N = mean of small networks; H frames).
  At EVERY look of the rule (the replan tick and every later look of --rule-every) at frame j (a later look: j + q), in the main phase and in the pre-roll:
       X [H, 3] = the reader's reads on the five-frame windows that end at the frames j-H+1 .. j, oldest first, per frame (gap read before any map, in m; presence p; closing output
       times CLOS_SCALE_MS = 10 m/s): the quantities of closeread_dump.py; closing input = ClosingEstimator.apply(X), in m/s, not clipped (the rule forms v_L = max(0, v_l + closing)).
       The estimate needs the frames j - need_frames + 1 .. j of the clip (need_frames = H + 4). If they are not all in the clip the closing input is exactly 0.0 (the behaviour of form RR)
       and est_ok is False. The reads come from the logged tokens of the clip (cache steps at or before the look frame only), cached per (clip, frame).
  Logs: rule_log_<run>.npz gets the arrays c_est (float64, NaN where there is no full history) and est_ok (bool), row for row with the other arrays; rule_log_mid_<run>.npz the same two
       keys; rule_log_pre_<run>.npz the same two keys for the pre-roll rows; the side file rule_log_<run>_rd.npz carries them too. The run text run_<run>.txt (same folder) gets one line
       "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256 of the estimator file>" (scored = tick and later looks of the main phase, preroll = those of the pre-roll).
  Every other form behaves exactly as in the parent: the new text is reached only when the form is RRe (the only unconditional additions are a no-op call at the end of main and an
  argparse flag that is refused with every other form).
Text of the parent (rzgfhqc) follows.
'''

HELPERS = r'''MID_EST = ("c_est", "est_ok")                       # the two keys the later-look log gets in form RRe (estclose)


def _find_closest():
    for c in (os.environ.get("CLOSEST_DIR", ""), HERE, os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "closeread")):
        if c and os.path.isfile(os.path.join(c, "closest.py")):
            return os.path.normpath(c)
    raise SystemExit("REFUSED: closest.py (the closing estimator format of card ADD79) was not found; set CLOSEST_DIR")


def load_estimator(path, rd, rule_log=""):
    """Loads the estimator file of form RRe (closest.ClosingEstimator) and opens the caches and the log columns of the form. Refuses a reader history other than the one the file format has."""
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --closing-est file not found: %s" % path)
    d = _find_closest()
    if d not in sys.path:
        sys.path.insert(0, d)
    import closest as CL
    est = CL.ClosingEstimator.load(path)
    if int(rd.hist) != int(CL.READER_HIST):
        raise SystemExit("REFUSED: the reader history is %d frames, the estimator format has %d" % (int(rd.hist), int(CL.READER_HIST)))
    STATE["est"] = {"est": est, "path": path, "sha256": W.sha256_file(path), "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}
    for k in MID_EST:
        STATE["cols"][k] = []
        STATE["mid"]["rows"][k] = []
    return est


def _head3(rd, E):
    """The head of the token reader on frame embeddings E [n, hist, 64]: (gap_m before any map, presence p, closing_ms), each [n] float32, exactly the quantities of closeread_dump.py
    (TokReader.head_from_embeds for p and the gap, reader_closing for the closing output; CLOS_SCALE_MS applied here)."""
    import tok_reader as TR
    x = np.asarray(E).reshape(len(E), -1)
    h = TR._gelu(x @ rd.w["head.mlp.0.weight"].T + rd.w["head.mlp.0.bias"])
    o = h @ rd.w["head.mlp.2.weight"].T + rd.w["head.mlp.2.bias"]
    assert o.shape[1] == 3, o.shape
    p = (1.0 / (1.0 + np.exp(-o[:, 0].astype(np.float64)))).astype(np.float32)
    g = (o[:, 1] * TR.GAP_SCALE_M).astype(np.float32)
    c = (o[:, 2].astype(np.float64) * CLOS_SCALE_MS).astype(np.float32)
    return g, p, c


def _est_fill(rd, wanted):
    """wanted: {(clip, frame): feats array of that clip}. Fills STATE["est"]["reads"][(clip, frame)] = (gap, p, closing) of the five-frame window ending at that frame. The frame embeddings
    (the per-frame part of the reader) are cached per (clip, frame), so windows that overlap share them. With STATE["est"]["cache"] False both caches are emptied first."""
    S = STATE["est"]
    if not S["cache"]:
        S["emb"].clear()
        S["reads"].clear()
    need_e = {}
    for (ci, f), feats in wanted.items():
        if (ci, f) in S["reads"]:
            continue
        for ff in range(f - rd.hist + 1, f + 1):
            if (ci, ff) not in S["emb"]:
                need_e[(ci, ff)] = feats
    keys = list(need_e)
    for i in range(0, len(keys), 64):
        part = keys[i:i + 64]
        E = rd.frame_embed(np.stack([np.asarray(need_e[k][k[1]], np.float32) for k in part]))
        S["embed_calls"] += 1
        for k, e_ in zip(part, E):
            S["emb"][k] = np.asarray(e_)
    todo = [k for k in wanted if k not in S["reads"]]
    if todo:
        win = np.stack([np.stack([S["emb"][(ci, f - rd.hist + 1 + t)] for t in range(rd.hist)]) for (ci, f) in todo])
        g, p, c = _head3(rd, win)
        for k, a, b, d in zip(todo, g, p, c):
            S["reads"][k] = (float(a), float(b), float(d))


def _est_at(states, rd, e_off):
    """The closing estimate (m/s) at the look on frame e = j + e_off of each window: X [H, 3] = the reads (gap, presence, closing) of the windows ending at frames e-H+1 .. e, oldest first,
    through the estimator. -> c_est [B] float64 (NaN where the frames e - need_frames + 1 .. e are not all in the clip) and est_ok [B] bool (full history and a finite estimate)."""
    S = STATE["est"]
    est = S["est"]
    H, need = int(est.H), int(est.need_frames(rd.hist))
    B = len(states)
    c_est, est_ok = np.full(B, np.nan), np.zeros(B, bool)
    rows, wanted = [], {}
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - need + 1 >= 0 and e < len(s.d["feats"]):
            rows.append((i, int(s.ci), e))
            for f in range(e - H + 1, e + 1):
                wanted[(int(s.ci), f)] = s.d["feats"]
    if rows:
        _est_fill(rd, wanted)
        X = np.stack([np.array([S["reads"][(ci, f)] for f in range(e - H + 1, e + 1)], np.float64) for _, ci, e in rows])
        v = est.apply(X)
        idx = [i for i, _, _ in rows]
        c_est[idx] = v
        est_ok[idx] = np.isfinite(v)
    return c_est, est_ok


def est_fed(c_est, est_ok):
    """the closing input of the rule: the estimate where there is one, exactly 0.0 where not (the closing input of form RR)."""
    return np.where(est_ok, c_est, 0.0)


def _est_tick(n):
    """the closing input at the tick-level look (frame j of each window); logs the estimate and its flag, one entry per call, row for row with the rule log."""
    states = STATE.get("est_states")
    if states is None or len(states) != n:
        raise RuntimeError("the closing estimate needs the states of this batch (ReaderPlanner not bound)")
    c_est, ok = _est_at(states, STATE["reader"], 0)
    STATE["cols"]["c_est"].append(c_est)
    STATE["cols"]["est_ok"].append(ok)
    return est_fed(c_est, ok)


def est_tick_arrays(log):
    """c_est and est_ok of the tick-level looks of the main phase, as arrays of the length of the rule log."""
    c = np.concatenate(STATE["cols"]["c_est"]) if STATE["cols"]["c_est"] else np.zeros(0)
    ok = np.concatenate(STATE["cols"]["est_ok"]) if STATE["cols"]["est_ok"] else np.zeros(0, bool)
    n = len(log.arrays()["ci"])
    if len(c) != n or len(ok) != n:
        raise RuntimeError("estimate columns have %d / %d rows, the rule log %d" % (len(c), len(ok), n))
    return {"c_est": c, "est_ok": ok}


def _count_ok(parts):
    full = short = 0
    for lst in parts:
        for a in lst:
            a = np.asarray(a, bool)
            full += int(a.sum())
            short += int(a.size - a.sum())
    return full, short


def est_line():
    """the text line of the run file: counts of the looks with a full history (est_ok) and without, scored (tick and later looks of the main phase) and pre-roll; None outside form RRe."""
    S = STATE.get("est")
    if S is None or not S.get("rule_log"):
        return None
    sf, ss = _count_ok([STATE["cols"]["est_ok"], STATE["mid"]["rows"]["est_ok"]])
    runner = STATE.get("runner")
    pf, ps = _count_ok([part for kind, post in (runner.posts if runner is not None else []) if kind == "model" for part in (post["cols"]["est_ok"], post["mid"]["rows"]["est_ok"])])
    return "est looks scored full %d short %d preroll full %d short %d file %s" % (sf, ss, pf, ps, S["sha256"])


def write_est_line():
    """appends est_line() to run_<run>.txt, the file beside the rule log (the queue writes its own lines there); nothing in any other form."""
    line = est_line()
    if line is None:
        return
    d, base = os.path.split(os.path.abspath(STATE["est"]["rule_log"]))
    run = base[len("rule_log_"):-4] if base.startswith("rule_log_") and base.endswith(".npz") else os.path.splitext(base)[0]
    with open(os.path.join(d, "run_" + run + ".txt"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


'''

ANCHORS = []
PARENT_LINES = {}
EXTENTS = {}


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def build(parent_text):
    """-> (new text, anchors [(name, new text)], parent_lines {name: first line}, extents {name: (first line, lines)}) from the text of the parent."""
    ANCHORS.clear()
    PARENT_LINES.clear()
    EXTENTS.clear()
    t = parent_text
    assert chr(13) not in t
    parent = t

    def E(old, new, name):
        nonlocal t
        PARENT_LINES[name] = parent[:parent.index(old)].count("\n") + 1
        EXTENTS[name] = (PARENT_LINES[name], old.count(chr(10)))
        t = once(t, old, new)
        ANCHORS.append((name, new))

    E(OLD_FIRST, HEADER + OLD_FIRST.replace('"""v8f_t1_eval_reader_rule_rzgfhqc.py -- ', 'v8f_t1_eval_reader_rule_rzgfhqc.py -- '), "X1 header docstring")
    E('FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh", "RRd")\n', 'FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh", "RRd", "RRe")\n', "X2 FORMS gets RRe")
    E('    elif form in ("RR", "RRc", "RRh", "RRd"):\n        used = np.minimum(', '    elif form in ("RR", "RRc", "RRh", "RRd", "RRe"):\n        used = np.minimum(', "X3 feed: RRe gap = min(g, Q(g)) as RR")
    E("def head_values_reader(model, feats, frame_ok=None, hist=None):\n", HELPERS + "def head_values_reader(model, feats, frame_ok=None, hist=None):\n", "X4 estimator helpers before head_values_reader")
    E('    c_fed = np.zeros(len(p), np.float32)\n    if STATE["form"] == "RRd":\n',
      '    c_fed = np.zeros(len(p), np.float32)\n    c_estv = _est_tick(len(p)) if STATE["form"] == "RRe" else None      # form RRe: the closing estimate of the tick-level look\n    if STATE["form"] == "RRd":\n',
      "X5 head_values_reader: the estimate at the tick-level look")
    E('    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else c_fed), ok\n',
      '    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else (c_estv if c_estv is not None else c_fed)), ok\n',
      "X6 head_values_reader: RRe returns the estimate as closing")
    E('        plan = X.MixPlanner.__call__(self, batch)\n', '        if STATE["form"] == "RRe":\n            STATE["est_states"] = self.states               # the estimate reads the logged tokens of the earlier frames of these windows\n        plan = X.MixPlanner.__call__(self, batch)\n',
      "X7 ReaderPlanner: states for the estimate")
    E('        if form in ("RR", "RRc", "RRh"):\n            p, g, okr = _reader_at(st, rd, q)\n', '        if form in ("RR", "RRc", "RRh", "RRe"):\n            p, g, okr = _reader_at(st, rd, q)\n',
      "X8 mid_looks: reader branch takes RRe")
    E('            hg, hc = used.astype(np.float32).astype(np.float64), (c_read if form == "RRh" else np.zeros(B, np.float64))\n',
      '            c_est, est_ok = _est_at(st, rd, q) if form == "RRe" else (None, None)       # form RRe: the estimate at frame j + q\n'
      '            hg, hc = used.astype(np.float32).astype(np.float64), (c_read if form == "RRh" else (est_fed(c_est, est_ok) if form == "RRe" else np.zeros(B, np.float64)))\n',
      "X9 mid_looks: closing of RRe is the estimate")
    E('        for f in MID_FIELDS:\n            M["rows"][f].append(np.asarray(row[f]))\n',
      '        if form == "RRe":\n            row["c_est"], row["est_ok"] = c_est, est_ok\n        for f in MID_FIELDS + (MID_EST if form == "RRe" else ()):\n            M["rows"][f].append(np.asarray(row[f]))\n',
      "X10 mid_looks: rows carry c_est and est_ok")
    E('    cols = {f: (np.concatenate(M["rows"][f]) if M["rows"][f] else np.zeros(0)) for f in MID_FIELDS}\n',
      '    cols = {f: (np.concatenate(M["rows"][f]) if M["rows"][f] else np.zeros(0)) for f in MID_FIELDS + (MID_EST if STATE["form"] == "RRe" else ())}\n',
      "X11 save_mid writes c_est and est_ok")
    E('        cols["pre_segment"] = np.concatenate(segcol) if segcol else np.zeros(0, np.int64)\n',
      '        cols["pre_segment"] = np.concatenate(segcol) if segcol else np.zeros(0, np.int64)\n'
      '        if STATE["form"] == "RRe":                     # the pre-roll rows get their own c_est / est_ok (the main-phase arrays of the rule log file are not theirs)\n'
      '            meta.pop("c_est", None)\n            meta.pop("est_ok", None)\n'
      '            for kk in MID_EST:\n'
      '                parts = [x for kind, post in runner.posts if kind == "model" for x in post["cols"][kk]]\n'
      '                cols[kk] = np.concatenate([np.asarray(x) for x in parts]) if parts else np.zeros(0)\n'
      '                assert len(cols[kk]) == len(cols["ci"]), (kk, len(cols[kk]), len(cols["ci"]))\n',
      "X12 finish_preroll: pre-roll rows carry their own c_est / est_ok")
    E('    ap.add_argument("--preroll-exec", choices=("plan", "logged"), default="plan")\n',
      '    ap.add_argument("--preroll-exec", choices=("plan", "logged"), default="plan")\n    ap.add_argument("--closing-est", default="")\n', "X13 flag --closing-est")
    E('    mine, rest = ap.parse_known_args(argv)\n',
      '    mine, rest = ap.parse_known_args(argv)\n'
      '    if mine.form == "RRe" and not mine.closing_est:\n        raise SystemExit("REFUSED: form RRe needs --closing-est NPZ (an estimator file of closest.py)")\n'
      '    if mine.form != "RRe" and mine.closing_est:\n        raise SystemExit("REFUSED: --closing-est belongs to form RRe only")\n',
      "X14 refusals of --closing-est")
    E('        if f not in ("RLb", "RLc", "RR", "RRc", "RRh"):\n            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc, RR, RRc or RRh")\n',
      '        if f not in ("RLb", "RLc", "RR", "RRc", "RRh", "RRe"):\n            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc, RR, RRc, RRh or RRe")\n',
      "X15 refusal of later looks admits RRe")
    E('    if f in GAP_PLANT or f in ("RR", "RRc", "RRh", "RRd"):\n', '    if f in GAP_PLANT or f in ("RR", "RRc", "RRh", "RRd", "RRe"):\n', "X16 reader needed by RRe")
    E('        if f in ("RR", "RRc", "RRh", "RRd"):\n            if not mine.qmap', '        if f in ("RR", "RRc", "RRh", "RRd", "RRe"):\n            if not mine.qmap', "X17 q map needed by RRe")
    E('        if f == "RRd":\n            if mine.slope is None or not (0.0 < mine.slope <= 2.0) or mine.k_diff < 1:\n',
      '        if f == "RRe":\n            load_estimator(mine.closing_est, STATE["reader"], mine.rule_log)\n'
      '        if f == "RRd":\n            if mine.slope is None or not (0.0 < mine.slope <= 2.0) or mine.k_diff < 1:\n',
      "X18 load the estimator")
    E('        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head"}.get(f, "zero"))\n',
      '        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head", "RRe": "head"}.get(f, "zero"))\n', "X19 mix closing of RRe is the head value (the estimate)")
    E('            log.save(mine.rule_log, meta)\n',
      '            log.save(mine.rule_log, dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta)\n', "X20 rule log carries c_est and est_ok")
    E('    finally:\n        finish_preroll()\n', '    finally:\n        finish_preroll()\n        write_est_line()\n', "X21 run text line at the end of main")
    ast.parse(t)
    return t, list(ANCHORS), dict(PARENT_LINES), dict(EXTENTS)



def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--parent", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.environ.get("EC_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % a.out)
    raw = open(a.parent, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha[:16] != PARENT_SHA16:
        sys.exit("REFUSED: parent sha256 %s is not %s" % (sha[:16], PARENT_SHA16))
    t, anchors, plines, _ = build(raw.decode("utf-8"))
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    out_sha = hashlib.sha256(t.encode("utf-8")).hexdigest()
    print("wrote", a.out, out_sha[:16], t.count("\n"), "lines (parent %d lines, sha256 %s)" % (raw.count(b"\n"), sha[:16]))
    print("edits (%d): parent line -> new line" % len(anchors))
    for name, new in anchors:
        key = new if len(new) < 400 else new[:200]
        print("   %-62s parent L%-5d new L%d" % (name, plines[name], t[:t.index(key)].count("\n") + 1))
    print("sha256", out_sha)


if __name__ == "__main__":
    main()
