"""make_shuf_wrapper.py -- writes v8f_t1_eval_reader_rule_rzgfhqet.py from v8f_t1_eval_reader_rule_rzgfhqe.py by asserted text edits (each old text exactly once).
Card PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K): the form RRe gets a second source of its closing-speed input, a TABLE by clip index and frame (new flag --closing-table NPZ), so that the
estimate of another clip (the surrogate of the card) can be fed to the rule. Everything else is the parent's text. Pattern of estclose/make_estclose_wrapper.py.
Usage: python make_shuf_wrapper.py [--parent PATH] [--out PATH]   (defaults: ../estclose/v8f_t1_eval_reader_rule_rzgfhqe.py -> ./v8f_t1_eval_reader_rule_rzgfhqet.py)
Refuses when the parent hash differs, refuses to overwrite the output (set SC_OVERWRITE_OWN=1 to rewrite it). Prints, per edit, the parent line and the line of the new text.
Standard library only."""
import argparse
import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "estclose", "v8f_t1_eval_reader_rule_rzgfhqe.py"))
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqet.py")
PARENT_SHA16 = "a747abeb51ef041f"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhqe.py -- written by make_estclose_wrapper.py'

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqet.py -- written by make_shuf_wrapper.py from v8f_t1_eval_reader_rule_rzgfhqe.py (card ADD83, study "shufclose").
  New flag --closing-table NPZ: admitted only with form RRe and exclusive with --closing-est. The table has the arrays clip_index (int), frame (int) and est (float, m/s; NaN where there is no
  value). With it the estimate at a look is the table value at (clip index of the window, frame of the look): the same clip index (s.ci) and the same frame (j for the tick look, j + q for a later
  look) that _est_at uses. A look whose (clip index, frame) has no row in the table, or whose row holds NaN or an infinite value, has est_ok False, c_est NaN and the closing input exactly 0.0
  (the behaviour of form RR). Nothing else of the form changes: the columns c_est / est_ok of the rule log, the later-look log and the pre-roll log, and the run-text line
  "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>" keep their format; the file hash in the line is that of the table. The table path reads no reader
  features and no estimator file; the reader of the rule (presence and gap) is unchanged.
  Every other form, and form RRe with --closing-est, behave exactly as in the parent.
Text of the parent (rzgfhqe) follows.
'''

TABLE_HELPERS = r'''TABLE_STRIDE = 1 << 20                              # key of the table = clip index * TABLE_STRIDE + frame (frames of a clip are far below the stride)


def load_table(path, rule_log=""):
    """Loads the table of form RRe with --closing-table (arrays clip_index, frame, est) and opens the log columns of the form. Refuses a file without the arrays, rows of different
    lengths, a frame outside [0, TABLE_STRIDE), a negative clip index or two rows with the same (clip index, frame)."""
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --closing-table file not found: %s" % path)
    with np.load(path, allow_pickle=False) as z:
        for k in ("clip_index", "frame", "est"):
            if k not in z.files:
                raise SystemExit("REFUSED: --closing-table file lacks the array %s" % k)
        ci, fr, v = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["frame"]).astype(np.int64), np.asarray(z["est"]).astype(np.float64)
    if not (ci.ndim == fr.ndim == v.ndim == 1 and len(ci) == len(fr) == len(v)):
        raise SystemExit("REFUSED: --closing-table arrays are not one-dimensional of one length")
    if len(ci) and (ci.min() < 0 or fr.min() < 0 or fr.max() >= TABLE_STRIDE):
        raise SystemExit("REFUSED: --closing-table has a negative clip index or a frame outside [0, %d)" % TABLE_STRIDE)
    key = ci * TABLE_STRIDE + fr
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    if len(key) > 1 and bool(np.any(key[1:] == key[:-1])):
        raise SystemExit("REFUSED: --closing-table has two rows with the same clip index and frame")
    STATE["est"] = {"est": None, "table": {"key": key, "val": v}, "path": path, "sha256": W.sha256_file(path), "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}
    for k in MID_EST:
        STATE["cols"][k] = []
        STATE["mid"]["rows"][k] = []
    return STATE["est"]["table"]


def _table_at(states, e_off):
    """The closing input of form RRe taken from the table at the look on frame e = j + e_off of each window: the value at (clip index, e). -> c_est [B] float64 (NaN where the table has no
    row or a value that is not finite) and est_ok [B] bool."""
    T = STATE["est"]["table"]
    B = len(states)
    c_est, est_ok = np.full(B, np.nan), np.zeros(B, bool)
    if B and len(T["key"]):
        ci = np.array([int(s.ci) for s in states], np.int64)
        e = np.array([int(s.j) + int(e_off) for s in states], np.int64)
        inside = (ci >= 0) & (e >= 0) & (e < TABLE_STRIDE)
        key = np.where(inside, ci * TABLE_STRIDE + e, -1)
        pos = np.searchsorted(T["key"], key)
        posc = np.minimum(pos, len(T["key"]) - 1)
        hit = inside & (pos < len(T["key"])) & (T["key"][posc] == key)
        v = np.where(hit, T["val"][posc], np.nan)
        est_ok = hit & np.isfinite(v)
        c_est = np.where(est_ok, v, np.nan)
    return c_est, est_ok


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

    E(OLD_FIRST, HEADER + OLD_FIRST.replace('"""v8f_t1_eval_reader_rule_rzgfhqe.py -- ', 'v8f_t1_eval_reader_rule_rzgfhqe.py -- '), "Y1 header docstring")
    E("def _est_at(states, rd, e_off):\n", TABLE_HELPERS + "def _est_at(states, rd, e_off):\n", "Y2 table helpers before _est_at")
    E('    S = STATE["est"]\n    est = S["est"]\n    H, need = int(est.H), int(est.need_frames(rd.hist))\n',
      '    S = STATE["est"]\n    if S.get("table") is not None:                       # --closing-table: the value of the table at (clip index, frame of the look)\n        return _table_at(states, e_off)\n'
      '    est = S["est"]\n    H, need = int(est.H), int(est.need_frames(rd.hist))\n',
      "Y3 _est_at reads the table when there is one")
    E('    ap.add_argument("--closing-est", default="")\n', '    ap.add_argument("--closing-est", default="")\n    ap.add_argument("--closing-table", default="")\n', "Y4 flag --closing-table")
    E('    if mine.form == "RRe" and not mine.closing_est:\n        raise SystemExit("REFUSED: form RRe needs --closing-est NPZ (an estimator file of closest.py)")\n'
      '    if mine.form != "RRe" and mine.closing_est:\n        raise SystemExit("REFUSED: --closing-est belongs to form RRe only")\n',
      '    if mine.closing_est and mine.closing_table:\n        raise SystemExit("REFUSED: --closing-est and --closing-table together (one source of the closing input only)")\n'
      '    if mine.form == "RRe" and not (mine.closing_est or mine.closing_table):\n        raise SystemExit("REFUSED: form RRe needs --closing-est NPZ (an estimator file of closest.py) or --closing-table NPZ")\n'
      '    if mine.form != "RRe" and mine.closing_est:\n        raise SystemExit("REFUSED: --closing-est belongs to form RRe only")\n'
      '    if mine.form != "RRe" and mine.closing_table:\n        raise SystemExit("REFUSED: --closing-table belongs to form RRe only")\n',
      "Y5 refusals of --closing-table")
    E('        if f == "RRe":\n            load_estimator(mine.closing_est, STATE["reader"], mine.rule_log)\n',
      '        if f == "RRe":\n            if mine.closing_table:\n                load_table(mine.closing_table, mine.rule_log)\n            else:\n                load_estimator(mine.closing_est, STATE["reader"], mine.rule_log)\n',
      "Y6 load the table")
    ast.parse(t)
    return t, list(ANCHORS), dict(PARENT_LINES), dict(EXTENTS)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--parent", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.environ.get("SC_OVERWRITE_OWN") != "1":
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
        print("   %-46s parent L%-5d new L%d" % (name, plines[name], t[:t.index(key)].count("\n") + 1))
    print("sha256", out_sha)


if __name__ == "__main__":
    main()
