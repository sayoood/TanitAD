"""make_midlook_closing.py -- writes v8f_t1_eval_reader_rule_rzgfhqc.py from v8f_t1_eval_reader_rule_rzgfhq.py by asserted text edits (each old text exactly once).
Study "midlook-closing": the later looks of the rule (--rule-every below the replan interval) for the two forms that feed a closing speed, RRc and RRh. One refusal is widened, the
reader branch of mid_looks takes the forms RRc and RRh, form RRh feeds the reader's own closing output at a later look, and the mid-look log gets one new key (c_read).
Design: MIDLOOK_CLOSING_DESIGN.md. The parent, the pinned tool v8f_t1_eval.py and the distrule files are not changed. Standard library only.
Usage: python make_midlook_closing.py (in this folder). Refuses to overwrite the output (set MC_OVERWRITE_OWN=1 to rewrite it)."""
import ast
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py")
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqc.py")
PARENT_SHA16 = "92a8034cc0f7b0d2"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.\n'

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqc.py -- written by make_midlook_closing.py from v8f_t1_eval_reader_rule_rzgfhq.py (study "midlook-closing"): the later looks of the rule for RRc and RRh.
  --rule-every below the replan interval is accepted for the forms RLb, RLc, RR (as in the parent) and now also RRc and RRh. The form RRd is still REFUSED (its closing speed is a gap rate
  that needs an earlier read; the later looks do not carry one). Nothing else changes: with --rule-every at or above the replan interval, and for every later look of RLb, RLc, RR, the
  arrays are those of the parent (the later looks of RR feed closing 0 as before).
  At a later look q (frame j + q) of the forms RRc and RRh the reader branch of mid_looks is used (presence, gap through the map, from the logged tokens of frame j + q, as RR does):
    RRc  closing speed = the label closing at the clip and frame j + q (the mix branch closing == "label"); where the label has none and a lead is present, exactly 0 (the reader value of
         the form is the zero array, as at the tick).
    RRh  closing speed = the reader's own closing output (third output of the token reader times CLOS_SCALE_MS) on the same token window as the gap read at j + q: cache steps
         j + q - hist + 1 .. j + q of the window's own clip, frames at or before j + q only (new helper _reader_closing_at). NaN where that window does not exist; presence needs it.
    RR   unchanged: closing 0 at every later look.
  The mid-look log (rule_log_mid_<run>.npz) has one more key, c_read: the reader's closing read at the later look, logged in every reader form (NaN in the label forms), fed only in RRh.
Text of the parent (rzgfhq) follows.
v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.
'''

HELPER = r'''def _reader_closing_at(states, rd, e_off):
    """The reader's closing output (m/s, [B] float64; NaN where there is no complete history in the clip) on the logged tokens of frame j + e_off of each window: the same token window
    (cache steps e-hist+1 .. e of the window's own clip, frames at or before e only) and the same row rule as the gap read of _reader_at; reader_closing applies CLOS_SCALE_MS. It is
    the quantity the tick-level look of form RRh feeds on frame e (head_values_reader). Nothing but tokens is read."""
    B = len(states)
    c = np.full(B, np.nan)
    rows, toks = [], []
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):
            rows.append(i)
            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))
    if rows:
        c[rows] = reader_closing(rd, np.stack(toks))
    return c


'''

ANCHORS = []


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def main():
    if os.path.exists(OUT) and os.environ.get("MC_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % OUT)
    raw = open(SRC, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha[:16] != PARENT_SHA16:
        sys.exit("REFUSED: parent sha256 %s is not %s" % (sha[:16], PARENT_SHA16))
    t = raw.decode("utf-8")
    assert "\r" not in t
    n0 = t.count("\n")

    def E(old, new, name):
        nonlocal t
        t = once(t, old, new)
        ANCHORS.append(name)

    E(OLD_FIRST, HEADER, "C1 header docstring")
    E('MID_FIELDS = tuple(W.LOG_FIELDS) + ("q", "excess", "nchg")\n',
      'MID_FIELDS = tuple(W.LOG_FIELDS) + ("q", "excess", "nchg", "c_read")      # c_read: the reader closing read at the later look (midlook-closing)\n',
      "C2 MID_FIELDS gets c_read")
    E("def mid_looks(pl, plan):\n", HELPER + "def mid_looks(pl, plan):\n", "C3 helper _reader_closing_at before mid_looks")
    E('        if form == "RR":\n            p, g, okr = _reader_at(st, rd, q)\n',
      '        if form in ("RR", "RRc", "RRh"):\n            p, g, okr = _reader_at(st, rd, q)\n            c_read = _reader_closing_at(st, rd, q)       # logged in every reader form; fed only in RRh\n',
      "C4 mid_looks: reader branch takes RRc and RRh; reader closing read")
    E('            hg, hc = used.astype(np.float32).astype(np.float64), np.zeros(B, np.float64)\n',
      '            hg, hc = used.astype(np.float32).astype(np.float64), (c_read if form == "RRh" else np.zeros(B, np.float64))\n',
      "C5 mid_looks: closing of RRh is the reader read")
    E('            p_src, hg, hc = np.zeros(B, np.float64), np.full(B, np.nan), np.zeros(B, np.float64)\n',
      '            p_src, hg, hc = np.zeros(B, np.float64), np.full(B, np.nan), np.zeros(B, np.float64)\n            c_read = np.full(B, np.nan)\n',
      "C6 mid_looks: label forms log NaN as c_read")
    E('changed=res.changed, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0,',
      'changed=res.changed, g_src=g_src, c_src=c_src, c_read=c_read, p_src=p_src, v0=v0,',
      "C7 mid_looks: row carries c_read")
    E('        if f not in ("RLb", "RLc", "RR"):\n            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc or RR")\n',
      '        if f == "RRd":\n            raise SystemExit("REFUSED: --rule-every below the replan interval is not available for the rule form RRd (its closing speed is a gap rate over an earlier read; the later looks carry none)")\n'
      '        if f not in ("RLb", "RLc", "RR", "RRc", "RRh"):\n            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc, RR, RRc or RRh")\n',
      "C8 refusal widened to RRc and RRh, RRd named")
    ast.parse(t)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    out_sha = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
    print("wrote", OUT, out_sha, t.count("\n"), "lines (parent %d lines, sha256 %s)" % (n0, sha[:16]))
    print("anchors (%d):" % len(ANCHORS))
    for a in ANCHORS:
        print("  ", a)


if __name__ == "__main__":
    main()
