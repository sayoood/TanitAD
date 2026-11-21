"""make_wrapper_j.py -- writes v8f_t1_eval_reader_rule_rzgfhqetj.py from ../shufclose/v8f_t1_eval_reader_rule_rzgfhqet.py by asserted text edits (each old text exactly once).
Card PREREG_V8F_PROOF_ADD94 (GAPRULE10K). New argument --rule-join NPZ (default empty): the label source of the lead rule (X.STATE["label"], read by the rule planner only) is built from
the rule join when it is given and from --eval-join otherwise. The scoring does not read this object: --eval-join stays in the namespace handed to the pinned tool's cmd_run, and the
install call of the wrapper gets J = None, so no other LabelSource exists. The run record (rule log meta) gains rule_join and rule_join_sha256 (empty when the argument is not given).
A given rule join has to have the same clip names, off, n and lead_valid as the evaluation join (refused otherwise); a missing file is refused.
Usage: python make_wrapper_j.py [--parent PATH] [--out PATH]. Refuses a parent with another sha256 and refuses to overwrite the output (GJ_OVERWRITE_OWN=1 rewrites it).
Prints per edit the parent line and the new line, and a diff summary. Standard library only."""
import argparse
import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "shufclose", "v8f_t1_eval_reader_rule_rzgfhqet.py"))
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqetj.py")
PARENT_SHA16 = "02a58e655e5c1979"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhqet.py -- written by make_shuf_wrapper.py'
HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqetj.py -- written by make_wrapper_j.py from v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD94, study "gaprule").
  New flag --rule-join NPZ (default empty): the label source of the lead rule is built from this join; without it, from --eval-join as in the parent. Only the rule reads that source.
  The scoring keeps --eval-join (the namespace handed to the pinned tool is not changed). The rule log meta gains rule_join and rule_join_sha256.
  A rule join has to equal the evaluation join in clip names, off, n and lead_valid, else the run is refused.
  Every other behaviour is the parent's. Text of the parent (rzgfhqet) follows.
'''
RJ_HELPER = r'''def build_rule_label(a, rule_join, load_join):
    """-> LabelSource of the lead rule: from the rule join when one is given (checked against the evaluation join), else from the evaluation join (the parent's behaviour)."""
    J = load_join(a.eval_join)
    if not rule_join:
        return W.LabelSource(J)
    if not os.path.isfile(rule_join):
        raise SystemExit("REFUSED: --rule-join file not found: %s" % rule_join)
    JR = load_join(rule_join)
    for k in ("clip", "off", "n", "lead_valid"):
        if k not in J or k not in JR or not np.array_equal(np.asarray(J[k]), np.asarray(JR[k])):
            raise SystemExit("REFUSED: --rule-join differs from the evaluation join in %s" % k)
    print("rule join %s sha256 %s (the label source of the rule only; scoring keeps %s)" % (rule_join, W.sha256_file(rule_join), a.eval_join), flush=True)
    return W.LabelSource(JR)


'''


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def build(parent_text):
    t = parent_text
    assert chr(13) not in t
    parent = t
    anchors, plines = [], {}

    def E(old, new, name):
        nonlocal t
        plines[name] = parent[:parent.index(old)].count("\n") + 1
        t = once(t, old, new)
        anchors.append((name, new))

    E(OLD_FIRST, HEADER + OLD_FIRST.replace('"""v8f_t1_eval_reader_rule_rzgfhqet.py -- ', 'v8f_t1_eval_reader_rule_rzgfhqet.py -- '), "J1 header docstring")
    E("def main(argv=None):\n    argv = list(sys.argv[1:] if argv is None else argv)\n",
      "def main(argv=None):\n    argv = list(sys.argv[1:] if argv is None else argv)\n", "J2 anchor (unchanged text, used for ordering)")
    E('    ap.add_argument("--closing-table", default="")\n', '    ap.add_argument("--closing-table", default="")\n    ap.add_argument("--rule-join", default="")\n', "J3 flag --rule-join")
    E('    X.STATE["label"] = W.LabelSource(T.D.load_join(a.eval_join))\n',
      '    X.STATE["label"] = build_rule_label(a, mine.rule_join, T.D.load_join)\n', "J4 label source from the rule join")
    E('"eval_join": a.eval_join, "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "reader": mine.reader,',
      '"eval_join": a.eval_join, "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "rule_join": mine.rule_join,\n'
      '                    "rule_join_sha256": W.sha256_file(mine.rule_join) if mine.rule_join and os.path.isfile(mine.rule_join) else "", "reader": mine.reader,', "J5 run record fields")
    anchors.pop(1)
    plines.pop("J2 anchor (unchanged text, used for ordering)")
    idx = t.index("def _main_inner(")                    # the helper is a module-level function placed before _main_inner
    t = t[:idx] + RJ_HELPER + t[idx:]
    ast.parse(t)
    return t, anchors, plines


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--parent", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.environ.get("GJ_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % a.out)
    raw = open(a.parent, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha[:16] != PARENT_SHA16:
        sys.exit("REFUSED: parent sha256 %s is not %s" % (sha[:16], PARENT_SHA16))
    ptext = raw.decode("utf-8")
    t, anchors, plines = build(ptext)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    out_sha = hashlib.sha256(t.encode("utf-8")).hexdigest()
    print("wrote", a.out, out_sha[:16], t.count("\n"), "lines (parent %d lines, sha256 %s)" % (raw.count(b"\n"), sha[:16]))
    for name, new in anchors:
        key = new if len(new) < 400 else new[:200]
        print("   %-44s parent L%-5d new L%d" % (name, plines[name], t[:t.index(key)].count("\n") + 1))
    d = list(difflib.unified_diff(ptext.split("\n"), t.split("\n"), lineterm="", n=0))
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    dele = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    print("diff summary: %d lines added, %d lines removed (parent %d lines, new %d lines)" % (add, dele, ptext.count("\n"), t.count("\n")))
    print("sha256", out_sha)


if __name__ == "__main__":
    main()
