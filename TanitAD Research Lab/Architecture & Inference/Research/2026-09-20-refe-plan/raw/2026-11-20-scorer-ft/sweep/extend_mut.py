"""MM 2026-11-20: SCORER-FT copied P5-C code blocks into launch_gate.py / gate_checks.py, so five P5-C mutation targets now occur twice.
Make each target unique WITHOUT changing what it mutates: append to BOTH the target and its replacement the source text that follows the
target's FIRST occurrence (the P5-C block; the SCORER-FT copies come later in the file), whole lines at a time, until the target occurs exactly
once. Rewrites only the first two string literals of the tuples at the given line numbers.  usage: extend_mut.py <refe dir> <test.py> <lineno>..."""
import ast, json, pathlib, sys
R = pathlib.Path(sys.argv[1]); T = R / sys.argv[2]; LINES = {int(x) for x in sys.argv[3:]}
src = T.read_text(encoding="utf-8"); assert "\r" not in src
lines = src.split("\n")
edits = []
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Tuple) and node.lineno in LINES and len(node.elts) >= 3 and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts[:3]):
        f, old, new = (e.value for e in node.elts[:3])
        text = (R / f).read_text(encoding="utf-8")
        i = text.find(old); assert i >= 0, (node.lineno, "absent")
        j, ext, k = i + len(old), None, i + len(old) - 1
        while ext is None:                         # candidates in order: up to the next line end (without, then with, the newline)
            k = text.find("\n", k + 1); assert k >= 0, (node.lineno, "ran off the file")
            for cand in (text[j:k], text[j:k + 1]):
                if cand and text.count(old + cand) == 1:
                    ext = cand; break
        assert text.find(old + ext) == i, "the unique extension is not the P5-C (first) occurrence"
        mutated = text.replace(old + ext, new + ext, 1); assert mutated != text
        for e, val in ((node.elts[1], old + ext), (node.elts[2], new + ext)):
            assert e.lineno == e.end_lineno == node.lineno, "multi-line literal: edit by hand"
            edits.append((e.lineno, e.col_offset, e.end_col_offset, json.dumps(val, ensure_ascii=False)))
        print("EXTENDED", node.lineno, f, "by", len(ext), "chars:", repr(ext[:80]))
assert len(edits) == 2 * len(LINES), (len(edits), LINES)
for ln, c0, c1, lit in sorted(edits, key=lambda e: (e[0], -e[1])):
    s = lines[ln - 1].encode("utf-8")
    lines[ln - 1] = (s[:c0] + lit.encode("utf-8") + s[c1:]).decode("utf-8")
T.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("ZZEXTEND_OK", T.name, len(LINES))
