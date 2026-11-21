"""PR-4a: every EMPTY-CONTAINER return in alpamayo_records.py, enumerated.

⚠️ AN AST CENSUS IS AN ENUMERATION AID, NOT A GUARD. A census once read 0
suspects on BOTH the fixed and the broken trainer. Its job here is only to make
sure nothing is MISSED from the hand-classification in RESULT.md -- the
classification itself is human and is stated there with its reasoning.

The census carries its own CONTROL: it is run over the PRE-FIX file as well, and
must find MORE suspects there. A census that reports the same on both files has
measured nothing.

ASCII-only stdout.
usage: sibling_sweep.py <fixed.py> <prefix.py> <out.json>
"""
import ast
import io
import json
import sys

EMPTY = {"dict": "{}", "list": "[]", "set": "set()", "str": '""', "tuple": "()"}


def is_empty_literal(node):
    if isinstance(node, ast.Dict) and not node.keys:
        return "{}"
    if isinstance(node, ast.List) and not node.elts:
        return "[]"
    if isinstance(node, ast.Tuple) and not node.elts:
        return "()"
    if isinstance(node, ast.Constant) and node.value == "":
        return '""'
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    # {"clips": 0} -- a dict whose only content is a zero count
    if isinstance(node, ast.Dict) and len(node.keys) == 1:
        v = node.values[0]
        if isinstance(v, ast.Constant) and v.value == 0:
            return "{k: 0}"
    return None


def census(path):
    src = io.open(path, encoding="utf-8").read()
    assert src.strip(), "source must be READABLE -- a blank read is not a zero"
    tree = ast.parse(src)
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                lit = is_empty_literal(node.value)
                if lit:
                    hits.append({"function": fn.name, "line": node.lineno,
                                 "returns": lit})
            # a per-row swallow: `except: <assign empty>` or `except: continue`
            if isinstance(node, ast.ExceptHandler):
                for sub in node.body:
                    if isinstance(sub, ast.Assign) and is_empty_literal(sub.value):
                        hits.append({"function": fn.name, "line": sub.lineno,
                                     "returns": "except-assign %s"
                                                % is_empty_literal(sub.value)})
                    if isinstance(sub, ast.Continue):
                        hits.append({"function": fn.name, "line": sub.lineno,
                                     "returns": "except-continue"})
    return sorted(hits, key=lambda h: h["line"]), len(src)


fixed, nf = census(sys.argv[1])
prefix, npf = census(sys.argv[2])
print("FIXED  file %s  bytes=%d  suspects=%d" % (sys.argv[1][-40:], nf, len(fixed)))
for h in fixed:
    print("   line %-4d %-22s -> %s" % (h["line"], h["function"], h["returns"]))
print()
print("PRE-FIX (CONTROL) bytes=%d  suspects=%d" % (npf, len(prefix)))
for h in prefix:
    print("   line %-4d %-22s -> %s" % (h["line"], h["function"], h["returns"]))
print()
same = ([(h["function"], h["returns"]) for h in fixed]
        == [(h["function"], h["returns"]) for h in prefix])
print("CONTROL: census differs between the two files: %s  (must be True, or the"
      " census is reading nothing)" % (not same))
json.dump({"fixed": fixed, "prefix": prefix, "census_discriminates": not same},
          open(sys.argv[3], "w", encoding="utf-8"), indent=1)
print("WROTE %s" % sys.argv[3])
