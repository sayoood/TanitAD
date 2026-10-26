"""SMOKE: stub tool trees for the chain's flag-guard mutation test.  Each stub is the REAL tool (tool_81aa3e36) with ONE guard-relevant fact removed; the chain must REFUSE each.
python make_stubs.py <out dir>"""
import os
import sys

SRC = "E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36/stack/scripts/v8f_t1_eval.py"
out = sys.argv[1]
s = open(SRC, encoding="utf-8", newline="").read()
M = {
    "no_dump_heads": ('add_argument("--dump-heads"', 'add_argument("--dump-hexds"'),
    "no_dump_ticks": ('add_argument("--dump-ticks"', 'add_argument("--dump-ticxs"'),
    "no_causal": ('LIMIT_MODES = ("logged", "off", "causal", "causal_all")', 'LIMIT_MODES = ("logged", "off", "causal_all")'),
    "dump_heads_only_in_docstring": ('p.add_argument("--dump-heads", action="store_true"', 'p.add_argument("--dumpheads", action="store_true"'),
    "broken_control_replan": ('add_argument("--replan-steps"', 'add_argument("--replan-stepz"'),
    "broken_control_limit": ('LIMIT_MODES = ("logged", "off", "causal", "causal_all")', 'LIMIT_MODES = ("off", "causal", "causal_all")'),
}
for name, (a, b) in M.items():
    assert s.count(a) == 1, (name, s.count(a))
    d = os.path.join(out, name, "stack", "scripts")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "v8f_t1_eval.py"), "w", encoding="utf-8", newline="").write(s.replace(a, b, 1))
    print("stub", name)
