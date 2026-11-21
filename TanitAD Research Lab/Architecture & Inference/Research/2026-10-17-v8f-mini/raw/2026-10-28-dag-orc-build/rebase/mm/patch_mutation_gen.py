"""MM 2026-10-28: A1 added a second raw-sha check (AMEND1.md, mutant a1-amend-no-sha) with the same first line, so the pristine prereg-no-sha
anchor matched twice in the reader (ANCHOR ERROR). Narrow it to the PREREG check's own return line."""
p = "E:/Projects/TanitAD-artifacts/dagorc_rebase/rebase_mutation.py"
m = open(p, encoding="utf-8").read()
anchor = 'out = HERE / "read" / "dag_orc_mutation.py"\n'
assert m.count(anchor) == 1
old = """    ("prereg-no-sha", "prereg_problems skips the raw sha256", 'if hashlib.sha256(raw).hexdigest() != pin:', 'if False:'),"""
new = ("""    ("prereg-no-sha", "prereg_problems skips the raw sha256", 'if hashlib.sha256(raw).hexdigest() != pin:\\\\n        return [f"{path} is not the pinned"""
       """ PREREG_V8F_DAG_ORC.md', 'if False:\\\\n        return [f"{path} is not the pinned PREREG_V8F_DAG_ORC.md'),""")
add = ("# MM 2026-10-28: the AMEND1 sha check (A1) repeats the PREREG check's first line, so the pristine prereg-no-sha anchor matched twice"
       " (ANCHOR ERROR): narrow it to the PREREG check's own return line\n"
       "rep(" + repr(old) + ",\n    " + repr(new) + ")\n")
open(p, "w", encoding="utf-8", newline="\n").write(m.replace(anchor, add + anchor))
print("patched")
