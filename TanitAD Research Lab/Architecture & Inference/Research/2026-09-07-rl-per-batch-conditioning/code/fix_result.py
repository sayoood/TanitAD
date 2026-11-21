# -*- coding: utf-8 -*-
"""Correct one imprecise sentence in RESULT.md before it lands."""
import pathlib

p = pathlib.Path(r"C:\Users\Admin\tanitad-perbatch\pkg\RESULT.md")
s = p.read_text(encoding="utf-8")

OLD = ("  production RL script in the repo that builds one** \u2014 actually uses, and it "
       "plumbs five\n  of the twelve forward parameters (`nav_cmd`, `v0`, `lan` + "
       "`frames`/`steps`).")
NEW = ("  production RL script in the repo that builds one** \u2014 actually uses, and it "
       "passes\n  **three of the twelve conditioning channels** (`nav_cmd`, `v0`, `lan`; "
       "plus the\n  positional `frames` and explicit `steps`), which is the gap "
       "`refc_adapter` was created\n  to close in the first place.")

assert OLD in s, "RESULT.md anchor not found"
p.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
print("RESULT.md corrected")
