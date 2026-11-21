#!/usr/bin/env python
"""make_save_n1.py -- note N1 of card ADD79: writes closeread_save_n1.py from closeread_save.py by ONE counted text edit (the float bar of the table sub-check per arm)."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST = os.path.join(HERE, "closeread_save.py"), os.path.join(HERE, "closeread_save_n1.py")
raw = io.open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("c15066be6b1bd95d"), "closeread_save.py is not the reviewed file"
s = raw.decode("utf-8")
a = 'checks["e0d2_" + tag], diffs["e0d2_" + tag] = bool(same and d_tab <= 1e-12), d_tab'
b = 'checks["e0d2_" + tag], diffs["e0d2_" + tag] = bool(same and d_tab <= (1e-12 if arm == "L" else 1e-5)), d_tab      # note N1 of card ADD79: float32 forward on two batch sizes'
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("wrote closeread_save_n1.py", hashlib.sha256(io.open(DST, "rb").read()).hexdigest()[:16])
