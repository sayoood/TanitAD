"""make_read_n2b.py -- writes read_i10k_n2b.py from read_i10k_n2.py by ONE asserted text edit: where control k5 takes its two cells from the JSON of the pinned
evaluation tool. The reader was built against a test structure with the cells directly under intervals.metrics; the tool writes arms.cl.intervals.metrics.<cell>.mean
(the key path already written in K5_BANKED.json and used for the banked values). Cells, tolerance and the checkpoint hash check of k5 are unchanged.
The first reader stopped inside the controls function with KeyError 'intervals' before it printed any control value, name or outcome difference."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
t = io.open(os.path.join(HERE, "read_i10k_n2.py"), encoding="utf-8").read()
old = '            m = j["intervals"]["metrics"]\n'
new = ('            mm = j["arms"]["cl"]["intervals"]["metrics"]          # the pinned tool: arms.cl.intervals.metrics.<cell>.mean (K5_BANKED.json key_path)\n'
       '            m = {c: float(mm[c]["mean"]) for c in ("fde_last_m", "ade_dense_m")}\n')
assert t.count(old) == 1
t = t.replace(old, new)
io.open(os.path.join(HERE, "read_i10k_n2b.py"), "w", encoding="utf-8", newline="\n").write(t)
print("wrote read_i10k_n2b.py")
