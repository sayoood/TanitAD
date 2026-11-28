"""make_read_ip1c.py -- writes read_ip1_c.py from read_ip1.py by ONE asserted text edit (card PREREG_V8F_PROOF_ADD44, control c1, part head_in_run_equals_head_file):
the field qhead_sha256 is written by the read-out wrapper into the side file rule_log_IRQ_<checkpoint>_qh.npz, not into rule_log_IRQ_<checkpoint>.npz where the first reader
looked (logged in TIMES_NOTE.md before the rerun; no difference had been printed). Nothing else changes. Usage: python make_read_ip1c.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
raw = open(os.path.join(HERE, "read_ip1.py"), "rb").read()
assert bytes([13]) not in raw, "CR line ends"
t = raw.decode("utf-8")
old = '            rl = os.path.join(pull, "IRQ_%s" % b, "ticks", "rule_log_IRQ_%s.npz" % b)\n'
new = '            rl = os.path.join(pull, "IRQ_%s" % b, "ticks", "rule_log_IRQ_%s_qh.npz" % b)          # the side file of the read-out wrapper holds qhead_sha256\n'
assert t.count(old) == 1, t.count(old)
t = t.replace(old, new)
with io.open(os.path.join(HERE, "read_ip1_c.py"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(t)
print("wrote read_ip1_c.py; ZZ MAKE-READ-IP1C DONE")
