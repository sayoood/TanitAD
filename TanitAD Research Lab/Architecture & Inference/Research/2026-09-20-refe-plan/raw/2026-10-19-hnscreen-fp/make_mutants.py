"""Build three mutant trees from work/: each reverts ONE part of the fix in launch_gate.py (and leaves every test as shipped).
  mut_a   revert rule (a): the "not filled" refusal (the empty / incomplete dict again runs no check)
  mut_b   revert rule (b): an expected flag absent from the argv is skipped again (`continue`, no refusal)
  mut_ab  both reverted == the original launch_gate.py's behaviour"""
import os
import shutil

root = r"C:\Users\Admin\par18\hnfp"
src = os.path.join(root, "work")
gate = os.path.join(src, "refe", "launch_gate.py")
s = open(gate, "rb").read().decode("utf-8")
A_OLD = "    if fp_unfilled:\n"
A_NEW = "    if False:  # MUTANT-A: rule (a) reverted\n"
B_OLD = """            out.append(f"{name}: {f} is in fingerprints_expected but the argv does not carry it exactly once with one value ({len(vals or ())} found): the content check cannot run")\n            continue\n"""
B_NEW = """            continue  # MUTANT-B: rule (b) reverted\n"""
assert s.count(A_OLD) == 1 and s.count(B_OLD) == 1
for name, text in (("mut_a", s.replace(A_OLD, A_NEW)), ("mut_b", s.replace(B_OLD, B_NEW)), ("mut_ab", s.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW))):
    dst = os.path.join(root, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    open(os.path.join(dst, "refe", "launch_gate.py"), "wb").write(text.encode("utf-8"))
    print(name, "built", text.count("MUTANT-"))
