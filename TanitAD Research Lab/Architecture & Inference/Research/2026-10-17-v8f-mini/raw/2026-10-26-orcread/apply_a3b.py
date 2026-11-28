"""A3, part 2: the reader names Amendment 3 in its PREREG id and banks the amended text (the landed PREREG + Amendment 3) beside the A2 copy,
with a test that it extends A2 and carries the pair rule. Run after apply_a3.py; originals kept as *.pre_a3b."""
import io
import shutil
import subprocess
import sys

H = "C:/Users/Admin/orcread/"
TIP = sys.argv[1]
A3 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t1kin/AMEND_ORC3.md"
pre = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "show", f"{TIP}:Project Steering/PREREG_V8F_ORECOVER.md"], capture_output=True, check=True).stdout
assert pre == open(H + "PREREG_V8F_ORECOVER.A2.md", "rb").read(), "the landed PREREG is not the A2 copy: rebase"
add = open(A3, "rb").read()
assert b"\r\n" not in add and b"## Amendment 3" in add and b"## Amendment 3" not in pre
open(H + "PREREG_V8F_ORECOVER.A3.md", "wb").write(pre + add)


def rep(name, pairs):
    p = H + name
    shutil.copyfile(p, p + ".pre_a3b")
    s = io.open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:90])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="").write(s)


rep("orc_read.py", [
    ('+ Amendment 2 (A2-2026-10-26)"', '+ Amendment 2 (A2-2026-10-26) + Amendment 3 (A3-2026-10-26)"'),
])
rep("test_orc_read.py", [
    ('PREREG_A2 = HERE / "PREREG_V8F_ORECOVER.A2.md"           # the landed text after Amendment 2 (pinned by sha256 at landing)',
     'PREREG_A2 = HERE / "PREREG_V8F_ORECOVER.A2.md"           # the landed text after Amendment 2 (pinned by sha256 at landing)\n'
     'PREREG_A3 = HERE / "PREREG_V8F_ORECOVER.A3.md"           # the landed text after Amendment 3: the A2 text plus the amendment, appended'),
    ('''def test_a2_table_is_complete_no_reachable_UNPLACED():''',
     '''def test_a3_text_extends_a2_and_carries_the_pair_rule():
    a2, a3 = PREREG_A2.read_bytes(), PREREG_A3.read_bytes()
    assert a3.startswith(a2) and b"## Amendment 3" in a3[len(a2):] and b"## Amendment 3" not in a2
    t = a3[len(a2):].decode("utf-8")
    assert "ORC-0 names only `--ego-perturb-p 0`" in t and "identical except `--out`, `--ego-perturb-p`, and the ORC-only law flags" in t.replace("\\n", " ")
    for name, w in orc.LAW.items():                                  # the reader's law is the amendment's, value for value
        assert f"`{name} {' '.join(str(x) for x in w)}`" in t, name
    assert "Amendment 3 (A3-2026-10-26)" in orc.PREREG_ID


def test_a2_table_is_complete_no_reachable_UNPLACED():'''),
])
print("A3 copy + PREREG_ID + test written")
