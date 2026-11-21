"""Merge the R3 stream into the v7F merge copy.

train_v6_staged.py : 3-way `git merge-file` in LF space -- base = the tip blob (LF), ours = the merge copy
                     (R1/R4 + R2 + R6; CRLF working copy -> LF), theirs = the R3 fix (CRLF -> LF). Written back
                     to the working copy in its own EOL (CRLF). Conflicts are REPORTED, never auto-resolved.
test_v6_effective_weights.py : the merge copy == tip (verified byte-identical to the snapshot), so R3's version
                     is taken; the tip blob is CRLF and R3 kept CRLF -> kept as-is.
test_tactical_label_reach_v6.py : NEW from R3 -> copied.
declared_vs_built_v6.py : NOT taken from R3 -- the union module is already in the merge copy.
"""
import subprocess, sys, tempfile
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
TIP = "c36b6dddc18ef8d43debc610591b58432748e104"
R3 = REPO / "products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r3/code/fix"
M = Path("C:/Users/Admin/v7f_merge")
CRLF, LF = b"\r\n", b"\n"


def lf(b: bytes) -> bytes:
    return b.replace(CRLF, LF)


rel = "stack/scripts/train_v6_staged.py"
base = subprocess.run(["git", "-C", str(REPO), "show", f"{TIP}:{rel}"], capture_output=True, check=True).stdout
ours_raw = (M / rel).read_bytes()
theirs = lf((R3 / rel).read_bytes())
ours = lf(ours_raw)
assert CRLF not in base, "tip blob expected LF"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "ours").write_bytes(ours); (td / "base").write_bytes(base); (td / "theirs").write_bytes(theirs)
    r = subprocess.run(["git", "merge-file", "-p", "-L", "merge(R1R4+R2+R6)", "-L", "tip", "-L", "R3",
                        str(td / "ours"), str(td / "base"), str(td / "theirs")], capture_output=True)
merged = r.stdout
n_conf = merged.count(b"<<<<<<< merge(R1R4+R2+R6)")
print(f"{rel}: merge-file rc={r.returncode} conflicts={n_conf} | ours {len(ours)} B, theirs {len(theirs)} B, "
      f"merged {len(merged)} B")
out_eol = CRLF if CRLF in ours_raw else LF
(M / "train_v6_staged.merged_LF.py").write_bytes(merged)
if n_conf == 0 and r.returncode == 0:
    (M / rel).write_bytes(merged.replace(LF, out_eol) if out_eol == CRLF else merged)
    print("  -> written to the merge copy (EOL", "CRLF" if out_eol == CRLF else "LF", ")")
else:
    print("  -> NOT written: resolve the conflicts in", M / "train_v6_staged.merged_LF.py")

for rel in ("stack/tests/test_v6_effective_weights.py", "stack/tests/test_tactical_label_reach_v6.py"):
    (M / rel).write_bytes((R3 / rel).read_bytes())
    print(f"{rel}: taken from R3 ({len((R3 / rel).read_bytes())} B)")
sys.exit(0 if n_conf == 0 and r.returncode == 0 else 1)
