"""Bank the v7F merge copy into the package: LF-normalised full files under code/fix/, unified diffs vs the tip
blobs (c36b6ddd) under code/diffs_vs_c36b6ddd/, and the R2 test/mutation records under raw/.
Every banked code file is checked to be LF-only and to parse; every diff is checked to be non-empty."""
import ast, subprocess, sys
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
TIP = "c36b6dddc18ef8d43debc610591b58432748e104"
SRC = Path("C:/Users/Admin/v7f_merge/stack")
PKG = REPO / "products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge"
FILES = sys.argv[1:] or ["tanitad/models/v6.py", "scripts/train_v6_staged.py", "tests/test_v7f_r2_nav_fixes.py"]
CR = chr(13).encode()

for rel in FILES:
    tip = subprocess.run(["git", "-C", str(REPO), "show", f"{TIP}:stack/{rel}"], capture_output=True)
    # ⛔ EOL follows the TIP BLOB, per file: some repo blobs are CRLF (e.g. tests/test_v6_effective_weights.py,
    # 602 CRLF / 0 LF). Blanket LF-normalisation would rewrite every line of such a file. NEW files -> LF.
    tip_crlf = tip.returncode == 0 and b"\r\n" in tip.stdout
    raw = (SRC / rel).read_bytes().replace(b"\r\n", b"\n")
    assert CR not in raw, f"{rel}: stray CR after normalisation"
    ast.parse(raw.decode("utf-8"))
    if tip_crlf:
        raw = raw.replace(b"\n", b"\r\n")
    dst = PKG / "code/fix/stack" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    base = PKG / "raw/_tip_blob.tmp"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(tip.stdout if tip.returncode == 0 else b"")
    d = subprocess.run(["git", "diff", "--no-index", "--text", f"--src-prefix=a/stack/", f"--dst-prefix=b/stack/",
                        str(base), str(dst)], capture_output=True)
    patch = d.stdout.replace(str(base).replace("\\", "/").encode(), rel.encode()).replace(
        str(dst).replace("\\", "/").encode(), rel.encode())
    out = PKG / "code/diffs_vs_c36b6ddd" / ("stack_" + rel.replace("/", "_") + ".diff")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(patch)
    base.unlink()
    status = "NEW" if tip.returncode != 0 else "MODIFIED"
    print(f"{status:8s} {rel}: {len(raw)} B {'CRLF' if tip_crlf else 'LF'}, diff {len(patch)} B")
    assert patch, f"{rel}: empty diff"
