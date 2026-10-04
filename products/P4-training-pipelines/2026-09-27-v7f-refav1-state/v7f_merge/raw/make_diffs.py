"""Unified diffs vs the tip with repo-relative headers, each PROVEN to `git apply` onto the tip blob and to
reproduce the banked code/fix file byte-for-byte."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
TIP = "c36b6dddc18ef8d43debc610591b58432748e104"
PKG = REPO / "products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge"
FIX = PKG / "code/fix/stack"
OUT = PKG / "code/diffs_vs_c36b6ddd"
rels = [p.relative_to(FIX).as_posix() for p in FIX.rglob("*") if p.is_file()]
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.diff"):
    old.unlink()
ok = 0
for rel in sorted(rels):
    tip = subprocess.run(["git", "-C", str(REPO), "show", f"{TIP}:stack/{rel}"], capture_output=True)
    is_new = tip.returncode != 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a, b = td / "a/stack" / rel, td / "b/stack" / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIX / rel, b)
        if not is_new:
            a.parent.mkdir(parents=True, exist_ok=True)
            a.write_bytes(tip.stdout)
        left = "/dev/null" if is_new else f"a/stack/{rel}"
        # core.autocrlf=false here too: otherwise a CRLF-stored file's hunks lose their CRs in the patch
        d = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--no-prefix", "--text",
                            left, f"b/stack/{rel}"], cwd=td, capture_output=True)
        patch = d.stdout
        if not is_new:
            patch = patch.replace(f"b/stack/{rel}".encode(), f"b/stack/{rel}".encode())
        # prove: apply onto the tip blob (or nothing) in a scratch repo and compare bytes
        chk = td / "chk"
        (chk / "stack" / rel).parent.mkdir(parents=True, exist_ok=True)
        if not is_new:
            (chk / "stack" / rel).write_bytes(tip.stdout)
        pf = td / "p.diff"
        pf.write_bytes(patch)
        # core.autocrlf=false: otherwise git on this box writes CRLF and the byte check reads a false mismatch
        ap = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-p1", str(pf)], cwd=chk,
                            capture_output=True, text=True)
        same = (chk / "stack" / rel).exists() and (chk / "stack" / rel).read_bytes() == (FIX / rel).read_bytes()
    name = "stack_" + rel.replace("/", "_") + ".diff"
    (OUT / name).write_bytes(patch)
    print(f"{'NEW' if is_new else 'MOD'} {rel}: diff {len(patch)} B | git apply rc={ap.returncode} "
          f"{ap.stderr.strip()[:120]} | reproduces code/fix: {same}")
    ok += int(ap.returncode == 0 and same)
print(f"{ok}/{len(rels)} diffs apply cleanly and reproduce their file")
sys.exit(0 if ok == len(rels) else 1)
