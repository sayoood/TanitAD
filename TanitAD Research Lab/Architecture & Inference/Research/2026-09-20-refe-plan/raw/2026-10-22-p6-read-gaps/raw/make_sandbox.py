"""Overlay sandbox = mirror-tip eval/ + refe/ (pkx) with the package's changed files on top: what the landed tree will look like. Tests run here."""
import shutil, sys
from pathlib import Path
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
root = Path("C:/Users/Admin/par18/p6gaps")
sb = root / "sandbox" / PK
if (root / "sandbox").exists():
    shutil.rmtree(root / "sandbox")
shutil.copytree(root / "pkx" / PK / "eval", sb / "eval", ignore=shutil.ignore_patterns("raw", "__pycache__"))
shutil.copytree(root / "pkx" / PK / "refe", sb / "refe", ignore=shutil.ignore_patterns("__pycache__"))
# overlay
for p in (root / "code" / PK).rglob("*"):
    if p.is_file() and "__pycache__" not in p.parts:
        d = sb / p.relative_to(root / "code" / PK)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
print("sandbox at", sb)
