"""Regenerate MANIFEST.txt of the RECOVER-FT package: path | bytes | sha256 (MANIFEST.txt itself is not listed)."""
import hashlib
import os

ROOT = "E:/Projects/TanitAD-artifacts/dagland/recover_ft"
rows = ["# RECOVER-FT package manifest (path | bytes | sha256). NOT in git; the Master Mind lands stack_patch/stack/. MANIFEST.txt itself is not listed.",
        "# stack_patch/stack/<path> mirrors the repo path stack/<path>. Base of the trainer patch: mirror tip 408cd81f, stack/scripts/v8f_mini_train.py sha256 0d708b3cb841bc1f1832b09fe61e3d8421a4d3f59769f4435c0b22bb77e268ce."]
for d, _, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(d, f).replace("\\", "/")
        if p.endswith("/MANIFEST.txt") or "__pycache__" in p:
            continue
        b = open(p, "rb").read()
        rows.append(f"{p} | {len(b)} | {hashlib.sha256(b).hexdigest()}")
rows = rows[:2] + sorted(rows[2:])
open(f"{ROOT}/MANIFEST.txt", "w", newline="\n").write("\n".join(rows) + "\n")
print(len(rows) - 2, "files")
