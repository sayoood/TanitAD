"""Builds TARGET.patch (tree_base -> tree_patch), patched/ (the full changed files), PINS.txt into DEST (the ONLY write to F:). LF, UTF-8. Reads the trees only.
Usage: package_target.py"""
import difflib
import hashlib
import os

S = "C:/Users/Admin/v8proof_c/target"
DEST = "F:/TanitAD-scratch/v8proof/spatial/target"
CHANGED = ["stack/tanitad/models/hjepa.py", "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tests/test_v8f_target_levers.py"]
UNCHANGED_PINNED = ["stack/tanitad/models/v8f_beta.py", "stack/scripts/v8f_t1_eval.py", "stack/tests/test_v8f_cell_predictor.py", "stack/tests/test_v8f_motion_levers.py"]


def rd(p):
    with open(p, "rb") as fh:
        return fh.read()


sha = lambda b: hashlib.sha256(b).hexdigest()
os.makedirs(DEST, exist_ok=True)
patch = []
for rel in CHANGED:
    new = rd(f"{S}/tree_patch/{rel}")
    assert b"\r" not in new, rel
    try:
        old = rd(f"{S}/tree_base/{rel}")
    except FileNotFoundError:
        old = None
    nl = new.decode("utf-8").splitlines(keepends=True)
    if old is None:
        d = difflib.unified_diff([], nl, fromfile="/dev/null", tofile=f"b/{rel}", n=3)
        hdr = f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n"
    else:
        assert b"\r" not in old
        d = difflib.unified_diff(old.decode("utf-8").splitlines(keepends=True), nl, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)
        hdr = f"diff --git a/{rel} b/{rel}\n"
    body = "".join(d)
    if body and not body.endswith("\n"):
        body += "\n\\ No newline at end of file\n"
    patch.append(hdr + body)
with open(f"{DEST}/TARGET.patch", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("".join(patch))
pins, base_pins = {}, {}
for rel in CHANGED + UNCHANGED_PINNED:
    pins[rel] = sha(rd(f"{S}/tree_patch/{rel}"))
    if rel in CHANGED:
        dst = f"{DEST}/patched/{rel[len('stack/'):]}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(rd(f"{S}/tree_patch/{rel}"))
    if os.path.exists(f"{S}/tree_base/{rel}"):
        base_pins[rel] = sha(rd(f"{S}/tree_base/{rel}"))
with open(f"{DEST}/PINS.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# PINS: sha256 of the files of tree_patch (paths relative to the repo root; patched/ holds the changed ones without the leading stack/).\n")
    fh.write("# base = spatial/motion/tree_patch (the MOTION result). changed = base sha256 -> patched sha256\n")
    for rel in CHANGED:
        if rel in base_pins:
            fh.write(f"changed  {rel}  {base_pins[rel]} -> {pins[rel]}\n")
        else:
            fh.write(f"new      {rel}  {pins[rel]}\n")
    fh.write("# unchanged files the chain pins (same sha256 in base and patched)\n")
    for rel in UNCHANGED_PINNED:
        assert base_pins[rel] == pins[rel], rel
        fh.write(f"same     {rel}  {pins[rel]}\n")
print("packaged", {r: pins[r][:8] for r in pins})
