"""make_tree.py -- export the STOPBAL code commit of record to a read tree (zero training, no GPU).

AMENDMENT_1_CODE.md names the code commit 5ecdd7e6.  The read runs the instrument of that commit:
stack/scripts/v8f_t1_eval.py is byte-identical there to the one that rolled the DAG-OACC baseline out (sha256 61e140ed, PINS.txt of
v8f-dagoacc-t1), so the band baseline and this read share the T1 instrument.  The export is a git archive with core.autocrlf=false
(byte-exact blobs).  Writes TREE_PINS.json next to this file.  Usage: python make_tree.py
"""
import hashlib
import io
import json
import os
import subprocess
import tarfile

COMMIT = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"    # the STOPBAL code of record (AMENDMENT_2_CODE.md; PATHS byte-identical to 5ecdd7e6, READ_NOTE_COMMIT.md)
MIRROR = "C:/Users/Admin/tanitad-push"
TREE = "F:/TanitAD-scratch/stopbal_read/tree"
HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ("stack/tanitad", "stack/scripts", "taniteval")
# the baseline's own pins (E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/PINS.txt); the files the STOPBAL diff touched are expected to differ
BASE_PINS = {
    "stack/scripts/v8f_t1_eval.py": "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397",
    "stack/tanitad/models/v8f_mini.py": "8ec5e19037b681242a13f3642818b2a82e84395b0b5436bc5ff17f39822c5b3a",
    "stack/tanitad/data/v8f_mini_data.py": "0e4a8cea438532424addb14d3c28a8f0b34f6c1ca33562384c3e389141f49588",
    "taniteval/taniteval/floors_kin.py": "d9c7315df1af2633a0e1069f15c19171c1261a8345ced219074d16c2840e5d1d",
    "taniteval/taniteval/four_families.py": "968393297a6a0914466e92a30bba7eb8f4b745aec758fa9201ee226e153544df",
    "taniteval/taniteval/ci.py": "38db571e368f02dc4c0d9df7b47333ebbb082e43695617ba0dfba881668cd73e",
    "taniteval/tools/t1_eval.py": "1f558f3cfc55b857cb3395b5b85e9f64b20f3fa516c455b3da54676dffe0d16d",
    "stack/tanitad/data/v8f_dagger.py": "b20cc058b17e1029ddb138dc84f434333bf6b0b2746c5b76e996cef17c176bd3",
}
MUST_EQUAL = ("stack/scripts/v8f_t1_eval.py", "taniteval/taniteval/four_families.py", "taniteval/taniteval/floors_kin.py",
              "taniteval/tools/t1_eval.py", "stack/tanitad/data/v8f_dagger.py")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert not os.path.exists(TREE), f"{TREE} exists; remove it by its literal path first"
    g = ["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", MIRROR]
    tip = subprocess.run(g + ["rev-parse", "--verify", COMMIT + "^{commit}"], capture_output=True, encoding="utf-8", check=True).stdout.strip()
    assert tip == COMMIT, tip
    raw = subprocess.run(g + ["archive", "--format=tar", COMMIT] + list(PATHS), capture_output=True, check=True).stdout
    os.makedirs(TREE)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        n = 0
        for m in t.getmembers():
            if m.isfile():
                n += 1
        t.extractall(TREE)
    pins, same = {}, {}
    for rel, base in BASE_PINS.items():
        h = sha(os.path.join(TREE, rel))
        blob = subprocess.run(g + ["show", f"{COMMIT}:{rel}"], capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == h, f"export differs from the commit blob: {rel}"
        pins[rel] = h
        same[rel] = (h == base)
    bad = [r for r in MUST_EQUAL if not same[r]]
    assert not bad, f"instrument files differ from the baseline pins: {bad}"
    rec = {"commit": COMMIT, "tree": TREE, "paths": list(PATHS), "n_files": n, "tar_bytes": len(raw), "sha256": pins,
           "equal_to_baseline_pin": same,
           "note": "v8f_mini.py and v8f_mini_data.py carry the STOPBAL diff and differ from the baseline pins by design; ci.py is the "
                   "commit's estimator module (newer than the baseline's pin). The parity dump (README_READ.md) is the control that "
                   "the changed model code reproduces the baseline on a pre-change checkpoint."}
    json.dump(rec, open(os.path.join(HERE, "TREE_PINS.json"), "w"), indent=1)
    print("tree", TREE, "files", n, "equal_to_baseline", same)


if __name__ == "__main__":
    main()
