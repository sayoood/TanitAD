"""make_tree_s1.py -- export the STOPPERC arm-A code commit of record (3e264bbf) to the stage-1 read tree (zero training, no GPU).

Why a NEW tree: arm A's checkpoints carry cfg fields (`stop_aux`, `w_stop_aux`) and state-dict keys (`stop_head.*`, `stop_pres_head.*`) that the STOPBAL tree's (715ef8fc) V8FMiniConfig / strict load do not know, so the
dump and STOPDEC steps must build them from 3e264bbf.  `git diff 715ef8fc 3e264bbf` over the exported paths touches exactly four files (checked below): stack/scripts/stop_aux_w_from_smoke.py (new), stack/scripts/v8f_mini_train.py,
stack/tanitad/data/v8f_stopaux.py (new), stack/tanitad/models/v8f_mini.py.  The instrument files (v8f_t1_eval.py, four_families.py, floors_kin.py, t1_eval.py, v8f_dagger.py, ci.py, v8f_mini_data.py) must be BYTE-IDENTICAL
to the STOPBAL tree's pins (TREE_PINS.json of theory/stopbal/read); v8f_mini.py must differ (the stop-aux diff).  git archive with core.autocrlf=false.  Writes TREE_PINS_S1.json beside this file.
Usage: python make_tree_s1.py [--tree DIR]     (default F:/TanitAD-scratch/stopperc_read/tree; refuses an existing folder)"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

COMMIT = "3e264bbf3fa2f6f17bea448de09aa972faaf7725"
BASE_COMMIT = "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
MIRROR = "C:/Users/Admin/tanitad-push"
TREE = "F:/TanitAD-scratch/stopperc_read/tree"
HERE = os.path.dirname(os.path.abspath(__file__))
STOPBAL_PINS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), "theory", "stopbal", "read", "TREE_PINS.json")
PATHS = ("stack/tanitad", "stack/scripts", "taniteval")
MUST_EQUAL = ("stack/scripts/v8f_t1_eval.py", "taniteval/taniteval/four_families.py", "taniteval/taniteval/floors_kin.py", "taniteval/tools/t1_eval.py",
              "stack/tanitad/data/v8f_dagger.py", "taniteval/taniteval/ci.py", "stack/tanitad/data/v8f_mini_data.py")
MUST_DIFFER = ("stack/tanitad/models/v8f_mini.py",)
EXPECTED_DIFF = {"stack/scripts/stop_aux_w_from_smoke.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_stopaux.py", "stack/tanitad/models/v8f_mini.py"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(tree=TREE):
    assert not os.path.exists(tree), f"{tree} exists; remove it by its literal path first"
    g = ["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", MIRROR]
    tip = subprocess.run(g + ["rev-parse", "--verify", COMMIT + "^{commit}"], capture_output=True, encoding="utf-8", check=True).stdout.strip()
    assert tip == COMMIT, tip
    par = subprocess.run(g + ["rev-parse", COMMIT + "^"], capture_output=True, encoding="utf-8", check=True).stdout.strip()
    assert par == BASE_COMMIT, f"3e264bbf's parent is {par}, expected {BASE_COMMIT}"
    diff = set(subprocess.run(g + ["diff", "--name-only", BASE_COMMIT, COMMIT, "--"] + list(PATHS), capture_output=True, encoding="utf-8", check=True).stdout.split())
    assert diff == EXPECTED_DIFF, f"the diff over the exported paths is {sorted(diff)}, expected {sorted(EXPECTED_DIFF)}"
    raw = subprocess.run(g + ["archive", "--format=tar", COMMIT] + list(PATHS), capture_output=True, check=True).stdout
    os.makedirs(tree)
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        n = sum(1 for m in t.getmembers() if m.isfile())
        t.extractall(tree)
    base = json.load(open(STOPBAL_PINS, encoding="utf-8"))
    assert base["commit"] == BASE_COMMIT, base["commit"]
    pins, same = {}, {}
    for rel in MUST_EQUAL + MUST_DIFFER:
        h = sha(os.path.join(tree, rel))
        blob = subprocess.run(g + ["show", f"{COMMIT}:{rel}"], capture_output=True, check=True).stdout
        assert hashlib.sha256(blob).hexdigest() == h, f"export differs from the commit blob: {rel}"
        pins[rel] = h
        same[rel] = (h == base["sha256"].get(rel))
    bad = [r for r in MUST_EQUAL if base["sha256"].get(r) is not None and not same[r]]
    unpinned = [r for r in MUST_EQUAL if base["sha256"].get(r) is None]
    assert not bad, f"instrument files differ from the STOPBAL tree pins: {bad}"
    assert not any(same[r] for r in MUST_DIFFER), "v8f_mini.py equals the STOPBAL tree's: the export is the wrong commit"
    rec = {"commit": COMMIT, "base_commit": BASE_COMMIT, "tree": tree, "paths": list(PATHS), "n_files": n, "tar_bytes": len(raw), "sha256": pins, "equal_to_stopbal_tree_pin": same,
           "diff_files_vs_715ef8fc": sorted(diff), "instrument_files_without_a_stopbal_pin": unpinned,
           "note": "the OFF-checkpoint bit-identity of this tree against the STOPBAL tree is a separate, measured check (stopperc_s1_read.py identity; README_READ_S1.md)"}
    json.dump(rec, open(os.path.join(HERE, "TREE_PINS_S1.json"), "w"), indent=1)
    print("TREE", tree, n, "files; diff vs 715ef8fc:", sorted(diff))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else TREE)
