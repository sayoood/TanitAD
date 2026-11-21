#!/usr/bin/env python3
"""The H-NT-DECIDE-1 trainer-side change set as ONE unified diff against the launch commit fb47a58, built from the
COMMITTED blobs (``git cat-file <base-rev>:<path>``), with every result written in its file's COMMITTED EOL
convention -- so ``git apply`` works on a faithful checkout and the diff is minimal.

    python make_patch.py <out-dir> [--repo D:/Projects/TanitAD] [--base-rev fb47a58]

Why the base is read from git, not from a local tree: the local ``tree_fb47a58`` copy carries EOL DRIFT (124 of 400
sampled files differ from their committed blob, all LF-content-equal; ``refc_v3_train.py`` is committed CRLF and was
LF in the copy -- measured 2026-10-21), and a diff built on it does NOT apply to the real commit.

The manifest carries, per path, ``git_blob_result`` (the blob a faithful landing produces) and ``git_blob_result_lf``
(the blob if a ``core.autocrlf`` add normalises a CRLF-committed file -- the same content, a whole-file EOL rewrite)."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent / "nt_tree"
FILES = (
    "stack/scripts/launch_gate.py",
    "stack/scripts/refc_v3_train.py",
    "stack/tanitad/data/v2_dataset.py",
    "stack/tanitad/refs/refcv6_max_speed.py",
    "stack/tanitad/train/declared_vs_built.py",
    "stack/tanitad/train/refcv8_train.py",
    "stack/tanitad/train/nt_guard.py",
    "stack/tanitad/configs/refcv8_nt/nt_heldout.json",
    "stack/tanitad/configs/refcv8_nt/.gitattributes",
    "stack/tanitad/configs/refcv8_nt/AMENDMENT_ACK_NT.txt",
    "stack/ops/runs.d/nt-a1-s0.argv.json",
    "stack/ops/refcv8nt_eval_remap_thor.json",
    "stack/scripts/nt_prelaunch.sh",
    "stack/ops/runs.d/nt-a1r-s1.argv.json",
    "stack/tests/test_declared_vs_built.py",
    "stack/tests/test_nt_decide.py",
    "stack/tests/test_v2_dataset_frame_index.py",
    # two SOURCE pins the E4 trainer patch moved (found by the faithful-vs-landed classification, 2026-10-21 23:50):
    # the milestone save now also writes --snapshot-steps; the train-split LABEL reader now passes all_unknown
    "stack/tests/test_refc_v3_save_before_eval.py",
    "stack/tests/test_refcv8_speed_label.py",
)
NUL = bytes([0])
CRLF, LF = bytes([13, 10]), bytes([10])


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + NUL + data).hexdigest()


def committed(repo: str, rev: str, rel: str) -> bytes | None:
    r = subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{rev}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return None
    r = subprocess.run(["git", "-C", repo, "cat-file", "blob", f"{rev}:{rel}"], capture_output=True, check=True)
    return r.stdout


def is_crlf(b: bytes) -> bool:
    return b.count(CRLF) * 2 > b.count(LF)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--repo", default="D:/Projects/TanitAD")
    ap.add_argument("--base-rev", default="fb47a58")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base_commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "--verify", a.base_rev + "^{commit}"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    chunks, man = [], {}
    for rel in FILES:
        bb = committed(a.repo, base_commit, rel)
        nb = (NEW / rel).read_bytes()
        if CRLF in nb:
            raise SystemExit(f"{rel}: CRLF in the nt_tree file (nt_tree is kept LF; the EOL is applied here)")
        crlf = bool(bb) and is_crlf(bb)
        if bb is not None and bb.replace(CRLF, LF) == nb:
            raise SystemExit(f"{rel}: no content change vs {a.base_rev} -- not part of the change set")
        res = nb.replace(LF, CRLF) if crlf else nb
        old = bb.decode("utf-8").splitlines(keepends=True) if bb is not None else []
        new = res.decode("utf-8").splitlines(keepends=True)
        d = list(difflib.unified_diff(old, new, fromfile=("a/" + rel) if bb is not None else "/dev/null",
                                      tofile="b/" + rel, n=3, lineterm="\n"))
        # difflib's own header/hunk lines end in "\n"; content lines keep their own ending (CRLF where committed so)
        d.insert(0, f"diff --git a/{rel} b/{rel}\n" + ("" if bb is not None else "new file mode 100644\n"))
        d = [l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in d]
        chunks.append("".join(d))
        man[rel] = {"git_blob_base": git_blob(bb) if bb is not None else None,
                    "git_blob_result": git_blob(res),
                    "git_blob_result_lf": git_blob(res.replace(CRLF, LF)),
                    "committed_crlf": crlf, "new_file": bb is None, "n_lines": len(new)}
    diff = "".join(chunks).encode("utf-8")
    (out / "nt_decide_stack.diff").write_bytes(diff)
    (out / "nt_decide_stack.manifest.json").write_text(json.dumps(
        {"base": f"{a.base_rev} = {base_commit} (committed blobs, read from {a.repo})", "files": man,
         "check": "after applying: git hash-object --no-filters <path> (or git rev-parse <rev>:<path> once committed) "
                  "must equal git_blob_result; == git_blob_result_lf means a core.autocrlf add rewrote the EOL",
         "diff_sha256": hashlib.sha256(diff).hexdigest()}, indent=1), encoding="utf-8", newline="\n")
    print(len(chunks), "files in the diff;", len(diff), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
