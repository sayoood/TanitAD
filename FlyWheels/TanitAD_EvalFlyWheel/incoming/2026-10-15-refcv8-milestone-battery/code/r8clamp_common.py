"""r8clamp_common.py -- shared constants + the TREE PROVENANCE check for the clamp-ON battery path (AMENDMENT A-R8-CLAMP-1).

NEW file (EvalFlyWheel, 2026-10-18).  Nothing a running chain imports is edited.  Pure stdlib: importable without torch / numpy.

The clamp (`cfg.plan_vmax_clamp`, `refcv6_selection.cap_plan_to_limit`, the call site in `refc_v3.py`) exists ONLY in the verified
export `E:/refcv8_eval_kit/tree_de4e15b` (git archive of commit de4e15b, stack + taniteval; see its PROVENANCE.txt).  The export was made with
core.autocrlf=true: LF blobs arrive CRLF, so every file is verified by LF-NORMALISED byte equality with `git cat-file blob de4e15b:<path>`.

ASSERT THE SHAPE BEFORE COMPARING (CLAUDE.md, the empty-string blob hole): both sides must be non-empty and the git object name must be 40
characters, else the answer is INCONCLUSIVE and the caller REFUSES -- never "match".
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

TREE = Path(os.environ.get("R8_CLAMP_TREE", "E:/refcv8_eval_kit/tree_de4e15b"))
COMMIT = "de4e15b"
COMMIT_FULL = "de4e15b19a39131f1fd058f09c433e5bd302b4fa"
GIT_DIR = os.environ.get("R8_CLAMP_GITDIR", "C:/Users/Admin/tanitad-push/.git")
#: the refcv7 battery code (run_battery_r7 / r7_panel / r7_roll / roll_seed_r7 / microbatch / gpu_lock ...) is NOT in the export (no FlyWheels dir):
#: it is the launch tree's, the very code that rolled the banked dumps.
BATTERY_CODE = Path(os.environ.get(
    "R8_CLAMP_BATTERY_CODE",
    "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"))

#: PROVENANCE.txt "checked 6/6 load-bearing files" + the two files a roll executes that the clamp commit also touched (loader, refcv3_arm)
LOAD_BEARING = (
    "stack/tanitad/refs/refcv6_selection.py",
    "stack/tanitad/refs/refc_v3.py",
    "stack/scripts/refc_v3_train.py",
    "stack/tanitad/train/declared_vs_built.py",
    "taniteval/taniteval/ci.py",
    "stack/tanitad/refs/refc.py",
    "stack/tanitad/eval/refcv7_loader.py",
    "taniteval/tools/refcv3_arm.py",
)
#: positive content markers (the clamp must be IN the tree, not merely a file that hashes right)
MARKERS = {
    "stack/tanitad/refs/refcv6_selection.py": ("def cap_plan_to_limit", "plan_cap_binding"),
    "stack/tanitad/refs/refc_v3.py": ("plan_vmax_clamp", "traj_precap", "plan_v_lim_ms", "cap_plan_to_limit"),
}
#: per-row keys the ON forward adds to `out` (refc_v3.py + cap_plan_to_limit's diag); appended to r7_roll.EXTRA_ROW_KEYS in ON mode
CLAMP_KEYS = ("traj_precap", "plan_cap_binding", "plan_cap_speed_before", "plan_cap_speed_after", "plan_cap_excess_ms",
              "plan_cap_v0_over_limit", "plan_v_lim_ms")
#: what the ON forward MUST emit on every call (the brief: "assert the model emits plan_cap_binding / traj_precap")
ON_REQUIRED_OUT = ("traj_precap", "plan_cap_binding")


class ProvenanceError(SystemExit):
    """The tree is not the verified clamp-carrying export (or could not be shown to be): REFUSE."""


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def _git(git_dir: str, *args: str) -> bytes:
    p = subprocess.run(["git", "--git-dir", git_dir, *args], capture_output=True, timeout=180)
    return p.stdout if p.returncode == 0 else b""


def git_blob(git_dir: str, commit: str, path: str) -> bytes:
    return _git(git_dir, "cat-file", "blob", f"{commit}:{path}")


def git_blob_sha(git_dir: str, commit: str, path: str) -> str:
    return _git(git_dir, "rev-parse", f"{commit}:{path}").decode("ascii", "replace").strip()


def check_provenance(tree=None, commit: str = COMMIT, git_dir: str | None = None, paths=LOAD_BEARING, markers=MARKERS,
                     expect_full: str | None = COMMIT_FULL, blob_reader=None, sha_reader=None) -> dict:
    """Positive provenance of `tree` against `commit`.  Returns a record; raises ProvenanceError on ANY failure or inconclusive read.

    `blob_reader(path) -> bytes` / `sha_reader(path) -> str` are test seams (default: `git cat-file blob` / `git rev-parse`)."""
    tree = Path(tree or TREE)
    git_dir = git_dir or GIT_DIR
    blob_reader = blob_reader or (lambda p: git_blob(git_dir, commit, p))
    sha_reader = sha_reader or (lambda p: git_blob_sha(git_dir, commit, p))
    prov = tree / "PROVENANCE.txt"
    if not prov.is_file():
        raise ProvenanceError(f"[r8clamp] REFUSING: {prov} is absent -- not the verified export")
    first = prov.read_text(encoding="utf-8", errors="replace").splitlines()[0] if prov.stat().st_size else ""
    if expect_full and expect_full not in first:
        raise ProvenanceError(f"[r8clamp] REFUSING: {prov} first line does not name the commit {expect_full}: {first[:200]!r}")
    rec: dict = {"tree": str(tree), "commit": commit, "provenance_first_line": first[:300], "files": {}, "markers": {}}
    for p in paths:
        f = tree / p
        if not f.is_file():
            raise ProvenanceError(f"[r8clamp] REFUSING: load-bearing file {p} is absent from {tree}")
        a, b = lf(f.read_bytes()), lf(blob_reader(p))
        sha = sha_reader(p)
        if len(a) == 0 or len(b) == 0 or len(sha) != 40:
            raise ProvenanceError(f"[r8clamp] INCONCLUSIVE (refusing): {p}: tree {len(a)} B / git blob {len(b)} B / object name {len(sha)} chars "
                                  f"(an empty or short git read is a failed query, never a match)")
        if a != b:
            raise ProvenanceError(f"[r8clamp] REFUSING: {p} differs from {commit} after LF normalisation "
                                  f"(tree {len(a)} B vs blob {len(b)} B)")
        rec["files"][p] = {"git_blob": sha, "lf_bytes": len(a), "lf_equal": True}
    for p, needles in markers.items():
        txt = lf((tree / p).read_bytes()).decode("utf-8", "replace")
        got = {n: txt.count(n) for n in needles}
        if min(got.values()) < 1:
            raise ProvenanceError(f"[r8clamp] REFUSING: {p} lacks the clamp marker(s) {[n for n, c in got.items() if c < 1]} -- "
                                  f"this is not the clamp-carrying tree")
        rec["markers"][p] = got
    rec["ok"] = True
    return rec


def _main(argv=None) -> int:
    """`python r8clamp_common.py check --out <json>`: write the provenance record (ok true) or the refusal (ok false).  Assert on the JSON, not the exit code."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        rec = check_provenance()
    except SystemExit as exc:
        rec = {"ok": False, "reason": str(exc)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"ok": bool(rec.get("ok")), "out": a.out}))
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
