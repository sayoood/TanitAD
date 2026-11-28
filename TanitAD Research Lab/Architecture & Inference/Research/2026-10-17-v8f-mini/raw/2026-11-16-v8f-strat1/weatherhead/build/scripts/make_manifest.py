"""make_manifest.py -- the deliverable manifest of the WH build: sha256 of every changed / new file against a4357e79 (tree vs base_tree), every build script, the launch files, and the test logs.  Writes MANIFEST_WH.md and diff_a4357e79.patch."""
import difflib
import hashlib
import os
import sys

WH = "F:/TanitAD-scratch/strat1/whead"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows, patch = [], []
    changed = ["stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_eval.py", "stack/scripts/v8f_t1_eval.py", "stack/tests/test_v8f_weatherhead.py", "stack/tests/v8f_mini_tip_snapshot_whead.txt"]
    for rel in changed:
        new = f"{WH}/tree/{rel}"
        import subprocess
        r = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "core.autocrlf=false", "show", f"a4357e79:{rel}"], capture_output=True)
        a = r.stdout.decode("utf-8").splitlines(True) if r.returncode == 0 else []        # the base file straight from the object store (no export folder needed)
        b = open(new, encoding="utf-8", newline="").read().splitlines(True)
        d = list(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=2))
        patch += d
        rows.append((f"tree/{rel}", "edited" if a else "new", sha(new), sum(1 for x in d if x[:1] in "+-" and x[:3] not in ("+++", "---")), "repo path (tree/ of the work folder) -- ONE copy until the MM commits it"))
    for f in sorted(os.listdir(f"{WH}/build")):
        if f.startswith("wh_") or f.startswith("test_wh") or f == "make_manifest.py":
            rows.append((f"build/{f}", "new", sha(f"{WH}/build/{f}"), "", "work folder; the MM copies it next to the STRAT1 study scripts -- ONE copy"))
    for f in ("launch_wh.sh", "job_wh.cmd", "launch_wh_wmi.ps1", "env.sh", "BUILD_REPORT_WH.md", "HANDOFF.md"):
        if os.path.exists(f"{WH}/{f}"):
            rows.append((f, "new", sha(f"{WH}/{f}"), "", "work folder -- ONE copy"))
    for d in ("logs",):
        for f in sorted(os.listdir(f"{WH}/{d}")):
            if f.endswith((".log", ".txt")) and os.path.getsize(f"{WH}/{d}/{f}") < 2_000_000:
                rows.append((f"{d}/{f}", "log", sha(f"{WH}/{d}/{f}"), "", "work folder"))
    open(f"{WH}/diff_a4357e79.patch", "w", encoding="utf-8", newline="").write("".join(patch))
    out = ["# MANIFEST_WH: deliverables of the form (a) build (every artifact and where it lives)", "", "| path (under `F:/TanitAD-scratch/strat1/whead/`) | status vs a4357e79 | sha256 | changed lines | where it lives |", "|---|---|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | `{r[2]}` | {r[3]} | {r[4]} |" for r in rows]
    out += ["", f"Diff of the tree against `a4357e79`: `diff_a4357e79.patch` (sha256 `{sha(WH + '/diff_a4357e79.patch')}`).", "Nothing was committed, pushed or landed. Nothing is on a pod, on box 2 or on HF."]
    open(f"{WH}/MANIFEST_WH.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("MANIFEST_WH.md", len(rows), "rows")


if __name__ == "__main__":
    main()
