#!/usr/bin/env python3
"""Where does the NT patch land? Read-only against a repo: for each patched path, the blob at <base-rev> (must equal the
manifest's ``git_blob_base`` -- the positive control that tree_fb47a58 IS fb47a58) and at <head-rev>; then a per-file
3-way merge (``git merge-file``: base = fb47a58, ours = HEAD, theirs = nt_tree) in a scratch dir, so the landing agent
learns, before touching the index, which files merge clean against the CURRENT head and gets the merged result plus
a HEAD-based diff.

A blob id is admissible only if it is 40 HEX chars: a failing ``git rev-parse HEAD:<path>`` echoes its argument, and
``HEAD:stack/tanitad/train/refcv8_train.py`` is exactly 40 characters (measured 2026-10-21 -- the length-only shape
check passed it as a blob).

    python landing_check.py --repo D:/Projects/TanitAD --manifest out/patch/nt_decide_stack.manifest.json \
        --new nt_tree --out out/patch/landing_<head>
Writes nothing to the repo (``git show`` / ``git rev-parse`` / ``git merge-file -p`` on scratch copies only)."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")
NUL = bytes([0])


def git_blob(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob " + str(len(data)).encode() + NUL + data).hexdigest()


def git(repo, *args, binary=False, check=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=180)
    if check and r.returncode != 0:
        raise SystemExit(f"git {args} failed: {r.stderr.decode(errors='replace')[:300]}")
    return r


def blob_id(repo, rev, path):
    r = git(repo, "rev-parse", "--verify", "--quiet", f"{rev}:{path}")
    h = r.stdout.decode().strip()
    return h if (r.returncode == 0 and HEX40.match(h)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--new", required=True, help="the tree holding the patched files (nt_tree)")
    ap.add_argument("--base-rev", default="fb47a58")
    ap.add_argument("--head-rev", default="HEAD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, new, out = Path(a.repo), Path(a.new), Path(a.out)
    m = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    head = git(repo, "rev-parse", "--verify", a.head_rev, check=True).stdout.decode().strip()
    base = git(repo, "rev-parse", "--verify", a.base_rev + "^{commit}", check=True).stdout.decode().strip()
    assert HEX40.match(head) and HEX40.match(base), (head, base)
    out.mkdir(parents=True, exist_ok=True)
    rows, merged_diff = {}, []
    base_tree = Path(a.new).parent / "tree_fb47a58"

    def lf(x: bytes) -> bytes:
        return x.replace(b"\r\n", b"\n")

    def is_crlf(x: bytes) -> bool:                      # the blob's own convention: majority of its line ends
        return x.count(b"\r\n") * 2 > x.count(b"\n")

    for rel, v in m["files"].items():
        b_id, h_id = blob_id(repo, base, rel), blob_id(repo, head, rel)
        bb = git(repo, "cat-file", "blob", b_id, check=True).stdout if b_id else b""
        mine_base = (base_tree / rel).read_bytes() if (base_tree / rel).is_file() else b""
        row = {"base_blob_repo": b_id, "head_blob": h_id,
               # the strict control (blob id) and the content control (LF-normalised): a CRLF-committed blob fails the
               # first and passes the second -- refc_v3_train.py is committed CRLF at fb47a58 (measured 2026-10-21)
               "control_base_blob_eq": (b_id == v["git_blob_base"]) if not v["new_file"] else (b_id is None),
               "control_base_content_eq_lf": lf(bb) == lf(mine_base),
               "base_blob_crlf": is_crlf(bb) if bb else None}
        newb = lf((new / rel).read_bytes())
        if not v["new_file"] and h_id is None:
            r = git(repo, "cat-file", "-e", f"{head}:{rel}")
            row["state"] = "HEAD_UNREADABLE" if r.returncode != 0 else "HEAD_BLOB_ID_UNREADABLE"
            row["stderr"] = r.stderr.decode(errors="replace")[:200]
            rows[rel] = row
            continue
        hb = git(repo, "cat-file", "blob", h_id, check=True).stdout if h_id else b""
        row["head_blob_crlf"] = is_crlf(hb) if hb else None
        merged = None
        if h_id is not None and lf(hb) == newb:
            row["state"] = "ALREADY_LANDED"
        elif v["new_file"] and h_id is None:
            row["state"] = "NEW_CLEAN"
            merged = newb
        elif not v["new_file"] and lf(hb) == lf(bb):
            row["state"] = "HEAD_EQ_BASE"
            merged = newb
        else:
            row["state"] = "NEW_BUT_PRESENT_AT_HEAD" if v["new_file"] else "HEAD_MOVED"
            sd = out / "merge" / rel.replace("/", "__")
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "ours_head").write_bytes(lf(hb))
            (sd / "base").write_bytes(lf(bb))
            (sd / "theirs_nt").write_bytes(newb)
            r = subprocess.run(["git", "merge-file", "-p", "-L", "HEAD", "-L", "fb47a58", "-L", "nt_tree",
                                str(sd / "ours_head"), str(sd / "base"), str(sd / "theirs_nt")],
                               capture_output=True, timeout=180)
            row["merge_conflicts"] = r.returncode       # git merge-file: number of conflicts (<0 on error)
            (sd / "merged").write_bytes(r.stdout)
            row["head_vs_base_changed_lines"] = sum(1 for l in difflib.unified_diff(
                lf(bb).decode("utf-8", "replace").splitlines(), lf(hb).decode("utf-8", "replace").splitlines(), n=0)
                if l[:1] in "+-" and l[:3] not in ("+++", "---"))
            if r.returncode == 0:
                merged = r.stdout
        if merged is not None:
            if row.get("head_blob_crlf") or (h_id is None and row.get("base_blob_crlf")):
                merged = merged.replace(b"\n", b"\r\n")  # keep the file's committed convention: a minimal diff
            row["landed_blob_expected"] = git_blob(merged)
            row["landed_blob_lf"] = git_blob(lf(merged))
            dst = out / "landed" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(merged)
            d = list(difflib.unified_diff(lf(hb).decode("utf-8").splitlines(keepends=True),
                                          lf(merged).decode("utf-8").splitlines(keepends=True),
                                          fromfile=("a/" + rel) if h_id else "/dev/null", tofile="b/" + rel, n=3))
            row["diff_lines_vs_head"] = sum(1 for l in d if l[:1] in "+-" and l[:3] not in ("+++", "---"))
            if d:
                d.insert(0, f"diff --git a/{rel} b/{rel}\n" + ("" if h_id else "new file mode 100644\n"))
                merged_diff.append("".join(l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in d))
        rows[rel] = row
    (out / "nt_decide_stack_vs_head.diff").write_text("".join(merged_diff), encoding="utf-8", newline="\n")
    rep = {"repo": str(repo), "head": head, "base": base, "files": rows,
           "all_content_controls_pass": all(r["control_base_content_eq_lf"] for r in rows.values()),
           "note": "nt_decide_stack_vs_head.diff is LF-normalised (review copy); land the files under landed/ "
                   "(each keeps its committed EOL convention) and assert git rev-parse HEAD:<p> == landed_blob_expected",
           "n_conflicted": sum(1 for r in rows.values() if r.get("merge_conflicts", 0) != 0)}
    (out / "LANDING_CHECK.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    for rel, r in rows.items():
        print(f"{r['state']:<24} conflicts={r.get('merge_conflicts', '-')!s:<3} ctl={r['control_base_content_eq_lf']!s:<5} "
              f"crlf={r.get('head_blob_crlf')!s:<5} dl={r.get('diff_lines_vs_head', '-')!s:<5} "
              f"headchg={r.get('head_vs_base_changed_lines', '-')!s:<5} {rel}")
    print("HEAD", head[:12], "content controls", rep["all_content_controls_pass"], "conflicted", rep["n_conflicted"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
