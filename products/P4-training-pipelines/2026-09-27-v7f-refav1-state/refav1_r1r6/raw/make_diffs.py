"""Package builder: full modified files -> code/fix/<repo path>, unified diffs vs the c36b6ddd
blobs -> code/diffs_vs_c36b6ddd/<path>.diff (LF-normalised, so the diff shows content, not EOL),
and a JSON table of (repo path, base blob at c36b6ddd, new blob = git hash-object of the
LF-normalised file = the blob that lands). Reads the base via `git show` (read-only)."""
import difflib, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

REPO = "D:/Projects/TanitAD"
SRC = Path("C:/Users/Admin/refav1_r1r6")
PKG = Path(sys.argv[1])
FILES = sys.argv[2:]
out = []
for rel in FILES:
    src = SRC / rel
    new = src.read_bytes()
    base = subprocess.run(["git", "-C", REPO, "show", f"c36b6ddd:{rel}"], capture_output=True)
    base_b = base.stdout if base.returncode == 0 else None
    base_blob = (subprocess.run(["git", "-C", REPO, "rev-parse", f"c36b6ddd:{rel}"],
                                capture_output=True, text=True).stdout.strip()
                 if base_b is not None else "NEW")
    new_lf = new.replace(b"\r\n", b"\n")
    new_blob = hashlib.sha1(b"blob %d\0" % len(new_lf) + new_lf).hexdigest()
    dst = PKG / "code" / "fix" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    a = [] if base_b is None else base_b.decode("utf-8").replace("\r\n", "\n").splitlines(True)
    b = new_lf.decode("utf-8").splitlines(True)
    d = difflib.unified_diff(a, b, fromfile=("/dev/null" if base_b is None else f"a/{rel} (c36b6ddd)"),
                             tofile=f"b/{rel} (R1-R6)")
    dd = PKG / "code" / "diffs_vs_c36b6ddd" / (rel.replace("/", "_") + ".diff")
    dd.parent.mkdir(parents=True, exist_ok=True)
    txt = "".join(d)
    dd.write_bytes(txt.encode("utf-8"))
    eol = "CRLF" if b"\r\n" in new else "LF"
    n_add = sum(1 for l in txt.splitlines() if l.startswith("+") and not l.startswith("+++"))
    n_del = sum(1 for l in txt.splitlines() if l.startswith("-") and not l.startswith("---"))
    out.append({"repo_path": rel, "base_blob_c36b6ddd": base_blob, "new_blob_lf": new_blob,
                "eol_in_package": eol, "lines_added": n_add, "lines_removed": n_del,
                "package_path": str(dst.relative_to(PKG)).replace(os.sep, "/"),
                "diff_path": str(dd.relative_to(PKG)).replace(os.sep, "/")})
(PKG / "code" / "files_table.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for r in out:
    print(f"{r['repo_path']}: base {r['base_blob_c36b6ddd']} -> new {r['new_blob_lf']} "
          f"({r['eol_in_package']}, +{r['lines_added']}/-{r['lines_removed']})")
