"""Write the package's LANDING_READY.txt (PI ruling 2026-09-26, coordinated landing: the Master Mind is the
single committer; agents never `git add`).

    python make_landing.py                              # (re)write the whole list
    python make_landing.py --append raw/step5000 RESULT.md  # APPEND a new section for these paths only

Format: a heading line `## <date> REFCV7-BATTERY...`, then one path per line relative to
D:/Projects/TanitAD, each preceded by a `#` comment carrying its `git hash-object --no-filters` blob.
Every file under the package's SPEC.md / code/ / raw/ is listed, EXCEPT __pycache__, *.partial.json and
anything over 19 MiB. ⛔ Before listing, every text file is scanned for a raw clip UUID
(`sanitize_for_bank.scan`); a hit REFUSES the whole landing file.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = Path("D:/Projects/TanitAD")
REPO_R = REPO.resolve()
sys.path.insert(0, str(HERE))
import sanitize_for_bank as S  # noqa: E402


def blob(p: Path) -> str:
    r = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True, text=True,
                       cwd="C:/Users/Admin")
    b = r.stdout.strip()
    if len(b) != 40:
        raise SystemExit(f"[landing] hash-object gave {b!r} for {p} -- INCONCLUSIVE, refusing")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heading", default=None)
    ap.add_argument("--append", nargs="*", default=None, help="package-relative paths to append")
    a = ap.parse_args()
    files = []
    for sub in (a.append if a.append else ("SPEC.md", "RESULT.md", "code", "raw")):
        p = PKG / sub
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [q for q in sorted(p.rglob("*")) if q.is_file()]
    files = [f for f in files if "__pycache__" not in f.parts and not f.name.endswith(".partial.json")
             and f.stat().st_size <= 19 * 2**20]
    hits = S.scan(PKG)
    if hits:
        raise SystemExit(f"[landing] raw clip ids found, refusing: {hits[:5]}")
    head = a.heading or f"## {time.strftime('%Y-%m-%d')} REFCV7-BATTERY (EvalFlyWheel, refcv7 four-family milestone battery; written {time.strftime('%H:%M')} Berlin)"
    lines = [head, "# every path is relative to D:/Projects/TanitAD; the comment above each is its "
             "`git hash-object --no-filters` blob at the time this file was written"]
    for f in files:
        # ⚠️ 2026-10-04: D: is now `subst D: E:\` and Path.resolve() follows the subst, so the package
        # resolves to E:\Projects\TanitAD\... -- take the path relative to whichever spelling holds it
        rel = (f.relative_to(REPO_R) if f.is_relative_to(REPO_R) else f.relative_to(REPO)).as_posix()
        lines.append(f"# {blob(f)}  {f.stat().st_size} B")
        lines.append(rel)
    if a.append:
        with open(PKG / "LANDING_READY.txt", "a", encoding="utf-8") as fh:
            fh.write("\n" + "\n".join(lines) + "\n")
    else:
        (PKG / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[landing] {len(files)} files listed; uuid hits 0")


if __name__ == "__main__":
    main()
