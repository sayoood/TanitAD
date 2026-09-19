"""Rebuild the clip list for ``tools/clipid_scan.py --clips`` — OUTSIDE the repo.

The list is 4,719 clip ids, so it is never banked. This rebuilds it from a source that
holds it and writes it ONLY if it reproduces the digest the floor was recorded with
(``tools/clipid_baseline.json`` → ``_clips_list``). A wrong or partial source is refused
here, before the guard ever sees it.

    python rebuild_v7_list.py <out.txt> --mirror C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov
    python rebuild_v7_list.py <out.txt> --sha-json <corpus copy>/camera/camera_sha256.json

⛔ Refuses an output path inside the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for up in Path(__file__).resolve().parents:
        if (up / "tools" / "clipid_scan.py").exists():
            return up
    raise SystemExit("tools/clipid_scan.py not found above this script")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mirror", help="directory of <clip id>.mp4 files")
    src.add_argument("--sha-json", help="the corpus's camera/camera_sha256.json")
    a = ap.parse_args(argv)
    repo = _repo_root()
    out = Path(a.out).resolve()
    if out.is_relative_to(repo):
        raise SystemExit(f"⛔ refusing to write the clip list inside the repository ({repo})")
    sys.path.insert(0, str(repo / "tools"))
    from clipid_scan import list_identity
    rec = json.loads((repo / "tools" / "clipid_baseline.json").read_text(encoding="utf-8"))
    rec = rec.get("_clips_list")
    if not rec:
        raise SystemExit("⛔ the baseline records no named list; nothing to rebuild against")
    if a.mirror:
        ids = [p.name[:36] for p in Path(a.mirror).glob("*.mp4")]
    else:
        d = json.loads(Path(a.sha_json).read_text(encoding="utf-8"))
        ids = [k.split("/")[-1][:36] for k in d]
    got = list_identity(ids)
    if (got["n"], got["sha256_sorted"]) != (rec["n"], rec["sha256_sorted"]):
        raise SystemExit(
            f"⛔ this source does not reproduce the recorded list {rec.get('name')!r} "
            f"(want n={rec['n']}, sha256 {rec['sha256_sorted'][:16]}…; got n={got['n']}, "
            f"sha256 {got['sha256_sorted'][:16]}…). Nothing written.")
    out.write_text("\n".join(sorted(set(ids))) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {got['n']} ids to {out} — reproduces {rec.get('name')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
