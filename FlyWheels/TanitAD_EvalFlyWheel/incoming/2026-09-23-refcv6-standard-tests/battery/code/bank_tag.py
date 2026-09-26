"""Bank one battery tag directory into the repo package, sanitized (sha12 only, no raw clip ids).

    python bank_tag.py <raw/tag_dir> <package/raw/tag> [--with-dumps]

* every top-level *.json / *.log of the tag dir -> sanitized copy;
* --with-dumps: the refcv6 S2 dumps (`dump_s*`: ep*.npz, refcv6_extras.npz, and a SANITIZED
  manifest.json; NOT decisions/*.npz, which stay on the dev box) packed as `dumps_<tag>.tar.gz`. These are the per-window arrays a
  zero-GPU re-analysis needs. The panel/S6 dumps are derivable from them plus the banked baseline
  dumps and are not banked. ⚠️ A re-analysis needs the clip ids back (the lead-block join is keyed
  on them): the kit's `_v2manifest.pt` maps `clip_index` -> clip id on the dev box.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanitize_for_bank as S  # noqa: E402


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in (".json", ".log", ".md", ".txt"):
            S.sanitize_file(p, dst / p.name)
            n += 1
    out = {"files": n}
    if "--with-dumps" in sys.argv:
        tgz = dst / f"dumps_{src.name}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            for d in sorted(src.glob("dump_s*")):
                for p in sorted(d.rglob("*")):
                    if not p.is_file():
                        continue
                    if "decisions" in p.relative_to(d).parts:
                        continue          # ~25 KB/window of fan telemetry; stays on the dev box
                    arc = f"{d.name}/{p.relative_to(d).as_posix()}"
                    if p.name == "manifest.json":
                        obj = S._walk(json.load(open(p, encoding="utf-8")))
                        b = json.dumps(obj, indent=1, default=str).encode("utf-8")
                        ti = tarfile.TarInfo(arc)
                        ti.size = len(b)
                        tf.addfile(ti, io.BytesIO(b))
                    elif p.suffix == ".npz" or p.name == "ABLATION.txt":
                        if p.suffix == ".npz":
                            import numpy as np
                            with np.load(p) as z:
                                bad = [k for k in z.files if z[k].dtype.kind in ("U", "S", "O")]
                            if bad:
                                raise SystemExit(f"[bank] {p}: string arrays {bad} -- refusing")
                        tf.add(str(p), arcname=arc)
                    elif p.suffix == ".json":
                        obj = S._walk(json.load(open(p, encoding="utf-8")))
                        b = json.dumps(obj, indent=1, default=str).encode("utf-8")
                        ti = tarfile.TarInfo(arc)
                        ti.size = len(b)
                        tf.addfile(ti, io.BytesIO(b))
        out["dumps_tar"] = str(tgz)
        out["dumps_tar_bytes"] = os.path.getsize(tgz)
        if out["dumps_tar_bytes"] > 19 * 2**20:        # landing rule: nothing over 20 MB
            os.remove(tgz)
            out["dumps_tar"] = None
            out["dumps_tar_removed"] = "over 19 MiB -- kept on the dev box only"
    big = [str(p) for p in dst.rglob("*") if p.is_file() and p.stat().st_size > 19 * 2**20]
    for p in big:
        os.remove(p)
    out["removed_over_19MiB"] = big
    hits = S.scan(dst)
    out["uuid_hits"] = hits
    print(json.dumps(out))
    if hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
