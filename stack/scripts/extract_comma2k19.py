"""Extract comma2k19 chunk zips with Windows-safe names.

comma2k19 route folders contain '|' (e.g. 'b0c9...|2018-07-27--06-03-57'),
which is an illegal path character on Windows — plain extractall() fails.
This extractor rewrites '|' -> '_' in member paths (the loader is
name-agnostic: route id = folder name, whatever it looks like). On Linux
(RunPod) plain unzip works and this script is a no-op-equivalent.

Usage:
    python scripts/extract_comma2k19.py <chunk.zip> <dest_dir>
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def safe_extract(zip_path: str | Path, dest: str | Path) -> int:
    dest = Path(dest)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            safe_name = info.filename.replace("|", "_")
            target = dest / safe_name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            n += 1
    return n


if __name__ == "__main__":
    zp, dst = sys.argv[1], sys.argv[2]
    print(f"extracted {safe_extract(zp, dst)} files to {dst}")
