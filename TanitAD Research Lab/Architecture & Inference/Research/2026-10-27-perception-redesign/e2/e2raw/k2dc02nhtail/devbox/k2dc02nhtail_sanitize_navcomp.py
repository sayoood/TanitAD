"""k2dc02nhtail_sanitize_navcomp.py <src dir> <dst dir> -- copy the STRATEGIC outputs for banking with every dataset clip id removed.

The lander's clip-id scan refuses raw PhysicalAI clip UUIDs in the repo. In the per-arm navcomp_*.json they sit only under
time_base_control.per_clip[].clip; each is replaced by its position ("clip#<i>") and the file records that it was sanitized.
Any other string that still looks like a clip UUID makes this script refuse. STRATEGIC_NAVCOMP.json / .md are copied unchanged."""
import json
import re
import shutil
import sys
from pathlib import Path

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def main(argv):
    src, dst = Path(argv[1]), Path(argv[2])
    dst.mkdir(parents=True, exist_ok=False)
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if not p.name.startswith("navcomp_"):
            shutil.copy2(p, dst / p.name)
            assert not UUID.search((dst / p.name).read_text(encoding="utf-8")), f"clip id left in {p.name}"
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for i, row in enumerate(((d.get("time_base_control") or {}).get("per_clip") or [])):
            if isinstance(row, dict) and "clip" in row:
                row["clip"] = f"clip#{i}"
                n += 1
        d["_sanitized"] = f"MM 2026-11-20: {n} clip ids under time_base_control.per_clip[].clip replaced by their index for the repo (clip-id scan)"
        t = json.dumps(d, indent=1, default=str)
        left = UUID.findall(t)
        assert not left, f"{p.name}: {len(left)} clip-id-like strings remain, e.g. {left[0][:8]}"
        (dst / p.name).write_text(t, encoding="utf-8")
        print(f"{p.name}: {n} clip ids replaced")
    print("SANITIZE OK", dst)


if __name__ == "__main__":
    main(sys.argv)
