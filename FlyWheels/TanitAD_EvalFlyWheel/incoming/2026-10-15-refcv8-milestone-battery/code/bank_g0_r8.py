"""bank_g0_r8.py -- bank a refcv8 G0 work dir into the package SANITIZED and SLIM (sha12 only, nothing > 5 MB).

    python bank_g0_r8.py <work_dir E:/refcv8_eval_kit/battery/step5000> <package raw/step5000> --tag step5000

Writes: G0_SUMMARY_<tag>.json/.md (summarize_g0_r8.py), g0.r8_repoint.json, the small logs (sanitized), the probe record.
The full g0.json (~10 MB), the batch cache and the collated dumps STAY on E: (listed in the README manifest).
Refuses (exit 1) on any UUID hit in the destination.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/"
                   "battery/code")
import sanitize_for_bank as S  # noqa: E402
import summarize_g0_r8 as SM  # noqa: E402

MAX = 5 * 2**20


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else src.name
    dst.mkdir(parents=True, exist_ok=True)
    out = {"banked": [], "left_on_E": []}
    g0 = src / "g0.json"
    if g0.exists() or g0.with_suffix(".partial.json").exists():
        s = SM.summarize(g0)
        json.dump(s, open(dst / f"G0_SUMMARY_{tag}.json", "w", encoding="utf-8"), indent=1, default=str)
        (dst / f"G0_SUMMARY_{tag}.md").write_text(SM.to_md(s, tag), encoding="utf-8")
        out["banked"] += [f"G0_SUMMARY_{tag}.json", f"G0_SUMMARY_{tag}.md"]
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name in ("g0.json", "g0.partial.json") or "batch_cache" in str(p):
            continue
        if p.suffix.lower() not in (".json", ".log", ".txt", ".md"):
            continue
        rel = p.relative_to(src)
        if p.stat().st_size > MAX:
            out["left_on_E"].append(f"{rel} ({p.stat().st_size} B)")
            continue
        S.sanitize_file(p, dst / rel)
        out["banked"].append(str(rel).replace("\\", "/"))
    big = [str(p) for p in dst.rglob("*") if p.is_file() and p.stat().st_size > MAX]
    for p in big:
        Path(p).unlink()
    out["removed_over_5MiB"] = big
    out["uuid_hits"] = S.scan(dst)
    print(json.dumps(out))
    if out["uuid_hits"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
