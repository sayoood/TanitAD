"""E-DEC-27 step 1 — fetch ONLY the label chunks the 130 held-out val clips need.

PI authorised the data job. Corrected cost: **7.13 GiB**, not the 3.58 GiB first
quoted — the join needs BOTH `obstacle.offline` (3.58) and `egomotion` (3.55),
resolved by the same chunk index (`build_obstacle_join.py:780`, `resolve(ego_idx,
EGOMOTION, cid)`). The first quote priced only the obstacle half.

⛔ 92 chunks of 3,146. The repo totals 147.2 GiB per kind; we take 2.9 % of it.
⛔ The token is read IN PLACE from Keys.txt and never printed, logged or passed as
an argument (`hf_hub_download(token=...)` only).
⭐ Resumable: `hf_hub_download` skips a file already present with the right etag,
so a re-run after an interruption costs nothing.

Layout written is the one `find_label_zips` probes:
    <hf-cache>/labels/<kind>/<kind>.chunk_NNNN.zip
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

try:
    import truststore
    truststore.inject_into_ssl()          # certifi fails behind this TLS proxy
except Exception as e:                     # pragma: no cover
    print("truststore unavailable:", e)

SPD = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
CACHE = SPD / "hfcache"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
KINDS = ("obstacle.offline", "egomotion")


def main() -> int:
    from huggingface_hub import hf_hub_download

    tok = re.search(r"hf_[A-Za-z0-9]+",
                    (R / "Keys.txt").read_text(encoding="utf-8", errors="replace"))
    if not tok:
        raise SystemExit("[pull] no HF token in Keys.txt")
    tok = tok.group(0)

    need = sorted({int(l.strip()) for l in open(SPD / "need_chunks.txt") if l.strip()})
    print(f"[pull] {len(need)} chunks x {len(KINDS)} kinds -> {CACHE}", flush=True)
    t0, done, got = time.time(), 0, 0
    for kind in KINDS:
        dst = CACHE / "labels" / kind
        dst.mkdir(parents=True, exist_ok=True)
        for c in need:
            name = f"{kind}.chunk_{c:04d}.zip"
            out = dst / name
            if out.exists() and out.stat().st_size > 0:
                done += 1
                continue
            p = hf_hub_download(REPO, f"labels/{kind}/{name}", repo_type="dataset",
                                token=tok, local_dir=str(CACHE))
            got += 1
            done += 1
            if got % 10 == 0:
                el = time.time() - t0
                mb = sum(f.stat().st_size for f in CACHE.rglob("*.zip")) / 2 ** 20
                print(f"[pull] {done}/{len(need)*len(KINDS)}  fetched {got}  "
                      f"{mb:.0f} MiB  {el/60:.1f} min", flush=True)
    mb = sum(f.stat().st_size for f in CACHE.rglob("*.zip")) / 2 ** 20
    n = len(list(CACHE.rglob("*.zip")))
    print(f"\n[pull] DONE {n} zips, {mb/1024:.2f} GiB, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    # ⛔ verify by CONTENT, not by count: a truncated zip has a size but no
    # readable central directory, and the join would read it as "clip absent".
    import zipfile
    bad = []
    for f in sorted(CACHE.rglob("*.zip")):
        try:
            with zipfile.ZipFile(f) as z:
                if z.testzip() is not None or not z.namelist():
                    bad.append(f.name)
        except Exception:
            bad.append(f.name)
    print(f"[pull] zip integrity: {len(bad)} bad of {n}"
          + (f"  -> {bad[:5]}" if bad else "  (all readable)"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
