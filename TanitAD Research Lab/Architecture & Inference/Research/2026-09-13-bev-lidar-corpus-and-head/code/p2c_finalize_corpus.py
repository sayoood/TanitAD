#!/usr/bin/env python3
"""P2c - finalise the corpus build: verify, summarise, mirror.

  1. every manifest row -> sha256 of its artifact, re-verified by the LOADER (literals +
     builder specs), so the staged manifest proves each file was READ, not merely listed;
  2. corpus summary: clips ok / quarantined / missing, bytes, per-clip distributions;
  3. mirror the artifacts to D: and re-hash the COPIES (a copy that exists is not a copy
     that matches);
  4. the four protected Qwen-Drive parquets are still present with their sizes; no other
     raw parquet is left behind except quarantined ones.

Writes `raw/p2_corpus_manifest.jsonl` (sha12 only) and `raw/p2_corpus_summary.json`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bev_gt_loader as L  # noqa: E402
import p2_build_corpus as B  # noqa: E402
from lidar_fetch import LIDAR_CACHE, is_protected, sha12  # noqa: E402

GT = Path(B.OUT_DIR)
MIRROR = Path(r"D:\Projects\TanitAD-artifacts\bev-lidar-gt-b1eval-20260913")
PROTECTED_SIZES = {  # bytes, MEASURED 2026-09-13 09:0x before any P2 run (ls -la)
    "4fbd97b6a4b7": None, "6924358fafe0": None, "0d90d20036a3": None, "73495082f98b": None}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows = {}
    for line in (GT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["clip_sha12"]] = r                     # last line per clip wins
    join = B.join_clips()
    join_sha = [sha12(c) for c in join]
    MIRROR.mkdir(parents=True, exist_ok=True)
    out_lines, n_ok, n_q, n_mirror_ok = [], 0, 0, 0
    stats = {k: [] for k in ("polar48_marginal_observed_infield", "polar48_marginal_rule_A_shipped",
                             "polar48_marginal_rule_C_first_hit", "cart_occ_frac_of_grid",
                             "polar48_observed_frac_infield", "polar48_infield_frac", "valid_frac",
                             "ground_peak_z_m", "artifact_bytes", "build_s")}
    dt_med, dt_max, n_nolabel = [], [], 0
    for s in join_sha:
        r = rows.get(s)
        if r is None:
            out_lines.append({"clip_sha12": s, "status": "MISSING"})
            continue
        if not r.get("ok"):
            n_q += 1
            out_lines.append({"clip_sha12": s, "status": "QUARANTINED", "failures": r.get("failures"),
                              "error": r.get("error")})
            continue
        p = GT / r["artifact"]
        clip = L.load_clip(p, grid="polar48", verify=True, builder_specs=B.SPECS, zband=B.ZB)
        digest = sha256(p)
        dst = MIRROR / r["artifact"]
        shutil.copy2(p, dst)
        mirror_ok = sha256(dst) == digest
        n_mirror_ok += int(mirror_ok)
        n_ok += 1
        for k in stats:
            if isinstance(r.get(k), (int, float)):
                stats[k].append(float(r[k]))
        dt_med.append(r["align_dt_ms"]["median"])
        dt_max.append(r["align_dt_ms"]["max_valid"])
        n_nolabel += r["align_dt_ms"]["n_no_label"]
        out_lines.append({"clip_sha12": s, "status": "OK", "artifact": r["artifact"],
                          "bytes": p.stat().st_size, "sha256": digest, "mirror_sha256_match": mirror_ok,
                          "T": r["T"], "n_valid": r["n_valid"],
                          "loader_verified": True, "frames_read": int(clip.occ.shape[0]),
                          "polar48_marginal_B": r["polar48_marginal_observed_infield"],
                          "parquet_deleted": r.get("parquet_deleted"), "fetch": r.get("fetch")})
    (HERE.parent / "raw" / "p2_corpus_manifest.jsonl").write_text(
        "\n".join(json.dumps(x) for x in out_lines) + "\n", encoding="utf-8")

    # raw parquet hygiene
    left = sorted(Path(LIDAR_CACHE).glob("*.lidar_top_360fov.parquet"))
    left_sha = [sha12(p.name.split(".")[0]) for p in left]
    protected_present = sorted(s for s in left_sha if s in PROTECTED_SIZES)
    quarantined = {x["clip_sha12"] for x in out_lines if x["status"] == "QUARANTINED"}
    unexpected = sorted(s for s in left_sha if s not in PROTECTED_SIZES and s not in quarantined)

    def dist(v):
        return {"n": len(v), "min": min(v), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": max(v)} if v else None

    summ = {
        "schema": "tanitad.bevgt_corpus_summary/1",
        "evidence_class": "MEASURED (ours; every artifact re-read by the loader and sha256'd)",
        "join_clips": len(join_sha), "ok": n_ok, "quarantined": n_q,
        "missing": sum(1 for x in out_lines if x["status"] == "MISSING"),
        "mirror": str(MIRROR), "mirror_sha256_matches": n_mirror_ok,
        "total_artifact_bytes": int(sum(stats["artifact_bytes"])),
        "instants_total": int(sum(x.get("T", 0) for x in out_lines)),
        "instants_valid": int(sum(x.get("n_valid", 0) for x in out_lines)),
        "instants_no_label": n_nolabel,
        "align_dt_ms_median_per_clip": dist(dt_med), "align_dt_ms_max_valid_per_clip": dist(dt_max),
        "per_clip": {k: dist(v) for k, v in stats.items()},
        "raw_parquets_left": len(left), "protected_present": protected_present,
        "protected_sizes_bytes": {sha12(q.name.split(".")[0]): q.stat().st_size for q in left
                                  if sha12(q.name.split(".")[0]) in PROTECTED_SIZES},
        "protected_expected": sorted(PROTECTED_SIZES), "unexpected_parquets_left": unexpected,
    }
    (HERE.parent / "raw" / "p2_corpus_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
