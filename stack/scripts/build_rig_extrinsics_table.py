#!/usr/bin/env python3
"""build_rig_extrinsics_table.py -- the PER-CLIP front-wide mount pose table.

Emits the JSON that ``refc_v3_train.py --agent-rig-extrinsics`` consumes as a
PER-CLIP bank: ``{clip_id: {qx, qy, qz, qw, x, y, z}}``, read from
PhysicalAI-AV's OWN ``calibration/sensor_extrinsics`` parquets. Nothing here is
fitted, assumed or inherited.

WHY A TABLE AND NOT A CONSTANT
------------------------------
Retraction class **C28** -- *a constant where the quantity is per-clip*. The
front-wide mount height is **not** a constant: three values circulate in this
repo (1.22 / 1.43 / 1.5 m) and all three are wrong. ``ground_range_prior``
back-projects a pixel through the road plane, so the range it supervises is
directly PROPORTIONAL to the mount height -- a 1.5 m camera on a 1.245 m clip
biases every range it teaches by +20 %, and the head learns that bias as
geometry.

THE CLIP SET IS ASSERTED, NOT ASSUMED
-------------------------------------
The parity corpus is ``physicalai-train-e438721ae894`` (2,400 clips ->
2,376 episodes after the 24-clip skip ``f09e44db``). This script verifies the
clip list it read against ``parity_manifest.json``'s committed
``clip_id_sha256_sorted`` BEFORE writing anything, so a re-selected or
truncated clip set is refused rather than silently producing a table for a
different corpus.

    python stack/scripts/build_rig_extrinsics_table.py \\
        --root C:/Users/Admin/tanitad-data/physicalai \\
        --corpus physicalai-train-e438721ae894 --out rig_extrinsics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parent.parent
if str(STACK) not in sys.path:
    sys.path.insert(0, str(STACK))

SENSOR = "camera_front_wide_120fov"


def _front_wide_rows(df):
    """The front-wide rows of a calibration parquet, clip_id as a column."""
    import pandas as pd                                        # noqa: F401
    if "clip_id" not in df.columns:
        df = df.reset_index()
    name_col = ("name" if "name" in df.columns else
                "sensor_name" if "sensor_name" in df.columns else None)
    if name_col is None:
        df = df.rename(columns={"level_0": "clip_id", "level_1": "name"})
        name_col = "name"
    return df[df[name_col].astype(str) == SENSOR]


def clip_list(root: Path):
    """clip_id -> chunk, UNIONED over every selection table on the box.

    r0_selection holds 500 clips and phase0_selection 3,000; the parity corpus
    is 2,400 and is a subset of neither alone. Reading only one is the
    absence-at-one-location error, and it produced a MISMATCH refusal on the
    first run of this script -- correctly.
    """
    import pandas as pd
    out = {}
    for name in ("phase0_selection.parquet", "r0_selection.parquet"):
        p = root / "r0" / name
        if not p.exists():
            continue
        sel = pd.read_parquet(p)
        out.update(dict(zip(sel["clip_id"].astype(str),
                            sel["chunk"].astype(int))))
    if not out:
        raise SystemExit("[extr] no selection table under %s/r0" % root)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the PhysicalAI root holding calibration/ and r0/")
    ap.add_argument("--corpus", default="physicalai-train-e438721ae894",
                    help="the parity manifest key whose clip digest must match")
    ap.add_argument("--clips", default=None,
                    help="optional explicit comma-separated clip_id list "
                         "(otherwise the whole r0 selection is used)")
    ap.add_argument("--clips-json", default=None,
                    help="a JSON list of clip_ids -- the honest source for the "
                         "parity set, whose 2,400 names live in the v2 cache "
                         "(HF) and in no local selection table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-digest-mismatch", action="store_true",
                    help="write the table even if the clip set does not match "
                         "the committed parity digest. Stamped into the file.")
    a = ap.parse_args(argv)

    root = Path(a.root)
    chunk_of = clip_list(root)
    if a.clips_json:
        want = sorted(json.loads(Path(a.clips_json).read_text(
            encoding="utf-8")))
    elif a.clips:
        want = [c.strip() for c in a.clips.split(",") if c.strip()]
    else:
        want = sorted(chunk_of)
    print("[extr] r0 selection: %d clips; asked for %d"
          % (len(chunk_of), len(want)), flush=True)

    # -- the POSITIVE assertion that this is the parity clip set ------------
    digest = hashlib.sha256(
        "\n".join(sorted(want)).encode("utf-8")).hexdigest()
    from tanitad.data import parity as pa
    ent = pa.manifest_entry(a.corpus) or {}
    want_dig = (ent.get("clip_membership") or {}).get("clip_id_sha256_sorted")
    match = (want_dig is not None and digest == want_dig)
    print("[extr] clip_id_sha256_sorted read %s / manifest %s -> %s"
          % (digest[:16], (want_dig or "NONE")[:16],
             "MATCH" if match else "MISMATCH"), flush=True)
    if not match and not a.allow_digest_mismatch:
        raise SystemExit(
            "[extr] REFUSING: the clip set does not match the committed "
            "parity digest for %r. Parity is sacred -- a table built over a "
            "re-selected clip set would silently supply cameras for a "
            "different corpus. Pass --allow-digest-mismatch to write it "
            "anyway (the mismatch is stamped into the file)." % a.corpus)

    by_chunk: dict = {}
    for c in want:
        k = chunk_of.get(c)
        if k is None:
            continue
        by_chunk.setdefault(int(k), []).append(c)

    import pandas as pd
    out, missing_chunk, missing_clip = {}, [], []
    cal = root / "calibration" / "sensor_extrinsics"
    for k in sorted(by_chunk):
        p = cal / ("sensor_extrinsics.chunk_%04d.parquet" % k)
        if not p.exists():
            missing_chunk.append(k)
            missing_clip.extend(by_chunk[k])
            continue
        rows = _front_wide_rows(pd.read_parquet(p))
        have = {}
        for r in rows.itertuples(index=False):
            have[str(r.clip_id)] = dict(
                qx=float(r.qx), qy=float(r.qy), qz=float(r.qz),
                qw=float(r.qw), x=float(r.x), y=float(r.y), z=float(r.z))
        for c in by_chunk[k]:
            if c in have:
                out[c] = have[c]
            else:
                missing_clip.append(c)

    if not out:
        raise SystemExit(
            "[extr] REFUSING to write an EMPTY table -- zero clips resolved "
            "from %d chunks. That is indistinguishable from a failed read, "
            "and an empty table would make every per-clip camera None while "
            "config.json still read mount_pose_scope PER-CLIP." % len(by_chunk))

    zs = sorted(v["z"] for v in out.values())
    xs = sorted(v["x"] for v in out.values())

    def pitch(v):
        n = math.sqrt(v["qx"]**2 + v["qy"]**2 + v["qz"]**2 + v["qw"]**2) or 1.0
        qx, qy, qz, qw = (v[k] / n for k in ("qx", "qy", "qz", "qw"))
        # R[2,2] of the cam->vehicle rotation applied to the camera's +z
        b2 = 1 - 2 * (qx * qx + qy * qy)
        return math.degrees(math.asin(max(-1.0, min(1.0, -b2))))

    ps = sorted(pitch(v) for v in out.values())
    n_dist = len(set(round(z, 6) for z in zs))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    stats = {
        "n_clips_requested": len(want), "n_clips_written": len(out),
        "n_clips_missing": len(missing_clip),
        "n_chunks_missing": len(missing_chunk),
        "clip_id_sha256_sorted": digest,
        "parity_manifest_digest": want_dig,
        "parity_digest_match": match,
        "height_m": {"min": zs[0], "median": zs[len(zs) // 2], "max": zs[-1],
                     "n_distinct": n_dist},
        "forward_offset_m": {"min": xs[0], "max": xs[-1]},
        "pitch_deg": {"min": ps[0], "max": ps[-1]},
        "sensor": SENSOR,
        "source": "PhysicalAI-AV calibration/sensor_extrinsics parquets",
    }
    Path(str(a.out) + ".stats.json").write_text(json.dumps(stats, indent=2),
                                                encoding="utf-8")
    print("[extr] wrote %s  %d/%d clips" % (a.out, len(out), len(want)))
    print("[extr] height  min %.4f  median %.4f  max %.4f  (%d distinct)"
          % (zs[0], zs[len(zs) // 2], zs[-1], n_dist))
    print("[extr] forward %.4f - %.4f m   pitch %+.3f .. %+.3f deg"
          % (xs[0], xs[-1], ps[0], ps[-1]))
    if missing_clip:
        print("[extr] MISSING %d clips over %d absent chunks (e.g. %s)"
              % (len(missing_clip), len(missing_chunk), missing_clip[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
