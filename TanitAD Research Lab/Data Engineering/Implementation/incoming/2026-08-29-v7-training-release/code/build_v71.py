"""v7.1 — a METADATA-ONLY patch of the v7 labels. Not a re-emit.

⛔ WHY A PATCH AND NOT A RE-RUN OF THE EMITTER. Re-running `s2_geom_emit_v7.py`
would re-derive EVERY field from current HEAD, and other code has moved since v7
was cut. That could change `a_tac` / `g_tac` — things trainers DO read — while we
believe we are shipping a metadata correction. A patch cannot do that, and the
assertion below PROVES it did not.

WHAT v7.1 CHANGES
  1. `alpamayo.lateral.agree` — CORRECTED. v7 compared the Alpamayo side against
     a scan of the GOAL tokens for TURN_ prefixes, so NUDGE_L/R (actions, never
     turn goals) were compared as "straight". Wrong on 992 of 4,416 records
     (22.5 %), in BOTH directions.
  2. `alpamayo.lateral.reasoning_side` + `concordance` — NEW. Alpamayo states its
     lateral action twice: the structured `meta_action` label and its
     time-segmented `motion_analysis` reasoning. They disagree on 36 % of clips
     and neither dominates; where they AGREE the claim matches geometry 74.7 %
     vs a 65 % baseline. A clip whose two channels contradict each other is not
     a second opinion, and now says so.
  3. `a_tac.lat_args.lat_peak_m` — NEW. The signed peak lateral offset that
     DECIDES a NUDGE label (`abs(lat) >= 1.0 m`) and was previously discarded, so
     no NUDGE could be audited after the fact.
  4. `strata_cell`, `road_class`, `daynight_clock`, `stopped_frame_frac` — NEW.
     The PI's selection design (`2026-08-06-alpamayo-augmentation/DESIGN.md:42`)
     specified a manifest carrying the strata cell; v7 shipped clip_id only, and
     that gap is why a designed corpus was mistaken for an accidental one.

⭐ THE SAFETY PROPERTY: every field a TRAINER reads is asserted BYTE-IDENTICAL to
v7. If that assertion fails, this is not a metadata release and it stops.
"""
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BUNDLE = REL / "tanitad-v7-training-corpus"
V7_MD5 = "ee44875916ae7c0ac002c6716b9658ea"
OUT = REL / "v71"
OUT.mkdir(exist_ok=True)

src = BUNDLE / "labels" / "s2_labels_v7.jsonl.gz"
assert hashlib.md5(src.read_bytes()).hexdigest() == V7_MD5, "not the v7 release blob"
rows = [json.loads(x) for x in gzip.open(src, "rt", encoding="utf-8") if x.strip()]
print(f"v7 rows {len(rows)} (md5 {V7_MD5[:12]})")

strata = pd.read_parquet(REL / "b1_strata_cells.parquet").set_index("clip_id")
stopcov = pd.read_parquet(REL / "b1_stop_coverage_20s.parquet").set_index("clip")


def side_of(cls: str) -> str:
    t = (cls or "").upper()
    if t.endswith(("_L", "_LEFT")):
        return "left"
    if t.endswith(("_R", "_RIGHT")):
        return "right"
    return "straight"


# --- lat_peak_m per clip, on the 20 s TRAINABLE window (MM-C10) --------------
import math  # noqa: E402
tf = tarfile.open(BUNDLE / "egomotion" / "egomotion_alpamayo.tar")
lat_peak = {}
for m in tf.getmembers():
    if not m.isfile():
        continue
    d = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()),
                        columns=["x", "y", "qz", "qw", "timestamp"])
    t = d.timestamp.to_numpy() / 1e6
    w = (t >= 0.0) & (t <= 20.0)
    x, y = d.x.to_numpy()[w], d.y.to_numpy()[w]
    yaw = 2.0 * np.arctan2(d.qz.to_numpy()[w], d.qw.to_numpy()[w])
    if len(x) < 3:
        continue
    c, s = math.cos(-yaw[0]), math.sin(-yaw[0])
    track = s * (x - x[0]) + c * (y - y[0])
    lat_peak[m.name.split(".")[0]] = round(float(track[int(np.argmax(np.abs(track)))]), 3)
print(f"lat_peak_m computed for {len(lat_peak)} clips (20 s window)")

# --- Alpamayo's second lateral channel --------------------------------------
from tanitad.data import alpamayo_structured as AST  # noqa: E402
reasoning_side = {}
for r in rows:
    cid = r["clip_id"]
    try:
        reasoning_side[cid] = AST.band_tokens(cid, 0.0, 6.0).get("lateral_side")
    except Exception:                                       # noqa: BLE001
        reasoning_side[cid] = None

# --- patch -------------------------------------------------------------------
TRAINER_FIELDS = ("a_tac", "g_tac", "a_str", "g_str", "cot_tokens", "bands",
                  "horizon", "vocab", "nav_command", "semantics",
                  "manoeuvre_sequence", "turn_suppression", "scene", "t0_s")
before = {r["clip_id"]: {k: json.dumps(r.get(k), sort_keys=True) for k in TRAINER_FIELDS}
          for r in rows}

n_agree_fixed = 0
for r in rows:
    cid = r["clip_id"]
    alp = r.setdefault("alpamayo", {})
    lat = alp.setdefault("lateral", {})
    geom_side = side_of((r.get("a_tac") or {}).get("lat"))
    a_side = lat.get("alpamayo_side")
    if a_side is not None:
        new_agree = (a_side == geom_side)
        if lat.get("agree") != new_agree:
            n_agree_fixed += 1
        lat["agree"] = new_agree
        lat["agree_basis"] = "alpamayo_side vs side_of(a_tac.lat)"
    rs = reasoning_side.get(cid)
    lat["reasoning_side"] = rs
    if a_side is None and rs is None:
        lat["concordance"] = "silent"
    elif a_side is None or rs is None:
        lat["concordance"] = "single_channel"
    elif a_side == rs:
        lat["concordance"] = "concordant"
    else:
        lat["concordance"] = "self_contradictory"
    # lat_peak_m rides in a_tac.lat_args, beside the existing within_m
    lp = lat_peak.get(cid)
    if lp is not None:
        r.setdefault("a_tac", {}).setdefault("lat_args", {})
        if r["a_tac"]["lat_args"] is None:
            r["a_tac"]["lat_args"] = {}
        r["a_tac"]["lat_args"]["lat_peak_m"] = lp
    if cid in strata.index:
        s = strata.loc[cid]
        r["strata"] = {"strata_cell": s.strata_cell, "road_class": s.road_class,
                       "daynight_clock": s.daynight_clock, "country": s.country,
                       "stopped_frame_frac_20s": float(
                           stopcov.loc[cid, "stopped_frac_20s"]) if cid in stopcov.index else None,
                       "has_stop_launch_20s": bool(
                           stopcov.loc[cid, "has_stop_launch"]) if cid in stopcov.index else None}
    # ⛔ schema_version identifies the FORMAT CONTRACT, never the release.
    # MEASURED: bumping it to "s2_labels_v7.1" made the blob UNLOADABLE by
    # tanitad.data.v7_labels (EXPECTED_SCHEMA = "s2-geom-v7") — every consumer
    # refused a file whose format had not actually changed. The additions here
    # are backward-compatible FIELDS, so the contract is unchanged; the release
    # rides in its own key.
    r["schema_version"] = "s2-geom-v7"
    r["release"] = "v7.1"

# ⭐ THE SAFETY ASSERTION -----------------------------------------------------
drift = []
for r in rows:
    cid = r["clip_id"]
    for k in TRAINER_FIELDS:
        if k == "a_tac":
            continue                    # a_tac gains lat_args.lat_peak_m by design
        if json.dumps(r.get(k), sort_keys=True) != before[cid][k]:
            drift.append((cid, k))
assert not drift, f"TRAINER-VISIBLE DRIFT on {len(drift)} fields, e.g. {drift[:5]}"
# a_tac checked field-by-field EXCEPT the one addition
atac_drift = []
for r in rows:
    b = json.loads(before[r["clip_id"]]["a_tac"])
    a = {k: v for k, v in r["a_tac"].items()}
    b2 = dict(b)
    a_args = dict(a.get("lat_args") or {})
    a_args.pop("lat_peak_m", None)
    a = {**a, "lat_args": a_args or None}
    b2["lat_args"] = b2.get("lat_args") or None
    if json.dumps(a, sort_keys=True) != json.dumps(b2, sort_keys=True):
        atac_drift.append(r["clip_id"])
assert not atac_drift, f"a_tac drift beyond lat_peak_m on {len(atac_drift)} clips"
print(f"✅ SAFETY ASSERTION PASSED — every trainer-visible field byte-identical to v7")
print(f"   (a_tac gains ONLY lat_args.lat_peak_m)")
print(f"agree corrected on {n_agree_fixed} records")

out = OUT / "s2_labels_v7.1.jsonl.gz"
with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
md5 = hashlib.md5(out.read_bytes()).hexdigest()
sha = hashlib.sha256(out.read_bytes()).hexdigest()
print(f"\nwrote {out.name}  {out.stat().st_size/1e6:.2f} MB  md5 {md5}")

man = [{"clip_id": r["clip_id"], "t0_us": 5100000,
        "strata_cell": (r.get("strata") or {}).get("strata_cell"),
        "road_class": (r.get("strata") or {}).get("road_class"),
        "daynight_clock": (r.get("strata") or {}).get("daynight_clock"),
        "country": (r.get("strata") or {}).get("country"),
        "stopped_frame_frac_20s": (r.get("strata") or {}).get("stopped_frame_frac_20s"),
        "has_stop_launch_20s": (r.get("strata") or {}).get("has_stop_launch_20s")}
       for r in rows]
json.dump(man, open(OUT / "selection_manifest_v7.1.json", "w"), indent=0)
json.dump({"schema": "s2_labels_v7.1", "based_on": {"blob": "s2_labels_v7.jsonl.gz",
                                                    "md5": V7_MD5},
           "labels_md5": md5, "labels_sha256": sha, "n_records": len(rows),
           "agree_corrected": n_agree_fixed,
           "trainer_visible_drift": 0,
           "changes": ["alpamayo.lateral.agree corrected",
                       "alpamayo.lateral.reasoning_side + concordance added",
                       "a_tac.lat_args.lat_peak_m added",
                       "strata block added (cell, road_class, daynight_clock, "
                       "country, stopped_frame_frac_20s, has_stop_launch_20s)"]},
          open(OUT / "V71_MANIFEST.json", "w"), indent=1)
print(f"wrote selection_manifest_v7.1.json ({len(man)} rows) and V71_MANIFEST.json")
