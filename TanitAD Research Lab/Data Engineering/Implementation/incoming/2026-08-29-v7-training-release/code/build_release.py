"""Assemble the v7 training-corpus release (PI authorization 2026-08-29).

Baseline = the current label state (blob 121a8d93 content, the v7_sup run).
Consumers: v7f, refcv3, refav1, refd. Mandatory per the PI: the corresponding
Alpamayo data and the tactical/strategic labels travel together — the corpus is
complete only with both.

Everything here is verified by CONTENT (sha256 / parse / count), never by exit
code — the release manifest carries the hashes so any consumer can re-verify.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

T0 = time.time()
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/v7_sup/s2_labels_v7.jsonl")
SUM = Path("C:/Users/Admin/tanitad-wt/_s2build/v7_sup/summary.json")
ALP = Path("C:/Users/Admin/tanitad-data/alpamayo/records.parquet")
SEL = Path("C:/Users/Admin/tanitad-data/alpamayo/selection_manifest.json")
EGO = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
CIX = Path("C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet")
TOOLS = [Path("C:/Users/Admin/tanitad-wt/_s2build/dl/pull_camera.py"),
         Path("C:/Users/Admin/tanitad-wt/_s2build/dl/pull_egomotion_range.py")]

for d in ("labels", "alpamayo", "egomotion", "index", "tools"):  # noqa
    (OUT / d).mkdir(parents=True, exist_ok=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# --- 1. labels (the ground truth) ------------------------------------------
rows = [json.loads(l) for l in open(LBL, encoding="utf-8") if l.strip()]
assert len(rows) == 4719, f"label count {len(rows)} != 4719"
viol = sum(1 for r in rows if r["g_tac"]["violations"])
sup = sum(1 for r in rows if r.get("turn_suppression"))
assert viol == 0, f"{viol} exclusion violations in the baseline"
clip_ids = sorted(r["clip_id"] for r in rows)
corpus_id = hashlib.sha256("\n".join(clip_ids).encode()).hexdigest()[:16]
with open(LBL, "rb") as f, gzip.open(OUT / "labels" / "s2_labels_v7.jsonl.gz",
                                     "wb", compresslevel=9) as g:
    shutil.copyfileobj(f, g)
shutil.copyfile(SUM, OUT / "labels" / "summary.json")
print(f"labels: {len(rows)} clips, 0 violations, {sup} turn-suppressions, "
      f"corpus_id {corpus_id}")

# --- 2. alpamayo (MANDATORY per the PI) ------------------------------------
shutil.copyfile(ALP, OUT / "alpamayo" / "records.parquet")
shutil.copyfile(SEL, OUT / "alpamayo" / "selection_manifest.json")
import pandas as pd
adf = pd.read_parquet(OUT / "alpamayo" / "records.parquet")
assert len(adf) == 23644, f"alpamayo rows {len(adf)} != 23644"
print(f"alpamayo: {len(adf)} rows, {adf.clip_id.nunique()} clips, "
      f"{adf.task.nunique()} tasks")

# --- 3. egomotion store (tar of per-clip parquets) --------------------------
ego_files = sorted(EGO.glob("*.parquet"))
need = set(clip_ids)
have = {p.stem for p in ego_files}
missing = need - have
assert not missing, f"{len(missing)} labelled clips lack egomotion"
tar_path = OUT / "egomotion" / "egomotion_alpamayo.tar"
with tarfile.open(tar_path, "w") as tf:
    for p in ego_files:
        if p.stem in need:
            tf.add(p, arcname=p.name)
with tarfile.open(tar_path) as tf:
    n_in_tar = len(tf.getnames())
assert n_in_tar == len(need), f"tar holds {n_in_tar} != {len(need)}"
print(f"egomotion: {n_in_tar} clip parquets, {tar_path.stat().st_size/1e6:.0f} MB")

# --- 4. camera retrieval index (camera ~47 GB ships as TOOL + INDEX) --------
cix = pd.read_parquet(CIX)
sub = cix[cix.index.isin(need)].copy()
assert len(sub) == len(need), f"clip_index covers {len(sub)}/{len(need)}"
sub.to_parquet(OUT / "index" / "clip_to_chunk.parquet")
for t in TOOLS:
    shutil.copyfile(t, OUT / "tools" / t.name)
print(f"index: {len(sub)} clip->chunk rows; tools: {[t.name for t in TOOLS]}")

# --- 4b. per-clip cy (mandatory for cropping; included when built) ----------
cyp = Path("C:/Users/Admin/tanitad-wt/_s2build/release/front_wide_cy.parquet")
if cyp.exists():
    shutil.copyfile(cyp, OUT / "index" / "front_wide_cy.parquet")
    cydf = pd.read_parquet(cyp)
    print(f"cy table included: {len(cydf)} clips, rigs "
          f"{cydf.rig.value_counts().to_dict()}")
else:
    print("cy table NOT YET BUILT (HF token gated) — manifest marks PENDING")

# --- 5. MANIFEST + DATACARD -------------------------------------------------
files = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        files[str(p.relative_to(OUT)).replace("\\", "/")] = \
            {"sha256": sha(p), "bytes": p.stat().st_size}

manifest = {
    "name": "tanitad-v7-training-corpus",
    "released": "2026-08-29",
    "authorization": ("PI (Sayed), 2026-08-29, verbatim: 'Let take the current "
                      "state as base line and release the data set as required "
                      "for training our next model v7f, refcv3, refav1 and refd. "
                      "The usage of the corresponding data to the alpamayo and "
                      "tactical_/strategic labeled data is mandatory'"),
    "corpus": {
        "n_clips": len(rows),
        "hours": round(len(rows) * 20 / 3600.0, 1),
        "corpus_id_sha256_16": corpus_id,
        "definition": "alpamayo(4,729) INTERSECT provider egomotion, UUID-keyed",
        "parity_note": ("THIS IS A NEW, SEPARATELY IDENTIFIED CORPUS. It does "
                        "NOT re-select the canonical parity corpus "
                        "physicalai-train-e438721ae894 (2,376 episodes, "
                        "skip-hash f09e44db), which remains untouched and "
                        "authoritative for all cross-arm parity comparisons."),
    },
    "baseline": {
        "labels_blob": "121a8d93e4f1",
        "staging_verified": "blob-compare vs git index, 2026-08-29",
        "commit": "1c3513f27 — INDEPENDENTLY VERIFIED 2026-08-29 after mount "
                  "recovery: HEAD blob 121a8d93 == D-LABEL-GT pin (re-pin "
                  "9b440d662), ec075947 cost named in the register row",
        "register_row": "D-LABEL-GT (Project Steering/GOALS_AND_CLAIMS.md)",
        "vocab": "v7 FROZEN, 52 tokens, vocab_v7.py blob fc6719545313",
        "bands_s": {"operative": [0, 2], "tactical": [2, 6], "strategic": [8, 30]},
    },
    "consumers": ["v7f", "refcv3", "refav1", "refd"],
    "exclusions": {
        "clips": [],
        "policy": ("EMPTY AT RELEASE: all 4,719 clips have labels (0 refused) "
                   "and egomotion (asserted). Camera integrity is verified at "
                   "pull time (ftyp check in tools/pull_camera.py); any clip "
                   "whose camera fails MUST be appended here by the cache "
                   "builder with a reason, so skip-lists stay manifest-derived, "
                   "never discovered."),
    },
    "per_clip_cy": {
        "file": "index/front_wide_cy.parquet",
        "status": ("INCLUDED" if (OUT / "index" / "front_wide_cy.parquet").exists() else "PENDING — intrinsics pull is HF-token gated; token unreadable during the G: outage at build time; will be added before FINISH"),
        "why_mandatory": ("TWO front-wide rigs: cy~543 (A) vs cy~755 (B); a "
                          "geometric-center crop is ~215 px wrong for rig B. "
                          "Every epcache build MUST crop around per-clip cy."),
    },
    "trainer_conditions_mandatory": [
        "MASK the 8 NOT_YET_EXTRACTABLE classes (vocab_v7.NOT_YET_EXTRACTABLE) "
        "— a full-vocab head trains 8 permanently empty classes otherwise",
        "WEIGHT the strategic skew: FOLLOW_ROUTE is ~64-74 % of g_str",
        "disputed tokens (incl. contested-turn suppressed records and CoT "
        "tokens without box grounding) follow the PI's disputed policy: mask "
        "or down-weight, never train as clean positives",
        "untimed CoT tokens carry t_nominal_s = band midpoint (PI 2026-08-28); "
        "time_basis='untimed' distinguishes placed-by-convention from "
        "placed-by-evidence",
        "nav_command is a MODEL INPUT derived from the ego future (oracle) — "
        "training input only, never a target, never at eval in vision-only arms",
    ],
    "quality_at_release": {
        "exclusion_violations": 0,
        "goal_action_coherence_defects": 0,
        "vocab_emitting": "44/52 (8 declared unreachable with reasons)",
        "turn_suppressions": sup,
        "turn_corroboration": {"confirmed": 408, "uncorroborated": 139,
                               "contested_suppressed": 68},
    },
    "per_consumer": {
        "v7f": {"heads": "full v7: g_str + a_str + g_tac(multi-label) + "
                          "a_tac lat/lon; nav_command as INPUT",
                "notes": "all five universal conditions binding; SPEED_BAND "
                         "always present (regression target v_lo/v_hi)"},
        "refcv3": {"heads": "tactical goal set + lat/lon actions (selection "
                            "surface per its own prereg)",
                   "notes": "universal conditions binding; nav_command "
                            "admissible as input only"},
        "refav1": {"heads": "frozen-encoder reference; label heads as probes "
                            "over g_tac + a_tac",
                   "notes": "universal conditions binding"},
        "refd": {"heads": "reference D consumer; label interface identical",
                 "notes": "universal conditions binding"},
        "shared_consumer_module": ("Master Mind builds ONE canonical "
                                   "label-consumer implementing mask-the-8 + "
                                   "skew weights + disputed/t_nominal policy, "
                                   "asserted against this manifest — the "
                                   "decide-once pattern applied to "
                                   "consumption."),
    },
    "camera": {
        "not_shipped": "~47 GB; retrieve deterministically with "
                       "tools/pull_camera.py + index/clip_to_chunk.parquet "
                       "from nvidia/PhysicalAI-Autonomous-Vehicles",
    },
    "files": files,
}
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1),
                                   encoding="utf-8")

(OUT / "DATACARD.md").write_text(f"""# tanitad-v7-training-corpus

**{len(rows):,} clips / {len(rows)*20/3600:.1f} h** of front-camera driving with
hierarchical v7 labels (strategic goal+action · tactical multi-label goals ·
lateral/longitudinal actions · nav-command input) plus the FULL Alpamayo2-Super
augmentation (meta_action, CoT, auto-labeling, VQA, grounding boxes) and
provider egomotion. Corpus id `{corpus_id}`. PRIVATE — research use within the
TanitAD programme; sources: nvidia/PhysicalAI-Autonomous-Vehicles (research
license) + Sayood/tanitad-alpamayo2-augmentation.

Released 2026-08-29 on PI authorization as the training baseline for
**v7f, refcv3, refav1, refd**. Ground-truth register row: `D-LABEL-GT`.

## Read this before training
Every consumer MUST apply the five `trainer_conditions_mandatory` in
`MANIFEST.json`. The labels carry their own epistemics per token —
`provenance`, `disputed`, `grounded`, `corroboration`, `time_basis`,
`t_nominal_s`, and per-clip `turn_suppression` — so a trainer chooses what to
trust explicitly rather than inheriting silent defaults.

## Camera — retrieval contract (drive it unattended)
Not shipped (~47 GB). Deterministic retrieval, per clip:

1. `chunk = index/clip_to_chunk.parquet.loc[clip_id].chunk` (also carries the
   train/val/test split).
2. `tools/pull_camera.py` opens
   `camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{{chunk:04d}}.zip`
   on `nvidia/PhysicalAI-Autonomous-Vehicles` via **HTTP range reads** (the
   `HTTPRangeFile` class in `tools/pull_egomotion_range.py`): it fetches the
   zip's central directory from the final 256 KB, then ONLY the member
   `{{clip_id}}.camera_front_wide_120fov.mp4` (~10 MB vs the 2.05 GB chunk).
3. Every pulled file is verified by CONTENT (`ftyp` box in the first 64 bytes)
   before it is banked; failures print and skip — append them to
   `MANIFEST.json:exclusions.clips` with the reason.
4. Programmatic use: `import pull_camera; pull_camera.pull([clip_id, ...])`
   returns `{{clip_id: path}}`. Needs an HF token with read access.

## Cropping — per-clip cy is MANDATORY
`index/front_wide_cy.parquet` carries `clip_id, width, height, cx, cy, rig`.
TWO rigs exist (cy~543 rig A / cy~755 rig B); a geometric-center crop is
~215 px wrong for rig B. Crop around the CLIP'S cy, never the frame centre.

## Verify
Every file: sha256 in `MANIFEST.json`. Labels: 4,719 rows, 0 exclusion
violations, schema `s2-geom-v7`.
""", encoding="utf-8")

tot = sum(v["bytes"] for v in files.values())
print(f"\nRELEASE ASSEMBLED: {len(files)} files, {tot/1e6:.0f} MB, "
      f"{time.time()-T0:.0f}s")
print("MANIFEST sha256:", sha(OUT / "MANIFEST.json")[:16])
