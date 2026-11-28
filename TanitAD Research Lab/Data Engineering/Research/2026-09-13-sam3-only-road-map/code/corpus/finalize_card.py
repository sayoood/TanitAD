"""Top-level release manifest AUG_2026_09_MANIFEST.json + DATACARD.md section for the 2026-09-15 augmentations, written into the
stage (then pushed with push_augment.py). The card edit is additive: one pointer paragraph after the intro, one section at the end;
the rest of the card is byte-identical (asserted). Numbers come from the component manifests, not from prose."""
import hashlib, json, re, sys, time
from pathlib import Path
import truststore; truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download

REPO = "Sayood/tanitad-v7-training-corpus"; STAGE = Path(r"D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage")
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"D:/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
api = HfApi(token=tok); NL = "\n"
tmp = STAGE.parent / "hub_copies"
card_old = Path(hf_hub_download(REPO, "DATACARD.md", repo_type="dataset", token=tok, local_dir=str(tmp))).read_text(encoding="utf-8")
sem = json.loads(Path(hf_hub_download(REPO, "semantic_maps/SEMANTIC_MAPS_MANIFEST.json", repo_type="dataset", token=tok, local_dir=str(tmp))).read_text())
lid = json.loads((STAGE / "lidar_bev_gt" / "LIDAR_BEV_GT_MANIFEST.json").read_text(encoding="utf-8"))
agt = json.loads((STAGE / "agents" / "AGENTS_MANIFEST.json").read_text(encoding="utf-8"))
tools = STAGE / "tools"; tools.mkdir(exist_ok=True)                        # ship the loader FIRST, so the manifest hashes the file that ships
src = Path(r"<scratchpad>/corpus/stage_tools/augment_loader.py")
(tools / "augment_loader.py").write_bytes(src.read_bytes())
files = json.loads((STAGE / "stage_files.json").read_text(encoding="utf-8"))
files["tools/augment_loader.py"] = {"sha256": hashlib.sha256((tools / "augment_loader.py").read_bytes()).hexdigest(), "bytes": (tools / "augment_loader.py").stat().st_size}
import pandas as pd
ci = pd.read_parquet(STAGE / "calibration" / "camera_intrinsics.parquet"); se = pd.read_parquet(STAGE / "calibration" / "sensor_extrinsics.parquet")

man = {"schema": "tanitad.corpus_augmentation_release/1", "release": "aug-2026-09", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": REPO,
       "corpus_id": "a48251e89c7a8603", "clips": 4719,
       "request": "PI 2026-09-15: the dataset should include all augmentations, the Alpamayo outputs, the newest nav commands, tactical and strategic labels and the semantic maps",
       "additive": "no pre-existing file was modified except DATACARD.md (one pointer paragraph + one appended section); alpamayo/vqa_bank_500.json is new",
       "components": {
           "camera": {"status": "SHIPPED, unchanged", "paths": "camera/<clip_id>.mp4", "verify": "camera/camera_sha256.json"},
           "egomotion / timestamps / index / splits": {"status": "SHIPPED, unchanged", "verify": "MANIFEST.json"},
           "alpamayo": {"status": "SHIPPED, unchanged", "paths": "alpamayo/records.parquet, alpamayo/selection_manifest.json",
                        "what": "Alpamayo2-Super augmentation: trajectory, meta_action, chain_of_causation, auto-labeling, VQA, grounding boxes (records.parquet byte-identical to Sayood/tanitad-alpamayo2-augmentation)",
                        "added_2026_09_15": "alpamayo/vqa_bank_500.json: the VQA question bank the vqa_qid column refers to, byte-identical to Sayood/tanitad-alpamayo2-augmentation@cedbf57c"},
           "labels": {"status": "SHIPPED, unchanged", "current_release": "v8.1", "paths": "labels/s2_labels_v8_{train,eval}.jsonl.gz",
                      "what": "strategic goal + action, tactical multi-label goals, lateral / longitudinal actions, nav_command + nav_30s, tac_SIT, lane_change_text, speed_max_input",
                      "verify": "V8_MANIFEST.json files.{train,eval} (never build_chain)"},
           "semantic_maps": {"status": "IN_PROGRESS (published every 6 h from Thor; the live status is semantic_maps/SEMANTIC_MAPS_MANIFEST.json:status)",
                             "paths": "semantic_maps/gt/<clip_id>.sam3mapgt.npz (all checks passed), semantic_maps/gt_flagged/<clip_id>.sam3mapgt.npz (failed only where nothing contradicts the map: parked / stopped ego, or near-field road unlabelled; reason per clip in the manifest), semantic_maps/worldmap/<clip_id>.worldmap.npz, semantic_maps/logs/logs_<NNNN>.tar",
                             "manifest": "semantic_maps/SEMANTIC_MAPS_MANIFEST.json", "schema": "tanitad.sam3_map_gt/2", "published_when_this_file_was_written": sem["counts"]["published"]},
           "lidar_bev_gt": {"status": "COMPLETE", "paths": "lidar_bev_gt/<clip_id>.bevgt.npz", "manifest": "lidar_bev_gt/LIDAR_BEV_GT_MANIFEST.json",
                            "schema": "tanitad.lidar_bev_gt/2", "clips": lid["clips"], "by_subset": lid["by_subset"], "by_split": lid["by_split"]},
           "agents/obstacle_offline": {"status": "COMPLETE (upstream labels, not an augmentation)", "paths": "agents/obstacle_offline/obstacle_offline_<NNN>.parquet + index.parquet",
                                       "manifest": "agents/AGENTS_MANIFEST.json", "clips_with_source_file": agt["clips_with_source_file"],
                                       "clips_with_zero_cuboids": agt["clips_with_zero_cuboids"], "clips_without_source_file": agt["clips_without_source_file"], "rows": agt["rows"]},
           "calibration": {"status": "COMPLETE (upstream rows)", "paths": "calibration/camera_intrinsics.parquet, calibration/sensor_extrinsics.parquet",
                           "rows": {"camera_intrinsics": len(ci), "sensor_extrinsics": len(se)}, "clips": int(ci["clip_id"].nunique())},
           "tools/augment_loader.py": {"what": "loaders for every component above + sha256 verification against the manifests"}},
       "files": {p: v for p, v in files.items() if p.startswith(("calibration/", "agents/", "tools/", "alpamayo/"))}}
(STAGE / "AUG_2026_09_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")

cls = ", ".join(f"{k} {v:,}" for k, v in sorted(agt["label_class_counts"].items(), key=lambda kv: -kv[1])[:6])
anchor = "Ground-truth register row: `D-LABEL-GT`." + NL
assert card_old.count(anchor) == 1, "card anchor"
pointer = (NL + "⭐ **2026-09-15 — augmentations added (release `aug-2026-09`):** per-frame SAM3 semantic maps (in progress), LiDAR BEV ground truth "
           "(315 clips), obstacle tracks and calibration. Everything is additive — see *Augmentations added 2026-09-15* at the end and "
           "`AUG_2026_09_MANIFEST.json`." + NL)
section = f"""
## Augmentations added 2026-09-15 (release `aug-2026-09`)

PI 2026-09-15: *"the data set should include all augmentations, the already generated alpamayo outputs, our already generated newest nav
commands, tactical and strategic labels and the semantic maps"*. **Everything here is additive** — no earlier file changed apart from this
card. Index of all components: `AUG_2026_09_MANIFEST.json`. Loaders and verification: `tools/augment_loader.py`.

| component | path | clips | status |
|---|---|---|---|
| Alpamayo2-Super outputs | `alpamayo/records.parquet` + `alpamayo/vqa_bank_500.json` | 4,719 | shipped before, unchanged; the VQA question bank added |
| labels v8.1 — strategic, tactical, nav | `labels/s2_labels_v8_{{train,eval}}.jsonl.gz` | 4,719 | shipped before, unchanged, newest release |
| **semantic maps (SAM3)** | `semantic_maps/gt/` (+ `gt_flagged/`), `semantic_maps/worldmap/` | 4,719 target | ⏳ **in progress** — read `semantic_maps/SEMANTIC_MAPS_MANIFEST.json:status` |
| **LiDAR BEV ground truth** | `lidar_bev_gt/<clip_id>.bevgt.npz` | {lid['clips']} | complete |
| obstacle tracks (upstream) | `agents/obstacle_offline/` | {agt['clips_with_source_file']:,} files | complete |
| calibration (upstream) | `calibration/` | {ci['clip_id'].nunique():,} | complete |

### Semantic maps — `semantic_maps/`

* **What.** Per-frame BEV semantic ground truth, one `<clip_id>.sam3mapgt.npz` per clip (schema `tanitad.sam3_map_gt/2`), sampled from the
  clip's composed world-frame map (`semantic_maps/worldmap/<clip_id>.worldmap.npz`, 0.1 m cells).
* **Time axis = the v2ep EPISODE grid** the BEV head trains on: `t_query = linspace(t_cam[0], t_cam[-1], int(span_s·10))` (~200 frames per
  clip), each frame the first camera frame at or after `t_query`; `T_world_rig` is egomotion interpolated at that frame.
* **Frame and grids.** Rig frame, +x forward, **+y LEFT**, origin rear axle on the road plane. `cart_frac` [T,9,120,64] at 0.5 m
  (0–60 m ahead, ±16 m; column 0 is the RIGHT edge) · `polar_frac` [T,9,24,20] and `polar48_frac` [T,9,48,40] over the 120° field to 60 m
  (column 0 = LEFT) · `fine_codes` [T,600,320] at 0.1 m (codes 0–7, 255 = not seen).
* **Channels** — soft cell fractions, uint8, divide by 255: seen-no-map-class · drivable · lane / road line · crosswalk · arrow / text ·
  non-drivable edge · hatched area · sidewalk / verge · not seen.
* **How it was made.** SAM3 (841 M) on the front-wide camera only, ground from the ego path (no LiDAR), production configuration approved
  against the PI-confirmed map r (fp16 fusion encoder; decoder and backbone fp32). Before launch the corpus driver reproduced the validated
  maps **bit-identically** (every array of both files).
* ⛔ **LABEL-ONLY and NON-CAUSAL.** Every frame of a clip builds its map, so a frame's label contains the future. A training target and an
  evaluation reference — **never an inference input** (inference is vision-only).
* **Per-clip quality gate** — seen share > 0.2, drivable > 0.05, cell fractions sum to 1, ≥ 90 % of the ego's future path on drivable, not
  mirrored. `orientation` reads `untestable` on straight paths and `undecided` in symmetric surroundings — neither is a failure.
* **Two tiers.** `semantic_maps/gt/` holds the clips that passed **every** check. `semantic_maps/gt_flagged/` holds clips that failed
  the gate only where **nothing contradicts the map** — every other check passed, and either (a) the path check had nothing to test: a
  **parked or stopped ego** leaves no seen cell on its own future path (MEASURED: 20 of the 4,719 clips never move 2 m ahead within 3 s),
  or (b) the path's off-road share is only **unlabelled road** ("seen, no class" — the near field in front of the bonnet when creeping in
  a queue) with **no path cell on a non-drivable class** and ≥ 50 % on road classes. Both first cases were checked against the camera:
  a car parked at night facing an embankment, and a car queued behind a truck; both maps plausible. Flagged clips carry their reason and
  the class counts under their path in `SEMANTIC_MAPS_MANIFEST.json:clips_flagged`; the loader needs `allow_flagged=True`.
  **Any other failure is not shipped**; it is listed under `not_published` with its failed checks.
* ⚠️ **Not measured on this corpus:** per-class accuracy against an independent reference. The LiDAR-based checks were run on the
  validation clips, not on these 4,719.
* **Verify:** every file carries the sha256 the Hub reported after its commit in `SEMANTIC_MAPS_MANIFEST.json:clips[*].files`.

### LiDAR BEV ground truth — `lidar_bev_gt/`

* {lid['clips']} clips ({lid['by_subset'].get('b1eval', 0)} `b1eval` + {lid['by_subset'].get('b1train200', 0)} `b1train200`), schema `tanitad.lidar_bev_gt/2`: occupancy, observed and
  camera-visible grids, plus `label_valid` per frame, **on the same grids and the same time axis as the semantic maps**.
* The deskew sign was corrected on 2026-09-13; schema /1 artifacts were wrong and none is shipped. Quarantined builds are excluded.
* ⛔ The top LiDAR is not a deployed sensor: target and evaluation reference only.
* Builder, checks and loader: TanitAD repo `TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/`.

### Obstacle tracks — `agents/obstacle_offline/` (upstream, not an augmentation)

* The published `obstacle.offline` cuboids (source `scene:obstacles:autolabels:v2`), rows unchanged: {agt['clips_with_source_file']:,} clips with a source file
  ({agt['clips_with_zero_cuboids']} of them with zero cuboids) and {agt['clips_without_source_file']} with no upstream file. {agt['rows']:,} rows. Top classes: {cls}.
* One parquet row group per clip with a `clip_id` column added — filter on it; `index.parquet` maps clip → shard and carries the source sha256.

### Calibration — `calibration/` (upstream rows)

* `camera_intrinsics.parquet` ({len(ci):,} rows): width, height, cx, cy, f-theta `fw_poly_0..4` / `bw_poly_0..4` for every camera of every corpus clip.
* `sensor_extrinsics.parquet` ({len(se):,} rows): each sensor's pose in the rig frame (quaternion qx qy qz qw + translation, metres).
* The semantic maps used the `camera_front_wide_120fov` rows.

### Verify

```
python tools/augment_loader.py --root <snapshot> verify semantic_maps
python tools/augment_loader.py --root <snapshot> verify lidar_bev_gt
python tools/augment_loader.py --root <snapshot> verify agents
python tools/augment_loader.py --root <snapshot> verify calibration
```
"""
card_new = card_old.replace(anchor, anchor + pointer).rstrip(NL) + NL + section
assert card_new.startswith(card_old.split(anchor)[0] + anchor) and card_old.split(anchor)[1].rstrip(NL) in card_new, "additive edit"
(STAGE / "DATACARD.md").write_text(card_new, encoding="utf-8", newline=NL)
listed = json.loads((STAGE / "stage_files.json").read_text(encoding="utf-8"))
for p in ("AUG_2026_09_MANIFEST.json", "DATACARD.md", "tools/augment_loader.py"):
    f = STAGE / p; listed[p] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
(STAGE / "stage_files.json").write_text(json.dumps(listed, indent=1), encoding="utf-8")
print("card", len(card_old), "->", len(card_new), "bytes | manifest components", len(man["components"]), "| semantic maps published", sem["counts"]["published"])
