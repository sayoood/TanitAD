"""Re-issue MANIFEST.json now that the camera ships as BYTES, not as a recipe.

The v1 manifest described the camera as a RETRIEVAL TOOL (`tools/pull_camera.py`)
because the frames were to be pulled by the consumer. The PI changed that scope:
the 120 deg front camera ships INSIDE the private dataset. So the manifest must
now describe shipped, validated, hashed bytes -- and carry the validation verdict
that gates handover.

Everything here is read from the VALIDATION ARTIFACTS, never re-asserted by hand.
"""
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BUNDLE = REL / "tanitad-v7-training-corpus"
CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")

val = json.load(open(REL / "validation.json"))
shas = json.load(open(REL / "camera_sha256.json"))
fails = json.load(open(REL / "validation_failures.json"))
man = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))

assert val["verdict"] == "PASS", f"refusing to re-issue on verdict {val['verdict']}"
assert val["camera_ok"] == 4719 and not fails, (val["camera_ok"], len(fails))
tot = sum(v["bytes"] for v in shas.values())
assert len(shas) == 4719

# ship the per-file hash table with the bundle
(BUNDLE / "camera").mkdir(exist_ok=True)
dst = BUNDLE / "camera" / "camera_sha256.json"
dst.write_text(json.dumps(shas, indent=0, sort_keys=True), encoding="utf-8")

man["camera"] = {
    "status": "SHIPPED_IN_DATASET",
    "changed_from_v1": ("v1 shipped a RETRIEVAL TOOL (tools/pull_camera.py); the PI "
                        "directed that the frames themselves ship inside the private "
                        "dataset. The tool is retained for provenance/reproduction."),
    "feature": "camera_front_wide_120fov",
    "path": "camera/<clip_id>.mp4",
    "n_files": len(shas),
    "bytes_total": tot,
    "gb_total": round(tot / 1e9, 2),
    "per_file_sha256": "camera/camera_sha256.json",
    "validated": "every file decode-probed; see validation.md / validation.json",
    "per_clip_cy_is_mandatory_for_this_camera": {
        "file": "index/front_wide_cy.parquet",
        "why": ("TWO front-wide rigs (cy~543 A / cy~755 B). A geometric-centre crop "
                "is ~215 px wrong for rig B, which is the MAJORITY (2,723 of 4,719). "
                "Every epcache build from these mp4s MUST crop around the per-clip cy "
                "in index/front_wide_cy.parquet."),
        "rig_split": val["rig_split"],
    },
}
man["validation"] = {
    "verdict": val["verdict"],
    "validated_on": "2026-08-30",
    "gate": "code/validate_release.py",
    "joins": val["joins"],
    "camera_decode_probe": f"{val['camera_ok']}/{val['n_clips']}",
    "exclusions_written": len(fails),
    "checks": ["mp4 ftyp magic", "first+last frame decode, mean > 2.0 (zeros trap)",
               "duration 20 +/- 3 s",
               "frame count vs duration x fps +/- 5 % (truncation)",
               "clip-UUID joins across labels/camera/egomotion/cy/alpamayo",
               "per-file sha256"],
    "gate_defects_found_and_fixed_before_use": [
        "use-after-free reading PyAV stream attrs after container close (segfault, "
        "exit 139, no traceback) -- reads moved inside the container block",
        "seek(dur-0.5s) could yield no frame on a HEALTHY clip, reporting 129/3176 "
        "(~4%) as corrupt; hand-decoding three showed all 605 frames present. Fixed "
        "with back-off + full-decode fallback: 129 -> 0. See RETRACTION_LOG DE-C153 "
        "(class C83, an instrument that invents the defect it reports).",
    ],
}
man["exclusions"] = {"count": 0, "policy": man.get("exclusions", {}).get(
    "policy", "append-with-reason; never a silent skip"), "entries": []}

out = BUNDLE / "MANIFEST.json"
out.write_text(json.dumps(man, indent=1), encoding="utf-8")
sha = hashlib.sha256(out.read_bytes()).hexdigest()
print(f"MANIFEST re-issued: {out}")
print(f"  camera: {len(shas)} files, {tot/1e9:.2f} GB, verdict {val['verdict']}, "
      f"exclusions {len(fails)}")
print(f"  MANIFEST sha256 {sha[:16]}")
