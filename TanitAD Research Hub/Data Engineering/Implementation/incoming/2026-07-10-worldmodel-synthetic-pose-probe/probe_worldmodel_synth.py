"""Pose/action probe for PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios.

Backlog P0.1 (Data Engineering) — the GATING question for this 264k-clip / 8.3 TB
OpenMDW-1.1 corpus: does it ship an ego pose / action track?  If yes, the loader is
a near-zero-cost mirror of ``cosmos_drive.py`` (shared ``poses_to_signals`` + 9-ch
contract).  If no, the corpus is IDM/H7-gated (needs a trained inverse-dynamics head)
or video-only.  The HF card (read 2026-07-09) lists RGB + captions + metadata but no
pose; this probe settles it against the ACTUAL repo file tree, not the card prose.

The repo has ~264k clips x ~14 files ~= millions of paths, so a full ``list_repo_files``
walk hangs.  Instead this NAVIGATES the tree API structurally (root -> families ->
sampled clips -> clip fields), which is definitive because the per-clip layout is
homogeneous, and downloads ONE ``description/*.json`` to check whether pose is
embedded in the caption metadata.  Network-only; no GPU, no full download.

Run:  python probe_worldmodel_synth.py

TLS note (dev box): a TLS-intercepting proxy means only the Windows system trust
store has the issuing CA, so we inject it via ``truststore``.  Falls back cleanly.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios"
TREE = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
POSE_TOKENS = ("pose", "ego", "action", "traj", "trajectory", "odom", "imu",
               "calib", "intrinsic", "extrinsic", "vehicle", "can", "control",
               "steer", "accel", "yaw", "velocity", "gnss", "gps")


def _prep_tls() -> str:
    try:
        import truststore
        truststore.inject_into_ssl()
        return "truststore"
    except Exception:
        try:
            import os
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            return "certifi"
        except Exception:
            return "system"


def probe(clips_per_family: int = 3) -> dict:
    tls = _prep_tls()
    import httpx

    def tree(path: str = "") -> list[dict]:
        url = TREE + ("/" + path if path else "")
        r = httpx.get(url, timeout=30.0, follow_redirects=True,
                      params={"recursive": "false"})
        r.raise_for_status()
        return r.json()

    t0 = time.time()
    root = tree()
    families = [e["path"] for e in root if e["type"] == "directory"]
    root_files = [e["path"] for e in root if e["type"] == "file"]

    field_census: collections.Counter = collections.Counter()
    ext_census: collections.Counter = collections.Counter()
    pose_hits: list[str] = []
    sampled_clips: list[str] = []
    per_family_listed: dict[str, int] = {}

    for fam in families:
        fam_entries = tree(fam)
        clip_dirs = [e["path"] for e in fam_entries if e["type"] == "directory"]
        per_family_listed[fam] = len(clip_dirs)          # API caps a page at 1000
        for clip in clip_dirs[:clips_per_family]:
            sampled_clips.append(clip)
            for sub in tree(clip):                        # clip -> {video, description}
                field_census[sub["path"].split("/")[-1]] += 1
                if sub["type"] == "directory":
                    for f in tree(sub["path"]):           # files inside video/description
                        name = f["path"].split("/")[-1]
                        ext_census[Path(name).suffix.lower() or "<none>"] += 1
                        if any(tok in f["path"].lower() for tok in POSE_TOKENS):
                            pose_hits.append(f["path"])

    # inspect one description json for an embedded pose key
    desc_sample = None
    front = [c for c in sampled_clips]
    if front:
        from huggingface_hub import hf_hub_url
        djs = f"{front[0]}/description/front_wide.json"
        try:
            r = httpx.get(hf_hub_url(REPO_ID, djs, repo_type="dataset"),
                          timeout=30.0, follow_redirects=True)
            j = r.json()
            desc_sample = {
                "path": djs,
                "top_keys": list(j.keys()) if isinstance(j, dict) else type(j).__name__,
                "has_pose_key": bool(isinstance(j, dict) and any(
                    tok in k.lower() for k in j.keys() for tok in POSE_TOKENS)),
                "metadata_keys": list((j.get("metadata") or {}).keys())
                if isinstance(j, dict) else [],
                "raw_preview": json.dumps(j)[:700],
            }
        except Exception as e:
            desc_sample = {"path": djs, "error": f"{type(e).__name__}: {e}"}

    has_pose = bool(pose_hits) or bool((desc_sample or {}).get("has_pose_key"))
    return {
        "repo_id": REPO_ID,
        "tls_backend": tls,
        "probe_seconds": round(time.time() - t0, 1),
        "families": families,
        "root_files": root_files,
        "per_family_clip_dirs_listed_first_page": per_family_listed,
        "clips_sampled": len(sampled_clips),
        "per_clip_fields": dict(field_census),
        "file_ext_census_in_sample": dict(ext_census),
        "pose_action_file_hits": pose_hits,
        "num_pose_action_hits": len(pose_hits),
        "description_sample": desc_sample,
        "verdict": "NO-POSE (IDM/H7 or video-only)" if not has_pose
        else "POSE-PRESENT (mirror cosmos loader)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-family", type=int, default=3)
    ap.add_argument("--out", default="worldmodel_synth_probe.json")
    args = ap.parse_args()

    rep = probe(args.clips_per_family)
    Path(args.out).write_text(json.dumps(rep, indent=2))

    print(f"=== WMS pose probe ({rep['probe_seconds']}s, tls={rep['tls_backend']}) ===")
    print("families:", rep["families"])
    print("per-clip fields (subdirs):", rep["per_clip_fields"])
    print("file ext census (sample):", rep["file_ext_census_in_sample"])
    print(f"pose/action file hits: {rep['num_pose_action_hits']}", rep["pose_action_file_hits"][:10])
    ds = rep["description_sample"] or {}
    print("description top_keys:", ds.get("top_keys"))
    print("description metadata_keys:", ds.get("metadata_keys"))
    print("description has_pose_key:", ds.get("has_pose_key"))
    print("VERDICT:", rep["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
