#!/usr/bin/env python3
"""Build the val40 agents JSONL join ON THE DEV BOX, by IMPORTING
`stack/scripts/build_obstacle_join.py`'s functions — never forking its logic.

Why a driver exists at all: `build_obstacle_join.main` sources clips through
`corpus_first_clips`, which wants a v2 corpus of `*.v2ep.pt` payloads. This box holds the
val40 episode POSES-ONLY view (`ep_*.pt`, the episode contract, sha256-verified against
`manifest_EVALPOD_val40.json`) plus the per-clip label parquets (sha256-verified against the
Thor leadwork index — see `recover_val40_uuids.py`, 16/16 checks). So this driver substitutes
ONLY the ingest: poses from the ep cache, clip identity from the verified clip map, parquets
from the durable extract. Everything that computes — `EgoTrack`, `join_clip` (registration,
rig->world->ego composition, span rule, P4 occ flag), `open_out`, `write_records`,
`verify_with_reader`, `md5_of`, `assert_occ_matches_fov_mask` — is `build_obstacle_join`'s
own code, imported.

Outputs (--out, plain jsonl): the join + `<out>.meta.json` (same meta shape as
`build_obstacle_join.main`, plus this driver's provenance) and, with --xz-copy, an
xz-compressed byte-identical copy for the repo.

CPU-only. Touches no episode cache, no pod, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
for p in (REPO / "stack" / "scripts", REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_obstacle_join as boj  # noqa: E402  (the imported, never-forked logic)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("build_val40_join_local")
    ap.add_argument("--poses-dir", default="C:/Users/Admin/tanitad-caches/"
                    "val40-poses-20260818/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--labels-dir", default="C:/Users/Admin/tanitad-caches/"
                    "val40-obstacle-20260818")
    ap.add_argument("--clip-map", default=str(Path(__file__).resolve().parents[1]
                                              / "raw" / "val40_clipmap.json"))
    ap.add_argument("--out", required=True, help="plain jsonl output path")
    ap.add_argument("--xz-copy", default=None,
                    help="also write an xz-compressed byte-identical copy here")
    ap.add_argument("--tol-s", type=float, default=boj.DEFAULT_TOL_S)
    ap.add_argument("--hfov-deg", type=float, default=boj.HFOV_DEG_DEFAULT)
    a = ap.parse_args(argv)
    t0 = time.time()

    # the same pre-write self-check main() runs, for the same reason
    identity_check = boj.assert_occ_matches_fov_mask(a.hfov_deg)
    print(f"[join-local] P4 predicate self-check OK at hfov {a.hfov_deg} deg "
          f"({identity_check['n_cells']} cells, 0 disagree)", flush=True)

    import torch
    cm = json.loads(Path(a.clip_map).read_text(encoding="utf-8"))
    poses_dir = Path(a.poses_dir)
    labels = Path(a.labels_dir)
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fh = boj.open_out(out_p)
    per_clip, skipped = [], {"no_obstacle": [], "no_egomotion": [],
                             "registration_failed": [], "bad_clip": []}
    tot_frames = tot_boxes = tot_vis = n_joined = 0
    ls = boj._lead_source()
    import pandas as pd
    try:
        for i in range(len(cm)):
            cid = cm[str(i)]
            tag = f"[join-local] ep {i:03d} {cid}"
            ep = torch.load(str(poses_dir / f"ep_{i:05d}.pt"),
                            map_location="cpu", weights_only=False)
            poses = np.asarray(ep["poses"], dtype=np.float64)
            ego_p = labels / "egomotion" / f"{cid}.egomotion.parquet"
            obs_p = labels / "obstacle.offline" / f"{cid}.obstacle.offline.parquet"
            if not ego_p.exists():
                skipped["no_egomotion"].append(cid)
                print(f"{tag}: NO egomotion parquet — SKIP", flush=True)
                continue
            if not obs_p.exists():
                skipped["no_obstacle"].append(cid)
                print(f"{tag}: NO obstacle.offline — SKIP (frames stay NO_LABEL "
                      f"downstream, never free flow)", flush=True)
                continue
            try:
                ego = boj.EgoTrack(pd.read_parquet(ego_p))
                obs = pd.read_parquet(obs_p)
                recs, st = boj.join_clip(cid, poses, ego, obs,
                                         tol_s=a.tol_s, hfov_deg=a.hfov_deg)
            except ls.RegistrationError as e:
                skipped["registration_failed"].append(cid)
                print(f"{tag}: REGISTRATION FAILED — SKIP ({e})", flush=True)
                continue
            except (ValueError, KeyError) as e:
                skipped["bad_clip"].append(cid)
                print(f"{tag}: BAD CLIP DATA — SKIP ({e!r})", flush=True)
                continue
            boj.write_records(fh, recs)
            per_clip.append(st)
            n_joined += 1
            tot_frames += st["n_labelled"]
            tot_boxes += st["n_agent_boxes"]
            tot_vis += st["n_visible_boxes"]
            print(f"{tag}: frames {st['n_labelled']}/{st['n_frames']} labelled "
                  f"span {st['label_span_s'][0]:.2f}-{st['label_span_s'][1]:.2f}s "
                  f"tracks {st['n_tracks']} boxes {st['n_agent_boxes']} "
                  f"visible {st['visible_frac']} "
                  f"reg_res {st['registration']['residual_m']['median']}m "
                  f"b {st['registration']['b']:.6f}s", flush=True)
    finally:
        fh.close()

    if n_joined == 0:
        print("JOIN_FAILED no episode could be joined", flush=True)
        return 1

    try:
        ver = boj.verify_with_reader(out_p)
        ok = (ver["n_records"] == tot_frames and ver["n_clips"] == n_joined)
        print(f"[join-local] reader-verify: {ver} vs built (records {tot_frames}, "
              f"clips {n_joined}) -> {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            return 1
    except Exception as e:                                     # noqa: BLE001
        print(f"[join-local] WARNING: reader-verify unavailable here ({e!r}); "
              f"schema stays pinned by stack/tests/test_obstacle_join.py", flush=True)
        ver = {"error": repr(e)}

    digest = boj.md5_of(out_p)
    summary = {"n_episodes": n_joined, "n_episodes_requested": len(cm),
               "n_frames": tot_frames, "n_agent_boxes": tot_boxes,
               "visible_frac": round(tot_vis / tot_boxes, 4) if tot_boxes else None,
               "skipped": {s: v for s, v in skipped.items() if v},
               "out": str(out_p), "md5": digest,
               "wall_s": round(time.time() - t0, 1)}
    meta = {
        "task": "val40 obstacle.offline -> episode agents join (dev-box driver; "
                "logic = stack/scripts/build_obstacle_join.py, imported)",
        "driver": "build_val40_join_local.py",
        "corpus_key": "physicalai-val-0c5f7dac3b11",
        "clip_identity": "raw/val40_clipmap.json — 16/16 verification checks "
                         "(see uuid_recovery_verify.json)",
        "poses_source": "poses-only val40 view, sha256-verified vs "
                        "manifest_EVALPOD_val40.json (S3)",
        "labels_source": "per-clip parquets extracted from the local NVIDIA chunk "
                         "zips; bytes sha256-verified vs the Thor leadwork index "
                         "(S4); MANIFEST.md5 beside them",
        "args": vars(a), "summary": summary, "reader_verify": ver,
        "per_clip": per_clip,
        "conventions": {
            "frame": "per-frame EGO frame, +x fwd +y LEFT (refb_labels.ego_frame "
                     "via bev_raster.ego_frame_agents)",
            "frame_idx": "EPISODE index space; times recovered per episode by "
                         "lead_source.register_poses_to_time (grid ~0.1007 s)",
            "composition": "rig@sample -> world at the sample's OWN timestamp -> "
                           "ego@frame (build_obstacle_join.join_clip)",
            "NO_LABEL": "absent (clip, frame) line — outside the label span or "
                        "clip without obstacle.offline; NEVER emitted as empty "
                        "agents (that means labelled CLEAR)",
            "sample_rule": f"per-track NEAREST within {a.tol_s} s",
        },
        "p4_predicate_identity": dict(
            boj.P4_PREDICATE_IDENTITY, hfov_deg_used=float(a.hfov_deg),
            hfov_is_sensor_default=bool(a.hfov_deg == boj.HFOV_DEG_DEFAULT),
            self_check=identity_check),
        "_evidence_class": "MEASURED (ours; artifact = the jsonl + this meta)",
    }
    meta_p = Path(str(out_p) + ".meta.json")
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if a.xz_copy:
        xz_p = Path(a.xz_copy)
        xz_p.parent.mkdir(parents=True, exist_ok=True)
        raw = out_p.read_bytes()
        xz_p.write_bytes(lzma.compress(raw, preset=6))
        # roundtrip proof: decompressed bytes identical to the plain file
        back = lzma.decompress(xz_p.read_bytes())
        same = hashlib.md5(back).hexdigest() == digest
        print(f"[join-local] xz copy {xz_p} ({xz_p.stat().st_size / 1e6:.1f} MB) "
              f"roundtrip-md5 {'OK' if same else 'MISMATCH'}", flush=True)
        if not same:
            return 1
        Path(str(xz_p) + ".meta.json").write_text(
            json.dumps({**meta, "xz_of_md5": digest}, indent=1), encoding="utf-8")

    print(f"JOIN_DONE {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
