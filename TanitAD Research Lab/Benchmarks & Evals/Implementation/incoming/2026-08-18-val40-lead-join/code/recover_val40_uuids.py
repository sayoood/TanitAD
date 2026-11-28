#!/usr/bin/env python3
"""Recover + VERIFY the 40 full clip UUIDs of val40 (`physicalai-val-0c5f7dac3b11`), in
val-list order, and emit the `--clip-map` file `taniteval.dump_lead_join` consumes.

Why this exists: the banked tier-0 dumps carry only 4-char eid prefixes (packed-ASCII
`episode_id`), and `attach_lead` refuses ambiguous prefixes — the val join needs full UUIDs
in val-list order (DUMP_LEAD_WIRING.md, 2026-08-18).

THE ADMISSIBLE STANDARD (task brief): two independent sources agreeing on the 40 UUIDs.
This script verifies FOUR mutually independent constraints and refuses on any mismatch:

  S1  Thor leadwork index (INHERITED, in-repo):
      `…/2026-08-18-thor-stranded-rescue/rescued_beyond_a11/leadwork/val40_lead_index.json`
      — ep_00000..ep_00039 -> full clip_id + HF chunk + per-clip label sha256s.
  S2  `stack/tanitad/data/deployed_val40_clip_digests.json` (in-repo, itself cross-checked
      against `val40_lead_index_ANON.json` clip_sha8s): 40 sha256(clip_id) digests
      + digest_of_digests. Independent of S1's ORDER (sorted-set comparison).
  S3  The poses cache (MEASURED here, loaded from the .pt files, not the sidecar json):
      `C:/Users/Admin/tanitad-caches/val40-poses-20260818/physicalai-val-0c5f7dac3b11/ep_*.pt`
      — packed `episode_id` big-endian-decodes to the clip UUID's first 4 chars, per
      episode, in file order; poses_sha256 re-hashed and checked against the committed
      eval-pod manifest `manifest_EVALPOD_val40.json` (the canonical val40 identity).
  S4  NVIDIA's own label chunk zips (MEASURED here, external to every program artifact):
      `<labels-root>/{egomotion,obstacle.offline}/<kind>.chunk_{chunk:04d}.zip` must contain
      member `{clip_id}.{kind}.parquet` for exactly the chunk S1 names, and the member
      BYTES must sha256-match S1's per-clip `egomotion_sha256`/`obstacle_sha256`.
      A wrong UUID or wrong chunk cannot pass this: the member would not exist.

Outputs (all in --out-dir):
  val40_clipmap.json          {eid(int-as-str): clip_uuid} — the `--clip-map` file
  val40_clips.json            full per-episode record (uuid, chunk, id4, sha256s, checks)
  uuid_recovery_verify.json   the verification result, every check with its count
Exit non-zero on ANY failed check. No network, no GPU, CPU-only, read-only on sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]

THOR_INDEX = (REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
              / "2026-08-18-thor-stranded-rescue/rescued_beyond_a11/leadwork"
              / "val40_lead_index.json")
DIGESTS = REPO / "stack/tanitad/data/deployed_val40_clip_digests.json"
EVALPOD_MANIFEST = (REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
                    / "2026-07-26-s3-decision-grade/artifacts/manifest_EVALPOD_val40.json")


def decode_packed(iv: int) -> str | None:
    """Big-endian ASCII unpacking — same rule as dump_lead_join._decode_packed."""
    if iv < (1 << 24):
        return None
    b = iv.to_bytes(8, "big").lstrip(b"\x00")
    try:
        s = b.decode("ascii")
    except UnicodeDecodeError:
        return None
    return s if s.isprintable() else None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("recover_val40_uuids")
    ap.add_argument("--poses-dir", default="C:/Users/Admin/tanitad-caches/"
                    "val40-poses-20260818/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--labels-root", default="C:/Users/Admin/tanitad-data/physicalai/labels")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    failed = []

    def check(name, ok, detail):
        checks[name] = {"ok": bool(ok), **detail}
        if not ok:
            failed.append(name)
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # ---- S1: the Thor leadwork index ------------------------------------- #
    idx = json.loads(THOR_INDEX.read_text(encoding="utf-8"))
    keys = sorted(idx.keys())
    want = [f"ep_{i:05d}.pt" for i in range(40)]
    check("S1_keys_are_ep00000_39", keys == want,
          {"n": len(keys), "first": keys[:1], "last": keys[-1:]})
    clips = [idx[k]["clip_id"] for k in want]
    check("S1_40_distinct_uuids", len(set(clips)) == 40, {"n_distinct": len(set(clips))})
    id4s = [c[:4] for c in clips]
    check("S1_40_distinct_id4_prefixes", len(set(id4s)) == 40,
          {"n_distinct": len(set(id4s)),
           "note": "prefix->clip resolution is unambiguous inside this join"})
    pfx_ok = all(idx[k].get("prefix") == c[:4] for k, c in zip(want, clips))
    check("S1_internal_prefix_consistent", pfx_ok, {})

    # ---- S2: the committed digest file (order-independent set match) ------ #
    dg = json.loads(DIGESTS.read_text(encoding="utf-8"))
    want_digests = sorted(dg["clip_id_digests"])
    have_digests = sorted(sha256_bytes(c.encode("utf-8")) for c in clips)
    check("S2_sha256_set_matches_deployed_digests", have_digests == want_digests,
          {"n": len(want_digests), "corpus_key": dg.get("corpus_key")})
    dod = sha256_bytes("\n".join(want_digests).encode("utf-8"))
    check("S2_digest_of_digests", dod == dg.get("digest_of_digests"),
          {"digest_of_digests": dod, "formula": "sha256('\\n'.join(sorted(digests)))"})
    css = sha256_bytes("\n".join(sorted(clips)).encode("utf-8"))
    check("S2_clip_id_sha256_sorted", css == dg.get("clip_id_sha256_sorted"),
          {"clip_id_sha256_sorted": css,
           "formula": "sha256('\\n'.join(sorted(clip_ids)))"})
    check("S2_corpus_key", dg.get("corpus_key") == "physicalai-val-0c5f7dac3b11",
          {"corpus_key": dg.get("corpus_key")})

    # ---- S3: poses cache .pt files (measured) vs packed ids + manifest ---- #
    import torch
    man = json.loads(EVALPOD_MANIFEST.read_text(encoding="utf-8"))
    man_rows = {r["file"]: r for r in man["episodes"]}
    n_id4, n_sha, n_T = 0, 0, 0
    per_ep = []
    for i, k in enumerate(want):
        p = Path(a.poses_dir) / k
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        eid_packed = int(d["episode_id"])
        id4 = decode_packed(eid_packed)
        poses = d["poses"]
        sha = sha256_bytes(poses.numpy().tobytes()
                           if hasattr(poses, "numpy") else bytes(poses))
        mr = man_rows[k]
        ok_id4 = id4 == clips[i][:4] and eid_packed == mr["episode_id"]
        ok_sha = sha == mr["poses_sha256"]
        ok_T = int(poses.shape[0]) == int(mr["T"]) == int(idx[k]["T"])
        n_id4 += ok_id4
        n_sha += ok_sha
        n_T += ok_T
        per_ep.append({"file": k, "eid": i, "clip_id": clips[i],
                       "chunk": int(idx[k]["chunk"]), "id4": id4,
                       "episode_id_packed": eid_packed, "T": int(poses.shape[0]),
                       "poses_sha256": sha,
                       "id4_matches_uuid": ok_id4, "poses_sha_matches_manifest": ok_sha})
    check("S3_packed_id4_matches_uuid_prefix_all40", n_id4 == 40, {"n_ok": n_id4})
    check("S3_poses_sha256_matches_evalpod_manifest_all40", n_sha == 40, {"n_ok": n_sha})
    check("S3_T_consistent_all40", n_T == 40, {"n_ok": n_T})

    # ---- S4: NVIDIA chunk zips — member existence + byte sha256 ----------- #
    lr = Path(a.labels_root)
    ego_ok = ego_sha_ok = obs_ok = obs_sha_ok = 0
    obs_absent_expected = []
    for i, k in enumerate(want):
        rec = idx[k]
        cid, chunk = clips[i], int(rec["chunk"])
        ez = lr / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
        with zipfile.ZipFile(ez) as z:
            names = {Path(n).name: n for n in z.namelist()}
            m = names.get(f"{cid}.egomotion.parquet")
            if m:
                ego_ok += 1
                b = z.read(m)
                s_ok = (sha256_bytes(b) == rec["egomotion_sha256"]
                        and len(b) == int(rec["egomotion_bytes"]))
                ego_sha_ok += s_ok
                per_ep[i]["egomotion_member_ok"] = True
                per_ep[i]["egomotion_sha_ok"] = bool(s_ok)
        oz = lr / "obstacle.offline" / f"obstacle.offline.chunk_{chunk:04d}.zip"
        if rec.get("obstacle") != "present":
            # S1 says absent-in-chunk; verify the member truly is absent there.
            absent = True
            if oz.exists():
                with zipfile.ZipFile(oz) as z:
                    absent = f"{cid}.obstacle.offline.parquet" not in {
                        Path(n).name for n in z.namelist()}
            obs_absent_expected.append({"file": k, "clip_id": cid, "chunk": chunk,
                                        "absent_confirmed": bool(absent)})
            per_ep[i]["obstacle_member_ok"] = None
            per_ep[i]["obstacle_absent_confirmed"] = bool(absent)
            continue
        with zipfile.ZipFile(oz) as z:
            names = {Path(n).name: n for n in z.namelist()}
            m = names.get(f"{cid}.obstacle.offline.parquet")
            if m:
                obs_ok += 1
                b = z.read(m)
                s_ok = (sha256_bytes(b) == rec["obstacle_sha256"]
                        and len(b) == int(rec["obstacle_bytes"]))
                obs_sha_ok += s_ok
                per_ep[i]["obstacle_member_ok"] = True
                per_ep[i]["obstacle_sha_ok"] = bool(s_ok)
    check("S4_egomotion_member_in_named_chunk_all40", ego_ok == 40, {"n_ok": ego_ok})
    check("S4_egomotion_bytes_sha256_match_thor_index_all40", ego_sha_ok == 40,
          {"n_ok": ego_sha_ok})
    check("S4_obstacle_member_in_named_chunk_39of39", obs_ok == 39, {"n_ok": obs_ok})
    check("S4_obstacle_bytes_sha256_match_thor_index_39of39", obs_sha_ok == 39,
          {"n_ok": obs_sha_ok})
    check("S4_single_obstacle_absent_is_ep_00037",
          [r["file"] for r in obs_absent_expected] == ["ep_00037.pt"]
          and all(r["absent_confirmed"] for r in obs_absent_expected),
          {"absent": obs_absent_expected})

    # ---- emit ------------------------------------------------------------- #
    clipmap = {str(i): clips[i] for i in range(40)}
    (out_dir / "val40_clipmap.json").write_text(
        json.dumps(clipmap, indent=1), encoding="utf-8")
    (out_dir / "val40_clips.json").write_text(json.dumps({
        "_what": "val40 (physicalai-val-0c5f7dac3b11) full clip UUIDs in val-list order, "
                 "with per-clip HF label chunk + verification bits",
        "_confidentiality": dg.get("confidentiality"),
        "corpus_key": "physicalai-val-0c5f7dac3b11",
        "episodes": per_ep}, indent=1), encoding="utf-8")
    verify = {
        "_what": "four-constraint verification of the recovered val40 clip UUIDs",
        "_evidence_class": {
            "S1": "INHERITED (Thor leadwork index, in-repo, rescued 2026-08-18)",
            "S2": "INHERITED (deployed_val40_clip_digests.json, in-repo; itself "
                  "cross-checked against val40_lead_index_ANON.json clip_sha8)",
            "S3": "MEASURED (this run: .pt episode_id decoded + poses re-hashed vs "
                  "the committed eval-pod manifest)",
            "S4": "MEASURED (this run: NVIDIA chunk-zip member bytes sha256 vs S1)",
        },
        "n_failed": len(failed), "failed": failed, "checks": checks,
    }
    (out_dir / "uuid_recovery_verify.json").write_text(
        json.dumps(verify, indent=1), encoding="utf-8")
    print(f"[recover_val40_uuids] {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + str(failed)}"
          f" -> {out_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
