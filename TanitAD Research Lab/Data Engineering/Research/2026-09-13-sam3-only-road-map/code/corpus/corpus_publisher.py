"""Publisher for the corpus SAM3 semantic maps (PI 2026-09-15: "push the final dataset to my hf account ... and the semantic maps").
Incremental and resumable. Every clip the production ledger records OK and that is not yet published goes to the private corpus repo in
commits of at most 91 files (HF: keep manual commits around 50-100 files); a clip counts as published only after the Hub reports, for
every one of its files, the sha256 and size computed here (get_paths_info lfs.sha256). SEMANTIC_MAPS_MANIFEST.json is regenerated and
committed LAST on every run. Refuses to upload when the account's private storage plus the pending bytes would pass
PRIVATE_CEILING_GB (Pro: 1 TB private included, beyond it pay-as-you-go -- the PI's hard ceiling) or when this repo alone passes
REPO_CEILING_GB.
Repo layout: semantic_maps/gt/<clip_id>.sam3mapgt.npz (every check passed) · semantic_maps/gt_flagged/<clip_id>.sam3mapgt.npz (failed only where
nothing contradicts the map: see flag_of) · semantic_maps/worldmap/<clip_id>.worldmap.npz ·
semantic_maps/logs/logs_<NNNN>.tar (one per commit: <clip_id>/{report.json, ledger.json, refine.json, consensus.json,
consensus_atlas.png, stages.log}) · semantic_maps/SEMANTIC_MAPS_MANIFEST.json
Usage: corpus_publisher.py [--final] [--max-clips N] [--dry-run]   (--final: status COMPLETE when no clip is pending)"""
import argparse, collections, hashlib, io, json, os, sys, tarfile, time
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path("/home/nvidia/sam3map/corpus"); OUT = ROOT / "out"; PUB = ROOT / "publish"
REPO = "Sayood/tanitad-v7-training-corpus"; CAMERA_REV = "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"
DRIVER = Path("/home/nvidia/sam3map/eval/sam3map_prod.py")
CLIPS_PER_COMMIT = 45; PRIVATE_CEILING_GB = 900.0; REPO_CEILING_GB = 150.0
api = HfApi()


def log(msg):
    PUB.mkdir(exist_ok=True)
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
    print(line, flush=True)
    with open(PUB / "publisher.log", "a") as fh:
        fh.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha12(clip):
    return hashlib.sha256(clip.encode()).hexdigest()[:12]


UNTESTABLE = "path untestable: no seen cell on the ego's future path"
ROAD, NON_DRIVABLE = (1, 2, 3, 4, 6), (5, 7)


def reasons(row):
    """why a clip failed sam3map_prod.export_v2ep's gate, from its ledger row (same thresholds)"""
    e = (row or {}).get("export") or {}
    if not e:
        return ["no export report"]
    r = []
    if e.get("cart_seen_share", 0) <= 0.2:
        r.append("cart seen share <= 0.2")
    if e.get("cart_drivable_share", 0) <= 0.05:
        r.append("cart drivable share <= 0.05")
    if e.get("cells_bad_sum", 1) != 0:
        r.append("cell fractions do not sum to 1")
    real = (e.get("ego_future_path_on_drivable") or {}).get("real")
    if real is None:
        r.append(UNTESTABLE)
    elif real < 0.9:
        r.append("ego future path on drivable < 0.9")
    if e.get("orientation") == "MIRRORED?":
        r.append("mirrored")
    return r


def path_classes(npz):
    """class codes of the seen fine-grid cells under the ego's future path, with export_v2ep's geometry (next 30 frames, x >= 2 m)"""
    z = np.load(npz, allow_pickle=True); fc = z["fine_codes"]; T = z["T_world_rig"]; cnt = collections.Counter()
    for n in range(len(T)):
        fut = T[n + 1: n + 31, :2, 3]
        if not len(fut):
            continue
        q = (fut - T[n, :2, 3]) @ T[n, :2, :2]
        i = np.floor(q[:, 0] / 0.1).astype(int); j = np.floor((q[:, 1] + 16.0) / 0.1).astype(int)
        ok = (i >= 0) & (i < fc.shape[1]) & (j >= 0) & (j < fc.shape[2]) & (q[:, 0] >= 2.0)
        c = fc[n, i[ok], j[ok]]
        cnt.update(int(v) for v in c[c != 255])
    return {str(k): v for k, v in sorted(cnt.items())}


def flag_of(row, npz):
    """a failed clip ships in the FLAGGED tier only when nothing contradicts its map: every other check passed, and either the path
    check had nothing to test (a parked / stopped ego) or the path's off-road share is only UNLABELLED road ('seen, no class', typically
    the near field in front of the bonnet at walking pace) with no path cell on a non-drivable class and >= 50 % on road classes"""
    r = reasons(row)
    if r == [UNTESTABLE]:
        return "path untestable (parked / stopped ego)", None
    if r == ["ego future path on drivable < 0.9"] and npz.exists():
        pc = path_classes(npz); tot = sum(pc.values()); road = sum(v for k, v in pc.items() if int(k) in ROAD)
        if tot and sum(v for k, v in pc.items() if int(k) in NON_DRIVABLE) == 0 and road / tot >= 0.5:
            return "near path unlabelled (seen, no class), no path cell on a non-drivable class", pc
        return None, pc
    return None, None


def gt_path(clip, tier):
    """validated clips -> semantic_maps/gt/ (every check passed); flagged clips -> semantic_maps/gt_flagged/ (see flag_of; MEASURED
    2026-09-15: the first corpus failures were a car parked facing an embankment and a car creeping in a queue, both maps plausible)"""
    return f"semantic_maps/{'gt' if tier == 'validated' else 'gt_flagged'}/{clip}.sam3mapgt.npz"


def private_used_gb():
    me = api.whoami()["name"]; tot = 0; this = 0
    for lister, info in ((api.list_models, api.model_info), (api.list_datasets, api.dataset_info), (api.list_spaces, api.space_info)):
        for r in lister(author=me):
            i = info(r.id, expand=["usedStorage", "private"]); u = getattr(i, "used_storage", None) or 0
            tot += u if i.private else 0
            this += u if r.id == REPO else 0
    return tot / 1e9, this / 1e9


def tar_logs(batch, clips, last_row, tmp):
    p = tmp / f"logs_{batch:04d}.tar"
    with tarfile.open(p, "w") as tf:
        for clip in clips:
            s = sha12(clip)
            members = [(OUT / f"{s}.sam3mapgt.report.json", "report.json"), (OUT / "logs" / f"{s}.refine.json", "refine.json"),
                       (OUT / "logs" / f"{s}.consensus.json", "consensus.json"), (OUT / "logs" / f"{s}.consensus_atlas.png", "consensus_atlas.png"),
                       (OUT / "logs" / f"{s}.log", "stages.log")]
            blobs = [(name, src.read_bytes()) for src, name in members if src.exists()] + [("ledger.json", json.dumps(last_row[s], indent=1).encode())]
            for name, data in blobs:
                ti = tarfile.TarInfo(f"{clip}/{name}"); ti.size = len(data); ti.mtime = 0; ti.mode = 0o644
                tf.addfile(ti, io.BytesIO(data))
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--max-clips", type=int, default=10 ** 9)
    ap.add_argument("--dry-run", action="store_true", help="print the plan (tiers, paths, failure reasons); upload nothing")
    a = ap.parse_args()
    PUB.mkdir(exist_ok=True); tmp = PUB / "tmp"; tmp.mkdir(exist_ok=True)
    order = [l.strip() for l in open(ROOT / "production_order.txt") if l.strip()]; by_sha = {sha12(c): c for c in order}
    rows = []
    for l in (open(OUT / "manifest.jsonl") if (OUT / "manifest.jsonl").exists() else []):
        try:
            rows.append(json.loads(l))
        except ValueError:                                                     # the driver may be appending this very line
            pass
    last_row = {}; ok = set(); fails = collections.Counter(); status_last = {}; checks_failed = set()
    for r in rows:                                                             # last_row = the OK row when there is one, else the latest row
        s = r["clip_sha12"]
        if s not in ok:
            last_row[s] = r; status_last[s] = r.get("status")
        if r.get("status") == "OK":
            ok.add(s); last_row[s] = r; status_last[s] = "OK"
        elif str(r.get("status", "")).startswith("FAIL"):
            fails[s] += 1
            checks_failed |= {s} if r.get("status") == "FAIL-EXPORT-CHECKS" else set()
    published = {}
    if (PUB / "published.jsonl").exists():
        for l in open(PUB / "published.jsonl"):
            e = json.loads(l); published[e["clip_id"]] = e
    flags = {s: flag_of(last_row[s], OUT / f"{s}.sam3mapgt.npz") for s in checks_failed - ok if status_last.get(s) == "FAIL-EXPORT-CHECKS"}
    tier = {s: "validated" for s in ok}; tier.update({s: "flagged" for s, (f, _) in flags.items() if f})
    todo = [by_sha[s] for s in (sha12(c) for c in order) if s in tier and by_sha[s] not in published][:a.max_clips]
    log(f"ledger rows {len(rows)} | OK {len(ok)} | flagged {sum(1 for v in tier.values() if v == 'flagged')} | published {len(published)} | to publish {len(todo)}")
    if a.dry_run:
        for c in todo:
            print("PLAN", sha12(c), tier[sha12(c)], gt_path(c, tier[sha12(c)]).split("/")[1])
        for s in sorted(checks_failed - ok):
            print("FAILED", s, reasons(last_row[s]), "| flag:", flags.get(s, (None, None))[0], "| path classes:", flags.get(s, (None, None))[1])
        return
    batch = 1 + max([e["batch"] for e in published.values()], default=0)
    queued = 0
    if todo:                                                                   # one storage census per run; pending bytes accumulate
        priv, this = private_used_gb()
        log(f"storage: account private {priv:.2f} GB, this repo {this:.2f} GB (ceilings {PRIVATE_CEILING_GB} / {REPO_CEILING_GB})")
    for i in range(0, len(todo), CLIPS_PER_COMMIT):
        clips = todo[i:i + CLIPS_PER_COMMIT]
        files = {}
        for c in clips:
            s = sha12(c)
            for src, dst in ((OUT / f"{s}.sam3mapgt.npz", gt_path(c, tier[s])), (OUT / f"{s}.worldmap.npz", f"semantic_maps/worldmap/{c}.worldmap.npz")):
                files[dst] = (src, c)
        tarp = tar_logs(batch, clips, last_row, tmp); tar_dst = f"semantic_maps/logs/logs_{batch:04d}.tar"
        pending = sum(src.stat().st_size for src, _ in files.values()) + tarp.stat().st_size
        if priv + (queued + pending) / 1e9 > PRIVATE_CEILING_GB or this + (queued + pending) / 1e9 > REPO_CEILING_GB:
            log(f"REFUSED batch {batch}: private {priv:.1f} GB, this repo {this:.1f} GB, this run {(queued + pending) / 1e9:.3f} GB (ceilings {PRIVATE_CEILING_GB} / {REPO_CEILING_GB})")
            sys.exit(2)
        queued += pending
        local = {dst: (sha256(src), src.stat().st_size) for dst, (src, _) in files.items()}; local[tar_dst] = (sha256(tarp), tarp.stat().st_size)
        ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for dst, (src, _) in files.items()] + [CommitOperationAdd(path_in_repo=tar_dst, path_or_fileobj=str(tarp))]
        t0 = time.time()
        n_flag = sum(1 for c in clips if tier[sha12(c)] != "validated")
        info = api.create_commit(REPO, repo_type="dataset", operations=ops,
                                 commit_message=f"semantic_maps batch {batch:04d}: {len(clips)} clips (SAM3 front-camera map r, schema tanitad.sam3_map_gt/2)"
                                                + (f", {n_flag} flagged" if n_flag else ""))
        remote = {f.path: f for f in api.get_paths_info(REPO, list(local), repo_type="dataset", expand=True)}
        bad = [p for p, (h, n) in local.items() if p not in remote or remote[p].size != n or not remote[p].lfs or remote[p].lfs.sha256 != h]
        if bad:
            log(f"VERIFY-FAIL batch {batch} commit {info.oid[:10]}: {len(bad)} of {len(local)} files, e.g. {bad[:3]}"); sys.exit(3)
        with open(PUB / "published.jsonl", "a") as fh:
            for c in clips:
                fh.write(json.dumps({"clip_id": c, "sha12": sha12(c), "batch": batch, "commit": info.oid, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tier": tier[sha12(c)], "flag": (flags.get(sha12(c)) or (None, None))[0],
                                     "files": {dst: {"sha256": local[dst][0], "bytes": local[dst][1]} for dst in (gt_path(c, tier[sha12(c)]), f"semantic_maps/worldmap/{c}.worldmap.npz")},
                                     "logs_tar": {"path": tar_dst, "sha256": local[tar_dst][0], "bytes": local[tar_dst][1]}}) + "\n")
                published[c] = {"batch": batch}
        log(f"batch {batch:04d}: {len(clips)} clips, {len(local)} files, {pending / 1e6:.1f} MB, commit {info.oid[:10]}, {time.time() - t0:.0f}s, sha256 verified on the Hub")
        tarp.unlink(); batch += 1
    write_manifest(order, last_row, ok, fails, checks_failed, status_last, a.final, flags)


def write_manifest(order, last_row, ok, fails, checks_failed, status_last, final, flags):
    published = {}
    if (PUB / "published.jsonl").exists():
        for l in open(PUB / "published.jsonl"):
            e = json.loads(l); published[e["clip_id"]] = e
    prod_md = {}
    if published:
        first = next(iter(published)); z = np.load(OUT / f"{sha12(first)}.sam3mapgt.npz", allow_pickle=True); prod_md = json.loads(str(z["meta_json"]))
        prod_md.get("source", {}).pop("clip_sha12", None); prod_md.get("source", {}).pop("split", None)
    given_up = {s for s, n in fails.items() if s not in ok and (s in checks_failed or n >= 2)}     # = sam3map_prod.given_up with PROD_MAX_FAILS=2
    skip = set((OUT / "skip.txt").read_text().split()) if (OUT / "skip.txt").exists() else set()
    clips, flagged, not_pub = {}, {}, {}
    for c in order:
        s = sha12(c)
        if c in published:
            r = last_row[s]; e = published[c]
            entry = {"sha12": s, "split": r.get("split"), "frames_5hz": r.get("frames"), "frames_v2ep": (r.get("export") or {}).get("frames"),
                     "checks": {k: v for k, v in (r.get("export") or {}).items() if k != "bytes"}, "files": e["files"], "logs_tar": e["logs_tar"]["path"],
                     "commit": e["commit"]}
            if e.get("tier", "validated") == "validated":
                clips[c] = entry
            else:
                flagged[c] = dict(entry, failed_checks=reasons(r), flag=e.get("flag") or (flags.get(s) or (None, None))[0], path_classes=(flags.get(s) or (None, None))[1])
        else:
            st = ("SKIPPED-CRASH-LOOP" if s in skip else f"GIVEN_UP {status_last.get(s)}" if s in given_up else "NOT_PUBLISHED_YET" if s in ok
                  else f"RETRYING {status_last.get(s)}" if s in status_last else "PENDING")
            not_pub[c] = {"sha12": s, "status": st, "error": (last_row.get(s) or {}).get("error")}
            if s in given_up and (last_row.get(s) or {}).get("export"):
                not_pub[c]["checks"] = {k: v for k, v in last_row[s]["export"].items() if k != "bytes"}
                not_pub[c]["failed_checks"] = reasons(last_row[s])
                if (flags.get(s) or (None, None))[1] is not None:
                    not_pub[c]["path_classes"] = flags[s][1]
    pending = sum(1 for v in not_pub.values() if v["status"].split(" ")[0] in ("PENDING", "NOT_PUBLISHED_YET", "RETRYING"))
    status = "COMPLETE" if final and pending == 0 else "IN_PROGRESS"
    man = {"schema": "tanitad.semantic_maps_manifest/1", "component": "semantic_maps", "status": status, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "what": "per-frame BEV semantic ground truth from the SAM3 front-camera map r, one file per clip, on the v2ep EPISODE grid the BEV head trains on",
           "label_only": "non-causal (every frame of the clip builds the map): a training TARGET, never an inference input (inference is vision-only)",
           "corpus": {"repo": REPO, "clips_in_corpus": len(order), "camera_revision_used": CAMERA_REV, "camera_file": "camera/<clip_id>.mp4 (front-wide 120 fov)"},
           "counts": {"published": len(clips), "published_flagged": len(flagged), "flagged_by_reason": dict(collections.Counter(v["flag"] for v in flagged.values())), "not_published": len(not_pub),
                      "by_status": dict(collections.Counter(v["status"] for v in not_pub.values())),
                      "not_published_by_failed_check": dict(collections.Counter(x for v in not_pub.values() for x in v.get("failed_checks", []))),
                      "published_by_split": dict(collections.Counter(v["split"] for v in clips.values()))},
           "tiers": {"clips": "semantic_maps/gt/: every export check passed",
                     "clips_flagged": "semantic_maps/gt_flagged/: failed the gate only where nothing contradicts the map -- every other check passed and either the "
                                      "ego-future-path check had nothing to test (parked / stopped ego) or the path's off-road share is only unlabelled road "
                                      "('seen, no class', the near field in front of the bonnet at walking pace) with no path cell on a non-drivable class and "
                                      ">= 50 % on road classes; path_classes counts the codes under the path (1 drivable, 2 line, 3 crosswalk, 4 arrow, 6 hatched, "
                                      "0 seen-no-class, 5 edge, 7 sidewalk) -- opt in explicitly",
                     "not_published": "any other failed check (seen or drivable share too low, path on drivable < 0.9, mirrored), or not produced yet"},
           "gt_file_schema": prod_md,
           "production": {"host": "Jetson Thor", "driver": "sam3map_prod.py", "driver_md5": hashlib.md5(DRIVER.read_bytes()).hexdigest(),
                          "approved_config": "spdF4a: SAM3 841M, fp16 fusion encoder, fp32 decoder + backbone, 19 prompts batched, async CPU post",
                          "approval": "TanitAD repo, TanitAD Research Lab/Data Engineering/Research/2026-09-13-sam3-only-road-map/RESULT.md sections 19-20 (commits dbca361, c8eab57)",
                          "reproduction_check": "2026-09-15 corpus driver vs production_test2 on a validated clip: GT npz 9/9 arrays and worldmap 8/8 arrays bit-identical",
                          "per_clip_checks": "cart_seen_share > 0.2, cart_drivable_share > 0.05, no bad cell sums, ego future path on drivable >= 0.9, orientation not MIRRORED"},
           "verify": "every file in clips[*].files carries the sha256 the Hub reported after its commit; hf_hub_download + sha256 must match",
           "clips": clips, "clips_flagged": flagged, "not_published": not_pub}
    p = PUB / "SEMANTIC_MAPS_MANIFEST.json"
    if p.exists():                                                             # unchanged apart from the timestamp: no commit
        old = json.loads(p.read_text()); old.pop("generated", None)
        if old == {k: v for k, v in man.items() if k != "generated"} and not final:
            log(f"manifest unchanged ({len(clips)} published) - not re-uploaded"); return
    up = PUB / "SEMANTIC_MAPS_MANIFEST.json.upload"; up.write_text(json.dumps(man, indent=1))
    info = api.upload_file(path_or_fileobj=str(up), path_in_repo="semantic_maps/SEMANTIC_MAPS_MANIFEST.json", repo_id=REPO, repo_type="dataset",
                           commit_message=f"semantic_maps manifest: {len(clips)} clips published + {len(flagged)} flagged, status {status}")
    os.replace(up, p)                                                          # the local copy = what the Hub holds
    log(f"manifest: {len(clips)} published + {len(flagged)} flagged, {len(not_pub)} not published ({man['counts']['by_status']}), status {status}, "
        f"commit {getattr(info, 'oid', str(info))[:10]}")


if __name__ == "__main__":
    main()
