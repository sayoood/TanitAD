#!/usr/bin/env python3
"""Stage the inputs of a clip list on the pod (idempotent, ledgered). Pod only; the HF token is read in place and never printed.

Per clip it stages, under --root (layout chosen so that the stack's own helpers work: the mp4 sits under a directory named r0, and
<root>/calibration/physicalai_front_wide_intrinsics.csv is the per-clip table the builder reads):
  r0/camera/<clip>.mp4                   camera video (HF Sayood/tanitad-v7-training-corpus, sha256 vs camera/camera_sha256.json)
  ts/<clip>.timestamps.parquet           from timestamps/timestamps.tar        (the same source as stage A, v2ep_pod.py:128-142)
  ego/<clip>.parquet                     from egomotion/egomotion_alpamayo.tar (the same source as stage A, v2ep_pod.py:144-170)
  obs/<clip>.parquet                     the clip's obstacle.offline rows (HF agents/obstacle_offline shard; one row group per clip)
  calibration/physicalai_front_wide_intrinsics.csv   copy of stage A's per-clip table (intrinsics_dev.csv)
  calibration/front_wide_extrinsics.csv  camera_front_wide_120fov rows of calibration/sensor_extrinsics.parquet
  ledger.jsonl                           one line per clip and attempt: ok flag, reason, sizes, sha256 of the mp4, seconds
A clip is OK when mp4 (sha verified), timestamps and egomotion exist; `obs` is required for a train clip (a train clip with no obstacle
source is "not stageable" and the card's replacement rule applies) and optional for an evaluation clip (flagged has_obs=false).
Shared downloads (tars, shards) go to --hf-dir and are re-creatable pulls; nothing is deleted.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enc_common import Layout, read_ids, sha256_file   # noqa: E402

HF = "Sayood/tanitad-v7-training-corpus"
INTR_SRC = "/workspace/v8proof/stageA/ops/intrinsics_dev.csv"
FRONT_WIDE = "camera_front_wide_120fov"
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = open("/root/.hf_token").read().strip()
    return _TOK


def log(m):
    print(f"[stage {time.strftime('%H:%M:%S')}] {m}", flush=True)


def hf_dl(path, local_dir, **kw):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF, path, repo_type="dataset", token=tok(), local_dir=local_dir, **kw)


def mp4_ledger(hf_dir):
    j = json.load(open(hf_dl("camera/camera_sha256.json", hf_dir), encoding="utf-8"))
    if isinstance(j, dict) and "files" in j and isinstance(j["files"], dict):
        j = j["files"]

    def hf_sha(c):
        for k in (c, f"{c}.mp4", f"camera/{c}.mp4"):
            if k in j:
                v = j[k]
                return v if isinstance(v, str) else v.get("sha256")
        return None
    return hf_sha


def extract_tar(tar_path, want, suffix, dest_fn, take_first=True):
    """Stream a tar once; write members of wanted clips (clip id = basename up to the first '.') that end with `suffix`."""
    n = 0
    seen = set()
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile():
                continue
            b = os.path.basename(m.name)
            if not b.endswith(suffix):
                continue
            cid = b.split(".")[0]
            if cid in want and cid not in seen:
                os.makedirs(os.path.dirname(dest_fn(cid)), exist_ok=True)
                with open(dest_fn(cid), "wb") as fh:
                    fh.write(t.extractfile(m).read())
                seen.add(cid)
                n += 1
                if len(seen) == len(want):
                    break
    return seen


def stage_shared(L, hf_dir, clips, need_obs):
    """Timestamps, egomotion, obstacle rows, calibration tables for the clips that lack them. Returns {clip: has_obs}."""
    import pandas as pd
    os.makedirs(f"{L.root}/calibration", exist_ok=True)
    if not os.path.exists(L.intr_csv):
        shutil.copy2(INTR_SRC, L.intr_csv)
    want_ts = {c for c in clips if not os.path.exists(L.ts(c))}
    if want_ts:
        p = hf_dl("timestamps/timestamps.tar", hf_dir)
        got = extract_tar(p, want_ts, ".timestamps.parquet", L.ts)
        log(f"timestamps: {len(got)}/{len(want_ts)} extracted")
    want_ego = {c for c in clips if not os.path.exists(L.ego(c))}
    if want_ego:
        p = hf_dl("egomotion/egomotion_alpamayo.tar", hf_dir)
        got = extract_tar(p, want_ego, ".parquet", L.ego)
        log(f"egomotion: {len(got)}/{len(want_ego)} extracted")
    # extrinsics
    ex = pd.read_parquet(hf_dl("calibration/sensor_extrinsics.parquet", hf_dir))
    ex = ex[ex["sensor_name"].astype(str) == FRONT_WIDE]
    have = set()
    if os.path.exists(L.ext_csv):
        have = set(pd.read_csv(L.ext_csv, dtype={"clip_id": str})["clip_id"])
    if not set(clips) <= have:
        allc = sorted(have | set(clips))
        sub = ex[ex["clip_id"].isin(allc)][["clip_id", "qx", "qy", "qz", "qw", "x", "y", "z"]]
        old = pd.read_csv(L.ext_csv, dtype={"clip_id": str}) if os.path.exists(L.ext_csv) else None
        if old is not None:
            sub = pd.concat([old, sub[~sub["clip_id"].isin(old["clip_id"])]])
        sub.to_csv(L.ext_csv, index=False)
    # obstacle rows
    idx = pd.read_parquet(hf_dl("agents/obstacle_offline/index.parquet", hf_dir)).set_index("clip_id")
    has_obs = {}
    need = [c for c in clips if not os.path.exists(L.obs(c))]
    by_shard = {}
    for c in need:
        if c in idx.index and int(idx.loc[c, "rows"]) > 0:
            by_shard.setdefault(str(idx.loc[c, "shard"]), []).append(c)
    for shard, cs in sorted(by_shard.items()):
        p = hf_dl(f"agents/obstacle_offline/{shard}", hf_dir)
        df = pd.read_parquet(p)
        df = df[df["clip_id"].isin(cs)]
        os.makedirs(f"{L.root}/obs", exist_ok=True)
        for c, g in df.groupby("clip_id"):
            g.to_parquet(L.obs(str(c)), index=False)
        log(f"obstacle shard {shard}: {len(cs)} clips wanted, {df['clip_id'].nunique()} written")
    for c in clips:
        has_obs[c] = os.path.exists(L.obs(c))
    return has_obs


def stage_mp4(L, c, want_sha, tries=3):
    t0 = time.time()
    p = L.mp4(c)
    got = sha256_file(p) if os.path.exists(p) else None
    k = 0
    while got != want_sha:
        k += 1
        if k > tries:
            return None, got, time.time() - t0, f"mp4 sha mismatch after {tries} downloads ({str(got)[:12]} != {str(want_sha)[:12]})"
        try:
            hf_dl(f"camera/{c}.mp4", L.cam_root, force_download=(k > 1))
        except Exception as e:                                       # noqa: BLE001
            if k >= tries:
                return None, got, time.time() - t0, f"download {type(e).__name__}: {str(e)[:120]}"
            continue
        got = sha256_file(p) if os.path.exists(p) else None
    return os.path.getsize(p), got, time.time() - t0, None


def ledger_ok(L):
    ok = {}
    if os.path.exists(L.ledger):
        for ln in open(L.ledger, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    d = json.loads(ln)
                    ok[d["clip"]] = d
                except Exception:                                    # noqa: BLE001
                    pass
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True, help="file of clip ids (a leading position column or a trailing role column is ignored)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hf-dir", default="/workspace/v8proof/cropread/hfdl")
    ap.add_argument("--kind", choices=["eval", "train"], required=True)
    ap.add_argument("--first", type=int, default=0, help="stage only the first N ids of the list (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    a = ap.parse_args(argv)
    L = Layout(a.root)
    os.makedirs(L.root, exist_ok=True)
    os.makedirs(a.hf_dir, exist_ok=True)
    clips = read_ids(a.list)
    if a.first:
        clips = clips[:a.first]
    free = shutil.disk_usage(a.root).free / 1e9
    if free < a.min_free_gb:
        log(f"STOP: disk free {free:.1f} GB < {a.min_free_gb}")
        return 7
    done = ledger_ok(L)
    todo = [c for c in clips if not (done.get(c, {}).get("ok") and os.path.exists(L.mp4(c))
                                     and os.path.getsize(L.mp4(c)) == done[c].get("mp4_bytes"))]
    log(f"{a.kind}: {len(clips)} clips, {len(clips) - len(todo)} already staged, {len(todo)} to do")
    hf_sha = mp4_ledger(a.hf_dir)
    has_obs = stage_shared(L, a.hf_dir, clips, need_obs=(a.kind == "train"))

    def one(c):
        want = hf_sha(c)
        row = {"clip": c, "kind": a.kind, "t": time.strftime("%F %T"), "has_obs": bool(has_obs.get(c)), "ok": False}
        if want is None:
            row["reason"] = "no HF sha256 for the mp4"
            return row
        nb, got, secs, err = stage_mp4(L, c, want)
        row.update(dl_s=round(secs, 2), mp4_bytes=nb, mp4_sha256=got)
        if err:
            row["reason"] = err
            return row
        miss = [n for n, f in (("timestamps", L.ts(c)), ("egomotion", L.ego(c))) if not os.path.exists(f)]
        if miss:
            row["reason"] = "missing " + "+".join(miss)
            return row
        if a.kind == "train" and not row["has_obs"]:
            row["reason"] = "no obstacle source for this clip"
            return row
        row["ok"] = True
        return row

    n_ok = n_bad = 0
    t0 = time.time()
    with open(L.ledger, "a", buffering=1, encoding="utf-8") as lf, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(one, todo):
            lf.write(json.dumps(row) + "\n")
            if row["ok"]:
                n_ok += 1
            else:
                n_bad += 1
                log(f"NOT STAGED {row['clip'][:8]}: {row.get('reason')}")
            if (n_ok + n_bad) % 25 == 0:
                log(f"progress {n_ok + n_bad}/{len(todo)} ok={n_ok} bad={n_bad} {(time.time() - t0) / (n_ok + n_bad):.1f} s/clip wall")
    secs = time.time() - t0
    log(f"done: ok={n_ok} not_staged={n_bad} wall={secs:.0f}s workers={a.workers}")
    print("ZZ STAGE DONE ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
