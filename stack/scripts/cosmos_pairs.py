"""Extract matched weather PAIRS from a Cosmos-Drive-Dreams generation shard.

The SC-05/D8 redesign (2026-07-08) needs the SAME scene rendered under clear
and degraded weather to isolate the weather axis from scene content. The first
60 members of shard part-000 were 60 distinct scenes, so pairs require a full
scan: pass 1 streams the tar index only and groups members by (base_id, chunk);
pass 2 extracts one clear + one degraded variant for the selected groups and
fetches their vehicle_pose tars.

Runs on the pod (CPU-only, streaming — no RAM pressure next to the trainer):
  python scripts/cosmos_pairs.py --shard .../generation.tar.gz.part-000 \
      --out /workspace/data/cosmos/pairs --n-pairs 24
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

CLEAR = {"sunny", "morning", "golden_hour"}
DEGRADED = {"foggy", "rainy", "snowy", "night"}
WEATHER_RE = re.compile(
    r"_(Foggy|Golden_hour|Morning|Night|Rainy|Snowy|Sunny)$", re.IGNORECASE)


def parse_member(name: str):
    """`.../<uuid>_<t0>_<t1>_<chunk>_<Weather>.mp4` -> (base, chunk, weather)."""
    stem = Path(name).stem
    m = WEATHER_RE.search(stem)
    if not m:
        return None
    weather = m.group(1).lower()
    rest = stem[:m.start()]
    mc = re.search(r"_(\d+)$", rest)
    if not mc:
        return None
    return rest[:mc.start()], int(mc.group(1)), weather


def stream(shard: str):
    """Iterate members of a split-gzip shard, tolerating the truncated tail."""
    try:
        with tarfile.open(fileobj=gzip.open(shard, "rb"), mode="r|") as tf:
            yield from ((m, tf) for m in tf)
    except (EOFError, tarfile.ReadError):
        return


def scan_index(shard: str) -> dict:
    groups: dict = {}
    n = 0
    for m, _ in stream(shard):
        if not (m.isfile() and m.name.endswith(".mp4")):
            continue
        n += 1
        p = parse_member(m.name)
        if p:
            groups.setdefault((p[0], p[1]), {})[p[2]] = m.name
        if n % 2000 == 0:
            print(f"[pairs] indexed {n} mp4s, {len(groups)} groups", flush=True)
    print(f"[pairs] pass1 done: {n} mp4s, {len(groups)} (base,chunk) groups")
    return groups


def select_pairs(groups: dict, n_pairs: int) -> dict[str, tuple]:
    """Pick groups having >=1 clear and >=1 degraded variant; prefer richer."""
    cands = []
    for key, wmap in groups.items():
        cl = [w for w in wmap if w in CLEAR]
        dg = [w for w in wmap if w in DEGRADED]
        if cl and dg:
            cands.append((len(wmap), key, wmap[cl[0]], wmap[dg[0]]))
    cands.sort(reverse=True)
    wanted = {}
    for _, key, clear_name, deg_name in cands[:n_pairs]:
        wanted[clear_name] = key
        wanted[deg_name] = key
    print(f"[pairs] selected {len(wanted) // 2} pairs "
          f"of {len(cands)} candidates")
    return wanted


def extract(shard: str, wanted: dict, out: Path) -> int:
    got = 0
    vid_dir = out / "generation"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for m, tf in stream(shard):
        if m.name in wanted and m.isfile():
            m.name = Path(m.name).name                  # flatten
            tf.extract(m, vid_dir, filter="data")
            got += 1
            if got == len(wanted):
                break
    print(f"[pairs] pass2 extracted {got}/{len(wanted)} mp4s")
    return got


def fetch_poses(out: Path) -> int:
    from huggingface_hub import hf_hub_download
    repo = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"
    pose_root = out / "vehicle_pose"
    pose_root.mkdir(parents=True, exist_ok=True)
    base_ids = sorted({"_".join(p.stem.split("_")[:3])
                       for p in (out / "generation").glob("*.mp4")})
    ok = 0
    for cid in base_ids:
        if (pose_root / cid).is_dir():
            ok += 1
            continue
        try:
            t = hf_hub_download(repo, f"vehicle_pose/{cid}.tar",
                                repo_type="dataset", local_dir=str(out))
            with tarfile.open(t) as pf:
                pf.extractall(pose_root / cid, filter="data")
            ok += 1
        except Exception as e:
            print(f"[pairs] pose {cid}: {type(e).__name__}")
    print(f"[pairs] poses: {ok}/{len(base_ids)} base ids")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)

    groups = scan_index(args.shard)
    wanted = select_pairs(groups, args.n_pairs)
    if not wanted:
        raise SystemExit("[pairs] no clear+degraded groups in this shard slice")
    extract(args.shard, wanted, out)
    fetch_poses(out)
    manifest = {}
    for name, (base, chunk) in wanted.items():
        manifest.setdefault(f"{base}_{chunk}", []).append(Path(name).name)
    (out / "pairs_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[pairs] manifest -> {out / 'pairs_manifest.json'}")


if __name__ == "__main__":
    main()
