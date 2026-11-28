#!/usr/bin/env python3
"""The dev-box PROXY SUBSET: 10,000 live-bank frames, one manifest shared by the decoder-side proxy (M3/M4a/M4b)
and the scorer-only fine-tune (measure 5). CPU only; reads local files; writes one JSON.

POOL. The live run trains on the pod's `/workspace/data/refe_navtrain/train_grow`. Its dev-box-built part sits on
this box in `D:/Projects/TanitAD/data/refe_navtrain10` -- `r0/targets_rank0.jsonl` (rank-0 teacher targets, 10 Hz)
and `aug/targets_aug.jsonl` (rank-1 goal-augmented twins) -- and was pushed to the pod as the `DEV10` shards
(`code/devbox_upload.py`; `.upload_state_DEV10*.json` record the bytes sent). The fixed-candidate scorer labels for
the same scenes are local too (`sc_r0_s*/scorer_targets.jsonl`, `sc_aug_s*/scorer_targets_rank1.jsonl`).
⚠️ INHERITED until the pod index says so: that every pool scene is IN train_grow (grow_assemble may have rejected a
row). `finalize` checks it against the pod index (proxy_pod_extract.py, phase A) and replaces absent picks.

SELECTION (pre-registered before any pod data is seen). Stratified by drive log, proportional: each log gets
round(n_log x N / pool) frames by largest remainder, chosen by a per-log seeded shuffle (seed 20260927); `finalize`
replaces a dropped pick with the next frame of the SAME log's shuffle. A frame carries all of its ranks (0, and 1
when the scene has an augmented twin). On-policy coverage is RECORDED per (frame, rank), never selected on: the
proxy's on-policy arm then sees the live bank's own coverage density for this population.

    python refe/proxy_manifest.py candidates [--n 10000]                    # provisional (no pod data)
    python refe/proxy_manifest.py finalize --index <pod_index.tar>          # after phase A
Writes raw/2026-09-27-training-measures/proxy_manifest_10k.json. ZZMANIFEST_OK <counts>.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
BANK = Path("D:/Projects/TanitAD/data/refe_navtrain10")
OUT = PKG / "raw" / "2026-09-27-training-measures" / "proxy_manifest_10k.json"
SEED = 20260927
JPEG_KB_MEASURED = 202.0          # mean over 48,584 local OpenScene navtest camera JPEGs (4 cameras), 2026-09-27
SCENE_CTX_BYTES = 64 * 256 * 2    # [64 = 4 cams x 16 registers, 256] bf16 -- the trajectory decoder's input
VISUAL_CTX_BYTES = 7680 * 256 * 2  # [4 cams x 1,920 patches, 256] bf16 -- the scoring decoder's input


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(p) -> tuple:
    """complete lines only; (rows, n_unparseable)"""
    with open(p, "rb") as f:
        buf = f.read()
    cut = buf.rfind(b"\n")
    rows, bad = [], 0
    for line in buf[:cut + 1].decode("utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def fkey(r) -> tuple:
    return (r["log_name"], r["token"], int(r.get("step", 0)))


def valid_target(r) -> bool:
    try:
        return (isinstance(r.get("image"), list) and len(r["image"]) == 4 and len(r["ego"]) == 7
                and len(r["traj"]) == 20 and len(r["traj"][0]) == 3 and r.get("log_name") and r.get("token"))
    except Exception:
        return False


def fixed_label_keys(pattern: str) -> set:
    """(log, token, step, rank) of every fixed-candidate scorer row -- keys only."""
    keys = set()
    for p in sorted(glob.glob(str(BANK / pattern))):
        with open(p, "rb") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))))
                except Exception:
                    continue
    return keys


def build_pool():
    r0, bad0 = read_rows(BANK / "r0" / "targets_rank0.jsonl")
    r1, bad1 = read_rows(BANK / "aug" / "targets_aug.jsonl")
    pool, rej = {}, 0
    for r in r0:
        if int(r.get("rank", 0)) != 0 or not valid_target(r):
            rej += 1
            continue
        pool.setdefault(fkey(r), {"r0": r, "r1": None})
    n1 = 0
    for r in r1:
        k = fkey(r)
        if k in pool and int(r.get("rank", 1)) == 1 and valid_target(r):
            pool[k]["r1"] = r
            n1 += 1
    fx0 = fixed_label_keys("sc_r0_s*/scorer_targets.jsonl")
    fx1 = fixed_label_keys("sc_aug_s*/scorer_targets_rank1.jsonl")
    src = {"rank0": {"path": str(BANK / "r0" / "targets_rank0.jsonl"), "rows": len(r0), "unparseable": bad0,
                     "sha256": sha256(BANK / "r0" / "targets_rank0.jsonl")},
           "rank1": {"path": str(BANK / "aug" / "targets_aug.jsonl"), "rows": len(r1), "unparseable": bad1,
                     "sha256": sha256(BANK / "aug" / "targets_aug.jsonl")},
           "fixed_label_keys": {"rank0": len(fx0), "rank1": len(fx1)}}
    return pool, fx0, fx1, src, {"rejected_rank0_rows": rej, "rank1_attached": n1}


def ranking(pool: dict) -> dict:
    """{log: [frame keys in that log's seeded order]} -- the whole pre-registered preference order."""
    by = {}
    for k in sorted(pool):
        by.setdefault(k[0], []).append(k)
    for lg, ks in by.items():
        random.Random(f"{SEED}|{lg}").shuffle(ks)
    return by


def allocate(by: dict, n: int) -> dict:
    """largest-remainder proportional quota per log (deterministic tie-break by log name)"""
    tot = sum(len(v) for v in by.values())
    raw = {lg: len(v) * n / tot for lg, v in by.items()}
    q = {lg: int(x) for lg, x in raw.items()}
    rest = n - sum(q.values())
    for lg in sorted(raw, key=lambda g: (-(raw[g] - q[g]), g))[:rest]:
        q[lg] += 1
    return q


def frame_entry(k, e, fx0, fx1) -> dict:
    ranks = {"0": {"fixed_labels": (k[0], k[1], k[2], 0) in fx0}}
    if e["r1"] is not None:
        ranks["1"] = {"fixed_labels": (k[0], k[1], k[2], 1) in fx1}
    return {"log_name": k[0], "token": k[1], "step": k[2], "images": list(e["r0"]["image"]),
            "cameras": e["r0"].get("cameras"), "ranks": ranks}


def select(pool, fx0, fx1, n, present=None) -> tuple:
    by = ranking(pool)
    q = allocate(by, n)
    frames, short = [], 0
    for lg in sorted(by):
        take = [k for k in by[lg] if present is None or k in present][:q[lg]]
        short += q[lg] - len(take)
        frames += [frame_entry(k, pool[k], fx0, fx1) for k in take]
    if short and present is not None:                   # a log without enough present frames: fill from the
        extra = [k for lg in sorted(by) for k in by[lg][q[lg]:] if k in present]   # remaining order, round-robin
        have = {(f["log_name"], f["token"], f["step"]) for f in frames}
        frames += [frame_entry(k, pool[k], fx0, fx1) for k in extra if k not in have][:short]
    return frames, q


def estimates(frames) -> dict:
    n = len(frames)
    imgs = {os.path.basename(i) for f in frames for i in f["images"]}
    return {"frames": n, "rows": sum(len(f["ranks"]) for f in frames), "jpegs": 4 * n,
            "jpeg_basenames_unique": len(imgs) == 4 * n,
            "jpeg_gb_est": round(4 * n * JPEG_KB_MEASURED * 1024 / 1e9, 2),
            "cache_scene_ctx_gb": round(n * SCENE_CTX_BYTES / 1e9, 3),
            "cache_visual_ctx_gb": round(n * VISUAL_CTX_BYTES / 1e9, 2),
            "basis": f"JPEG mean {JPEG_KB_MEASURED} KB MEASURED on 48,584 local OpenScene navtest frames; cache bf16"}


def load_index(tar_path) -> dict:
    """phase-A tar (proxy_pod_extract.py emit-index) -> {train_grow keys, set lines by key}"""
    out = {"train_grow": set(), "sets": {}, "done": None}
    with tarfile.open(tar_path, "r|") as t:
        for m in t:
            if not m.isfile():
                continue
            data = t.extractfile(m).read().decode("utf-8")
            if m.name.endswith("train_grow_keys.tsv"):
                for line in data.splitlines()[1:]:
                    f_, off, nb, lg, tok, st, rk = line.split("\t")
                    out["train_grow"].add((lg, tok, int(st), int(rk)))
            elif m.name.endswith("sets_index.tsv"):
                for line in data.splitlines()[1:]:
                    f_, off, nb, lg, tok, st, rk, ck, lv, ok = line.split("\t")
                    if ok == "1":
                        out["sets"].setdefault((lg, tok, int(st), int(rk)), []).append(
                            {"file": f_, "offset": int(off), "nbytes": int(nb), "ckpt_step": int(ck),
                             "label_version": int(lv)})
            elif m.name == "index/DONE.json":        # the name proxy_pod_extract.py's pod script writes, LAST
                out["done"] = json.loads(data)
    if out["done"] is None:
        raise SystemExit(f"{tar_path}: no index/DONE.json -- a TRUNCATED index is not an index")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("candidates", "finalize"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--index", default=None, help="phase-A tar from the pod (finalize)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    pool, fx0, fx1, src, stats = build_pool()
    present = None
    cov = None
    if a.stage == "finalize":
        if not a.index:
            print("  REFUSING: finalize needs --index <pod_index.tar>")
            return 4
        idx = load_index(a.index)
        present = {k for k in pool if (k[0], k[1], k[2], 0) in idx["train_grow"]}
    frames, q = select(pool, fx0, fx1, a.n, present)
    if a.stage == "finalize":
        n_cov = {"0": 0, "1": 0}
        lines = 0
        nbytes = 0
        for f in frames:
            for rk, info in f["ranks"].items():
                sets = idx["sets"].get((f["log_name"], f["token"], f["step"], int(rk)), [])
                info["onpolicy"] = [[s["ckpt_step"], s["label_version"]] for s in sets]
                n_cov[rk] += bool(sets)
                lines += len(sets)
                nbytes += sum(s["nbytes"] for s in sets)
        cov = {"frames_rank0_with_a_set": n_cov["0"], "rank1_rows_with_a_set": n_cov["1"],
               "set_lines": lines, "set_bytes": nbytes, "pool_present_in_train_grow": len(present),
               "index_done": idx["done"]}
    man = {"version": 1, "status": "final" if a.stage == "finalize" else "provisional (pod index not yet applied)",
           "seed": SEED, "n_requested": a.n, "rule": __doc__.split("SELECTION")[1].split("python refe")[0].strip(),
           "sources": src, "pool": {"scenes": len(pool), "logs": len({k[0] for k in pool}), **stats},
           "quota_per_log": q, "estimates": estimates(frames), "coverage": cov,
           "frames": frames, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tool_sha256": sha256(__file__)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=None, separators=(",", ":"))
    os.replace(tmp, a.out)
    e = man["estimates"]
    print(f"  pool {len(pool):,} scenes / {man['pool']['logs']} logs; picked {e['frames']:,} frames ({e['rows']:,} "
          f"rows); JPEGs {e['jpegs']:,} ~{e['jpeg_gb_est']} GB; fixed labels rank0 "
          f"{sum(f['ranks']['0']['fixed_labels'] for f in frames):,}/{len(frames):,}; {time.time() - t0:.0f} s")
    if cov:
        print(f"  coverage: {json.dumps({k: v for k, v in cov.items() if k != 'index_done'})}")
    print(f"ZZMANIFEST_OK {man['status'].split()[0]} frames={e['frames']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
