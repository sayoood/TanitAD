"""q5_sample.py -- S2 / S3 of AMENDMENT 7 through s0p_extract_train's own sampler (`draw_sample`, unedited), plus the C9 counts.

S1 seed 20261111 (sha256 843e4017...), S2 seed 20261115, S3 seed 20261116 (built only on SAMPLES-DISAGREE).  The draw is the dry run of s0p_extract_train.py (CPU, no cache read):
    python q5_sample.py --seed 20261115 --tree <repo root> --join <join_train.npz> --tracks <native_tracks.pt> --out-dir <dir> [--s0p-dir <dir with s0p_extract_train.py>]
"""
import argparse
import json
import os
import subprocess
import sys

SEEDS = {"S1": 20261111, "S2": 20261115, "S3": 20261116}
S1_SHA = "843e4017de3c40ae91618ac878cc2f361c798a62633eb9fa6805b457b4c1609e"


def draw(seed, tree, join, tracks, out_dir, s0p_dir, python=sys.executable):
    cmd = [python, os.path.join(s0p_dir, "s0p_extract_train.py"), "--tree", tree, "--cache-dir", os.path.dirname(tracks), "--join", join, "--tracks", tracks,
           "--out-dir", out_dir, "--seed", str(seed), "--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    if r.returncode != 0:
        raise RuntimeError(f"sample draw rc {r.returncode}: {r.stderr[-300:]}")
    s = json.load(open(os.path.join(out_dir, "sample.json"), encoding="utf-8"))
    if seed == SEEDS["S1"]:
        assert s["sha256"] == S1_SHA, f"S1 sample sha {s['sha256']} != {S1_SHA}"
    return s


def counts(sample):
    return {"n": len(sample["y"]), "n_pos": int(sum(sample["y"])), "n_clips": len(set(sample["clip"])), "sha256": sample["sha256"]}


def overlap(a, b):
    ka, kb = set(zip(a["clip"], a["ws"])), set(zip(b["clip"], b["ws"]))
    return {"windows": len(ka & kb), "clips": len(set(a["clip"]) & set(b["clip"])), "clip_union": len(set(a["clip"]) | set(b["clip"]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--join", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--s0p-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "s0p"))
    a = ap.parse_args()
    print(json.dumps(counts(draw(a.seed, a.tree, a.join, a.tracks, a.out_dir, a.s0p_dir)), indent=1))
