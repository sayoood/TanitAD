#!/usr/bin/env python3
"""Writes the A2 FLOOR heads: the never-trained aux drivable head of seeds 0, 1, 2 (PREREG_AUX_DRIVABLE_S1.md, Amendment A2.1 / A2.4), in the trainer's `aux_drv/1` file format,
to  D:/Projects/TanitAD-artifacts/aux_navtest_forward/init_heads/seed<k>/aux_drv.pt   (feed off; see below).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe make_init_heads.py [--out <dir>] [--fixture <aux_drv.pt>]

What it ASSERTS (it exits non-zero on any failure, and prints ZZINITHEADS_OK only when all hold):
  1. FEED OFF AND FEED ON SHARE ONE INITIAL HEAD PER SEED.  `AuxDrivable(cfg(feed="off", seed=k)).head` and `AuxDrivable(cfg(feed="on", seed=k)).head` have bit-identical
     state_dicts (the head is built from its own fork_rng stream `manual_seed(cfg.seed)`; the feed module has its own fork, seed + 1), so ONE file serves an H1 arm and an H2 arm of
     the same seed.  (A2.1: "feed off and feed on share one initial head per seed".)
  2. THE FILE ROUND-TRIPS: `aux_navtest_forward.load_aux_head` (strict load, the loader the forward tool uses) rebuilds a head whose state_dict is bit-identical to the one built here.
  3. THE SEEDS DIFFER (0 vs 1 vs 2): three distinct heads, not one head written three times.
  4. THE SEED-0 FILE EQUALS THE EXISTING FIXTURE `fixture_fresh_head_seed0/aux_drv.pt` bit-exactly (head tensors, config, store pin), i.e. the head whose 32-token plumbing read
     (P3 pooled AUC 0.8245) triggered A2 is exactly the seed-0 floor.
Nothing here reads a trunk, a GPU or the C3 cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PKG, "eval"))
sys.path.insert(0, os.path.join(PKG, "refe"))
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402
import aux_navtest_forward as FWD  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts/aux_navtest_forward"
OUT = f"{ART}/init_heads"
FIXTURE = f"{ART}/fixture_fresh_head_seed0/aux_drv.pt"
SEEDS = (0, 1, 2)


def cfg(seed: int, feed: str) -> "AUX.AuxDrivableConfig":
    return AUX.AuxDrivableConfig(tap="visual", depth=2, feed=feed, w_aux=0.0, seed=seed)       # make_fresh_head_file's own config (A2.1: tap visual, depth 2)


def same_state(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fixture", default=FIXTURE)
    a = ap.parse_args(argv)
    heads, manifest = {}, {}
    for seed in SEEDS:
        off, on = AUX.AuxDrivable(cfg(seed, "off")), AUX.AuxDrivable(cfg(seed, "on"))
        assert off.feed is None and on.feed is not None
        assert same_state(off.head.state_dict(), on.head.state_dict()), f"seed {seed}: feed off and feed on built DIFFERENT heads"
        path = FWD.make_fresh_head_file(os.path.join(a.out, f"seed{seed}", "aux_drv.pt"), seed=seed, depth=2, feed="off")
        head, info = FWD.load_aux_head(path, "cpu")
        assert same_state(head.state_dict(), off.head.state_dict()), f"seed {seed}: the written file does not round-trip to the built head"
        heads[seed] = off.head.state_dict()
        manifest[f"seed{seed}"] = {"path": os.path.abspath(path), "sha256": sha256(path), "update": info["update"], "n_params": info["n_params"], "config": info["config"],
                                   "feed_on_head_bit_identical": True}
        print(f"seed {seed}: {path}  sha256 {manifest[f'seed{seed}']['sha256'][:16]}  params {info['n_params']}  feed-on head bit-identical: True", flush=True)
    assert not same_state(heads[0], heads[1]) and not same_state(heads[0], heads[2]) and not same_state(heads[1], heads[2]), "two seeds produced the same head"
    fx = torch.load(a.fixture, map_location="cpu", weights_only=False)
    mine = torch.load(manifest["seed0"]["path"], map_location="cpu", weights_only=False)
    assert same_state(fx["head"], mine["head"]), "seed-0 head != the existing fixture's head"
    assert fx["meta"]["config"] == mine["meta"]["config"] and fx["meta"]["store_sha256"] == mine["meta"]["store_sha256"] and fx["format"] == mine["format"] and fx["update"] == mine["update"]
    manifest["seed0_equals_fixture"] = {"fixture": os.path.abspath(a.fixture), "fixture_sha256": sha256(a.fixture), "head_bit_identical": True, "config_identical": True}
    print(f"seed-0 head == fixture head, bit-exact ({a.fixture})", flush=True)
    with open(os.path.join(a.out, "init_heads_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("ZZINITHEADS_OK 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
