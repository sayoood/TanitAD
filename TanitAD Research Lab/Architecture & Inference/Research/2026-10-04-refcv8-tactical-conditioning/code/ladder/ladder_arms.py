"""SPEC_WPB_LADDER (registered, sha256 e1b6aff7...) + amendment A1 (cbd64fa2...) -- the ARM TABLE, the argv builder and
the one-variable audit. Every arm's argv is DERIVED here from refcv7-r101-s0's canonical argv (the launch tree's
`stack/ops/runs.d/refcv7-r101-s0.argv.json`), never typed by hand, and audited against its base BEFORE it runs.

Rig (SPEC sec. 1): `--size tiny`, resnet34 ImageNet (`--trunk-name resnet34.a1_in1k --trunk-in-channels 9`), `--batch 8
--lr 1e-4 --warmup 200 --log-every 50 --eval-every N --eval-batches 64`, the same N for every arm. Execution-only tokens
the rig also sets (identical on every arm, so no audit can see them): `--trunk-compile` dropped (a CPU-side compile
cost per chunk), `--workers 2` (Thor host RAM: the 2026-10-04 global OOM), `--save-every 250` (the chunked resume).

Speed (SPEC sec. 1, MM ruling Q5): EVERY arm, V0 included, feeds the past-only N2 with the trained unknown row; refcv7's
v8 future-max sidecars are REMOVED from every arm. X10 (MM ruling 2026-10-04): `--pose-sync-sidecar` on every arm.
`--grad-share-every 250` is an instrument (no gradient, no RNG) and is carried by every arm so V0 has readings too.

Usage:  python ladder_arms.py --tree <tree> --out <W> [--n N] [--k K]   -> <W>/argv/<arm>.json + <W>/argv/AUDIT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

V9 = "/home/nvidia/refcv8_v9labels/release"
POSE_SYNC = "/home/nvidia/data/refcv6_pose_sync_sidecar.jsonl"

#: the tokens the RIG sets (removed from refcv7's argv, re-added with the rig's values) -- SPEC sec. 1
RIG_FLAGS = ("--size", "--trunk-name", "--trunk-in-channels", "--batch", "--lr", "--warmup", "--log-every",
             "--eval-every", "--eval-batches", "--steps", "--save-every", "--workers", "--out", "--seed",
             "--trunk-compile")
#: refcv7's speed source -- REFUSED on refcv8 / every ladder arm (X3)
REMOVED = ("--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval")
#: the speed-only set a V0 argv may carry (MM I-0 assertion 1: nothing else refcv8-live)
SPEED_ONLY_SET = ("--r8-speed-input", "--r8-speed-unknown-p", "--r8-v9-labels", "--r8-v9-labels-eval",
                  "--r8-v9-md5", "--r8-v9-eval-md5")

#: L1's one variable: the refcv8 recipe bundle (SPEC sec. 2 V-R8 + MM ruling Q2 + the 693-box mask)
R8_BUNDLE = [("--refcv8", []), ("--r8-nav-from-v9", []), ("--r8-rc-variant", ["A50"]),
             ("--r8-rc-noise-along-m", ["2.0"]), ("--r8-rc-noise-lat-m", ["0.75"]), ("--r8-rc-dropout", ["0.3"]),
             ("--r8-nav-args-dropout", ["0.5"]), ("--w-r8-cons", ["0.05"]), ("--r8-n-alloc", ["32"]),
             ("--r8-alloc-emit", []), ("--w-r8-alloc-l1", ["1.0"]), ("--w-r8-listwise", ["1.0"]),
             ("--w-r8-v9-cons", ["0.05"]),
             ("--join-defect-masks", ["/home/nvidia/refcv8_ladder/tree/stack/tanitad/configs/"
                                      "refcv8_join_label_defects.json"])]


def pairs(argv):
    out, i = [], 0
    while i < len(argv):
        f, j = argv[i], i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        out.append((f, list(argv[i + 1:j])))
        i = j
    return out


def flat(ps):
    return [t for f, v in ps for t in [f] + list(v)]


def setf(ps, f, v):
    ps = [(a, b) for a, b in ps if a != f]
    return ps + [(f, list(v))]


def drop(ps, *fs):
    return [(a, b) for a, b in ps if a not in fs]


def base_v0(r7, n: int, out_root: str, seed: int = 0):
    ps = drop(pairs(r7), *RIG_FLAGS, *REMOVED)
    rig = [("--size", ["tiny"]), ("--trunk-name", ["resnet34.a1_in1k"]), ("--trunk-in-channels", ["9"]),
           ("--batch", ["8"]), ("--lr", ["1e-4"]), ("--warmup", ["200"]), ("--log-every", ["50"]),
           ("--eval-every", [str(n)]), ("--eval-batches", ["64"]), ("--steps", [str(n)]), ("--save-every", ["250"]),
           ("--workers", ["2"]), ("--seed", [str(seed)]),
           ("--r8-speed-input", ["n2"]), ("--r8-speed-unknown-p", ["0.45"]),
           ("--r8-v9-labels", [f"{V9}/v9_labels_train.npz"]), ("--r8-v9-labels-eval", [f"{V9}/v9_labels_eval139.npz"]),
           ("--pose-sync-sidecar", [POSE_SYNC]), ("--grad-share-every", ["250"])]
    return ps + rig


def arms(r7, n: int, k: str | None, out_root: str) -> dict:
    """{arm: (argv pairs, base arm, the declared delta [(flag, value)], role)}; `--out` is set per arm."""
    v0 = base_v0(r7, n, out_root)
    vr8 = v0 + R8_BUNDLE
    A = {}

    def add(name, ps, base, delta, role):
        A[name] = (setf(ps, "--out", [f"{out_root}/{name}/run"]), base, delta, role)
    add("V0", v0, None, [], "L1 base")
    add("V-R8", vr8, "V0", R8_BUNDLE, "L1 treatment; L2/L3/L4/L4b/L5 base")
    add("V-R8d", vr8 + [("--r8-derange-feed", []), ("--r8-rc-roll", [])], "V-R8",
        [("--r8-derange-feed", []), ("--r8-rc-roll", [])], "L1 deliberate regression")
    add("V0r", setf(v0, "--seed", ["1"]), "V0", [("--seed", ["1"])], "L1 replicate")
    add("V-R8r", setf(vr8, "--seed", ["1"]), "V-R8", [("--seed", ["1"])], "L1 replicate")
    if k is not None:
        vtac = setf(vr8, "--w-tac-v6", [k])
        add("V-TACk", vtac, "V-R8", [("--w-tac-v6", [k])], "L2 treatment")
        add("V-TACk-roll", vtac + [("--r8-roll-targets", ["tac"])], "V-TACk", [("--r8-roll-targets", ["tac"])],
            "L2 deliberate regression")
        add("V-TACkr", setf(vtac, "--seed", ["1"]), "V-TACk", [("--seed", ["1"])], "L2 replicate")
    vmap = setf(vr8, "--w-map-hires", ["4.0"])
    add("V-MAP4", vmap, "V-R8", [("--w-map-hires", ["4.0"])], "L3 treatment")
    add("V-MAP4-roll", vmap + [("--r8-roll-targets", ["map"])], "V-MAP4", [("--r8-roll-targets", ["map"])],
        "L3 deliberate regression")
    add("V-MAP4r", setf(vmap, "--seed", ["1"]), "V-MAP4", [("--seed", ["1"])], "L3 replicate")
    add("V-VSHUF", vr8 + [("--r8-roll-speed-input", [])], "V-R8", [("--r8-roll-speed-input", [])],
        "L4 deliberate regression")
    # amendment A1
    ve8 = vr8 + [("--r8-speed-enc8", [])]
    add("V-R8-E8", ve8, "V-R8", [("--r8-speed-enc8", [])], "L4b treatment")
    add("V-R8-E8-roll", ve8 + [("--r8-roll-speed-input", [])], "V-R8-E8", [("--r8-roll-speed-input", [])],
        "L4b deliberate regression")
    vdrv = vr8 + [("--r8-critic-drivable", []), ("--w-r8-drivable", ["1.0"])]
    add("V-R8-DRV", vdrv, "V-R8", [("--r8-critic-drivable", []), ("--w-r8-drivable", ["1.0"])], "L5 treatment")
    add("V-R8-DRV-roll", vdrv + [("--r8-roll-targets", ["map"])], "V-R8-DRV", [("--r8-roll-targets", ["map"])],
        "L5 deliberate regression")
    return A


#: run order = priority order (SPEC sec. 2, then A1); V-TACk* need k (sized on V-R8's first 20 % of readings)
ORDER = ("V0", "V-R8", "V-R8d", "V0r", "V-R8r", "V-TACk", "V-TACk-roll", "V-TACkr", "V-MAP4", "V-MAP4-roll",
         "V-MAP4r", "V-VSHUF", "V-R8-E8", "V-R8-E8-roll", "V-R8-DRV", "V-R8-DRV-roll")


def audit(A: dict) -> dict:
    """The ONE-VARIABLE audit on the ACTUAL argv (SPEC sec. 1): arm minus base must be exactly the declared delta,
    ignoring `--out`; plus MM I-0 assertion 1 (V0 carries no live refcv8 flag beyond the speed-only set)."""
    rep, ok = {}, True
    for name, (ps, base, delta, _role) in A.items():
        if base is None:
            continue
        a = {f: v for f, v in ps if f != "--out"}
        b = {f: v for f, v in A[base][0] if f != "--out"}
        got = sorted([(f, v) for f, v in a.items() if b.get(f, None) != v])
        gone = sorted(f for f in b if f not in a)
        want = sorted((f, list(v)) for f, v in delta)
        good = (got == want and not gone)
        ok &= good
        rep[name] = {"base": base, "PASS": good, "delta_found": got, "removed": gone, "delta_declared": want}
    v0 = dict(A["V0"][0])
    live = [f for f in v0 if (f.startswith("--r8-") or f.startswith("--w-r8-") or f == "--refcv8")
            and f not in SPEED_ONLY_SET]
    rep["I0_assert1_V0_speed_only"] = {"PASS": not live, "live_refcv8_flags": live}
    ok &= not live
    # X10 parity (MM 2026-10-05): EVERY arm, V0 included, trains on the corrected clock with the SAME sidecar -- a
    # common token, so no L-rung reads the clock as a second variable
    ps_vals = {n_: dict(ps).get("--pose-sync-sidecar") for n_, (ps, *_r) in A.items()}
    ps_ok = all(v == [POSE_SYNC] for v in ps_vals.values())
    rep["pose_sync_on_every_arm"] = {"PASS": ps_ok, "sidecar": POSE_SYNC,
                                     "arms_without": sorted(k for k, v in ps_vals.items() if v != [POSE_SYNC])}
    ok &= ps_ok
    stale = [n for n, (ps, *_r) in A.items() if any(f in REMOVED for f, _v in ps)]
    rep["no_v8_sidecar_anywhere"] = {"PASS": not stale, "arms": stale}
    ok &= not stale
    rep["PASS"] = bool(ok)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", default=None)
    ap.add_argument("--root", default=None, help="run-dir root (default: <out>/arms)")
    a = ap.parse_args()
    r7 = json.loads((Path(a.tree) / "stack/ops/runs.d/refcv7-r101-s0.argv.json").read_text(encoding="utf-8"))["argv"]
    root = a.root or f"{a.out}/arms"
    A = arms(r7, a.n, a.k, root)
    rep = audit(A)
    d = Path(a.out) / "argv"
    d.mkdir(parents=True, exist_ok=True)
    for name, (ps, base, delta, role) in A.items():
        argv = flat(ps)
        rec = {"arm": name, "base": base, "role": role, "n_steps": a.n, "k": a.k, "argv": argv,
               "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest()}
        (d / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (d / "AUDIT.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v["PASS"] for k, v in rep.items() if isinstance(v, dict)}))
    print("AUDIT", "PASS" if rep["PASS"] else "FAIL")
    return 0 if rep["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
