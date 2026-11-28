"""SPEC_WPB_LADDER (registered, sha256 e1b6aff7...) + amendment A1 (cbd64fa2...) -- the ARM TABLE, the argv builder and
the one-variable audit. Every arm's argv is DERIVED here from refcv7-r101-s0's canonical argv (the launch tree's
`stack/ops/runs.d/refcv7-r101-s0.argv.json`), never typed by hand, and audited against its base BEFORE it runs.

Rig (SPEC sec. 1): `--size tiny`, resnet34 ImageNet (`--trunk-name resnet34.a1_in1k --trunk-in-channels 9`), `--batch 8
--lr 1e-4 --warmup 200 --log-every 50 --eval-every N --eval-batches 64`, the same N for every arm. Execution-only tokens
the rig also sets (identical on every arm, so no audit can see them): `--trunk-compile` dropped (a CPU-side compile
cost per chunk), `--workers 2` (Thor host RAM: the 2026-10-04 global OOM), `--save-every 250` (the chunked resume).

Speed -- PI RULING 2026-10-11 (Sayed; supersedes SPEC sec. 1 / MM ruling Q5's past-only N2): "yes use the quantified
label as input, it is not a training signal, it is the maximal wished speed. it is ok that it is matching the future
speed in some cases since it also must be achieved in the future. It is important to assure the road type quantization
and avoid the exact value of the future speed as target speed." => EVERY arm, V0 included, feeds refcv7's 4-way
ROAD-TYPE set-speed sidecar through `--r8-speed-input label` (the two sidecar paths are carried VERBATIM from refcv7's
argv, never typed here) with the trained unknown row; the trainer refuses a sidecar that carries anything but the four
road-type steps. Route -- PI 2026-10-11: "just give the nav command and the distance" => NO route checkpoint on any
refcv8 arm (`--r8-no-rc`); the per-frame nav token (`--r8-nav-from-v9`) and its turn distance stay. L4b (the 8-step N2
encoding) needs the N2 ladder and is therefore not built under the label source (SPEED_SOURCE).
X10 (MM ruling 2026-10-04): `--pose-sync-sidecar` on every arm.
`--grad-share-every 250` is an instrument (no gradient, no RNG) and is carried by every arm so V0 has readings too.
COST (MEASURED 2026-10-11, V0 chunk 1 on Thor): one reading inside the fp32 replay ran > 20 min with every term and
killed V0 at its cap with no checkpoint. Every arm therefore also carries `--grad-share-terms tac_v6,map_hires` (the
only terms a consumer reads; the rest fold into gs_*_rest, ~16 min / reading) and `--grad-share-at` = gs_at(N): ONE
row inside ladder_score's size_k window (0, 0.2 N] and TWO inside the L2/L3 window (0.5 N, N] -- 500,2000,2500 at
N = 3000, i.e. ~48 min per arm instead of ~5 h (every term, every 250th row; SPEC A2). Common tokens on every arm, so no audit sees them.

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
#: PI 2026-10-11: the source of the 4-way set-speed channel on EVERY arm = refcv7's road-type sidecar ("label"); the
#: past-only N2 of the earlier ruling is "n2" (then the sidecars are REMOVED and the E8 arms are built again)
SPEED_SOURCE = "label"
#: refcv7's road-type sidecar flags: carried from refcv7's argv under "label"
ROAD_TYPE_SIDECARS = ("--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval")
#: removed from refcv7's argv: only under the past-only source (REFUSED there, X3)
REMOVED = () if SPEED_SOURCE == "label" else ROAD_TYPE_SIDECARS
#: the 8-step N2 encoding arms (amendment A1): `--r8-speed-enc8` needs `--r8-speed-input n2` (the trainer refuses it otherwise)
E8_ARMS = ("V-R8-E8", "V-R8-E8-roll")
#: the speed-only set a V0 argv may carry (MM I-0 assertion 1: nothing else refcv8-live)
SPEED_ONLY_SET = ("--r8-speed-input", "--r8-speed-unknown-p", "--r8-v9-labels", "--r8-v9-labels-eval",
                  "--r8-v9-md5", "--r8-v9-eval-md5")

#: L1's one variable: the refcv8 recipe bundle (SPEC sec. 2 V-R8 + MM ruling Q2 + the 693-box mask)
#: PI 2026-10-11: "just give the nav command and the distance" => `--r8-no-rc`, and no `--r8-rc-*` flag at all (the route
#: checkpoint is never fed, so its variant / noise / dropout are dead flags); `--r8-nav-from-v9` + `--r8-nav-args-dropout`
#: keep the nav command and the turn distance (nav_args) flowing.
R8_BUNDLE = [("--refcv8", []), ("--r8-nav-from-v9", []), ("--r8-no-rc", []),
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


GS_EVERY = 250
GS_TERMS = "tac_v6,map_hires"


def gs_at(n: int, every: int = GS_EVERY) -> str:
    """The reading rows: the last multiple of `every` inside (0, 0.2 n] (size_k), and two inside (0.5 n, n] (L2/L3)."""
    a = (n // 5) // every * every
    b = (2 * n // 3) // every * every
    c = (5 * n // 6) // every * every
    assert 0 < a <= 0.2 * n and 0.5 * n < b < c <= n, (n, a, b, c)
    return f"{a},{b},{c}"


def base_v0(r7, n: int, out_root: str, seed: int = 0):
    ps = drop(pairs(r7), *RIG_FLAGS, *REMOVED)
    if SPEED_SOURCE == "label":
        _have = {f: v for f, v in ps if f in ROAD_TYPE_SIDECARS}
        if set(_have) != set(ROAD_TYPE_SIDECARS) or any(len(v) != 1 for v in _have.values()):
            raise SystemExit("[ladder_arms] the label source needs refcv7's road-type sidecar pair in its argv "
                             f"({ROAD_TYPE_SIDECARS}); found {_have}")
    rig = [("--size", ["tiny"]), ("--trunk-name", ["resnet34.a1_in1k"]), ("--trunk-in-channels", ["9"]),
           ("--batch", ["8"]), ("--lr", ["1e-4"]), ("--warmup", ["200"]), ("--log-every", ["50"]),
           ("--eval-every", [str(n)]), ("--eval-batches", ["64"]), ("--steps", [str(n)]), ("--save-every", ["250"]),
           ("--workers", ["2"]), ("--seed", [str(seed)]),
           ("--r8-speed-input", [SPEED_SOURCE]), ("--r8-speed-unknown-p", ["0.45"]),
           ("--r8-v9-labels", [f"{V9}/v9_labels_train.npz"]), ("--r8-v9-labels-eval", [f"{V9}/v9_labels_eval139.npz"]),
           ("--pose-sync-sidecar", [POSE_SYNC]), ("--grad-share-every", [str(GS_EVERY)]),
           ("--grad-share-terms", [GS_TERMS]), ("--grad-share-at", [gs_at(n)])]
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
    add("V-R8d", vr8 + [("--r8-derange-feed", [])], "V-R8",
        [("--r8-derange-feed", [])], "L1 deliberate regression")      # PI 2026-10-11: no RC => no --r8-rc-roll
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
    # amendment A1 -- L4b needs the N2 ladder; under the label source `--r8-speed-enc8` is refused by the trainer
    if SPEED_SOURCE == "n2":
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
_ORDER_ALL = ("V0", "V-R8", "V-R8d", "V0r", "V-R8r", "V-TACk", "V-TACk-roll", "V-TACkr", "V-MAP4", "V-MAP4-roll",
              "V-MAP4r", "V-VSHUF", "V-R8-E8", "V-R8-E8-roll", "V-R8-DRV", "V-R8-DRV-roll")
ORDER = tuple(a for a in _ORDER_ALL if SPEED_SOURCE == "n2" or a not in E8_ARMS)


def audit(A: dict, r7=None) -> dict:
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
    # PI 2026-10-11: the speed source is the SAME on every arm (a common token, so no rung reads it as a variable) -- the
    # label source with refcv7's road-type sidecar pair (verbatim from its argv when `r7` is given) and the unknown row
    r7d = dict(pairs(r7)) if r7 is not None else {}
    bad_src = sorted(n for n, (ps, *_r) in A.items()
                     if dict(ps).get("--r8-speed-input") != [SPEED_SOURCE]
                     or dict(ps).get("--r8-speed-unknown-p") != ["0.45"])
    bad_side = []
    for n, (ps, *_r) in A.items():
        d = dict(ps)
        for f in ROAD_TYPE_SIDECARS:
            want = r7d.get(f) if r7 is not None else dict(A["V0"][0]).get(f)
            if SPEED_SOURCE == "label" and (d.get(f) is None or len(d[f]) != 1 or d.get(f) != want):
                bad_side.append((n, f))
            if SPEED_SOURCE != "label" and f in d:
                bad_side.append((n, f))
    rep["speed_source_on_every_arm"] = {"PASS": not bad_src and not bad_side, "source": SPEED_SOURCE,
                                        "arms_off_source": bad_src, "sidecar_mismatch": sorted(bad_side)}
    ok &= not bad_src and not bad_side
    # PI 2026-10-11 (route): every refcv8 arm carries --r8-no-rc and NO --r8-rc-* flag (a dead RC flag would be stamped
    # and read by nothing); the nav token + its turn distance stay (--r8-nav-from-v9, --r8-nav-args-dropout)
    rc_bad = sorted(n for n, (ps, *_r) in A.items()
                    if any(f == "--refcv8" for f, _v in ps)
                    and (("--r8-no-rc" not in dict(ps)) or any(f.startswith("--r8-rc-") for f, _v in ps)
                         or "--r8-nav-from-v9" not in dict(ps)))
    rep["no_route_checkpoint_on_refcv8_arms"] = {"PASS": not rc_bad, "arms": rc_bad}
    ok &= not rc_bad
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
    rep = audit(A, r7)
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
