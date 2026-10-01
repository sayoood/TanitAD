#!/usr/bin/env python3
"""M4a -- SLOWED-TARGET TWINS: offline, append-only, DEFAULT OFF in the trainer (measures.py).

For every teacher target row of the chosen source ranks, emit time-rescaled copies ("twins") that keep
the SAME PATH at `factor` x the speed, in the bank's own row format, so winner-takes-all sees slow
targets for the same scene and keeps slow modes in the 64-proposal fan.

THE COPY is `slow_copies.slow_copy(traj, factor, dt=0.2)` -- the ONE shared implementation (validity:
eval/selftest_slow_copies.py). This tool never re-implements it; the twin's `traj` IS its output, and a
factor-1.0 twin reproduces the source target bit-for-bit (pinned by selftest_measures.py).
⚠️ Stated, not hidden (slow_copies.py docstring): a copy starts at `factor` x the source's initial speed,
so it asks for an immediate slow-down from the ego's measured speed. This tool MEASURES how large that
step is (the implied first-interval deceleration) and writes it into its manifest.

THE ROW FORMAT (what the trainer reads -- train.py `TargetBank`):
  files     `<dir>/targets_rank<R>.jsonl`, one JSON object per line (train.py:504, stray check :519-524)
  fields    image (list of 4 camera paths, :653), log_name (:656), ego (7 floats, :714), goal (:727),
            traj (20 x [x, y, heading], ego frame, 0.2 s apart, :728); rank / token / step key the SCENE
            (:616-620) and the scorer lookups (:675, :694)
  producer  build_teacher_rollouts.make_row (build_teacher_rollouts.py:396-405); the bank's own validity
            predicate is code/grow_assemble.py `_target_ok` (:77-82) -- every twin passes it.
A twin row = its source row with
  traj        slow_copy(source traj, factor)
  rank        a NEW rank (rank_base + src_index * n_factors + factor_index): the scene gains a member,
              and the trainer's duplicate-rank guard (train.py:624-630) stays meaningful
  slow        {factor, src_rank, construction, slow_copies_sha256, divergence_m, ...}: the marker the
              trainer REFUSES without --slow-twins, and under which it keys the twin's scorer
              supervision by the SOURCE rank (measures.scorer_rank)
  source      "slow_twin"
  traj_world  DROPPED (no consumer reads it -- grep; absent is safer than a fast world trajectory
              sitting beside a slow ego-frame one)

⛔ NEVER WRITE INTO A LIVE `--grow` BANK DIRECTORY. The live trainer's copy has no twin guard and picks up
a new `targets_rank*.jsonl` at its next epoch boundary with no flag and no identity check. This tool
refuses an output directory that holds source rank files unless `--into-live-bank` is passed, which is
reserved for the declared switch procedure (see the M4a deployment notes).
⚠️ Twins must NOT be dumped for on-policy labelling: `onpolicy_label.py` treats a row's target as the
teacher (EP is normalised by its advance) and a row's rank as the ROUTE rank. They need no labels: a
twin's scorer supervision is its source's labelled set.

    python slow_twins.py --bank <dir> --out <dir> [--factors 0.75,0.5] [--src-ranks 0] [--rank-base 2]
                         [--min-divergence-m 0.3] [--fraction 1.0] [--limit N] [--dry-run]
Prints ZZSLOWTWINS_OK <json counts>; writes <out>/slow_twins_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slow_copies as SC  # noqa: E402  the ONE implementation of a slowed copy

TOOL_VERSION = 1
DT = 0.2
EGO_SPEED_INDEX = 6        # ego = [vx, vy, ax, ay, yaw_rate, steering, speed] (build_teacher_rollouts.py:381-383)


def _sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_rows(path) -> tuple:
    """(rows, n_unparseable, bytes_read): COMPLETE lines only -- a torn tail is not yet a row."""
    if not os.path.exists(path):
        return [], 0, 0
    with open(path, "rb") as f:
        buf = f.read()
    cut = buf.rfind(b"\n")
    if cut < 0:
        return [], 0, 0
    rows, bad = [], 0
    for line in buf[:cut + 1].decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad, cut + 1


def scene_key(r) -> tuple:
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)))


def keep_fraction(r, fraction: float) -> bool:
    """A deterministic per-SCENE subset: every factor and source rank of a scene decide together."""
    if fraction >= 1.0:
        return True
    h = hashlib.sha1("|".join(map(str, scene_key(r))).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64) < fraction


def rank_map(src_ranks, factors, rank_base: int) -> dict:
    """{(src_rank, factor): twin rank} -- dense, deterministic, recorded in every twin and the manifest."""
    return {(int(r), float(f)): int(rank_base) + i * len(factors) + j
            for i, r in enumerate(src_ranks) for j, f in enumerate(factors)}


def divergence_m(a, b) -> float:
    """Max per-step displacement between two ego-frame trajectories (augment_search.divergence_m)."""
    return max(math.dist(p[:2], q[:2]) for p, q in zip(a, b))


def decel_limited(src_traj, factor: float, v0: float, a_max: float) -> np.ndarray:
    """OPTIONAL VARIANT (--decel-limit), NOT the shared slow copy: the SAME path, re-timed to
        sigma(t) = min( s_src(t), max( s_src(factor t), s_brake(t) ) )
    s_src   the source's own arc length at t (linear in time between samples, the origin at t = 0);
    s_src(factor t)  the slow copy's arc length -- the copy's poses lie on the path at exactly that;
    s_brake the least distance reachable from the MEASURED speed v0 braking at a_max:
            v0 t - a_max t^2 / 2 until standstill.
    So the twin never asks for more than a_max from v0 (the pure copy starts at factor x v0 -- see the
    manifest's kinematics), never gets AHEAD of the teacher, and equals the slow copy's profile once the
    braking envelope has dropped below it. factor 1.0 returns the source to float rounding (not bits).
    Re-timed by measures.retime_to_profile (the M4b primitive), float64."""
    import torch
    import measures as MS
    src = np.asarray(src_traj, dtype=np.float64)
    T = src.shape[0]
    pts = np.concatenate([np.zeros((1, 2)), src[:, :2]], 0)
    c = np.concatenate([[0.0], np.cumsum(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(-1) + 1e-12))])
    t = DT * np.arange(1, T + 1)
    s_src = c[1:]
    s_copy = np.interp(float(factor) * t, DT * np.arange(0, T + 1), c)
    v0 = max(float(v0), 0.0)
    t_stop = v0 / a_max
    s_brake = np.where(t < t_stop, v0 * t - 0.5 * a_max * t * t, 0.5 * v0 * v0 / a_max)
    sigma = np.minimum(s_src, np.maximum(s_copy, s_brake))
    return MS.retime_to_profile(torch.from_numpy(src), torch.from_numpy(sigma)).numpy()


def make_twin(row: dict, factor: float, rank: int, sc_sha: str, decel_limit: float = 0.0) -> tuple:
    """(twin row, divergence_m). By default the twin's `traj` is exactly `slow_copy` of the source's;
    with decel_limit > 0 it is the decel-limited variant (named as such in the row)."""
    src = np.asarray(row["traj"], dtype=np.float64)
    if decel_limit > 0.0:
        tw_traj = decel_limited(src, float(factor), row["ego"][EGO_SPEED_INDEX], float(decel_limit))
        construction = (f"decel_limited(a_max={float(decel_limit)!r}): min(source, max(slow_copy profile, "
                        f"braking envelope from v0)) via measures.retime_to_profile")
    else:
        tw_traj = SC.slow_copy(src, float(factor), dt=DT)
        construction = "refe/slow_copies.py::slow_copy"
    tw = {k: v for k, v in row.items() if k != "traj_world"}
    tw["traj"] = tw_traj.tolist()
    tw["rank"] = int(rank)
    tw["source"] = "slow_twin"
    d = divergence_m(row["traj"], tw["traj"])
    tw["slow"] = {"factor": float(factor), "src_rank": int(row.get("rank", 0)),
                  "src_source": row.get("source"), "construction": construction,
                  "decel_limit": float(decel_limit) if decel_limit > 0.0 else None,
                  "slow_copies_sha256": sc_sha, "dt": DT, "divergence_m": round(d, 4),
                  "traj_world_dropped": "traj_world" in row, "tool": "slow_twins.py",
                  "version": TOOL_VERSION}
    return tw, d


def first_interval_decel(row_ego, traj) -> float:
    """Constant deceleration from the ego's MEASURED speed v0 that yields the trajectory's mean speed over
    its first 0.2 s: vbar = |p_1| / dt = v0 - a dt / 2  =>  a = 2 (v0 - vbar) / dt  (m/s^2)."""
    v0 = float(row_ego[EGO_SPEED_INDEX])
    vbar = math.hypot(traj[0][0], traj[0][1]) / DT
    return 2.0 * (v0 - vbar) / DT


def _pct(xs, qs=(5, 25, 50, 75, 95, 99, 100)) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs}


def _append_lines(path, rows) -> None:
    """Append whole lines; terminate a torn tail first (grow_assemble's contract: append-only)."""
    if not rows:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        if torn:
            with open(path, "ab") as f:
                f.write(b"\n")
    data = "".join(json.dumps(r) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def run(a) -> int:
    factors = [float(x) for x in str(a.factors).split(",") if x.strip()]
    src_ranks = [int(x) for x in str(a.src_ranks).split(",") if x.strip()]
    for f in factors:
        if not (0.0 < f <= 1.0):
            print(f"  REFUSING: factor {f} outside (0, 1] (slow_copy's domain)")
            return 4
    bank, out = Path(a.bank), Path(a.out)
    live = (out.resolve() == bank.resolve()
            or any((out / f"targets_rank{r}.jsonl").exists() for r in src_ranks))
    if live and not a.into_live_bank and not a.dry_run:
        print(f"  REFUSING: {out} holds the source rank files -- it is a BANK directory. A live --grow "
              f"trainer would pick twin files up at its next epoch boundary with no flag and no identity "
              f"check. Write to a separate directory, or pass --into-live-bank ONLY inside the declared "
              f"switch procedure.")
        return 4
    rmap = rank_map(src_ranks, factors, a.rank_base)
    if set(rmap.values()) & set(src_ranks):
        print(f"  REFUSING: twin ranks {sorted(rmap.values())} overlap the source ranks {src_ranks}")
        return 4
    sc_sha = _sha256(Path(SC.__file__))
    sources, src_rows = {}, {}
    for r in src_ranks:
        p = bank / f"targets_rank{r}.jsonl"
        if not p.exists():
            print(f"  REFUSING: no {p.name} in {bank}")
            return 4
        rows, bad, nb = read_rows(p)
        twins_in_src = sum(1 for x in rows if isinstance(x.get("slow"), dict))
        if twins_in_src:
            print(f"  REFUSING: {p.name} already holds {twins_in_src} twin rows -- no twin of a twin")
            return 4
        wrong = sum(1 for x in rows if int(x.get("rank", 0)) != r)
        if wrong:
            print(f"  REFUSING: {wrong} rows of {p.name} do not carry rank {r}")
            return 4
        sources[p.name] = {"bytes_read": nb, "rows": len(rows), "unparseable": bad}
        src_rows[r] = rows[:a.limit] if a.limit else rows
    # a twin rank must not collide with a GENUINE rank of the bank (or of the output directory)
    for (r, f), R in rmap.items():
        for d in {bank, out}:
            p = d / f"targets_rank{R}.jsonl"
            if p.exists():
                rows, _, _ = read_rows(p)
                genuine = [x for x in rows if not isinstance(x.get("slow"), dict)]
                want_dl = float(a.decel_limit) if a.decel_limit > 0 else None
                clash = [x for x in rows if isinstance(x.get("slow"), dict) and
                         ((int(x["slow"]["src_rank"]), float(x["slow"]["factor"])) != (r, f)
                          or x["slow"].get("decel_limit") != want_dl)]
                if genuine or clash:
                    print(f"  REFUSING: twin rank {R} (src rank {r}, factor {f}) collides with "
                          f"{len(genuine)} genuine / {len(clash)} other-twin rows in {p}")
                    return 4
    counts, kin_src, kin_tw = {}, [], {f: [] for f in factors}
    to_write: dict = {}
    for (r, f), R in rmap.items():
        outp = out / f"targets_rank{R}.jsonl"
        have = {scene_key(x) for x in read_rows(outp)[0]} if outp.exists() else set()
        c = {"src_rank": r, "factor": f, "emitted": 0, "already_present": 0,
             "skipped_divergence": 0, "skipped_fraction": 0}
        add = []
        for row in src_rows[r]:
            if not keep_fraction(row, a.fraction):
                c["skipped_fraction"] += 1
                continue
            if scene_key(row) in have:
                c["already_present"] += 1
                continue
            tw, d = make_twin(row, f, R, sc_sha, a.decel_limit)
            if d < a.min_divergence_m:
                c["skipped_divergence"] += 1       # a near-duplicate target double-weights its source
                continue
            add.append(tw)
            kin_tw[f].append(first_interval_decel(row["ego"], tw["traj"]))
            c["emitted"] += 1
        counts[str(R)] = c
        to_write[outp] = add
    for r in src_ranks:
        for row in src_rows[r]:
            kin_src.append(first_interval_decel(row["ego"], row["traj"]))
    kin = {"definition": "a = 2 (v0 - |p_1| / 0.2 s) / 0.2 s: the constant deceleration from the ego's "
                         "measured speed that yields the trajectory's first-interval mean speed (m/s^2); frac_gt_4 / frac_gt_8 count values above 4.01 / 8.01 (a 0.01 m/s^2 rounding allowance)",
           "source_targets": {"n": len(kin_src), **_pct(kin_src),
                              "frac_gt_4": round(float(np.mean(np.asarray(kin_src) > 4.01)), 4)
                              if kin_src else None,
                              "frac_gt_8": round(float(np.mean(np.asarray(kin_src) > 8.01)), 4)
                              if kin_src else None},
           "twins": {str(f): {"n": len(v), **_pct(v),
                              "frac_gt_4": round(float(np.mean(np.asarray(v) > 4.01)), 4) if v else None,
                              "frac_gt_8": round(float(np.mean(np.asarray(v) > 8.01)), 4) if v else None}
                     for f, v in kin_tw.items()}}
    manifest = {"tool": "slow_twins.py", "version": TOOL_VERSION, "tool_sha256": _sha256(__file__),
                "slow_copies": str(Path(SC.__file__).resolve()), "slow_copies_sha256": sc_sha,
                "bank": str(bank), "out": str(out), "factors": factors, "src_ranks": src_ranks,
                "construction": ("decel_limited" if a.decel_limit > 0 else "slow_copy"),
                "decel_limit": a.decel_limit if a.decel_limit > 0 else None,
                "rank_map": {str(R): {"src_rank": r, "factor": f} for (r, f), R in rmap.items()},
                "min_divergence_m": a.min_divergence_m, "fraction": a.fraction, "limit": a.limit,
                "dry_run": bool(a.dry_run), "sources": sources, "counts": counts,
                "kinematics": kin, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for p, rows in to_write.items():
            _append_lines(p, rows)
    mpath = Path(a.manifest) if a.manifest else (None if a.dry_run else out / "slow_twins_manifest.json")
    if mpath is not None:
        mpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(mpath) + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, mpath)
    print(f"  twins per rank: {json.dumps(counts)}")
    print(f"  implied first-interval deceleration (m/s^2): source {json.dumps(kin['source_targets'])}")
    for f, v in kin["twins"].items():
        print(f"      twins x{f}: {json.dumps(v)}")
    print("ZZSLOWTWINS_OK " + json.dumps({k: v["emitted"] for k, v in counts.items()})
          + (" (dry run: nothing written)" if a.dry_run else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="directory holding targets_rank<r>.jsonl (read only)")
    ap.add_argument("--out", required=True, help="directory the twin rank files are APPENDED to")
    ap.add_argument("--factors", default="0.75,0.5", help="slow_copy factors in (0, 1]")
    ap.add_argument("--src-ranks", default="0", help="source ranks to twin (0 = logged intent)")
    ap.add_argument("--rank-base", type=int, default=2, help="first twin rank (0/1 are route ranks)")
    ap.add_argument("--min-divergence-m", type=float, default=0.3,
                    help="skip a twin whose max per-step displacement from its source is below this "
                         "(a near-duplicate); 0.3 = the live bank's augmentation tau")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="twin a deterministic per-scene subset of this size (dilution control)")
    ap.add_argument("--limit", type=int, default=0, help="first N source rows per rank only (tests)")
    ap.add_argument("--decel-limit", type=float, default=0.0, metavar="A",
                    help="OPTIONAL VARIANT (0 = off = the shared slow copy): floor the copy's profile by "
                         "braking at A m/s^2 from the measured ego speed, cap it at the source (decel_limited)")
    ap.add_argument("--dry-run", action="store_true", help="count and measure; write no bank file")
    ap.add_argument("--manifest", default=None, help="manifest path (default <out>/slow_twins_manifest.json)")
    ap.add_argument("--into-live-bank", action="store_true",
                    help="allow writing next to the source rank files (the declared switch ONLY)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
