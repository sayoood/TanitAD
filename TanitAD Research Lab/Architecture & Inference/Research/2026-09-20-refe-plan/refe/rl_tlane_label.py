#!/usr/bin/env python3
"""RL-S1 TL-1 (PI 2026-10-12, "paper-pure, smaller pool"): the PAPER-PURE teacher-lane labels of the RL scenes, produced from the frozen-fan dump.

    python refe/rl_tlane_label.py --sets <dir of onpolicy_rlfan_*.jsonl> --bank <train_grow dir> --out <dir> --repair-last-heading \\
        [--keys rl_train_scenes.tsv] [--shard I/N] [--workers 10] [--expect-ckpt-step 10075] [--limit N]
Writes <out>/teacher_lane_r0_s<I>of<N>.jsonl (the `teacher_lane_r*.jsonl` glob that `train.OnPolicyBank._apply_teacher_lane` and
`rls1_checks.tlane_check` read).  Markers: `ZZTLANE_PLAN <n>`, `ZZTLANE_DONE <written> <skipped_done> <failed>`; exit 0 only when every planned set is labelled.

WHAT THIS IS, AND WHY IT IS THE SAME LABEL AS SFT-4's.  `onpolicy_relabel_teacher_lane.py` labels QUEUE rows (`props_*`, written by the old dumper
`onpolicy_dump.py`).  A queue row is exactly two things beside the key: the model's 64 proposals `props` [M][T][3] and `teacher`, which is NOT a teacher
model's output at dump time: it is `TargetBank.rows[i]["traj"]`, the teacher trajectory BANKED in `targets_rank<r>.jsonl` (onpolicy_dump.py:141).  Both are
already in hand when the frozen-fan chain runs: the proposals are in the `onpolicy_set` record `rl_fan_dump.py` writes (traj + yaw, the SAME two-step rounding as
the old dumper's `props`), and the teacher trajectory is a row of the bank.  So no queue, no old dumper and no teacher model on a GPU is needed: this tool
REBUILDS the queue row from (dump record, bank row) and hands it to `onpolicy_relabel_teacher_lane.label_log` UNMODIFIED (same Scorer, same DriveRL calculators,
stride 2, same 9 leaves, `--repair-last-heading` as SFT-4's rows carry).  Nothing about the label is re-implemented here; the only new code is the join.
The join is proven row-for-row against real queue rows by `eval/rl_tlane_identity.py`, and the suite pins every refusal below with a mutation.

REFUSES (exit 4, nothing written): a dump record that is not an `onpolicy_set` of 64 x 20, of another `ckpt_step`, of rank != 0 or a key listed twice; a set
with no bank row, or two bank rows (never picks one); a dumped key outside --keys; a missing --repair-last-heading (SFT-4's labels carry
`repair: last_heading_hold`; without the flag these would be a DIFFERENT label).  Exit 3 = some sets could not be labelled (the failures are named).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

TOOL = "rl_tlane_label/1"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 3, 4
N_PROP, HORIZON = 64, 20
CKPT_STEP = 10075                      # model_final.pt's step (rls1_checks.CKPT_STEP)
CHUNK = 12                             # sets per job: one scenario build + 12 scorings; a kill costs at most one chunk (~2 min), the tail balances


class Refused(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------- the join
def shard_of(log: str, n: int) -> int:
    """Identical to rl_fan_dump.shard_of / rl_terms_label.shard_of (a test pins it): dump shard i and label shard i hold the same logs."""
    return zlib.crc32(log.encode("utf-8")) % n


def parse_shard(s: str):
    try:
        i, n = (int(x) for x in s.split("/"))
    except Exception:
        raise Refused(f"--shard {s!r}: need I/N")
    if not (n >= 1 and 0 <= i < n):
        raise Refused(f"--shard {s}: need 0 <= I < N")
    return i, n


def key_of(rec) -> tuple:
    return (rec["log_name"], rec["token"], int(rec["step"]), int(rec.get("rank", 0)))


def props_from_set(rec) -> list:
    """The queue row's `props` [M][T][3] from an `onpolicy_set` record: x, y from `traj`, yaw from `yaw`.  Exact, because both dumpers round x / y to 5 decimals
    and yaw to 5 (then 6, which is idempotent): `eval/rl_tlane_identity.py` compares this to real queue rows."""
    traj, yaw = rec["traj"], rec["yaw"]
    if len(traj) != N_PROP or len(yaw) != N_PROP:
        raise Refused(f"{key_of(rec)}: {len(traj)} proposals / {len(yaw)} yaw rows, need {N_PROP}")
    out = []
    for k in range(N_PROP):
        if len(traj[k]) != HORIZON or len(yaw[k]) != HORIZON or any(len(p) != 2 for p in traj[k]):
            raise Refused(f"{key_of(rec)}: proposal {k} is not {HORIZON} x (x, y) + {HORIZON} yaw")
        out.append([[float(traj[k][t][0]), float(traj[k][t][1]), float(yaw[k][t])] for t in range(HORIZON)])
    return out


def queue_row(rec, teacher) -> dict:
    """The row `onpolicy_dump.py` would have queued for this set (its field set; `kind` and the key included)."""
    return {"kind": "onpolicy_props", "log_name": rec["log_name"], "token": rec["token"], "step": int(rec["step"]), "rank": int(rec.get("rank", 0)),
            "ckpt_step": int(rec["ckpt_step"]), "teacher": teacher, "aug": rec.get("aug"), "props": props_from_set(rec)}


def read_sets(sets_dir, expect_ckpt_step, shard=(0, 1), keys=None):
    """-> ({key: record}, n_files).  Every record is checked; a bad one is a refusal, never a skip."""
    # the chain labels shard i while the dump is still WRITING shard i+1 into the same directory: read only this shard's own file (a half-written last line of another
    # shard's file is not a defect of this shard).  No file of that name (a single-file dump, the tests) = read them all and keep this shard's logs.
    files = (sorted(glob.glob(os.path.join(sets_dir, f"onpolicy_rlfan_*_s{shard[0]}of{shard[1]}.jsonl")))
             or sorted(glob.glob(os.path.join(sets_dir, "onpolicy_rlfan_*.jsonl"))))
    recs = {}
    if not files:
        raise Refused(f"no onpolicy_rlfan_*.jsonl in {sets_dir}")
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for n, ln in enumerate(fh, 1):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    raise Refused(f"{fp}:{n}: not JSON (a torn dump line: finish the dump first)")
                if r.get("kind") != "onpolicy_set":
                    raise Refused(f"{fp}:{n}: kind {r.get('kind')!r}, need onpolicy_set")
                k = key_of(r)
                if int(r.get("ckpt_step", -1)) != expect_ckpt_step:
                    raise Refused(f"{fp}:{n}: {k} has ckpt_step {r.get('ckpt_step')}, expected {expect_ckpt_step} (never label another checkpoint's fan)")
                if k[3] != 0:
                    raise Refused(f"{fp}:{n}: {k} is rank {k[3]}; RL-S1 labels rank-0 sets only")
                if keys is not None and (k[1], k[0]) not in keys:
                    raise Refused(f"{fp}:{n}: {k[:2]} is not in --keys (not a scene of the partition)")
                if shard_of(k[0], shard[1]) != shard[0]:
                    continue
                if k in recs:
                    raise Refused(f"{fp}:{n}: key {k} is dumped twice")
                recs[k] = r
    return recs, len(files)


def read_teachers(bank_dir, wanted):
    """{key: teacher traj} for the wanted keys from <bank>/targets_rank*.jsonl, as train.TargetBank reads them (rows verbatim).  Missing or doubled = refusal."""
    files = sorted(glob.glob(os.path.join(bank_dir, "targets_rank*.jsonl")))
    if not files:
        raise Refused(f"no targets_rank*.jsonl in {bank_dir}")
    want_tok = {k[1] for k in wanted}
    got, dup = {}, []
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("token") not in want_tok:
                    continue
                k = (r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0)))
                if k not in wanted:
                    continue
                if k in got:
                    dup.append(k)
                got[k] = r["traj"]
    if dup:
        raise Refused(f"the bank holds {len(dup)} wanted keys twice (first {dup[0]}); refusing to pick one")
    missing = [k for k in wanted if k not in got]
    if missing:
        raise Refused(f"{len(missing)} dumped sets have no bank row (first {missing[0]}): the dump and the bank are not the same corpus")
    for k, tr in got.items():
        if len(tr) != HORIZON or any(len(p) != 3 for p in tr):
            raise Refused(f"{k}: the banked teacher trajectory is not {HORIZON} x 3")
    return got


# ------------------------------------------------------------------------------------------------------------- resume / output
def valid_label_row(d) -> bool:
    """A row the trainer would ACCEPT and that carries the SFT-4 recipe: 4-part key, a ckpt_step, `repair: last_heading_hold`, 64 proposals each with a finite
    centre-line distance (what `train.OnPolicyBank._apply_teacher_lane` needs), and the teacher's own leaves."""
    try:
        props = d["props"]
        return (d.get("kind") == "teacher_lane_labels" and len(d["key"]) == 4 and int(d["ckpt_step"]) >= 0 and d.get("repair") == "last_heading_hold"
                and len(props) == N_PROP and isinstance(d.get("teacher"), dict)
                and all(p.get("center_line.CenterLine.info") is not None and math.isfinite(float(p["center_line.CenterLine.info"])) for p in props))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def read_done(out_dir):
    """{(key, ckpt_step)} of the VALID rows already in any teacher_lane_r*.jsonl of out_dir.  A torn line or a row that fails `valid_label_row` is not done:
    a rerun labels it again."""
    done = set()
    for fp in sorted(glob.glob(os.path.join(out_dir, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                    if valid_label_row(d):
                        k = tuple(d["key"])
                        done.add(((k[0], k[1], int(k[2]), int(k[3])), int(d["ckpt_step"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
                    continue
    return done


def terminate_torn_line(path):
    """A kill mid-write leaves a last line without its newline; appending to it would fuse the next row onto the fragment and LOSE that row (found by the suite).
    Terminate it, so the fragment stays one undecodable line (the trainer and `rls1_checks` skip such a line)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "ab") as g:
                g.write(b"\n")


def plan_jobs(recs, teachers, done, chunk=CHUNK):
    """-> [(log, [queue rows])] in sorted (log, token) order, chunked, skipping what `done` already holds."""
    by = collections.defaultdict(list)
    for k in sorted(recs, key=lambda k: (k[0], k[1])):
        if (k, int(recs[k]["ckpt_step"])) in done:
            continue
        by[k[0]].append(queue_row(recs[k], teachers[k]))
    jobs = []
    for lg in sorted(by):
        rows = by[lg]
        jobs += [(lg, rows[i:i + chunk]) for i in range(0, len(rows), chunk)]
    return jobs


def run(args, label_fn=None, init_fn=None) -> int:
    """The whole tool.  `label_fn` / `init_fn` are the test seams: with label_fn the jobs run serially in this process (no pool)."""
    if not args.repair_last_heading:
        raise Refused("--repair-last-heading is required: SFT-4's teacher-lane rows carry repair 'last_heading_hold'; without it these would be a different label")
    shard = parse_shard(args.shard)
    keys = None
    if args.keys:
        sys.path.insert(0, str(HERE))
        import rl_fan_dump as RFD
        keys = set()
        for p in args.keys:
            keys |= set(RFD.read_scenes(p))
    recs, nfiles = read_sets(args.sets, args.expect_ckpt_step, shard, keys)
    if args.limit:
        recs = {k: recs[k] for k in sorted(recs, key=lambda k: (k[0], k[1]))[: args.limit]}
    if not recs:
        raise Refused(f"no set of shard {args.shard} in {args.sets}")
    teachers = read_teachers(args.bank, recs)
    os.makedirs(args.out, exist_ok=True)
    done = read_done(args.out)
    jobs = plan_jobs(recs, teachers, done)
    n_todo = sum(len(r) for _, r in jobs)
    print(f"ZZTLANE_PLAN {len(recs)} sets in {nfiles} files (shard {args.shard}), {len(recs) - n_todo} already labelled, {n_todo} to do in {len(jobs)} jobs, "
          f"{args.workers} workers", flush=True)
    dst = os.path.join(args.out, f"teacher_lane_r0_s{shard[0]}of{shard[1]}.jsonl")
    wrote = failed = 0
    errors = []
    t0 = time.time()
    init_args = (0, args.stride, bool(args.repair_last_heading))          # rank 0, the bank's stride, the repair: ONE tuple for the pool and the serial path

    def sink(res, fh):
        nonlocal wrote, failed
        lines, nfail, err = res
        if err:
            errors.append(err)
            print("  " + err, flush=True)
        fh.writelines(lines)
        fh.flush()
        wrote += len(lines)
        failed += nfail
        print(f"  {wrote}/{n_todo} written, {failed} failed, {(time.time() - t0) / max(wrote, 1):.2f} s/set (wall, all workers)", flush=True)

    terminate_torn_line(dst)
    with open(dst, "a", encoding="utf-8", newline="\n") as fh:
        if label_fn is not None:
            if init_fn:
                init_fn(*init_args)
            for job in jobs:
                sink(label_fn(job), fh)
        else:
            import onpolicy_relabel_teacher_lane as TL
            with mp.get_context("spawn").Pool(args.workers, initializer=TL._init, initargs=init_args) as pool:
                for res in pool.imap_unordered(TL.label_log, jobs):
                    sink(res, fh)
    # a labelled row that came back is checked against the set it belongs to: the key and the checkpoint, never trusted from the worker's say-so
    ok_keys = {k: False for k in recs}
    for fp in sorted(glob.glob(os.path.join(args.out, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as f2:
            for ln in f2:
                try:
                    d = json.loads(ln)
                    k = tuple(d["key"])
                    k = (k[0], k[1], int(k[2]), int(k[3]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
                    continue
                if k in ok_keys and int(d.get("ckpt_step", -1)) == int(recs[k]["ckpt_step"]) and valid_label_row(d):
                    ok_keys[k] = True
    missing = [k for k, v in ok_keys.items() if not v]
    print(f"ZZTLANE_DONE {wrote} {len(recs) - n_todo} {len(missing)}", flush=True)
    if missing:
        print(f"ZZTLANE_UNLABELLED {len(missing)} sets, first {missing[0]}; {len(errors)} job errors, first: {errors[0] if errors else 'none (per-set failures)'}", flush=True)
        return EXIT_FAILED
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="dir of onpolicy_rlfan_*.jsonl (rl_fan_dump.py output)")
    ap.add_argument("--bank", required=True, help="the training bank dir holding targets_rank0.jsonl (the teacher trajectory of each row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="*", default=None, help="token<TAB>log TSV(s) the dump must stay inside")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--expect-ckpt-step", type=int, default=CKPT_STEP)
    ap.add_argument("--repair-last-heading", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N sets in (log, token) order")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except Refused as e:
        print(f"ZZTLANE_REFUSED {e}", flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
