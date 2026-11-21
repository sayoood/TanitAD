#!/usr/bin/env python3
"""PAPER-PURE lane labels (PI 2026-10-04: "let's do the paper pure way"): the TEACHER SIMULATOR's own lane signals for the
model's proposals, from the labeller's own Scorer (onpolicy_label.Scorer, DriveRL's calculators, stride 2) -- the same
code path that produced every existing scorer label, keeping the leaves that path computed and the bank never stored.

Per proposal (aggregated over prefixes by score_proposal_rollout's own rule: `.info` / event leaves = max, rewards = min):
  center_line.CenterLine.reward     the teacher's lane-centring factor (1.0 centred .. 0.75 at the line; a soft score
                                    in the teacher's per-step product reward)
  center_line.CenterLine.info       distance to the teacher's centreline (m, max over the horizon)
  off_road.CrossLane.reward         1.0 / 0.4 solid-line crossing / 0.2 wrong-way (soft score in the teacher's reward)
  off_road.CrossLane.lane_change_info, .solid_lane_change_info   lane-line crossing events (the latter ends an episode)
  off_road.OffRoad.info, ddc.violation, dac.violation             as the bank already holds them (consistency check)
Side files keyed like train.OnPolicyBank (log, token, step, rank) + ckpt_step, from the on-policy QUEUE rows (which carry
the proposals AND the teacher rollout the Scorer needs first). Map knowledge enters only through the teacher simulator,
exactly as in the paper; nothing here is a model input.
    python refe/onpolicy_relabel_teacher_lane.py --queue <dir> --files 'props_r0_*' --rank 0 --out <dir> [--workers 8]
                                                 [--repair-last-heading]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
LEAVES = ("center_line.CenterLine.reward", "center_line.CenterLine.info", "off_road.CrossLane.reward",
          "off_road.CrossLane.info", "off_road.CrossLane.lane_change_info", "off_road.CrossLane.solid_lane_change_info",
          "off_road.OffRoad.info", "ddc.violation", "dac.violation")
_S: dict = {}


def _init(rank, stride, repair):
    import onpolicy_label as OL
    _S["OL"] = OL
    _S["S"] = OL.Scorer(rank, stride)
    _S["repair"] = repair


def label_log(job):
    log, rows = job
    OL, S = _S["OL"], _S["S"]
    out, failed = [], 0
    try:
        scs = S.scenarios(log, sorted({r["token"] for r in rows}))
    except Exception as e:                                                   # noqa: BLE001
        return [], len(rows), f"{log}: scenario build {type(e).__name__}: {str(e)[:120]}"
    for r in rows:
        try:
            sc = scs[r["token"]]
            tr = np.asarray(r["teacher"], float)
            P = np.asarray(r["props"], float)
            PL = P
            if _S["repair"]:
                from planner import repair_last_heading
                PL = np.asarray(repair_last_heading(P), float)
            cands = [("teacher", OL._t(tr[:, :2]), OL._t(tr[:, 2]))] + \
                    [(k, OL._t(PL[k, :, :2]), OL._t(PL[k, :, 2])) for k in range(P.shape[0])]
            got, _nd = S.score(sc, r, cands)

            def leaves(c):
                return {k: (None if got[c].get(k) is None else float(got[c].get(k))) for k in LEAVES}
            out.append(json.dumps({"kind": "teacher_lane_labels",
                                   "key": [r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))],
                                   "ckpt_step": int(r["ckpt_step"]), "repair": "last_heading_hold" if _S["repair"] else None,
                                   "teacher": leaves("teacher"), "props": [leaves(k) for k in range(P.shape[0])]}) + "\n")
        except Exception:                                                    # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repair-last-heading", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, f"teacher_lane_r{a.rank}.jsonl")
    done = set()
    if os.path.exists(dst):
        for ln in open(dst, encoding="utf-8"):
            try:
                d = json.loads(ln)
                done.add((tuple(d["key"]), d["ckpt_step"]))
            except (json.JSONDecodeError, KeyError):
                pass
    by: dict = {}
    n = 0
    for fp in sorted(glob.glob(os.path.join(a.queue, a.files))):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "teacher" not in r or "props" not in r or int(r.get("rank", 0)) != a.rank:
                continue
            k = ((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"]))
            if k in done:
                continue
            done.add(k)
            by.setdefault(r["log_name"], []).append(r)
            n += 1
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    print(f"rank {a.rank}: {n} samples over {len(by)} logs, {a.workers} workers, repair {a.repair_last_heading}", flush=True)
    t0 = time.time()
    k = f_ = 0
    with open(dst, "a", encoding="utf-8") as f, mp.get_context("spawn").Pool(
            a.workers, initializer=_init, initargs=(a.rank, a.stride, a.repair_last_heading)) as pool:
        for lines, failed, err in pool.imap_unordered(label_log, list(by.items())):
            if err:
                print("  " + err, flush=True)
            f.writelines(lines)
            f.flush()
            k += len(lines)
            f_ += failed
            print(f"  {k}/{n} written, {f_} failed, {(time.time() - t0) / max(k, 1):.2f} s/sample", flush=True)
    print(f"ZZTEACHER_LANE_DONE {k} {f_}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
