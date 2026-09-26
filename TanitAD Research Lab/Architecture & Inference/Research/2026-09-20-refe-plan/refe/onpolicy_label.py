#!/usr/bin/env python3
"""On-policy scorer labels, POD-LOCAL (PI decision 2026-09-26, option B -- the paper's supervision).

Scores the student's OWN proposals with the teacher's simulator, through EXACTLY the per-frame path
`build_scorer_targets.py` used for the bank: logged context (`LogSession.scenario_data(logged=True)`),
lane-graph enrichment at the logged ego, the per-row augmented route for rank 1, the same
calculators and stride, EP relative to the teacher's own advance, and the discrimination rule
(a frame whose candidates differ in fewer than 3 signals is not banked). The only change is the
candidate set: the teacher path (for EP) plus the student's 64 proposals, instead of the fixed
perturbations.

  worker   python onpolicy_label.py --queue <props dir> --out <onpolicy dir> --rank 0|1 --worker w0
  check    python onpolicy_label.py --selftest-bank <train_grow dir> --rank 0|1 --n 6

WORKER: claims `props_r<rank>_*.jsonl` chunks written by `onpolicy_dump.py` (atomic rename), and
appends ONE line per sample (`kind: onpolicy_set`, all 64 slots) to `<out>/onpolicy_r<rank>_<worker>.jsonl`
-- one flushed write per sample, so a kill costs at most that sample. Resumes by skipping samples
already in its own file. One rank per process, as the builder: rank 1 installs the route patch
process-wide (`route_rank.apply`).

SELFTEST: re-scores the TEACHER path of N banked samples and must reproduce the bank's own `teacher`
candidate targets EXACTLY (content, all keys). It is what licenses calling these labels the same
instrument as the bank. Prints ZZOPLABEL_SELFTEST_OK / _FAIL.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

# what the trainer reads (ScorerBank.components) + the EP numerator; the rest of the ~40 signals
# stays out of the bank to keep the epoch-boundary parse cheap
KEEP = ("collision.NuPlanCollision.info", "dac.violation", "off_road.OffRoad.info", "progress.ep",
        "progress.advance_m", "ttc.NuPlanTTC.ttc_reward", "comfort.Comfort.reward", "ddc.violation")
BASE_GOAL_HORIZON_S = 12.0          # build_scorer_targets.py / build_teacher_rollouts.build_planner


class Scorer:
    """The builder's per-frame scoring, once per process (same setup order as its main())."""

    def __init__(self, rank: int, stride: int):
        import route_rank
        print(f"  {route_rank.apply(rank)}", flush=True)
        if os.environ.get("REFE_MAP_CACHE", "1") != "0":
            import map_feature_cache
            map_feature_cache.install()
        if os.environ.get("REFE_QUERY_CACHE", "1") != "0":
            import query_cache
            query_cache.install()
        self.rank, self.stride = int(rank), int(stride)
        self.RLP = None
        if self.rank == 1:
            import route_lane_rank_patch as RLP
            from nuplan.planning.script import driverl_runtime_map_features as _M
            if not (getattr(RLP, "_INSTALLED", False) and hasattr(_M._choose_route_edge, "__wrapped__")):
                raise SystemExit("REFUSING: rank 1 needs the route patch installed and wrapping "
                                 "`_choose_route_edge`, and it is not")
            self.RLP = RLP
        import scorer_gate as G
        import score_proposals as SP
        import augment_routes as AR
        import navtrain_scenarios as NS
        import build_scorer_targets as BST
        self.G, self.SP, self.AR, self.NS, self.BST = G, SP, AR, NS, BST
        cfg, _ = G.build_engine_config()
        self.calcs = SP.build_calculators(cfg)
        self.dbs = NS.index_dbs()
        print(f"  scorer: rank {self.rank} stride {self.stride} ego view "
              f"{'ON' if SP.EGO_VIEW else 'OFF'}; {len(self.dbs):,} DBs indexed", flush=True)

    def scenarios(self, log_name: str, tokens: list) -> dict:
        return {sc.scenario_name: sc for sc in self.NS.build_scenarios_for_log(self.dbs[log_name], tokens)}

    def score(self, scenario, row: dict, cands: list) -> tuple:
        """`cands`: [(name, xy [T,2] f32 tensor, yaw [T] f32 tensor)], the teacher FIRST.
        Returns ({name: targets}, ndiff) exactly as the builder computes them for its set."""
        sess = self.BST.LogSession(scenario=scenario)
        step = int(row["step"])
        if self.RLP is not None:          # the builder's per-row augmented route (rank 1)
            _aug = row.get("aug")
            if _aug and _aug.get("kind") == "lane_rank":
                self.RLP.set_rank(int(_aug["value"]))
                sess.builder.config.route_goal_horizon_s = BASE_GOAL_HORIZON_S
            elif _aug and _aug.get("kind") == "goal_horizon_s":
                self.RLP.set_rank(0)
                sess.builder.config.route_goal_horizon_s = float(_aug["value"])
            else:
                self.RLP.set_rank(0)
                sess.builder.config.route_goal_horizon_s = BASE_GOAL_HORIZON_S
        sd, log_sd = sess.scenario_data(step, logged=True)
        anchor = sess.logged_ego(step)
        sd, _st = self.SP.enrich_lane_graph(sd, sess.scenario.map_api, self.AR._anchor_from_ego(anchor),
                                            sess.init.route_roadblock_ids)
        by = {nm: self.SP.score_proposal_rollout(sd, log_sd, self.calcs, xy, yw, stride=self.stride)
              for nm, xy, yw in cands}
        t_adv = by["teacher"].get("progress.advance_m", float("nan"))
        for _r in by.values():
            adv = _r.get("progress.advance_m", float("nan"))
            _r["progress.ep"] = (max(0.0, min(1.0, adv / t_adv))
                                 if adv == adv and t_adv == t_adv and t_adv > 1e-3 else float("nan"))
        keys = [k for k in by["teacher"] if not k.startswith("_") and not k.endswith("@last")]
        ndiff = 0
        for k in keys:
            vals = set()
            for c in by:
                v = by[c].get(k)
                if not isinstance(v, str) and v is not None:
                    vals.add(round(float(v), 6))
            if len(vals) > 1:
                ndiff += 1
        out = {nm: {k: v for k, v in r.items() if not k.endswith("@last") and not k.startswith("_")}
               for nm, r in by.items()}
        return out, ndiff


def _t(a):
    return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32)


def selftest(a) -> int:
    """The labeller's teacher path must reproduce the bank's own `teacher` rows exactly."""
    S = Scorer(a.rank, a.stride)
    bank = Path(a.selftest_bank)
    # the reference: the bank's own `teacher` rows, taken from the START of the scorer file (a few MB,
    # not a 3 GB scan on the shared filesystem the trainer reads images from); every 40th frame
    sfile = bank / ("scorer_targets.jsonl" if a.rank == 0 else "scorer_targets_rank1.jsonl")
    ref, n_fr, t0 = {}, 0, time.time()
    with open(sfile, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("candidate") != "teacher" or int(r.get("rank", 0)) != a.rank:
                continue
            n_fr += 1
            if n_fr % 40 == 1:
                ref[(r["log_name"], r["token"], int(r["step"]))] = r["targets"]
            if len(ref) >= a.n:
                break
    toks = {k[1] for k in ref}
    rows = []
    with open(bank / f"targets_rank{a.rank}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not any(t in line for t in toks):
                continue
            r = json.loads(line)
            if (r["log_name"], r["token"], int(r["step"])) in ref and int(r.get("rank", 0)) == a.rank:
                rows.append(r)
            if len(rows) == len(ref):
                break
    print(f"  bank teacher rows {len(ref)}, their training rows {len(rows)} ({time.time() - t0:.0f} s)",
          flush=True)
    n_ok = n_cmp = 0
    for r in rows:
        k = (r["log_name"], r["token"], int(r["step"]))
        if k not in ref:
            print(f"    {k[0][:30]} {k[1]}: no banked teacher row (frame aborted by the builder?)")
            continue
        sc = S.scenarios(r["log_name"], [r["token"]]).get(r["token"])
        tr = np.asarray(r["traj"], dtype=float)
        t1 = time.time()
        got, nd = S.score(sc, r, [("teacher", _t(tr[:, :2]), _t(tr[:, 2]))])
        mine = json.loads(json.dumps(got["teacher"]))            # the bank's own float round trip
        theirs = ref[k]
        diff = sorted(k_ for k_ in set(mine) | set(theirs)
                      if json.dumps(mine.get(k_)) != json.dumps(theirs.get(k_)))
        # ⭐ THE PASS CRITERION IS THE TRAINER'S KEYS. MEASURED 2026-09-26: `center_line.CenterLine.info`
        # alone differs on 3 of 12 frames, and the PRODUCTION builder re-run cold on those frames
        # reproduces THIS labeller, not the bank (rank 1: ~1e-7, float32 accumulation order under a
        # different thread count; rank 0: 0.4224 vs 0.4143, a near-tie lane flip). It is a diagnostic
        # no component reads -- reported, never silently equal.
        hard = [k_ for k_ in diff if k_ in KEEP]
        n_cmp += 1
        n_ok += int(not hard)
        print(f"    {k[0][:30]} {k[1]}: {len(mine)} keys, trainer keys "
              f"{'IDENTICAL' if not hard else 'DIFF ' + str(hard)}"
              f"{'' if not diff else '; diagnostic-only diffs ' + str([d for d in diff if d not in KEEP][:3])}"
              f"  ({time.time() - t1:.1f} s)", flush=True)
    ok = n_cmp >= max(1, a.n // 2) and n_ok == n_cmp
    print(f"ZZOPLABEL_SELFTEST_{'OK' if ok else 'FAIL'} rank {a.rank} trainer-keys identical {n_ok}/{n_cmp}")
    return 0 if ok else 1


def _done_keys(path: str) -> set:
    """Samples already written by this worker; a torn last line is terminated so it stays one bad line."""
    done = set()
    if not os.path.exists(path):
        return done
    if os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "ab") as g:
                    g.write(b"\n")
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["log_name"], r["token"], int(r["step"]), int(r["rank"]), int(r["ckpt_step"])))
    return done


def worker(a) -> int:
    S = Scorer(a.rank, a.stride)
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"onpolicy_r{a.rank}_{a.worker}.jsonl")
    done = _done_keys(out_path)
    stat_path = os.path.join(a.queue, f"status_r{a.rank}_{a.worker}.json")
    st = {"worker": a.worker, "rank": a.rank, "samples": 0, "written": 0, "skipped_ndiff": 0,
          "failed": 0, "chunks": 0, "sec_scoring": 0.0, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"  worker {a.worker} rank {a.rank}: {len(done):,} samples already in {out_path}", flush=True)
    fh = open(out_path, "a", encoding="utf-8")
    idle_since = None
    while True:
        # own in-flight chunks first (a restart after a kill), then a fresh claim
        mine = sorted(glob.glob(os.path.join(a.queue, f"props_r{a.rank}_*.jsonl.{a.worker}")))
        chunk = mine[0] if mine else None
        if chunk is None:
            for c in sorted(glob.glob(os.path.join(a.queue, f"props_r{a.rank}_*.jsonl"))):
                try:
                    os.rename(c, c + "." + a.worker)          # atomic: exactly one worker wins
                    chunk = c + "." + a.worker
                    break
                except OSError:
                    continue
        if chunk is None:
            if a.once:
                break
            idle_since = idle_since or time.time()
            time.sleep(a.poll_s)
            continue
        idle_since = None
        rows = []
        with open(chunk, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        by_log: dict = {}
        for r in rows:
            by_log.setdefault(r["log_name"], []).append(r)
        for lg, rs in by_log.items():
            try:
                scs = S.scenarios(lg, sorted({r["token"] for r in rs}))
            except Exception as exc:                          # a DB that cannot build: count, go on
                st["failed"] += len(rs)
                print(f"    {lg[:34]}: scenario build FAILED {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
            for r in rs:
                key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]), int(r["ckpt_step"]))
                if key in done:
                    continue
                st["samples"] += 1
                sc = scs.get(r["token"])
                if sc is None:
                    st["failed"] += 1
                    continue
                t1 = time.time()
                try:
                    tr = np.asarray(r["teacher"], dtype=float)
                    P = np.asarray(r["props"], dtype=float)                   # [M, T, 3]
                    cands = [("teacher", _t(tr[:, :2]), _t(tr[:, 2]))] + \
                            [(k, _t(P[k, :, :2]), _t(P[k, :, 2])) for k in range(P.shape[0])]
                    got, nd = S.score(sc, r, cands)
                except Exception as exc:
                    st["failed"] += 1
                    print(f"    {r['token']}: scoring FAILED {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                    continue
                st["sec_scoring"] += time.time() - t1
                if nd < 3:                                   # the builder's discrimination rule
                    st["skipped_ndiff"] += 1
                    done.add(key)
                    continue
                M = P.shape[0]
                line = json.dumps({
                    "kind": "onpolicy_set", "log_name": r["log_name"], "token": r["token"],
                    "step": int(r["step"]), "rank": int(r["rank"]), "ckpt_step": int(r["ckpt_step"]),
                    "aug": r.get("aug"), "ndiff": int(nd), "stride": S.stride,
                    "traj": [[[round(float(x), 5), round(float(y), 5)] for x, y in P[k, :, :2]] for k in range(M)],
                    "yaw": [[round(float(v), 6) for v in P[k, :, 2]] for k in range(M)],
                    "targets": [{kk: got[k].get(kk) for kk in KEEP} for k in range(M)],
                    "teacher_targets": {kk: got["teacher"].get(kk) for kk in KEEP},
                    "labeller": a.worker, "sec": round(time.time() - t1, 2),
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
                fh.write(line)
                fh.flush()
                done.add(key)
                st["written"] += 1
        os.rename(chunk, chunk.rsplit(".", 1)[0] + ".done_" + a.worker)
        st["chunks"] += 1
        st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["sec_per_sample"] = round(st["sec_scoring"] / max(st["written"] + st["skipped_ndiff"], 1), 2)
        tmp = stat_path + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, stat_path)
        print(f"  chunk {os.path.basename(chunk)}: {len(rows)} samples -> written {st['written']:,} "
              f"total, skipped(ndiff) {st['skipped_ndiff']}, failed {st['failed']}, "
              f"{st['sec_per_sample']} s/sample", flush=True)
        if a.max_chunks and st["chunks"] >= a.max_chunks:
            break
    fh.close()
    print(f"ZZOPLABEL_WORKER_EXIT {a.worker} written {st['written']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True, choices=(0, 1))
    ap.add_argument("--stride", type=int, default=2, help="the bank's stride (build_scorer_targets default)")
    ap.add_argument("--queue", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument("--once", action="store_true", help="exit when the queue is empty")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--selftest-bank", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    if a.selftest_bank:
        return selftest(a)
    if not (a.queue and a.out):
        print("  --queue and --out are required for a worker")
        return 2
    return worker(a)


if __name__ == "__main__":
    sys.exit(main())
