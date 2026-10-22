"""P6 / D1 -- the ONLINE PDM labeller (SPEC_P6_PAPER_FAITHFUL.md sections 2-3). L1: zero extra forward, lag 0.

THE PAPER'S SUPERVISION (Fig. 2, p. 8): every training sample's scorer is supervised by the NAVSIM PDM verdict on THAT sample's OWN current proposals.
Until P6 we scored banked proposals from an older checkpoint (the `OnPolicyBank` + `onpolicy_relabel_pdm.py` route, label lag in thousands of steps).
Here the trainer hands each micro-batch's detached proposals [B, M, T, 3] to a CPU pool, and the SAME proposals are what the loss is taken on:

    F(k) forward with grad -> submit(k) -> collect(k-1) -> B(k-1) -> F(k+1) ...        (the deferred backward lives in train.py)

What a worker does is exactly `onpolicy_relabel_pdm.label_log`'s body, minus the scenario build (the parts come from the D9 cache, `build_metric_cache.py`):
    parts = load(metric_cache/<log>/<token>[__<twin>].pkl.xz)
    plans8 = navsim_pdm_targets.executed_plans8(proposals, repair=True)          # the A7 last-heading repair, then to_navsim: the SAME transform the planner uses
    t = navsim_pdm_targets.score_plans(parts, plans8)                            # NAVSIM's own simulator + scorer
    labels = clip(stack(nc, dac, ep, ttc, c, ddc), 0, 1)                         # OnPolicyBank.PDM_ORDER, the clip OnPolicyBank applies

Backends: "pdm" (the real thing; workers import the NAVSIM stack, optionally under another interpreter -- the pod's labeller venv is not the trainer's) and
"synthetic" (a deterministic analytic stand-in: a CONTROL for the dev-box rehearsal and the tests; the P6 launch profile FORBIDS it).

THREE THINGS THIS MODULE REFUSES TO DO SILENTLY
* label the wrong proposals: every job carries the sha256 of the proposals it was handed and returns the sha256 of the proposals it actually scored; the
  trainer compares both with its own copy (a stale/misrouted label raises). `REFE_LAB_MUTATE=stale` is the deliberate-regression arm: it hands the worker
  the PREVIOUS micro-batch's proposals, and the check must go RED.
* hide a failing labeller: a missing parts file or a worker error masks that sample (it gets the trajectory loss only, and is COUNTED by reason); if more than
  `max_fail_frac` of the sets seen so far failed, it raises.
* wait forever: `collect` raises after `timeout_s`.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import lzma
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

COMPONENTS = ("nc", "dac", "ep", "ttc", "c", "ddc")        # = train.OnPolicyBank.PDM_ORDER = the score head's 6 outputs
BACKENDS = ("pdm", "synthetic")
MUTATE_ENV = "REFE_LAB_MUTATE"
PARTS_FORMAT = "tanitad.p6parts/1"
_W: dict = {}                                               # per-worker state (set by the initializer)


# ------------------------------------------------------------------------------------------------ keys and paths
def twin_suffix(aug) -> str:
    """'' for a rank-0 sample and for a goal-horizon twin (they share the LOGGED route's cache); '__lane_rank_<k>' for a lane-rank twin, whose PDM-Closed
    must run on the twin's augmented route (SPEC_P6 D7)."""
    if isinstance(aug, dict) and aug.get("kind") == "lane_rank":
        return f"__lane_rank_{aug.get('value')}"
    return ""


def parts_path(root, log_name: str, token: str, aug=None) -> Path:
    return Path(root) / log_name / f"{token}{twin_suffix(aug)}.pkl.xz"


def sha_of(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def load_parts(path):
    """The 6-tuple `navsim_pdm_targets.score_plans` takes: (pdm_traj, ego, obs, centerline, lane_ids, drivable_area_map). Reads our own D9 format AND a
    NAVSIM MetricCache pickle (the M-LAB cached design)."""
    with open(path, "rb") as fh:
        obj = pickle.loads(lzma.decompress(fh.read()))
    if isinstance(obj, dict) and obj.get("format") == PARTS_FORMAT:
        return tuple(obj["parts"])
    return (obj.trajectory, obj.ego_state, obj.observation, obj.centerline, list(obj.route_lane_ids), obj.drivable_area_map)


def labels_from_targets(t: dict) -> np.ndarray:
    """score_plans' dict -> [M, 6] float32 in COMPONENTS order, clipped to [0, 1] exactly as OnPolicyBank._apply_pdm_labels does."""
    return np.clip(np.stack([np.asarray(t[c], np.float32) for c in COMPONENTS], -1), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ synthetic backend (a control, never a launch)
def synthetic_labels(props: np.ndarray, salt: int = 0) -> np.ndarray:
    """[M, T, 3] -> [M, 6] in (0, 1): a smooth deterministic function of the PROPOSALS (lateral offset, progress, yaw, jerk) so that (a) labels differ across
    proposals, (b) they move when the proposals move (the stale-label mutation changes them), (c) the BCE gradient reaches the scorer."""
    p = np.asarray(props, np.float64)
    x, y, yaw = p[..., 0], p[..., 1], p[..., 2]
    lat = np.abs(y).mean(-1)
    prog = x[:, -1]
    yawm = np.abs(yaw).mean(-1)
    jerk = np.abs(np.diff(x, 2, axis=-1)).mean(-1) + np.abs(np.diff(y, 2, axis=-1)).mean(-1)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))
    cols = [sig(2.0 - lat), sig(1.5 - 0.5 * np.abs(y[:, -1])), sig(0.08 * prog - 0.5), sig(1.0 - 2.0 * yawm), sig(3.0 - 20.0 * jerk),
            sig(0.5 - 0.1 * lat + 0.05 * salt)]
    return np.clip(np.stack(cols, -1), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ worker side
def _pdm_init(metric_cache: str, lru: int = 8):
    import sys
    here = str(Path(__file__).resolve().parent)
    for q in (here, str(Path(here).parent / "code")):
        if q not in sys.path:
            sys.path.insert(0, q)
    import navsim_pdm_targets as PT
    PT._setup()
    _W.update(PT=PT, root=metric_cache, lru=OrderedDict(), lru_n=int(lru))


def _synth_init(*_a):
    _W.update(root=None)


def _job(backend: str, meta: dict, props: np.ndarray) -> dict:
    """ONE sample: [M, T, 3] proposals -> {"labels": [M, 6] | None, "sha": sha256 of the proposals SCORED, "reason": why None, "cpu_s": ...}."""
    t0 = time.process_time()
    sha = sha_of(props)
    if backend == "synthetic":
        if os.environ.get(MUTATE_ENV) == "no_labels":           # the labeller-OFF control: every sample unlabelled => the score term must read EXACTLY 0.0
            return {"labels": None, "sha": sha, "reason": "no_parts", "cpu_s": time.process_time() - t0}
        return {"labels": synthetic_labels(props), "sha": sha, "reason": None, "cpu_s": time.process_time() - t0}
    PT = _W["PT"]
    path = parts_path(_W["root"], meta["log_name"], meta["token"], meta.get("aug"))
    key = str(path)
    parts = _W["lru"].get(key)
    if parts is None:
        if not path.is_file():
            return {"labels": None, "sha": sha, "reason": "no_parts", "cpu_s": time.process_time() - t0}
        parts = load_parts(path)
        _W["lru"][key] = parts
        while len(_W["lru"]) > _W["lru_n"]:
            _W["lru"].popitem(last=False)
    else:
        _W["lru"].move_to_end(key)
    labels = labels_from_targets(PT.score_plans(parts, PT.executed_plans8(props, repair=True)))
    return {"labels": labels, "sha": sha, "reason": None, "cpu_s": time.process_time() - t0}


# ------------------------------------------------------------------------------------------------ a pool of workers under ANOTHER interpreter
class _SubprocPool:
    """N `label_worker.py` subprocesses under `python`, each a blocking request/response; a thread pool of the same size borrows one per job."""

    def __init__(self, python: str, workers: int, metric_cache: str):
        import queue
        import subprocess
        import sys
        self._procs, self._free = [], queue.Queue()
        for _ in range(max(int(workers), 1)):
            p = subprocess.Popen([python, "-u", str(Path(__file__).resolve().parent / "label_worker.py"), metric_cache], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=sys.stderr)
            self._procs.append(p)
            self._free.put(p)
        self._ex = cf.ThreadPoolExecutor(len(self._procs))

    def submit(self, _fn, backend, meta, props):
        def run():
            p = self._free.get()
            try:
                # raw bytes + shape, NOT ndarray pickles: the trainer (numpy 2.x) and the NAVSIM venv (numpy 1.x) cannot unpickle each other's arrays
                pickle.dump((meta, tuple(props.shape), np.ascontiguousarray(props, dtype=np.float64).tobytes()), p.stdin, protocol=4)
                p.stdin.flush()
                res = pickle.load(p.stdout)
            finally:
                self._free.put(p)
            if "error" in res:
                raise RuntimeError("label worker: " + res["error"])
            if res.get("labels") is not None:
                res["labels"] = np.frombuffer(res["labels"], dtype=np.float32).reshape(res["labels_shape"]).copy()
            return res
        return self._ex.submit(run)

    def shutdown(self, wait=False, cancel_futures=True):
        self._ex.shutdown(wait=False, cancel_futures=True)
        for p in self._procs:
            try:
                p.stdin.close()
                p.terminate()
            except Exception:                                                 # noqa: BLE001
                pass


# ------------------------------------------------------------------------------------------------ trainer side
class OnlineLabeller:
    def __init__(self, backend: str = "pdm", workers: int = 12, metric_cache: str | None = None, timeout_s: float = 900.0, python: str | None = None,
                 n_prop: int = 64, horizon: int = 20, max_fail_frac: float = 0.02, min_sets_for_fail_check: int = 200, chunks: int = 1):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        # SPEC_P6 3.1 "a set may be split into proposal chunks across workers for latency": `chunks` k > 1 scores each sample's M proposals as k jobs of M/k
        # CONTIGUOUS proposals and reassembles them in order.  score_plans scores every proposal against PDM-Closed alone (navsim_pdm_targets.score_plans: ep_j =
        # raw_j / max(raw_0, raw_j)), so a label does not depend on which other proposals share its job; the G-LIVE-LAB identity check (D1a) re-proves it bit for
        # bit against the OFFLINE labeller at the launch value.  MEASURED 2026-10-18 on the 8.5-CPU P6 pod (p6_cpu_probe pool --mode paced, trainer clock):
        # 1 job per set reads a 2.3-3.8 % stall at 4-8 workers; 2 jobs per set 1.9 % at 4, 0.41 % at 6, 0.05-0.15 % at 8 (raw/2026-10-18-p6-gate-diagnosis/).
        if int(chunks) < 1 or int(n_prop) % int(chunks):
            raise ValueError(f"chunks must be >= 1 and divide n_prop ({n_prop}); got {chunks}")
        self.chunks = int(chunks)
        if backend == "pdm" and not metric_cache:
            raise ValueError("the pdm backend needs --metric-cache (the D9 parts directory)")
        self.backend, self.workers, self.metric_cache = backend, int(workers), metric_cache
        self.timeout_s, self.python = float(timeout_s), python or None
        self.n_prop, self.horizon = int(n_prop), int(horizon)
        self.max_fail_frac, self.min_sets = float(max_fail_frac), int(min_sets_for_fail_check)
        self._ex = None
        self._inflight: dict = {}
        self._prev_props = None
        self.stats = dict(n_sets=0, n_labelled=0, n_no_parts=0, n_failed=0, stall_s=0.0, cpu_s=0.0, queue_sum=0, n_submits=0, lag_hist={})
        self.tot_sets = self.tot_failed = 0                     # cumulative, for the fail-rate guard (take_stats resets the window)
        self.mutate = os.environ.get(MUTATE_ENV, "")

    # ---- lifecycle
    def _executor(self):
        if self._ex is None:
            n = max(self.workers, 1)
            if self.backend == "synthetic":
                self._ex = cf.ThreadPoolExecutor(n)
            elif self.python:
                self._ex = _SubprocPool(self.python, n, self.metric_cache)       # the NAVSIM stack lives in another interpreter than the trainer's
            else:
                import multiprocessing as mp
                self._ex = cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_pdm_init, initargs=(self.metric_cache,))
        return self._ex

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = None

    # ---- the pipeline
    def submit(self, stamp: int, step: int, metas: list, props: np.ndarray) -> dict:
        """`props` [B, M, T, 3] float64 -- the DETACHED proposals of micro-batch `stamp`. Returns the handle `collect` takes."""
        props = np.ascontiguousarray(props, dtype=np.float64)
        sent = props
        if self.mutate == "stale" and self._prev_props is not None and self._prev_props.shape == props.shape:
            sent = self._prev_props                             # DELIBERATE REGRESSION (D1c): the previous micro-batch's proposals are labelled instead
        self._prev_props = props.copy()
        ex = self._executor()
        pend = sum(1 for h in self._inflight.values() for fs in h["futs"] for f in fs if not f.done())
        self.stats["queue_sum"] += pend if self.chunks == 1 else pend / self.chunks           # in SETS (jobs / chunks)
        self.stats["n_submits"] += 1
        if props.shape[1] % self.chunks:
            raise ValueError(f"{props.shape[1]} proposals per sample do not split into {self.chunks} chunks")
        m = props.shape[1] // self.chunks
        cuts = [slice(c * m, (c + 1) * m) for c in range(self.chunks)]                            # contiguous along M; reassembled in this order
        futs = [[ex.submit(_job, self.backend, meta, sent[i, sl]) for sl in cuts] for i, meta in enumerate(metas)]   # (_SubprocPool ignores _job: its workers call it)
        h = {"stamp": int(stamp), "step": int(step), "futs": futs, "shas": [sha_of(props[i]) for i in range(len(metas))],
             "cshas": [[sha_of(props[i, sl]) for sl in cuts] for i in range(len(metas))], "cuts": cuts, "t_submit": time.perf_counter()}
        self._inflight[int(stamp)] = h
        return h

    def collect(self, h: dict, step_now: int) -> tuple:
        """-> (labels [B, M, 6] float32, mask [B] float32, info). Blocks (that wait IS the GPU stall, accumulated). Raises on a timeout, a proposal-identity
        mismatch, or a failure rate above `max_fail_frac`."""
        t0 = time.perf_counter()
        B = len(h["futs"])
        labels = np.zeros((B, self.n_prop, len(COMPONENTS)), np.float32)
        mask = np.zeros(B, np.float32)
        reasons = []
        try:
            for i, fs in enumerate(h["futs"]):
                failed = missing = None
                for c, f in enumerate(fs):                               # every chunk is awaited and identity-checked, even after a sibling failed
                    rem = self.timeout_s - (time.perf_counter() - t0)
                    try:
                        r = f.result(timeout=max(rem, 0.001))
                    except cf.TimeoutError:
                        raise RuntimeError(f"the online labeller did not return micro-batch {h['stamp']} within {self.timeout_s:.0f} s (queue stalled)") from None
                    except Exception as e:                               # noqa: BLE001 -- a worker error: counted (once per SAMPLE), masked, rate-guarded below
                        failed = failed or f"{type(e).__name__}: {str(e)[:120]}"
                        continue
                    if r["sha"] != h["cshas"][i][c]:
                        raise RuntimeError(f"LABEL/PROPOSAL IDENTITY VIOLATED on micro-batch {h['stamp']} sample {i}"
                                           f"{f' chunk {c}/{len(fs)}' if len(fs) > 1 else ''}: the worker scored proposals that are not the ones the "
                                           f"trainer took its loss on (sha {r['sha'][:12]} != {h['cshas'][i][c][:12]}). A stale or misrouted label is refused.")
                    self.stats["cpu_s"] += r["cpu_s"]
                    if r["labels"] is None:
                        missing = missing or r["reason"]
                        continue
                    labels[i, h["cuts"][c]] = r["labels"]
                if failed is not None:
                    self.stats["n_failed"] += 1
                    self.tot_failed += 1
                    reasons.append(failed)
                elif missing is not None:
                    self.stats["n_no_parts"] += 1
                    reasons.append(missing)
                else:
                    mask[i] = 1.0
                if mask[i] == 0.0:
                    labels[i] = 0.0                                      # a sample with ANY unlabelled chunk carries no label at all (never a partial set)
        finally:
            self._inflight.pop(h["stamp"], None)
        self.stats["stall_s"] += time.perf_counter() - t0
        self.stats["n_sets"] += B
        self.tot_sets += B
        self.stats["n_labelled"] += int(mask.sum())
        lag = int(step_now) - h["step"]
        self.stats["lag_hist"][lag] = self.stats["lag_hist"].get(lag, 0) + B
        bad = self.tot_failed
        if self.tot_sets >= self.min_sets and bad / self.tot_sets > self.max_fail_frac:
            raise RuntimeError(f"the online labeller failed on {bad} of {self.tot_sets} sets (> {100 * self.max_fail_frac:.1f} %); last: {reasons[-1:]}")
        return labels, mask, {"stamp": h["stamp"], "lag_steps": lag, "reasons": reasons, "latency_s": time.perf_counter() - h["t_submit"]}

    def take_stats(self, reset: bool = True) -> dict:
        s = dict(self.stats)
        s["lag_hist"] = dict(s["lag_hist"])
        if reset:
            self.stats.update(n_sets=0, n_labelled=0, n_no_parts=0, n_failed=0, stall_s=0.0, cpu_s=0.0, queue_sum=0, n_submits=0)
            self.stats["lag_hist"] = {}
        return s
