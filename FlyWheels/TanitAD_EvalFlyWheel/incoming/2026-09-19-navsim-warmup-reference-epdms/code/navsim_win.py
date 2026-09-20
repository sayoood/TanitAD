#!/usr/bin/env python3
"""E1 wrapper: run an UNMODIFIED official NavSim devkit entrypoint on Windows.

The devkit (navsim@0a380a9) is imported, not copied and not edited. Before its
hydra ``main()`` runs, this wrapper applies

  * ONE behaviour patch, only if ``--patch-loader`` is given:
      ``MetricCacheLoader._load_metric_cache_paths`` (navsim/common/dataloader.py:306-317)
      extracts the scene token as ``cache_path.split("/")[-2]``. nuPlan's
      ``save_cache_metadata`` writes ``str(WindowsPath)`` = backslash paths, so on
      Windows that split yields ONE element and ``[-2]`` raises IndexError. The patch
      replaces ONLY the separator: ``re.split(r"[\\\\/]", p)[-2]``. On any path the
      original accepts it returns the identical token (pinned by
      ``test_navsim_win_patch.py``, which also shows the original failing).

  * OBSERVATION-ONLY hooks (they return the wrapped function's result unchanged):
      - ``pdm_score`` (in the runner module's namespace): records per token the
        agent poses, the metric-cache human poses, v0, scene type, call time;
      - ``compute_final_scores`` (two-stage runner): dumps the per-token frame it
        returns (weights, endpoints, EC, score) as CSV for control C5;
      - ``MetricCacheProcessor.compute_and_save_metric_cache``: per-scene wall time
        and RSS for the navhard price.

  * A RAM guard thread: samples process RSS and system available memory every 2 s
    and ABORTS the run (exit 3) if available memory falls below ``--ram-floor-mb``.

Every patch and hook applied is written to ``<out>/<label>_manifest.json``.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

SCRIPTS = {
    "metric_caching": "navsim.planning.script.run_metric_caching",
    "pdm_score": "navsim.planning.script.run_pdm_score",
    "pdm_score_one_stage": "navsim.planning.script.run_pdm_score_one_stage",
}

# env.sh defaults (C:\Users\Admin\navsim\env.sh) — applied only if unset, and recorded.
ENV_DEFAULTS = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "C:/Users/Admin/navsim/data/maps",
    "NAVSIM_EXP_ROOT": "C:/Users/Admin/navsim/exp",
    "NAVSIM_DEVKIT_ROOT": "C:/Users/Admin/navsim/devkit",
    "OPENSCENE_DATA_ROOT": "C:/Users/Admin/navsim/data/openscene",
}


# --------------------------------------------------------------------------- #
# The one behaviour patch                                                      #
# --------------------------------------------------------------------------- #
def token_from_cache_path(p: str) -> str:
    """Separator-agnostic ``p.split("/")[-2]``. Tokens are hex, so a backslash can
    never be part of one; on a forward-slash path this is the original expression."""
    return re.split(r"[\\/]", p.strip())[-2]


def _load_metric_cache_paths_patched(self, cache_path):
    """Verbatim copy of dataloader.py:306-317 except the token expression."""
    metadata_dir = cache_path / "metadata"
    metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
    with open(str(metadata_file), "r") as f:
        cache_paths = f.read().splitlines()[1:]
    metric_cache_dict = {token_from_cache_path(cache_path): cache_path for cache_path in cache_paths}
    return metric_cache_dict


def _src_sha(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# RAM guard                                                                    #
# --------------------------------------------------------------------------- #
class Monitor(threading.Thread):
    """Abort policy (v2, 2026-09-19 11:58, after the v1 guard killed a 194 MB import on a
    transient dip caused by OTHER processes): abort if available < floor for
    ``sustain`` CONSECUTIVE samples, or immediately below ``hard_floor_mb``. Every
    sub-floor sample is counted and reported, so a transient dip is visible, not hidden."""

    def __init__(self, floor_mb: float, interval_s: float, on_abort, sustain: int = 3,
                 hard_floor_mb: float = 2000.0):
        super().__init__(daemon=True)
        self.floor_mb, self.interval_s, self.on_abort = floor_mb, interval_s, on_abort
        self.sustain, self.hard_floor_mb = sustain, hard_floor_mb
        self.proc = psutil.Process()
        self.t0 = time.time()
        self.peak_rss_mb = 0.0
        self.min_avail_mb = float("inf")
        self.n = 0
        self.n_below_floor = 0
        self.consecutive = 0
        self.trace = []            # (t_s, rss_mb, avail_mb) every 10th sample
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss / 2**20
                avail = psutil.virtual_memory().available / 2**20
                tree = rss
                for c in self.proc.children(recursive=True):
                    try:
                        tree += c.memory_info().rss / 2**20
                    except Exception:                            # noqa: BLE001
                        pass
            except Exception:                                    # noqa: BLE001
                self._stop.wait(self.interval_s)
                continue
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            self.peak_tree_rss_mb = max(getattr(self, "peak_tree_rss_mb", 0.0), tree)
            self.min_avail_mb = min(self.min_avail_mb, avail)
            if self.n % 10 == 0:
                self.trace.append((round(time.time() - self.t0, 1), round(rss, 1), round(avail, 1)))
            self.n += 1
            if avail < self.floor_mb:
                self.n_below_floor += 1
                self.consecutive += 1
            else:
                self.consecutive = 0
            if avail < self.hard_floor_mb or self.consecutive >= self.sustain:
                # a process-pool worker would otherwise be ORPHANED and keep holding RAM
                kids = self.proc.children(recursive=True)
                for k in kids:
                    try:
                        k.kill()
                    except Exception:                            # noqa: BLE001
                        pass
                self.on_abort(f"system available memory {avail:.0f} MB (floor {self.floor_mb:.0f} MB for "
                              f"{self.consecutive} consecutive samples; hard floor {self.hard_floor_mb:.0f} MB); "
                              f"own RSS {rss:.0f} MB; killed {len(kids)} child process(es)")
                os._exit(3)
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()

    def summary(self) -> dict:
        return {"wall_s": round(time.time() - self.t0, 2), "peak_rss_mb": round(self.peak_rss_mb, 1),
                "peak_process_tree_rss_mb": round(getattr(self, "peak_tree_rss_mb", self.peak_rss_mb), 1),
                "min_system_available_mb": (None if self.min_avail_mb == float("inf") else round(self.min_avail_mb, 1)),
                "n_samples": self.n, "n_samples_below_floor": self.n_below_floor,
                "sample_interval_s": self.interval_s, "ram_floor_mb": self.floor_mb,
                "abort_policy": f"available<{self.floor_mb:.0f}MB for {self.sustain} consecutive samples, "
                                f"or <{self.hard_floor_mb:.0f}MB once",
                "trace_every_10th_sample__t_rss_avail": self.trace}


def git_sha(repo: str) -> str | None:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:                                            # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True, type=Path, help="where manifest/timing/hook dumps go")
    ap.add_argument("--patch-loader", action="store_true", help="apply the MetricCacheLoader separator patch")
    ap.add_argument("--dump-final-scores", type=Path, default=None)
    ap.add_argument("--ram-floor-mb", type=float, default=3000.0)
    ap.add_argument("--interval-s", type=float, default=2.0)
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="-- then hydra overrides")
    a = ap.parse_args()
    overrides = [o for o in a.overrides if o != "--"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    env_applied = {}
    for k, v in ENV_DEFAULTS.items():
        if not os.environ.get(k):
            os.environ[k] = v
            env_applied[k] = v
    manifest = {
        "label": a.label, "script": a.script, "module": SCRIPTS[a.script], "overrides": overrides,
        "argv": sys.argv, "python": sys.version, "cwd": os.getcwd(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "hash_randomization_flag": sys.flags.hash_randomization,
        "env": {k: os.environ.get(k) for k in ENV_DEFAULTS}, "env_defaults_applied_by_wrapper": env_applied,
        # The C: runtime (C:/Users/Admin/navsim-crun) is a verified COPY without .git; the SHA is
        # read from the SOURCE checkout it was copied from (raw/devkit_copy_verify.json).
        "navsim_devkit_root_used": os.environ["NAVSIM_DEVKIT_ROOT"],
        "navsim_devkit_sha": git_sha("D:/Archive/devbox-C/navsim/devkit"),
        "nuplan_devkit_sha": git_sha("D:/Archive/devbox-C/navsim/nuplan-devkit"),
        "imported_from": {},
        "patches": [], "hooks": [], "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    man_path = a.out_dir / f"{a.label}_manifest.json"
    records = {"pdm_score_calls": [], "cache_scenes": []}

    def write_all(status: str, extra: dict | None = None):
        manifest["status"] = status
        manifest["ended_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["resources"] = mon.summary()
        if extra:
            manifest.update(extra)
        man_path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        (a.out_dir / f"{a.label}_hooks.json").write_text(json.dumps(records, default=str), encoding="utf-8")

    mon = Monitor(a.ram_floor_mb, a.interval_s,
                  on_abort=lambda why: (print(f"\nE1_RAM_GUARD_ABORT: {why}", flush=True),
                                        write_all("ABORTED_RAM_GUARD", {"abort_reason": why})))
    mon.start()

    import importlib
    mod = importlib.import_module(SCRIPTS[a.script])
    import navsim as _nv, nuplan as _np                            # assert WHICH tree was imported
    manifest["imported_from"] = {"navsim": _nv.__file__, "nuplan": _np.__file__, "script": mod.__file__,
                                 "fcntl": getattr(sys.modules.get("fcntl"), "__file__", None)}

    # ---- behaviour patch -------------------------------------------------- #
    if a.patch_loader:
        from navsim.common import dataloader as _dl
        orig = _dl.MetricCacheLoader._load_metric_cache_paths
        _dl.MetricCacheLoader._load_metric_cache_paths = _load_metric_cache_paths_patched
        manifest["patches"].append({
            "name": "MetricCacheLoader._load_metric_cache_paths: token = re.split(r'[\\\\/]', p)[-2]",
            "target": "navsim/common/dataloader.py:306-317",
            "original_source_sha16": _src_sha(orig),
            "patched_source_sha16": _src_sha(_load_metric_cache_paths_patched),
            "why": "save_cache_metadata writes str(WindowsPath) (backslashes); the original split('/')[-2] raises IndexError on Windows",
        })

    # ---- observation-only hooks ------------------------------------------- #
    if hasattr(mod, "pdm_score"):
        _orig_pdm = mod.pdm_score

        def pdm_score_hook(metric_cache, model_trajectory, *args, **kw):
            t = time.perf_counter()
            out = _orig_pdm(metric_cache, model_trajectory, *args, **kw)
            dt = time.perf_counter() - t
            try:
                ht = metric_cache.human_trajectory
                dcs = metric_cache.ego_state.dynamic_car_state
                records["pdm_score_calls"].append({
                    "token": Path(str(metric_cache.file_path)).parent.name,
                    "scene_type": str(metric_cache.scene_type),
                    "log_name": metric_cache.log_name,
                    "t_wall": round(time.time(), 3), "pdm_score_s": round(dt, 4),
                    "agent_poses": model_trajectory.poses.tolist(),
                    "agent_sampling": [model_trajectory.trajectory_sampling.num_poses,
                                       model_trajectory.trajectory_sampling.interval_length],
                    "human_poses": (None if ht is None else ht.poses.tolist()),
                    "human_sampling": (None if ht is None else [ht.trajectory_sampling.num_poses,
                                                                ht.trajectory_sampling.interval_length]),
                    "v0_mps": float(dcs.rear_axle_velocity_2d.magnitude()),
                    "row": {k: (float(out[0][k].iloc[0]) if k in out[0].columns else None) for k in (
                        "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
                        "lane_keeping", "history_comfort", "multiplicative_metrics_prod", "pdm_score")},
                })
            except Exception as e:                                   # noqa: BLE001
                records["pdm_score_calls"].append({"hook_error": repr(e)})
            return out

        mod.pdm_score = pdm_score_hook
        manifest["hooks"].append(f"{SCRIPTS[a.script]}.pdm_score (observation only)")

    if a.dump_final_scores is not None and hasattr(mod, "compute_final_scores"):
        _orig_cfs = mod.compute_final_scores

        def cfs_hook(df):
            res = _orig_cfs(df)
            try:
                d = res.copy()
                if "frame_type" in d.columns:
                    d["frame_type"] = d["frame_type"].astype(str)
                d.to_csv(a.dump_final_scores, index=False)
                manifest["final_scores_dump"] = str(a.dump_final_scores)
            except Exception as e:                                   # noqa: BLE001
                manifest["final_scores_dump_error"] = repr(e)
            return res

        mod.compute_final_scores = cfs_hook
        manifest["hooks"].append(f"{SCRIPTS[a.script]}.compute_final_scores (dump only)")

    if a.script == "metric_caching":
        from navsim.planning.metric_caching import metric_cache_processor as _mcp
        _orig_cs = _mcp.MetricCacheProcessor.compute_and_save_metric_cache

        def cs_hook(self, scenario):
            t = time.perf_counter()
            res = _orig_cs(self, scenario)
            records["cache_scenes"].append({
                "token": scenario.token, "synthetic_by_len17": len(scenario.token) == 17,
                "s": round(time.perf_counter() - t, 3),
                "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
                "t_wall": round(time.time(), 3), "ok": res is not None})
            return res

        _mcp.MetricCacheProcessor.compute_and_save_metric_cache = cs_hook
        manifest["hooks"].append("MetricCacheProcessor.compute_and_save_metric_cache (timing only)")

    write_all("RUNNING")
    atexit.register(lambda: None)
    sys.argv = [mod.__file__] + overrides
    status, code = "OK", 0
    try:
        mod.main()
    except SystemExit as e:                                          # hydra exits on error
        code = int(e.code) if isinstance(e.code, int) else 1
        status = "OK" if code == 0 else f"SYSTEM_EXIT_{code}"
    except BaseException as e:                                       # noqa: BLE001
        status, code = f"EXCEPTION {type(e).__name__}: {e}", 1
        import traceback
        traceback.print_exc()
    finally:
        mon.stop()
        write_all(status)
    print(f"E1_WRAPPER_DONE label={a.label} status={status} wall_s={mon.summary()['wall_s']} "
          f"peak_rss_mb={mon.summary()['peak_rss_mb']} min_avail_mb={mon.summary()['min_system_available_mb']} "
          f"pdm_calls={len(records['pdm_score_calls'])} cache_scenes={len(records['cache_scenes'])}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
