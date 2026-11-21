#!/usr/bin/env python3
"""W3 wrapper: run an UNMODIFIED official NAVSIM **v1.1** entrypoint on Windows (NAVSIM VENV).

The v1.1 devkit (``autonomousvision/navsim`` v1.1 branch @ ``3e8291b``, unpacked at
``V11_TREE``) is IMPORTED through ``PYTHONPATH`` — never copied, never edited. The C: venv
(``C:/Users/Admin/navsim-crun/venv``) carries navsim **2.0.0** as a ``.pth`` that is APPENDED to
``sys.path`` after site-packages, so a ``PYTHONPATH`` entry wins; this wrapper does not trust
that — it ASSERTS ``navsim.__file__`` is inside ``V11_TREE`` and REFUSES otherwise (C8).

Before the devkit's hydra ``main()`` runs it applies

* at most ONE behaviour patch, only with ``--patch-loader``: v1.1's
  ``MetricCacheLoader._load_metric_cache_paths`` (``navsim/common/dataloader.py:170-181``) keys
  the cache by ``cache_path.split("/")[-2]``; nuPlan's ``save_cache_metadata`` writes
  ``str(WindowsPath)``. The replacement is **E1's function, imported from E1's package**
  (``…/2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py``): the same body with only
  the separator made agnostic. ``tests/test_v1_loader_patch.py`` pins that E1's body equals
  v1.1's body line for line except the token expression, and control C7 shows the defect on a
  real v1.1 cache before the patch is ever used for a score.
* OBSERVATION-ONLY hooks (they return the wrapped result unchanged):
    - ``run_pdm_score.pdm_score``: per token the agent poses, v0, the result row, the
      scorer's raw progress of both proposals and their multiplier products (so "max compliant
      progress ≤ 5 m ⇒ EP ≡ 1" can be counted), and the call's wall time;
    - ``MetricCacheProcessor.compute_metric_cache``: per-scene wall time and RSS (the price).
* E1's RAM guard (``Monitor``): abort (exit 3) if system available memory stays below
  ``--ram-floor-mb`` for 3 consecutive 2 s samples, or once below 2,000 MB.
* process priority BELOW_NORMAL (a training run shares this CPU).

Every patch, hook, import path and env value is written to ``<out>/<label>_manifest.json``.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import psutil

V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V11_SHA = "3e8291bfa89ff247231e0227778840cd0a036896"
NUPLAN_TREE = "C:/Users/Admin/navsim-crun/nuplan-devkit"
NUPLAN_SHA = "ce3c323af01c0d7ec5672f7832ef53f9c679aab0"      # tag nuplan-devkit-v1.2 (v1.1 pin)
NUPLAN_SOURCE = "D:/Archive/devbox-C/navsim/nuplan-devkit"   # the checkout the C: tree was copied from
E1_WRAPPER = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
              "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")

SCRIPTS = {
    "metric_caching": "navsim.planning.script.run_metric_caching",
    "pdm_score": "navsim.planning.script.run_pdm_score",
}

#: D: ONLY (PI 2026-09-19). Applied only if unset; recorded either way.
ENV_DEFAULTS = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "D:/Archive/devbox-C/navsim/data/maps",
    "NAVSIM_EXP_ROOT": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1",
    "NAVSIM_DEVKIT_ROOT": V11_TREE,
    "OPENSCENE_DATA_ROOT": "D:/Archive/devbox-C/navsim/data/openscene",
}


def load_e1():
    """E1's wrapper module, imported by path (its ``main`` is guarded, import is inert)."""
    spec = importlib.util.spec_from_file_location("e1_navsim_win", E1_WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def assert_tree(mod_file: str, tree: str, what: str) -> None:
    if not _norm(mod_file).startswith(_norm(tree) + "/"):
        raise SystemExit(f"⛔ {what} imported from {mod_file}, not under {tree} — refusing (C8)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--patch-loader", action="store_true")
    ap.add_argument("--ram-floor-mb", type=float, default=3000.0)
    ap.add_argument("--interval-s", type=float, default=2.0)
    ap.add_argument("--record-poses", action="store_true",
                    help="keep per-token agent poses in the hooks file (large on full runs)")
    ap.add_argument("overrides", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    overrides = [o for o in a.overrides if o != "--"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        prio = "BELOW_NORMAL"
    except Exception as e:                                          # noqa: BLE001
        prio = f"unchanged ({e!r})"

    env_applied = {}
    for k, v in ENV_DEFAULTS.items():
        if not os.environ.get(k):
            os.environ[k] = v
            env_applied[k] = v
    e1 = load_e1()
    manifest = {
        "label": a.label, "script": a.script, "module": SCRIPTS[a.script], "overrides": overrides,
        "argv": sys.argv, "python": sys.version, "executable": sys.executable, "cwd": os.getcwd(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "priority": prio,
        "env": {k: os.environ.get(k) for k in ENV_DEFAULTS}, "env_defaults_applied_by_wrapper": env_applied,
        "navsim_v11_tree": V11_TREE, "navsim_v11_sha": V11_SHA,
        "nuplan_tree": NUPLAN_TREE, "nuplan_sha_expected": NUPLAN_SHA,
        # the C: tree is E1's COPY (no .git) of the D: source checkout; its blob-level
        # verification against ce3c323 (1321/1324 blobs identical, 3 non-runtime, explained) is
        # E1's raw/devkit_copy_verify.json — INHERITED here, the source SHA is MEASURED:
        "nuplan_sha_measured": e1.git_sha(NUPLAN_TREE),
        "nuplan_source_checkout": NUPLAN_SOURCE,
        "nuplan_source_sha_measured": e1.git_sha(NUPLAN_SOURCE),
        "nuplan_copy_verification": ("E1 raw/devkit_copy_verify.json: repos.nuplan-devkit "
                                     "n_blobs 1324, n_match 1321, 3 diffs explained (setup.py "
                                     "packaging, docs/make.bat eol, nuboard html)"),
        "e1_wrapper": E1_WRAPPER, "imported_from": {}, "patches": [], "hooks": [],
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    man_path = a.out_dir / f"{a.label}_manifest.json"
    records = {"pdm_score_calls": [], "cache_scenes": []}
    # ⛔ THE AGGREGATION HAZARD (MEASURED on navhard 2026-09-19: the v2 two-stage runner scored all
    # 5,912 scenarios and then DIED in aggregation — the whole compute lost because the CSV is
    # written only after it). Every per-token row is therefore STREAMED to a JSONL file the moment
    # the devkit produces it, flushed per line, so a crash anywhere after scoring costs nothing.
    jsonl_path = a.out_dir / f"{a.label}_rows.jsonl"
    jsonl = open(jsonl_path, "w", encoding="utf-8", buffering=1)
    manifest["rows_jsonl"] = str(jsonl_path)

    def write_all(status: str, extra: dict | None = None):
        manifest["status"] = status
        manifest["ended_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["resources"] = mon.summary()
        if extra:
            manifest.update(extra)
        man_path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        (a.out_dir / f"{a.label}_hooks.json").write_text(json.dumps(records, default=str),
                                                          encoding="utf-8")

    mon = e1.Monitor(a.ram_floor_mb, a.interval_s,
                     on_abort=lambda why: (print(f"\nW3_RAM_GUARD_ABORT: {why}", flush=True),
                                           write_all("ABORTED_RAM_GUARD", {"abort_reason": why})))
    mon.start()

    mod = importlib.import_module(SCRIPTS[a.script])
    import navsim as _nv
    import nuplan as _np
    manifest["imported_from"] = {"navsim": _nv.__file__, "nuplan": _np.__file__, "script": mod.__file__,
                                 "fcntl": getattr(sys.modules.get("fcntl"), "__file__", None)}
    assert_tree(_nv.__file__, V11_TREE, "navsim")
    assert_tree(mod.__file__, V11_TREE, SCRIPTS[a.script])
    assert_tree(_np.__file__, NUPLAN_TREE, "nuplan")
    for key in ("nuplan_sha_measured", "nuplan_source_sha_measured"):
        if manifest[key] not in (None, NUPLAN_SHA):
            raise SystemExit(f"⛔ {key} = {manifest[key]}, expected {NUPLAN_SHA}")

    # ---- the one behaviour patch --------------------------------------------- #
    if a.patch_loader:
        from navsim.common import dataloader as _dl
        orig = _dl.MetricCacheLoader._load_metric_cache_paths
        _dl.MetricCacheLoader._load_metric_cache_paths = e1._load_metric_cache_paths_patched
        manifest["patches"].append({
            "name": "MetricCacheLoader._load_metric_cache_paths: token = re.split(r'[\\\\/]', p)[-2]",
            "target": "navsim/common/dataloader.py:170-181 (v1.1 @ 3e8291b)",
            "replacement": f"{E1_WRAPPER}::_load_metric_cache_paths_patched (E1, imported)",
            "original_source_sha16": e1._src_sha(orig),
            "patched_source_sha16": e1._src_sha(e1._load_metric_cache_paths_patched),
            "why": ("save_cache_metadata writes str(WindowsPath) (backslashes); the original "
                    "split('/')[-2] raises IndexError on Windows (control C7)"),
        })

    # ---- observation-only hooks ---------------------------------------------- #
    if hasattr(mod, "pdm_score"):
        _orig_pdm = mod.pdm_score

        def pdm_score_hook(metric_cache, model_trajectory, future_sampling, simulator, scorer):
            t = time.perf_counter()
            out = _orig_pdm(metric_cache=metric_cache, model_trajectory=model_trajectory,
                            future_sampling=future_sampling, simulator=simulator, scorer=scorer)
            dt = time.perf_counter() - t
            try:
                dcs = metric_cache.ego_state.dynamic_car_state
                prod = scorer._multi_metrics.prod(axis=0)
                rec = {
                    "token": Path(str(metric_cache.file_path)).parent.name,
                    "t_wall": round(time.time(), 3), "pdm_score_s": round(dt, 4),
                    "v0_mps": float(dcs.rear_axle_velocity_2d.magnitude()),
                    "progress_raw_m": [float(x) for x in scorer._progress_raw],   # [PDM-Closed, agent]
                    "multiplier_prod": [float(x) for x in prod],
                    "max_compliant_progress_m": float((scorer._progress_raw * prod).max()),
                    "row": {k: float(v) for k, v in out.__dict__.items()},
                }
                if a.record_poses:
                    rec["agent_poses"] = model_trajectory.poses.tolist()
                records["pdm_score_calls"].append(rec)
                jsonl.write(json.dumps({k: v for k, v in rec.items() if k != "agent_poses"}) + "\n")
                jsonl.flush()
            except Exception as e:                                       # noqa: BLE001
                records["pdm_score_calls"].append({"hook_error": repr(e)})
                jsonl.write(json.dumps({"hook_error": repr(e)}) + "\n")
                jsonl.flush()
            return out

        mod.pdm_score = pdm_score_hook
        manifest["hooks"].append(f"{SCRIPTS[a.script]}.pdm_score (observation only)")

    if a.script == "metric_caching":
        from navsim.planning.metric_caching import metric_cache_processor as _mcp
        _orig_cs = _mcp.MetricCacheProcessor.compute_metric_cache

        def cs_hook(self, scenario):
            t = time.perf_counter()
            res = _orig_cs(self, scenario)
            records["cache_scenes"].append({
                "token": scenario.token, "log_name": scenario.log_name,
                "s": round(time.perf_counter() - t, 3),
                "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
                "t_wall": round(time.time(), 3), "ok": res is not None})
            return res

        _mcp.MetricCacheProcessor.compute_metric_cache = cs_hook
        manifest["hooks"].append("MetricCacheProcessor.compute_metric_cache (timing only)")

    write_all("RUNNING")
    sys.argv = [mod.__file__] + overrides
    status, code = "OK", 0
    try:
        mod.main()
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        status = "OK" if code == 0 else f"SYSTEM_EXIT_{code}"
    except BaseException as e:                                        # noqa: BLE001
        status, code = f"EXCEPTION {type(e).__name__}: {e}", 1
        import traceback
        traceback.print_exc()
    finally:
        mon.stop()
        try:
            jsonl.close()
        except Exception:                                             # noqa: BLE001
            pass
        write_all(status)
    print(f"W3_WRAPPER_DONE label={a.label} status={status} wall_s={mon.summary()['wall_s']} "
          f"peak_rss_mb={mon.summary()['peak_rss_mb']} min_avail_mb={mon.summary()['min_system_available_mb']} "
          f"pdm_calls={len(records['pdm_score_calls'])} cache_scenes={len(records['cache_scenes'])}",
          flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
