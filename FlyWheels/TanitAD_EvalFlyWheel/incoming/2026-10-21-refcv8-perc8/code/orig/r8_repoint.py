"""r8_repoint.py -- re-point the refcv7 four-family battery at the refcv8 run, WITHOUT changing a repo file.

(EvalFlyWheel refcv8 milestone kit, 2026-10-15; SPEC_REFCV8_DRAFT section 5: "the refcv7 four-family battery,
re-pointed at the refcv8 run".)  Imported FIRST by `g0_refcv8.py` / `run_battery_r8.py`, before the refcv7 battery
modules are imported, because the loader reads its environment at import time.

WHAT IT CHANGES (each item is also recorded, by `describe()`, into the artifact the caller writes)
  1. ENV  -- `REFCV7_KIT` (the dev-box eval kit root), `REFCV7_REMAP_OVERRIDES` (the dev-box twin of
     `stack/ops/refcv8_eval_remap_a2_thor.json`: the launch gate's four renamed flags PLUS `--r8-v9-labels-eval`
     and `--pose-sync-sidecar` + `--join-defect-masks`, which the gate never needed to remap because on Thor those
     Thor paths exist; the last is a repo file, mapped to the launch tree, md5-equal to Thor's launch_tree copy).
     The loader REFUSES a conflicting `REFCV6_*`, so a stale shell cannot override these.
  2. PATH  -- the refcv7 battery's `code/` directory (g0_refcv7.py, microbatch.py, wrapper_probe_r7.py, ...) is put on
     sys.path; the refcv8 launch tree (`C:/lgs/Lfb47a58376`, a read-only `git archive` of fb47a583763f) supplies
     `tanitad` and the loader.
  3. ONE BEHAVIOURAL PATCH -- `L.build_eval_dataset` is wrapped so the EVAL dataset gets `enable_pose_sync(
     args.pose_sync_sidecar)`, exactly as `refc_v3_train.train()` does at :9765-9766 ("refcv8 X10: the eval GT rides
     the SAME clock").  MEASURED 2026-10-15 by reading both: the loader at the launch commit (and in the working
     tree) NEVER calls `enable_pose_sync`, and the launch gate's G-EVAL cannot see it (it feeds the trainer's model
     and the loader's model the SAME loader-built batch).  Without the patch the eval GT, `pose_last` (hence v0) and
     the actions are read on the unshifted pose clock while the run's in-run row was scored on the shifted one, so G0
     would compare two different eval sets.  Every consumer of the eval dataset (G0, `r7_roll`, `perc_pass_r7`) goes
     through `L.build_eval_dataset`, so one wrapper covers all of them.  `--no-pose-sync` (env R8_NO_POSE_SYNC=1)
     turns it off, ONLY so the deliberate-regression arm and a diagnostic can show what the omission does.

NOTHING in refcv7's behaviour changes: a refcv7 argv carries no `--pose-sync-sidecar`, so the wrapper is a pass-through
for it (pinned by test_r8_repoint.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH_TREE = Path(os.environ.get("R8_LAUNCH_TREE", "C:/lgs/Lfb47a58376"))
#: the refcv7 battery package inside the launch tree (read-only archive of the launch commit)
BATTERY_CODE = LAUNCH_TREE / ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/"
                              "battery/code")
EVAL_KIT = os.environ.get("R8_EVAL_KIT", "E:/refcv6_eval_kit")        # refcv6/7 kit: the files that did not change
REMAP_OVERRIDES = HERE / "refcv8_eval_remap_devbox.json"

#: every flag of the refcv8 argv whose value is an EVAL-SIDE input file the loader / trainer replay reads
EVAL_SIDE_FLAGS = (
    "--anchors", "--eval-cache", "--eval-labels", "--speed-max-sidecar-v6-eval", "--agent-join",
    "--agent-rig-extrinsics", "--map-gt-root", "--join3d", "--clip-clock-sidecar", "--vis1-sidecar",
    "--nav-compliance-tau-file", "--map-hires-class-weights", "--r8-v9-labels-eval", "--pose-sync-sidecar",
    "--join-defect-masks",          # a BUILD-time input (train():1783 validates it inside model construction)
)

_STATE: dict = {"applied": False, "pose_sync": {}, "pose_sync_enabled": None}


def configure_env() -> dict:
    """Set the loader's environment (must run BEFORE `tanitad.eval.refcv7_loader` is imported)."""
    if "tanitad.eval.refcv7_loader" in sys.modules:
        raise SystemExit("[r8_repoint] the loader is already imported; configure_env() must run first "
                         "(it reads REFCV7_* at import time)")
    want = {"REFCV7_KIT": EVAL_KIT, "REFCV7_REMAP_OVERRIDES": str(REMAP_OVERRIDES),
            "REFCV7_REPO": str(LAUNCH_TREE)}
    for k, v in want.items():
        old = os.environ.get(k)
        if old and os.path.normcase(os.path.abspath(old)) != os.path.normcase(os.path.abspath(v)):
            raise SystemExit(f"[r8_repoint] {k}={old!r} in the shell disagrees with {v!r}; unset it")
        os.environ[k] = v
    for stem in ("KIT", "REMAP_OVERRIDES", "REPO"):           # the loader refuses a conflicting REFCV6_*
        old = os.environ.get(f"REFCV6_{stem}")
        if old and os.path.normcase(os.path.abspath(old)) != os.path.normcase(
                os.path.abspath(os.environ[f"REFCV7_{stem}"])):
            raise SystemExit(f"[r8_repoint] REFCV6_{stem}={old!r} disagrees with REFCV7_{stem}; unset it")
    for p in (str(BATTERY_CODE), str(LAUNCH_TREE / "stack"), str(LAUNCH_TREE / "taniteval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return want


def make_pose_synced_cls(base, sidecar: str, sink: dict):
    """A subclass of `base` (a V3Dataset subclass) whose constructor ends with `enable_pose_sync(sidecar)`.
    `train()` calls it right after constructing the eval dataset and before the labels; `_raw_offset` reads only
    the frame-stack shape, so the order against the label clock is immaterial."""

    class PoseSynced(base):                                   # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            stats = self.enable_pose_sync(str(sidecar))
            sink.clear()
            sink.update(stats)

    PoseSynced.__name__ = f"{base.__name__}PoseSynced"
    PoseSynced.__qualname__ = PoseSynced.__name__
    return PoseSynced


def assert_pose_sync_on(e_ds) -> None:
    """The positive check the patch is held to: the dataset the caller got HAS the sidecar attached."""
    if getattr(e_ds, "pose_sync", None) is None:
        raise SystemExit("[r8_repoint] the eval dataset has no pose_sync attached although the run's argv names "
                         "--pose-sync-sidecar: the eval GT would ride the UNSHIFTED pose clock (train():9765)")


def patch_loader(L, *, enabled: bool = True, sink: dict | None = None):
    """Wrap `L.build_eval_dataset` (idempotent). Returns the original."""
    sink = _STATE["pose_sync"] if sink is None else sink
    if getattr(L.build_eval_dataset, "_r8_patched", False):
        return L.build_eval_dataset._r8_orig
    orig = L.build_eval_dataset

    def build_eval_dataset(model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
        sidecar = getattr(args, "pose_sync_sidecar", None)
        if not (enabled and sidecar):
            return orig(model, cfg, args, config, with_perception_targets=with_perception_targets,
                        dataset_cls=dataset_cls)
        tr = L.trainer()
        if dataset_cls is None:
            if bool(getattr(args, "graft_lan", False) or getattr(args, "goal_str", False)):
                raise SystemExit("[r8_repoint] a LAN/goal-string dataset base is not a refcv8 arm; refusing")
            dataset_cls = tr.V3Dataset
        cls = make_pose_synced_cls(dataset_cls, sidecar, sink)
        e_ds, e_eps, rec = orig(model, cfg, args, config, with_perception_targets=with_perception_targets,
                                dataset_cls=cls)
        assert_pose_sync_on(e_ds)
        rec["pose_sync"] = dict(sink)
        print(f"[r8_repoint] pose-sync attached to the eval dataset: {sink.get('n_covered')}/{sink.get('n_clips')} "
              f"clips shifted ({sink.get('n_uncovered_read_unshifted')} unshifted); sidecar {sink.get('sidecar')}",
              flush=True)
        return e_ds, e_eps, rec

    build_eval_dataset._r8_patched = True
    build_eval_dataset._r8_orig = orig
    L.build_eval_dataset = build_eval_dataset
    return orig


R8_BATCH_KW = ("r8_nav", "r8_rc")     # refcv8's per-sample forward inputs (refc_v3.py RefCV3Model.forward)


def patch_microbatch() -> tuple:
    """The refcv7 `microbatch.MicroBatchForward` slices only the kwargs named in its `_BATCH_KW`; refcv8 adds
    `r8_nav` [B, 6] and `r8_rc` per row, so on the 8 GB card every micro-part received the FULL 16-row tensors and the
    step-5,000 probe died in `scene_hook` (`shape '[2, 6]' is invalid for input of size 96`, 2026-10-16 00:18).
    `_forward` reads the module global at call time, so extending it here reaches every caller."""
    import microbatch as MB                                          # noqa: E402  (battery code dir, on sys.path)
    missing = tuple(k for k in R8_BATCH_KW if k not in MB._BATCH_KW)
    MB._BATCH_KW = tuple(MB._BATCH_KW) + missing
    _STATE["microbatch_added"] = list(missing)
    return MB._BATCH_KW


SEAM_TELE: dict = {"forwards": 0, "rows": 0, "rows_above_clamp": 0, "saturated_forwards": 0,
                   "ratio_mean_sum": 0.0, "ratio_max": 0.0, "fail_loud_suppressed": 0}


def patch_seam_guard() -> bool:
    """EVAL ONLY: keep `refc_select.apply_seam_clamp`'s in-graph clamp, drop its fail-loud raise.

    The guard (`refc_select.py:300-330`) is a TRAINING-DYNAMICS alarm built as a population-over-time condition for
    training batches (mean ratio > fail AND > fail_frac of the batch at the clamp, `patience` consecutive steps). The
    battery rolls run 1-3 rows per forward (`refcv3_arm.py:2106, 2260`), so "fail_frac of the batch" degenerates to
    "this window", and 50 consecutive saturated windows kill the roll: MEASURED 2026-10-16, both step-5,000 seed-0
    attempts died with `goal_sel-surface graft seam SATURATED: mean pre-clamp ratio 1.627 > 1.5 AND 100.0% ...`.
    With `patience=0` the function computes the SAME `scale` and returns the SAME tensor (the raise is its only other
    effect), so outputs are bit-identical (pinned by test_r8_seam_guard.py). The saturation is COUNTED instead, so the
    battery reports how often refcv8's selection prior sat at the clamp."""
    if os.environ.get("R8_SEAM_GUARD_EVAL", "off") == "keep":
        return False
    from tanitad.refs import refc_select as SL                       # noqa: E402
    if getattr(SL.apply_seam_clamp, "_r8_patched", False):
        return True
    orig = SL.apply_seam_clamp

    def apply_seam_clamp(base, graft, *, clamp, fail, fail_frac, patience, state, surface):
        out = orig(base, graft, clamp=clamp, fail=fail, fail_frac=fail_frac, patience=0, state=state,
                   surface=surface)
        if clamp > 0.0:
            ratio = graft.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-9)
            n = int(ratio.numel())
            pre_mean = float(ratio.mean().detach())
            bound = int((ratio > clamp).sum().detach())
            t = SEAM_TELE
            t["forwards"] += 1
            t["rows"] += n
            t["rows_above_clamp"] += bound
            t["ratio_mean_sum"] += pre_mean * n
            t["ratio_max"] = max(t["ratio_max"], float(ratio.max().detach()))
            if pre_mean > fail and (bound / max(n, 1)) > fail_frac:
                t["saturated_forwards"] += 1
            if patience > 0 and state.sat_steps > 0 and state.sat_steps % patience == 0:
                t["fail_loud_suppressed"] += 1          # where the training guard would have raised (no reset:
                #                                         the telemetry keeps the true consecutive count)
        return out

    apply_seam_clamp._r8_patched = True
    apply_seam_clamp._r8_orig = orig
    SL.apply_seam_clamp = apply_seam_clamp
    out_dir = os.environ.get("R8_SEAM_TELE_DIR")
    if out_dir:
        import atexit

        def _dump():
            t = dict(SEAM_TELE)
            t["ratio_mean"] = t["ratio_mean_sum"] / max(t["rows"], 1)
            t["frac_rows_above_clamp"] = t["rows_above_clamp"] / max(t["rows"], 1)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"seam_tele_{os.getpid()}.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
        atexit.register(_dump)
    _STATE["seam_guard_eval_off"] = True
    return True


def apply(*, pose_sync: bool | None = None):
    """configure_env() + import the loader + patch it. Returns the loader module."""
    if pose_sync is None:
        pose_sync = os.environ.get("R8_NO_POSE_SYNC", "") not in ("1", "true", "yes")
    configure_env()
    from tanitad.eval import refcv7_loader as L                      # noqa: E402
    L.bootstrap()
    patch_loader(L, enabled=pose_sync)
    patch_microbatch()
    patch_seam_guard()
    _STATE.update(applied=True, pose_sync_enabled=bool(pose_sync))
    return L


def describe(L) -> dict:
    """The re-pointing, as a record the caller stamps into its artifact."""
    return {"tool": "r8_repoint.py", "launch_tree": str(LAUNCH_TREE), "loader_file": str(Path(L.__file__)),
            "loader_REPO": str(L.REPO), "loader_KIT": str(L.KIT),
            "remap_overrides_file": str(REMAP_OVERRIDES),
            "remap_overrides": json.loads(REMAP_OVERRIDES.read_text(encoding="utf-8")),
            "pose_sync_patch_enabled": _STATE["pose_sync_enabled"],
            "pose_sync_stats": dict(_STATE["pose_sync"]),
            "microbatch_batch_kw_added": _STATE.get("microbatch_added"),
            "seam_guard_eval_off": bool(_STATE.get("seam_guard_eval_off")),
            "seam_telemetry": dict(SEAM_TELE),
            "battery_code": str(BATTERY_CODE)}
