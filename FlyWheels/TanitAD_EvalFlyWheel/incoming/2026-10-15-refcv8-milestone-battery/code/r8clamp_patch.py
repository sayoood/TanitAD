"""r8clamp_patch.py -- run the refcv8 battery (`run_battery_r8.py` and its roll child) against the CLAMP-CARRYING tree `tree_de4e15b`, with
`cfg.plan_vmax_clamp` ON or OFF.   (EvalFlyWheel, AMENDMENT A-R8-CLAMP-1, 2026-10-18.)   NEW file; no existing file is edited.

PRECEDENT: `r7m_patch.py` (the same battery pointed at a refcv7 checkpoint).  Same trick: monkey-patch the `r8_repoint` / `r8_battery_patch`
module attributes BEFORE the battery runs, in the parent AND in the roll child (patches are per process).

WHAT `install(mode)` DOES   (mode = "on" | "off";  env `R8_CLAMP_MODE` carries it to the child)
  0. REFUSES unless `r8clamp_common.check_provenance()` passes: PROVENANCE.txt names de4e15b19a39..., every load-bearing file is LF-normalised
     byte-equal to `git cat-file blob de4e15b:<path>` (both reads non-empty, object name 40 chars), and the clamp markers are IN the files.
  1. `r8_repoint.LAUNCH_TREE := tree_de4e15b` (and env R8_LAUNCH_TREE) so `tanitad` / `taniteval` / the loader come from the clamp tree;
     `r8_repoint.BATTERY_CODE := the fb47a58 battery code` (the export has no FlyWheels dir; that code is what rolled the banked dumps).
  2. `r8_repoint.patch_loader` is wrapped.  After the original patch it
       a. asserts `tanitad` and the loader were imported FROM the tree (a wrong tree would make OFF a trivial "pass" and ON a no-op);
       b. POSE CLOCK.  The de4e15b loader ALREADY enables the pose shift (`refcv7_loader.py:801-802`: whenever the argv names
          `--pose-sync-sidecar`), which fb47a58's loader never did -- so `R8_NO_POSE_SYNC=1` alone would NOT give the uncorrected clock the banked
          `_unc` trees were rolled on.  Under R8_NO_POSE_SYNC=1 the sidecar is blanked for the dataset build and the result is asserted
          UNSHIFTED; otherwise it is asserted SHIFTED.  (`enable_pose_sync` is idempotent, so the r8 wrapper + the loader both enabling it is harmless);
       c. wraps `L.build_model`: ON sets `model.cfg.plan_vmax_clamp = True` AFTER the build (the argv carries no `--plan-vmax-clamp`; the
          trainer's own build-time refusals are replicated: needs `max_speed_onehot_v6`, refuses `refcv7_select`, needs `hier`), OFF asserts it is False;
       d. registers a forward hook on the model.  ON: every forward MUST emit `traj_precap` and `plan_cap_binding` (else RuntimeError = REFUSE) and
          the clamp must not be INERT (after 30 rows a limit that is +inf everywhere is a refusal); OFF: neither key may be present;
       e. ON: appends the clamp keys to `r7_roll.EXTRA_ROW_KEYS`, so `refcv6_extras.npz` carries `<cond>.traj_precap`, `.plan_cap_binding`,
          `.plan_v_lim_ms`, ... beside the clamped `<cond>.traj`.
  3. `r8_battery_patch.rewrite_child_cmd`: `roll_seed_r8.py` -> `roll_seed_r8clamp.py` (the child that calls install() first).

NOTHING here changes a number in OFF mode except which tree runs; ON mode changes exactly `out["traj"]` (inference only; the training path is
`not self.training`-gated in the tree).  K2 (OFF parity vs a banked dump) and K4 (ON vs the post-hoc clamp) are what test that, on the GPU, later.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r8clamp_common as C  # noqa: E402

CHILD = HERE / "roll_seed_r8clamp.py"
INERT_AFTER_ROWS = 30
_STATE: dict = {"installed": False, "mode": None, "provenance": None, "pose_clock": None, "build": None, "child_rewrites": [],
                "extra_row_keys_added": [], "loader_file": None, "tanitad_file": None}
STATS: dict = {"mode": None, "n_forwards": 0, "n_rows": 0, "n_rows_finite_limit": 0, "n_binding": 0, "n_v0_over_limit": 0,
               "n_forwards_with_clamp_keys": 0, "n_forwards_without_clamp_keys": 0}


def _under(p, root) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def mode_from_env() -> str:
    m = os.environ.get("R8_CLAMP_MODE", "")
    if m not in ("on", "off"):
        raise SystemExit(f"[r8clamp] R8_CLAMP_MODE must be 'on' or 'off' (got {m!r}); the wrapper sets it")
    return m


# ------------------------------------------------------------------------------------------------ the model hooks
def make_forward_hook(mode: str):
    """The per-forward assertion.  ON: the clamp keys must be there (and the limit not +inf everywhere).  OFF: they must be absent."""
    def hook(module, args, kwargs, out):
        import torch
        if not isinstance(out, dict):
            raise RuntimeError(f"[r8clamp] the model returned {type(out).__name__}, not a dict")
        STATS["n_forwards"] += 1
        has = [k for k in C.ON_REQUIRED_OUT if k in out]
        if mode == "off":
            if has or "plan_cap_binding" in out:
                raise RuntimeError(f"[r8clamp] OFF mode but the forward emitted {has}: the clamp ran -- K2 would not be a flag-OFF read")
            return
        missing = [k for k in C.ON_REQUIRED_OUT if k not in out]
        if missing:
            STATS["n_forwards_without_clamp_keys"] += 1
            raise RuntimeError(f"[r8clamp] ON mode but the forward did NOT emit {missing}: the clamp is not active "
                               f"(cfg.plan_vmax_clamp={getattr(getattr(module, 'cfg', None), 'plan_vmax_clamp', None)!r}); REFUSING to roll")
        STATS["n_forwards_with_clamp_keys"] += 1
        bind = out["plan_cap_binding"]
        STATS["n_rows"] += int(bind.numel())
        STATS["n_binding"] += int(bind.sum())
        lim = out.get("plan_v_lim_ms")
        if torch.is_tensor(lim):
            STATS["n_rows_finite_limit"] += int(torch.isfinite(lim).sum())
        ov = out.get("plan_cap_v0_over_limit")
        if torch.is_tensor(ov):
            STATS["n_v0_over_limit"] += int(ov.sum())
        if STATS["n_rows"] >= INERT_AFTER_ROWS and STATS["n_rows_finite_limit"] == 0:
            raise RuntimeError(f"[r8clamp] ON mode: {STATS['n_rows']} rows seen and EVERY limit is +inf -- the clamp is built and INERT "
                               f"(no ceiling channel reached it); REFUSING")
    return hook


def _enable_or_assert_off(model, cfg, mode: str) -> dict:
    mcfg = model.cfg
    before = bool(getattr(mcfg, "plan_vmax_clamp", False))
    rec = {"mode": mode, "cfg_plan_vmax_clamp_after_build": before, "cfg_is_model_cfg": cfg is mcfg}
    if mode == "off":
        if before:
            raise RuntimeError("[r8clamp] OFF mode but the BUILT cfg already has plan_vmax_clamp=True (the argv carries --plan-vmax-clamp)")
    else:
        # the trainer's own build-time refusals (refc_v3.py:1132-1143), replicated because the flag is set AFTER the build
        if not bool(getattr(mcfg, "max_speed_onehot_v6", False)):
            raise RuntimeError("[r8clamp] plan_vmax_clamp without max_speed_onehot_v6: every limit would be +inf (built and INERT)")
        if bool(getattr(mcfg, "refcv7_select", False)):
            raise RuntimeError("[r8clamp] plan_vmax_clamp with refcv7_select: the scorer's pick replaces the emitted plan after the clamp")
        if not bool(getattr(mcfg, "hier", False)):
            raise RuntimeError("[r8clamp] plan_vmax_clamp on a FLAT build: the ceiling channel is read through the hierarchy hook")
        if getattr(model, "training", False):
            raise RuntimeError("[r8clamp] the model is in training mode; the clamp is inference-only")
        mcfg.plan_vmax_clamp = True
        if cfg is not mcfg and hasattr(cfg, "plan_vmax_clamp"):
            cfg.plan_vmax_clamp = True
        rec["cfg_plan_vmax_clamp_now"] = bool(getattr(mcfg, "plan_vmax_clamp", False))
        if not rec["cfg_plan_vmax_clamp_now"]:
            raise RuntimeError("[r8clamp] could not set cfg.plan_vmax_clamp")
    model.register_forward_hook(make_forward_hook(mode), with_kwargs=True)
    rec["forward_hook"] = "registered (ON: traj_precap + plan_cap_binding required each forward; OFF: absent)"
    return rec


# ------------------------------------------------------------------------------------------------ the loader hook
def _post_loader(L, pose_enabled: bool, mode: str) -> None:
    import tanitad
    tree = C.TREE
    if not _under(tanitad.__file__, tree / "stack") or not _under(L.__file__, tree / "stack"):
        raise SystemExit(f"[r8clamp] REFUSING: tanitad is {tanitad.__file__} / the loader is {L.__file__}, not under {tree}/stack "
                         f"(a wrong tree makes OFF a trivial pass and ON a no-op)")
    _STATE["loader_file"], _STATE["tanitad_file"] = str(L.__file__), str(tanitad.__file__)
    # -- the pose clock ------------------------------------------------------------------------------------------------ #
    cur = L.build_eval_dataset
    if not getattr(cur, "_r8clamp_pose", False):
        def build_eval_dataset(model, cfg, args, config, **kw):
            saved = getattr(args, "pose_sync_sidecar", None)
            if saved and not pose_enabled:
                args.pose_sync_sidecar = None          # the tree's own loader would enable the shift whenever the argv names a sidecar
            try:
                res = cur(model, cfg, args, config, **kw)
            finally:
                if saved:
                    args.pose_sync_sidecar = saved
            attached = getattr(res[0], "pose_sync", None) is not None
            if saved and pose_enabled and not attached:
                raise SystemExit("[r8clamp] corrected-clock pass but the eval dataset has no pose shift attached")
            if (not pose_enabled) and attached:
                raise SystemExit("[r8clamp] R8_NO_POSE_SYNC=1 (standalone / UNCORRECTED clock) but the eval dataset HAS the pose shift attached")
            _STATE["pose_clock"] = {"policy": ("corrected: shift attached" if pose_enabled else "standalone/uncorrected: sidecar blanked, shift NOT attached"),
                                    "argv_names_sidecar": bool(saved), "shift_attached": attached}
            return res
        build_eval_dataset._r8clamp_pose = True
        L.build_eval_dataset = build_eval_dataset
    # -- build_model: the flag + the hook ---------------------------------------------------------------------------- #
    cb = L.build_model
    if not getattr(cb, "_r8clamp_build", False):
        def build_model(*a, **k):
            m, c, ar, r = cb(*a, **k)
            r["r8clamp"] = _enable_or_assert_off(m, c, mode)
            _STATE["build"] = r["r8clamp"]
            return m, c, ar, r
        build_model._r8clamp_build = True
        L.build_model = build_model
    # -- ON: the extras carry the clamp keys -------------------------------------------------------------------------- #
    if mode == "on":
        import r7_roll as RR
        add = [k for k in C.CLAMP_KEYS if k not in RR.EXTRA_ROW_KEYS]
        RR.EXTRA_ROW_KEYS = tuple(RR.EXTRA_ROW_KEYS) + tuple(add)
        _STATE["extra_row_keys_added"] = list(add)


def install(mode: str | None = None) -> dict:
    mode = mode or mode_from_env()
    if mode not in ("on", "off"):
        raise SystemExit(f"[r8clamp] mode must be on|off, got {mode!r}")
    if _STATE["installed"]:
        if _STATE["mode"] != mode:
            raise SystemExit(f"[r8clamp] already installed as {_STATE['mode']!r}")
        return _STATE
    os.environ["R8_CLAMP_MODE"] = mode
    STATS["mode"] = mode
    _STATE["provenance"] = C.check_provenance()                    # raises ProvenanceError (SystemExit) -> the run never starts
    if not (C.BATTERY_CODE / "run_battery_r7.py").is_file():
        raise SystemExit(f"[r8clamp] REFUSING: the battery code {C.BATTERY_CODE} has no run_battery_r7.py")
    old = os.environ.get("R8_LAUNCH_TREE")
    if old and os.path.normcase(os.path.abspath(old)) != os.path.normcase(os.path.abspath(str(C.TREE))):
        raise SystemExit(f"[r8clamp] R8_LAUNCH_TREE={old!r} disagrees with the clamp tree {C.TREE}; unset it")
    os.environ["R8_LAUNCH_TREE"] = str(C.TREE)
    import r8_repoint
    r8_repoint.LAUNCH_TREE = Path(C.TREE)                         # read at CALL time by configure_env() / describe()
    r8_repoint.BATTERY_CODE = Path(C.BATTERY_CODE)
    orig_patch_loader = r8_repoint.patch_loader

    def patch_loader(L, *, enabled: bool = True, sink=None):
        r = orig_patch_loader(L, enabled=enabled, sink=sink)
        _post_loader(L, bool(enabled), mode)
        return r

    r8_repoint.patch_loader = patch_loader                       # apply() resolves it as a module global at call time
    import r8_battery_patch as BP
    orig = BP.rewrite_child_cmd

    def rewrite_child_cmd(cmd, **kw):
        out = [str(c) for c in orig(cmd, **kw)]
        for i, c in enumerate(out):
            if c.replace("\\", "/").endswith("/roll_seed_r8.py"):
                out[i] = str(CHILD)
                _STATE["child_rewrites"].append({"from": c, "to": str(CHILD), "by": "r8clamp_patch", "mode": mode})
                break
        return out

    BP.rewrite_child_cmd = rewrite_child_cmd
    _STATE.update(installed=True, mode=mode)
    sd = os.environ.get("R8_CLAMP_STATS_DIR")
    if sd:
        def _dump():
            Path(sd).mkdir(parents=True, exist_ok=True)
            (Path(sd) / f"clamp_stats_{mode}_{os.getpid()}.json").write_text(json.dumps(describe(), indent=1, default=str), encoding="utf-8")
        atexit.register(_dump)
    return _STATE


def describe() -> dict:
    return json.loads(json.dumps({"tool": "r8clamp_patch.py", "state": _STATE, "forward_stats": STATS}, default=str))
