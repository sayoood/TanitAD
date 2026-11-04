"""mapsel_roll_r9_e2.py -- the UNCHANGED refcv9 MAPSEL roll (FlyWheels ``mapsel_roll.py``) driven for a refcv9 run dir: refcv9 STAGED, DEV windows only, inference seed pinned.

NEW file (E2-S6C, 2026-11-04).  The FlyWheels module is NOT copied or edited: it is exec'd from ``--mapsel-code`` AFTER the pins below are in place.  What it adds:
  (i)   MODEL BUILD -- none needed.  mapsel_roll -> roll_seed_r8_pf -> ``r7_roll`` builds through ``tanitad.eval.refcv7_loader.build_model`` (r7_roll.py:45, :352-355), and that
        loader replays every refcv9 attach (refcv7_loader.py:397-403 _r9rig, :437 _bda, :438 _bc5, :439 _p3f, :440 _r9ph, :441 _bw, :690 _bbf, :694 _pwf; :1006 T1 history on the eval
        dataset).  What the roll does NOT replay is the per-forward STAGING (``_bw.stage`` then ``_bc5.stage``): ``run_battery_r9_e2.install_staging`` is hooked on the first import of
        ``r7_roll`` -- i.e. BEFORE ``mapsel_roll`` captures ``_Base = RR.Refcv6Proxy`` -- and the wrapper REFUSES (before the roll, and again at the first forward) if ``_Base`` is not the
        staging proxy or the proxy's model is not the StagedModel.
  (ii)  DEV WINDOWS -- ``--windows-json`` is read, restricted to the DEV clips of DEV_HOLD_SPLIT.json by ``run_battery_r9_e2.make_dev_window_filter`` (refuses HOLD / empty / count
        mismatch against the split), and the DEV copy (next to ``--trace-out``) replaces it in the child argv.
  (iii) ``--infer-seed N`` (required) is the child's ``--seed`` (the roll SAMPLES; the seed is the INFERENCE seed).  A conflicting ``--seed`` refuses.
  (iv)  NO DEV-BOX DEFAULT -- the 7 env pins of the E2 roll (R8_LAUNCH_TREE, PF_R8_CODE, PF_NF_HOLD, R8_EVAL_KIT, R9E2_REMAP, R9E2_SPLIT, R9E2_PF_CODE) are required;
        ``L2_PF_CODE`` / ``L2_R8_CODE`` are set from R9E2_PF_CODE / PF_R8_CODE, and the dev-box remap ``mapsel_roll`` assigns to ``r8_repoint.REMAP_OVERRIDES`` is IGNORED (module pinned to
        R9E2_REMAP; the attempt is recorded).  The trace's meta gains an ``r9e2`` block (infer seed, dev filter census, stage census, remap pin).
UNVERIFIED end to end: no real K2 / K0 roll exists; ``main`` is exercised only against fake modules (test_r9e2_s6c.py).

    python mapsel_roll_r9_e2.py --mapsel-code <FlyWheels .../2026-10-25-refcv9-mapsel/code> --infer-seed 0 --trace-out T.npz --rast-dir R --map-gt-root G --require-seal \
        -- --ckpt <run>/model_ema.pt --config <run>/config.json --dump-dir D --windows-json W.json --device cuda --out-json O.json --os-only
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_battery_r9_e2 as R9  # noqa: E402  (the E2 roll's pins / DEV filter / staging proxy)

STAGE_CLS_NAME = "StagingRefcv6Proxy"
OWN_FLAGS = ("--mapsel-code", "--split", "--infer-seed")


def _sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def split_argv(argv):
    """argv -> (own dict, mapsel's own flags before `--`, the battery child argv after `--`).  Our three flags are stripped; the rest reaches mapsel_roll unchanged."""
    argv = list(argv)
    if "--" not in argv:
        raise SystemExit("[mapsel-r9e2] usage: mapsel_roll_r9_e2.py --mapsel-code D --infer-seed N [mapsel opts] -- <battery child argv>")
    i = argv.index("--")
    pre, child = argv[:i], argv[i + 1:]
    own = {}
    for f in OWN_FLAGS:
        if f in pre:
            j = pre.index(f)
            if j + 1 >= len(pre):
                raise SystemExit(f"[mapsel-r9e2] {f} needs a value")
            own[f], pre = pre[j + 1], pre[:j] + pre[j + 2:]
    for f in ("--mapsel-code", "--infer-seed"):
        if f not in own:
            raise SystemExit(f"[mapsel-r9e2] refusing: {f} is required")
    try:
        own["--infer-seed"] = int(own["--infer-seed"])
    except ValueError:
        raise SystemExit(f"[mapsel-r9e2] --infer-seed must be an integer, got {own['--infer-seed']!r}")
    return own, pre, child


def pin_child_seed(child, seed):
    """The child's ``--seed`` IS the inference seed: set it, or REFUSE a conflicting one."""
    child = list(child)
    if "--seed" in child:
        j = child.index("--seed")
        if j + 1 >= len(child) or str(child[j + 1]) != str(seed):
            raise SystemExit(f"[mapsel-r9e2] refusing: child --seed {child[j + 1] if j + 1 < len(child) else None!r} conflicts with --infer-seed {seed}")
        return child
    return child + ["--seed", str(seed)]


def dev_windows_arg(child, trace_out, dev, hold, split, census):
    """Replace the child's ``--windows-json`` by its DEV restriction (written next to ``trace_out``); the E2 roll's own filter does the checks."""
    child = list(child)
    if "--windows-json" not in child or child.index("--windows-json") + 1 >= len(child):
        raise SystemExit("[mapsel-r9e2] refusing: the child argv has no --windows-json (nothing to restrict to DEV)")
    j = child.index("--windows-json") + 1
    full = json.loads(Path(child[j]).read_text(encoding="utf-8"))
    kept = R9.make_dev_window_filter(lambda *_: full, dev, hold, split, census)(None, None)
    out = Path(trace_out).resolve().parent / "windows_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept), encoding="utf-8")
    child[j] = str(out)
    census["windows_json_dev"] = str(out)
    return child


class _PinnedModule(types.ModuleType):
    def __setattr__(self, k, v):
        pins = self.__dict__.get("_r9e2_pins", {})
        if k in pins:
            self.__dict__.setdefault("_r9e2_attempts", []).append((k, str(v)))
            return
        super().__setattr__(k, v)


def pin_module_attr(mod, name, value):
    """After this, ``mod.<name> = x`` is IGNORED (recorded in ``mod._r9e2_attempts``): mapsel_roll's dev-box remap assignment cannot override the E2 remap."""
    mod.__dict__.setdefault("_r9e2_pins", {})[name] = value
    mod.__class__ = _PinnedModule
    mod.__dict__[name] = value
    return mod


def stage_proxy_installed(base) -> bool:
    return any(c.__name__ == STAGE_CLS_NAME for c in getattr(base, "__mro__", ()))


def require_staging(M, stager):
    """REFUSE unless mapsel_roll's captured ``_Base`` is the staging proxy and the stager carries the trainer's order (_bw then _bc5)."""
    if not stage_proxy_installed(getattr(M, "_Base", None)):
        raise SystemExit("[mapsel-r9e2] refusing: mapsel_roll._Base is not the staging proxy -- the refcv9 stage hook did not fire before mapsel_roll captured Refcv6Proxy")
    if stager is None or [lab for lab, _ in stager.mods] != ["_bw", "_bc5"]:
        raise SystemExit(f"[mapsel-r9e2] refusing: the stager is {None if stager is None else [lab for lab, _ in stager.mods]}, need ['_bw', '_bc5']")


def guard_first_call(M):
    """Wrap ``MapSelProxy.__call__``: at the FIRST forward the proxy's model must be the StagedModel, else refuse (a missing stage must not roll silently)."""
    proxy, orig, seen = M.MapSelProxy, M.MapSelProxy.__call__, {"n": 0}

    def call(self, *a, **kw):
        if seen["n"] == 0 and not isinstance(self.__dict__.get("model"), R9.StagedModel):
            raise SystemExit("[mapsel-r9e2] refusing at the first forward: the proxy's model is not the StagedModel (a refcv9 stage would be skipped)")
        seen["n"] += 1
        return orig(self, *a, **kw)
    proxy.__call__ = call


def load_mapsel(path):
    spec = importlib.util.spec_from_file_location("mapsel_roll", str(path))
    M = importlib.util.module_from_spec(spec)
    sys.modules["mapsel_roll"] = M
    spec.loader.exec_module(M)
    return M


def main(argv=None) -> int:
    own, pre, child = split_argv(sys.argv[1:] if argv is None else argv)
    pins = R9.require_env()
    dev, hold, split = R9.load_dev_hold(pins["R9E2_SPLIT"])
    code = Path(own["--mapsel-code"])
    if not (code / "mapsel_roll.py").is_file():
        raise SystemExit(f"[mapsel-r9e2] {code}/mapsel_roll.py not found")
    trace_out = R9._val("--trace-out", pre)
    if not trace_out:
        raise SystemExit("[mapsel-r9e2] mapsel opts carry no --trace-out")
    os.environ["L2_PF_CODE"], os.environ["L2_R8_CODE"] = pins["R9E2_PF_CODE"], pins["PF_R8_CODE"]
    RP = R9.pin_repoint()
    pin_module_attr(RP, "REMAP_OVERRIDES", Path(pins["R9E2_REMAP"]))
    census = {}
    child = pin_child_seed(dev_windows_arg(child, trace_out, dev, hold, split, census), own["--infer-seed"])
    R9.child_stage_hook()                                       # BEFORE mapsel_roll imports r7_roll (through roll_seed_r8_pf)
    mp = code / "mapsel_roll.py"
    sys.argv = [str(mp)] + pre + ["--"] + child
    M = load_mapsel(mp)
    stager = getattr(sys.modules.get("r7_roll"), "R9E2_STAGER", None)
    require_staging(M, stager)
    guard_first_call(M)
    M.REMAP_REC.update(file=pins["R9E2_REMAP"], sha256=_sha256_file(pins["R9E2_REMAP"]), r9e2_pinned=True,
                       mapsel_devbox_remap_ignored=[a[1] for a in getattr(RP, "_r9e2_attempts", [])])
    orig_save = M.save

    def save(final, extra=None):
        return orig_save(final, {**(extra or {}), "r9e2": {"infer_seed": own["--infer-seed"], "split": pins["R9E2_SPLIT"], "dev_filter": census, "stage": stager.census(),
                                                            "remap_pin": pins["R9E2_REMAP"], "wrapper": Path(__file__).name}})
    M.save = save
    rc = M.main()
    if rc in (0, None) and stager.n_forwards == 0:
        raise SystemExit("[mapsel-r9e2] the roll ended rc 0 but NO forward was staged")
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
