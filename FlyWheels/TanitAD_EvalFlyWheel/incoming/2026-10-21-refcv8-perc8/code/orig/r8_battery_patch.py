"""r8_battery_patch.py -- the STAGE-2 (post-G0) re-pointing of the refcv7 four-family battery at refcv8.

(EvalFlyWheel refcv8 milestone kit, 2026-10-15.)  `r8_repoint.py` (G0, UNCHANGED, imported by the queued G0 chain) re-points
the loader.  This module adds what the rolls / panels need on top of it, WITHOUT editing a repo file and WITHOUT touching
`r8_repoint.py` (a running G0 chain imports that file at every python start):

  D1. BASELINE PATHS.  `r7_panel.BASELINES` names `C:/Users/Admin/refcv5cmp/out/...`; that tree was MOVED by the 2026-10-15
      C: cleanup to `D:/Projects/TanitAD-artifacts/refcv5cmp/out/...` (MEASURED: the C: path does not exist; the S2 window
      source `b_refcv4b` is among them, so `run_battery_r7._run` dies at `RR.s2_window_list`).  `fix_baselines` re-points
      the dict IN PLACE, by content (manifest + ep000.npz + decisions/ep000.npz), and REFUSES if a baseline is nowhere.
  D2. REFCV7-50,400 AS A PAIRED BASELINE.  The battery already pairs banked dumps through `build_panel_dump(baselines=)` and
      `build_s6_dump(baselines=)` (GT bit-equality and model-free-arm equality are asserted there).  `add_refcv7_baselines`
      registers `b_refcv7_50400_s{0,1}` = `<refcv7 battery>/dump_s{0,1}` as extra baselines and extends `PAIRS`, so
      refcv8-vs-refcv7 `os` is produced by the battery's OWN paired machinery, per inference seed, on the SAME windows.
      Registering the label in `P.TIERS` is REQUIRED: `build_s6_dump` evaluates `TIERS[arm]` for every baseline arm.
  D3. CHILD PROCESSES.  `run_battery_r7._run` spawns `roll_seed_r7.py` as a child process from the refcv7 battery dir: the
      child would NOT carry the pose-sync patch (sys.path / monkey-patches are per process).  `roll_seed_r8.py` applies the
      re-pointing first; `rewrite_child_cmd` swaps it into the command.
  D4. G0 NON-BLOCKING MODE (explicit, stamped).  `run_battery_r7._run` raises SystemExit on a G0 that is not PASS.  For the
      step-5,000 EARLY READ the Master Mind wants the numbers even when G0 reads FAIL-as-coded (refcv7's own did at step 5,000
      and at 50,400); `load_battery_module(g0_nonblocking=True)` patches that ONE raise by an exact-anchor source edit (the
      anchor must occur exactly once, else it refuses) and the run stamps `G0_NONBLOCKING` into the summary.  SPEC section 5:
      "No refcv8 battery number counts unless its G0 passes" -- such numbers are PROVISIONAL and say so.

Nothing here changes a number: every edit is a path, a spawn command, or the G0 stop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint  # noqa: E402

#: where the 2026-10-15 cleanup put the refcv5cmp baselines (first existing wins; the C: original last)
ARTIFACT_ROOTS = [
    "D:/Projects/TanitAD-artifacts/refcv5cmp/out",
    "E:/Projects/TanitAD-artifacts/refcv5cmp/out",
    "C:/Users/Admin/c_cleanup_archive_2026-10-15_b/refcv5cmp/out",
    "C:/Users/Admin/refcv5cmp/out",
]
#: the refcv7-50,400 battery bank (banked dumps hold the full 8-slot 0.5-6 s plans; ckpt md5 b418d0fc...)
R7_BASE_ROOT = os.environ.get("R8_R7BASE_ROOT", "D:/refcv7_eval_kit/battery/step50400_a6text")
R7_BASE_MD5 = "b418d0fc4a92a6848c246a6a7c50207b"
R7_BASE_LABEL = "b_refcv7_50400_s{s}"

_REC: dict = {"baseline_repoint": {}, "refcv7_baselines": {}, "child_rewrites": [], "g0_nonblocking": False,
              "baselines_dropped": {}}


def dump_ok(d) -> bool:
    d = Path(d)
    return all((d / p).exists() for p in ("manifest.json", "ep000.npz", "decisions/ep000.npz"))


def apply():
    """r8_repoint.apply() + import the battery panel module + repoint its baselines. Returns (L, P)."""
    L = r8_repoint.apply()
    import r7_panel as P                                    # noqa: E402  (BATTERY_CODE is on sys.path)
    fix_baselines(P)
    return L, P


def apply_child():
    """What a CHILD process (roll_seed_r8 / perc_pass_r8) needs: the loader re-pointing + the pose-sync patch."""
    return r8_repoint.apply()


def fix_baselines(P, roots=None) -> dict:
    """Re-point `P.BASELINES` (in place; `build_s6_dump`'s default argument is the same dict object)."""
    roots = list(roots or ARTIFACT_ROOTS)
    out = {}
    for lab, p in list(P.BASELINES.items()):
        if dump_ok(p):
            out[lab] = {"path": p, "moved": False}
            continue
        name = Path(p).name
        hit = next((str(Path(r) / name) for r in roots if dump_ok(Path(r) / name)), None)
        if hit is None:
            raise SystemExit(f"[r8_battery_patch] baseline {lab}: {p} is absent and {name} is in none of {roots}")
        P.BASELINES[lab] = hit
        out[lab] = {"path": hit, "moved": True, "was": p}
    _REC["baseline_repoint"] = out
    return out


def add_refcv7_baselines(P, RB=None, root: str | None = None, seeds=(0, 1)) -> dict:
    """Register refcv7-50,400 `os` (per inference seed) as a paired baseline + the PAIRS that read it."""
    root = root or R7_BASE_ROOT
    added = {}
    for s in seeds:
        lab = R7_BASE_LABEL.format(s=s)
        d = f"{root}/dump_s{s}"
        if not dump_ok(d):
            raise SystemExit(f"[r8_battery_patch] refcv7-50,400 baseline dump {d} is incomplete / absent")
        P.REFCV6_38K[lab] = d                              # baselines_available() includes any entry whose manifest exists
        P.TIERS[lab] = "T1"                                # build_s6_dump does TIERS[arm] for EVERY baseline (KeyError otherwise:
                                                           # found by the CPU smoke 2026-10-15 23:17, r7_panel.py:627)
        added[lab] = d
        if RB is not None:
            for pair in (("b_refcv7_50400_s%d" % s, "os", "os_minus_refcv7_50400_s%d" % s),
                         ("ha0_ext", "b_refcv7_50400_s%d" % s, "refcv7_50400_s%d_minus_ha0ext" % s)):
                if pair not in RB.PAIRS:
                    RB.PAIRS.append(pair)
    _REC["refcv7_baselines"] = added
    return added


def windows_of(dump_dir) -> dict:
    """{clip_id: sorted ws list} of a banked dump (manifest clip_id -> ep file -> `ws`), the EXACT window identity."""
    import numpy as np
    d = Path(dump_dir)
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    out = {}
    for e in man["episodes"]:
        with np.load(d / f"ep{int(e['file_index']):03d}.npz") as z:
            out[str(e["clip_id"])] = sorted(int(x) for x in z["ws"])
    return out


def window_set(w: dict) -> set:
    return {(c, int(x)) for c, v in w.items() for x in v}


def drop_unusable_optional(P, windows: dict) -> dict:
    """The OPTIONAL baselines (refcv6@38k, refcv5v2, ...) must carry every S2 window or `build_panel_dump` raises
    mid-battery after hours of rolling.  A required baseline (refcv4b = the window source, refcv7-50,400) that fails
    refuses; an optional one is dropped from the registries and the drop is recorded."""
    want = window_set(windows)
    required = {"b_refcv4b"} | {R7_BASE_LABEL.format(s=s) for s in (0, 1)}
    for reg in (P.BASELINES, P.REFCV6_38K):
        for lab, d in list(reg.items()):
            if reg is P.REFCV6_38K and not os.path.exists(os.path.join(d, "manifest.json")):
                continue
            try:
                have = window_set(windows_of(d))
                ok = want <= have
                why = None if ok else f"{len(want - have)} S2 windows absent from the baseline"
            except Exception as exc:                        # noqa: BLE001 -- recorded
                ok, why = False, f"{type(exc).__name__}: {str(exc)[:160]}"
            if not ok:
                if lab in required:
                    raise SystemExit(f"[r8_battery_patch] REQUIRED baseline {lab} cannot pair: {why}")
                _REC["baselines_dropped"][lab] = why
                del reg[lab]
    return dict(_REC["baselines_dropped"])


# ---------------------------------------------------------------- the battery module, one exact-anchor edit
_ANCHOR = ('        if v["G0"] != "PASS":\n'
           '            if not a.g0_text_verdict:\n')
_REPLACEMENT = (
    '        if v["G0"] != "PASS" and R8_G0_NONBLOCKING and not a.g0_text_verdict:\n'
    '            summary["G0_NONBLOCKING"] = {\n'
    '                "G0_as_coded": v["G0"], "reasons": v["reasons"][:40],\n'
    '                "stamp": "G0 did NOT pass as coded; the battery proceeded by explicit request (EARLY READ). "\n'
    '                         "SPEC_REFCV8 section 5: no refcv8 battery number counts unless its G0 passes -- every "\n'
    '                         "number in this run is PROVISIONAL and is not bar evidence."}\n'
    '            print("[battery] G0 = %s -- NOT STOPPING (r8 G0-nonblocking early read; numbers PROVISIONAL)" % v["G0"], '
    'flush=True)\n'
    '            bank()\n'
    '        elif v["G0"] != "PASS":\n'
    '            if not a.g0_text_verdict:\n')


def patched_source(src: str) -> str:
    n = src.count(_ANCHOR)
    if n != 1:
        raise SystemExit(f"[r8_battery_patch] the G0-stop anchor occurs {n} times in run_battery_r7.py (need exactly 1); "
                         f"the upstream file changed -- refusing to patch")
    return src.replace(_ANCHOR, _REPLACEMENT)


def load_battery_module(g0_nonblocking: bool = False):
    """`run_battery_r7` as module `run_battery_r7` (so children / siblings that `import run_battery_r7` agree), from its
    ORIGINAL path (`__file__`, hence HERE, are the launch tree's, so roll_seed / g0 / SPEC paths resolve), with the one
    G0-stop edit when asked."""
    p = r8_repoint.BATTERY_CODE / "run_battery_r7.py"
    src = p.read_text(encoding="utf-8")
    mod = types.ModuleType("run_battery_r7")
    mod.__file__ = str(p)
    mod.R8_G0_NONBLOCKING = bool(g0_nonblocking)
    _REC["g0_nonblocking"] = bool(g0_nonblocking)
    code = compile(patched_source(src) if g0_nonblocking else src, str(p), "exec", dont_inherit=True)
    sys.modules["run_battery_r7"] = mod
    exec(code, mod.__dict__)                                # noqa: S102 -- our own launch-tree file, anchor-checked
    return mod


# ---------------------------------------------------------------- child processes
def rewrite_child_cmd(cmd, *, smoke_cpu: bool = False, os_only_all: bool = False):
    """Swap `roll_seed_r7.py` for `roll_seed_r8.py` (the child must carry the re-pointing + pose-sync policy).
    `os_only_all` adds `--os-only` to EVERY seed's roll (SPEC A3 already does that for seeds after the first): the corrected-clock
    pass needs only `os` + the model-free controls, not the seed-0 perturbation / sensitivity arms."""
    cmd = [str(c) for c in cmd]
    for i, c in enumerate(cmd):
        if c.replace("\\", "/").endswith("/roll_seed_r7.py"):
            cmd[i] = str(HERE / "roll_seed_r8.py")
            if smoke_cpu and "--smoke-cpu-fp32-trunk" not in cmd:
                cmd.append("--smoke-cpu-fp32-trunk")
            if os_only_all and "--os-only" not in cmd:
                cmd.append("--os-only")
            _REC["child_rewrites"].append({"from": c, "to": cmd[i], "smoke_cpu": bool(smoke_cpu),
                                           "os_only_all": bool(os_only_all)})
            break
    return cmd


class SubprocessShim:
    """Stands in for the `subprocess` module inside the battery module: identical, except the roll child is rewritten."""

    def __init__(self, smoke_cpu: bool = False, os_only_all: bool = False):
        self._smoke = bool(smoke_cpu)
        self._os_only_all = bool(os_only_all)

    def __getattr__(self, k):
        return getattr(subprocess, k)

    def run(self, cmd, *a, **k):
        return subprocess.run(rewrite_child_cmd(cmd, smoke_cpu=self._smoke, os_only_all=self._os_only_all), *a, **k)


def describe() -> dict:
    return json.loads(json.dumps(_REC, default=str))
