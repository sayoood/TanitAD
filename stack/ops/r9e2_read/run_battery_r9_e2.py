"""run_battery_r9_e2.py -- the refcv8 per-frame battery (``run_battery_r8_pf.py``) driven for a refcv9 run dir: DEV windows only, refcv9 mechanisms STAGED.

NEW file (refcv9 E2-ROLL, 2026-11-04).  Existing files are NOT edited.  Three differences against ``run_battery_r8_pf.py`` (HANDOFF_famroll.md section 1):
  (i)   DEV FILTER  -- ``r7_roll.s2_window_list`` (all banked refcv4b S2 windows of the kit) is replaced by a filter that keeps only the DEV clips of
        DEV_HOLD_SPLIT.json (clip key sha256(clip_id)[:12] == ``refcv7_loader.sha12``).  The PARENT writes ``--windows-json``; the child only re-reads it, so
        patching the parent suffices.  HOLD must never be scored: the kept set is asserted disjoint from HOLD and equal, per clip and in total, to the split's own
        S2 window counts (REFUSES otherwise).  Applied the moment ``r7_roll`` is first imported (it is only importable after ``r8_repoint.apply()``).
  (ii)  STAGING     -- the child (``roll_seed_r9_e2.py`` / ``camblind_roll_r9_e2.py``) rebinds ``r7_roll.Refcv6Proxy`` to a subclass whose RAW model is wrapped so
        that EVERY inner ``model(...)`` call first calls ``_bw.stage`` then ``_bc5.stage`` (the trainer's order, refc_v3_train.py:5920-5922 / 7158-7160;
        r9_e1_dump.py:135-148,432 stages ``_bc5``).  The staged batch is the window's dataset item expanded to the CALL's own batch size (a pfq stage must have
        R.shape[0] == the forward batch, slot_query_select.py:360; both stages are ONE-SHOT) and ``agent_ep`` = the window's episode id taken from the eval
        dataset exactly as the trainer emits it (``int(ep.episode_id)``, refc_v3_train.py:3997/4029), because the roll builds the eval dataset with
        ``with_perception_targets=False`` (no agent join => the item has no ``agent_ep``).  K0 (no flag built) => both stages are no-ops.
  (iii) NO DEV-BOX DEFAULT -- R8_LAUNCH_TREE, PF_R8_CODE, PF_NF_HOLD, R8_EVAL_KIT, R9E2_REMAP, R9E2_SPLIT, R9E2_PF_CODE come from the environment; the run
        REFUSES when one is unset.  ``r8_repoint.REMAP_OVERRIDES`` is set from R9E2_REMAP in-process (the technique camblind_roll_r9.py uses; it is read at
        ``configure_env()`` time), so the ``r8_repoint.py:42`` one-line edit is NOT needed for this entry point.
  (iv)  THOR GPU GATE (FIXED 2026-11-12) -- on Jetson Thor ``gpu_gate.gate`` is replaced, the moment the battery imports it, by ``thor_gate_factory``: the
        dev-box gate reads memory.used (Thor: "[N/A]") and free RAM through powershell.exe, so on Thor it can only WAIT (see the THOR_MARKER block).
Pose-clock policy: E2 runs the CORRECTED clock => ``--no-baselines`` REQUIRED and ``R8_NO_POSE_SYNC`` must be UNSET (run_battery_r8.py:63-70).
UNVERIFIED: no real K0 / K2 roll exists; the MM runs it on Thor.

    python run_battery_r9_e2.py --ckpt <run>/model_ema.pt --config <run>/config.json --no-g0 --no-baselines --tag E2_K2_pf --infer-seeds 0,1 --lock-held --out-root <dir>
"""
from __future__ import annotations

import csv
import hashlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PINS = ("R8_LAUNCH_TREE", "PF_R8_CODE", "PF_NF_HOLD", "R8_EVAL_KIT", "R9E2_REMAP", "R9E2_SPLIT", "R9E2_PF_CODE")
CHILD = HERE / "roll_seed_r9_e2.py"


def sha12(clip_id) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]      # == refcv7_loader.sha12


def require_env(env=None, pins=ENV_PINS) -> dict:
    env = os.environ if env is None else env
    missing = [k for k in pins if not str(env.get(k, "")).strip()]
    if missing:
        raise SystemExit(f"[r9e2] refusing: unset env pin(s) {missing} -- no dev-box default is taken on Thor")
    if str(env.get("R8_NO_POSE_SYNC", "")).strip():
        raise SystemExit("[r9e2] refusing: R8_NO_POSE_SYNC is set -- E2 reads the CORRECTED pose clock (K0 trained on it); unset it")
    return {k: str(env[k]) for k in pins}


def load_dev_hold(path):
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    dev, hold = {str(x).lower() for x in j["DEV"]["sha12"]}, {str(x).lower() for x in j["HOLD"]["sha12"]}
    if not dev or not hold or dev & hold:
        raise SystemExit(f"[r9e2] {path}: DEV / HOLD empty or not disjoint")
    return dev, hold, j


def make_dev_window_filter(orig, dev, hold, split=None, census=None):
    """r7_roll.s2_window_list(baseline_dump, kit_clip_ids) -> the same dict restricted to DEV clips.  REFUSES if a kept clip is HOLD, nothing is kept, or the kept
    windows differ from the split's own per-clip / total S2 counts (``split["DEV"]["windows_per_episode"][sha12][0]`` / ``n_windows_s2``)."""
    census = {} if census is None else census

    def s2_window_list(baseline_dump, kit_clip_ids):
        full = orig(baseline_dump, kit_clip_ids)
        kept = {c: w for c, w in full.items() if sha12(c) in dev}
        if not kept:
            raise SystemExit("[r9e2] the DEV filter kept no clip")
        if any(sha12(c) in hold for c in kept):
            raise SystemExit("[r9e2] a HOLD clip survived the DEV filter -- refusing to roll it")
        nw = sum(len(v) for v in kept.values())
        if split is not None:
            wpe = split["DEV"]["windows_per_episode"]
            bad = [sha12(c) for c, w in kept.items() if int(wpe[sha12(c)][0]) != len(w)]
            if bad or set(wpe) != {sha12(c) for c in kept} or nw != int(split["DEV"]["n_windows_s2"]):
                raise SystemExit(f"[r9e2] the DEV windows differ from DEV_HOLD_SPLIT.json: {nw} windows / {len(kept)} clips, per-clip mismatches {bad[:5]} "
                                 f"(split: {split['DEV']['n_windows_s2']} / {len(wpe)})")
        census.update(n_clips_all=len(full), n_clips_dev=len(kept), n_clips_dropped=len(full) - len(kept), n_windows_dev=nw,
                      n_windows_dropped=sum(len(v) for c, v in full.items() if c not in kept))
        return kept
    return s2_window_list


# ---- refcv9 staging ---------------------------------------------------------------------------------------------------------------- #
def item_batch(item, n):
    """The window's dataset item (unbatched) -> a B=n batch of VIEWS (expand: no copy); non-tensors are skipped."""
    import torch
    return {k: v[None].expand(n, *v.shape) for k, v in item.items() if torch.is_tensor(v)}


def default_mods():
    from tanitad.train import bev_warp_flag as _bw
    from tanitad.train import box_c5_flag as _bc5
    return [("_bw", _bw), ("_bc5", _bc5)]          # the trainer's order (refc_v3_train.py:5920-5922)


class Stager:
    def __init__(self, mods, episodes_of=None):
        self.mods, self.episodes_of = list(mods), episodes_of
        self.n_forwards, self.n_stage = 0, {lab: 0 for lab, _ in self.mods}
        self.batch_sizes, self.probe = {}, {}

    def episode_id(self, ctx, item):
        import torch
        eps = self.episodes_of() if self.episodes_of else None
        if eps is not None and getattr(ctx, "index", None) is not None:
            return int(eps[int(ctx.index[0])].episode_id)
        for k in ("agent_ep", "episode_id"):
            if k in item:
                return int(torch.as_tensor(item[k]).reshape(-1)[0])
        return None

    def stage(self, model, ctx, frames):
        import torch
        n, dev, item = int(frames.shape[0]), frames.device, getattr(ctx, "item", None)
        if item is None:
            raise RuntimeError("[r9e2-stage] a forward with no window item on the ctx -- nothing to stage from")
        batch = item_batch(item, n)
        eid = self.episode_id(ctx, item)
        if eid is not None:
            batch["agent_ep"] = torch.full((n,), eid, dtype=torch.long)       # absent => _bc5.stage REFUSES (pfq) / is a no-op (K0): fail loud by delegation
        for lab, mod in self.mods:
            mod.stage(model, batch, dev)
            self.n_stage[lab] += 1
        self.n_forwards += 1
        self.batch_sizes[n] = self.batch_sizes.get(n, 0) + 1
        if not self.probe:
            self.probe = {"bc5_pfq_built": getattr(getattr(model, "_perception", None), "box_pfq", None) is not None,
                          "bw_built": getattr(model, "_r9_bev_warp", None) is not None}

    def census(self) -> dict:
        return {"n_forwards": self.n_forwards, "n_stage_calls": dict(self.n_stage), "batch_sizes": dict(self.batch_sizes),
                "mechanisms_built": dict(self.probe), "order": [lab for lab, _ in self.mods]}


class StagedModel:
    """Wraps the RAW model: every call (the proxy makes up to four per window) stages first.  Everything else delegates."""

    def __init__(self, model, ctx, stager):
        d = self.__dict__
        d["_model"], d["_ctx"], d["_stager"] = model, ctx, stager

    def __getattr__(self, k):
        return getattr(self.__dict__["_model"], k)

    def __setattr__(self, k, v):
        setattr(self.__dict__["_model"], k, v)

    def __call__(self, *a, **kw):
        d = self.__dict__
        frames = a[0] if a else kw.get("frames")
        if frames is None:
            raise RuntimeError("[r9e2-stage] a model call without frames")
        d["_stager"].stage(d["_model"], d["_ctx"], frames)
        return d["_model"](*a, **kw)


def make_staging_proxy_cls(base, stager):
    class StagingRefcv6Proxy(base):
        def __init__(self, model, ctx, *a, **kw):
            super().__init__(model, ctx, *a, **kw)                 # reads model.modules() on the RAW model
            self.__dict__["model"] = StagedModel(model, ctx, stager)
    return StagingRefcv6Proxy


def install_staging(RR, mods=None):
    """On the imported ``r7_roll`` module: capture the eval dataset (episode ids) and rebind Refcv6Proxy (install_patches.load_model reads it at call time)."""
    holder = {"ds": None}
    orig = RR.L.build_eval_dataset

    def capture(*a, **kw):
        out = orig(*a, **kw)
        holder["ds"] = out[0]
        return out
    RR.L.build_eval_dataset = capture
    stager = Stager(mods or default_mods(), episodes_of=lambda: getattr(holder["ds"], "episodes", None))
    RR.Refcv6Proxy = make_staging_proxy_cls(RR.Refcv6Proxy, stager)
    RR.R9E2_STAGER = stager
    return stager


# ---- import hook: run fn(module) right after `name` is first imported ---------------------------------------------------------------- #
class _PostImport(importlib.abc.MetaPathFinder):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader, fn = spec.loader, self.fn
        orig = loader.exec_module

        def exec_module(module):
            orig(module)
            fn(module)
        loader.exec_module = exec_module
        return spec


def hook_on_import(name, fn):
    if name in sys.modules:
        fn(sys.modules[name])
        return None
    f = _PostImport(name, fn)
    sys.meta_path.insert(0, f)
    return f


# ---- Jetson Thor: the battery's GPU gate (FIXED 2026-11-12) ---------------------------------------------------------------------------- #
#: MEASURED 2026-11-12 10:08-10:18Z, the FIRST REAL E2 roll on Thor (runs/E2_READ_K2M1N20, roll_K0.out): run_battery_r7.gate_wait -> gpu_gate.gate WAITED
#: 8 times and would have waited 12 h, holding thor_gpu.lock.  gpu_gate.py is the DEV-BOX gate (RTX 4060): it int()s `nvidia-smi --query-gpu=memory.used`
#: (Thor answers "[N/A]" -> ValueError -> probe FAILED -> WAIT) and reads free RAM through powershell.exe.  The roll's DRY run skips the lock and the gate,
#: so no test and no dry run ever reached it on Thor.  On Thor NO memory figure is admissible (TRAPS.md: unified memory; nvidia-smi / free / tegrastats /
#: VmRSS all lie), so the Thor gate reads none: the caller holds thor_gpu.lock (--lock-held; r9e2_roll.sh takes it before any GPU stage) and NO OTHER
#: compute app is on the GPU.  `--query-compute-apps` does list a live CUDA process on Thor (MEASURED 2026-11-12: own pid, python, "134 MiB"; empty when
#: idle).  Stricter than the dev box: ANY other app (python or not) is a WAIT.  A probe that cannot be read is a WAIT (fail closed, as gpu_gate.PROBE_ERRORS).
THOR_MARKER = Path("/etc/nv_tegra_release")
THOR_RULE = ("THOR (unified memory: no memory probe is admissible, TRAPS.md): the caller holds thor_gpu.lock (--lock-held) and no OTHER compute app is on "
             "the GPU (nvidia-smi --query-compute-apps); memory.used and free RAM are NOT read")
_THOR_PROBE_ERRORS = (subprocess.SubprocessError, OSError, ValueError, IndexError)


def is_thor(marker=None) -> bool:
    return Path(THOR_MARKER if marker is None else marker).exists()


def _compute_apps_query() -> str:
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=True).stdout


def thor_gate_factory(query=None):
    """A drop-in for ``gpu_gate.gate`` on Thor: same keywords; ``max_used_mib`` / ``min_free_ram_gb`` are accepted and NOT applied (THOR_RULE)."""
    q = _compute_apps_query if query is None else query

    def gate(max_used_mib=4300, min_free_ram_gb=8.0, self_pid=None, self_pids=None):
        own = {int(p) for p in (self_pids or ()) if p is not None}
        own.add(os.getpid() if self_pid is None else int(self_pid))
        g = {"platform": "thor", "gpu_used_mib": None, "free_ram_gb": None, "self_pids_excluded": sorted(own), "rule": THOR_RULE,
             "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            apps = [(int(row[0].strip()), row[1].strip() if len(row) > 1 else "")
                    for row in csv.reader(io.StringIO(q())) if row and any(c.strip() for c in row)]
        except _THOR_PROBE_ERRORS as exc:                       # FAIL CLOSED: an unreadable probe (or an unparsable row) is a WAIT, never a PASS
            return {**g, "ok": False, "probe_ok": False, "probe_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "compute_apps": None, "python_compute": None}
        other = [(p, n) for p, n in apps if p not in own]
        return {**g, "ok": not other, "probe_ok": True, "compute_apps": other, "python_compute": [(p, n) for p, n in other if "python" in n.lower()]}
    gate.r9e2_thor = True
    return gate


def install_thor_gate(argv, marker=None, query=None):
    """On Thor: REFUSE without ``--lock-held``, else replace ``gpu_gate.gate`` the moment the battery imports gpu_gate.  Elsewhere: nothing (the dev-box gate stays)."""
    if not is_thor(marker):
        return None
    if "--lock-held" not in argv:
        raise SystemExit("[r9e2] refusing: on Jetson Thor the battery GPU gate can read no memory figure (memory.used is [N/A]) -- the caller must hold "
                         "thor_gpu.lock and pass --lock-held")
    return hook_on_import("gpu_gate", lambda m: setattr(m, "gate", thor_gate_factory(query)))


def pin_repoint(env=None):
    """Import r8_repoint from PF_R8_CODE and point it at the env pins BEFORE any apply() (REMAP_OVERRIDES is read at configure_env() time)."""
    env = os.environ if env is None else env
    sys.path.insert(0, str(env["PF_R8_CODE"]))
    import r8_repoint as RP
    if Path(str(RP.LAUNCH_TREE)) != Path(env["R8_LAUNCH_TREE"]) or str(RP.EVAL_KIT) != env["R8_EVAL_KIT"]:
        raise SystemExit(f"[r9e2] r8_repoint read LAUNCH_TREE={RP.LAUNCH_TREE} EVAL_KIT={RP.EVAL_KIT}, not the env pins")
    RP.REMAP_OVERRIDES = Path(env["R9E2_REMAP"])
    return RP


def child_stage_hook():
    return hook_on_import("r7_roll", install_staging)


def _val(flag, argv):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--no-baselines" not in argv:
        raise SystemExit("[r9e2] refusing: --no-baselines is REQUIRED on the corrected pose clock")
    split_path = _val("--split", argv)
    if split_path:                                   # an own switch: strip it before the battery's argparse sees it
        i = argv.index("--split")
        del argv[i:i + 2]
        os.environ.setdefault("R9E2_SPLIT", split_path)
    pins = require_env()
    install_thor_gate(argv)                          # AFTER the env refusals (their tests run on Thor too), BEFORE the battery imports gpu_gate
    dev, hold, split = load_dev_hold(pins["R9E2_SPLIT"])
    census = {}
    pin_repoint()
    sys.path.insert(0, pins["R9E2_PF_CODE"])
    import run_battery_r8_pf as RBPF               # imports roll_seed_r8_train_pf (no side effects)
    RBPF.PF_CHILD = CHILD                          # pf_rewrite reads the global at call time: the child roll is OUR staged child
    hook_on_import("r7_roll", lambda m: setattr(m, "s2_window_list", make_dev_window_filter(m.s2_window_list, dev, hold, split, census)))
    sys.argv = [sys.argv[0]] + argv
    rc = RBPF.main()
    out_root, tag = _val("--out-root", argv), _val("--tag", argv)
    if out_root and tag and (Path(out_root) / tag).is_dir():
        (Path(out_root) / tag / "r9e2_dev_filter.json").write_text(json.dumps(
            {"split": pins["R9E2_SPLIT"], "dev_clips": len(dev), "hold_clips": len(hold), "filter": census, "child": str(CHILD)}, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
