"""extract_ema_r8.py -- derive the milestone-layout pair (`ckpt_<S>.pt`, `ckpt_ema_<S>.pt`) from a ROLLING full `ckpt.pt`, the TRAINER'S WAY.  CPU only.

    python extract_ema_r8.py extract    --src <full ckpt.pt> --config E:/refcv8_eval_kit/ckpt/config.json --out-dir E:/refcv8_eval_kit/ckpt --expect-step 40000 \
                                        [--provenance thor_md5=... ...] [--force]
    python extract_ema_r8.py selftest   --out <json>                          # tiny synthetic model: WeightEMA round trip, with its deliberate-regression arms
    python extract_ema_r8.py check-real --raw E:/refcv8_eval_kit/ckpt/ckpt_30000.pt --ema E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt \
                                        --config E:/refcv8_eval_kit/ckpt/config.json --out <json>      # what extract would write vs the real 30k pair

WHY A SECOND ROUTE.  `slim_rolling_ckpt_r8.py` (landed, used for step 10,500) re-implements `WeightEMA.model_state_dict` over the checkpoint's tensors; it builds no model.
THIS script is the independent cross-route the Master Mind asked for: it builds the model from the run config EXACTLY as the battery's loader does
(`r8_repoint.apply()` -> `refcv7_loader.build_model`, strict load), constructs the REAL `WeightEMA` with the run's decay, `load_state_dict(ckpt['ema'], model)`,
and writes `ckpt_<S>.pt` + `ckpt_ema_<S>.pt` with the TRAINER'S OWN `_save_ema_model` (imported from `scripts/refc_v3_train.py`, not copied).  It then CROSS-CHECKS
its EMA payload bit-for-bit against `slim_rolling_ckpt_r8.derive_ema_model` (two independent derivations must agree on every key).

⛔ THE TRAP THIS SCRIPT HAD TO WORK AROUND (found by reading, pinned by the selftest's regression arm): `refcv7_loader.build_model` ends with
`for p in model.parameters(): p.requires_grad_(False)` (eval build).  `WeightEMA` shadows "every TRAINABLE parameter and every floating buffer", so a `WeightEMA(model, d)`
built on the loader's model shadows NO parameter and `load_state_dict(ckpt['ema'])` raises (key sets differ).  The training-time trainable pattern is restored FROM THE
CHECKPOINT'S OWN SHADOW KEYS (a tied parameter is trainable if ANY of its names is shadowed), and the rebuilt `WeightEMA`'s key set must then equal the checkpoint's exactly
(`WeightEMA.load_state_dict` refuses otherwise): the checkpoint, not a guess, decides what was trainable.

MEMORY (the dev box's COMMIT limit binds, not RAM): the full 1.56 GB `ckpt.pt` is loaded once, the optimiser state dropped at once; peak ~2.5 GB.  `--min-commit-gb` (default 4)
refuses to start below that free commit (Windows).  No GPU is touched; `CUDA_VISIBLE_DEVICES=-1` is set by the script itself.
UNVERIFIED until a real full `ckpt.pt` is on the dev box (none is, 2026-10-18): the selftest and `check-real` are what exist now.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
KIT = "E:/refcv8_eval_kit"
META_KEYS = {"model", "step", "ema_decay", "num_updates", "kind"}          # _save_ema_model's payload (refc_v3_train.py:5206-5207)


def _md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def free_commit_gb() -> float | None:
    """Available commit (page file + RAM) in GB on Windows; None elsewhere.  The dev box's binding limit (CLAUDE.md memory: 'commit limit binds')."""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def _alias_groups(sd: dict) -> list:
    """Partition of the state dict's tensors by shared storage (tied tensors), as sorted name tuples."""
    import torch
    g: dict = {}
    for k, t in sd.items():
        if torch.is_tensor(t):
            g.setdefault((t.untyped_storage().data_ptr(), int(t.storage_offset()), tuple(t.shape), tuple(t.stride())), []).append(k)
    return sorted(tuple(sorted(v)) for v in g.values() if len(v) > 1)


def restore_trainable_from_shadow(model, shadow_keys) -> dict:
    """Set `requires_grad` on every Parameter to its TRAINING-time value, read from the checkpoint's own shadow keys.  A tied parameter (one object, several
    state_dict names) is trainable iff ANY of its names is shadowed -- done per OBJECT, so an alias visited later cannot undo it."""
    import torch
    by_obj: dict = {}
    for name, t in model.state_dict(keep_vars=True).items():
        if isinstance(t, torch.nn.Parameter):
            by_obj.setdefault(id(t), [t, []])[1].append(name)
    n_tr = 0
    for p, names in by_obj.values():
        want = any(n in shadow_keys for n in names)
        p.requires_grad_(want)
        n_tr += int(want)
    return {"n_parameters": len(by_obj), "n_trainable_restored": n_tr, "n_shadow_keys": len(shadow_keys)}


def extract_core(model, ck: dict, out_dir, *, step: int, decay: float, save_ema, force: bool = False, raw_path_prebuilt=None) -> dict:
    """`model` already has `ck['model']` loaded (the loader's strict load).  Writes the pair, verifies it, returns the record.  Pure of the loader: tested on a tiny model."""
    import torch
    from slim_rolling_ckpt_r8 import derive_ema_model
    from tanitad.train.weight_ema import WeightEMA
    out = Path(out_dir)
    raw_p, ema_p = out / f"ckpt_{step}.pt", out / f"ckpt_ema_{step}.pt"
    ema_sd = ck["ema"]
    rec: dict = {"step": step, "decay": decay}
    rec["trainable_restore"] = restore_trainable_from_shadow(model, set(ema_sd["shadow"]))
    ema = WeightEMA(model, decay)
    ema.load_state_dict(ema_sd, model)                       # raises EMAStateError on a key / shape / decay disagreement
    if int(ema.num_updates) != int(step):
        raise ValueError(f"EMA num_updates {ema.num_updates} != step {step}")
    if not force:
        for p in (raw_p, ema_p):
            if p.exists() and raw_path_prebuilt is None:
                raise FileExistsError(f"{p} exists (pass --force)")
        if ema_p.exists():
            raise FileExistsError(f"{ema_p} exists (pass --force)")
    # ---- the files: raw = the trainer's milestone layout; ema = the trainer's OWN writer ------------------------------------------------------- #
    if raw_path_prebuilt is None:
        tmp = raw_p.with_suffix(".pt.part")
        torch.save({"model": model.state_dict(), "step": int(step)}, tmp)
        os.replace(tmp, raw_p)
    tmp = ema_p.with_suffix(".pt.part")
    save_ema(ema, model, int(step), str(tmp))
    os.replace(tmp, ema_p)
    # ---- verification by CONTENT, against an INDEPENDENT derivation (freed step by step: the commit limit binds) ---------------------------------- #
    numel = int(ema.numel)
    del ema
    gc.collect()
    derived, dstats = derive_ema_model(ck["model"], ema_sd)
    got = torch.load(ema_p, map_location="cpu", weights_only=False)
    problems = []
    if set(got) != META_KEYS:
        problems.append(f"EMA file meta keys {sorted(got)} != {sorted(META_KEYS)}")
    if got.get("kind") != "ema_model_only" or int(got.get("step", -1)) != step or int(got.get("num_updates", -1)) != step:
        problems.append(f"EMA meta wrong: kind={got.get('kind')!r} step={got.get('step')} num_updates={got.get('num_updates')}")
    if float(got.get("ema_decay", -1)) != float(decay):
        problems.append(f"EMA decay {got.get('ema_decay')} != {decay}")
    gm = got["model"]
    if set(gm) != set(derived) or set(gm) != set(ck["model"]):
        problems.append("EMA model key set != the independent derivation's / the checkpoint's")
    n_bad, n_diff = 0, 0
    for k, v in gm.items():
        d, live = derived.get(k), ck["model"].get(k)
        if torch.is_tensor(v):
            if d is None or v.dtype != d.dtype or v.shape != d.shape or not torch.equal(v, d):
                n_bad += 1
            if v.is_floating_point() and live is not None and not torch.equal(v, live):
                n_diff += 1
        elif v != d:
            n_bad += 1
    if n_bad:
        problems.append(f"{n_bad} tensors differ from slim_rolling_ckpt_r8.derive_ema_model (two derivations DISAGREE)")
    if n_diff == 0:
        problems.append("every EMA tensor equals the live weight: the EMA file would be pointless")
    if _alias_groups(gm) != _alias_groups(ck["model"]):
        problems.append("tied-tensor (shared storage) partition differs between the EMA file and the checkpoint")
    n_keys = len(gm)
    del got, gm, derived
    gc.collect()
    raw = torch.load(raw_p, map_location="cpu", weights_only=False)
    if set(raw) != {"model", "step"} or int(raw["step"]) != step or set(raw["model"]) != set(ck["model"])             or any(not torch.equal(raw["model"][k], v) for k, v in ck["model"].items() if torch.is_tensor(v)):
        problems.append("raw milestone file != ck['model']")
    del raw
    gc.collect()
    rec.update(n_keys=n_keys, n_float_tensors_ema_differs_from_live=n_diff, slim_derive_stats=dstats, problems=problems, ok=not problems,
               raw={"path": str(raw_p), "md5": _md5(raw_p)}, ema={"path": str(ema_p), "md5": _md5(ema_p)}, shadow_numel=numel)
    return rec


def _trainer_save_ema():
    """-> (save_ema fn, source string).  The trainer's own `_save_ema_model`, via the loader's `trainer()` (imported, never copied)."""
    import r8_repoint
    L = r8_repoint.apply(pose_sync=False)
    tr = L.trainer()
    return tr._save_ema_model, f"{tr.__file__}::_save_ema_model", L


def cmd_extract(a) -> int:
    rec: dict = {"tool": "extract_ema_r8.py", "src": a.src, "config": a.config, "started": time.strftime("%FT%T"), "ok": False}
    try:
        gb = free_commit_gb()
        rec["free_commit_gb_at_start"] = gb
        if gb is not None and gb < a.min_commit_gb:
            raise RuntimeError(f"free commit {gb:.1f} GB < --min-commit-gb {a.min_commit_gb}: refusing (the dev box's commit limit binds; a failed allocation can hit the live GPU job)")
        import torch
        save_ema, src_of_save, L = _trainer_save_ema()
        rec["save_ema_source"] = src_of_save
        ck = torch.load(a.src, map_location="cpu", weights_only=False)
        for k in ("model", "step", "ema"):
            if k not in ck:
                raise ValueError(f"source has no {k!r} (keys: {sorted(ck)[:12]})")
        step = int(ck["step"])
        if a.expect_step is not None and step != a.expect_step:
            raise ValueError(f"source step is {step}, expected {a.expect_step}")
        ck = {"model": ck["model"], "step": step, "ema": ck["ema"]}          # drop the optimiser state NOW
        gc.collect()
        out = Path(a.out_dir)
        raw_p = out / f"ckpt_{step}.pt"
        if (raw_p.exists() or (out / f"ckpt_ema_{step}.pt").exists()) and not a.force:
            raise FileExistsError(f"{raw_p} or its EMA twin exists (pass --force)")
        tmp = raw_p.with_suffix(".pt.part")
        torch.save({"model": ck["model"], "step": step}, tmp)               # the loader's strict build reads THIS file (a 0.4 GB load, not the 1.56 GB one)
        os.replace(tmp, raw_p)
        config = L.load_config(a.config)
        decay = float((config.get("ema") or {}).get("decay"))
        model, cfg, targs, mrec = L.build_model(config, str(raw_p), "cpu")
        rec["model_build"] = {"strict_state_dict": mrec.get("state_dict"), "ema_decay_from_config": decay}
        r = extract_core(model, ck, out, step=step, decay=decay, save_ema=save_ema, force=a.force, raw_path_prebuilt=True)
        rec.update(r)
        rec["src_md5"] = _md5(a.src)
        rec["provenance"] = dict(kv.split("=", 1) for kv in a.provenance if "=" in kv)
        rec["note"] = ("DERIVED on the dev box from a rolling ckpt.pt with the trainer's own _save_ema_model; NOT a trainer-written milestone file "
                       "(MILESTONES=(5000,15000,20000,30000), refc_v3_train.py:232).  Same layout, same tensors; cross-checked against slim_rolling_ckpt_r8.derive_ema_model.")
        (out / f"ckpt_{step}.provenance.extract_ema.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    except Exception as exc:                                                    # noqa: BLE001 -- the json line IS the result
        rec["reason"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    print(json.dumps({k: rec.get(k) for k in ("ok", "step", "reason", "problems", "raw", "ema")}, default=str))
    return 0 if rec.get("ok") else 1


# ------------------------------------------------------------------------------------------------ the synthetic known-value control
def _tiny():
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(6, 6)
            self.bn = nn.BatchNorm1d(6)                                      # float buffers (shadowed) + an int buffer (not)
            self.out = nn.Linear(6, 3)
            self.out_tied = self.out                                         # a TIED parameter: one object, two state_dict names
            self.frozen = nn.Linear(3, 3)
            self.frozen.requires_grad_(False)                                # frozen in TRAINING: never shadowed
            self.register_buffer("anchors", torch.randn(4, 2))               # a float buffer: shadowed

        def forward(self, x):
            return self.frozen(self.out(self.bn(self.lin1(x))))
    return Tiny()


def selftest(save_ema=None) -> dict:
    import tempfile
    import torch
    from tanitad.train.weight_ema import EMAStateError, WeightEMA
    res: dict = {"checks": {}}
    ok = res["checks"]
    torch.manual_seed(0)
    A = _tiny()
    E = WeightEMA(A, 0.9)
    opt = torch.optim.SGD([p for p in A.parameters() if p.requires_grad], lr=0.05)
    for _ in range(25):
        x = torch.randn(8, 6)
        loss = A(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        E.update(A)
    GT = {k: v.detach().clone() for k, v in E.model_state_dict(A).items()}        # the trainer's EMA payload, from the objects that TRAINED
    ck_mem = {"model": A.state_dict(), "step": 25, "ema": E.state_dict(), "optimizer": {"junk": torch.zeros(3)}}
    _td = tempfile.mkdtemp()
    torch.save(ck_mem, Path(_td) / "ckpt.pt")                                       # through a FILE: tied tensors share storage in it, exactly as in a real ckpt.pt
    ck = torch.load(Path(_td) / "ckpt.pt", map_location="cpu", weights_only=False)
    ck = {"model": ck["model"], "step": 25, "ema": ck["ema"]}

    def loader_model():                                                            # what refcv7_loader.build_model hands back: strict-loaded, then ALL requires_grad False
        torch.manual_seed(123)
        B = _tiny()
        B.load_state_dict(ck["model"])
        for p in B.parameters():
            p.requires_grad_(False)
        return B

    if save_ema is None:
        def save_ema(ema, model, step, path):                                      # LOCAL COPY of refc_v3_train._save_ema_model -- used only when the trainer is not importable
            torch.save({"model": ema.model_state_dict(model), "step": int(step), "ema_decay": ema.decay, "num_updates": int(ema.num_updates), "kind": "ema_model_only"}, path)
        res["save_ema_source"] = "LOCAL COPY (trainer not imported)"
    else:
        res["save_ema_source"] = "the trainer's own _save_ema_model"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        r = extract_core(loader_model(), ck, out, step=25, decay=0.9, save_ema=save_ema, raw_path_prebuilt=None)
        ok["extract_ok_no_problems"] = bool(r["ok"]) and not r["problems"]
        got = torch.load(out / "ckpt_ema_25.pt", map_location="cpu", weights_only=False)["model"]
        ok["ema_file_equals_the_training_EMA_on_every_key"] = set(got) == set(GT) and all(torch.equal(got[k], GT[k]) for k in GT)
        shadow_names = set(E.shadow)
        ok["shadowed_keys_equal_the_shadow"] = all(torch.equal(got[k], E.shadow[k].to(got[k].dtype)) for k in shadow_names)
        live = A.state_dict()
        non_shadow = [k for k in got if k not in shadow_names and k not in ("out_tied.weight", "out_tied.bias")]
        ok["non_shadowed_keys_equal_the_live_value"] = bool(non_shadow) and all(torch.equal(got[k], live[k]) for k in non_shadow)
        ok["frozen_and_int_keys_are_among_them"] = {"frozen.weight", "bn.num_batches_tracked"} <= set(non_shadow)
        ok["tied_alias_equals_its_twin"] = torch.equal(got["out_tied.weight"], got["out.weight"]) and torch.equal(got["out_tied.bias"], got["out.bias"])
        ok["ema_differs_from_live_on_shadowed_floats"] = r["n_float_tensors_ema_differs_from_live"] >= 4
        ok["meta_keys_are_the_trainers"] = set(torch.load(out / "ckpt_ema_25.pt", map_location="cpu", weights_only=False)) == META_KEYS
        ok["raw_file_is_the_milestone_layout"] = set(torch.load(out / "ckpt_25.pt", map_location="cpu", weights_only=False)) == {"model", "step"}
        # ---- refusals ------------------------------------------------------------------------------------------------------------------ #
        try:
            extract_core(loader_model(), ck, out, step=25, decay=0.9, save_ema=save_ema)
            ok["refuses_to_overwrite"] = False
        except FileExistsError:
            ok["refuses_to_overwrite"] = True
    # ---- the DELIBERATE-REGRESSION arms: each must read RED ---------------------------------------------------------------------------------- #
    B = loader_model()                                                                  # (a) NOT restoring the trainable pattern: WeightEMA covers no parameter
    try:
        W = WeightEMA(B, 0.9)
        W.load_state_dict(ck["ema"], B)
        ok["REGRESSION_without_requires_grad_restore_is_refused"] = False
    except EMAStateError:
        ok["REGRESSION_without_requires_grad_restore_is_refused"] = True
    bad = dict(ck, ema=dict(ck["ema"], decay=0.8))                                      # (b) a decay that is not the run's
    with tempfile.TemporaryDirectory() as td:
        try:
            extract_core(loader_model(), bad, Path(td), step=25, decay=0.9, save_ema=save_ema)
            ok["REGRESSION_wrong_decay_is_refused"] = False
        except EMAStateError:
            ok["REGRESSION_wrong_decay_is_refused"] = True
        stale = dict(ck, ema=dict(ck["ema"], num_updates=24))                           # (c) an EMA that is not at the checkpoint's step
        try:
            extract_core(loader_model(), stale, Path(td), step=25, decay=0.9, save_ema=save_ema)
            ok["REGRESSION_ema_not_at_step_is_refused"] = False
        except ValueError:
            ok["REGRESSION_ema_not_at_step_is_refused"] = True
        shadow = {k: v.clone() for k, v in ck["ema"]["shadow"].items()}                 # (d) a corrupted shadow tensor must make the file DIFFER from the training EMA
        shadow["lin1.weight"] = shadow["lin1.weight"] + 1e-3
        corrupt = dict(ck, ema=dict(ck["ema"], shadow=shadow))
        r2 = extract_core(loader_model(), corrupt, Path(td), step=25, decay=0.9, save_ema=save_ema)
        got2 = torch.load(Path(td) / "ckpt_ema_25.pt", map_location="cpu", weights_only=False)["model"]
        ok["REGRESSION_corrupt_shadow_reads_RED_against_the_training_EMA"] = not torch.equal(got2["lin1.weight"], GT["lin1.weight"])
    res["ok"] = all(ok.values())
    return res


def cmd_selftest(a) -> int:
    save = None
    try:
        save, src, _ = _trainer_save_ema()
    except Exception as exc:                                                            # noqa: BLE001
        print(f"[selftest] trainer _save_ema_model NOT importable here ({type(exc).__name__}: {str(exc)[:200]}); using the local copy", file=sys.stderr)
    r = selftest(save)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps({"ok": r["ok"], "save_ema_source": r["save_ema_source"], "failed": [k for k, v in r["checks"].items() if not v]}))
    return 0 if r["ok"] else 1


# ------------------------------------------------------------------------------------------------ the real 30k pair
def check_real_pair(raw_path, ema_path, config_path) -> dict:
    """What `extract` would write, asserted on the REAL pair (key names + shapes + dtypes from the model file; the EMA file's meta; the non-shadow invariants)."""
    import torch
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    ema = torch.load(ema_path, map_location="cpu", weights_only=False)
    rm, em = raw["model"], ema["model"]
    chk: dict = {}
    chk["ema_meta_keys_are_the_trainers"] = set(ema) == META_KEYS
    chk["raw_meta_keys_are_the_milestone_layout"] = set(raw) == {"model", "step"}
    chk["kind"] = ema.get("kind") == "ema_model_only"
    chk["step_equal"] = int(raw["step"]) == int(ema["step"]) == int(ema["num_updates"])
    chk["decay_is_the_runs"] = float(ema["ema_decay"]) == float((cfg.get("ema") or {}).get("decay"))
    chk["key_set_equal"] = set(rm) == set(em)
    shapes = {k: (tuple(rm[k].shape), str(rm[k].dtype)) for k in rm if torch.is_tensor(rm[k])}
    chk["shape_dtype_equal_on_every_tensor"] = all(torch.is_tensor(em.get(k)) and (tuple(em[k].shape), str(em[k].dtype)) == v for k, v in shapes.items())
    differ = [k for k in rm if k in em and torch.is_tensor(rm[k]) and torch.is_tensor(em[k]) and not torch.equal(rm[k], em[k])]
    non_float_differ = [k for k in differ if not rm[k].is_floating_point()]
    non_float_differ += [k for k in rm if k in em and torch.is_tensor(rm[k]) and not rm[k].is_floating_point() and rm[k].dtype != em[k].dtype]
    chk["only_float_tensors_differ"] = not non_float_differ
    chk["ema_differs_from_live_somewhere"] = len(differ) >= 1
    chk["tied_partition_equal"] = _alias_groups(rm) == _alias_groups(em)
    n_shadow_cfg = (cfg.get("ema") or {}).get("n_tensors")
    rep = {"n_keys": len(rm), "n_tensors_differing_from_live": len(differ), "config_ema_n_tensors": n_shadow_cfg,
           "n_tied_groups": len(_alias_groups(rm)), "step": int(raw["step"]), "ema_decay": float(ema["ema_decay"]), "checks": chk,
           "ok": all(chk.values())}
    rep["note"] = ("`n_tensors_differing_from_live` counts STATE-DICT KEYS (a tied parameter counts once per name) and excludes shadowed tensors the EMA happens to equal; "
                   "it is reported beside config.ema.n_tensors, not asserted equal to it")
    return rep


def cmd_check_real(a) -> int:
    r = check_real_pair(a.raw, a.ema, a.config)
    r["raw"], r["ema"], r["raw_md5"], r["ema_md5"] = a.raw, a.ema, _md5(a.raw), _md5(a.ema)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"ok": r["ok"], "failed": [k for k, v in r["checks"].items() if not v], "n_keys": r["n_keys"], "n_differ": r["n_tensors_differing_from_live"]}))
    return 0 if r["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--src", required=True)
    e.add_argument("--config", default=f"{KIT}/ckpt/config.json")
    e.add_argument("--out-dir", default=f"{KIT}/ckpt")
    e.add_argument("--expect-step", type=int, default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--min-commit-gb", type=float, default=4.0)
    e.add_argument("--provenance", nargs="*", default=[])
    s = sub.add_parser("selftest")
    s.add_argument("--out", required=True)
    c = sub.add_parser("check-real")
    c.add_argument("--raw", required=True)
    c.add_argument("--ema", required=True)
    c.add_argument("--config", default=f"{KIT}/ckpt/config.json")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "selftest": cmd_selftest, "check-real": cmd_check_real}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
