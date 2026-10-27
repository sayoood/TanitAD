"""pin_digest.py -- flags-absent identity of the ARGV paths round 3 touches, pristine (aeb5446 = 9506371 stack/) vs
modified, in the stepid_digest report format (``{"arms": {name: {"out": {...}, "state": {...}, ...}}}``):

  argv_pins      -- refc_v3_train._pin_refcv8 on refcv8 argvs WITHOUT --r9-train-only (test_refcv8_emit_schedule's
                    accepted + refused argvs, and refcv8-r101-s0's committed launch argv): vars(cfg.refcv8) or the
                    refusal text; map_only_ft.pin_args on the r8 map-only argv;
  profiles       -- every launch-gate profile EXCEPT refcv9-warm, canonical JSON per key (forbidden_levers separately:
                    round 3 adds ONE entry to refcv8's, inherited by refcv8-nt / refcv8-mapft / v8f / v8f_steer);
  gate_argv      -- profile_argv_rules of refcv8 / refcv8-mapft on refcv8-r101-s0's launch argv (+ the map-only flag);
  assert_tool    -- assert_map_only_finetune.main (legacy form, no --sets) on four fixed synthetic checkpoint pairs: the
                    exit code and the full JSON (paths are the same fixed files for both trees).

  python -X utf8 pin_digest.py --tree <tree> --out <json> [--fixtures <dir>]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import contextlib
import sys
import types
from pathlib import Path


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _fixtures(d: Path):
    import torch
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(5)
    sd = {"core.decoder.w": torch.randn(3, 4, generator=g), "core.encoder.bn.running_mean": torch.zeros(4),
          "core.encoder.bn.num_batches_tracked": torch.tensor(7),
          "_map_hires.lift.proj.weight": torch.randn(2, 2, generator=g),
          "_map_hires.refine.cls.weight": torch.randn(8, 2, generator=g),
          "_map_hires.near.lift.unobserved": torch.zeros(2),
          "_map_hires.near_refine.0.c1.weight": torch.randn(2, 2, generator=g)}
    src = {"model": sd, "step": 48852, "kind": "ema_model_only"}
    pairs = {"identical": copy.deepcopy(src)}
    m = copy.deepcopy(src)
    m["model"]["_map_hires.refine.cls.weight"] += 0.5
    pairs["map_change"] = m
    n = copy.deepcopy(src)
    n["model"]["core.decoder.w"] = torch.nextafter(n["model"]["core.decoder.w"], n["model"]["core.decoder.w"] + 1)
    pairs["planner_change"] = n
    raw = copy.deepcopy(sd)
    raw["_map_hires.refine.cls.weight"] = raw["_map_hires.refine.cls.weight"] + 1.0
    pairs["rolling_with_ema"] = {"model": raw, "opt": {}, "step": 120, "data_pos": {}, "param_names": [],
                                 "ema": {"schema": "weight_ema/1", "decay": 0.999, "num_updates": 120,
                                         "shadow": {"_map_hires.refine.cls.weight":
                                                    sd["_map_hires.refine.cls.weight"] + 0.1}}}
    if not (d / "src.pt").exists():
        torch.save(src, d / "src.pt")
        for k, v in pairs.items():
            torch.save(v, d / f"dst_{k}.pt")
    return d / "src.pt", {k: d / f"dst_{k}.pt" for k in pairs}


def run(tree: Path, fixtures: Path) -> dict:
    stack = tree / "stack"
    for p in (str(stack), str(stack / "scripts"), str(stack / "tests")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    import refc_v3_train as T
    import launch_gate as LG
    import assert_map_only_finetune as AT
    from tanitad.refs import refcv8_conditioning as r8c
    from tanitad.train import map_only_ft as M
    res = {"tanitad_file": tanitad.__file__, "torch": torch.__version__, "arms": {}}

    # ---- argv pins ------------------------------------------------------------------------------------------------ #
    ON8 = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05",
           "--r8-n-alloc", "8", "--w-r8-alloc-l1", "1.0"]
    launch = json.loads((stack / "ops" / "runs.d" / "refcv8-r101-s0.argv.json").read_text(encoding="utf-8"))["argv"]
    argvs = {"emit2000_warm": ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "2000", "--init-from", "c.pt"],
             "emit_tiny": ON8 + ["--r8-alloc-emit"],
             "emit_warm_nostart": ON8 + ["--r8-alloc-emit", "--init-from", "c.pt"],
             "emit0": ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0"],
             "emit0_warm": ON8 + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0", "--init-from", "c.pt"],
             "start_no_emit": ON8 + ["--r8-alloc-emit-start", "2000"],
             "tf_default": ON8, "tf_hold": ON8 + ["--r8-tf-start", "0.25", "--r8-tf-end", "0.25"],
             "refcv8_r101_s0_launch": list(launch)}
    out = {}
    for k, a in argvs.items():
        args = T.build_parser().parse_args(a)
        cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
        try:
            T._pin_refcv8(cfg, args)
            out[f"pin8.{k}"] = repr(sorted(vars(cfg.refcv8).items()))
        except SystemExit as e:
            out[f"pin8.{k}"] = "REFUSED: " + str(e)
    mapft = ["--arm", "hier", "--out", "X", "--refcv8", "--map-hires", "on", "--map-hires-near-lift-m", "20",
             "--map-hires-near-refine-blocks", "1", "--w-map-hires", "1.0", "--opt", "dd", "--init-from", "e.pt",
             "--conflict-detector", "off", "--r8-train-only-map"]
    for k, a in {"mapft_ok": mapft, "mapft_no_init": [x for x in mapft if x not in ("--init-from", "e.pt")]}.items():
        try:
            M.pin_args(T.build_parser().parse_args(a))
            out[f"mof_pin.{k}"] = "OK"
        except SystemExit as e:
            out[f"mof_pin.{k}"] = "REFUSED: " + str(e)
    res["arms"]["argv_pins"] = {"out": out, "state": {}, "n_out": len(out), "n_state": 0, "loss": None}

    # ---- the profiles (every one but refcv9-warm) ------------------------------------------------------------------ #
    out, fl = {}, {}
    for name, prof in LG.PROFILES.items():
        if name == "refcv9-warm":
            continue
        for key in sorted(prof):
            if key == "forbidden_levers":
                fl[name] = [[list(e[0]), e[1]] for e in prof[key]]
                out[f"{name}.forbidden_levers_minus_r9"] = _h([e for e in fl[name] if "--r9-train-only" not in e[0]])
                out[f"{name}.forbidden_levers_r9_entries"] = repr([e for e in fl[name] if "--r9-train-only" in e[0]])
            else:
                out[f"{name}.{key}"] = _h(prof[key])
    res["arms"]["profiles"] = {"out": out, "state": {}, "n_out": len(out), "n_state": 0, "loss": None}

    # ---- the gate's argv rules on refcv8's committed launch argv --------------------------------------------------- #
    out = {}
    for name, a in {"refcv8": list(launch), "refcv8-mapft": list(launch) + ["--r8-train-only-map"]}.items():
        r, pend = LG.profile_argv_rules(LG.PROFILES[name], a, {}, [])
        out[f"{name}.refusals"] = repr(r)
        out[f"{name}.pending"] = repr(pend)
        out[f"{name}.r9_schedule"] = repr(LG.r9_schedule_reasons(LG.PROFILES[name], a)) \
            if hasattr(LG, "r9_schedule_reasons") else "absent"
    res["arms"]["gate_argv"] = {"out": out, "state": {}, "n_out": len(out), "n_state": 0, "loss": None}

    # ---- the assert tool, legacy form ------------------------------------------------------------------------------ #
    src, dsts = _fixtures(fixtures)
    out = {}
    for k, d in dsts.items():
        j = fixtures / f"_out_{k}.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = AT.main(["--src", str(src), "--dst", str(d), "--json", str(j)])
        out[f"at.{k}.exit"] = repr(rc)
        out[f"at.{k}.json"] = _h(json.loads(j.read_text(encoding="utf-8")))
        out[f"at.{k}.stdout"] = hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()
    res["arms"]["assert_tool"] = {"out": out, "state": {}, "n_out": len(out), "n_state": 0, "loss": None}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="F:/refcv9_code3/work/at_fixtures")
    a = ap.parse_args()
    r = run(Path(a.tree), Path(a.fixtures))
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("tanitad from", r["tanitad_file"], {k: v["n_out"] for k, v in r["arms"].items()})
