# -*- coding: utf-8 -*-
"""THE ROLLABILITY BATTERY for --max-speed-input (E16).

The bar `--tac-goal-tok-head` had to clear before it was allowed into the 40k run:
  1. 0 missing / 0 unexpected under strict=True on a REAL trained checkpoint;
  2. `param_breakdown` accounts for the new params EXACTLY
     (recorded == rebuilt, and the lines sum to the total);
  3. the REAL `refcv3_arm.cross_check_config` PASSES -- not a reimplementation.

Both arms are run in one breath: the ON checkpoint AND the OFF control, so a
pass is a measurement rather than a battery that cannot fail.
ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

REPO = "C:/Users/Admin/tanitad-wt"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, REPO)


def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm = by_path("refcv3_arm_battery",
              os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
from tanitad.refs import refc_v3 as v3          # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def battery(run_dir: str, label: str, expect_ms: bool) -> dict:
    print("=" * 78)
    print(f"ARM {label}   {run_dir}")
    print("=" * 78)
    ck_p = os.path.join(run_dir, "ckpt.pt")
    cfg_p = os.path.join(run_dir, "config.json")
    with open(cfg_p, encoding="utf-8") as fh:
        config = json.load(fh)

    # ---- what the RECORD says --------------------------------------------
    rec_bd = {k: int(v) for k, v in config["param_breakdown"].items()}
    seam = (config.get("seams") or {}).get("max_speed_input")
    if seam is None:                       # the stamp may sit at top level
        for k in ("seam_stamp", "seams", "hierarchy_seams"):
            if isinstance(config.get(k), dict) and "max_speed_input" in config[k]:
                seam = config[k]["max_speed_input"]
    print(f"  record: param_breakdown has {len(rec_bd)} lines, "
          f"max_speed_inject={'max_speed_inject' in rec_bd}")
    print(f"  record: max_speed_input flag = {config.get('max_speed_input')}, "
          f"mode = {config.get('max_speed_mode')}")
    check(("max_speed_inject" in rec_bd) == expect_ms,
          f"{label}: recorded ledger {'names' if expect_ms else 'does NOT name'}"
          f" max_speed_inject")

    # ---- (3) THE REAL INSTRUMENT ----------------------------------------- #
    # load_model runs rebuild_config -> cross_check_config -> strict load and
    # SystemExits on any contradiction. Calling it IS the cross-check.
    model, cfg, targs, prov = arm.load_model(ck_p, cfg_p, device="cpu")
    checks = prov["config_cross_checks"]
    conflicts = [k for k, v in checks.items()
                 if v["config_json"] is not None and v["rebuilt"] != v["config_json"]]
    check(not conflicts,
          f"{label}: REAL refcv3_arm.cross_check_config PASS on "
          f"{len(checks)} facts ({sorted(checks)}); conflicts={conflicts}")

    # ---- (1) 0 missing / 0 unexpected ------------------------------------ #
    sd = prov["state_dict_load"]
    check(sd["missing_keys"] == [] and sd["unexpected_keys"] == [],
          f"{label}: strict load 0 missing / 0 unexpected "
          f"(missing={sd['missing_keys'][:4]}, "
          f"unexpected={sd['unexpected_keys'][:4]}, "
          f"tolerated_inert={sd.get('tolerated_inert_buffers')})")
    # ...and LITERALLY strict=True, on a freshly rebuilt model, so the claim
    # does not rest on the adapter's strict=False + report.
    fresh = v3.RefCV3Model(cfg)
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    try:
        fresh.load_state_dict(ck["model"], strict=True)
        strict_ok, strict_err = True, ""
    except RuntimeError as ex:                              # noqa: BLE001
        strict_ok, strict_err = False, str(ex)[:400]
    check(strict_ok, f"{label}: model.load_state_dict(..., strict=True) -> "
                     f"{'OK' if strict_ok else strict_err}")

    # ---- (2) param accounting -------------------------------------------- #
    reb_bd = {k: int(v) for k, v in v3.param_breakdown_v3(model).items()}
    check(reb_bd == rec_bd,
          f"{label}: recorded ledger == rebuilt ledger "
          f"(only-in-rebuilt={sorted(set(reb_bd) - set(rec_bd))}, "
          f"only-in-record={sorted(set(rec_bd) - set(reb_bd))})")
    lines = sum(v for k, v in reb_bd.items() if k != "total")
    check(lines == reb_bd["total"],
          f"{label}: ledger lines sum to total ({lines} == {reb_bd['total']})")

    ms_keys = sorted(k for k in ck["model"] if "max_speed" in k)
    check(bool(ms_keys) == expect_ms,
          f"{label}: checkpoint {'carries' if expect_ms else 'carries NO'} "
          f"max_speed weights -> {ms_keys}")
    if expect_ms:
        built = sum(p.numel() for p in model.max_speed_cond.parameters())
        check(reb_bd.get("max_speed_inject") == built,
              f"{label}: max_speed_inject line {reb_bd.get('max_speed_inject')} "
              f"== measured module params {built}")
        check(cfg.max_speed_input is True and cfg.max_speed_cfg.enabled,
              f"{label}: the ROLLED config carries the flag "
              f"(max_speed_input={cfg.max_speed_input}, "
              f"mode={cfg.max_speed_cfg.mode})")
    else:
        check(getattr(model, "max_speed_cond", "absent") is None,
              f"{label}: rebuilt model has no conditioner")
    return reb_bd


SP = os.path.dirname(os.path.abspath(__file__))
on = battery(os.path.join(SP, "msi_on"), "ON  (--max-speed-input)", True)
off = battery(os.path.join(SP, "msi_off"), "OFF (control)", False)

print("=" * 78)
d = on["total"] - off["total"]
line = on.get("max_speed_inject")
check(d == line,
      f"CROSS-ARM: ON total - OFF total = {d} == the max_speed_inject line "
      f"{line} -- the opt-in costs EXACTLY the conditioner and nothing else")
print("=" * 78)
print("BATTERY RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})")
for f in FAIL:
    print("   -", f)
sys.exit(1 if FAIL else 0)
