# -*- coding: utf-8 -*-
"""Every refusal path --max-speed-input adds, EXERCISED (not asserted in prose).

Each probe carries its POSITIVE control in the same breath: the same call with
the offending condition removed must NOT refuse. A guard that always fires is
indistinguishable from a broken build.
ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

REPO = "C:/Users/Admin/tanitad-wt"
sys.path.insert(0, os.path.join(REPO, "stack"))


def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = by_path("trainer_probe", os.path.join(REPO, "stack", "scripts",
                                           "refc_v3_train.py"))
import torch                                       # noqa: E402
from tanitad.data import v7_labels as v7l          # noqa: E402
from tanitad.refs import max_speed_input as msi    # noqa: E402
from tanitad.refs import refc_v3 as v3             # noqa: E402

V8 = "C:/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz"
V72 = ("C:/Users/Admin/tanitad-wt/_s2build/release/v72/"
       "s2_labels_v7.2_eval.jsonl.gz")
FAIL, N = [], 0


def refuses(fn, want_in, label, exc=SystemExit):
    global N
    N += 1
    try:
        fn()
    except exc as e:                                        # noqa: BLE001
        got = str(e)
        ok = want_in.lower() in got.lower()
        print(("  PASS  " if ok else "  FAIL  ")
              + f"{label}\n         -> {got[:190]}")
        if not ok:
            FAIL.append(f"{label}: message lacks {want_in!r}")
        return
    except Exception as e:                                  # noqa: BLE001
        print(f"  FAIL  {label}: raised {type(e).__name__} not "
              f"{exc.__name__}: {str(e)[:160]}")
        FAIL.append(label)
        return
    print(f"  FAIL  {label}: DID NOT REFUSE")
    FAIL.append(label)


def allows(fn, label):
    global N
    N += 1
    try:
        fn()
    except Exception as e:                                  # noqa: BLE001
        print(f"  FAIL  CONTROL {label}: refused when it must not: "
              f"{type(e).__name__} {str(e)[:180]}")
        FAIL.append("control " + label)
        return
    print(f"  PASS  CONTROL {label}: allowed")


def args(*extra, arm="hier"):
    return tr.build_parser().parse_args(
        ["--arm", arm, "--size", "small", "--out", "/x",
         "--v2-cache", "/x/cache"] + list(extra))


def pin(a):
    base = v3.refc_v3_sized_config(a.size, hier=(a.arm == "hier"))
    return tr._pin_trainer_cfg(base, a)


# ---- fixtures for the loader-level probes ------------------------------- #
def _label(clip_id, smi):
    lab = object.__new__(v7l.V7Label)
    object.__setattr__(lab, "clip_id", clip_id)
    object.__setattr__(lab, "_oracle", {"speed_max_input": smi})
    return lab


def _manifest():
    return v7l.LabelManifest(path="/fake", md5="deadbeef", n_records=1,
                             schema_version="x", vocab="v7.0",
                             allow_oracle_nav=True)


def _ds(labels):
    """The minimum surface `enable_max_speed` reads: v7_by_sid / index /
    episodes. A real V3Dataset would need a corpus; the method under test
    touches only these three."""
    d = types.SimpleNamespace()
    d.v7_by_sid = {i: lab for i, lab in enumerate(labels)}
    d.episodes = [types.SimpleNamespace(episode_id=i) for i in range(len(labels))]
    d.index = [(i, 0) for i in range(len(labels))]
    return d


def _run_enable(labels, mode="quantized"):
    return tr.V3Dataset.enable_max_speed(_ds(labels), _manifest(), mode)


GOOD = {"v_max_ms": 11.19, "units": "m/s", "v_max_bucket_kmh": 50,
        "v_max_bucket_ms": 13.8889,
        "bucket_steps_kmh": list(msi.POSTED_LIMIT_STEPS_KMH)}


def _build_flat():
    cfg = v3.refc_v3_flat_config()
    cfg.max_speed_input = True
    return v3.RefCV3Model(cfg)


def _smoke_model(on):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.max_speed_input = on
    torch.manual_seed(0)
    return cfg, v3.RefCV3Model(cfg).eval()


def _frames(cfg, b=2):
    g = torch.Generator().manual_seed(3)
    h, w = cfg.core.encoder.image_hw()
    return torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                      generator=g)


def _supplied_no_seam():
    cfg, m = _smoke_model(False)
    with torch.no_grad():
        m(_frames(cfg), nav_cmd=torch.tensor([0, 1]),
          v0=torch.tensor([3.0, 7.0]), steps=2,
          v_max_ms=torch.tensor([13.9, 25.0]))


def _seam_no_supply():
    cfg, m = _smoke_model(True)
    with torch.no_grad():
        m(_frames(cfg), nav_cmd=torch.tensor([0, 1]),
          v0=torch.tensor([3.0, 7.0]), steps=2)


def _both_present():
    cfg, m = _smoke_model(True)
    with torch.no_grad():
        m(_frames(cfg), nav_cmd=torch.tensor([0, 1]),
          v0=torch.tensor([3.0, 7.0]), steps=2,
          v_max_ms=torch.tensor([13.9, 25.0]),
          v_max_valid=torch.tensor([1.0, 1.0]))


print("=" * 78)
print("1. THE DEAD-FLAG REFUSALS (trainer, before any GPU work)")
print("=" * 78)
refuses(lambda: tr._check_max_speed_args(args("--max-speed-input")),
        "--v7-labels", "--max-speed-input without --v7-labels")
allows(lambda: tr._check_max_speed_args(args("--max-speed-input",
                                             "--v7-labels", V8)),
       "--max-speed-input WITH --v7-labels")
allows(lambda: tr._check_max_speed_args(args()), "no flag at all")
refuses(lambda: tr._check_max_speed_args(
            args("--max-speed-input", "--v7-labels", V8,
                 "--eval-cache", "/x/e", "--eval-every", "100")),
        "--eval-labels", "in-training eval without --eval-labels")

print()
print("=" * 78)
print("2. THE INERT-MODE REFUSAL (a knob that selects nothing)")
print("=" * 78)
refuses(lambda: pin(args("--max-speed-mode", "raw")),
        "INERT", "--max-speed-mode raw WITHOUT --max-speed-input")
allows(lambda: pin(args("--max-speed-input", "--max-speed-mode", "raw")),
       "--max-speed-mode raw WITH --max-speed-input")

print()
print("=" * 78)
print("3. THE FLAT-ARM REFUSAL (a seam with nowhere to inject)")
print("=" * 78)
refuses(lambda: pin(args("--max-speed-input", arm="flat")),
        "FLAT", "--max-speed-input on --arm flat (trainer)")
refuses(_build_flat, "FLAT",
        "max_speed_input=True on a flat RefCV3Config (model)", exc=ValueError)

print()
print("=" * 78)
print("4. THE UNITS REFUSAL -- and the TRAINER's error NAMES THE FIELD")
print("=" * 78)
refuses(lambda: msi.read_max_speed_field({"v_max_ms": 13.9}),
        "units", "read_max_speed_field on an undeclared payload",
        exc=msi.MaxSpeedUnitsMissing)
allows(lambda: msi.read_max_speed_field({"v_max_ms": 13.9, "units": "m/s"}),
       "read_max_speed_field on a DECLARED payload")
_no_units = {k: v for k, v in GOOD.items() if k != "units"}
refuses(lambda: _run_enable([_label("clip-A", _no_units)]),
        "speed_max_input",
        "enable_max_speed on a record whose units are UNDECLARED")
allows(lambda: _run_enable([_label("clip-A", GOOD)]),
       "enable_max_speed on a DECLARED record")

print()
print("=" * 78)
print("5. THE SHIPPED-VALUE CROSS-CHECKS (ladder and bucket)")
print("=" * 78)
_bad_ladder = dict(GOOD, bucket_steps_kmh=[30, 50, 70, 100])
refuses(lambda: _run_enable([_label("clip-B", _bad_ladder)]),
        "two experiments", "a record shipping a DIFFERENT ladder")
_bad_bucket = dict(GOOD, v_max_bucket_kmh=70)
refuses(lambda: _run_enable([_label("clip-C", _bad_bucket)]),
        "shipped", "a record whose SHIPPED bucket disagrees with the snap")

print()
print("=" * 78)
print("6. THE EMPTY-CHANNEL REFUSAL (the field is a v8 addition)")
print("=" * 78)
refuses(lambda: _run_enable([_label("clip-D", None),
                             _label("clip-E", None)]),
        "v8 addition", "enable_max_speed where NO clip carries the block")

print()
print("=" * 78)
print("7. THE MODEL'S TWO-WAY SILENT-DROP REFUSAL")
print("=" * 78)
refuses(_supplied_no_seam, "SILENTLY",
        "v_max_ms supplied to a build with no seam", exc=ValueError)
refuses(_seam_no_supply, "no v_max_ms reached",
        "seam built but nothing fed", exc=ValueError)
allows(_both_present, "seam built AND fed")

print()
print("=" * 78)
print("8. THE ORACLE GATE (the record declares oracle:true / ego-future)")
print("=" * 78)
_m_noflag = v7l.LabelManifest(path="/f", md5="x", n_records=1,
                              schema_version="x", vocab="v7.0",
                              allow_oracle_nav=False)
refuses(lambda: v7l.oracle_max_speed(_label("clip-F", GOOD), _m_noflag),
        "ORACLE", "oracle_max_speed without allow_oracle_nav",
        exc=v7l.OracleNavRefused)
allows(lambda: v7l.oracle_max_speed(_label("clip-F", GOOD), _manifest()),
       "oracle_max_speed WITH allow_oracle_nav")

print("=" * 78)
print(f"REFUSAL PROBE: {N} probes, "
      + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED"))
for f in FAIL:
    print("   -", f)
sys.exit(1 if FAIL else 0)
