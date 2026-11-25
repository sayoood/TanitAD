#!/usr/bin/env python3
"""CTRL-CLIP (PREREG_SCORERFT_PICK.md 3.2) on CPU fixtures: refe/clip_probe.py and the trainer's new grad_norm_pre / clip_coef log keys (train.py).

    python test_clip_probe.py            # whole file (module-level fixtures); prints ZZCLIPPROBE_PASS <n>

  V1  known norms 2.0 / 0.5 record b = 1 / 0, the wrapped clip is bit-identical to the bare clip; RED: CLIP_PROBE_MUTATE=post_clip (the norm recorded after clipping) turns V1, and the probe, INVALID
  the pass-through wrapper on the real trainer: K norms, stopped before step K, no checkpoint, the trainer's logged keys == the wrapper's
  the states CLIP-0 / CLIP-SAME / CLIP-T / CLIP-INVALID from the paired read (n_on, n_off, occ, cmin)
  2b  NO BEHAVIOUR CHANGE: the parameters after N steps are bit-identical with and without the new logging (a source-mutated copy of train.py with the two keys and the capture removed);
      the control (current vs current) is identical too, and a copy whose optimiser step is skipped is DIFFERENT (the comparison can fail).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import clip_probe as CP  # noqa: E402
import test_p6_levers as P6  # noqa: E402
import test_rank_loss as TR  # noqa: E402

# THE PROBE'S READER AND ITS IMPORT CLOSURE (MM 2026-11-25; pod gate FAIL D3: `clip_probe.judge` imports scorerft_read from <package>/eval/scorerft, which the package did not carry, so this file raised
# ModuleNotFoundError in G-SUITE while the probe itself ran from a side copy).  They are NAMED HERE AS LITERALS so that `p7_ship_list derive` (a static scan of every string constant that resolves to a file) lists
# them and `build_pkg.py` ships them with every profile whose suite carries this test (P / Ps); scorerft_read imports p5c_read and field_read by sys.path, field_read imports field_rawdac (MEASURED, import trace).
# A reader import added later is not found by the scan: `main` asserts the whole closure imports from the package tree and that no other eval/ module is loaded.
SCORERFT_READ_FILES = ("eval/scorerft/scorerft_read.py", "eval/p5c/p5c_read.py", "eval/field/field_read.py", "eval/field/field_rawdac.py")

N = 0


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def form_rec(binding, norms=None, coefs=None, **over):
    k = len(binding)
    norms = norms or [2.0 if b else 0.5 for b in binding]
    coefs = coefs or [(1.0 / (n + 1e-6)) if b else 1.0 for b, n in zip(binding, norms)]
    d = {"argv": ["--steps", "1000", "--out", "/x"], "argv_sha256": "0" * 64, "rank_w": 0.25, "k": k, "stopped_before_step_k": True, "rc": None, "norms": norms, "coefs": coefs, "binding": list(binding),
         "logged_norm": norms, "logged_coef": coefs, "order": {}, "n_train_rows": k, "resume_events": 0, "checkpoints_written": [], "metrics_path": "", "onpolicy_step0": None}
    d.update(over)
    return d


def chk_v1_and_red():
    r = CP.selfcheck_v1()
    ok(r["pass"] and r["detail"]["norm_2.0"]["binding"] is True and r["detail"]["norm_0.5"]["binding"] is False and r["detail"]["norm_2.0"]["grads_bit_identical"], r)
    old = os.environ.get(CP.MUTATE)
    os.environ[CP.MUTATE] = "post_clip"
    try:
        bad = CP.selfcheck_v1()
    finally:
        if old is None:
            os.environ.pop(CP.MUTATE, None)
        else:
            os.environ[CP.MUTATE] = old
    ok(not bad["pass"] and bad["detail"]["norm_2.0"]["recorded_norm"] < 1.01 and bad["detail"]["norm_2.0"]["ok"] is False, bad)       # RED: the post-clip norm of a 2.0 gradient reads ~1.0
    ok(abs(CP._coef(torch.tensor(2.0), 1.0) - 1.0 / (2.0 + 1e-6)) < 1e-6 and CP._coef(torch.tensor(0.5), 1.0) == 1.0, "torch's own coefficient")


def _mini_c_metrics(td, rows_by_step: dict, path_name="banked.jsonl"):
    p = Path(td) / path_name
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for s in sorted(rows_by_step):
            f.write(json.dumps(rows_by_step[s]) + "\n")
    return str(p)


def chk_states_and_invalid(td):
    """the paired read on synthetic form records with a real banked-rows stand-in: CLIP-0, CLIP-SAME occ=a/8, CLIP-T on / off, CLIP-INVALID (V2, V3, V4 and V1 each)."""
    k = 8
    order = {str(s): {"epoch": 0, "pos": 256 * (s + 1), "samples": 256 * (s + 1), "batch": 4, "lr": 5e-5} for s in range(k)}
    base_row = lambda s, extra=None: {"step": s, "epoch": 0, "pos": 256 * (s + 1), "samples": 256 * (s + 1), "batch": 4, "lr": 5e-5, "traj_L1": 0.5 + s, "score": 0.1, "winners": 4, "assign_d": 0.0, "scorer_cov": 1.0,
                                       "pos_L1": 0.1, "yaw_L1": 0.1, **{f"bce_{h}": 0.1 for h in ("nc", "dac", "ep", "ttc", "c", "ddc")}, **(extra or {})}   # noqa: E731
    banked = _mini_c_metrics(td, {s: base_row(s) for s in range(k)})
    mk = lambda path_name, extra=None: _mini_c_metrics(td, {s: base_row(s, extra) for s in range(k)}, path_name)    # noqa: E731
    mc, mp = mk("fc.jsonl"), mk("fp.jsonl", {"rank_L": 0.3, "rank_w": 0.25, "grad_norm_pre": 0.5, "clip_coef": 1.0})
    v1 = {"id": "V1", "pass": True, "fails": []}

    def judge(bc, bp, **over):
        fc = form_rec(bc, order=order, metrics_path=mc)
        fp = form_rec(bp, order=order, metrics_path=mp, **over)
        return CP.judge(fc, fp, banked, k, check_c_sha=False, v1=v1)
    z = judge([0] * 8, [0] * 8)
    ok(z["valid"] and z["state"] == "CLIP-0" and z["n_on"] == 0 and z["n_off"] == 0 and z["pl"] is None, z)
    s = judge([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0])
    ok(s["state"] == "CLIP-SAME occ=2/8" and s["occ_P"] == s["occ_C"] == 2, s["state"])
    t = judge([1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
    ok(t["state"].startswith("CLIP-T on=2 off=0 occ_P=3/8 occ_C=1/8 cmin_P=") and t["pl"] == "PL4" and t["n_on"] == 2, t["state"])
    off = judge([1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0])
    ok(off["state"].startswith("CLIP-T on=0 off=1 ") and off["n_off"] == 1, off["state"])
    ok(abs(t["cmin_P"] - 1.0 / (2.0 + 1e-6)) < 1e-9 and t["N0_P"] == 2.0 and t["N0_C"] == 2.0, t)
    # --- validity failures each give CLIP-INVALID / PL3
    inv = CP.judge(form_rec([0] * 8, order=order, metrics_path=mc), form_rec([0] * 8, order=order, metrics_path=mp), banked, k, check_c_sha=False, v1={"id": "V1", "pass": False, "fails": ["x"]})
    ok(inv["state"] == "CLIP-INVALID" and inv["pl"] == "PL3" and not inv["valid"], inv["state"])
    bad_order = {**order, "3": {**order["3"], "pos": 999}}
    v3 = CP.judge(form_rec([0] * 8, order=order, metrics_path=mc), form_rec([0] * 8, order=bad_order, metrics_path=mp), banked, k, check_c_sha=False, v1=v1)
    ok(v3["state"] == "CLIP-INVALID" and not v3["validity"]["V3"]["pass"] and v3["validity"]["V3"]["detail"]["P"]["steps_differing"] == [3], v3["validity"]["V3"])
    nan = judge([0] * 8, [0] * 8, norms=[float("nan")] + [0.5] * 7)
    ok(nan["state"] == "CLIP-INVALID" and not nan["validity"]["V4"]["pass"], nan["validity"]["V4"])
    for over in ({"resume_events": 1}, {"checkpoints_written": ["model_step24.pt"]}, {"stopped_before_step_k": False}, {"logged_norm": [9.0] * 8}):
        r = judge([0] * 8, [0] * 8, **over)
        ok(r["state"] == "CLIP-INVALID" and not r["validity"]["V4"]["pass"], (over, r["validity"]["V4"]))
    drift = _mini_c_metrics(td, {s_: dict(base_row(s_, {"rank_L": 0.3, "rank_w": 0.25}), traj_L1=(0.5 + s_) + (1e-12 if s_ == 0 else 0.0)) for s_ in range(k)}, "fp_drift.jsonl")
    v2 = CP.judge(form_rec([0] * 8, order=order, metrics_path=mc), form_rec([0] * 8, order=order, metrics_path=drift), banked, k, check_c_sha=False, v1=v1)
    ok(v2["state"] == "CLIP-INVALID" and not v2["validity"]["V2"]["pass"], v2["validity"]["V2"])                   # a step-0 metric off by 1e-12: C is not reusable
    ok(not CP.judge(form_rec([0] * 8, order=order, metrics_path=mc), form_rec([0] * 8, order=order, metrics_path=mp), banked, k, check_c_sha=True, v1=v1)["validity"]["V2"]["pass"], "the C sha is enforced when asked")


def _run_form(td, tag, extra=(), k=2, steps=6):      # accum 1 and k = 2 of a 4-sample epoch: the stop comes before the epoch ends (an epoch end writes a snapshot)
    bank = P6.make_bank(td)
    import online_labeller as OL
    old = OL.synthetic_labels
    OL.synthetic_labels = TR._rich_labels()
    try:
        return CP.run_form(P6.argv_p6(bank, str(td / f"o_{tag}"), steps=steps, accum=1, extra=extra), td / f"scratch_{tag}", k=k)
    finally:
        OL.synthetic_labels = old


def chk_wrapper_on_the_real_trainer(td):
    c = _run_form(td, "c", ("--rank-w", "0.0"))
    p = _run_form(td, "p", ("--rank-w", "0.5", "--rank-heads", "pick"))
    for f in (c, p):
        ok(len(f["norms"]) == 2 and all(math.isfinite(x) and x > 0 for x in f["norms"]) and f["stopped_before_step_k"] and f["checkpoints_written"] == [] and f["resume_events"] == 0, f)
        ok(f["logged_norm"] == f["norms"][:len(f["logged_norm"])] and len(f["logged_norm"]) == 2 and f["logged_coef"] == f["coefs"][:2], f)                    # the trainer's keys == the wrapper's, bit for bit
        ok(all((c_ < 1.0) == b and 0.0 < c_ <= 1.0 for c_, b in zip(f["coefs"], f["binding"])), f)
        ok(all(f["order"][str(s)]["pos"] > 0 for s in range(2)), f["order"])
    ok(c["rank_w"] == 0.0 and p["rank_w"] == 0.5, (c["rank_w"], p["rank_w"]))
    ok(p["norms"][0] != c["norms"][0], "the pick term changes the step-0 global norm (the first global measure of the term's size)")
    # the full read on the two REAL forms, C's own rows standing in for the banked file
    out = CP.judge(c, p, c["metrics_path"], 2, check_c_sha=False)
    ok(out["valid"] is True and out["state"].split()[0] in ("CLIP-0", "CLIP-SAME", "CLIP-T") and out["k"] == 2, (out["state"], out["validity"]))
    ok(out["N0_C"] == c["norms"][0] and out["N0_P"] == p["norms"][0], out)
    old = os.environ.get(CP.MUTATE)
    os.environ[CP.MUTATE] = "post_clip"
    try:
        bad = CP.judge(c, p, c["metrics_path"], 2, check_c_sha=False)                                              # RED: the mutated V1 turns the whole probe INVALID
    finally:
        if old is None:
            os.environ.pop(CP.MUTATE, None)
        else:
            os.environ[CP.MUTATE] = old
    ok(bad["state"] == "CLIP-INVALID" and bad["pl"] == "PL3", bad["state"])
    # the probe wrote no checkpoint and the scratch is only the probe's own
    ok(not list(Path(td).glob("scratch_*/**/*.pt")) and not list(Path(td).glob("o_*/*.pt")), "no checkpoint anywhere")


def _final_tensors(out_dir: Path) -> dict:
    cks = sorted(out_dir.glob("*.pt"))
    ok(cks, f"no checkpoint in {out_dir}")
    ck = torch.load(str(cks[-1]), map_location="cpu", weights_only=False)
    flat = {}

    def walk(x, pre):
        if isinstance(x, torch.Tensor):
            flat[pre] = x.detach().clone()
        elif isinstance(x, dict):
            for k_, v in x.items():
                walk(v, f"{pre}/{k_}")
    walk(ck, "")
    ok(flat, "no tensors")
    return flat


def _load_variant(src_text: str, path: Path):
    path.write_text(src_text, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("train", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_with(mod, td, tag, argv_extra=()):
    import online_labeller as OL
    old_mod, old_lab = sys.modules.get("train"), OL.synthetic_labels
    sys.modules["train"] = mod
    OL.synthetic_labels = TR._rich_labels()
    try:
        bank = P6.make_bank(td)
        argv = P6.argv_p6(bank, str(td / f"id_{tag}"), steps=3, accum=2, extra=argv_extra)
        rc = mod.run(mod.build_parser().parse_args(argv))
    finally:
        OL.synthetic_labels = old_lab
        if old_mod is None:
            sys.modules.pop("train", None)
        else:
            sys.modules["train"] = old_mod
    ok(rc in (0, None), (tag, rc))
    return _final_tensors(td / f"id_{tag}")


def _clip_coef_f32(g):
    """The trainer's clip_coef recomputed the way train.py computes it: in FLOAT32 on a tensor (gn_pre is a float32 tensor there). EXACT equality is the bar, no tolerance."""
    return float(torch.clamp(1.0 / (torch.tensor(g, dtype=torch.float32) + 1e-6), max=1.0))


def _clip_coef_f64(g):
    """The OLD recomputation: 1.0 / (g + 1e-6) evaluated in python DOUBLE, cast to float32 only afterwards. Kept ONLY as the RED arm."""
    return float(torch.clamp(torch.tensor(1.0 / (g + 1e-6)), max=1.0))


def chk_clip_coef_precision():
    """The precision of the clip_coef recomputation: the float32 path is bit-exact against the trainer's tensor op on float32 norms; the old double path is not (RED)."""
    gen = torch.Generator().manual_seed(20261124)
    g32 = 1.0 + 9.0 * torch.rand(5000, generator=gen, dtype=torch.float32)  # float32 norms in [1, 10)
    trainer = torch.clamp(1.0 / (g32 + 1e-6), max=1.0)  # the trainer's op: float32 tensor in, float32 tensor out
    vals = [float(v) for v in g32]  # the logged form: float(gn_pre), a python float holding the float32 value exactly
    bad32 = sum(1 for v, t in zip(vals, trainer.tolist()) if _clip_coef_f32(v) != t)
    ok(bad32 == 0, f"CONTROL: the float32 recomputation is bit-exact on 5000 float32 norms (mismatches {bad32})")
    bad64 = sum(1 for v, t in zip(vals, trainer.tolist()) if _clip_coef_f64(v) != t)
    ok(bad64 >= 1, f"RED: the old double-precision recomputation DIFFERS from the trainer on at least one norm (mismatches {bad64})")


def chk_logging_changes_no_behaviour(td):
    """2b: the parameters after 3 steps are bit-identical with and without the grad_norm_pre / clip_coef logging."""
    src = (HERE / "train.py").read_text(encoding="utf-8")
    cap_old = "gn_pre = torch.nn.utils.clip_grad_norm_("
    keys_old = '"grad_norm_pre": float(gn_pre), "clip_coef": float(torch.clamp(1.0 / (gn_pre + 1e-6), max=1.0))}'
    ok(src.count(cap_old) == 1 and src.count(keys_old) == 1, "the logging lines are where this test expects them")
    no_log = src.replace(cap_old, "torch.nn.utils.clip_grad_norm_(").replace(keys_old, "}")
    ok("gn_pre" not in no_log.replace("# the PRE-clip global norm", ""), "the variant carries no gn_pre")
    skip_step = src.replace("            opt.step()\n            opt.zero_grad(set_to_none=True)\n", "            opt.zero_grad(set_to_none=True)\n", 1)
    ok(skip_step != src, "the sensitivity variant must differ")
    cur = _load_variant(src, td / "train_cur.py")
    a = _train_with(cur, td, "cur1")
    b = _train_with(cur, td, "cur2")
    ok(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "CONTROL: the tiny run is deterministic (current vs current is bit-identical)")
    nl = _load_variant(no_log, td / "train_nolog.py")
    c = _train_with(nl, td, "nolog")
    ok(a.keys() == c.keys() and all(torch.equal(a[k], c[k]) for k in a), "the parameters with the logging are BIT-IDENTICAL to the parameters without it")
    sk = _load_variant(skip_step, td / "train_skip.py")
    d = _train_with(sk, td, "skip")
    ok(any(not torch.equal(a[k], d[k]) for k in a if k in d), "RED: a trainer that skips the optimiser step is DIFFERENT (the comparison can fail)")
    # the new keys are in every logged row and in the metrics of the unmodified run
    rows = [json.loads(ln) for ln in (td / "id_cur1" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    tr = [r for r in rows if "traj_L1" in r]
    ok(tr and all(math.isfinite(r["grad_norm_pre"]) and 0.0 < r["clip_coef"] <= 1.0 and r["clip_coef"] == _clip_coef_f32(r["grad_norm_pre"]) for r in tr), tr[:1])
    nl_rows = [json.loads(ln) for ln in (td / "id_nolog" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    ok(not any("grad_norm_pre" in r for r in nl_rows), "the variant logs no norm")
    # the S0 allow-list: the keys are rank-only for CTRL-S0 (a P-only key), the way T's rank keys were
    sys.path.insert(0, str(HERE.parent / "eval" / "scorerft"))
    import scorerft_read as SR
    ok(SR.S0_RANK_ONLY.match("grad_norm_pre") and SR.S0_RANK_ONLY.match("clip_coef") and not SR.S0_RANK_ONLY.match("grad_norm_post"), "S0_RANK_ONLY")


def chk_reader_closure():
    """the reader the probe's `judge` imports is part of the PACKAGE (the pod G-SUITE failure): every file of SCORERFT_READ_FILES exists under the package root, and importing scorerft_read loads exactly those eval/ modules."""
    pkg = HERE.parent
    ok(all((pkg / f).is_file() for f in SCORERFT_READ_FILES), [f for f in SCORERFT_READ_FILES if not (pkg / f).is_file()])
    sys.path.insert(0, str(pkg / "eval" / "scorerft"))
    import scorerft_read as SR
    loaded = sorted(Path(m.__file__).resolve().relative_to(pkg.resolve()).as_posix() for m in list(sys.modules.values())
                    if getattr(m, "__file__", None) and str(pkg.resolve()).lower() in str(Path(m.__file__).resolve()).lower() and Path(m.__file__).resolve().relative_to(pkg.resolve()).parts[0] == "eval")
    ok(loaded == sorted(SCORERFT_READ_FILES), ("the reader's loaded eval/ modules differ from SCORERFT_READ_FILES (add the new one, so the ship list carries it):", loaded))
    ok(hasattr(SR, "ctrl_s0"), "scorerft_read.ctrl_s0")


def main() -> int:
    chk_reader_closure()
    print(f"  reader_closure: ok ({N})", flush=True)
    chk_v1_and_red()
    print(f"  v1_and_red: ok ({N})", flush=True)
    chk_clip_coef_precision()
    print(f"  clip_coef_precision: ok ({N})", flush=True)
    td = P6._mkdtemp("clipprobe_")
    chk_states_and_invalid(td)
    print(f"  states_and_invalid: ok ({N})", flush=True)
    chk_wrapper_on_the_real_trainer(td)
    print(f"  wrapper_on_the_real_trainer: ok ({N})", flush=True)
    chk_logging_changes_no_behaviour(td)
    print(f"  logging_changes_no_behaviour: ok ({N})", flush=True)
    print(f"ZZCLIPPROBE_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
