"""ACCEPTANCE TEST for ``derive_rt_seed.py`` (D-R8-CKPT-RULE AMENDMENT 1b) -- TANITAD VENV, CPU, no forward, no GPU.

AMENDMENT 1b: "Control that must read a known value: ``derive_rt_seed.py`` applied to the seed-0 rows reproduces ``R8_RT`` exactly."

What is run: the groundwork's ``derive_rt_seed.py`` UNCHANGED, as its own process, on the Thor-CUDA smoke rows ``rows_R8_LEGAL.jsonl`` of BOTH
EMA checkpoints (5 navtest tokens, VALIDATION ONLY -- no score is produced).  The expected seam is the one ``run_bridge8.py --retime``
(the bridge's OWN R8_RT derivation, run_bridge8.py:281-297) wrote on Thor in the same run: ``seam_R8_RT.npz`` + manifest.  Every array
(token, fingerprint, source, poses, knots, sampling, arm) must be bit-identical and the manifest must name the registered table
(``3319b003...``), the registered stage (``b4bc7b79...``), ``derived_from rows_R8_LEGAL.jsonl``.

The expected values are the banked Thor artifacts (independent of the code under test: a different machine, a different OS, a different
process).  The deliberate-regression arms prove the comparison is sensitive:

  knot_perturbed_in_expected   one knot of the BANKED seam is nudged by 1e-3 m                  -> the comparison must report a difference
  knot_perturbed_in_rows       one knot of the SOURCE ROWS is nudged by 0.01 m                  -> the derived seam must differ from the bank
  stage_is_noop (script)       a copy of derive_rt_seed.py whose retime call is replaced by identity -> must differ
  wrong_source_rows            seed-1 rows fed as the seed-0 source                               -> must differ
  other_checkpoint             the 48,852 rows against the 40k bank                              -> must differ
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
NK = "C:/Users/Admin/refcv8_pf/navsim_ckpt"
SMOKE = f"{NK}/raw_thor_smoke"
PKG = f"{NK}/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim"
DERIVE = f"{NK}/code/derive_rt_seed.py"
RETIME = f"{NK}/thor_bundle/stage/retime"
INPUTS = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TOKENS = f"{PKG}/raw/smoke5_tokens.json"
BATT = f"{NK}/batt_fh/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"

DERIVE_SHA256 = "bebdc5c1ab41b775a60c80d4d79510bc7af6f4b4d12c374526985408e260e278"   # the file under acceptance
STAGE_SHA256 = "b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee"
RT_DEPLOY_SHA256 = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"

NEEDED = [DERIVE, f"{RETIME}/retime_stage.py", f"{RETIME}/refcv8_retime_deploy_gains_pf.json", INPUTS, TOKENS,
          f"{PKG}/code/run_bridge8.py", f"{SMOKE}/smoke_ema40000/bridge_navtest/seam_R8_RT.npz", f"{BATT}/r8_repoint.py",
          "C:/lgs/Lfb47a58376/stack/tanitad/__init__.py"]
pytestmark = pytest.mark.skipif(any(not os.path.exists(p) for p in NEEDED),
                                reason="dev-box fixtures absent: " + ", ".join(p for p in NEEDED if not os.path.exists(p)))


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _env():
    env = dict(os.environ)
    env.update(TANITAD_REPO="C:/lgs/Lfb47a58376", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1", R8_BATTERY_CODE=BATT,
               PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval")
    env.pop("R8_MUTATION", None)
    return env


def derive(tmp, ck, src_arm="R8_LEGAL", dst_arm="R8_RT", rows_path=None, script=DERIVE):
    """Run derive_rt_seed.py as its own process; returns the out dir."""
    out = os.path.join(str(tmp), f"out_{ck}_{src_arm}_{dst_arm}")
    os.makedirs(out, exist_ok=True)
    rows = rows_path or f"{SMOKE}/smoke_ema{ck}/bridge_navtest/rows_{src_arm}.jsonl"
    shutil.copy(rows, os.path.join(out, f"rows_{src_arm}.jsonl"))
    p = subprocess.run([sys.executable, "-X", "utf8", script, "--pkg-code", f"{PKG}/code", "--out", out, "--inputs", INPUTS,
                        "--split", "navtest", "--src-arm", src_arm, "--dst-arm", dst_arm,
                        "--retime-table", f"{RETIME}/refcv8_retime_deploy_gains_pf.json",
                        "--retime-table-sha256", RT_DEPLOY_SHA256, "--retime-stage-file", f"{RETIME}/retime_stage.py",
                        "--tokens-file", TOKENS], capture_output=True, text=True, env=_env(), cwd=str(tmp), timeout=900)
    assert f"[derive_rt] {dst_arm}: 5 rows, missing 0, partial False" in p.stdout, (p.stdout + p.stderr)[-2000:]
    return out


def compare_seams(got_dir, got_arm, want_dir, want_arm) -> list:
    """Every difference between two seams (arrays bit-for-bit + the manifest fields that name the table / stage / source). [] == identical."""
    diffs = []
    a, b = np.load(f"{got_dir}/seam_{got_arm}.npz"), np.load(f"{want_dir}/seam_{want_arm}.npz")
    if sorted(a.files) != sorted(b.files):
        diffs.append(f"files {sorted(a.files)} != {sorted(b.files)}")
    for k in b.files:
        if k == "arm":
            continue
        if k not in a.files or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
            diffs.append(f"array {k}")
    ma = json.load(open(f"{got_dir}/seam_{got_arm}.manifest.json", encoding="utf-8"))
    mb = json.load(open(f"{want_dir}/seam_{want_arm}.manifest.json", encoding="utf-8"))
    for k in ("n_rows", "n_missing", "partial", "n_model_rows", "n_cv_standin_rows", "n_expected", "split"):
        if ma.get(k) != mb.get(k):
            diffs.append(f"manifest {k}: {ma.get(k)} != {mb.get(k)}")
    for k in ("table_sha256", "stage_sha256", "knots_s", "gains", "call"):
        if (ma.get("retime") or {}).get(k) != (mb.get("retime") or {}).get(k):
            diffs.append(f"manifest retime.{k}")
    return diffs


# ---------------------------------------------------------------------------------------------------------------------------
def test_the_file_under_acceptance_is_the_one_pinned():
    assert _sha(DERIVE) == DERIVE_SHA256
    assert _sha(f"{RETIME}/retime_stage.py") == STAGE_SHA256
    assert _sha(f"{RETIME}/refcv8_retime_deploy_gains_pf.json") == RT_DEPLOY_SHA256


@pytest.mark.parametrize("ck", ["40000", "48852"])
def test_seed0_rows_reproduce_the_bridges_own_R8_RT_exactly(tmp_path, ck):
    out = derive(tmp_path, ck)
    want = f"{SMOKE}/smoke_ema{ck}/bridge_navtest"
    assert compare_seams(out, "R8_RT", want, "R8_RT") == []
    m = json.load(open(f"{out}/seam_R8_RT.manifest.json", encoding="utf-8"))
    assert m["retime"]["table_sha256"] == "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"
    assert m["retime"]["stage_sha256"] == "b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee"
    assert m["derived_from"] == "rows_R8_LEGAL.jsonl" and m["partial"] is False and m["n_rows"] == 5 and m["n_cv_standin_rows"] == 0


@pytest.mark.parametrize("ck", ["40000", "48852"])
def test_seed1_rows_reproduce_the_banked_R8_RT_s1(tmp_path, ck):
    """The seed-1 seam is what the rule needs and run_bridge8 cannot make; the bank is the same script on Thor (cross-platform determinism)."""
    out = derive(tmp_path, ck, src_arm="R8_LEGAL_s1", dst_arm="R8_RT_s1")
    assert compare_seams(out, "R8_RT_s1", f"{SMOKE}/smoke_ema{ck}/bridge_navtest", "R8_RT_s1") == []
    m = json.load(open(f"{out}/seam_R8_RT_s1.manifest.json", encoding="utf-8"))
    assert m["derived_from"] == "rows_R8_LEGAL_s1.jsonl"


def test_the_stage_acts_on_every_token(tmp_path):
    out = derive(tmp_path, "40000")
    a = np.load(f"{out}/seam_R8_RT.npz")["knots"]
    rows = [json.loads(ln) for ln in open(f"{SMOKE}/smoke_ema40000/bridge_navtest/rows_R8_LEGAL.jsonl", encoding="utf-8")]
    src = {r["token"]: np.asarray(r["knots"], np.float32) for r in rows}
    toks = [str(t) for t in np.load(f"{out}/seam_R8_RT.npz")["token"]]
    moved = [float(np.abs(a[i] - src[t]).max()) for i, t in enumerate(toks)]
    assert min(moved) > 0.1, moved           # the registered gains move EVERY plan (smallest per-token max |knot change| 0.1169 m)
    assert abs(max(moved) - 2.281787872314453) < 1e-6   # RT_ASSERT.json (Thor, MEASURED): max_abs_knot_change_vs_source_m, ema40000 R8_RT


# ---------------------------------------------------------------------------------------------------------------------------
# deliberate-regression arms: each must make the comparison REPORT a difference
# ---------------------------------------------------------------------------------------------------------------------------
def test_mutation_knot_perturbed_in_expected_is_detected(tmp_path):
    out = derive(tmp_path, "40000")
    want = os.path.join(str(tmp_path), "want_perturbed")
    os.makedirs(want)
    src = f"{SMOKE}/smoke_ema40000/bridge_navtest"
    shutil.copy(f"{src}/seam_R8_RT.manifest.json", want)
    z = dict(np.load(f"{src}/seam_R8_RT.npz"))
    z["knots"] = z["knots"].copy()
    z["knots"][2, 5, 1] += np.float32(1e-3)
    np.savez(f"{want}/seam_R8_RT.npz", **z)
    assert compare_seams(out, "R8_RT", want, "R8_RT") == ["array knots"]


def test_mutation_knot_perturbed_in_source_rows_is_detected(tmp_path):
    rows = [json.loads(ln) for ln in open(f"{SMOKE}/smoke_ema40000/bridge_navtest/rows_R8_LEGAL.jsonl", encoding="utf-8")]
    rows[2]["knots"][4][0] += 0.01
    bad = os.path.join(str(tmp_path), "rows_perturbed.jsonl")
    with open(bad, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    out = derive(tmp_path, "40000", rows_path=bad)
    d = compare_seams(out, "R8_RT", f"{SMOKE}/smoke_ema40000/bridge_navtest", "R8_RT")
    assert "array knots" in d and "array poses" in d


def test_mutation_stage_is_noop_in_the_script_is_detected(tmp_path):
    src = open(DERIVE, encoding="utf-8").read()
    old = 'knots = B.retime_plan8(RS, table, np.asarray(r["knots"], np.float32), np.asarray(r["prior_emitted_path"], np.float32))'
    assert src.count(old) == 1
    mut = os.path.join(str(tmp_path), "derive_rt_seed_noop.py")
    open(mut, "w", encoding="utf-8").write(src.replace(old, 'knots = np.asarray(r["knots"], np.float32)'))
    out = derive(tmp_path, "40000", script=mut)
    d = compare_seams(out, "R8_RT", f"{SMOKE}/smoke_ema40000/bridge_navtest", "R8_RT")
    assert "array knots" in d and "array poses" in d


def test_mutation_wrong_source_rows_is_detected(tmp_path):
    out = derive(tmp_path, "40000", rows_path=f"{SMOKE}/smoke_ema40000/bridge_navtest/rows_R8_LEGAL_s1.jsonl")
    assert "array knots" in compare_seams(out, "R8_RT", f"{SMOKE}/smoke_ema40000/bridge_navtest", "R8_RT")


def test_mutation_other_checkpoint_is_detected(tmp_path):
    out = derive(tmp_path, "48852")
    d = compare_seams(out, "R8_RT", f"{SMOKE}/smoke_ema40000/bridge_navtest", "R8_RT")
    assert "array knots" in d and "array token" not in d        # same tokens, different plans: the comparison separates the checkpoints
