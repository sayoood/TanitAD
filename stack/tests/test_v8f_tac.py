"""PREREG_V8F_TAC: the T -> O intervention on v8F-mini (does the operative layer USE the tactical layer?). CPU only, < 90 s.

What is pinned, and the control that must go red for each:
  * ``forward_diag`` with T's OWN code reproduces ``forward_infer`` BIT-exactly (the identity control: the hook changes nothing else),
    and a different code MOVES O's trajectory (O is code-sensitive in this world, so a zero below is a measurement, not a dead probe);
    the hook reads only INFERENCE_INPUT_KEYS; a wrong-shaped override is refused.
  * ``v8f_mini_eval.py run --tac shuffle`` gives O another EPISODE's predicted code on every window and changes O's output; ``--tac
    oracle`` gives O the GT code where the window has a v9 row; ``--tac pred`` reproduces a plain run; combining --tac with --plan beta or
    --nav-shuffle is refused.
  * the reader (code/v8f_tac_read.py): USED only when shuf - pred > 0 and separated at ALL FOUR checkpoints; DECORATIVE only when separated
    at none; LARGE headroom only below -2F at both seeds; it REFUSES a shuffle that did not change the code, a tacpred run that does not
    reproduce the banked eval, and a missing file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
READER = PKG / "code" / "v8f_tac_read.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_tac", Path(__file__).resolve().parent / "test_v8f_mini.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
torch.set_num_threads(1)


def _code_sensitive(model, seed=0):
    """Give O a non-zero residual head and a strong code embedding, so its output depends on T's code (zero-init starts at P10)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(model.head_o.parameters()) + list(model.e_code.parameters()):
            p.add_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def test_forward_diag_identity_is_bit_exact_and_a_different_code_moves_o(world):
    b, J, _ = make_batch(world)
    model, _ = tiny_model(J)
    _code_sensitive(model).eval()
    with torch.no_grad():
        o = model.forward_infer(b)
        o2 = model.forward_diag(b, o["code"])
        assert torch.equal(o2["traj30"], o["traj30"]) and torch.equal(o2["code"], o["code"])     # IDENTITY: exactly the same
        assert o2["_keys_read"] <= M.INFERENCE_INPUT_KEYS, o2["_keys_read"] - M.INFERENCE_INPUT_KEYS
        o3 = model.forward_diag(b, o["code"].roll(1, 0))                                          # another window's code
        assert (o3["traj30"] - o["traj30"]).abs().max() > 1e-3, "O must be code-sensitive here, or a zero means nothing"
        assert torch.equal(o3["code"], o["code"].roll(1, 0)), "out['code'] is the code O read"
        with pytest.raises(ValueError, match="code_override"):
            model.forward_diag(b, o["code"][:, :5])


def _ckpt(world, tmp_path):
    TR = _load("v8f_mini_train_for_tac", REPO / "stack" / "scripts" / "v8f_mini_train.py")
    out = tmp_path / "run"
    assert TR.main(["--out", str(out), "--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "4", "--log-every", "2",
                    "--warmup", "1", "--ckpt-every", "4", "--threads", "1", "--seed", "3"]) == 0
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    g = torch.Generator().manual_seed(1)
    for k, v in ck["model"].items():                     # the same perturbation as _code_sensitive, on the saved state
        if k.startswith("head_o.") or k.startswith("e_code."):
            ck["model"][k] = v + torch.randn(v.shape, generator=g) * 0.3
    p = tmp_path / "ck_sens.pt"
    torch.save(ck, p)
    return p


def _run(EV, world, ck, out, tag, *extra):
    root = Path(world["root"])
    return EV.main(["run", "--ckpt", str(ck), "--tag", tag, "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
                    "--eval-kin", str(root / "eval_kin.npz"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out),
                    "--tracks", str(root / "native_tracks.pt"), "--n-boot", "20", "--batch", "16", "--lru", "4", *extra])


def test_eval_tac_modes_intervene_on_t_to_o_only(world, tmp_path):
    EV = _load("v8f_mini_eval_for_tac", REPO / "stack" / "scripts" / "v8f_mini_eval.py")
    ck = _ckpt(world, tmp_path)
    out = tmp_path / "ev"
    assert _run(EV, world, ck, out, "plain") == 0
    for mode, tag in (("pred", "c-tacpred"), ("shuffle", "c-tacshuf"), ("oracle", "c-tacorc")):
        assert _run(EV, world, ck, out, tag, "--tac", mode) == 0, mode
    z = {t: np.load(out / f"eval_{t}.npz") for t in ("plain", "c-tacpred", "c-tacshuf", "c-tacorc")}
    assert np.array_equal(z["plain"]["pred"], z["c-tacpred"]["pred"]), "--tac pred IS a plain run"
    assert "tac_code_shift" not in z["c-tacpred"].files
    js = json.load(open(out / "eval_c-tacshuf.json", encoding="utf-8"))
    m = js["tac_manipulation"]
    assert js["tac"] == "shuffle" and m["frac_code_changed"] >= 0.95 and m["n_oracle_fallback_no_v9_row"] == 0, m
    assert np.abs(z["c-tacshuf"]["pred"] - z["plain"]["pred"]).max() > 1e-3, "the donor code must move O's plan"
    mo = json.load(open(out / "eval_c-tacorc.json", encoding="utf-8"))["tac_manipulation"]
    assert mo["code_shift_mean"] > 0 and mo["n_oracle_fallback_no_v9_row"] == int(z["c-tacorc"]["tac_oracle_fallback"].sum())
    assert np.array_equal(z["c-tacorc"]["eid"], z["plain"]["eid"])
    for extra in (("--nav-shuffle",), ("--plan", "beta")):
        with pytest.raises(SystemExit, match="REFUSED"):
            _run(EV, world, ck, out, "bad", "--tac", "shuffle", *extra)


# ------------------------------------------------------------------------------------------------ the reader
def _world_files(d, shuf_eff=(0.3, 0.3, 0.3, 0.3), orc_eff=(-0.5, -0.5, -0.5, -0.5), m2b_gap=0.0, changed=1.0, repro=0.0, drop=None):
    rng = np.random.default_rng(4)
    eid = np.repeat(np.arange(30), 8)
    base = 2.2 + rng.normal(0, 0.3, eid.size) + np.repeat(rng.normal(0, 0.2, 30), 8)
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    for arm in ("M1", "M2b"):
        for s in (0, 1):
            c = f"{arm}_s{s}"
            pred = base + (m2b_gap * s if arm == "M2b" else 0.0) + 0.01 * s
            noise = rng.normal(0, 0.02, eid.size)
            files = {"": pred, "-tacpred": pred + repro, "-tacshuf": pred + shuf_eff[k] + noise, "-tacorc": pred + orc_eff[k] + noise}
            for suf, ade in files.items():
                if drop == f"{c}{suf}":
                    continue
                z = dict(eid=eid, ade6=ade, pred=np.zeros((eid.size, 30, 2)))
                if suf in ("-tacshuf", "-tacorc"):
                    z["tac_code_shift"] = np.where(np.arange(eid.size) < changed * eid.size, 0.2, 0.0)
                    z["tac_oracle_fallback"] = np.zeros(eid.size, bool)
                np.savez(d / f"eval_{c}{suf}.npz", **z)
            k += 1
    return d


def _read(d, capsys=None):
    R = _load("v8f_tac_read", READER)
    out = d / "tac.json"
    rc = R.main(["--eval-dir", str(d), "--out", str(out), "--n-boot", "300", "--no-families"])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.mark.skipif(not READER.exists(), reason="the v8F-mini package is not in this checkout")
def test_reader_decision_table(tmp_path):
    rc, r = _read(_world_files(tmp_path / "used"))
    assert rc == 0 and r["TAC_USED"]["verdict"] == "USED" and r["TAC_HEADROOM_T0"]["M2b"]["verdict"] == "LARGE", r["reading"]
    assert "USED + LARGE" in r["next"] and r["checks"]["M2b_s1"]["shuffle_frac_code_changed"] == 1.0
    rc, r = _read(_world_files(tmp_path / "deco", shuf_eff=(0.0, 0.0, 0.0, 0.0)))
    assert rc == 0 and r["TAC_USED"]["verdict"] == "DECORATIVE" and "DECORATIVE" in r["next"], r["reading"]
    rc, r = _read(_world_files(tmp_path / "part", shuf_eff=(0.3, 0.3, 0.0, 0.3)))
    assert rc == 0 and r["TAC_USED"]["verdict"] == "PARTIAL", r["reading"]
    rc, r = _read(_world_files(tmp_path / "neg", shuf_eff=(0.3, 0.3, -0.3, 0.3)))
    assert rc == 0 and r["TAC_USED"]["verdict"] == "PARTIAL", "a donor code that HELPS is never USED"
    rc, r = _read(_world_files(tmp_path / "allneg", shuf_eff=(-0.3, -0.3, -0.3, -0.3)))
    assert rc == 0 and r["TAC_USED"]["verdict"] == "PARTIAL", "separated everywhere (the wrong way) is neither USED nor DECORATIVE"
    rc, r = _read(_world_files(tmp_path / "small", orc_eff=(-0.5, -0.5, -0.5, -0.5), m2b_gap=0.6))   # F_M2b 0.6 -> 2F 1.2 > 0.5
    assert rc == 0 and r["TAC_HEADROOM_T0"]["M2b"]["verdict"] == "SMALL" and "USED + SMALL" in r["next"], r["reading"]
    rc, r = _read(_world_files(tmp_path / "pos", orc_eff=(-0.5, -0.5, 0.5, -0.5)))
    assert rc == 0 and r["TAC_HEADROOM_T0"]["M2b"]["verdict"] == "SMALL", "both seeds must be below -2F"


@pytest.mark.skipif(not READER.exists(), reason="the v8F-mini package is not in this checkout")
def test_reader_refuses_a_shuffle_that_did_not_happen_a_drifted_rerun_and_a_missing_file(tmp_path, capsys):
    assert _read(_world_files(tmp_path / "nochange", changed=0.5))[0] == 2
    assert "the intervention did not happen" in capsys.readouterr().err
    assert _read(_world_files(tmp_path / "drift", repro=0.01))[0] == 2
    assert "differs from the banked eval" in capsys.readouterr().err
    assert _read(_world_files(tmp_path / "miss", drop="M2b_s1-tacorc"))[0] == 2
    assert "eval_M2b_s1-tacorc.npz is missing" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
def test_queue_tac_stage_runs_three_interventions_on_four_checkpoints(tmp_path):
    """The box-2 queue's `tac` stage: M1 / M2b x seeds 0, 1 x {pred, shuffle, oracle}, the yaw-rate eval arguments on every call, tags
    the reader expects (eval_<ARM>_s<SEED>-tac{pred,shuf,orc}), and no --project / --families / --nav-shuffle / --plan."""
    import os
    import re
    import subprocess
    q = PKG / "code" / "v8f_mini_box2_queue.cmd"
    env = dict(os.environ, V8F_DRYRUN="1", REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"),
               V9=str(tmp_path / "v9"))
    r = subprocess.run(f'cmd /c ""{q}" tac"', env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    ln = [x for x in r.stdout.splitlines() if "v8f_mini_eval.py" in x]
    got = [(re.search(r"--tag (\S+)", x).group(1), re.search(r"--tac (\S+)", x).group(1)) for x in ln]
    want = [(f"{a}_s{s}-{suf}", m) for a in ("M1", "M2b") for s in (0, 1)
            for m, suf in (("pred", "tacpred"), ("shuffle", "tacshuf"), ("oracle", "tacorc"))]
    assert got == want, got
    assert all("--kin-mode yaw" in x and "--tracks" in x and f"{os.sep}ckpt_last.pt" in x for x in ln), ln
    assert not any(f in x for x in ln for f in ("--project", "--families", "--nav-shuffle", "--plan")), ln


def test_the_v8f_guard_ends_only_on_a_line_start_marker_of_its_chains():
    """v8f_guard.sh stays armed until ITS chain's DONE / STOP marker opens a line (the unanchored form matched a chain's own waiting line
    in ra1runs_guard.sh on 2026-10-20), and it knows chain_v8f_tac.sh's markers (else it never exits)."""
    import re
    g = (PKG / "code" / "box2_ops" / "v8f_guard.sh").read_text(encoding="utf-8")
    ln = [x for x in g.splitlines() if "$CHAIN_OUT" in x and "grep -q -E" in x]
    assert len(ln) == 1, ln
    rx = re.compile(re.search(r"grep -q -E '([^']+)'", ln[0]).group(1))
    for m in ("ZZV8FTAC-DONE 10-21 01:10 tip x: ...", "ZZV8FTAC-STOP box 2 unreachable", "ZZBETA-DONE 10-20 14:17", "ZZV8FMINI-STOP x"):
        assert rx.search(m), m
    for m in ("10-20 23:00 waiting for the predecessor chain (/c/Users/Admin/box2_ship/logs/chain_ra1_full5.out)",
              "10-20 23:00 waiting for ZZV8FTAC-DONE", "echo ZZV8FTAC-DONE", " ZZBETA-STOP indented"):
        assert not rx.search(m), m
    chain = (PKG / "code" / "box2_ops" / "chain_v8f_tac.sh").read_text(encoding="utf-8")
    assert set(re.findall(r'echo "(ZZ[A-Z0-9]+)-STOP', chain)) == {"ZZV8FTAC"}
