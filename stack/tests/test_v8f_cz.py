"""PREREG_V8F_CZ: T's ABSENT constraint slots.

MEASURED 2026-10-21 on PREREG_V8F_TAC Amendment 2's saved codes: T's loss MASKS an absent constraint, while `_code_gt` writes 0 there and O
is teacher-forced on it, so at inference T writes unsupervised values into those slots (92-95 % of its squared error in latcons /
stoplead). The lever (`cons_absent zero`, arm M2b-cz) gives T the target O already reads; the stage-1 instrument (`--tac absmask`)
sizes the best case at zero training. Pinned here: the lever's gradient (with an EXACTLY-zero control), its declared-vs-built check, the
eval intervention, both readers' decision tables and refusals, and the queue stages.
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
for p in (REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
GATE_R = PKG / "code" / "v8f_cz_gate_read.py"
CZ_R = PKG / "code" / "v8f_cz_read.py"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_cz", Path(__file__).resolve().parent / "test_v8f_mini.py")
_TAC = _load("test_v8f_tac_for_cz", Path(__file__).resolve().parent / "test_v8f_tac.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
TR = _T.TR
torch.set_num_threads(1)
C0 = M.N_GOAL + M.N_LAT + M.N_LON


def _cons_grad(world, mode, k_absent=(1, 2, 16)):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    model, cfg = tiny_model(J, cons_absent=mode)
    b, _, _ = make_batch(world, n=8, seed=0, hmax=cfg.hmax)
    b["lab_t"]["has"] = torch.ones_like(b["lab_t"]["has"])
    cons = b["lab_t"]["cons"].clone()
    cons[:, list(k_absent)] = float("nan")              # these constraints are ABSENT in every window of the batch
    b["lab_t"]["cons"] = cons
    out = model.loss(b, p_gt=0.0, gen=torch.Generator().manual_seed(0))
    g = torch.autograd.grad(out["terms"]["cons"], out["cons_std"], retain_graph=True)[0]
    return g, out, list(k_absent), torch.isfinite(cons)


def test_cons_absent_zero_supervises_absent_slots_and_mask_leaves_them_exactly_free(world):
    g_mask, _, ab, fin = _cons_grad(world, "mask")
    assert float(g_mask[:, ab].abs().sum()) == 0.0, "CONTROL: under mask an absent slot gets EXACTLY zero gradient from T's loss"
    assert float(g_mask[fin].abs().sum()) > 0.0, "... and the present slots do get one"
    g_zero, out, ab, _ = _cons_grad(world, "zero")
    assert float(g_zero[:, ab].abs().min()) > 0.0, "zero: every absent slot is now supervised"
    want = torch.sign(out["cons_std"][:, ab].detach())                  # Huber toward 0: the gradient has the sign of the prediction
    assert torch.equal(torch.sign(g_zero[:, ab]), want), "the absent-slot target is 0 (the value _code_gt writes)"
    with pytest.raises(ValueError, match="cons_absent"):
        M.V8FMiniConfig(cons_absent="drop")


def test_m2b_cz_declares_and_builds_cons_absent_zero_and_a_lie_is_refused(world):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    a = _T._args(["--arm", "M2b-cz"])
    cfg, model, opt, info = _T._built(a, J)
    assert cfg.cons_absent == "zero" and model.built_levers()["cons_absent"] == "zero"
    assert TR.declared_vs_built(a, cfg, model, opt, info) == []
    cfg, model, opt, info = _T._built(a, J, cons_absent="mask")            # declared zero, built mask
    assert any("cons_absent" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    a = _T._args(["--arm", "M2b"])
    cfg, model, opt, info = _T._built(a, J)
    assert cfg.cons_absent == "mask" and TR.declared_vs_built(a, cfg, model, opt, info) == [], "every other arm is unchanged"
    a = _T._args(["--arm", "M2b", "--cons-absent", "zero"])
    assert TR.make_config(a).cons_absent == "zero", "the override reaches the config"
    a = _T._args(["--arm", "M2b-cz", "--w-cons", "0"])
    cfg, model, opt, info = _T._built(a, J)
    assert any("does nothing" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    assert TR.ARM_SPECS["M2b-cz"] == dict(TR.ARM_SPECS["M2b"], cons_absent="zero"), "M2b-cz is M2b with ONE lever moved"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert "cons_absent" in dests and TR.FLAG_CHECKS.get("cons_absent") == "lever:cons_absent" and dests <= set(TR.FLAG_CHECKS)


def test_eval_absmask_zeroes_only_the_gt_absent_constraint_slots(world, tmp_path):
    EV = _load("v8f_mini_eval_for_cz", REPO / "stack" / "scripts" / "v8f_mini_eval.py")
    ck = _TAC._ckpt(world, tmp_path)
    out = tmp_path / "ev"
    runs = {"abs": ("--tac", "absmask"), "tcons": ("--tac", "oracle", "--tac-groups", "all-constraints"),
            "gcons": ("--tac", "oracle", "--tac-groups", "constraints")}
    for t, args in runs.items():
        assert _TAC._run(EV, world, ck, out, f"c-{t}", *args) == 0, t
    z = {t: np.load(out / f"eval_c-{t}.npz") for t in runs}
    t_code = np.concatenate([z["gcons"]["tac_code"][:, :C0], z["tcons"]["tac_code"][:, C0:]], 1)     # T's own code, every slot
    g_cons = z["gcons"]["tac_code"][:, C0:]                                                          # the GT constraints (0 = absent)
    a = z["abs"]["tac_code"]
    assert np.array_equal(a[:, :C0], t_code[:, :C0]), "the decisions are T's own, untouched"
    changed = a[:, C0:] != t_code[:, C0:]
    assert changed.any(), "the world must contain absent constraints"
    assert np.all(a[:, C0:][changed] == 0.0) and np.all(g_cons[changed] == 0.0), "only slots the GT marks absent, and only to 0"
    m = json.load(open(out / "eval_c-abs.json", encoding="utf-8"))["tac_manipulation"]
    assert m["mode"] == "absmask" and 0.0 < m["absent_slot_frac"] < 1.0 and m["n_code_dims_replaced"] is None
    for bad in (("--tac", "absmask", "--tac-groups", "londec"), ("--tac", "absmask", "--tac-alpha", "0.5")):
        with pytest.raises(SystemExit, match="REFUSED"):
            _TAC._run(EV, world, ck, out, "bad", *bad)


# ---- stage 1: the gate reader --------------------------------------------------------------------------------------------------------
def _gate_files(d, d_abs=-0.5, full=-1.0, m2b_gap=0.0, mode="absmask", frac=0.4, drop=None, ref_shift=0.0, d_abs_s1=None):
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(30), 8)
    base = 2.2 + rng.normal(0, 0.3, eid.size) + np.repeat(rng.normal(0, 0.2, 30), 8)
    d.mkdir(parents=True, exist_ok=True)
    prev = {}
    for arm in ("M1", "M2b"):
        for s in (0, 1):
            c = f"{arm}_s{s}"
            pred = base + (m2b_gap * s if arm == "M2b" else 0.0)
            da = d_abs if (s == 0 or d_abs_s1 is None) else d_abs_s1
            for suf, ade in (("tacpred", pred), ("tacorc", pred + full), ("tacabs", pred + da + rng.normal(0, 0.01, eid.size))):
                if drop == f"{c}-{suf}":
                    continue
                np.savez(d / f"eval_{c}-{suf}.npz", eid=eid, ade6=ade, pred=np.zeros((eid.size, 30, 2)))
            json.dump({"tac_manipulation": {"mode": mode, "absent_slot_frac": frac, "code_shift_mean": 0.3}},
                      open(d / f"eval_{c}-tacabs.json", "w", encoding="utf-8"))
            prev[c] = {"ade6_pred": float(pred.mean()) + ref_shift}
    json.dump({"rows": prev}, open(d / "tac_read.json", "w", encoding="utf-8"))
    return d


def _read_gate(d):
    R = _load("v8f_cz_gate_read", GATE_R)
    out = d / "gate.json"
    rc = R.main(["--eval-dir", str(d), "--tac-read", str(d / "tac_read.json"), "--out", str(out), "--n-boot", "300", "--no-families"])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.mark.skipif(not GATE_R.exists(), reason="the v8F-mini package is not in this checkout")
def test_cz_gate_reader_decision_table(tmp_path, capsys):
    rc, r = _read_gate(_gate_files(tmp_path / "pass"))
    assert rc == 0 and r["GATE"] == "PASS" and "trains M2b-cz" in r["next"], r["reading"]
    assert abs(r["rows"]["M2b_s0"]["share"] - 0.5) < 0.03
    rc, r = _read_gate(_gate_files(tmp_path / "floor", m2b_gap=0.5))                 # 2F 1.0 > 0.5
    assert rc == 0 and r["GATE"] == "FAIL" and "No training" in r["next"]
    rc, r = _read_gate(_gate_files(tmp_path / "worse", d_abs=+0.5))
    assert rc == 0 and r["GATE"] == "FAIL", "a worse absmask never passes"
    rc, r = _read_gate(_gate_files(tmp_path / "oneseed", d_abs_s1=+0.2))
    assert rc == 0 and r["GATE"] == "FAIL", "one seed is not enough"
    assert _read_gate(_gate_files(tmp_path / "mode", mode="oracle"))[0] == 2
    assert "expected absmask" in capsys.readouterr().err
    assert _read_gate(_gate_files(tmp_path / "frac", frac=0.0))[0] == 2
    assert "expected absmask" in capsys.readouterr().err
    assert _read_gate(_gate_files(tmp_path / "miss", drop="M2b_s1-tacabs"))[0] == 2
    assert "eval_M2b_s1-tacabs.npz is missing" in capsys.readouterr().err
    assert _read_gate(_gate_files(tmp_path / "ref", ref_shift=0.01))[0] == 2
    assert "PREREG_V8F_TAC read" in capsys.readouterr().err


# ---- stage 2: the training reader ----------------------------------------------------------------------------------------------------
def _cz_files(d, delta=-0.4, m2b_gap=0.1, shift_cz=0.05, shift_m2b=0.3, lever_cz="zero", steps_cz=19164, drop=None, delta_s1=None):
    rng = np.random.default_rng(4)
    eid = np.repeat(np.arange(30), 8)
    base = 2.2 + rng.normal(0, 0.3, eid.size) + np.repeat(rng.normal(0, 0.2, 30), 8)
    ev, ru = d / "eval", d / "runs"
    ev.mkdir(parents=True, exist_ok=True)
    for arm in ("M2b", "M2b-cz"):
        for s in (0, 1):
            t = f"{arm}_s{s}"
            dl = delta if (s == 0 or delta_s1 is None) else delta_s1
            ade = base + m2b_gap * s + (dl + rng.normal(0, 0.01, eid.size) if arm == "M2b-cz" else 0.0)
            if drop != t:
                np.savez(ev / f"eval_{t}.npz", eid=eid, ade6=ade, pred=np.zeros((eid.size, 30, 2)))
            json.dump({"raw": {"traj": {"ade6": float(ade.mean())}}}, open(ev / f"eval_{t}.json", "w", encoding="utf-8"))
            json.dump({"tac_manipulation": {"mode": "absmask", "code_shift_mean": shift_cz if arm == "M2b-cz" else shift_m2b}},
                      open(ev / f"eval_{t}-tacabs.json", "w", encoding="utf-8"))
            (ru / t).mkdir(parents=True, exist_ok=True)
            lev = {"wm": "b"} if arm == "M2b" else {"wm": "b", "cons_absent": lever_cz}
            json.dump({"arm": arm, "seed": s, "steps": steps_cz if arm == "M2b-cz" else 19164, "done": True, "n_clips": 4512,
                       "train_sha256": "t" * 64, "excluded_sha256": HOLD, "levers": lev}, open(ru / t / "summary.json", "w", encoding="utf-8"))
    return d


def _read_cz(d):
    R = _load("v8f_cz_read", CZ_R)
    out = d / "cz.json"
    rc = R.main(["--eval-dir", str(d / "eval"), "--runs-dir", str(d / "runs"), "--out", str(out), "--n-boot", "300", "--no-families"])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.mark.skipif(not CZ_R.exists(), reason="the v8F-mini package is not in this checkout")
def test_cz_reader_decision_table(tmp_path, capsys):
    rc, r = _read_cz(_cz_files(tmp_path / "ok"))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and abs(r["two_F"] - 0.2) < 1e-6, r["reading"]
    rc, r = _read_cz(_cz_files(tmp_path / "small", delta=-0.15))                       # beyond 0 but under 2F 0.2
    assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and "PRESENT-slot" in r["next"]
    rc, r = _read_cz(_cz_files(tmp_path / "nomech", delta=-0.15, shift_cz=0.2))       # 0.2 > 0.5 x 0.3
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM" and "Diagnose" in r["next"]
    rc, r = _read_cz(_cz_files(tmp_path / "worse", delta=+0.4))
    assert rc == 0 and r["VERDICT"] != "SUCCESS"
    rc, r = _read_cz(_cz_files(tmp_path / "oneseed", delta_s1=+0.4))
    assert rc == 0 and r["VERDICT"] != "SUCCESS", "one seed is not enough"
    assert _read_cz(_cz_files(tmp_path / "lever", lever_cz="mask"))[0] == 2
    assert "cons_absent mask, expected zero" in capsys.readouterr().err
    assert _read_cz(_cz_files(tmp_path / "steps", steps_cz=9000))[0] == 2
    assert "different step counts" in capsys.readouterr().err
    assert _read_cz(_cz_files(tmp_path / "miss", drop="M2b-cz_s1"))[0] == 2
    assert "eval_M2b-cz_s1.npz is missing" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
def test_queue_cz_stages(tmp_path):
    import os
    import re
    import subprocess
    q = PKG / "code" / "v8f_mini_box2_queue.cmd"
    env = dict(os.environ, V8F_DRYRUN="1", REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"),
               V9=str(tmp_path / "v9"))

    def dry(args):
        r = subprocess.run(f'cmd /c ""{q}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
        return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]
    rc, ln = dry("tacabs")
    assert rc == 0 and [(re.search(r"--tag (\S+)", x).group(1), re.search(r"--tac (\S+)", x).group(1)) for x in ln] == \
        [(f"{a}_s{s}-tacabs", "absmask") for a in ("M1", "M2b") for s in (0, 1)], ln
    rc, ln = dry("gatearm M2b-cz")
    assert rc == 0 and len(ln) == 1 and "--arm M2b-cz" in ln[0] and "--selfcheck-only" in ln[0] and "--kin-mode yaw" in ln[0], ln
    assert dry("gatearm")[0] == 2
    rc, ln = dry("tacabsarm M2b-cz 1")
    assert rc == 0 and len(ln) == 1 and "--tag M2b-cz_s1-tacabs" in ln[0] and "--tac absmask" in ln[0], ln
    rc, ln = dry("train M2b-cz 0")
    assert rc == 0 and len(ln) == 1 and "--arm M2b-cz --seed 0" in ln[0] and "--exclude 540:600" in ln[0], ln
