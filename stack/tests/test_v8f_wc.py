"""PREREG_V8F_WC: T's PRESENT constraint slots, by the loss weight.

PREREG_V8F_CZ's gate (2026-10-22) refuted the absent-slot route: O reads T's own code for most of its training, so T's absent-slot values
are O's input (R63). Its committed FAIL branch names T's PRESENT-slot accuracy as the next lever. Arm M2b-wc = M2b with ONE lever moved,
`--w-cons 4.0` (M2b: the default 0.5) -- an existing flag with a declared-vs-built entry, so the trainer's arm table is unchanged.
Pinned here: argv -> config -> loss (an analytic identity, with an EXACTLY-zero control on a module the constraint term cannot reach),
the built weight recorded in built_levers, the reader's decision table (stage s0 and final) and its refusals, and the queue stages.
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
WC_R = PKG / "code" / "v8f_wc_read.py"
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_T = _load("test_v8f_mini_for_wc", Path(__file__).resolve().parent / "test_v8f_mini.py")
world, make_batch, tiny_model = _T.world, _T.make_batch, _T.tiny_model
TR = _T.TR
torch.set_num_threads(1)


def _loss_at(world, w_cons):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    torch.manual_seed(0)
    model, cfg = tiny_model(J, w_cons=w_cons)
    b, _, _ = make_batch(world, n=8, seed=0, hmax=cfg.hmax)
    b["lab_t"]["has"] = torch.ones_like(b["lab_t"]["has"])
    out = model.loss(b, p_gt=0.0, gen=torch.Generator().manual_seed(0))
    model.zero_grad()
    out["terms"]["total"].backward()
    return model, cfg, out


def test_w_cons_reaches_the_loss_exactly_and_nothing_else(world):
    m05, c05, o05 = _loss_at(world, 0.5)
    m4, c4, o4 = _loss_at(world, 4.0)
    t05, t4 = o05["terms"], o4["terms"]
    assert float(t05["cons"]) > 0.0 and torch.equal(t05["cons"], t4["cons"]), "same weights, same batch: the same constraint term"
    want = (4.0 - 0.5) * c4.w_tac * float(t4["cons"])
    assert abs((float(t4["total"]) - float(t05["total"])) - want) < 1e-5, "the total moves by exactly 3.5 x w_tac x cons"
    # CONTROL: the constraint term cannot reach O's output head (O reads T's code, never T's loss), so its gradient is EXACTLY unchanged
    for (n, p5), (_, p4) in zip(m05.head_o.named_parameters(), m4.head_o.named_parameters()):
        if p5.grad is not None:
            assert torch.equal(p5.grad, p4.grad), f"head_o.{n}: the weight leaked into a module the constraint term cannot reach"
    # ... while T's own parameters DO see it: at least one T parameter's gradient changes
    t_ids = {id(p) for m in m4.trainable_groups()["T"] for p in (m.parameters() if isinstance(m, torch.nn.Module) else [m])}
    t_moved = [n for (n, p5), (_, p4) in zip(m05.named_parameters(), m4.named_parameters())
               if id(p4) in t_ids and p5.grad is not None and not torch.equal(p5.grad, p4.grad)]
    assert t_ids and t_moved, "no parameter of T's trainable group changed its gradient"
    assert m4.built_levers()["w"]["w_cons"] == 4.0 and m05.built_levers()["w"]["w_cons"] == 0.5
    assert m4.built_levers()["w"]["w_tac"] == c4.w_tac


def test_m2b_wc_argv_builds_w_cons_4_and_a_lie_is_refused(world):
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    a = _T._args(["--arm", "M2b", "--w-cons", "4.0"])
    assert TR.make_config(a).w_cons == 4.0, "the flag reaches the config"
    cfg, model, opt, info = _T._built(a, J)
    assert TR.declared_vs_built(a, cfg, model, opt, info) == [] and model.built_levers()["w"]["w_cons"] == 4.0
    cfg, model, opt, info = _T._built(a, J, w_cons=0.5)                     # declared 4.0, built 0.5
    assert any("w_cons" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    assert TR.build_parser().parse_args(["--out", "x"]).w_cons == 0.5, "M2b's default is 0.5: M2b-wc moves it 8x"
    assert TR.FLAG_CHECKS.get("w_cons") == "cfg:w_cons"
    assert "M2b-wc" not in TR.ARM_SPECS, "M2b-wc is M2b + one flag; the arm table is unchanged"


# ---- the reader ----------------------------------------------------------------------------------------------------------------------
C0, NC = M.N_GOAL + M.N_LAT + M.N_LON, len(M.CONS_NAMES)


def test_reader_layout_matches_the_model():
    R = _load("v8f_wc_read_layout", WC_R)
    assert (R.C0, R.N_CONS) == (C0, NC) and C0 + NC == M.CODE_DIM
    for g, r in R.GROUPS.items():
        assert [C0 + i for i in r] == list(M.CODE_GROUPS[g]), g


def _wc_files(d, delta=-0.4, m2b_gap=0.1, err_wc=0.3, err_m2b=0.6, w_wc=4.0, argv_wc="4.0", steps_wc=19164, drop=None, delta_s1=None,
              gt_shift=0.0, m2b_argv_wc=False, sc="PASS", seeds=(0, 1)):
    rng = np.random.default_rng(4)
    n_ep, per = 30, 8
    eid = np.repeat(np.arange(n_ep), per)
    N = eid.size
    base = 2.2 + rng.normal(0, 0.3, N) + np.repeat(rng.normal(0, 0.2, n_ep), per)
    g = np.zeros((N, M.CODE_DIM))
    g[:, :C0] = rng.normal(0, 1, (N, C0))
    present = rng.random((N, NC)) < 0.45
    g[:, C0:] = np.where(present, rng.normal(0, 1, (N, NC)), 0.0)
    fb = np.zeros(N, bool)
    fb[:5] = True                                                         # 5 windows without a v9 row
    ev, ru = d / "eval", d / "runs"
    ev.mkdir(parents=True, exist_ok=True)
    for arm in ("M2b", "M2b-wc"):
        for s in (0, 1):
            t = f"{arm}_s{s}"
            dl = delta if (s == 0 or delta_s1 is None) else delta_s1
            ade = base + m2b_gap * s + (dl + rng.normal(0, 0.01, N) if arm == "M2b-wc" else 0.0)
            if arm == "M2b" or s in seeds:
                if drop != t:
                    np.savez(ev / f"eval_{t}.npz", eid=eid, ade6=ade, pred=np.zeros((N, 30, 2)))
                json.dump({"raw": {"traj": {"ade6": float(ade.mean())}}}, open(ev / f"eval_{t}.json", "w", encoding="utf-8"))
            if s not in seeds:
                continue
            err = err_wc if arm == "M2b-wc" else err_m2b
            tc = g.copy()
            tc[:, C0:] = np.where(present, g[:, C0:] + rng.normal(0, err, (N, NC)), 0.0)        # absmask: present = T's, absent = 0
            gg = g.copy()
            if arm == "M2b-wc":
                gg[:, C0:] += gt_shift * present
            np.savez(ev / f"eval_{t}-tacabs.npz", eid=eid, ade6=ade, tac_code=tc, tac_oracle_fallback=fb)
            np.savez(ev / f"eval_{t}-tacorc.npz", eid=eid, ade6=ade - 1.0, tac_code=gg, tac_oracle_fallback=fb)
            json.dump({"tac_manipulation": {"mode": "absmask", "absent_slot_frac": 0.55}}, open(ev / f"eval_{t}-tacabs.json", "w", encoding="utf-8"))
            (ru / t).mkdir(parents=True, exist_ok=True)
            w = {"w_pen": 0.5, "w_wm": 0.1} if arm == "M2b" else {"w_pen": 0.5, "w_wm": 0.1, "w_tac": 1.0, "w_cons": w_wc}
            json.dump({"arm": "M2b", "seed": s, "steps": steps_wc if arm == "M2b-wc" else 19164, "done": True, "n_clips": 4512,
                       "train_sha256": "t" * 64, "excluded_sha256": HOLD, "levers": {"wm": "b", "w": w}},
                      open(ru / t / "summary.json", "w", encoding="utf-8"))
            av = ["--arm", "M2b", "--seed", str(s)] + (["--w-cons", argv_wc] if arm == "M2b-wc" else (["--w-cons", "4.0"] if m2b_argv_wc else []))
            json.dump({"argv": av, "sha": "x"}, open(ru / t / "argv.json", "w", encoding="utf-8"))
            json.dump({"declared_vs_built": sc}, open(ru / t / "selfcheck.json", "w", encoding="utf-8"))
    return d


def _read(d, stage="final"):
    R = _load("v8f_wc_read", WC_R)
    out = d / "wc.json"
    rc = R.main(["--stage", stage, "--eval-dir", str(d / "eval"), "--runs-dir", str(d / "runs"), "--out", str(out), "--n-boot", "300",
                 "--no-families"])
    return rc, (json.loads(out.read_text(encoding="utf-8")) if rc == 0 else None)


@pytest.mark.skipif(not WC_R.exists(), reason="the v8F-mini package is not in this checkout")
def test_wc_reader_decision_table(tmp_path):
    rc, r = _read(_wc_files(tmp_path / "ok"))
    assert rc == 0 and r["VERDICT"] == "SUCCESS" and abs(r["two_F"] - 0.2) < 1e-6 and r["mechanism"], r["reading"]
    assert r["rows"]["s0"]["present_rmse"]["M2b-wc"] < r["rows"]["s0"]["present_rmse"]["M2b"]
    assert r["n_windows_present"] < r["n_windows"], "windows without a v9 row are out of the mechanism"
    rc, r = _read(_wc_files(tmp_path / "small", delta=-0.15))                       # beyond 0 but under 2F 0.2
    assert rc == 0 and r["VERDICT"] == "FAIL-MECHANISM-OK" and "O's use" in r["next"]
    rc, r = _read(_wc_files(tmp_path / "nomech", delta=-0.15, err_wc=0.9))
    assert rc == 0 and r["VERDICT"] == "FAIL-NO-MECHANISM" and "INPUT / capacity" in r["next"]
    rc, r = _read(_wc_files(tmp_path / "worse", delta=+0.4))
    assert rc == 0 and r["VERDICT"] != "SUCCESS"
    rc, r = _read(_wc_files(tmp_path / "oneseed", delta_s1=+0.4))
    assert rc == 0 and r["VERDICT"] != "SUCCESS", "one seed is not enough"
    # stage s0: seed 0 alone decides whether seed 1 is trained; F still comes from BOTH banked M2b evals
    rc, r = _read(_wc_files(tmp_path / "s0go", seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "CONTINUE" and r["VERDICT"] is None and abs(r["two_F"] - 0.2) < 1e-6, r["reading"]
    rc, r = _read(_wc_files(tmp_path / "s0stop", delta=-0.15, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-MECHANISM-OK"
    rc, r = _read(_wc_files(tmp_path / "s0nomech", delta=+0.1, err_wc=0.9, seeds=(0,)), "s0")
    assert rc == 0 and r["DECISION"] == "STOP" and r["VERDICT"] == "FAIL-NO-MECHANISM"


@pytest.mark.skipif(not WC_R.exists(), reason="the v8F-mini package is not in this checkout")
def test_wc_reader_refusals(tmp_path, capsys):
    for name, kw, msg in (("wbuilt", dict(w_wc=0.5), "built with w_cons 0.5, expected 4.0"),
                          ("wargv", dict(argv_wc="2.0"), "argv --w-cons 2.0, expected 4.0"),
                          ("m2bargv", dict(m2b_argv_wc=True), "M2b_s0: argv carries --w-cons"),
                          ("steps", dict(steps_wc=9000), "different step counts"),
                          ("gate", dict(sc="FAIL"), "declared_vs_built is not PASS"),
                          ("gt", dict(gt_shift=0.5), "the GT code differs"),
                          ("miss", dict(drop="M2b-wc_s1"), "eval_M2b-wc_s1.npz is missing")):
        assert _read(_wc_files(tmp_path / name, **kw))[0] == 2, name
        assert msg in capsys.readouterr().err, name
    assert _read(_wc_files(tmp_path / "f", drop="M2b_s1", seeds=(0,)), "s0")[0] == 2
    assert "M2b's seed floor needs both seeds" in capsys.readouterr().err


def test_queue_wc_runs_the_command_it_prints():
    """A dry run can only print: the real line is checked statically -- the trainer appears ONCE per stage, inside the one command
    string, and the real run executes that string (the mutation proof found a real line that dropped the lever under a green dry run)."""
    q = (PKG / "code" / "v8f_mini_box2_queue.cmd").read_text(encoding="utf-8").replace("\r\n", "\n")
    for label, var in ((":trainwc", "TW"), (":gatewc", "GW")):
        blk = q.split(f"\n{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")
        code = [x for x in blk if x.strip() and not x.lower().startswith("rem ")]
        assert [x for x in code if "%TRAIN%" in x] == [x for x in code if x.startswith(f'set "{var}=')], label
        assert any(x.startswith(f'set "{var}="%PY%" "%TRAIN%" --arm M2b --w-cons %WC% ') for x in code), label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code, label
        assert sum(x.startswith(f"%{var}% ") for x in code) == 1, label
    assert '\nset "WC=4.0"\n' in q


@pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
def test_queue_wc_stages(tmp_path):
    import os
    import subprocess
    q = PKG / "code" / "v8f_mini_box2_queue.cmd"
    env = dict(os.environ, V8F_DRYRUN="1", REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"),
               V9=str(tmp_path / "v9"))

    def dry(args):
        r = subprocess.run(f'cmd /c ""{q}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
        return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]
    rc, ln = dry("gatewc")
    assert rc == 0 and len(ln) == 1 and "--arm M2b --w-cons 4.0" in ln[0] and "--selfcheck-only" in ln[0] and "gate_M2b-wc" in ln[0], ln
    rc, ln = dry("trainwc 1")
    assert rc == 0 and len(ln) == 1 and "--arm M2b --w-cons 4.0 --seed 1" in ln[0] and "M2b-wc_s1" in ln[0], ln
    assert "--exclude 540:600" in ln[0] and "--kin-mode yaw" in ln[0] and "--steps-mult 4" in ln[0], ln
    assert dry("trainwc")[0] == 2
    rc, ln = dry("evalwc 0")
    assert rc == 0 and len(ln) == 3, ln
    assert "--tag M2b-wc_s0 " in ln[0] and "--families" in ln[0] and "M2b-wc_s0" in ln[0], ln
    assert "--tag M2b-wc_s0-tacabs" in ln[1] and "--tac absmask" in ln[1], ln
    assert "--tag M2b-wc_s0-tacorc" in ln[2] and "--tac oracle" in ln[2] and "--tac-groups" not in ln[2], ln
    assert dry("evalwc")[0] == 2
