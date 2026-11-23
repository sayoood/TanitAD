"""Tests of the KI kind of the study checker (dscale10k_check_ki.py = the revision-2 RK checker 49a5a116 + patch_ki.py): RH's argv + EXACTLY `--dagger-base logged` (appended) + its own --out, the RH bank under bank/, dose 0.1,
human label (no label flag), no cache (K-INPUT, PREREG_V8F_KINPUT). CPU only, on the synthetic rig of test_dscale10k_check_rhk.py (the same Rig). Run from this folder:  python -I -m pytest -q test_ki_checker.py
The mutant arms the checker must refuse, each with a non-empty C-ARGV problem list: a second changed flag; another bank / another sha256; base `own` or another value under the KI name; the base flag missing; dose other than 0.1;
a label / mode / cache / kin-tb flag added; the base flag under the RH name."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KI_CHECKER = os.environ.get("DS_CHECK_KI_PATH") or os.path.join(os.path.dirname(HERE), "dscale10k_check_ki.py")
os.environ["DS_CHECK_RH_PATH"] = KI_CHECKER                                 # the base test module loads its NEW checker from here
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_dscale10k_check_rhk as T  # noqa: E402
from test_dscale10k_check_rhk import Rig, refused_by  # noqa: E402,F401

NEW = T.NEW
BASE = "logged"


def ki_toks(r, base=BASE, p=None, extra=None):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"KI_s{r.seed}", "train"))
    if base is not None:
        t = t + ["--dagger-base", base]
    if p is not None:
        t = r.setf(t, "--dagger-p", p)
    if extra:
        t = r.setf(t, *extra)
    return t


def mint_ki(r, toks=None, token=None, tag="", mod=None):
    mod = mod or NEW
    r.patch_heavy(mod)
    toks = toks if toks is not None else ki_toks(r)
    rp, bp = r.argv_file(f"KI_s{r.seed}{tag}", toks), r.argv_file(f"B0_s{r.seed}{tag}_ki", r.b0_toks)
    a = r.mint_ns(f"KI_s{r.seed}", rp, f"KI_s{r.seed}{tag}", pair=bp, token=token or r.gen, selfcheck=r.sc_rh)
    return mod.mint(a), a.gate_dir


def test_ki_the_arms_exist_and_the_original_kinds_are_unchanged():
    assert NEW.RH_ARMS["KI_s0"] == ("KI", 0) and NEW.RH_ARMS["KI_s1"] == ("KI", 1)
    assert NEW.RH_ARMS["RH_s0"] == ("RH", 0) and NEW.RH_ARMS["RHp3_s1"] == ("RHp3", 1) and NEW.RH_ARMS["RHK_s0"] == ("RHK", 0) and NEW.RH_ARMS["B0_s1"] == ("B0", 1)
    assert NEW.RH_ARMS["RK_s0"] == ("RK", 0) and NEW.RH_ARMS["RL_s1"] == ("RL", 1) and NEW.RH_ARMS["RA_s0"] == ("RA", 0)
    assert NEW.required_checks("KI_s0", "B") == NEW.required_checks("RH_s0", "B") and NEW.ARMS["KI_s1"] == NEW.ARMS["RH_s1"]
    assert NEW.KI_P == "0.1" and NEW.KI_BASE == [BASE] and NEW.KI_BASE_FLAG == "--dagger-base" and NEW.RH_TOKENS == ("causal_g1x", "causal_g1")


@pytest.mark.parametrize("seed", [0, 1])
def test_ki_mints_pass_and_binds_the_base_and_the_rh_bank(tmp_path, seed):
    r = Rig(tmp_path / f"ki{seed}", seed=seed)
    toks = ki_toks(r)
    a, b = NEW.parse_argv(toks), NEW.parse_argv(r.rh_toks)
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["--dagger-base", "--out"] and a["--dagger-base"] == [BASE] and len(toks) == len(r.rh_toks) + 2
    assert toks[-2:] == ["--dagger-base", BASE]
    assert NEW.rh_argv_problems(json.load(open(r.base))["argv"], r.tmpl_toks, toks, f"KI_s{seed}", r.gen) == []
    pr, gd = mint_ki(r, toks)
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["arm"] == f"KI_s{seed}" and tok["stage"] == "B" and tok["argv_sha256"] == NEW.argv_sha(toks)
    assert set(NEW.required_checks(f"KI_s{seed}", "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    assert tok["data"]["bank/bank.json"]["sha256"] == r.bank_sha and tok["checks"]["C-RH-BANKGEN"]["readout_token"] == r.gen
    r2 = Rig(tmp_path / f"ki{seed}_rh", seed=seed)                                                       # the RH and B0 kinds still mint: the new kind did not move them
    assert r2.mint("RH")[0] == [] and r2.mint("B0")[0] == []


def test_ki_the_stock_pair_check_is_not_patched():
    assert hashlib.sha256(open(T.PAIR_CHECK, "rb").read()).hexdigest() == "0dc30d9ef773be761a9b36a07d0fd73ef770e87d3384826230ba798db560869f"


# ---- the mutant arms: each must be refused, with a non-empty C-ARGV problem list
def _refused(r, toks, token=None, tag="m"):
    pr, _ = mint_ki(r, toks, token=token, tag=tag)
    assert pr, "the mutant was MINTED (the checker accepted it)"
    assert refused_by(pr, "C-ARGV"), pr
    return pr


def test_ki_mutant_a_second_changed_flag_is_refused(tmp_path):
    r = Rig(tmp_path / "ma")
    lr = r.tmpl_toks[r.tmpl_toks.index("--lr") + 1]
    pr = _refused(r, ki_toks(r, extra=("--lr", "2e-4" if lr != "2e-4" else "3e-4")))
    assert "--lr" in "\n".join(pr)
    pr = _refused(r, ki_toks(r) + ["--w-field", "2.0"], tag="m2")                                          # an added flag
    assert "--w-field" in "\n".join(pr)


def test_ki_mutant_b_another_bank_or_another_sha256_is_refused(tmp_path):
    r = Rig(tmp_path / "mb")
    t = ki_toks(r)
    other = os.path.join(r.tmp, "bank_k", f"s{r.seed}")                                                    # another bank directory (the K bank's place)
    pr = _refused(r, r.setf(t, "--dagger-bank", other), tag="bk")
    assert "bank" in "\n".join(pr)
    pr, _ = mint_ki(r, r.setf(t, "--dagger-bank-sha256", "ab" * 32), tag="bs")                             # another (well-formed) sha256: the binding and C-RH-BANKGEN compare it with bank.json
    assert pr and refused_by(pr, "C-RH-BANKGEN") and "sha256" in "\n".join(pr), pr


@pytest.mark.parametrize("base", ["own", "Logged", "LOGGED", "base", "none"])
def test_ki_mutant_c_another_base_value_under_the_ki_name_is_refused(tmp_path, base):
    r = Rig(tmp_path / "mc")
    pr = _refused(r, ki_toks(r, base=base))
    assert "--dagger-base" in "\n".join(pr)


def test_ki_mutant_d_the_base_flag_missing_is_refused(tmp_path):
    r = Rig(tmp_path / "md")
    pr = _refused(r, ki_toks(r, base=None))
    assert "--dagger-base" in "\n".join(pr)


@pytest.mark.parametrize("p", ["0.3", "0.5", "0.25", "1.0", "0.10", "0"])
def test_ki_mutant_e_any_dose_but_0_1_is_refused(tmp_path, p):
    r = Rig(tmp_path / "me")
    pr = _refused(r, ki_toks(r, p=p))
    assert "--dagger-p" in "\n".join(pr)


@pytest.mark.parametrize("flag,val", [("--dagger-mode", "add"), ("--dagger-mode", "swap"), ("--dagger-label", "kin_v1"), ("--dagger-label", "speed_v1"), ("--dagger-label-cache", "/root/rollhist/kin_cache/s0_kin_v1.npz"),
                                      ("--dagger-kin-tb", "0.5")])
def test_ki_a_third_lever_flag_is_refused_under_the_ki_name(tmp_path, flag, val):
    r = Rig(tmp_path / "mg")
    pr = _refused(r, ki_toks(r) + [flag, val], tag="mg")
    assert flag in "\n".join(pr)


def test_ki_the_base_flag_doubled_is_refused(tmp_path):
    r = Rig(tmp_path / "mh")
    t = ki_toks(r) + ["--dagger-base", BASE]
    try:
        pr, _ = mint_ki(r, t, tag="m2")
    except ValueError as e:                                                                                   # the stock parser refuses a repeated flag outright
        assert "appears twice" in str(e)
    else:
        assert pr, "a doubled base flag was MINTED"


@pytest.mark.parametrize("name", ["RH_s0", "RK_s0"])
def test_ki_the_rh_and_rk_names_may_not_carry_the_base_flag(tmp_path, name):                                  # `--dagger-base logged` is the KI arm's own token
    r = Rig(tmp_path / "mf")
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", name, "train")) + ["--dagger-base", BASE]
    r.patch_heavy(NEW)
    rp, bp = r.argv_file(name + "_base", t), r.argv_file("B0_s0_base", r.b0_toks)
    a = r.mint_ns(name, rp, name + "_base", pair=bp, token=r.gen, selfcheck=r.sc_rh)
    pr = NEW.mint(a)
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_ki_the_base_pair_is_dropped_before_the_stock_pair_check(tmp_path):
    r = Rig(tmp_path / "mj")
    pr, gd = mint_ki(r, ki_toks(r))
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["checks"]["C-RH-PAIR"]["pass"] is True
