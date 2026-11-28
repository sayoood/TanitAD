"""Tests of the RA kind of the study checker (dscale10k_check_ra.py = dscale10k_check_rhk.py 1bd28255 + patch_rl.py): RH's argv + EXACTLY `--dagger-mode add` + its own --out, the RH bank under bank/, dose 0.1.
CPU only, on the synthetic rig of test_dscale10k_check_rhk.py (the same Rig). Run from this folder:  python -I -m pytest -q test_ra_checker.py
The five mutant arms the checker must refuse, each with a non-empty C-ARGV problem list: a second changed flag; another bank / another sha256; `swap` under the RA name; the mode flag missing; dose 0.3."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RA_CHECKER = os.environ.get("DS_CHECK_RA_PATH") or os.path.join(os.path.dirname(HERE), "dscale10k_check_ra.py")
os.environ["DS_CHECK_RH_PATH"] = RA_CHECKER                                 # the base test module loads its NEW checker from here
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_dscale10k_check_rhk as T  # noqa: E402
from test_dscale10k_check_rhk import Rig, refused_by  # noqa: E402,F401

NEW = T.NEW
MODE = "add"


def ra_toks(r, mode=MODE, p=None, extra=None):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"RA_s{r.seed}", "train"))
    if mode is not None:
        t = t + ["--dagger-mode", mode]
    if p is not None:
        t = r.setf(t, "--dagger-p", p)
    if extra:
        t = r.setf(t, *extra)
    return t


def mint_ra(r, toks=None, token=None, tag="", mod=None):
    mod = mod or NEW
    r.patch_heavy(mod)
    toks = toks if toks is not None else ra_toks(r)
    rp, bp = r.argv_file(f"RA_s{r.seed}{tag}", toks), r.argv_file(f"B0_s{r.seed}{tag}_rl", r.b0_toks)
    a = r.mint_ns(f"RA_s{r.seed}", rp, f"RA_s{r.seed}{tag}", pair=bp, token=token or r.gen, selfcheck=r.sc_rh)
    return mod.mint(a), a.gate_dir


def test_ra_the_arms_exist_and_the_original_kinds_are_unchanged():
    assert NEW.RH_ARMS["RA_s0"] == ("RA", 0) and NEW.RH_ARMS["RA_s1"] == ("RA", 1)
    assert NEW.RH_ARMS["RH_s0"] == ("RH", 0) and NEW.RH_ARMS["RHp3_s1"] == ("RHp3", 1) and NEW.RH_ARMS["RHK_s0"] == ("RHK", 0) and NEW.RH_ARMS["B0_s1"] == ("B0", 1)
    assert NEW.required_checks("RA_s0", "B") == NEW.required_checks("RH_s0", "B") and NEW.ARMS["RA_s1"] == NEW.ARMS["RH_s1"]
    assert NEW.RA_P == "0.1" and NEW.RA_MODE == [MODE] and NEW.RH_TOKENS == ("causal_g1x", "causal_g1")


@pytest.mark.parametrize("seed", [0, 1])
def test_ra_mints_pass_and_binds_the_mode_and_the_rh_bank(tmp_path, seed):
    r = Rig(tmp_path / f"rl{seed}", seed=seed)
    toks = ra_toks(r)
    a, b = NEW.parse_argv(toks), NEW.parse_argv(r.rh_toks)
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["--dagger-mode", "--out"] and a["--dagger-mode"] == [MODE] and len(toks) == len(r.rh_toks) + 2
    assert NEW.rh_argv_problems(json.load(open(r.base))["argv"], r.tmpl_toks, toks, f"RA_s{seed}", r.gen) == []
    pr, gd = mint_ra(r, toks)
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["arm"] == f"RA_s{seed}" and tok["stage"] == "B" and tok["argv_sha256"] == NEW.argv_sha(toks)
    assert set(NEW.required_checks(f"RA_s{seed}", "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    assert tok["data"]["bank/bank.json"]["sha256"] == r.bank_sha and tok["checks"]["C-RH-BANKGEN"]["readout_token"] == r.gen
    r2 = Rig(tmp_path / f"rl{seed}_rh", seed=seed)                                                       # the RH and B0 kinds still mint: the new kind did not move them
    assert r2.mint("RH")[0] == [] and r2.mint("B0")[0] == []


def test_ra_the_stock_pair_check_is_not_patched():
    assert hashlib.sha256(open(T.PAIR_CHECK, "rb").read()).hexdigest() == "0dc30d9ef773be761a9b36a07d0fd73ef770e87d3384826230ba798db560869f"


# ---- the five mutant arms: each must be refused, with a non-empty C-ARGV problem list
def _refused(r, toks, token=None, tag="m"):
    pr, _ = mint_ra(r, toks, token=token, tag=tag)
    assert pr, "the mutant was MINTED (the checker accepted it)"
    assert refused_by(pr, "C-ARGV"), pr
    return pr


def test_ra_mutant_a_second_changed_flag_is_refused(tmp_path):
    r = Rig(tmp_path / "ma")
    lr = r.tmpl_toks[r.tmpl_toks.index("--lr") + 1]
    pr = _refused(r, ra_toks(r, extra=("--lr", "2e-4" if lr != "2e-4" else "3e-4")))
    assert "--lr" in "\n".join(pr)
    pr = _refused(r, ra_toks(r) + ["--w-field", "2.0"], tag="m2")                                          # an added flag
    assert "--w-field" in "\n".join(pr)


def test_ra_mutant_b_another_bank_or_another_sha256_is_refused(tmp_path):
    r = Rig(tmp_path / "mb")
    t = ra_toks(r)
    other = os.path.join(r.tmp, "bank_k", f"s{r.seed}")                                                    # another bank directory (the K bank's place)
    pr = _refused(r, r.setf(t, "--dagger-bank", other), tag="bk")
    assert "bank" in "\n".join(pr)
    pr, _ = mint_ra(r, r.setf(t, "--dagger-bank-sha256", "ab" * 32), tag="bs")                             # another (well-formed) sha256: C-ARGV cannot see the file; the binding and C-RH-BANKGEN compare it with bank.json
    assert pr and refused_by(pr, "C-RH-BANKGEN") and "sha256" in "\n".join(pr), pr
    assert "check C-ARGV did not pass" not in "\n".join(pr), "recorded fact: a well-formed foreign sha256 is not a C-ARGV problem; C-RH-BANKGEN and the binding refuse it"


def test_ra_mutant_c_swap_under_the_rl_name_is_refused(tmp_path):
    r = Rig(tmp_path / "mc")
    pr = _refused(r, ra_toks(r, mode="swap"))
    assert "--dagger-mode" in "\n".join(pr)


def test_ra_mutant_d_the_mode_flag_missing_is_refused(tmp_path):
    r = Rig(tmp_path / "md")
    pr = _refused(r, ra_toks(r, mode=None))
    assert "--dagger-mode" in "\n".join(pr)


@pytest.mark.parametrize("p", ["0.3", "0.5", "0.25", "1.0", "0.10", "0"])
def test_ra_mutant_e_any_dose_but_0_1_is_refused(tmp_path, p):
    r = Rig(tmp_path / "me")
    pr = _refused(r, ra_toks(r, p=p))
    assert "--dagger-p" in "\n".join(pr)


def test_ra_the_rh_name_may_not_carry_the_mode_flag(tmp_path):                                              # the add-mode lever under the RH name: RH admits no `--dagger-mode`
    r = Rig(tmp_path / "mf")
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", "RH_s0", "train")) + ["--dagger-mode", MODE]
    r.patch_heavy(NEW)
    rp, bp = r.argv_file("RH_s0_lab", t), r.argv_file("B0_s0_lab", r.b0_toks)
    a = r.mint_ns("RH_s0", rp, "RH_s0_lab", pair=bp, token=r.gen, selfcheck=r.sc_rh)
    pr = NEW.mint(a)
    assert pr and refused_by(pr, "C-ARGV"), pr
