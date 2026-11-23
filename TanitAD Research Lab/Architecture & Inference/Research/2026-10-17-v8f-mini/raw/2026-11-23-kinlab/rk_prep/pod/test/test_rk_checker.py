"""REVISION 2. Tests of the RK kind of the study checker (dscale10k_check_rk.py = the revision-1 checker 273befbc + patch_rk2.py): RH's argv + EXACTLY `--dagger-label kin_v1` + the execution-only pair
`--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` (Amendment 1) + its own --out, the RH bank under bank/, dose 0.1 (KINLAB). The cache pair is admitted once, for THIS seed only.
CPU only, on the synthetic rig of test_dscale10k_check_rhk.py (the same Rig). Run from this folder:  python -I -m pytest -q test_rk_checker.py
The mutant arms the checker must refuse, each with a non-empty C-ARGV problem list: a second changed flag; another bank / another sha256; `speed_v1` or `human` under the RK name; the label flag missing; dose other than 0.1; a mode flag added."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RK_CHECKER = os.environ.get("DS_CHECK_RK_PATH") or os.path.join(os.path.dirname(HERE), "dscale10k_check_rk.py")
os.environ["DS_CHECK_RH_PATH"] = RK_CHECKER                                 # the base test module loads its NEW checker from here
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_dscale10k_check_rhk as T  # noqa: E402
from test_dscale10k_check_rhk import Rig, refused_by  # noqa: E402,F401

NEW = T.NEW
LABEL = "kin_v1"


def cache_of(r, seed=None, kind="kin_v1"):
    return os.path.join(r.tmp, "kin_cache", f"s{r.seed if seed is None else seed}_{kind}.npz")


def rk_toks(r, label=LABEL, p=None, extra=None, cache="default"):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"RK_s{r.seed}", "train"))
    if label is not None:
        t = t + ["--dagger-label", label]
    if cache == "default":
        t = t + ["--dagger-label-cache", cache_of(r)]
    elif cache is not None:
        t = t + ["--dagger-label-cache", cache]
    if p is not None:
        t = r.setf(t, "--dagger-p", p)
    if extra:
        t = r.setf(t, *extra)
    return t


def mint_rk(r, toks=None, token=None, tag="", mod=None):
    mod = mod or NEW
    r.patch_heavy(mod)
    toks = toks if toks is not None else rk_toks(r)
    rp, bp = r.argv_file(f"RK_s{r.seed}{tag}", toks), r.argv_file(f"B0_s{r.seed}{tag}_rk", r.b0_toks)
    a = r.mint_ns(f"RK_s{r.seed}", rp, f"RK_s{r.seed}{tag}", pair=bp, token=token or r.gen, selfcheck=r.sc_rh)
    return mod.mint(a), a.gate_dir


def test_rk_the_arms_exist_and_the_original_kinds_are_unchanged():
    assert NEW.RH_ARMS["RK_s0"] == ("RK", 0) and NEW.RH_ARMS["RK_s1"] == ("RK", 1)
    assert NEW.RH_ARMS["RH_s0"] == ("RH", 0) and NEW.RH_ARMS["RHp3_s1"] == ("RHp3", 1) and NEW.RH_ARMS["RHK_s0"] == ("RHK", 0) and NEW.RH_ARMS["B0_s1"] == ("B0", 1)
    assert NEW.required_checks("RK_s0", "B") == NEW.required_checks("RH_s0", "B") and NEW.ARMS["RK_s1"] == NEW.ARMS["RH_s1"]
    assert NEW.RK_P == "0.1" and NEW.RK_LABEL == [LABEL] and NEW.RH_TOKENS == ("causal_g1x", "causal_g1")


@pytest.mark.parametrize("seed", [0, 1])
def test_rk_mints_pass_and_binds_the_label_and_the_rh_bank(tmp_path, seed):
    r = Rig(tmp_path / f"rk{seed}", seed=seed)
    toks = rk_toks(r)
    a, b = NEW.parse_argv(toks), NEW.parse_argv(r.rh_toks)
    assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == ["--dagger-label", "--dagger-label-cache", "--out"] and a["--dagger-label"] == [LABEL] and len(toks) == len(r.rh_toks) + 4
    assert a["--dagger-label-cache"] == [cache_of(r)] and toks[-2:] == ["--dagger-label-cache", cache_of(r)]
    assert NEW.rh_argv_problems(json.load(open(r.base))["argv"], r.tmpl_toks, toks, f"RK_s{seed}", r.gen) == []
    pr, gd = mint_rk(r, toks)
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["arm"] == f"RK_s{seed}" and tok["stage"] == "B" and tok["argv_sha256"] == NEW.argv_sha(toks)
    assert set(NEW.required_checks(f"RK_s{seed}", "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    assert tok["data"]["bank/bank.json"]["sha256"] == r.bank_sha and tok["checks"]["C-RH-BANKGEN"]["readout_token"] == r.gen
    r2 = Rig(tmp_path / f"rk{seed}_rh", seed=seed)                                                       # the RH and B0 kinds still mint: the new kind did not move them
    assert r2.mint("RH")[0] == [] and r2.mint("B0")[0] == []


def test_rk_the_stock_pair_check_is_not_patched():
    assert hashlib.sha256(open(T.PAIR_CHECK, "rb").read()).hexdigest() == "0dc30d9ef773be761a9b36a07d0fd73ef770e87d3384826230ba798db560869f"


# ---- the five mutant arms: each must be refused, with a non-empty C-ARGV problem list
def _refused(r, toks, token=None, tag="m"):
    pr, _ = mint_rk(r, toks, token=token, tag=tag)
    assert pr, "the mutant was MINTED (the checker accepted it)"
    assert refused_by(pr, "C-ARGV"), pr
    return pr


def test_rk_mutant_a_second_changed_flag_is_refused(tmp_path):
    r = Rig(tmp_path / "ma")
    lr = r.tmpl_toks[r.tmpl_toks.index("--lr") + 1]
    pr = _refused(r, rk_toks(r, extra=("--lr", "2e-4" if lr != "2e-4" else "3e-4")))
    assert "--lr" in "\n".join(pr)
    pr = _refused(r, rk_toks(r) + ["--w-field", "2.0"], tag="m2")                                          # an added flag
    assert "--w-field" in "\n".join(pr)


def test_rk_mutant_b_another_bank_or_another_sha256_is_refused(tmp_path):
    r = Rig(tmp_path / "mb")
    t = rk_toks(r)
    other = os.path.join(r.tmp, "bank_k", f"s{r.seed}")                                                    # another bank directory (the K bank's place)
    pr = _refused(r, r.setf(t, "--dagger-bank", other), tag="bk")
    assert "bank" in "\n".join(pr)
    pr, _ = mint_rk(r, r.setf(t, "--dagger-bank-sha256", "ab" * 32), tag="bs")                             # another (well-formed) sha256: C-ARGV cannot see the file; the binding and C-RH-BANKGEN compare it with bank.json
    assert pr and refused_by(pr, "C-RH-BANKGEN") and "sha256" in "\n".join(pr), pr
    assert "check C-ARGV did not pass" not in "\n".join(pr), "recorded fact: a well-formed foreign sha256 is not a C-ARGV problem; C-RH-BANKGEN and the binding refuse it"


@pytest.mark.parametrize("lab", ["speed_v1", "human", "kin_v2", "KIN_V1"])
def test_rk_mutant_c_another_label_under_the_rk_name_is_refused(tmp_path, lab):
    r = Rig(tmp_path / "mc")
    pr = _refused(r, rk_toks(r, label=lab))
    assert "--dagger-label" in chr(10).join(pr)


def test_rk_mutant_d_the_label_flag_missing_is_refused(tmp_path):
    r = Rig(tmp_path / "md")
    pr = _refused(r, rk_toks(r, label=None))
    assert "--dagger-label" in "\n".join(pr)


@pytest.mark.parametrize("p", ["0.3", "0.5", "0.25", "1.0", "0.10", "0"])
def test_rk_mutant_e_any_dose_but_0_1_is_refused(tmp_path, p):
    r = Rig(tmp_path / "me")
    pr = _refused(r, rk_toks(r, p=p))
    assert "--dagger-p" in "\n".join(pr)


def test_rk_a_mode_flag_is_refused_under_the_rk_name(tmp_path):
    r = Rig(tmp_path / "mg")
    pr = _refused(r, rk_toks(r) + ["--dagger-mode", "add"], tag="mg")
    assert "--dagger-mode" in chr(10).join(pr)


def test_rk_the_rh_name_may_not_carry_the_label_flag(tmp_path):                                              # the kinematic label under the RH name: RH admits no `--dagger-label`
    r = Rig(tmp_path / "mf")
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", "RH_s0", "train")) + ["--dagger-label", LABEL]
    r.patch_heavy(NEW)
    rp, bp = r.argv_file("RH_s0_lab", t), r.argv_file("B0_s0_lab", r.b0_toks)
    a = r.mint_ns("RH_s0", rp, "RH_s0_lab", pair=bp, token=r.gen, selfcheck=r.sc_rh)
    pr = NEW.mint(a)
    assert pr and refused_by(pr, "C-ARGV"), pr


# ---- revision 2: the cache token pair (Amendment 1)
def test_rk2_the_cache_pair_is_admitted_once_for_this_seed_only(tmp_path):
    for seed in (0, 1):
        r = Rig(tmp_path / f"c{seed}", seed=seed)
        pr, _ = mint_rk(r, rk_toks(r))
        assert pr == [], pr
        assert NEW.rh_argv_problems(json.load(open(r.base))["argv"], r.tmpl_toks, rk_toks(r), f"RK_s{seed}", r.gen) == []


def test_rk2_mutant_f_the_other_seeds_cache_is_refused(tmp_path):
    r = Rig(tmp_path / "mf0", seed=0)
    pr = _refused(r, rk_toks(r, cache=cache_of(r, seed=1)))
    assert "kin_cache" in "\n".join(pr) and "s0_kin_v1.npz" in "\n".join(pr)
    r1 = Rig(tmp_path / "mf1", seed=1)
    pr = _refused(r1, rk_toks(r1, cache=cache_of(r1, seed=0)))
    assert "s1_kin_v1.npz" in "\n".join(pr)


def test_rk2_mutant_g_a_cache_of_another_kind_or_folder_is_refused(tmp_path):
    r = Rig(tmp_path / "mg2")
    pr = _refused(r, rk_toks(r, cache=cache_of(r, kind="speed_v1")))
    assert "kin_cache" in "\n".join(pr)
    pr = _refused(r, rk_toks(r, cache=os.path.join(r.tmp, "elsewhere", f"s{r.seed}_kin_v1.npz")), tag="m2")
    assert "kin_cache" in "\n".join(pr)


def test_rk2_mutant_h_the_cache_pair_missing_or_doubled_is_refused(tmp_path):
    r = Rig(tmp_path / "mh")
    pr = _refused(r, rk_toks(r, cache=None))
    assert "--dagger-label-cache" in "\n".join(pr)
    t = rk_toks(r) + ["--dagger-label-cache", cache_of(r)]
    try:
        pr, _ = mint_rk(r, t, tag="m2")
    except ValueError as e:                                                                                   # the stock parser refuses a repeated flag outright
        assert "appears twice" in str(e)
    else:
        assert pr, "a doubled cache flag was MINTED"


def test_rk2_the_rh_and_b0_names_may_not_carry_the_cache_flag(tmp_path):
    r = Rig(tmp_path / "mi")
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", "RH_s0", "train")) + ["--dagger-label-cache", cache_of(r)]
    r.patch_heavy(NEW)
    rp, bp = r.argv_file("RH_s0_cache", t), r.argv_file("B0_s0_cache", r.b0_toks)
    a = r.mint_ns("RH_s0", rp, "RH_s0_cache", pair=bp, token=r.gen, selfcheck=r.sc_rh)
    pr = NEW.mint(a)
    assert pr and refused_by(pr, "C-ARGV"), pr
