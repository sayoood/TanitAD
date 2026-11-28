"""Tests of kit/dscale10k_check_l3.py (the LEVER3 arms D3K / D9K) and the identity test against the pod's dscale10k_check_s3.py.
Every new `*_problems` path has a deliberate-regression arm that must be non-empty. Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest kit/tests_l3 -q  (conftest.py loads both modules)."""
import copy
import difflib
import hashlib
import json
import os
import subprocess
import sys

import pytest

from conftest import KIT, L3, POD_S3, S3

sys.path.insert(0, KIT)
import l3_argv as A  # noqa: E402
import l3_chk as K  # noqa: E402

ROOT = "C:/Users/Admin/lever3"
SHA = "ab" * 32
ARMS = ("D3K_s0", "D3K_s1", "D9K_s0", "D9K_s1")


def fx(arm):
    seed = arm.split("_s")[1]
    return A.render(arm, SHA, ROOT), K.mapped_base(ROOT), K.mapped_u_template(int(seed), ROOT)


def setf(t, f, v):
    t = list(t)
    t[t.index(f) + 1] = v
    return t


# ---------------------------------------------------------------- the new arms: clean + regressions
@pytest.mark.parametrize("arm", ARMS)
def test_clean_argv_passes(arm):
    t, base, tm = fx(arm)
    assert L3.rh_argv_problems(base, tm, t, arm, "causal_g1x") == []


@pytest.mark.parametrize("arm", ARMS)
def test_regressions_refused(arm):
    t, base, tm = fx(arm)
    kind = arm[:3]
    other_steps = "9000" if kind == "D3K" else "3000"
    muts = {
        "lr": setf(t, "--lr", "2e-4"), "batch": setf(t, "--batch", "32"), "steps_swapped": setf(t, "--steps", other_steps), "warmup": setf(t, "--warmup", "300"),
        "dagger_p": setf(t, "--dagger-p", "0.3"), "dagger_p0": setf(t, "--dagger-p", "0"), "readout": setf(t, "--dagger-bank-readout", "causal_g1"), "kappa": setf(t, "--dagger-bank-kappa", "0.3"),
        "bank_other_seed": setf(t, "--dagger-bank", f"{ROOT}/bank/s{1 - int(arm[-1])}"), "bank_k": setf(t, "--dagger-bank", f"{ROOT}/bank_k/s{arm[-1]}"),
        "bank_sha_short": setf(t, "--dagger-bank-sha256", "abc"), "bank_placeholder": setf(t, "--dagger-bank-sha256", "@BANKSHA@"),
        "init_runs_layout": setf(t, "--init-ckpt", f"{ROOT}/runs/UE_s{arm[-1]}/A/ckpt_last.pt"), "init_other_seed": setf(t, "--init-ckpt", f"{ROOT}/init/UE_s{1 - int(arm[-1])}/A/ckpt_last.pt"),
        "out_template": setf(t, "--out", tm[tm.index("--out") + 1]), "out_other_arm": setf(t, "--out", f"{ROOT}/runs/RH_s0/train"),
        "data_flag": setf(t, "--tracks", "C:/elsewhere/tracks.pt"), "p10": setf(t, "--p10", "C:/elsewhere/p10.pt"),
        "extra_flag": list(t) + ["--w-field", "2.0"], "dropped_flag": [x for x in t if x != "--detach-upper"], "dropped_fitjoin": [x for x in t if x != "--fit-join"],
    }
    for name, m in muts.items():
        assert L3.rh_argv_problems(base, tm, m, arm, "causal_g1x"), f"{arm}: mutation {name} was not refused"


def test_token_mismatch_refused():
    t, base, tm = fx("D3K_s0")
    assert L3.rh_argv_problems(base, tm, t, "D3K_s0", "causal_g1")


def test_d3k_cannot_carry_9000_and_d9k_cannot_carry_3000():
    for arm, bad in (("D3K_s0", "9000"), ("D9K_s0", "3000")):
        t, base, tm = fx(arm)
        assert any("--steps" in p for p in L3.rh_argv_problems(base, tm, setf(t, "--steps", bad), arm, "causal_g1x"))


def test_steps_problems_stage_b():
    rec = {"S": 1, "n_clips": 14512}
    for arm, steps, ok in (("D3K_s0", "3000", True), ("D3K_s0", "9000", False), ("D9K_s1", "9000", True), ("D9K_s1", "3000", False), ("RH_s0", "3000", True), ("RH_s0", "9000", False)):
        toks = ["--arm", "M2b", "--steps", steps]
        assert (L3.steps_problems(arm, toks, rec, None, "B") == []) is ok, (arm, steps)


def test_pair():
    for seed in (0, 1):
        a3, _, _ = fx(f"D3K_s{seed}")
        a9, _, _ = fx(f"D9K_s{seed}")
        assert L3.l3_pair_problems(a3, a9, f"D3K_s{seed}", "causal_g1x") == []
        assert L3.l3_pair_problems(a9, a3, f"D9K_s{seed}", "causal_g1x") == []
        regress = {
            "lr": setf(a9, "--lr", "2e-4"), "dropped": [x for x in a9 if x != "--detach-upper"], "same_out": setf(a9, "--out", a3[a3.index("--out") + 1]),
            "steps_same": setf(a9, "--steps", "3000"), "bank": setf(a9, "--dagger-bank-sha256", "cd" * 32), "seed": setf(a9, "--seed", "7"), "init": setf(a9, "--init-ckpt", "C:/x/ckpt.pt"),
            "token": setf(a9, "--dagger-bank-readout", "causal_g1"), "extra": list(a9) + ["--w-field", "2.0"],
        }
        for n, m in regress.items():
            assert L3.l3_pair_problems(a3, m, f"D3K_s{seed}", "causal_g1x"), f"pair mutation {n} was not refused"
    assert L3.l3_pair_problems(a3, a9, "RH_s0", "causal_g1x")        # not an L3 arm


def test_required_checks_and_tables():
    for arm in ARMS:
        assert L3.required_checks(arm, "B") == L3.REQUIRED_DATA + L3.REQUIRED_UNION + L3.REQUIRED_TRAINER + L3.REQUIRED_RH_RH
        assert L3.RH_ARMS[arm][0] in L3.RH_LIKE and L3.ARMS[arm]["set"] == "U" and L3.ARMS[arm]["stage_b"]
    assert L3.GATE_TOKEN_FILE == "gate_gatetok.json"


def test_template_vs_base_is_accepted():
    for seed in (0, 1):
        assert L3.argv_diff_problems(K.mapped_base(ROOT), K.mapped_u_template(seed, ROOT), f"U_s{seed}", "B") == []


def test_dev_argv_equals_rh_template_except_declared(tmp_path):
    """D3K's argv == the RH_sK pod template with the paths, bank and sha mapped and nothing else (control b4, via l3_argv.verify_one)"""
    for arm in ARMS:
        t, _, _ = fx(arm)
        assert A.verify_one(arm, t, ROOT, SHA) == []


# ---------------------------------------------------------------- identity: every other arm, check and token field is UNCHANGED
def test_identity_source_is_the_patch_output():
    """the kit file is exactly patch_l3.py applied to the pod's file; every removed line of the diff is one of the known anchors"""
    out = KIT + "/_identity_tmp.py"
    r = subprocess.run([sys.executable, "-I", KIT + "/patch_l3.py", POD_S3, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    try:
        assert hashlib.sha256(open(out, "rb").read()).hexdigest() == hashlib.sha256(open(KIT + "/dscale10k_check_l3.py", "rb").read()).hexdigest()
    finally:
        os.remove(out)
    a = open(POD_S3, encoding="utf-8").read().splitlines()
    b = open(KIT + "/dscale10k_check_l3.py", encoding="utf-8").read().splitlines()
    removed = [ln[2:] for ln in difflib.ndiff(a, b) if ln.startswith("- ")]
    allowed = ("SEED3 build", "RH_TOKENS =", "def rh_stage_a_ckpt_problems", "want = f\"/runs/UE_s", "return [] if _norm_path(init).endswith(want)", "bad += rh_stage_a_ckpt_problems(init, seed)",
               "exp_added, exp_removed, own = set(RH_DECL)", "if kind == \"RH\":", "return [] if t == STAGE_B_STEPS", "if pair is None:", "\"gate_token.json\"", "TOKEN_SCHEMA =")
    unknown = [r for r in removed if not any(r.strip().startswith(x) or x in r for x in allowed) and r.strip()]
    assert not unknown, unknown


def test_identity_tables_and_required_checks():
    if S3 is None:
        pytest.skip("pod checker copy not present")
    for arm in S3.ARMS:
        assert S3.ARMS[arm] == L3.ARMS[arm], arm
        for st in ("A", "B"):
            assert S3.required_checks(arm, st) == L3.required_checks(arm, st), (arm, st)
    assert {k: v for k, v in L3.RH_ARMS.items() if k not in L3.L3_ARMS} == S3.RH_ARMS
    assert set(L3.ARMS) - set(S3.ARMS) == set(L3.L3_ARMS)
    assert L3.RH_LIKE[:len(S3.RH_LIKE)] == S3.RH_LIKE and L3.RH_LIKE[len(S3.RH_LIKE):] == ("D3K", "D9K")
    for n in ("TOKEN_SCHEMA", "WRITTEN_BY", "STAGE_B_STEPS", "RUN_STEPS_A", "RH_KAPPA", "BANK_SHARDS", "REQUIRED_TRAINER", "REQUIRED_DATA", "REQUIRED_UNION", "REQUIRED_RH_RH", "REQUIRED_RH_B0", "TRAINER_REL", "DATA_BIND_FLAGS"):
        assert getattr(S3, n) == getattr(L3, n), n


def _old_arm_argvs():
    tm = json.load(open(KIT + "/inputs/records/dscale10k/argv/U_s0.B.tmpl.json"))["argv"]
    base = json.load(open(KIT + "/inputs/records/dscale10k/argv/B_s0.B.tmpl.json"))["argv"]
    ue = "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt"
    rh = list(tm)
    rh = setf(rh, "--out", "/root/rollhist/runs/RH_s0/train")
    rh = setf(rh, "--init-ckpt", ue)
    rh = setf(rh, "--dagger-bank", "/root/rollhist/bank/s0")
    rh = setf(rh, "--dagger-bank-sha256", SHA) + ["--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2"]
    b0 = [x for x in setf(setf(setf(tm, "--out", "/root/rollhist/runs/B0_s0/train"), "--init-ckpt", ue), "--dagger-p", "0") if x not in ()]
    i = b0.index("--dagger-bank")
    b0 = b0[:i] + b0[i + 4:]
    j = b0.index("--dagger-seed")
    b0 = b0[:j] + b0[j + 2:]
    out = {"RH_s0": (rh, "causal_g1x"), "B0_s0": (b0, None), "RHp3_s0": (setf(rh, "--dagger-p", "0.3"), "causal_g1x"), "KI_s0": (setf(rh, "--out", "/root/rollhist/runs/KI_s0/train") + ["--dagger-base", "logged"], "causal_g1x"),
           "RHK_s0": (setf(setf(rh, "--out", "/root/rollhist/runs/RHK_s0/train"), "--dagger-bank", "/root/rollhist/bank_k/s0"), "causal_g0x")}
    return base, tm, out


def test_identity_old_arms_same_answers():
    """rh_argv_problems for every pre-existing kind, clean and mutated, equals the pod checker's answer line by line"""
    if S3 is None:
        pytest.skip("pod checker copy not present")
    base, tm, arms = _old_arm_argvs()
    n = 0
    for arm, (t, tok) in arms.items():
        mutants = [t, setf(t, "--lr", "2e-4"), [x for x in t if x != "--detach-upper"], list(t) + ["--w-field", "2.0"], setf(t, "--steps", "9000"), setf(t, "--init-ckpt", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt"),
                   setf(t, "--out", "/root/x/train"), setf(t, "--tracks", "/elsewhere.pt")]
        for m in mutants:
            assert L3.rh_argv_problems(base, tm, m, arm, tok) == S3.rh_argv_problems(base, tm, m, arm, tok), arm
            n += 1
    assert n >= 40
    rec = {"S": 1, "n_clips": 14512}
    for arm in S3.ARMS:
        for steps in ("3000", "9000", "19164"):
            for stage in ("A", "B"):
                toks = ["--arm", "M2b", "--steps", steps]
                try:
                    exp = S3.steps_problems(arm, toks, rec, None, stage)
                except Exception as e:                                                  # an exception must be the same exception
                    with pytest.raises(type(e)):
                        L3.steps_problems(arm, toks, rec, None, stage)
                    continue
                assert L3.steps_problems(arm, toks, rec, None, stage) == exp, (arm, steps, stage)
    t, tk = arms["RH_s0"]
    assert S3.rh_stage_a_ckpt_problems("/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", 0) == L3.rh_stage_a_ckpt_problems("/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", 0)
    assert S3.rh_stage_a_ckpt_problems(f"{ROOT}/init/UE_s0/A/ckpt_last.pt", 0) == L3.rh_stage_a_ckpt_problems(f"{ROOT}/init/UE_s0/A/ckpt_last.pt", 0) != []   # only the L3 kinds accept the init/ layout
