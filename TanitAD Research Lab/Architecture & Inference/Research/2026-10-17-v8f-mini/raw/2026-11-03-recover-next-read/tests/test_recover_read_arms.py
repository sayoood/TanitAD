"""Tests of the --r-arm extension of code/recover_read.py (RAPR / RTL / RP01 in the R role).  The 27 source tests (tests/test_recover_read.py) run unchanged against the same file; these cover what is new:
the run table per arm, the arm-parametrised controls (KR1 band, KR2 argv audit, KR3), ARM_SPEC against the shipped argv files, and the CLI.

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q tests/test_recover_read_arms.py        (CPU only, seconds)
"""
import copy
import importlib.util
import json
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SRC = f"{PK}/code/recover_read.py"
SRC_ORIG = "E:/Projects/TanitAD-artifacts/dagland/recover_read/code/recover_read.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
DL = "E:/Projects/TanitAD-artifacts/dagland"
ARMS = ("RAPR", "RTL", "RP01")


def _load(path, name="rr_arms_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load(os.environ.get("RECOVER_READ_PATH", SRC))


def ft_tokens(name):
    return open(f"{DL}/recover_ft_launch/box2_files/argv/{name}.train.txt", encoding="utf-8").read().split()


def nx_tokens(name):
    return open(f"{DL}/recover_next/argv/{name}.train.txt", encoding="utf-8").read().split()


# ------------------------------------------------------------------------------------------------ run table
def test_arm_R_table_equals_the_source_table():
    """--r-arm R is the source reader's table, row for row (the extension adds the 'reused' flag, False everywhere)."""
    src = _load(SRC_ORIG, "rr_source")
    a, b = R.run_table("R"), src.run_table()
    assert [{k: v for k, v in r.items() if k != "reused"} for r in a] == b and not any(r["reused"] for r in a)
    assert R.run_table() == a                                                       # the default arm is R


@pytest.mark.parametrize("arm", ARMS)
def test_arm_table(arm):
    t = R.run_table(arm)
    assert len(t) == 15 and len({r["tag"] for r in t}) == 15
    new, reused = [r for r in t if not r["reused"]], [r for r in t if r["reused"]]
    assert [r["tag"] for r in new] == [f"{arm}_s0", f"KA1{arm}_s0", f"{arm}_s1", f"FZ{arm}_s0", f"FZ{arm}_s1", f"{arm}C_s0", f"{arm}C_s1"]      # 7 rollouts, the primary contrast first
    assert {r["tag"] for r in reused} == {"C0_s0", "C0_s1", "C0REP_s0", "C0REP_s1", "FZC0_s0", "FZC0_s1", "C0C_s0", "C0C_s1"}                      # the 8 existing rollouts, not re-rolled
    assert all(r["ckpt_run"] == f"M2b_s{r['seed']}_recft_{arm}" for r in new)
    assert {r["ckpt_run"] for r in reused} == {"M2b_s0_recft_C0", "M2b_s1_recft_C0", "M2b_s0_recft_C0rep", "M2b_s1_recft_C0rep"}
    base = {r["tag"]: r for r in R.run_table("R")}
    for r in t:                                                                     # scene / feasible / role per row equal the source's row of the same ROLE and seed
        s_ = [x for x in base.values() if x["role"] == r["role"] and x["seed"] == r["seed"]][0]
        assert (r["role"], r["seed"], r["scene"], r["feasible"]) == (s_["role"], s_["seed"], s_["scene"], s_["feasible"]) and R.run_flags(r) == R.run_flags(s_)
    assert R.tag_of(t, "R", 1) == f"{arm}_s1" and R.tag_of(t, "RC", 0) == f"{arm}C_s0" and R.tag_of(t, "FZC0", 1) == "FZC0_s1"
    with pytest.raises(ValueError):
        R.run_table("RAPR2")


def test_tag_dir_reuse():
    t = R.run_table("RTL")
    assert [R.tag_dir(r, "O", "U") for r in t[:3]] == ["O", "U", "O"]               # RTL_s0 new, C0_s0 reused, KA1RTL_s0 new


@pytest.mark.parametrize("arm", ("R",) + ARMS)
def test_cli_list_runs_and_dry_run(arm):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    lr = subprocess.run([PY, "-I", SRC, "--r-arm", arm, "--list-runs"], capture_output=True, text=True, env=env)
    lines = [ln for ln in lr.stdout.splitlines() if ln.strip()]
    assert lr.returncode == 0 and len(lines) == (15 if arm == "R" else 7) and all(len(ln.split("\t")) == 6 for ln in lines)
    dr = subprocess.run([PY, "-I", SRC, "--r-arm", arm, "--dry-run"], capture_output=True, text=True, env=env)
    assert dr.returncode == 0 and dr.stdout.count("[REUSED") == (0 if arm == "R" else 8) and f"r_arm={arm}" in dr.stdout


# ------------------------------------------------------------------------------------------------ ARM_SPEC against the shipped argv files
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("s", [0, 1])
def test_arm_spec_diff_equals_the_shipped_argv_difference(arm, s):
    d = R.argv_diff(R.parse_argv(nx_tokens(f"{arm}_{s}")), R.parse_argv(ft_tokens(f"R_{s}")))
    want = R.ARM_SPEC[arm]["diff_vs_R"]
    assert d["added"] == want["added"] and d["removed"] == sorted(want["removed"]) and d["changed"] == want["changed"]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("s", [0, 1])
def test_argv_audit_passes_on_the_shipped_argv_and_fails_on_each_break(arm, s):
    x, c0, rep, r = nx_tokens(f"{arm}_{s}"), ft_tokens(f"C0_{s}"), ft_tokens(f"C0rep_{s}"), ft_tokens(f"R_{s}")
    a = R.argv_audit(x, c0, rep, s, arm, r)
    assert a["PASS"], [k for k, v in a.items() if v is False]
    assert R.argv_audit(x, c0, rep, s, "R", r)["PASS"] is (arm == "R")             # an arm audited as R fails (RAPR drops the bank, RTL's target, RP01's p)
    for other in set(ARMS) - {arm}:                                                 # an arm's argv audited as another arm fails
        assert R.argv_audit(x, c0, rep, s, other, r)["PASS"] is False
    extra = x + ["--lr-extra", "1"]                                                 # a lever outside the declared change
    assert R.argv_audit(extra, c0, rep, s, arm, r)["R_differs_from_R_argv_only_by_the_declared_change"] is False
    wrong_seed = [("0" if (i > 0 and x[i - 1] == "--recover-seed" and s == 1) else ("1" if (i > 0 and x[i - 1] == "--recover-seed") else t)) for i, t in enumerate(x)]
    assert R.argv_audit(wrong_seed, c0, rep, s, arm, r)["PASS"] is False
    assert "R_differs_from_R_argv_only_by_the_declared_change" not in R.argv_audit(x, c0, rep, s, arm)          # no reference argv: that check is absent, the rest still passes
    assert R.argv_audit(ft_tokens(f"R_{s}"), c0, rep, s)["PASS"] and R.argv_audit(ft_tokens(f"R_{s}"), c0, rep, s, "R", r)["PASS"]


def test_arm_R_audit_keys_are_the_source_keys():
    """For arm R the audit's key set is exactly the source's (so RECOVER_READ_R.json reproduces RECOVER_READ.json key for key)."""
    src = _load(SRC_ORIG, "rr_source2")
    a, b = R.argv_audit(ft_tokens("R_0"), ft_tokens("C0_0"), ft_tokens("C0rep_0"), 0), src.argv_audit(ft_tokens("R_0"), ft_tokens("C0_0"), ft_tokens("C0rep_0"), 0)
    assert a == b and list(a) == list(b)


# ------------------------------------------------------------------------------------------------ KR1 / KR3
def test_kr1_band_scales_with_p():
    zero = {"recover_n": 0, "recover_p": 0.0}
    sel = lambda share, p: {"recover_n": int(share * 100000), "recover_eligible": 100000, "recover_p": p}   # noqa: E731
    assert R.kr1_check(sel(0.30, 0.3), zero, zero, "R")["PASS"] and R.kr1_check(sel(0.30, 0.3), zero, zero, "RAPR")["PASS"] and R.kr1_check(sel(0.30, 0.3), zero, zero, "RTL")["PASS"]
    assert R.kr1_check(sel(0.10, 0.1), zero, zero, "RP01")["PASS"]
    assert R.kr1_check(sel(0.30, 0.3), zero, zero, "RP01")["PASS"] is False                                    # 0.3 is outside the p 0.1 band
    assert R.kr1_check(sel(0.10, 0.1), zero, zero, "R")["PASS"] is False                                       # and 0.1 is outside the p 0.3 band
    assert R.kr1_check(sel(0.10, 0.3), zero, zero, "RP01")["PASS"] is False                                    # the recorded p must be the arm's
    assert R.kr1_check(sel(0.10, 0.1), {"recover_n": 1, "recover_p": 0.0}, zero, "RP01")["PASS"] is False     # a comparator that selected anything fails
    k = R.kr1_check(sel(0.10, 0.1), zero, zero, "RP01")
    assert k["arm"] == "RP01" and k["band"] == [0.0833, 0.1167]
    assert "arm" not in R.kr1_check(sel(0.30, 0.3), zero, zero)                                                 # arm R: the source's keys only


def _gate(arm, s):
    """A gate selfcheck.json of the arm's shape, built from the real banked R_0 gate evidence (the bank arms) or with the table / bank removed (the a-priori arm), levers set to the arm's declared ones."""
    sc = json.load(open(f"{DL}/recover_ft_launch/evidence/R_0/selfcheck.json", encoding="utf-8"))
    sc = copy.deepcopy(sc)
    spec = R.ARM_SPEC[arm]
    lv = sc["levers"]["recover"]
    lv.update({"recover_p": spec["p"], "recover_target": spec["target"], "recover_seed": s, "recover_rejoin_steps": 10})
    if s == 1:
        sc["info"]["recover"]["bank_sha256"] = R.BANK_SHA[1]
        lv["recover_dev_bank_sha256"] = R.BANK_SHA[1]
    if not spec["bank"]:
        lv.update({"recover_dev_law": "apriori", "recover_dev_bank": None, "recover_dev_bank_sha256": None})
        sc["levers"]["recover_table"] = None
        del sc["info"]["recover"]
    return sc


def test_kr3_apriori_arm_reads_no_table_no_bank():
    for s in (0, 1):
        assert R.kr3_check(_gate("RAPR", s), s, "RAPR")["PASS"] is True
    base = lambda: _gate("RAPR", 0)                                                                             # noqa: E731
    sc = base(); sc["info"]["recover"] = {"table": {"n": 5}}
    assert R.kr3_check(sc, 0, "RAPR")["PASS"] is False                                                          # a table build record exists: the law was not the a-priori one
    sc = base(); sc["levers"]["recover_table"] = {"n": 5}
    assert R.kr3_check(sc, 0, "RAPR")["PASS"] is False
    sc = base(); sc["levers"]["recover"]["recover_dev_law"] = "empirical"
    assert R.kr3_check(sc, 0, "RAPR")["PASS"] is False
    sc = base(); sc["levers"]["recover"]["recover_dev_bank_sha256"] = R.BANK_SHA[0]
    assert R.kr3_check(sc, 0, "RAPR")["PASS"] is False
    sc = base(); sc["levers"]["recover"]["recover_p"] = 0.1
    assert R.kr3_check(sc, 0, "RAPR")["PASS"] is False
    assert R.kr3_check(base(), 1, "RAPR")["PASS"] is False                                                      # seed 1 expects recover_seed 1
    assert R.kr3_check(_gate("RAPR", 0), 0, "R")["PASS"] is False                                               # read as arm R (a table is required) the a-priori selfcheck fails


@pytest.mark.parametrize("arm", ("RTL", "RP01"))
def test_kr3_bank_arms_read_the_table_and_the_built_levers(arm):
    assert R.kr3_check(_gate(arm, 0), 0, arm)["PASS"] is True
    sc = _gate(arm, 0); sc["levers"]["recover"]["recover_p"] = 0.3 if arm == "RP01" else 0.1
    assert R.kr3_check(sc, 0, arm)["PASS"] is False                                                             # built p differs from the arm's declared one
    sc = _gate(arm, 0); sc["levers"]["recover"]["recover_target"] = "rejoin" if arm == "RTL" else "logged"
    assert R.kr3_check(sc, 0, arm)["PASS"] is False
    sc = _gate(arm, 0); sc["info"]["recover"]["kb3"]["overlap_with_eval"] = 1
    assert R.kr3_check(sc, 0, arm)["PASS"] is False                                                             # K-B3 still binds
    sc = _gate(arm, 0); sc["levers"]["recover"]["recover_dev_law"] = "apriori"
    assert R.kr3_check(sc, 0, arm)["PASS"] is False


def test_kr3_arm_R_is_the_source_check():
    sc = json.load(open(f"{DL}/recover_ft_launch/evidence/R_0/selfcheck.json", encoding="utf-8"))
    assert R.kr3_check(sc, 0)["PASS"] and R.kr3_check(sc, 0, "R") == R.kr3_check(sc, 0) and "built_levers" not in R.kr3_check(sc, 0)


# ------------------------------------------------------------------------------------------------ production-path rehearsal of file_controls for an arm (stand-ins; crash-freedom and wiring, not a result)
@pytest.mark.parametrize("arm", ARMS)
def test_file_controls_run_to_completion_for_each_arm_on_stand_ins(arm, tmp_path, monkeypatch):
    """A crash in file_controls would cost a 45-minute chain.  Build a recover dir whose R-role run is the ARM (its json files from the shipped argv, the gate-shaped selfcheck and a summary at the arm's p),
    redirect the arm's rollout tags to the existing R rollouts (read only), fake the checkpoint hashes, and require: no exception, the arm-specific control names present, and every json-only control PASS."""
    import argparse
    import hashlib
    import re
    import shutil
    R.heavy_init()
    real_sha = R.sha256_file
    monkeypatch.setattr(R, "sha256_file", lambda p: hashlib.sha256(str(p).encode()).hexdigest() if str(p).endswith("ckpt_last.pt") else real_sha(p))
    real_rd, rd = "E:/Projects/TanitAD-artifacts/v8f-recover", str(tmp_path / "rd")
    spec = R.ARM_SPEC[arm]
    for s in (0, 1):
        for base in ("R", "C0", "C0rep"):
            os.makedirs(f"{rd}/M2b_s{s}_recft_{base}", exist_ok=True)
            for f in ("summary.json", "argv.json", "gate_selfcheck.json"):
                shutil.copy(f"{real_rd}/M2b_s{s}_recft_{base}/{f}", f"{rd}/M2b_s{s}_recft_{base}/{f}")
        x = f"{rd}/M2b_s{s}_recft_{arm}"
        os.makedirs(x, exist_ok=True)
        summ = json.load(open(f"{real_rd}/M2b_s{s}_recft_R/summary.json", encoding="utf-8"))
        summ.update({"recover_p": spec["p"], "recover_n": int(spec["p"] * 100000), "recover_eligible": 100000})
        json.dump(summ, open(f"{x}/summary.json", "w"))
        json.dump({"argv": nx_tokens(f"{arm}_{s}")}, open(f"{x}/argv.json", "w"))
        json.dump(_gate(arm, s), open(f"{x}/gate_selfcheck.json", "w"))
        os.makedirs(f"{rd}/rnext_markers", exist_ok=True)
        json.dump({"evidence": {"ckpt_sha256": hashlib.sha256(f"{x}/ckpt_last.pt".encode()).hexdigest()}}, open(f"{rd}/rnext_markers/RUN_{arm}_{s}.done", "w"))
    shutil.copytree(f"{real_rd}/recft_markers", f"{rd}/recft_markers")
    pairs = [(f"KA1{arm}", "KA1R"), (f"FZ{arm}", "FZR"), (f"{arm}C", "RC"), (f"{arm}_s", "R_s")]            # arm tag -> the existing R rollout it stands in for (longest first)

    def redir0(p, prs):
        for a_, b_ in prs:
            p = p.replace(a_, b_)
        return p

    real_out, tmp_out = "C:/Users/Admin/rr_scratch/out", (tmp_path / "out").as_posix()
    os.makedirs(tmp_out)
    for r_ in R.run_table(arm):                                                     # the arm's own done markers (copies of the R rollouts' small json), so KREUSE reads a marker for every row
        if not r_["reused"]:
            src_tag = redir0(r_["tag"], pairs)
            shutil.copy(f"{real_out}/done_{src_tag}.txt", f"{tmp_out}/done_{r_['tag']}.txt")

    def redir(p):
        p = str(p).replace("\\", "/").replace(tmp_out, real_out)
        for a_, b_ in pairs:
            p = p.replace(a_, b_)
        return p
    for owner, name in ((R, "run_header"), (R.LR, "tickload"), (R.SR, "load_dump")):
        orig = getattr(owner, name)
        monkeypatch.setattr(owner, name, (lambda o: (lambda *a_, **k_: o(*[redir(v) if isinstance(v, str) else v for v in a_], **k_)))(orig))
    ns = argparse.Namespace(recover_dir=rd, r_arm=arm, out_dir=tmp_out, reuse_dir=real_out)
    C, _ = R.file_controls(ns, R.run_table(arm))
    assert "KCARD_recover_next_card_hashes_to_FROZEN_SHA" in C and C["KCARD_recover_next_card_hashes_to_FROZEN_SHA"]["PASS"]
    assert "KREUSE_every_rollout_done_marker_has_the_pulled_ckpt_sha_and_the_pinned_tool_sha" in C
    kr3 = "KR3_table_build_summary_exists_all_records_TRAIN" if spec["bank"] else "KR3_apriori_law_builds_no_table_and_reads_no_bank"
    assert kr3 in C and ("KR3_apriori_law_builds_no_table_and_reads_no_bank" in C) != spec["bank"]
    for name in ("KCARD_composite_card_hashes_to_FROZEN_SHA", "KCARD_recover_ft_card_hashes_to_FROZEN_SHA", "KP1_parity_train_and_window_list_sha_equal_DAGOACC", "KR1_selected_over_eligible_in_band_for_R_zero_for_C0", kr3):
        assert C[name]["PASS"], (name, C[name])
    assert all(C["KMARK_box2_marker_sha_equals_pulled_ckpt"][f"{arm}_{s}"]["PASS"] for s in (0, 1))                 # the arm's own markers (the C0 / C0rep ones carry the real shas, which the faked hash cannot match)
    kr2 = C["KR2_R_ckpt_differs_from_C0_and_C0_recipe_is_R_minus_the_lever"]
    assert all(kr2[f"s{s}"]["argv_audit"]["PASS"] and kr2[f"s{s}"]["R_role_run_differs_from_the_RECOVER_FT_R_run"] for s in (0, 1))
    assert set(C["KMARK_box2_marker_sha_equals_pulled_ckpt"]) == {f"{arm}_0", f"{arm}_1", "C0_0", "C0_1", "C0rep_0", "C0rep_1", "PASS"}
    assert re.match(r"^KFZ1", "KFZ1_production_receipt_every_item_FZR_FZC0") and set(C["KFZ1_production_receipt_every_item_FZR_FZC0"]) == {f"FZ{arm}_s0", "FZC0_s0", f"FZ{arm}_s1", "FZC0_s1", "PASS"}
    assert "KA1_bit_identity_rerun_of_R_s0" in C and set(C["KK1_RC_C0C_headers_carry_kbox_and_sg72"]) == {f"{arm}C_s0", "C0C_s0", f"{arm}C_s1", "C0C_s1", "PASS"}
