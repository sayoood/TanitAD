"""test_ds_read_b3.py -- unit tests of the DSCALE10K block-3 read (H-DATA-SCALE-1 sub-read): the label rule of card lines 195-197 reached by PLANTED ADE6 deltas (strict 2F_A bar, both signs, the floor guard,
the two readings of "U - BL"), the precedences, the block-3 item / step / checkpoint-root constants, the stage-A route jobs, K-EVALX at stage A, K-RE of the ADE6 means, sync_b1, the launcher's wait and bars.
Run:  env -u DS_BLOCK python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/read3/bt/t test_ds_read_b3.py"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read_b3 as B3  # noqa: E402
import ds_t0 as T  # noqa: E402
import sync_b1 as SY  # noqa: E402

PYX = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def in_block3(code):
    """run `code` in a fresh interpreter with DS_BLOCK=3 (the lib reads it at import) -> stdout"""
    r = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); " + code], env=dict(os.environ, DS_BLOCK="3", CUDA_VISIBLE_DEVICES="-1"), capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout.strip()


# ---------------------------------------------------------------------------------------------------------------------------------------- the label rule (card 195-197)
F_A = 0.05      # 2F_A = 0.10 m


@pytest.mark.parametrize("delta,want", [
    (-0.1001, "DATA > EPOCHS"),            # first (U) better (lower ADE) by more than 2F_A
    (-0.2, "DATA > EPOCHS"),
    (0.1001, "EPOCHS > DATA"),             # the reverse
    (0.3, "EPOCHS > DATA"),
    (-0.10, "NEITHER"),                    # exactly 2F_A is not MORE than 2F_A (strict)
    (0.10, "NEITHER"),
    (0.0, "NEITHER"),
    (-0.0999, "NEITHER"),
])
def test_verdict_strict_two_f_a_and_sign(delta, want):
    assert B3.verdict(delta, F_A) == want


@pytest.mark.parametrize("f_a", [None, 0.0, -0.01, float("nan"), float("inf")])
def test_a_floor_that_is_not_positive_and_finite_is_not_a_floor(f_a):
    assert B3.verdict(-5.0, f_a) == "NOT-YET" and B3.data_scale_label(-5.0, -5.0, f_a)["label"] == "NOT-YET"


def test_a_nan_delta_is_not_yet_never_neither():
    assert B3.verdict(float("nan"), F_A) == "NOT-YET" and B3.verdict(None, F_A) == "NOT-YET"
    assert B3.data_scale_label(float("nan"), -0.5, F_A)["label"] == "NOT-YET"


def test_both_readings_agree_gives_the_label_and_the_literal_reading_is_kept():
    for d, want in ((-0.3, "DATA > EPOCHS"), (0.3, "EPOCHS > DATA"), (0.02, "NEITHER")):
        r = B3.data_scale_label(d, d, F_A)
        assert r["label"] == want and r["label_literal"] == want and not r["reading_sensitive"] and r["two_F_A"] == pytest.approx(0.10)


def test_disagreeing_readings_are_reading_sensitive_not_a_single_label():
    r = B3.data_scale_label(-0.3, 0.02, F_A)          # U - BL says DATA > EPOCHS, UE - BL says NEITHER
    assert r["label"] == "READING-SENSITIVE" and r["reading_sensitive"] and r["label_literal"] == "DATA > EPOCHS" and r["label_ue_bl"] == "NEITHER"
    assert r["labels_by_reading"] == {"U_s0 - BL_s0 (literal)": "DATA > EPOCHS", "UE_s0 - BL_s0": "NEITHER"}
    r = B3.data_scale_label(0.3, -0.3, F_A)
    assert r["label"] == "READING-SENSITIVE"


def test_ue_minus_bl_flag_for_the_data_neutral_next_step():
    assert B3.data_scale_label(0.0, -0.5, F_A)["ue_minus_bl_better_than_2FA"] is True
    assert B3.data_scale_label(0.0, 0.5, F_A)["ue_minus_bl_better_than_2FA"] is False


def test_outcome_precedence_blocked_notyet_label_controls():
    lab = B3.data_scale_label(-0.3, -0.3, F_A)
    assert B3.outcome_b3(lab, ["UE_s0.A"], True, []) == ["BLOCKED"]
    assert B3.outcome_b3(lab, ["UE_s0.A"], False, ["x"]) == ["BLOCKED"]
    assert B3.outcome_b3(None, None, False, []) == ["NOT-YET"] and B3.outcome_b3(lab, None, False, []) == ["NOT-YET"]
    assert B3.outcome_b3(lab, None, True, []) == ["DATA > EPOCHS"]
    assert B3.outcome_b3(lab, None, True, ["K-RE"])[0].startswith("DATA > EPOCHS (CONTROL FAILED") and "not quotable" in B3.outcome_b3(lab, None, True, ["K-RE"])[0]


def test_pair_set_is_the_cards_three_quantities_plus_the_literal_reading():
    assert B3.PAIRS["U-B_s0.A"] == ("U_s0", "B_s0") and B3.PAIRS["BL-B_s0.A"] == ("BL_s0", "B_s0") and B3.PAIRS["UE-BL.A"] == ("UE_s0", "BL_s0") and B3.PAIRS["U-BL.A"] == ("U_s0", "BL_s0")
    assert B3.LABEL_PAIRS == ("U-BL.A", "UE-BL.A")


def test_kre_ade6_equal_and_differing():
    ban = {"ade6": {"A": 2.2222, "B": 2.1111}}
    assert B3.kre_ade6({"U_s0": 2.2222, "B_s0": 2.1111}, ban)["pass"]
    bad = B3.kre_ade6({"U_s0": 2.2223, "B_s0": 2.1111}, ban)
    assert not bad["pass"] and bad["mismatches"]


# ---------------------------------------------------------------------------------------------------------------------------------------- items, steps, checkpoint roots
def test_block3_constants_and_arm_names():
    assert L.B3_ITEMS == (("UE_s0", "A"), ("BL_s0", "A")) and L.STEPS_ARM == {"UE_s0": 61568, "BL_s0": 61568} and 4 * 15392 == 61568
    assert L.steps_of("UE_s0", "A") == 61568 and L.steps_of("BL_s0", "A") == 61568 and L.steps_of("B_s0", "A") == 19164 and L.steps_of("B_s0", "B") == 3000
    assert L.STAGES_OF["UE_s0"] == ("A",) and L.STAGES_OF["BL_s0"] == ("A",) and L.ARMS["UE_s0"] == 0 and L.ARMS["BL_s0"] == 0
    assert T.name_of("UE_s0") == "DSUEs0A" and T.name_of("BL_s0") == "DSBLs0A" and len({T.name_of(a) for a in B3.ARMS_A}) == 5
    assert L.READ_ITEMS == tuple((a, s) for a in ("B_s0", "U_s0", "B_rep", "R_s0") for s in L.STAGES_OF[a]) and len(L.READ_ITEMS) == 7        # the block-1 item set is unchanged


def test_block3_defaults_and_checkpoint_roots():
    out = in_block3("import ds_lib as L; print(L.DEFAULT_ITEMS); print(L.ck_root('UE_s0')); print(L.ck_root('B_s0')); print(L.ck_root('B_rep')); print(L.ck_root('U_s0')); print(L.ROUTE_ITEMS_B3)").splitlines()
    assert out[0] == "(('UE_s0', 'A'), ('BL_s0', 'A'))"
    assert out[1].replace("\\", "/") == os.path.normpath(HERE + "/ckpts").replace("\\", "/") and out[2].replace("\\", "/") == L.B1_ROOT + "/ckpts" and out[3].replace("\\", "/") == L.B1_ROOT + "/ckpts" and out[4].replace("\\", "/") == L.B1_ROOT + "/ckpts"
    assert out[5] == "(('B_s0', 'A'), ('U_s0', 'A'), ('B_rep', 'A'), ('UE_s0', 'A'), ('BL_s0', 'A'))"
    out = in_block3("import ds_lib as L; print(L.DEFAULT_ITEMS)")
    assert "UE_s0" in out
    other = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_lib as L; print(len(L.DEFAULT_ITEMS)); print(L.ck_root('B_s0'))"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"},
                           capture_output=True, encoding="utf-8").stdout.splitlines()
    assert other[0] == "7" and other[1].replace("\\", "/") == os.path.normpath(HERE + "/ckpts").replace("\\", "/")          # block 1 default: seven items, own folder


def _fake_ck(tmp_path, arm, steps, sha=None):
    d = tmp_path / L.item(arm, "A")
    d.mkdir(parents=True)
    (d / "ckpt_last.pt").write_bytes(b"ckpt-bytes-" + arm.encode())
    sha = sha or L.sha256_file(str(d / "ckpt_last.pt"))
    (d / "BANK_RECORD.json").write_text(json.dumps({"arm": arm, "stage": "A", "sha256": sha, "source": "hf", "commit": L.COMMIT, "steps": steps, "seed": 0}), encoding="utf-8")
    return str(tmp_path)


def test_bank_problem_wants_61568_steps_for_the_block3_arms_and_19164_for_the_others(tmp_path):
    for arm in ("UE_s0", "BL_s0"):
        assert L.bank_problem(arm, "A", _fake_ck(tmp_path / arm / "ok", arm, 61568)) is None
        why = L.bank_problem(arm, "A", _fake_ck(tmp_path / arm / "short", arm, 19164))
        assert why and "61568" in why and "19164" in why                                   # a block-1 length is refused for a block-3 arm
        assert L.bank_problem(arm, "A", _fake_ck(tmp_path / arm / "bad", arm, 61568, sha="0" * 64)) and "sha256" in L.bank_problem(arm, "A", str(tmp_path / arm / "bad"))
    assert L.bank_problem("B_s0", "A", _fake_ck(tmp_path / "b", "B_s0", 19164)) is None
    assert L.bank_problem("B_s0", "A", _fake_ck(tmp_path / "b61", "B_s0", 61568)) is not None                 # and a block-3 length is refused for a block-1 arm


# ---------------------------------------------------------------------------------------------------------------------------------------- stage-A route jobs and K-EVALX
def test_route_jobs_are_stage_a_for_five_arms_and_five_variants():
    out = in_block3("import ds_lib as L, ds_run as D; j = D.route_jobs(L.ROUTE_ITEMS_B3, 'o'); print(len(j)); print(sorted({x[0] for x in j})[:3]); print(D.ROUTE_STAGE); "
                    "z = [x for x in j if x[0] == 'DS-UE_s0-A_zero'][0]; print(z[2] == L.TREE_ZERO); r = [x for x in j if x[0] == 'DS-BL_s0-A_real'][0]; print(r[2] == L.TREE); "
                    "print('--nav-withheld' in z[1], '--nav-shuffle' in [x for x in j if x[0] == 'DS-BL_s0-A_donor'][0][1]); import re; print(any(re.search(r'-B_(real|donor|zero|realx|donorx)$', x[0]) for x in j)); "
                    "print([a for a in z[1] if 'ckpt' in str(a).lower() or a.endswith('ckpt_last.pt')][-1].replace(chr(92), '/').split('/')[-3:])").splitlines()
    assert out[0] == "25" and out[2] == "A" and out[3] == "True" and out[4] == "True" and out[5] == "True True" and out[6] == "False"
    assert "UE_s0.A" in out[7]
    blk1 = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_run as D; print(D.ROUTE_STAGE)"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"}, capture_output=True, encoding="utf-8").stdout.strip()
    assert blk1 == "B"                                                                                  # blocks 1 / 2 keep their stage-B route passes


def _route_dir(tmp_path, arms=B3.ARMS_A, variants=("real", "donor", "zero", "realx", "donorx")):
    for a in arms:
        for v in variants:
            (tmp_path / f"eval_DS-{a}-A_{v}.npz").write_bytes(b"x")
    return str(tmp_path)


def _kx(tmp_path, arms=B3.ARMS_A, ok=True, drop=None):
    items = {f"{a}.A {v}": {"identical": ok} for a in arms for v in ("real", "donor")}
    if drop:
        items.pop(drop)
    (tmp_path / "KEVALX.json").write_text(json.dumps({"pass": ok, "items": items}), encoding="utf-8")


def test_route_problem_needs_all_passes_and_a_kevalx_pass_over_every_stage_a_arm(tmp_path):
    (tmp_path / "a").mkdir()
    d = _route_dir(tmp_path / "a")
    assert "KEVALX.json missing" in B3.route_problem_b3(d)
    _kx(tmp_path / "a")
    assert B3.route_problem_b3(d) is None
    _kx(tmp_path / "a", drop="UE_s0.A donor")
    assert "not a PASS" in B3.route_problem_b3(d)                                                       # K-EVALX must cover the block-3 arms too
    _kx(tmp_path / "a", ok=False)
    assert "K-EVALX" in B3.route_problem_b3(d)
    (tmp_path / "b").mkdir()
    d2 = _route_dir(tmp_path / "b", variants=("real", "donor", "zero", "realx"))
    _kx(tmp_path / "b")
    assert "T0 route passes missing" in B3.route_problem_b3(d2) and "donorx" in B3.route_problem_b3(d2)
    (tmp_path / "c").mkdir()
    d3 = _route_dir(tmp_path / "c", arms=B3.ARMS_A[:4])
    _kx(tmp_path / "c")
    assert "BL_s0.A" in B3.route_problem_b3(d3)


def test_missing_dumps_write_not_yet(tmp_path):
    out = tmp_path / "r.json"
    rc = B3.main(["--dumps0", str(tmp_path / "x"), "--dumps3", str(tmp_path / "y"), "--out", str(out), "--b1-result", str(tmp_path / "nob1.json")])
    r = json.load(open(out))
    assert rc == 1 and r["outcome"] == ["NOT-YET"] and "T1 dumps missing" in r["reason"] and "UE_s0.A" in r["reason"] and r["block"] == 3


# ---------------------------------------------------------------------------------------------------------------------------------------- sync_b1
def test_sync_b1_copies_verifies_and_refuses_a_differing_target(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for f in SY.plan(str(src)):
        (src / f).write_bytes(f.encode())
    c, s, bad = SY.sync(str(src), str(dst))
    assert c == 32 and s == 0 and not bad
    c, s, bad = SY.sync(str(src), str(dst))
    assert c == 0 and s == 32 and not bad                                                              # idempotent
    f0 = SY.plan(str(src))[0]
    (dst / f0).write_bytes(b"different")
    c, s, bad = SY.sync(str(src), str(dst))
    assert bad and "REFUSED" in bad[0] and (dst / f0).read_bytes() == b"different"                      # never overwritten
    os.remove(src / SY.plan(str(src))[5])
    assert any("source missing" in b for b in SY.sync(str(src), str(tmp_path / "d2"))[2])


# ---------------------------------------------------------------------------------------------------------------------------------------- the launcher
def run_wait(tmp_path, b2_text):
    sh = open(HERE + "/launch_dscale10k_b3.sh", encoding="utf-8").read()
    body = sh.split("# WAIT-BEGIN")[1].split("# WAIT-END")[0]
    log = tmp_path / "b2.log"
    log.write_text(b2_text, encoding="utf-8")
    out = tmp_path / "b3.log"
    script = tmp_path / "w.sh"
    script.write_text('fail() { echo "FAILED: $1" >> "$LOG"; exit 1; }\nLOG=' + str(out).replace("\\", "/") + '\nB2LOG=' + str(log).replace("\\", "/") + "\n" + body + '\necho PROCEEDED >> "$LOG"\n', encoding="utf-8", newline="\n")
    bash = os.environ.get("GIT_BASH", "C:/Program Files/Git/usr/bin/bash.exe")
    r = subprocess.run([bash, str(script).replace("\\", "/")], env=dict(os.environ, DS_POLL_S="0", DS_POLL_N="2", PATH="/usr/bin:/bin:" + os.environ["PATH"]), capture_output=True, encoding="utf-8")
    return out.read_text(encoding="utf-8") if out.exists() else "", r.returncode


def test_launcher_waits_for_the_b2_done_sentinel(tmp_path):
    old = "[dscale10k b2 read] start t1\nZZ DSCALE10K B2 READ FAIL ZZ first attempt\n"
    (tmp_path / "a").mkdir()
    text, rc = run_wait(tmp_path / "a", old + "[dscale10k b2 read] start t2\nstuff\nZZ DSCALE10K B2 READ DONE ZZ\n")
    assert "PROCEEDED" in text and "FAILED" not in text                                                # the earlier FAIL is before the latest start: ignored
    (tmp_path / "b").mkdir()
    text, rc = run_wait(tmp_path / "b", old + "[dscale10k b2 read] start t2\nrunning\n")
    assert "FAILED: the block-2 read did not finish" in text and "PROCEEDED" not in text
    (tmp_path / "c").mkdir()
    text, rc = run_wait(tmp_path / "c", "[dscale10k b2 read] start t1\nZZ DSCALE10K B2 READ FAIL ZZ boom\n")
    assert "FAILED: the block-2 read ended in FAIL" in text and "PROCEEDED" not in text
    (tmp_path / "d").mkdir()
    text, rc = run_wait(tmp_path / "d", "[dscale10k b2 read] start t1\nZZ DSCALE10K B2 READ DONE ZZ\n[dscale10k b2 read] start t2\nre-running\n")
    assert "PROCEEDED" not in text                                                                      # an older DONE is never read while a re-run is in progress


def test_launcher_keeps_the_bars_the_order_and_the_sentinels():
    sh = open(HERE + "/launch_dscale10k_b3.sh", encoding="utf-8").read()
    assert '[ "$FREE" -lt 25 ]' in sh and "ZZ DSCALE10K B3 READ DONE ZZ" in sh and "ZZ DSCALE10K B3 READ FAIL ZZ" in sh and "DS_BLOCK=3" in sh and "wait_commit.py 5.5" in sh
    assert 'ds_read_b3.py --out "$R/out/RESULT_DSCALE10K_B3.json"' in sh
    order = [sh.index(x) for x in ("pytest", "ds_pull.py >>", "ds_pull.py --check-only", "sync_b1.py >>", "ds_run.py dumps", "ds_run.py mech", "ds_run.py route", "ds_read_b3.py --out")]
    assert order == sorted(order)
    assert "ds_pull.py --check-only --items B_s0.A U_s0.A B_rep.A UE_s0.A BL_s0.A" in sh                      # the three block-1 reference checkpoints and the two new ones are re-checked
    assert "env -u DS_BLOCK" in sh                                                                       # a pytest run must not inherit the block env var
    assert "CreateFlags = 0x08000200" in open(HERE + "/launch_wmi_b3.ps1", encoding="utf-8").read()
    assert "Git" + chr(92) + "usr" + chr(92) + "bin" + chr(92) + "bash.exe" in open(HERE + "/job_dscale10k_b3.cmd", encoding="utf-8").read()
    assert "RESULT_DSCALE10K_B1.json" in sh and "RESULT_DSCALE10K_B2.json" in sh and "KEVALX.json" in sh


# ---------------------------------------------------------------------------------------------------------------------------------------- the four families per pair (never pooled)
def _row(gain=False, cost=False):
    return {"status": "OK", "gain": gain, "cost": cost, "delta": 0.1, "n_windows": 5}


def test_pair_family_status_reads_each_family_from_its_own_headline_cells():
    rows = {c: _row() for f in B3.FAMILIES for c in B3.HEADLINE[f]}
    rows[B3.HEADLINE["LATERAL"][0]] = _row(gain=True)
    rows[B3.HEADLINE["TACTICAL"][1]] = _row(cost=True)
    st = B3.pair_family_status(rows)
    assert st["LATERAL"]["status"] == "gain" and st["TACTICAL"]["status"] == "cost" and st["LONGITUDINAL"]["status"] == "none" and st["STRATEGIC"]["status"] == "none"
    del rows[B3.HEADLINE["STRATEGIC"][0]], rows[B3.HEADLINE["STRATEGIC"][1]]
    assert B3.pair_family_status(rows)["STRATEGIC"]["status"] == "NOT-YET"                                # an unread family is NOT-YET, never "no gain"


def test_fam_rows_all_adds_the_strategic_cells_only_where_the_strategic_block_has_the_pair():
    pairs = {"U-B_s0.A": {"a": _row()}, "B_rep-B_s0.A": {"a": _row()}}
    strat = {"status": "OK", "pairs": {"U-B_s0.A": {"S1:pos_err_m@80": _row(gain=True), "RC-A:x": _row()}}}
    out = B3.fam_rows_all(pairs, strat)
    assert "S1:pos_err_m@80" in out["U-B_s0.A"] and "RC-A:x" in out["U-B_s0.A"] and set(out["B_rep-B_s0.A"]) == {"a"}
    assert set(B3.fam_rows_all(pairs, {"status": "NOT-YET"})["U-B_s0.A"]) == {"a"}


def test_parse_items_defaults_per_command_in_block3():
    out = in_block3("import ds_run as D; print(len(D.parse_items([]))); print(len(D.parse_items([], route=True))); print(D.parse_items(['UE_s0.A']))").splitlines()
    assert out[0] == "2" and out[1] == "5" and out[2] == "[('UE_s0', 'A')]"
