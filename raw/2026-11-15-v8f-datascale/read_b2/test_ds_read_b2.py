"""test_ds_read_b2.py -- unit tests of the DSCALE10K block-2 deciding read: every committed outcome of card section 6 reached by PLANTED per-seed arms (cells built by the card's own cell rule),
the NOT-YET / BLOCKED / NO-REACH precedences, the K-RE (B1) equality of reused seed-0 rows, the seed-1 argvs and item defaults.
Run:  python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/read2/bt/t test_ds_read_b2.py"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R  # noqa: E402
import ds_read_b2 as B2  # noqa: E402
import ds_run as RUN  # noqa: E402

CI = L.load_ci()
HEAD_CELLS = [c for f in R.FAMILIES for c in R.HEADLINE[f]]
HIGHER = set(R.ACCURACY) | set(R.STRAT_HIGHER)


def planted_rows(gain=(), cost=(), seed=1):
    """U - B rows of one seed for the eight headline cells, built by the card's cell rule (cell_delta, a floor, the paired episode-cluster bootstrap) from synthetic windows: the families in `gain` get a
    separated improvement above the floor on BOTH of their headline cells, the families in `cost` a separated worsening, the rest nothing."""
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(40), 15)
    rows = {}
    for f in R.FAMILIES:
        for c in R.HEADLINE[f]:
            sign = (1.0 if f in gain else 0.0) - (1.0 if f in cost else 0.0)       # +1 better, -1 worse
            b = rng.normal(1.0, 0.3, 600)
            shift = sign * 0.2 * (1.0 if c in HIGHER else -1.0)
            a = b + shift + rng.normal(0, 0.02, 600)
            rows[c] = R.OR.cell_delta(CI, a, b, eid, c in HIGHER, 0.05, 300)
    return rows


def fam_of(**kw):
    return B2.seed_families(planted_rows(**kw))


def test_planted_rows_follow_the_cell_rule():
    r = planted_rows(gain=["LATERAL"], cost=["TACTICAL"])
    assert all(r[c]["gain"] and not r[c]["cost"] for c in R.HEADLINE["LATERAL"])
    assert all(r[c]["cost"] and not r[c]["gain"] for c in R.HEADLINE["TACTICAL"])         # higher-is-better cells: a drop in accuracy is a cost
    assert all(not r[c]["gain"] and not r[c]["cost"] for c in R.HEADLINE["STRATEGIC"] + R.HEADLINE["LONGITUDINAL"])
    s = fam_of(gain=["STRATEGIC"])
    assert s["STRATEGIC"]["status"] == "gain" and s["LATERAL"]["status"] == "none"


@pytest.mark.parametrize("g0,c0,g1,c1,want", [
    (["LATERAL", "TACTICAL"], [], ["LATERAL", "TACTICAL", "STRATEGIC"], [], "DATA-HELPS"),
    (["LATERAL"], [], ["LATERAL"], [], "DATA-HELPS-PARTIAL"),
    ([], [], [], [], "DATA-NEUTRAL"),
    ([], ["LATERAL"], [], ["LATERAL"], "DATA-HURTS"),
    (["TACTICAL"], ["LATERAL"], ["TACTICAL"], ["LATERAL"], "MIXED"),
    (["LATERAL"], [], [], [], "SEEDS-DISAGREE"),                                         # a gain at one seed only
    ([], [], [], ["LONGITUDINAL"], "SEEDS-DISAGREE"),                                    # a cost at ONE seed only (the seed-1 cost must count)
    ([], ["LATERAL"], [], ["TACTICAL"], "SEEDS-DISAGREE"),                               # costs in different families
    (["LATERAL", "TACTICAL"], [], ["LATERAL"], ["STRATEGIC"], "SEEDS-DISAGREE"),         # a seed-1 cost blocks DATA-HELPS
])
def test_every_committed_outcome_from_planted_arms(g0, c0, g1, c1, want):
    assert B2.outcome_b2({0: fam_of(gain=g0, cost=c0, seed=2), 1: fam_of(gain=g1, cost=c1, seed=3)}, True) == [want]


def test_precedence_blocked_reach_notyet():
    ok = {0: fam_of(gain=["LATERAL", "TACTICAL"]), 1: fam_of(gain=["LATERAL", "TACTICAL"])}
    assert B2.outcome_b2(ok, True) == ["DATA-HELPS"]
    assert B2.outcome_b2(ok, False) == ["NO-REACH"]                                      # NO-REACH precedes the labels
    assert B2.outcome_b2(ok, True, blocked=["B_s1.B"]) == ["BLOCKED"]                    # a refused arm blocks everything
    assert B2.outcome_b2(ok, False, blocked=["B_s1.B"]) == ["BLOCKED"]
    assert B2.outcome_b2(ok, True, reach_read=False) == ["NOT-YET"]                      # the reach precondition is unread
    assert B2.outcome_b2({0: ok[0]}, True) == ["NOT-YET"]                                # one seed: no label


def test_an_unread_family_blocks_the_label_it_is_not_no_gain():
    s1 = fam_of(gain=["LATERAL", "TACTICAL"])
    s1["STRATEGIC"] = {"status": "NOT-YET", "gains": [], "costs": [], "missing": list(R.HEADLINE["STRATEGIC"])}
    assert B2.outcome_b2({0: fam_of(gain=["LATERAL", "TACTICAL"]), 1: s1}, True) == ["NOT-YET"]
    assert R.outcome_label({0: fam_of(gain=["LATERAL", "TACTICAL"]), 1: s1}, True) == ["DATA-HELPS"]     # the block-1 rule alone would have labelled it: the guard is the difference


def test_a_family_is_read_from_its_own_headline_cells_never_pooled():
    rows = planted_rows(gain=["LATERAL"], cost=["LONGITUDINAL"])
    s = B2.seed_families(rows)
    assert s["LATERAL"]["status"] == "gain" and s["LONGITUDINAL"]["status"] == "cost"    # a gain in one family and a cost in another stay separate
    assert B2.outcome_b2({0: s, 1: s}, True) == ["MIXED"]


# ---------------------------------------------------------------------------------------------------------------------------------------- K-RE (B1): reused seed-0 rows equal the recomputed ones exactly
def rows_with_nan():
    r = planted_rows(gain=["LATERAL"])
    r = R.slim(r)
    r["operative_0_2s:speed_mae_mps"] = dict(r["operative_0_2s:speed_mae_mps"], floor=float("nan"))
    return r


def test_kre_b1_exact_equality_and_nan():
    rec = rows_with_nan()
    banked = json.loads(json.dumps(rec, default=float))                                    # a banked JSON round trip (NaN survives as NaN)
    k = B2.kre_b1(rec, banked)
    assert k["pass"] and k["rows_compared"] > 0 and not k["mismatches"]


def test_kre_b1_detects_a_changed_field_a_missing_cell_and_nothing_compared():
    rec = R.slim(planted_rows(gain=["LATERAL"]))
    t = json.loads(json.dumps(rec, default=float))
    c = next(iter(t))
    t[c]["hi"] = t[c]["hi"] + 1e-9
    k = B2.kre_b1(rec, t)
    assert not k["pass"] and k["n_mismatches"] >= 1
    t2 = json.loads(json.dumps(rec, default=float))
    t2["a_cell_only_the_bank_has"] = dict(t2[c])
    k2 = B2.kre_b1(rec, t2)
    assert not k2["pass"] and k2["cells_banked_not_recomputed"] == ["a_cell_only_the_bank_has"]
    assert not B2.kre_b1(rec, {})["pass"]                                                  # zero rows compared is a failure


def fake_mech(r, delta):
    return {"r": r, "rmse_slot": {"A": 1.0 + r, "B": 1.0}, "mse_delta": {"point": delta, "lo": delta - 0.01, "hi": delta + 0.01, "separated": True}}


def test_kre_mech():
    a = {"U_vs_B_stage_A": fake_mech(-0.06, -0.04), "R_vs_U_stage_A": fake_mech(0.08, 0.05), "stage_A_T_floor": {"floor_mse": 0.01}}
    b = json.loads(json.dumps(a))
    assert B2.kre_mech(a, b)["pass"]
    b["R_vs_U_stage_A"]["rmse_slot"]["A"] += 1e-9
    assert not B2.kre_mech(a, b)["pass"]


# ---------------------------------------------------------------------------------------------------------------------------------------- seed-1 items, argvs, defaults, bank
def test_block2_item_set_and_not_yet_arms():
    assert set(L.B2_ITEMS) == {("B_s1", "A"), ("B_s1", "B"), ("U_s1", "A"), ("U_s1", "B")}
    assert len(L.READ_ITEMS) == 7 and all(L.ARMS[a] == 1 for a, _ in L.B2_ITEMS) and ("B_s1", "A") not in L.READ_ITEMS
    assert set(L.NOT_YET_B2) == {"UE_s0", "BL_s0", "X_s0", "B_s3", "U_s3", "B_s4", "U_s4"} and not set(L.NOT_YET_B2) & {"B_s1", "U_s1"}


def test_default_items_follow_ds_block():
    code = "import ds_lib as L, ds_run as R; print(sorted(R.parse_items([]))); print(L.DEFAULT_ITEMS == L.B2_ITEMS)"
    out = {}
    for blk in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=dict(os.environ, DS_BLOCK=blk, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-300:]
        out[blk] = r.stdout.splitlines()
    assert out["2"][1] == "True" and out["1"][1] == "False"
    assert out["2"][0] == str(sorted(L.B2_ITEMS)) and out["1"][0] == str(sorted(L.READ_ITEMS))


def test_seed1_argvs():
    a, od, tag = RUN.t1_argv("U_s1", "B")
    assert tag == "DS-U_s1-B_m10_causal_heads" and od == L.T1 and od.replace("\\", "/").startswith(HERE.replace("\\", "/")) and "--max-windows" not in a
    assert "U_s1.A".split(".")[0] in L.ARMS
    j = dict(RUN.mech_argvs("B_s1", "A", "OUT"))
    assert list(j)[0] == "DSBs1A_s0" and len(j) == 4
    assert RUN.route_argv("U_s1", "B", "donor", "o")[0] == "DS-U_s1-B_donor"


def test_preflight_names_every_unbanked_seed1_item(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "CK", str(tmp_path))
    monkeypatch.setattr(RUN.L, "disk_problem", lambda *a, **k: None)
    monkeypatch.setattr(RUN.L, "tree_problem", lambda *a, **k: None)
    monkeypatch.setattr(RUN, "import_check", lambda: None)
    assert set(RUN.preflight(list(L.B2_ITEMS))) == {"B_s1.A", "B_s1.B", "U_s1.A", "U_s1.B"}


def test_bank_problem_accepts_a_seed1_record(tmp_path):
    d = tmp_path / "B_s1.B"
    d.mkdir()
    (d / "ckpt_last.pt").write_bytes(b"x")
    (d / "BANK_RECORD.json").write_text(json.dumps({"arm": "B_s1", "stage": "B", "sha256": L.sha256_file(str(d / "ckpt_last.pt")), "source": "hf", "commit": L.COMMIT, "steps": 3000, "seed": 1}))
    assert L.bank_problem("B_s1", "B", str(tmp_path)) is None
    assert "steps" in L.bank_problem("B_s1", "A", str(tmp_path)) or "missing" in L.bank_problem("B_s1", "A", str(tmp_path))


def test_missing_dumps_write_not_yet(tmp_path):
    out = tmp_path / "r.json"
    rc = B2.main(["--dumps0", str(tmp_path / "n0"), "--dumps1", str(tmp_path / "n1"), "--out", str(out)])
    r = json.load(open(out))
    assert rc == 1 and r["outcome"] == ["NOT-YET"] and "T1 dumps missing" in r["reason"] and "UE_s0" in r["not_yet"] and "B_s1" not in r["not_yet"]


# ---------------------------------------------------------------------------------------------------------------------------------------- the launcher's wait on the block-1 sentinel
def run_wait(tmp_path, b1_text):
    sh = open(HERE + "/launch_dscale10k_b2.sh", encoding="utf-8").read()
    body = sh.split("# WAIT-BEGIN")[1].split("# WAIT-END")[0]
    log = tmp_path / "b1.log"
    log.write_text(b1_text, encoding="utf-8")
    out = tmp_path / "b2.log"
    script = tmp_path / "w.sh"
    script.write_text('fail() { echo "FAILED: $1" >> "$LOG"; exit 1; }\nLOG=' + str(out).replace("\\", "/") + '\nB1LOG=' + str(log).replace("\\", "/") + "\n" + body + '\necho PROCEEDED >> "$LOG"\n', encoding="utf-8", newline="\n")
    bash = os.environ.get("GIT_BASH", "C:/Program Files/Git/usr/bin/bash.exe")
    r = subprocess.run([bash, str(script).replace("\\", "/")], env=dict(os.environ, DS_POLL_S="0", DS_POLL_N="2", PATH="/usr/bin:/bin:" + os.environ["PATH"]), capture_output=True, encoding="utf-8")
    return out.read_text(encoding="utf-8") if out.exists() else "", r.returncode


def test_launcher_waits_for_the_b1_done_sentinel(tmp_path):
    old_fail = "[dscale10k b1 read] start t1\nZZ DSCALE10K B1 READ FAIL ZZ first attempt\n"
    (tmp_path / "a").mkdir()
    text, rc = run_wait(tmp_path / "a", old_fail + "[dscale10k b1 read] start t2\nstuff\nZZ DSCALE10K B1 READ DONE ZZ\n")
    assert "PROCEEDED" in text and "FAILED" not in text                                    # the earlier FAIL is before the latest start: ignored
    (tmp_path / "b").mkdir()
    text, rc = run_wait(tmp_path / "b", old_fail + "[dscale10k b1 read] start t2\nrunning\n")
    assert "FAILED: the block-1 read did not finish" in text and "PROCEEDED" not in text    # still running: never proceeds
    (tmp_path / "c").mkdir()
    text, rc = run_wait(tmp_path / "c", "[dscale10k b1 read] start t1\nZZ DSCALE10K B1 READ FAIL ZZ boom\n")
    assert "FAILED: the block-1 read ended in FAIL" in text and "PROCEEDED" not in text


def test_launcher_keeps_the_bar_and_the_sentinels():
    sh = open(HERE + "/launch_dscale10k_b2.sh", encoding="utf-8").read()
    assert '[ "$FREE" -lt 25 ]' in sh and "ZZ DSCALE10K B2 READ DONE ZZ" in sh and "ZZ DSCALE10K B2 READ FAIL ZZ" in sh and "DS_BLOCK=2" in sh
    assert 'ds_read_b2.py --out "$R/out/RESULT_DSCALE10K_B2.json"' in sh and "wait_commit.py 5.5" in sh
    assert "CreateFlags = 0x08000200" in open(HERE + "/launch_wmi_b2.ps1", encoding="utf-8").read()
    assert "Git" + chr(92) + "usr" + chr(92) + "bin" + chr(92) + "bash.exe" in open(HERE + "/job_dscale10k_b2.cmd", encoding="utf-8").read()
