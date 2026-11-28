"""W-COST replicate exception (PI 2026-11-21 ~13:30Z, "Hold replicate to parent R").

K2DC02s1 (= K2DC02's argv with `--seed` 0 -> 1 and `--out` renamed) was refused by G-LIVE W-COST because its OWN smoke read R = 1.0904 while K2DC02's read
1.5450 (MEASURED, Thor r9e2/gate3/<arm>/evidence/G-LIVE.json). A DECLARED replicate (`launch_gate.WCOST_REPLICATES`) is held to the PARENT's R; any other arm
keeps the own-smoke rule unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
PARENT = json.loads((RUNS / "r9e2_K2DC02.argv.json").read_text(encoding="utf-8"))["argv"]
REP = json.loads((RUNS / "r9e2_K2DC02s1.argv.json").read_text(encoding="utf-8"))["argv"]
FMT = ("[v3] refcv9 W-COST at step 0: critic-BCE / trajectory-loss = %s  (critic BCE 0.6931, traj 0.2772, n_drv 120, --w-r8-drivable 0.06472; "
       "target weighted ratio ~0.1 => w = 0.1 / ratio)")


def judge(argv, R="1.0904", prof="refcv9-e2"):
    return LG.wcost_reasons(LG.PROFILES[prof], argv, [FMT % R])


def test_i_replicate_with_its_own_lower_R_passes_and_records_both_readings():
    r, d = judge(REP, "1.0904")
    assert r == [], r
    assert d["R_parent"] == 1.545 and d["R_own"] == 1.0904 and d["replicate_of"] == "r9e2_K2DC02.argv.json"
    assert abs(d["R_own_over_parent"] - 1.0904 / 1.545) < 1e-12
    assert d["w_want"] == "0.06472" and d["deviation"] < 1e-3


def test_ii_replicate_with_a_wrong_weight_FAILS():
    argv = LG.set_flag(REP, "--w-r8-drivable", ["0.0917"])
    r, d = judge(argv, "1.0904")
    assert len(r) == 1 and "not a replicate" in r[0] and "'0.0917'" in r[0], r      # w moved vs the parent: caught by the argv-equality rule


def test_ii_b_the_weight_rule_itself_binds_to_the_parent_R(monkeypatch, tmp_path):
    # isolate the weight rule: a parent that carries the SAME wrong weight, so argv equality holds and only w * R_parent / 0.1 can refuse
    pdoc = {"argv": LG.set_flag(PARENT, "--w-r8-drivable", ["0.0917"])}
    (tmp_path / "r9e2_K2DC02.argv.json").write_text(json.dumps(pdoc), encoding="utf-8")
    monkeypatch.setattr(LG, "WCOST_RUNS_DIR", tmp_path)
    r, d = judge(LG.set_flag(REP, "--w-r8-drivable", ["0.0917"]), "1.0904")
    assert len(r) == 1 and "PARENT" in r[0] and "R = 1.545" in r[0] and "0.06472" in r[0], r


def test_iii_replicate_with_another_lever_moved_FAILS_naming_the_difference():
    argv = LG.set_flag(REP, "--w-map-hires", ["24.0"])
    r, d = judge(argv, "1.0904")
    assert len(r) == 1 and "not a replicate" in r[0] and "'24.0'" in r[0] and "'25.0'" in r[0], r
    assert d["replicate_diff"]


def test_iii_b_replicate_with_a_changed_length_FAILS():
    r, _ = judge(REP + ["--extra"], "1.0904")
    assert len(r) == 1 and "length" in r[0], r


def test_iv_the_parent_itself_is_unchanged_own_smoke_rule():
    r, d = judge(PARENT, "1.0904")                      # parent with the replicate's R: refused exactly as before
    assert len(r) == 1 and "the smoke's OWN W-COST ratio is R = 1.09" in r[0], r
    assert "R_parent" not in d and "replicate_of" not in d
    r2, d2 = judge(PARENT, "1.5450")
    assert r2 == [] and "R_parent" not in d2


def test_v_missing_parent_json_is_a_FAIL_never_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(LG, "WCOST_RUNS_DIR", tmp_path)
    r, d = judge(REP, "1.0904")                         # its own R would otherwise be refused too; the reason must be the parent's absence
    assert len(r) == 1 and "missing or unreadable" in r[0], r
    r2, _ = judge(REP, "1.5450")                        # even with an own R that would pass, no fallback
    assert len(r2) == 1 and "missing or unreadable" in r2[0], r2


def test_vi_pins_parent_R_weight_and_out_key():
    ent = LG.WCOST_REPLICATES["r9e2-K2DC02s1"]
    assert LG.wcost_weight_text(ent["parent_R"]) == LG.flag_values(PARENT, "--w-r8-drivable")[0] == LG.flag_values(REP, "--w-r8-drivable")[0]
    assert LG.wcost_out_name(REP) == "r9e2-K2DC02s1" and LG.wcost_out_name(PARENT) == "r9e2-K2DC02"
    assert (RUNS / ent["parent_argv"]).is_file()
    assert LG.wcost_replicate_diff(REP, PARENT) == []
    assert ent["ruling"].startswith("PI 2026-11-21")


@pytest.mark.parametrize("bad", [[], ["--seed", "1"]])
def test_replicate_diff_requires_both_flags_once_at_the_same_position(bad):
    assert LG.wcost_replicate_diff(bad, bad)
