"""Deliberate-regression arm of town_result.py: a perturbed control must make the outcome INVALID; the outcome logic must give the card's labels on known answers.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q -p no:cacheprovider tests/test_town_result.py   (reads town/raw only; CPU)"""
import os
import re
import sys
import copy
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import pytest

PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
sys.path.insert(0, PK)
import town_result as TR  # noqa: E402


@pytest.fixture(scope="module")
def I():
    return TR.load_inputs()


def label(J):
    return TR.evaluate(J)["outcome"]


def test_real_inputs_hold_every_control_and_read_the_card_label(I):
    r = TR.evaluate(I)
    assert r["controls"]["all_pass"] and r["outcome"] in TR.OUTCOMES and r["outcome_detail"]["logic_applied"]


def _set(path, value):
    def f(J):
        d = J
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return f


def _slope(arm, seed, **kw):
    def f(J):
        J["read"][arm][seed]["shift_committed"]["slope"].update(kw)
    return f


PERTURB = {
    "C1 one tiny delta on the executed tracks": _set(["gate_id", "s0", "max_abs_delta_tracks"], 1e-12),
    "C1 identity gate not PASS": _set(["gate_id", "s1", "PASS"], False),
    "C1 reference is the banked 12-thread run": _set(["gate_id", "s1", "against"], "RIC0_s1"),
    "C1 one episode dump unequal": _set(["gate_id", "s0", "episode_files_unequal"], ["ep0007.npz"]),
    "C2 tick-0 swapped inputs differ": _set(["gate_twin", "C0", "s0", "G-TWIN0", "swapped_inputs_bitwise_equal"], False),
    "C2 d at tick 0 not exactly 0.0": _set(["read", "REG", "s1", "shift_committed", "d_mean_tick0_max_abs"], 1e-9),
    "C2b twin swapped nothing": _set(["gate_twin", "C0", "s1", "G-TWINL", "later_swap_unequal"], 0),
    "C3a T zeroed on one call too few": _set(["receipt", "REG", "s0", "run_t_zeroed_calls"], 449),
    "C3a C0 run was zeroed": _set(["receipt", "C0", "s1", "zero_t_ego"], True),
    "C3a e_ego weights not restored": _set(["receipt", "REG", "s1", "e_ego_weight_sha256_after"], "0" * 64),
    "C3b regression slope survives (separated above 0)": _slope("REG", "s1", lo=0.002, hi=0.02, slope=0.01),
    "C3b regression slope survives (separated below 0)": _slope("REG", "s0", lo=-0.02, hi=-0.002, slope=-0.01),
    "frozen card prefix changed": _set(["card_prefix_sha256"], "0" * 64),
    "implementation choices changed": _set(["impl_sha256"], "0" * 64),
    "population changed (rows)": _set(["read", "C0", "s0", "n_rows"], 7196),
}


@pytest.mark.parametrize("name", sorted(PERTURB))
def test_perturbed_control_makes_the_outcome_INVALID(I, name):
    J = copy.deepcopy(I)
    PERTURB[name](J)
    r = TR.evaluate(J)
    assert r["outcome"] == "INVALID" and r["outcome_detail"]["logic_applied"] is False and r["outcome_detail"]["failed_controls"]
    assert "INVALID" in TR.render_md(r)


def _make(J, slope0, slope1, f1_0, f1_1):
    for s, sl, f1 in (("s0", slope0, f1_0), ("s1", slope1, f1_1)):
        J["read"]["C0"][s]["shift_committed"]["slope"].update(dict(zip(("slope", "lo", "hi"), sl)))
        J["read"]["C0"][s]["quality"]["all_replans"]["lon_macro_f1"].update(dict(zip(("delta", "lo", "hi"), f1)))
    return J


NEG, NS, POS = (-0.2, -0.3, -0.1), (0.0, -0.05, 0.05), (0.2, 0.1, 0.3)
WORSE, FLAT, BETTER = (-0.08, -0.12, -0.04), (0.0, -0.03, 0.03), (0.08, 0.04, 0.12)


@pytest.mark.parametrize("sl0,sl1,f0,f1,expected", [
    (NEG, NEG, WORSE, WORSE, "T-FEEDBACK"),
    (NS, NS, FLAT, FLAT, "T-STABLE"),
    (NS, NS, BETTER, FLAT, "T-STABLE"),             # 'not separated-worse' includes better
    (NS, NS, WORSE, WORSE, "MIXED"),                # this read's real pattern
    (NEG, NS, WORSE, WORSE, "MIXED"),               # one seed only
    (NEG, NEG, WORSE, FLAT, "MIXED"),
    (NEG, NEG, FLAT, FLAT, "MIXED"),
    (POS, POS, WORSE, WORSE, "MIXED"),              # separated, wrong sign
    (NS, POS, FLAT, FLAT, "MIXED"),
])
def test_outcome_logic_known_answers(I, sl0, sl1, f0, f1, expected):
    J = _make(copy.deepcopy(I), sl0, sl1, f0, f1)
    r = TR.evaluate(J)
    assert r["controls"]["all_pass"] and r["outcome"] == expected
    md = TR.render_md(r)
    assert f"**{expected}**" in md


def test_result_json_carries_the_read_numbers_unchanged(I):
    r = TR.evaluate(I)
    for s in TR.SEEDS:
        src = I["read"]["C0"][s]["shift_committed"]["slope"]
        assert r["seeds"][s]["slope"]["point"] == src["slope"] and r["seeds"][s]["slope"]["lo"] == src["lo"] and r["seeds"][s]["slope"]["hi"] == src["hi"]
        assert r["seeds"][s]["lon_f1"]["delta"] == I["read"]["C0"][s]["quality"]["all_replans"]["lon_macro_f1"]["delta"]


def test_written_result_is_the_current_output(I):
    on_disk = json.load(open(f"{PK}/RESULT_TOWN.json", encoding="utf-8"))
    fresh = json.loads(json.dumps(TR.evaluate(I)))
    assert on_disk == fresh and open(f"{PK}/RESULT_TOWN.md", encoding="utf-8").read() == TR.render_md(TR.evaluate(I)) + "\n"


def test_no_banned_vocabulary_in_the_written_files():
    banned = re.compile(r"\b(void|superseded|retracted|formerly|withdrawn|correction)\b|no longer|closed[- ]loop", re.I)
    for f in ("RESULT_TOWN.md", "RESULT_TOWN.json", "town_result.py", "info_extras.py"):
        assert not banned.search(open(f"{PK}/{f}", encoding="utf-8").read()), f
