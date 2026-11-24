"""Tests of box2_rhk_queue.py (card ADD140): rows, order, flags and wrappers of section 3."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box2_rhk_queue as Q  # noqa: E402


def fm(argv):
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
            out[argv[i]] = nxt
            i += 2 if nxt is not True else 1
        else:
            i += 1
    return out


def test_twelve_runs_in_order(tmp_path):
    runs = Q.build_runs(str(tmp_path))
    assert [r["arm"] for r in runs] == ["%s_s%d" % (n, m) for m in (0, 1, 2) for n in ("H_RLcK", "H_RQ50K", "H_RLcK_P30", "U_RLcK_P30")]


def test_flags_and_wrappers(tmp_path):
    for r in Q.build_runs(str(tmp_path)):
        f = fm(r["argv"])
        assert f["--readout"] == "causal_g" and f["--acc-gain"] == "0" and f["--path-exec"] is True
        assert f["--rule-every"] == "5" and f["--kappa-max"] == "0.2" and f["--replan-steps"] == "10"
        assert f["--nav"] == "deadreckon" and f["--scene"] == "logged" and f["--limit"] == "causal"
        pre = r["letter"] in ("k", "l")
        assert r["argv"][1].endswith("rzgfhq.py" if pre else "rzgfh.py")
        assert (f.get("--preroll-steps") == "30") == pre and (("--preroll-steps" in r["argv"]) == pre)
        assert f["--eval-grid"].endswith("grid_p30.json" if pre else "eval_grid_windows.json")
        assert f["--form"] == ("RR" if r["letter"] == "j" else "RLc")
        assert ("--reader" in f) == (r["letter"] == "j") and ("--qmap" in f) == (r["letter"] == "j")
        want = ("UE_s%d" if r["letter"] == "l" else "RH_s%d") % r["m"]
        assert ("/inputs/%s/" % want) in f["--ckpt"]
        if r["letter"] == "j":
            assert f["--reader"].endswith("TK16_s%d/ckpt_last.pt" % (0 if r["m"] == 2 else r["m"]))


def test_timeout_is_card_limit():
    assert Q.RUN_TIMEOUT_S == 15 * 60


def test_mutation_detected(tmp_path, monkeypatch):
    rows = dict(Q.ROWS)
    rows["k"] = ("H_RLcK_P30", "RH", "RLc", 1, 5, "gridp30", 30)  # acc gain 1: not readout K
    monkeypatch.setattr(Q, "ROWS", rows)
    bad = [r for r in Q.build_runs(str(tmp_path)) if fm(r["argv"])["--acc-gain"] != "0"]
    assert len(bad) == 3
