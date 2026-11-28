"""Tests of box2_crhb_queue.py (card ADD142): six runs, reader form, swapped reader seeds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box2_crhb_queue as Q  # noqa: E402


def test_runs_and_readers(tmp_path):
    runs = Q.build_runs(str(tmp_path))
    assert [r["arm"] for r in runs] == ["%s_s%d" % (n, m) for m in (0, 1, 2) for n in ("H_RQ50Kx", "U_RQ50Kx")]
    want_r = {0: 1, 1: 0, 2: 1}
    for r in runs:
        a = r["argv"]
        rd = a[a.index("--reader") + 1]
        qm = a[a.index("--qmap") + 1]
        assert rd.endswith("TK16_s%d/ckpt_last.pt" % want_r[r["m"]]) and qm.endswith("QMAP_LR16_s%d_q50.json" % want_r[r["m"]])
        assert a[a.index("--form") + 1] == "RR" and a[a.index("--acc-gain") + 1] == "0" and a[a.index("--rule-every") + 1] == "5"
        assert "--preroll-steps" not in a and a[1].endswith("rzgfh.py")
        assert ("/inputs/%s_s%d/" % ("RH" if r["name"].startswith("H_") else "UE", r["m"])) in a[a.index("--ckpt") + 1]


def test_mutation_reader_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "READER_SEED", {0: 0, 1: 1, 2: 0})
    a = Q.build_runs(str(tmp_path))[0]["argv"]
    assert not a[a.index("--reader") + 1].endswith("TK16_s1/ckpt_last.pt")
