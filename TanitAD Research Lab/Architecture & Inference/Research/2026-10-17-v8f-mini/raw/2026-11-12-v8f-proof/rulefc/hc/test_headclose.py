"""Tests of card ADD152 (HEADCLOSE10K): the generated queue, synthetic data only (and the pulled partners' argv.json files)."""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RF = os.path.normpath(os.path.join(HERE, ".."))
spec = importlib.util.spec_from_file_location("q_hc", os.path.join(HERE, "box2_headclose_queue.py"))
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
ROOT = "D:/tanitad/rhacc10k"
PARTNER_PULL = {"H_RQ50KV1": os.path.join(RF, "e1r", "pull"), "H_RQ50KV1_P30": os.path.join(RF, "e1a", "pull"),
                "H_RQ50KVz": os.path.join(RF, "e1r", "pull"), "H_RQ50KVz_P30": os.path.join(RF, "e1a", "pull"), "H_RQ50KC1": os.path.join(RF, "e1r", "pull")}


def test_queue_order_and_forms():
    runs = Q.build_runs(ROOT)
    assert [r["arm"] for r in runs] == ["H_RQ50KVz_s0", "H_RQ50KVz_P30_s0", "H_RQ50KV1_s0", "H_RQ50KV1_P30_s0", "H_RQ50KV1_s1", "H_RQ50KV1_P30_s1",
                                         "H_RQ50KV1_s2", "H_RQ50KV1_P30_s2"]
    for r in runs:
        a = r["argv"]
        assert a[1].endswith("/build/v8f_t1_eval_reader_rule_rzgfhqet.py") and a[a.index("--form") + 1] == "RRe"
        assert a[a.index("--rule-every") + 1] == "1"
        t = a[a.index("--closing-table") + 1]
        assert t.endswith("ZERO_TABLE.npz" if "Vz" in r["arm"] else "HEAD_TABLE.npz")
        assert ("--preroll-steps" in a) == ("P30" in r["arm"])


def test_ceiling_rows():
    runs = Q.build_runs(ROOT, ceiling=True)
    assert [r["arm"] for r in runs] == ["H_RQ50KC1_s0", "H_RQ50KC1_s1", "H_RQ50KC1_s2"]
    for r in runs:
        a = r["argv"]
        assert a[a.index("--form") + 1] == "RRc" and "--closing-table" not in a and a[1].endswith("rzgfhqet.py")


@pytest.mark.parametrize("ceiling", [False, True])
def test_argv_equals_partner_apart_from_the_card_items(ceiling):
    n = 0
    for r in Q.build_runs(ROOT, ceiling=ceiling):
        part = Q.PARTNER.get(r["name"], {"H_RQ50KV1": "H_RQ50KE1", "H_RQ50KV1_P30": "H_RQ50KE1_P30", "H_RQ50KC1": "H_RQ50KE1"}.get(r["name"]))
        pa = "%s_s%d" % (part, r["m"])
        ref = json.load(open(os.path.join(PARTNER_PULL[r["name"]], pa, "argv.json"), encoding="utf-8"))["argv"]
        got = [x.replace(r["arm"], pa) for x in r["argv"]]
        if "--closing-table" in got:
            i = got.index("--closing-table")
            del got[i:i + 2]
        got[got.index("--form") + 1] = "RR"
        want_wrapper = "rzgfhq.py" if "P30" in r["arm"] else "rzgfh.py"
        assert ref[1].endswith(want_wrapper)
        got[1] = ref[1]
        assert got == ref, (r["arm"], [(x, y) for x, y in zip(got, ref) if x != y])
        n += 1
    assert n == (3 if ceiling else 8)


def test_pinned_sha_entries():
    assert Q.PINNED_SHA["wrapper_et"] == "02a58e655e5c1979776e9034b3ec54718dc82ab9dc3271524edfe07782b7cf9e"
    for k in ("head_table", "zero_table"):
        assert len(Q.PINNED_SHA[k]) == 64


def _mkrun(tmp, arm, tag, vals, rule_extra=None):
    d = tmp / "runs" / arm
    (d / ("dump_" + tag)).mkdir(parents=True)
    (d / "ticks").mkdir()
    for i, v in enumerate(vals):
        np.savez(d / ("dump_" + tag) / ("ep%04d.npz" % i), plan=np.array([v, np.nan]), name=np.array(arm))
    np.savez(d / "ticks" / ("rule_log_%s.npz" % arm), g0=np.array([1.0, 2.0]), form=np.array("x"), **(rule_extra or {}))
    np.savez(d / "ticks" / ("rule_log_mid_%s.npz" % arm), q=np.arange(1, 10))
    return d


def _r(tmp):
    return {"name": "H_RQ50KVz", "m": 0, "arm": "H_RQ50KVz_s0", "tag": "H_RQ50KVz_s0_m10_causal_heads", "outdir": str(tmp / "runs" / "H_RQ50KVz_s0")}


def test_compare_partner_pass_and_fail(tmp_path):
    _mkrun(tmp_path, "H_RQ50KE1_s0", "H_RQ50KE1_s0_m10_causal_heads", [1.0, 2.0, 3.0])
    _mkrun(tmp_path, "H_RQ50KVz_s0", "H_RQ50KVz_s0_m10_causal_heads", [1.0, 2.0, 3.0], {"c_est": np.zeros(2)})
    ok, res = Q.compare_partner(str(tmp_path), _r(tmp_path))
    assert ok and res["dump_files"] == 3 and res["dump_numeric_arrays"] == 3 and "ep0000.npz" in res["non_numeric_differ"]
    assert json.load(open(tmp_path / "runs" / "H_RQ50KVz_s0" / "CONTROL_ZERO.json"))["PASS"] is True
    np.savez(tmp_path / "runs" / "H_RQ50KVz_s0" / "dump_H_RQ50KVz_s0_m10_causal_heads" / "ep0001.npz", plan=np.array([2.0 + 1e-12, np.nan]), name=np.array("x"))
    ok, res = Q.compare_partner(str(tmp_path), _r(tmp_path))
    assert not ok and res["dump_bad"][0][0] == "ep0001.npz"


def test_compare_partner_missing_key_and_file(tmp_path):
    _mkrun(tmp_path, "H_RQ50KE1_s0", "H_RQ50KE1_s0_m10_causal_heads", [1.0, 2.0], {"extra": np.ones(2)})
    _mkrun(tmp_path, "H_RQ50KVz_s0", "H_RQ50KVz_s0_m10_causal_heads", [1.0])
    ok, res = Q.compare_partner(str(tmp_path), _r(tmp_path))
    assert not ok and res["dump_bad"] == [["ep0001.npz", "missing"]] and res["logs"]["rule_log_X.npz"]["bad"] == ["missing:extra"]


def test_run_one_stops_on_failed_control(tmp_path, monkeypatch):
    _mkrun(tmp_path, "H_RQ50KE1_s0", "H_RQ50KE1_s0_m10_causal_heads", [1.0])
    r = [x for x in Q.build_runs(str(tmp_path)) if x["arm"] == "H_RQ50KVz_s0"][0]

    def fake_exec(argv, env, log, t):
        d = tmp_path / "runs" / "H_RQ50KVz_s0"
        (d / "dump_H_RQ50KVz_s0_m10_causal_heads").mkdir()
        np.savez(d / "dump_H_RQ50KVz_s0_m10_causal_heads" / "ep0000.npz", plan=np.array([9.0, np.nan]), name=np.array("y"))
        np.savez(d / "ticks" / "rule_log_H_RQ50KVz_s0.npz", g0=np.array([1.0, 2.0]), form=np.array("x"))
        np.savez(d / "ticks" / "rule_log_mid_H_RQ50KVz_s0.npz", q=np.arange(1, 10))
        return 0
    monkeypatch.setattr(Q, "execute", fake_exec)
    monkeypatch.setattr(Q, "free_vram_mib", lambda: 8000.0)
    monkeypatch.setattr(Q, "grid_clip_count", lambda p: 1)
    with pytest.raises(Q.Stop) as e:
        Q.run_one(str(tmp_path), r, {})
    assert "CONTROL-ZERO-FAILED" in str(e.value)
    assert not (tmp_path / "runs" / "H_RQ50KVz_s0" / "RUN_DONE.json").exists()
