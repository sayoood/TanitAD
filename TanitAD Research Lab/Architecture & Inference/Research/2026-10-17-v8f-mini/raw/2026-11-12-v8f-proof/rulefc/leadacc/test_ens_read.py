"""Test of card ADD153 amendment 1's ensemble read (CPU, fake files): the one-head set reproduces the run's own ACC.json, the ensemble of two identical heads equals one head,
and a wrong checkpoint sha256 stops the read."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "leadspeed"))
sys.path.insert(0, HERE)
import ens_read_box2 as E  # noqa: E402
import train_acc_box2 as TA  # noqa: E402
import train_full_box2 as F  # noqa: E402
from test_train_full import _fake  # noqa: E402

torch = pytest.importorskip("torch")


def test_ensemble_read(tmp_path):
    rng = np.random.default_rng(0)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=30, n_eval=3)
    base = ["--arm", "K25D", "--steps", "4", "--ckpt-every", "4", "--n-fit", "24", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard", "--no-marker"]
    outs = {}
    for nm, extra in (("a", []), ("b", ["--train-seed", "149"]), ("c", ["--label-run", "10"])):
        o = os.path.join(str(tmp_path), nm)
        r = TA.main(base + ["--out", o] + extra)
        assert "error" not in r, r.get("traceback")
        outs[nm] = (os.path.join(o, "head_4.pt"), r["train"]["ckpt_sha256"]["4"], os.path.join(o, "ACC.json"))
    common = ["--arm", "K25D", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-marker", "--expect-acc", outs["a"][2]]
    hs = ["%s=%s" % (outs[n][0], outs[n][1]) for n in ("a", "b", "c")]
    r = E.main(common + ["--out", os.path.join(str(tmp_path), "e"), "--heads"] + hs)
    assert "error" not in r, r.get("traceback")
    assert r["controls_all_pass"], r["controls"]
    assert r["ENS2"]["held_out"]["n_A"] == r["single"]["held_out"]["n_A"] > 0
    assert "consequence" in r and r["ENS2"]["name"].startswith("ACCHIST10K-ENS2-")
    # two copies of one head: the ensemble equals the head, the paired difference is exactly 0
    same = ["%s=%s" % (outs["a"][0], outs["a"][1])] * 2 + [hs[2]]
    r2 = E.main(common + ["--out", os.path.join(str(tmp_path), "e2"), "--heads"] + same)
    assert abs(r2["ENS2"]["held_out"]["A"] - r2["single"]["held_out"]["A"]) < 1e-12
    assert r2["ENS2_vs_single"]["held_out"]["dA"]["diff"] == 0.0
    # a wrong sha256 stops the read
    bad = ["%s=%s" % (outs["a"][0], "0" * 64)] + hs[1:]
    r3 = E.main(common + ["--out", os.path.join(str(tmp_path), "e3"), "--heads"] + bad)
    assert "differs" in r3["error"] and not r3["controls"]["e2_checkpoints"]["PASS"]
    json.dumps(r, default=float)


def test_ensemble_read_n(tmp_path):
    import ens_read_n_box2 as EN
    rng = np.random.default_rng(1)
    corp, ev, lf, jf = _fake(str(tmp_path), rng, n_corpus=30, n_eval=3)
    base = ["--arm", "K25D", "--steps", "4", "--ckpt-every", "4", "--n-fit", "24", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-guard", "--no-marker"]
    hs = []
    for nm, extra in (("a", []), ("b", ["--train-seed", "149"]), ("c", ["--label-run", "10"]), ("d", ["--train-seed", "149", "--label-run", "10"]), ("e", ["--train-seed", "150"])):
        o = os.path.join(str(tmp_path), nm)
        r = TA.main(base + ["--out", o] + extra)
        assert "error" not in r, r.get("traceback")
        hs.append("%s=%s" % (os.path.join(o, "head_4.pt"), r["train"]["ckpt_sha256"]["4"]))
    data = ["--arm", "K25D", "--corpus", corp, "--leadf", lf, "--evalc", ev, "--join", jf, "--no-marker"]
    r1 = E.main(data + ["--out", os.path.join(str(tmp_path), "e1"), "--expect-acc", os.path.join(str(tmp_path), "a", "ACC.json"), "--heads"] + hs[:3])
    assert r1["controls_all_pass"], r1["controls"]
    ens = os.path.join(str(tmp_path), "e1", "ENS.json")
    rn = EN.main(data + ["--out", os.path.join(str(tmp_path), "en"), "--expect-ens", ens, "--record", "ENS4", "--compare", "ENS2",
                         "--sets", "ENS2=0,1;ENS4=0,1,2,3;ENS5=0,1,2,3,4", "--heads"] + hs)
    assert "error" not in rn, rn.get("traceback")
    assert rn["controls_all_pass"], rn["controls"]
    assert rn["ENS2"]["held_out"]["A"] == r1["ENS2"]["held_out"]["A"]
    assert rn["ENS4"]["name"].startswith("ACCHIST10K-ENS4-") and "consequence" in rn
    # e1 mutation: a different compare set does not reproduce amendment 1's ENS2
    rm = EN.main(data + ["--out", os.path.join(str(tmp_path), "em"), "--expect-ens", ens, "--record", "ENS4", "--compare", "ENS2",
                         "--sets", "ENS2=0,4;ENS4=0,1,2,3", "--heads"] + hs)
    assert not rm["controls"]["e1_compare_reproduces"]["PASS"] and not rm["controls_all_pass"]
