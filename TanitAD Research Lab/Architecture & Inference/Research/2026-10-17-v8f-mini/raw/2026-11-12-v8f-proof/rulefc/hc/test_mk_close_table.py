"""Tests of mk_close_table.py (card ADD152). Dev box, CPU only (CUDA_VISIBLE_DEVICES=-1), synthetic data; never real data.
  LEADSPEED_DIR=<rulefc/leadspeed> python -I -m pytest -q test_mk_close_table.py"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
LS = os.path.normpath(os.path.join(HERE, "..", "leadspeed"))
os.environ.setdefault("LEADSPEED_DIR", LS)
for p in (HERE, LS):
    if p not in sys.path:
        sys.path.insert(0, p)
import mk_close_table as M  # noqa: E402


def test_build_tables_mean_nan_and_zero():
    a = np.array([np.nan] * 9 + [1.0, 3.0])
    b = np.array([np.nan] * 9 + [2.0, 5.0])
    head, zero = M.build_tables([{"ci": 1, "c148": a, "c149": b, "v": np.arange(11.0)}, {"ci": 0, "c148": a, "c149": b, "v": np.arange(11.0)}])
    assert head["est"][9] == 1.5 and head["est"][10] == 4.0
    assert np.isnan(head["est"][:9]).all()
    assert list(head["clip_index"][:11]) == [1] * 11 and list(head["frame"][:3]) == [0, 1, 2]
    assert (zero["est"] == 0.0).all() and np.array_equal(zero["clip_index"], head["clip_index"]) and np.array_equal(zero["frame"], head["frame"])


def test_build_tables_one_nan_gives_nan():
    a = np.array([np.nan] * 9 + [1.0, np.nan])
    b = np.array([np.nan] * 9 + [2.0, 5.0])
    head, _ = M.build_tables([{"ci": 0, "c148": a, "c149": b, "v": np.zeros(11)}])
    assert np.isnan(head["est"][10])


def test_build_tables_length_mismatch_refused():
    with pytest.raises(ValueError):
        M.build_tables([{"ci": 0, "c148": np.zeros(11), "c149": np.zeros(10), "v": np.zeros(11)}])


def _head(n_clips=3, T=12):
    cols = [{"ci": c, "c148": np.r_[np.full(9, np.nan), np.ones(T - 9)], "c149": np.r_[np.full(9, np.nan), np.ones(T - 9)], "v": np.zeros(T)} for c in range(n_clips)]
    return M.build_tables(cols)[0]


def test_check_rows_pass_and_fail():
    h = _head()
    assert M.check_rows(h, 3, {0: 12, 1: 12, 2: 12})["PASS"]
    assert not M.check_rows(h, 4, {0: 12, 1: 12, 2: 12, 3: 12})["PASS"]                      # a join clip without rows
    h2 = {k: v.copy() for k, v in h.items()}
    h2["est"][9] = np.nan
    assert not M.check_rows(h2, 3, {0: 12, 1: 12, 2: 12})["PASS"]                             # a NaN after the first window
    h3 = {k: v.copy() for k, v in h.items()}
    h3["est"][3] = 0.0
    assert not M.check_rows(h3, 3, {0: 12, 1: 12, 2: 12})["PASS"]                             # a value before the first window
    h4 = {k: v.copy() for k, v in h.items()}
    h4["frame"][5] = 6
    assert not M.check_rows(h4, 3, {0: 12, 1: 12, 2: 12})["PASS"]                             # a frame twice


def test_check_s():
    ok = M.check_s({"c148": 0.4255, "c149": 0.4446}, {"c148": 3430, "c149": 3430})
    assert ok["PASS"]
    assert not M.check_s({"c148": 0.4300, "c149": 0.4446}, {"c148": 3430, "c149": 3430})["PASS"]
    assert not M.check_s({"c148": 0.4255, "c149": 0.4446}, {"c148": 3430, "c149": 3429})["PASS"]


@pytest.fixture()
def fake(tmp_path):
    torch = pytest.importorskip("torch")
    import leadspeed as L
    rng = np.random.default_rng(0)
    ev = tmp_path / "eval"
    (ev / "clips").mkdir(parents=True)
    clips, man = ["cB", "cA", "cC"], []
    for c in clips:
        tok = torch.from_numpy(rng.integers(0, 120, size=(101, 40, 1024), dtype=np.uint8))
        torch.save({"pool4x10": tok, "v": torch.from_numpy(rng.uniform(0, 10, 101).astype(np.float32))}, str(ev / "clips" / (c + ".pt")))
        man.append({"clip": c, "file": "clips/%s.pt" % c, "shapes": {"pool4x10": [101, 40, 1024]}})
    with open(ev / "manifest.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in man) + "\n")
    n = 101
    join = tmp_path / "join.npz"
    np.savez(join, clip=np.array(["cA", "cB", "cC"]), off=np.array([0, n, 2 * n]), n=np.array([n, n, n]),
             lead_valid=np.ones(3 * n, int), has=np.ones(3 * n, int), lead_gap=np.concatenate([20.0 + 5.0 * np.sin(np.arange(n) / (4.0 + k)) for k in range(3)]))
    heads = []
    for name, seed in (("c148", 1), ("c149", 2)):
        torch.manual_seed(seed)
        h = L.make_head(1024, 64, temporal=True)
        p = tmp_path / ("%s.pt" % name)
        torch.save(h.state_dict(), str(p))
        heads.append("%s=%s=%s" % (name, p, hashlib.sha256(p.read_bytes()).hexdigest()))
    return tmp_path, ev, join, ",".join(heads)


def test_main_on_fake_cache(fake, capsys):
    tmp, ev, join, heads = fake
    out = tmp / "out"
    rc = M.main(["--out", str(out), "--evalc", str(ev), "--join", str(join), "--device", "cpu", "--heads", heads])
    res = json.load(open(out / "TABLE.json"))
    assert "QUEUE-COMPLETE" in capsys.readouterr().out
    assert rc == 2, res.get("error")                                    # random heads cannot reproduce row 217's r
    C = res["controls"]
    assert C["t1_ckpt"]["PASS"] and C["t1_rows"]["PASS"] and C["t1_v"]["PASS"] and not C["t1_r"]["PASS"]
    with np.load(out / "HEAD_TABLE.npz") as z:
        assert len(z["est"]) == 3 * 101 and int(np.isfinite(z["est"]).sum()) == 3 * 92
        ci = z["clip_index"]
        # clip index = position in the join: manifest clip "cB" is join index 1
        assert set(ci.tolist()) == {0, 1, 2}
        assert np.allclose(z["est"][np.isfinite(z["est"])], 0.5 * (z["c148"] + z["c149"])[np.isfinite(z["est"])])
    assert res["tables"]["head"] == hashlib.sha256((out / "HEAD_TABLE.npz").read_bytes()).hexdigest()


def test_main_wrong_ckpt_sha(fake):
    tmp, ev, join, heads = fake
    bad = heads.split(",")
    bad[0] = bad[0][:-4] + "0000"
    rc = M.main(["--out", str(tmp / "o2"), "--evalc", str(ev), "--join", str(join), "--device", "cpu", "--heads", ",".join(bad)])
    assert rc == 3 and "sha256" in json.load(open(tmp / "o2" / "TABLE.json"))["error"]


def test_main_planted_heads_reproduce_wants(fake, monkeypatch):
    """Mutation-style: with WANT_S set to what the fake heads give, t1_r passes; so the check reads the S it computes, not a constant."""
    tmp, ev, join, heads = fake
    M.main(["--out", str(tmp / "o3"), "--evalc", str(ev), "--join", str(join), "--device", "cpu", "--heads", heads])
    r = json.load(open(tmp / "o3" / "TABLE.json"))["eval141"]
    monkeypatch.setattr(M, "WANT_S", {"c148": r["c148"]["S"], "c149": r["c149"]["S"]})
    monkeypatch.setattr(M, "WANT_NS", r["c148"]["n_S"])
    rc = M.main(["--out", str(tmp / "o4"), "--evalc", str(ev), "--join", str(join), "--device", "cpu", "--heads", heads])
    assert rc == 0
