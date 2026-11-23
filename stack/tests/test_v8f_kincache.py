"""KINCACHE (BRIEF_BUILD_KINCACHE, MM design binding): ``--dagger-label-cache PATH``, the labeller's outputs for one bank written once (outside the GPU lock) and read back by every later trainer load.

What is pinned here, each with a test whose mutant turns it RED (``mutants_kincache/MUTANTS_RESULT.json``):
  * ROUND TRIP, bit for bit: labels read back from a written cache equal a fresh compute (slot, label, status_counts, is_kin, n_labelled, census), for speed_v1 and kin_v1, on the fixture bank and a planted one;
  * REFUSALS, each naming its field: bank.json / labeller bytes / dagger bytes / label kind / n_records / eligible index / record arrays / clip store; an unreadable or corrupt file; a truncated ``.part`` is never read;
  * ATOMIC write (a writer killed half way leaves the old cache, or none);
  * NO FLAG = the 0ca6f49d behaviour (golden copy of its v8f_dagger.py, the labeller bytes, the labeller runs and nothing is written);
  * REACHABILITY: through the trainer's argv with the flag and a cache, the swapped batch's ``gt`` and the plan loss / gradient equal the no-cache run EXACTLY;
  * the cache path and sha256 are BUILT levers (G-DVB), refused at ``--dagger-p 0``, without speed_v1 | kin_v1, and when the file changed after the load.
Run the WHOLE file (module fixtures are shared, and a ``-k`` run leaks them): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_kincache.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402
from test_v8f_dagger import world, fx10, make_batch  # noqa: E402,F401
from test_v8f_dagger_train import pack, DAG, TR, DG, M, dvb, _woken, TINY, _run, _summary, _selfcheck  # noqa: E402,F401
from test_v8f_recovery import eligible_windows_of, _same  # noqa: E402,F401
from test_v8f_recovery_wiring import gate_parts_rl  # noqa: E402,F401
from test_v8f_kinlab import kin_bank  # noqa: E402,F401

torch.set_num_threads(1)
GOLD_D = HERE / "golden_v8f_dagger_0ca6f49d.py"
GOLD_D_SHA256 = "69c513bcd4bd4fbd36b4c3c24b9365f8a0f31324f881cacc3e66f0b1f409b127"      # sha256 of `git show 0ca6f49d:stack/tanitad/data/v8f_dagger.py`
LABELLER_0CA6F49D_SHA256 = "795d380cb5bf0a04fb24b84392b6bd5f71df15bbc5ecb78a6267488225dd7440"   # sha256 of `git show 0ca6f49d:stack/tanitad/data/v8f_recovery.py` (this change does not touch it)
KINDS = ("speed_v1", "kin_v1")


# ------------------------------------------------------------------------------------------------ helpers
def _compute(f, bank, kind):
    fn = RV.kin_label_bank if kind == "kin_v1" else RV.label_bank
    return fn(bank.a, bank.ci, f.J["clip"], f.store.get_kin)


def _ident(f, bank, kind, read_clip=None, labeller_path=None, dagger_path=None):
    return G.label_cache_identity(bank, f.J["clip"], read_clip or f.store.get_kin, label_kind=kind, labeller_path=labeller_path or RV.__file__, dagger_path=dagger_path or G.__file__)


def _attach(bank, res, kind):
    if kind == "kin_v1":
        bank.attach_recovery(res["slot"], res["label"], res["status_counts"], kin=res["is_kin"])
    else:
        bank.attach_recovery(res["slot"], res["label"], res["status_counts"])


def _clone(f, bank, sha=None, mod=None):
    arrays = {k: np.array(v) for k, v in bank.a.items()}
    if mod is not None:
        mod(arrays)
    return G.DaggerBank(arrays, bank.header, clip_names=f.J["clip"], manifest=bank.manifest, sha256=sha or bank.sha256, path=bank.path)


def _assert_same_res(x, y, kind):
    """Bit for bit: same dtype, shape and bytes; same dict (keys in the same order); the census as the same canonical JSON."""
    for k in ("slot", "label") + (("is_kin",) if kind == "kin_v1" else ()):
        assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes(), k
    assert list(x["status_counts"].items()) == list(y["status_counts"].items())
    assert x["n_labelled"] == y["n_labelled"] and isinstance(x["n_labelled"], int)
    assert G.canon(x["census"]) == G.canon(y["census"])
    assert json.dumps(x["census"]) == json.dumps(y["census"])


def _fields(msg):
    m = re.search(r": \[([a-z_0-9, ]+)\] ", msg)
    assert m, msg
    return {s.strip() for s in m.group(1).split(",")}


def _refused(path, ident):
    with pytest.raises(G.BankRefused) as e:
        G.read_label_cache(path, ident)
    return str(e.value)


@pytest.fixture(scope="module")
def banks(fx10):
    return {"natural": fx10.bank, "planted": kin_bank(fx10, plant=10**9, kappa=0.02, da=0.6)}


@pytest.fixture(scope="module")
def cached(fx10, banks, tmp_path_factory):
    """A written cache per label kind for the planted bank: {kind: (path, sha256, fresh compute)}."""
    d = tmp_path_factory.mktemp("kincache")
    out = {}
    for kind in KINDS:
        res = _compute(fx10, banks["planted"], kind)
        path = d / f"{kind}.npz"
        out[kind] = (path, G.write_label_cache(path, _ident(fx10, banks["planted"], kind), res), res)
    return out


# ------------------------------------------------------------------------------------------------ round trip
@pytest.mark.parametrize("which", ["natural", "planted"])
@pytest.mark.parametrize("kind", KINDS)
def test_round_trip_a_written_then_loaded_cache_equals_a_fresh_compute_bit_for_bit(fx10, banks, tmp_path, kind, which):
    bank = banks[which]
    res = _compute(fx10, bank, kind)
    ident = _ident(fx10, bank, kind)
    path = tmp_path / "c.npz"
    sha = G.write_label_cache(path, ident, res)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest() and not Path(str(path) + ".part").exists()
    back, sha2 = G.read_label_cache(path, ident)
    assert sha2 == sha
    _assert_same_res(back, res, kind)
    assert res["n_labelled"] == int((bank.reason == G.OK).sum()) and len(res["label"]) > 0
    if which == "planted" and kind == "kin_v1":
        assert int(res["is_kin"].sum()) > 0, "the planted bank must carry kin_v1 labels, else the round trip of is_kin is vacuous"
    b1, b2 = _clone(fx10, bank), _clone(fx10, bank)                          # attached labels: the bank's own record equals, and so does the swapped batch
    _attach(b1, res, kind)
    _attach(b2, back, kind)
    assert b1.label_kind == b2.label_kind == kind
    for k in ("slot", "label", "has"):
        assert b1.rl[k].tobytes() == b2.rl[k].tobytes()
    assert b1.rl["status_counts"] == b2.rl["status_counts"] and (kind == "speed_v1" or b1.rl["kin"].tobytes() == b2.rl["kin"].tobytes())
    wins = eligible_windows_of(fx10, b1)
    batch = make_batch(fx10, wins)
    o1, o2 = G.DaggerSwap(b1, p=1.0, seed=3).apply(batch, 0), G.DaggerSwap(b2, p=1.0, seed=3).apply(batch, 0)
    assert _same(o1, o2), "the swapped batch from a loaded cache equals the one from a fresh compute"
    sha_b = G.write_label_cache(tmp_path / "c2.npz", ident, res)
    assert sha_b == sha, "the file is deterministic: the same labels give the same bytes (so the sha256 names the content)"


def test_the_arrays_hold_no_pickle_and_the_cache_is_a_plain_npz(cached):
    path, _, _ = cached["kin_v1"]
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"slot", "label", "is_kin", "status_counts_json", "census_json", "header_json"}
        assert all(z[k].dtype != object for k in z.files)
        hdr = json.loads(bytes(z["header_json"]).decode())
    assert set(G.LABEL_CACHE_FIELDS) <= set(hdr) and hdr["label_kind"] == "kin_v1" and hdr["version"] == G.LABEL_CACHE_VERSION
    with np.load(cached["speed_v1"][0], allow_pickle=False) as z:
        assert "is_kin" not in z.files


# ------------------------------------------------------------------------------------------------ refusals
def test_a_matching_header_is_accepted_for_both_kinds(fx10, banks, cached):
    for kind in KINDS:
        back, sha = G.read_label_cache(cached[kind][0], _ident(fx10, banks["planted"], kind))
        assert sha == cached[kind][1]
        _assert_same_res(back, cached[kind][2], kind)


def test_refusal_bank_json_changed(fx10, banks, cached):
    msg = _refused(cached["kin_v1"][0], _ident(fx10, _clone(fx10, banks["planted"], sha="0" * 64), "kin_v1"))
    assert _fields(msg) == {"bank_json_sha256"}, msg


def test_refusal_labeller_bytes_changed(fx10, banks, cached, tmp_path):
    alt = tmp_path / "v8f_recovery_changed.py"
    alt.write_bytes(Path(RV.__file__).read_bytes() + b"\n# one byte more\n")
    msg = _refused(cached["kin_v1"][0], _ident(fx10, banks["planted"], "kin_v1", labeller_path=alt))
    assert _fields(msg) == {"labeller_sha256"}, msg


def test_refusal_dagger_bytes_changed(fx10, banks, cached, tmp_path):
    alt = tmp_path / "v8f_dagger_changed.py"
    alt.write_bytes(Path(G.__file__).read_bytes() + b"\n# one byte more\n")
    msg = _refused(cached["kin_v1"][0], _ident(fx10, banks["planted"], "kin_v1", dagger_path=alt))
    assert _fields(msg) == {"dagger_sha256"}, msg


def test_refusal_label_kind_differs_in_both_directions(fx10, banks, cached):
    msg = _refused(cached["kin_v1"][0], _ident(fx10, banks["planted"], "speed_v1"))
    assert _fields(msg) == {"label_kind"}, msg
    msg = _refused(cached["speed_v1"][0], _ident(fx10, banks["planted"], "kin_v1"))
    assert _fields(msg) == {"label_kind"}, msg
    with pytest.raises(G.BankRefused):                                         # human has no cache at all
        _ident(fx10, banks["planted"], "human")


def test_refusal_eligible_index_differs(fx10, banks, cached):
    b = _clone(fx10, banks["planted"])
    b._ok_idx = b._ok_idx[:-1].copy()                                          # the eligible index alone (the record arrays are untouched)
    msg = _refused(cached["kin_v1"][0], _ident(fx10, b, "kin_v1"))
    assert _fields(msg) == {"eligible_index_sha256"}, msg


def test_refusal_record_count_differs(fx10, banks, cached):
    ident = _ident(fx10, banks["planted"], "kin_v1")
    msg = _refused(cached["kin_v1"][0], {**ident, "n_records": ident["n_records"] + 1})
    assert _fields(msg) == {"n_records"}, msg

    def grow(a):                                                               # a real bank with one record more
        for k in list(a):
            a[k] = np.concatenate([a[k], a[k][-1:]])
    big = _clone(fx10, banks["planted"], mod=grow)
    assert big.n == ident["n_records"] + 1
    assert "n_records" in _fields(_refused(cached["kin_v1"][0], _ident(fx10, big, "kin_v1")))


def test_refusal_record_arrays_differ(fx10, banks, cached):
    def bump(a):
        ok = np.flatnonzero(a["reason"] == G.OK)[0]
        a["p10f"][ok, -1, 0] += np.float32(0.25)
    msg = _refused(cached["kin_v1"][0], _ident(fx10, _clone(fx10, banks["planted"], mod=bump), "kin_v1"))
    assert _fields(msg) == {"records_sha256"}, msg


def test_refusal_clip_store_differs(fx10, banks, cached):
    def moved(name):
        d = dict(fx10.store.get_kin(name))
        pose = np.array(d["pose"], np.float64)
        pose[:, 0] += 1e-6
        d["pose"] = pose
        return d
    msg = _refused(cached["kin_v1"][0], _ident(fx10, banks["planted"], "kin_v1", read_clip=moved))
    assert _fields(msg) == {"clip_store_sha256"}, msg


def test_refusal_an_unreadable_truncated_or_corrupt_file(fx10, banks, cached, tmp_path):
    ident = _ident(fx10, banks["planted"], "kin_v1")
    raw = cached["kin_v1"][0].read_bytes()
    (tmp_path / "trunc.npz").write_bytes(raw[: len(raw) // 2])
    assert "[unreadable]" in _refused(tmp_path / "trunc.npz", ident)
    (tmp_path / "junk.npz").write_bytes(b"not a zip at all")
    assert "[unreadable]" in _refused(tmp_path / "junk.npz", ident)
    (tmp_path / "empty.npz").write_bytes(b"")
    assert "[unreadable]" in _refused(tmp_path / "empty.npz", ident)
    with np.load(cached["kin_v1"][0], allow_pickle=False) as z:                # an intact zip whose label array was edited: the header's arrays_sha256 refuses it
        arrs = {k: np.array(z[k]) for k in z.files}
    arrs["label"] = arrs["label"].copy()
    arrs["label"][0, 0, 0] += 1e-9
    G._save_npz_det(tmp_path / "edited.npz", arrs)
    assert _fields(_refused(tmp_path / "edited.npz", ident)) == {"arrays_sha256"} or "arrays_sha256" in _refused(tmp_path / "edited.npz", ident)


# ------------------------------------------------------------------------------------------------ atomic write
def test_the_write_is_atomic_a_writer_killed_half_way_leaves_the_old_cache_or_none(fx10, banks, cached, tmp_path, monkeypatch):
    ident = _ident(fx10, banks["planted"], "kin_v1")
    res = cached["kin_v1"][2]
    real = G._save_npz_det

    def dying(path, arrays):
        real(path, arrays)
        raw = Path(path).read_bytes()
        Path(path).write_bytes(raw[: len(raw) // 2])                           # the process dies with half the bytes written
        raise KeyboardInterrupt("killed mid-write")
    good = tmp_path / "good.npz"
    G.write_label_cache(good, ident, res)
    before = good.read_bytes()
    absent = tmp_path / "absent.npz"
    monkeypatch.setattr(G, "_save_npz_det", dying)
    with pytest.raises(KeyboardInterrupt):
        G.write_label_cache(good, ident, res)                                  # over an existing cache: it must stay byte-identical
    with pytest.raises(KeyboardInterrupt):
        G.write_label_cache(absent, ident, res)                                # no cache: none must appear
    monkeypatch.setattr(G, "_save_npz_det", real)
    assert good.read_bytes() == before and not absent.exists()
    assert Path(str(absent) + ".part").exists(), "the truncated half sits in <path>.part"
    assert "[unreadable]" in _refused(Path(str(absent) + ".part"), ident), "and a truncated .part is unreadable even if it were read"
    G.write_label_cache(absent, ident, res)                                    # the next write replaces the stale .part by a complete file and leaves no .part
    assert absent.exists() and not Path(str(absent) + ".part").exists()


# ------------------------------------------------------------------------------------------------ the trainer: flag, levers, reachability
def _parts(pack, kind, cache=None, p=1.0, extra=()):
    return gate_parts_rl(pack, [*DAG(pack, p, 3), "--dagger-label", kind, *(["--dagger-label-cache", str(cache)] if cache is not None else []), *extra])


def _block_labellers(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the labeller ran, but the cache should have been read")
    monkeypatch.setattr(TR.DR, "kin_label_bank", boom)
    monkeypatch.setattr(TR.DR, "label_bank", boom)


def _count_labellers(monkeypatch):
    n = {"kin_label_bank": 0, "label_bank": 0}
    for name in n:
        real = getattr(TR.DR, name)

        def wrap(*a, _real=real, _name=name, **k):
            n[_name] += 1
            return _real(*a, **k)
        monkeypatch.setattr(TR.DR, name, wrap)
    return n


def test_the_flag_exists_defaults_to_none_and_has_a_declared_vs_built_entry():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_label_cache == "" and TR.FLAG_CHECKS["dagger_label_cache"] == "dagger:dagger_label_cache"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)
    lv = G.DaggerSwap().built_levers()
    assert lv["dagger_label_cache"] is None and lv["dagger_label_cache_sha256"] is None


@pytest.mark.parametrize("kind", KINDS)
def test_reachability_trainer_argv_cache_written_then_read_gt_and_the_plan_loss_equal_the_no_cache_run_exactly(pack, tmp_path, monkeypatch, kind):
    cache = tmp_path / f"{kind}_cache.npz"
    base = _parts(pack, kind)                                                   # no flag: the labeller runs, as at 0ca6f49d
    assert base[5].built_levers()["dagger_label_cache"] is None and "label_cache" not in base[4]["dagger"]["speed_label"]
    w = _parts(pack, kind, cache)                                               # flag, PATH absent: compute, then write
    assert cache.is_file() and not Path(str(cache) + ".part").exists()
    assert w[4]["dagger"]["speed_label"]["label_cache"]["read"] is False
    _block_labellers(monkeypatch)                                               # flag, PATH present: the labeller must not run
    stat0 = (cache.stat().st_mtime_ns, cache.stat().st_size)
    r = _parts(pack, kind, cache)
    monkeypatch.undo()
    assert (cache.stat().st_mtime_ns, cache.stat().st_size) == stat0, "a load only READS the cache: the file is not rewritten"
    ctl_r = r[4]["dagger"]["speed_label"]
    assert ctl_r["label_cache"]["read"] is True and ctl_r["label_cache"]["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    for rl in (w[5].bank.rl, r[5].bank.rl):                                     # the attached labels: the same bytes as the no-cache run
        for k in ("slot", "label", "has"):
            assert rl[k].dtype == base[5].bank.rl[k].dtype and rl[k].tobytes() == base[5].bank.rl[k].tobytes(), k
        assert list(rl["status_counts"].items()) == list(base[5].bank.rl["status_counts"].items())
        assert (rl["kin"] is None) == (base[5].bank.rl["kin"] is None) and (kind == "speed_v1" or rl["kin"].tobytes() == base[5].bank.rl["kin"].tobytes())
    for k in ("status_counts", "n_labelled", "n_emitted", "n_eligible", "n_eligible_after", "census", "label"):
        assert base[4]["dagger"]["speed_label"][k] == ctl_r[k], k
    # G-DVB: the path and the sha256 are BUILT levers, declared == built, and the whole gate passes
    lv = r[5].built_levers()
    assert lv["dagger_label_cache"] == str(cache.resolve()) or Path(lv["dagger_label_cache"]).samefile(cache)
    assert lv["dagger_label_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert dvb(r[:9]) == [] and dvb(w[:9]) == [] and dvb(base[:9]) == []
    # the batch: gt (and everything else) equal exactly; the plan loss and its gradient equal to 0.0 difference
    a, cfg, _, _, info, dag_r, J, wins, clips, store = r
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    el = [x for x in wins if dag_r.bank.candidates([x[0]], [x[1]])[0][0] > 0][:24]
    b = D.collate([ds.item(x[0], x[1], 1, 3) for x in el])
    out_b, out_r = base[5].apply(b, 1), dag_r.apply(b, 1)
    assert base[5].last["n_label_replaced"] > 0 and dag_r.last["n_label_replaced"] == base[5].last["n_label_replaced"] and dag_r.last["n_label_kin"] == base[5].last["n_label_kin"]
    assert torch.equal(out_b["gt"], out_r["gt"]) and _same(out_b, out_r)
    cfg16 = M.V8FMiniConfig(**{**TINY, "hmax": 16, "wm": "b"})
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg16, M.P10Prior.synthetic(0), mu, sd))
    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]

    def grads(bb):
        for _, q in named:
            q.grad = None
        torch.manual_seed(20261026)
        T = model.loss(bb, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
        T["total"].backward()
        return float(T["total"].detach()), float(T["ade"].detach()), {n: (q.grad.detach().clone() if q.grad is not None else torch.zeros_like(q)) for n, q in named}
    tot_b, ade_b, g_b = grads(out_b)
    tot_r, ade_r, g_r = grads(out_r)
    assert tot_b - tot_r == 0.0 and ade_b - ade_r == 0.0, "the plan loss from a cached label equals the no-cache run to 0.0 difference"
    assert float(sum((g_b[n] - g_r[n]).abs().sum() for n in g_b)) == 0.0 and float(sum(x.abs().sum() for x in g_b.values())) > 0.0


def test_no_flag_is_the_0ca6f49d_behaviour_labeller_bytes_golden_dagger_module_and_the_labeller_runs(fx10, banks, pack, tmp_path, monkeypatch):
    # KINTB (2026-11-23) edited the labeller (v8f_recovery.py: T_B became a parameter), so its bytes are no longer 0ca6f49d's; test_v8f_kintb.py pins its behaviour at the default against golden_v8f_recovery_1a8b11a1.py.
    assert hashlib.sha256(GOLD_D.read_bytes()).hexdigest() == GOLD_D_SHA256
    spec = importlib.util.spec_from_file_location("golden_v8f_dagger_0ca6f49d", GOLD_D)
    gold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gold)
    assert gold.constants_sha256() == G.constants_sha256()
    for kind in ("human", "speed_v1", "kin_v1"):                                # the swap, with each label kind, equals 0ca6f49d's bit for bit
        bn, bg = _clone(fx10, banks["planted"]), gold.DaggerBank(*(lambda ar: (ar, banks["planted"].header))({k: np.array(v) for k, v in banks["planted"].a.items()}),
                                                                 clip_names=fx10.J["clip"], manifest=banks["planted"].manifest, sha256=banks["planted"].sha256, path=banks["planted"].path)
        if kind != "human":
            res = _compute(fx10, bn, kind)
            _attach(bn, res, kind)
            _attach(bg, res, kind)
        wins = eligible_windows_of(fx10, bn)
        batch = make_batch(fx10, wins)
        for step in (0, 5):
            assert _same(G.DaggerSwap(bn, p=1.0, seed=3).apply(batch, step), gold.DaggerSwap(bg, p=1.0, seed=3).apply(batch, step)), (kind, step)
        lv_n, lv_g = G.DaggerSwap(bn, p=1.0, seed=3).built_levers(), gold.DaggerSwap(bg, p=1.0, seed=3).built_levers()
        assert lv_n.pop("dagger_base") == "own"                                                      # KINPUT: the one new key since fde8b716 (this golden is 0ca6f49d's)
        assert lv_n.pop("dagger_kin_tb") == 1.0 and lv_n.pop("dagger_label_cache") is None and lv_n.pop("dagger_label_cache_sha256") is None and lv_n == lv_g, "no flag: the two new levers read None, every other lever is 0ca6f49d's"
    # the trainer with no flag: labeller runs once, nothing is written, the ctl record has no cache key, and a cache lying beside the bank is NOT read
    n = _count_labellers(monkeypatch)
    stray = Path(pack.bank) / "label_cache.npz"      # the place a "default path" mutant would look
    r = _parts(pack, "kin_v1")
    assert n == {"kin_label_bank": 1, "label_bank": 0} and "label_cache" not in r[4]["dagger"]["speed_label"]
    store_get = getattr(r[9], "get_kin", None) or r[9].get
    stray_ident = G.label_cache_identity(r[5].bank, r[6]["clip"], store_get, label_kind="kin_v1", labeller_path=RV.__file__, dagger_path=G.__file__)
    direct = RV.kin_label_bank(r[5].bank.a, r[5].bank.ci, r[6]["clip"], store_get)
    G.write_label_cache(stray, stray_ident, direct)
    try:
        n["kin_label_bank"] = 0
        r2 = _parts(pack, "kin_v1")
        assert n["kin_label_bank"] == 1, "no flag: a cache file lying beside the bank must not be read"
        assert r2[5].built_levers()["dagger_label_cache"] is None
        assert r2[5].bank.rl["label"].tobytes() == direct["label"].tobytes() and r2[5].bank.rl["slot"].tobytes() == direct["slot"].tobytes()
    finally:
        stray.unlink()


def test_a_truncated_part_file_left_behind_is_never_read_and_is_replaced(pack, tmp_path, monkeypatch):
    good = tmp_path / "good.npz"
    ref = _parts(pack, "speed_v1", good)
    raw = good.read_bytes()
    cache = tmp_path / "c.npz"
    Path(str(cache) + ".part").write_bytes(raw[: len(raw) // 3])
    n = _count_labellers(monkeypatch)
    w = _parts(pack, "speed_v1", cache)
    assert n["label_bank"] == 1, "PATH absent: the labeller ran (the .part was not read)"
    assert cache.is_file() and not Path(str(cache) + ".part").exists()
    assert w[4]["dagger"]["speed_label"]["label_cache"]["read"] is False
    assert cache.read_bytes() == raw and w[5].bank.rl["label"].tobytes() == ref[5].bank.rl["label"].tobytes()
    cache2 = tmp_path / "c2.npz"                                                # a truncated file AT PATH is a named refusal, not a silent recompute
    cache2.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(SystemExit) as e:
        _parts(pack, "speed_v1", cache2)
    assert "ZZDVB-REFUSEDZZ" in str(e.value) and "[unreadable]" in str(e.value)


def test_the_trainer_refuses_a_stale_cache_naming_the_field(pack, tmp_path, monkeypatch):
    cache = tmp_path / "k.npz"
    _parts(pack, "kin_v1", cache)
    with pytest.raises(SystemExit) as e:                                        # label kind: a kin_v1 cache under --dagger-label speed_v1
        _parts(pack, "speed_v1", cache)
    assert _fields(str(e.value)) == {"label_kind"}, str(e.value)
    alt = tmp_path / "rec_changed.py"
    alt.write_bytes(Path(RV.__file__).read_bytes() + b"\n# changed\n")
    with monkeypatch.context() as m:
        m.setattr(TR.DR, "__file__", str(alt))
        with pytest.raises(SystemExit) as e:
            _parts(pack, "kin_v1", cache)
    assert _fields(str(e.value)) == {"labeller_sha256"}, str(e.value)
    alt2 = tmp_path / "dag_changed.py"
    alt2.write_bytes(Path(G.__file__).read_bytes() + b"\n# changed\n")
    with monkeypatch.context() as m:
        m.setattr(TR.DG, "__file__", str(alt2))
        with pytest.raises(SystemExit) as e:
            _parts(pack, "kin_v1", cache)
    assert _fields(str(e.value)) == {"dagger_sha256"}, str(e.value)
    _parts(pack, "kin_v1", cache)                                               # and the untouched cache still loads


def test_g_dvb_refuses_the_flag_at_p_zero_without_a_label_and_when_the_file_changed_after_the_load(pack, tmp_path):
    cache = tmp_path / "g.npz"
    r = _parts(pack, "kin_v1", cache)
    assert dvb(r[:9]) == []
    msgs = dvb(_parts(pack, "kin_v1", cache, p=0.0, extra=[])[:9])              # p = 0: the lever is declared and does nothing
    assert any("dagger-label-cache" in m and "declared and does nothing" in m for m in msgs), msgs
    a = TR.build_parser().parse_args(["--out", str(pack.init_out), *__import__("test_v8f_dagger_train").GATE_ARGV, "--dagger-label-cache", str(cache)])      # no swap built at all
    assert any("dagger_label_cache" in m and "no DAgger swap was built" in m for m in TR.bad_dagger(a, None))
    with pytest.raises(SystemExit) as e:                                         # a human label with a cache at p > 0: refused at the load, naming why
        gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label-cache", str(cache)])
    assert "valid only with --dagger-label speed_v1 | kin_v1" in str(e.value)
    raw = cache.read_bytes()                                                      # the file changes after the load: the gate re-hashes it
    try:
        cache.write_bytes(raw + b"\x00")
        bad = dvb(r[:9])
        assert any("lever dagger_label_cache_sha256" in m for m in bad), bad
    finally:
        cache.write_bytes(raw)
    assert dvb(r[:9]) == []
    h = _parts(pack, "kin_v1")                                                     # declared cache, built without one: refused
    bad = dvb(r[:9], dag=h[5])
    assert any("lever dagger_label_cache:" in m and "declared" in m for m in bad), bad
    bad = dvb(h[:9], dag=r[5])                                                     # built from a cache, none declared: refused
    assert any("no --dagger-label-cache declared" in m for m in bad), bad


def test_selfcheck_only_with_an_absent_path_is_the_precompute_and_the_next_run_reads_it(pack, tmp_path, monkeypatch):
    cache = tmp_path / "pre" / "kin.npz"                                         # also: the directory is created
    out = tmp_path / "sc1"
    assert TR.main(_run(out, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-label-cache", str(cache), "--selfcheck-only"])) == 0
    assert cache.is_file() and not Path(str(cache) + ".part").exists()
    sc = _selfcheck(out)["info"]["dagger"]["speed_label"]
    assert sc["label_cache"]["read"] is False and sc["label_cache"]["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert set(sc["label_cache"]["header"]) == set(G.LABEL_CACHE_FIELDS)
    _block_labellers(monkeypatch)
    out2 = tmp_path / "train"
    assert TR.main(_run(out2, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--dagger-label-cache", str(cache), "--steps", "3"])) == 0
    monkeypatch.undo()
    lv = _summary(out2)["levers"]["dagger"]
    assert lv["dagger_label_cache"] == str(cache.resolve()) or Path(lv["dagger_label_cache"]).samefile(cache)
    assert lv["dagger_label_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest() == sc["label_cache"]["sha256"]
    sc2 = _selfcheck(out2)["info"]["dagger"]["speed_label"]
    assert sc2["label_cache"]["read"] is True and sc2["census"] == sc["census"] and sc2["status_counts"] == sc["status_counts"]
    h = tmp_path / "plain"                                                       # a run with no flag keeps today's key set
    assert TR.main(_run(h, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--steps", "2"])) == 0
    lvh = _summary(h)["levers"]["dagger"]
    assert lvh["dagger_label_cache"] is None and lvh["dagger_label_cache_sha256"] is None
    assert "label_cache" not in _selfcheck(h)["info"]["dagger"]["speed_label"]
