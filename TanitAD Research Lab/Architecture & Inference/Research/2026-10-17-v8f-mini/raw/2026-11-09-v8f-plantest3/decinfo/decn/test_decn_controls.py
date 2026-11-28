"""PREREG_V8F_DECN controls 1 to 6 with their deliberate mutations (IMPL_CHOICES_DECN.md section 6). CPU. Run through test_run.cmd."""
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch

import decn_lib as DN
import run_decn as RN
import select_extra as SE
from decn_lib import DL


def _syn(n_ep=40, per=6, frames=16, dim=32, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    eid = np.repeat([f"e{i:03d}" for i in range(n_ep)], per).astype("U36")
    n = len(eid)
    X = rng.standard_normal((n, frames, dim)).astype(np.float32)
    w = rng.standard_normal(dim)
    y = X @ w / np.sqrt(dim) + noise * rng.standard_normal((n, frames))
    return eid, X, y


class _D:
    pass


# ------------------------------------------------------------------------------------ chunked ridge == the gate's ridge
def test_chunked_ridge_equals_ridge_fit():
    rng = np.random.default_rng(0)
    xf = rng.standard_normal((7000, 24)).astype(np.float32) + 3.0
    yf = xf @ rng.standard_normal(24) + rng.standard_normal(7000)
    xh = rng.standard_normal((900, 24)).astype(np.float32)
    ref = DL.ridge_pred(xf, yf, xh)
    got = DN.ridge_pred_chunked(xf, yf, xh, chunk=1000)
    assert np.abs(ref - got).max() < 1e-8 * np.abs(ref).max() + 1e-9
    assert np.abs(ref - DN.ridge_pred_chunked(xf, yf, xh, chunk=7000)).max() < 1e-8 * np.abs(ref).max() + 1e-9


# ------------------------------------------------------------------------------------ control 1: target tie
def _tie(a, b):
    r2 = float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()); mad = float(np.abs(a - b).mean())
    return r2, mad, bool(r2 >= 0.99 and mad <= 0.10)


def test_c1_cache_v_ties_to_eval_target_on_982_tune_windows_and_mutation_fails():
    D, sp = DL.load_tune(read_feats=False)
    L = torch.load(DL.EVAL, weights_only=False); eid_all = np.array(L["eid"]); tt = np.asarray(L["t"])
    a, b, b_shift, vs = [], [], [], {}
    for j, di in enumerate(D.ds_idx):
        e = str(eid_all[di]); g = int(tt[di]) - 15
        if e not in vs:
            vs[e] = torch.load(f"{DL.W.SF}/cache/eval/clips/{e}.pt", weights_only=True)["v"].numpy().astype(np.float64)
        for k in range(16):
            if np.isfinite(D.vk[j, k]) and 1 <= g + k < len(vs[e]):
                a.append(D.vk[j, k]); b.append(vs[e][g + k]); b_shift.append(vs[e][g + k - 1])
    assert D.n == 982
    r2, mad, ok = _tie(np.array(a), np.array(b))
    print("control 1", r2, mad, len(a))
    assert ok
    # mutation: the source read one cache frame early leaves the tolerance
    assert not _tie(np.array(a), np.array(b_shift))[2]


# ------------------------------------------------------------------------------------ control 2: reproduction
@pytest.mark.parametrize("tag", ["M2b_s0", "M2b_s1"])
def test_c2_reproduction_of_decinfo_at_n53(tag):
    D, sp = DL.load_tune(read_feats=False)
    X = D.z1_ref[tag].numpy().astype(np.float32)
    runs = json.load(open(f"{DL.HERE}/runs/R0_{tag}.json", encoding="utf-8"))["rows"]
    pre = {t: [r["r2"] for r in runs if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == t][0] for t in ("speed", "yaw")}
    for t, y in (("speed", D.vk), ("yaw", D.yk)):
        r, _, _ = DN.ridge_read_c(D, X, y, y)
        print(tag, t, r, pre[t])
        assert abs(r - pre[t]) <= 0.01
    # mutation: targets shifted by one frame leave the tolerance
    rs, _, _ = DN.ridge_read_c(D, X[:, :-1], D.vk[:, 1:], D.vk[:, 1:])
    ry, _, _ = DN.ridge_read_c(D, X[:, :-1], D.yk[:, 1:], D.yk[:, 1:])
    assert abs(rs - pre["speed"]) > 0.01 or abs(ry - pre["yaw"]) > 0.01


# ------------------------------------------------------------------------------------ control 3: shuffled targets
def test_c3_shuffled_targets_at_or_below_005_and_mutation_fails():
    eid, X, y = _syn()
    D = _D(); D.nf, D.n = 160, len(eid)
    real, _, _ = DN.ridge_read_c(D, X, y, y)
    assert real > 0.6
    ysh = y.copy(); ysh[:D.nf] = DL.derange_tracks(y[:D.nf], eid[:D.nf], 5)
    r_sh, _, _ = DN.ridge_read_c(D, X, y, ysh)
    assert r_sh <= 0.05, r_sh
    assert not (real <= 0.05)                                      # mutation: no permutation reads at the true level


def test_c3_mlp_shuffled_target_reads_at_or_below_005():
    eid, X, y = _syn(n_ep=60, per=10, dim=16)
    D = _D(); D.nf, D.n, D.eid = 400, len(eid), eid
    D.es = np.array([e in DL.es_ids_of(set(eid[:400])) for e in eid]) & (np.arange(len(eid)) < 400)
    r, _ = DN.mlp_read_c(D, X, y, y, 30)
    ysh = y.copy(); ysh[:D.nf] = DL.derange_tracks(y[:D.nf], eid[:D.nf], 5)
    rs, _ = DN.mlp_read_c(D, X, y, ysh, 30)
    assert r > 0.5 and rs <= 0.05, (r, rs)


# ------------------------------------------------------------------------------------ control 4: planted signal at N1053 size
@pytest.mark.parametrize("dim,n_samples", [(128, 220000), (256, 220000), (512, 170000), (1024, 100000)])
def test_c4_planted_signal_recovered_at_n1053_size_and_zero_features_read_zero(dim, n_samples):
    rng = np.random.default_rng(dim)
    Xf = rng.standard_normal((n_samples, dim)).astype(np.float32)
    Xh = rng.standard_normal((3800, dim)).astype(np.float32)
    a, z = DN.planted_check_c(Xf, Xh)
    assert a >= 0.80 and z <= 0.02, (a, z)
    yf, yh = DL.planted_target(Xf, Xh)
    assert DL.r2(yh, DN.ridge_pred_chunked(Xf, yf[rng.permutation(len(yf))], Xh)) < 0.80          # mutation 1: target not a function of the features
    assert DL.r2(yh, DN.ridge_pred_chunked(Xf, yf, Xh)) > 0.02                                    # mutation 2: unzeroed features read the signal


# ------------------------------------------------------------------------------------ control 5: split
def test_c5_guard_mutations_raise():
    fit, held, es = {"a", "b", "c"}, {"d", "e"}, {"a"}
    ev, tr = {"x", "y"}, {"t1", "t2"}
    assert DN.split_guard_n(fit | tr, held, es, ev, tr)
    with pytest.raises(AssertionError):
        DN.split_guard_n(fit | tr | {"d"}, held, es, ev, tr)                # held-out episode in the fit set
    with pytest.raises(AssertionError):
        DN.split_guard_n(fit | tr, held, es, ev, tr | {"x"})                # eval clip among the additions
    with pytest.raises(AssertionError):
        DN.split_guard_n(fit | tr, held, es, ev, tr | {"d"})                # held-out episode among the additions
    with pytest.raises(AssertionError):
        DN.split_guard_n(fit | tr, held, es | {"d"}, ev, tr)               # early-stopping episode outside the fit set


def _tiny_world(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    nf, nh, C = 60, 20, 8
    eid = np.array([f"f{i // 10}" for i in range(nf)] + [f"h{i // 10}" for i in range(nh)], dtype="U36")
    X = rng.standard_normal((nf + nh, 16, C)).astype(np.float32)
    w = rng.standard_normal(C)
    v = X @ w + 0.1 * rng.standard_normal((nf + nh, 16))
    T = dict(eid=eid, vk=v, yk=-v, nf=np.array(nf), es=np.array([e in DL.es_ids_of({f"f{i}" for i in range(6)}) for e in eid]) & (np.arange(nf + nh) < nf),
             fit_eps=np.array(6), held_eps=np.array(2), z1_t=X)
    order = [f"c{i}" for i in range(4)]
    monkeypatch.setattr(DN, "LAT", str(tmp_path))
    for c in order:
        Xc = rng.standard_normal((5, 16, C)).astype(np.float32); vc = Xc @ w
        np.savez(os.path.join(str(tmp_path), f"{c}.npz"), z1_t=Xc, v=vc, yaw=-vc)
    return T, order


def test_c5_held_out_block_identical_across_n_and_fit_never_holds_it(tmp_path, monkeypatch):
    T, order = _tiny_world(tmp_path, monkeypatch)
    X_tr, ci = DN.load_train_arrays(order, 4, "z1_t")
    tt = RN.train_targets(order, 4)
    digs, X_h = [], []
    for k in (0, 2, 4):
        D, X, ids = DN.assemble(T, "z1_t", X_tr, ci, order, k, tt)
        digs.append(D.held_digest); X_h.append(X[D.nf:])
        assert D.nf == 60 + 5 * k and D.n == D.nf + 20 and D.n_fit_episodes == 6 + k
        assert set(D.eid[:D.nf]).isdisjoint(set(D.eid[D.nf:]))
        assert DN.split_guard_n(set(D.eid[:D.nf]), set(D.eid[D.nf:]), DL.es_ids_of(set(D.eid[:D.nf])), {"zz"}, ids)
        assert not D.es[D.nf:].any()
    assert len(set(digs)) == 1 and np.array_equal(X_h[0], X_h[1]) and np.array_equal(X_h[0], X_h[2])
    # mutation: a changed held-out target changes the digest
    T2 = dict(T); vk2 = T["vk"].copy(); vk2[70, 3] += 1.0; T2["vk"] = vk2
    D2, _, _ = DN.assemble(T2, "z1_t", X_tr, ci, order, 2, tt)
    assert D2.held_digest != digs[0]


def test_c5_real_split_matches_the_card():
    D, sp = DL.load_tune(read_feats=False)
    assert (len(sp["fit_ids"]), len(sp["held_ids"]), len(sp["fit"]), len(sp["held"])) == (53, 17, 744, 238)
    assert set(sp["tune"]).isdisjoint(set(sp["score"])) and set(sp["fit"]).isdisjoint(set(sp["held"]))
    ev = {f[:-3] for f in os.listdir(f"{DL.W.SF}/cache/eval/clips") if f.endswith(".pt")}
    c141, ce = DN.clip_lists()
    assert not (set(ce["clips"]) & ev) and not (set(c141["clips"]) & ev)
    assert not (set(ce["clips"]) & set(c141["clips"])) and len(ce["clips"]) == 859 and len(set(ce["clips"])) == 859
    assert not (set(ce["clips"]) & set(sp["fit_ids"]) | set(ce["clips"]) & set(sp["held_ids"]))


# ------------------------------------------------------------------------------------ the selection (card: salt, lowest sha256, excluded sets)
def test_selection_file_is_the_rule_and_is_pinned():
    c141, ce = DN.clip_lists()
    h = hashlib.sha256(open(os.path.join(HERE, "clips_extra.json"), "rb").read()).hexdigest()
    assert open(os.path.join(HERE, "clips_extra.sha256")).read().split()[0] == h
    ents = {json.loads(l)["clip"]: json.loads(l) for l in open(f"{DL.THEORY}/wmact/train_stream/box2_meta/manifest.jsonl", encoding="utf-8") if l.strip()}
    ev = sorted({f[:-3] for f in os.listdir(f"{DL.W.SF}/cache/eval/clips") if f.endswith(".pt")})
    pick, _, _ = SE.select(ents, c141["clips"], ev)
    assert pick == ce["clips"]
    assert all(ce["md5"][c] == ents[c]["md5"] for c in pick)
    # mutation: a different salt gives a different list
    assert SE.select(ents, c141["clips"], ev, salt="decn-v2")[0] != pick
    # mutation: an eval clip inside the manifest is refused
    bad = dict(ents); bad[ev[0]] = {"clip": ev[0]}
    with pytest.raises(AssertionError):
        SE.select(bad, c141["clips"], ev)


# ------------------------------------------------------------------------------------ control 6: clip integrity
def test_c6_md5_mismatch_raises_and_match_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(RN, "STREAM", str(tmp_path))
    payload = b"clip bytes" * 1000

    def fake_run(args, cwd=None, **kw):
        dest = os.path.join(cwd, RN.DEST_REL)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(fake_run.data)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(RN.time, "sleep", lambda s: None)
    fake_run.data = payload
    RN.pull("c", hashlib.md5(payload).hexdigest())
    fake_run.data = payload[:-1] + b"X"                                   # mutation: one byte differs
    with pytest.raises(RuntimeError, match="MD5"):
        RN.pull("c", hashlib.md5(payload).hexdigest())


def test_c6_network_failure_is_a_network_stop_not_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(RN, "STREAM", str(tmp_path))
    monkeypatch.setattr(RN.time, "sleep", lambda s: None)

    def boom(*a, **k):
        raise subprocess.CalledProcessError(255, "sftp")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ConnectionError):
        RN.pull("c", "0" * 32)


# ------------------------------------------------------------------------------------ the unit reader end to end on a tiny world
def test_read_unit_runs_end_to_end_and_reads_the_planted_relation(tmp_path, monkeypatch):
    T, order = _tiny_world(tmp_path, monkeypatch)
    X_tr, ci = DN.load_train_arrays(order, 4, "z1_t")
    tt = RN.train_targets(order, 4)
    D, X, ids = DN.assemble(T, "z1_t", X_tr, ci, order, 4, tt)
    monkeypatch.setattr(DN, "CAP_BIG", 4)
    rows, planted = DN.read_unit(D, X, "R0", "t", "N194", lambda *a: None)
    assert len(rows) == 8 and len(planted) == 2
    ridge = [r for r in rows if r["reader"] == "ridge" and r["m"] == 1 and r["target"] == "speed"][0]
    assert ridge["r2"] > 0.8 and ridge["r2_shuffled"] <= 0.05
    assert ridge["n_fit"] == D.nf * 16 and ridge["n_held"] == 20 * 16
    assert all(r["detail"]["cap"] == 4 for r in rows if r["reader"] == "mlp")


def test_stack_alignment():
    X = np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)
    S = DL.stack_ch(X, 4)
    assert S.shape == (2, 13, 12) and np.array_equal(S[:, 0, 0:3], X[:, 3]) and np.array_equal(S[:, -1, 9:12], X[:, 12])
