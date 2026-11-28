"""RALIGN fit and M-ALIGN gate on SYNTHETIC GAP48-shaped token files (written by this test into a tmp dir; no real GAP48 file is opened): fold assignment, NOT-EVALUABLE, OLS on a known affine,
M-ALIGN pass / fail / the same-fold mutants, the refusals, and the file-format knobs. Mutants are shown RED by calling the same checks against a deliberately broken function.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest ralign/tests/test_ralign_fit_gate.py -q --basetemp F:/TanitAD-scratch/ralign_pytest
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import ralign_common as K  # noqa: E402
import ralign_fit as F  # noqa: E402
import ralign_gate as G  # noqa: E402
import ralign_lib as L  # noqa: E402

SMALL = (4, 16)


# ============================================================================================================================ folds
def oracle_fold(sid, digit=0):
    return "A" if int("0123456789abcdef".index(hashlib.sha256(sid.encode()).hexdigest()[digit])) % 2 == 0 else "B"


def test_fold_rule_is_the_cards_and_the_replicate_uses_the_second_digit():
    ids = [f"scene-{i}" for i in range(400)]
    assert all(L.fold_of(s) == oracle_fold(s) for s in ids) and all(L.fold_of(s, 1) == oracle_fold(s, 1) for s in ids)
    assert sum(L.fold_of(s) == "A" for s in ids) in range(150, 250)                 # even hex digits: 8 of 16
    assert any(L.fold_of(s, 0) != L.fold_of(s, 1) for s in ids), "digit 1 must be able to swap folds (section 6.1)"
    h = hashlib.sha256(b"abc").hexdigest()                                          # ba7816bf...: 'b' = 11, odd -> B; second digit 'a' = 10 -> A
    assert h[0] == "b" and L.fold_of("abc") == "B" and L.fold_of("abc", 1) == "A"
    assert L.other_fold("A") == "B" and L.other_fold("B") == "A"
    with pytest.raises(ValueError, match="RALIGN_BAD_FOLD_DIGIT"):
        L.fold_of("abc", 2)
    with pytest.raises(ValueError, match="RALIGN_BAD_SCENE_ID"):
        L.fold_of("")


def test_fold_rule_mutants_are_distinguishable_from_the_card():
    ids = [f"scene-{i}" for i in range(200)]
    last_digit = lambda s: "A" if int(hashlib.sha256(s.encode()).hexdigest()[-1], 16) % 2 == 0 else "B"        # noqa: E731  MUTANT: the last digit
    inverted = lambda s: L.other_fold(L.fold_of(s))                                                              # noqa: E731  MUTANT: A and B swapped
    assert any(last_digit(s) != L.fold_of(s) for s in ids) and all(inverted(s) != L.fold_of(s) for s in ids)


# ============================================================================================================================ OLS
def test_ols_recovers_a_known_affine_and_matches_polyfit():
    g = torch.Generator().manual_seed(1)
    n = 600
    a_true = torch.linspace(-1.5, 2.0, SMALL[0] * SMALL[1], dtype=torch.float32).reshape(SMALL)
    b_true = torch.linspace(0.7, -0.4, SMALL[0] * SMALL[1], dtype=torch.float32).reshape(SMALL)
    R = torch.randn((n, *SMALL), generator=g) * 3.0
    X = a_true * R + b_true + 0.01 * torch.randn((n, *SMALL), generator=g)
    st = L.Stats(SMALL)
    st.add(R[:250], X[:250])
    st.add(R[250:], X[250:])                                                         # accumulation over two "scenes" equals one pass
    a, b, info = L.solve(st)
    assert a.dtype == torch.float32 and float((a - a_true).abs().max()) < 5e-3 and float((b - b_true).abs().max()) < 5e-3, info
    for (p, c) in ((0, 0), (2, 7), (3, 15)):
        k = np.polyfit(R[:, p, c].double().numpy(), X[:, p, c].double().numpy(), 1)
        assert abs(float(a[p, c]) - k[0]) < 1e-5 and abs(float(b[p, c]) - k[1]) < 1e-5, (p, c)
    assert info["n_degenerate"] == 0 and info["min_count"] == n
    wrong = L.Stats(SMALL)                                                           # RED: a fit on the swapped roles does NOT recover (a, b)
    wrong.add(X, R)
    aw, _, _ = L.solve(wrong)
    assert float((aw - a_true).abs().max()) > 0.1


def test_zero_variance_cell_is_degenerate_and_counted_not_silent():
    R = torch.randn((300, *SMALL))
    R[:, 1, 3] = 0.5                                                                  # constant render channel
    X = 2 * R + 1
    st = L.Stats(SMALL)
    st.add(R, X)
    a, b, info = L.solve(st)
    assert info["n_degenerate"] == 1 and float(a[1, 3]) == 0.0 and abs(float(b[1, 3]) - 2.0) < 1e-5     # intercept-only OLS: x = 2 * 0.5 + 1 = 2


# ============================================================================================================================ NOT-EVALUABLE
def tokens_for(tmp_path, n_a, n_b, ticks, shape=SMALL, name="tok", **kw):
    ids = K.scene_ids(n_a, n_b)
    pairs = K.synth_pairs(ids, ticks, shape, **kw)
    d = str(tmp_path / name)
    K.write_token_dir(pairs, d)
    return d, ids, pairs


def test_not_evaluable_below_200_pairs_per_cell_and_fitted_at_exactly_200(tmp_path):
    d, ids, _ = tokens_for(tmp_path, 2, 2, 99)                                       # 198 pairs per fold
    res, rc = F.fit(d, str(tmp_path / "o1"), expect_scenes=4, shape=SMALL)
    assert rc == F.EXIT_NOT_EVALUABLE and res["status"] == "NOT-EVALUABLE" and len(res["reasons"]) == 2
    assert not any(f.startswith("MAP_") for f in os.listdir(tmp_path / "o1")), "a map file was written for a NOT-EVALUABLE fit"
    d2, _, _ = tokens_for(tmp_path, 2, 2, 100, name="tok2")                          # 200 per fold: the boundary is inclusive
    res2, rc2 = F.fit(d2, str(tmp_path / "o2"), expect_scenes=4, shape=SMALL)
    assert rc2 == 0 and res2["status"] == "FITTED" and res2["folds"]["A"]["ols"]["min_count"] == 200
    R, X = torch.randn((200, *SMALL)), torch.randn((200, *SMALL))                    # at the Stats level a non-finite pair is dropped PER CELL: one NaN leaves that cell with 199 -> NOT EVALUABLE
    st = L.Stats(SMALL)
    st.add(R, X)
    assert L.solve(st)[2]["min_count"] == 200
    R[0, 1, 1] = float("nan")
    st = L.Stats(SMALL)
    st.add(R, X)
    with pytest.raises(ValueError, match="RALIGN_NOT_EVALUABLE 1 of 64 cells have fewer than 200 pairs"):
        L.solve(st)


def test_non_finite_tokens_are_refused_by_name(tmp_path):
    d, ids, pairs = tokens_for(tmp_path, 2, 2, 100)
    sid = ids[0]
    r, x = pairs[sid]
    r = r.clone()
    r[0, 0, 0] = float("nan")
    K.write_token_dir({sid: (r, x)}, d)
    with pytest.raises(ValueError, match="RALIGN_NONFINITE_PAIRS"):
        F.fit(d, str(tmp_path / "o"), expect_scenes=4, shape=SMALL)


# ============================================================================================================================ the fit end to end, the files, the refusals
def test_fit_writes_maps_with_hashes_and_the_fold_lists_and_recovers_the_affine(tmp_path):
    d, ids, _ = tokens_for(tmp_path, 3, 3, 100, a_true=0.6, b_true=0.3, noise=0.02)
    out = str(tmp_path / "fit")
    assert F.main(["fit", "--tokens-dir", d, "--out-dir", out, "--expect-scenes", "6", "--shape", "4,16"]) == 0
    idx = json.load(open(f"{out}/MAPS.json"))
    fr = json.load(open(f"{out}/FIT_RESULT.json"))
    assert idx["card_sha256"] == L.CARD_SHA256 and idx["applied_map_for_scene_in_fold"] == {"A": "B", "B": "A"} and fr["status"] == "FITTED"
    for f in "AB":
        m = L.load_map(f"{out}/MAP_{f}.pt", idx["maps"][f]["file_sha256"], idx["maps"][f]["tensor_sha256"], SMALL)
        assert m.a.dtype == torch.float32 and m.fold == f and sorted(m.meta["fold_scenes"]) == sorted(s for s in ids if L.fold_of(s) == f) == idx["maps"][f]["fold_scenes"]
        assert abs(float(m.a.mean()) - 0.6) < 0.02 and abs(float(m.b.mean()) - 0.3) < 0.02            # fp8-quantised synthetic tokens recover the generating affine
    assert set(idx["maps"]["A"]["fold_scenes"]).isdisjoint(idx["maps"]["B"]["fold_scenes"]) and len(fr["sources"]) == 6 and all(len(s["sha256"]) == 64 for s in fr["sources"])
    assert F.main(["fit", "--tokens-dir", d, "--out-dir", out, "--expect-scenes", "6", "--shape", "4,16"]) == F.EXIT_REFUSED, "an existing out-dir must be refused"


def test_fit_refusals_are_named(tmp_path):
    d, ids, pairs = tokens_for(tmp_path, 2, 2, 100)
    with pytest.raises(ValueError, match="RALIGN_SCENE_COUNT"):
        F.fit(d, str(tmp_path / "a"), expect_scenes=48, shape=SMALL)
    sid = ids[0]
    d2 = str(tmp_path / "ident")
    K.write_token_dir({**pairs, sid: (pairs[sid][0], pairs[sid][0])}, d2)                # a scene whose render equals its real tokens
    with pytest.raises(ValueError, match="RALIGN_IDENTITY_SCENE"):
        F.fit(d2, str(tmp_path / "b"), expect_scenes=4, shape=SMALL)
    d3 = str(tmp_path / "badname")
    K.write_token_dir(pairs, d3)
    os.replace(f"{d3}/{sid[:8]}.pt", f"{d3}/zzzzzzzz.pt")
    with pytest.raises(ValueError, match="RALIGN_FILE_NAME"):
        F.fit(d3, str(tmp_path / "c"), expect_scenes=4, shape=SMALL)
    sl = tmp_path / "scenes.json"
    sl.write_text(json.dumps({"scenes": [{"scene": s} for s in ids[1:]] + [{"scene": "x" * 36}]}))
    with pytest.raises(ValueError, match="RALIGN_SCENE_LIST_MISMATCH"):
        F.fit(d, str(tmp_path / "e"), expect_scenes=4, scene_list=str(sl), shape=SMALL)
    with pytest.raises(ValueError, match="RALIGN_PAIR_SHAPE"):
        F.fit(d, str(tmp_path / "f"), expect_scenes=4, shape=(4, 17))


def test_the_token_file_keys_and_dtype_are_configurable(tmp_path):
    ids = K.scene_ids(2, 2)
    pairs = K.synth_pairs(ids, 100, SMALL)
    d = str(tmp_path / "alt")
    K.write_token_dir(pairs, d, scene_key="sid", render_key="r", real_key="x")
    with pytest.raises(ValueError, match="RALIGN_PAIR_KEY_MISSING"):
        F.fit(d, str(tmp_path / "o1"), expect_scenes=4, shape=SMALL)
    res, rc = F.fit(d, str(tmp_path / "o2"), expect_scenes=4, shape=SMALL, keys={"scene_key": "sid", "render_key": "r", "real_key": "x"})
    assert rc == 0
    d32 = str(tmp_path / "f32")
    K.write_token_dir({s: (r.float(), x.float()) for s, (r, x) in pairs.items()}, d32)           # fp32 files are read as well
    assert F.fit(d32, str(tmp_path / "o3"), expect_scenes=4, shape=SMALL)[1] == 0


def test_identity_pair_writer(tmp_path):
    idx = F.write_identity(str(tmp_path / "id"), SMALL)
    assert idx["status"] == "IDMAP"
    for f in "AB":
        m = L.load_map(f"{tmp_path}/id/MAP_{f}.pt", idx["maps"][f]["file_sha256"], idx["maps"][f]["tensor_sha256"], SMALL)
        assert m.is_identity and bool(torch.signbit(m.b).all())


# ============================================================================================================================ M-ALIGN
def run_gate(tmp_path, d, fit_dir=None, **kw):
    fd = fit_dir or str(tmp_path / "fit_for_gate")
    if fit_dir is None:
        res, rc = F.fit(d, fd, expect_scenes=kw.get("expect", 4), shape=kw.get("shape", SMALL))
        assert rc == 0, res.get("reasons")
    maps, idx = G.load_maps(fd, kw.get("shape", SMALL))
    return G.gate(G.iter_pairs(d, shape=kw.get("shape", SMALL)), maps, idx, expect_scenes=kw.get("expect", 4)), fd


def offset_pairs(ids, ticks, shape, seed=0):
    """real = 0.6 * render + b_pattern + noise with a large fixed per-cell pattern b: the raw render-vs-real cosine is low (~0.5) and a held-out map recovers it (~1.0)."""
    g = torch.Generator().manual_seed(seed)
    b = 2.0 * torch.randn(shape, generator=g)
    out = {}
    for sid in ids:
        r = torch.randn((ticks, *shape), generator=g) * 2.0
        out[sid] = (K.fp8_like(r), K.fp8_like(0.6 * r + b + 0.05 * torch.randn((ticks, *shape), generator=g)))
    return out


def test_m_align_passes_when_a_held_out_map_recovers_the_real_tokens(tmp_path):
    ids = K.scene_ids(6, 6)
    d = str(tmp_path / "t")
    K.write_token_dir(offset_pairs(ids, 40, SMALL), d)
    res, _ = run_gate(tmp_path, d, expect=12)
    m = res["main"]
    assert res["verdict"] == G.V_PASS and res["gate_valid"], res["verdict"]
    assert m["median_before"] < 0.7 and m["median_after"] > 0.97 and m["rise"] > 0.3 and m["pass_rise"] and m["pass_level"]
    assert res["regression_arm"]["flagged"] and res["regression_arm"]["n_leak_scenes"] == 12 and m["leaks"] == [] and res["inconsistent_fold_lists"] == []
    lo, hi = m["ci_scene_cluster_B2000_seed0"]["rise"]
    assert lo <= m["rise"] <= hi


def test_m_align_fails_the_mechanism_when_render_and_real_are_unrelated(tmp_path):
    ids = K.scene_ids(6, 6)
    g = torch.Generator().manual_seed(4)
    pairs = {s: (K.fp8_like(torch.randn((40, *SMALL), generator=g)), K.fp8_like(torch.randn((40, *SMALL), generator=g))) for s in ids}
    d = str(tmp_path / "t")
    K.write_token_dir(pairs, d)
    res, _ = run_gate(tmp_path, d, expect=12)
    assert res["verdict"] == G.V_FAIL == "ALIGN-FAILS-MECH" and not res["main"]["pass_rise"] and not res["main"]["pass_level"] and res["gate_valid"]


def test_m_align_needs_both_the_rise_and_the_level(tmp_path):
    """A rise of >= 0.05 with a level below 0.90 is a FAIL; the thresholds are the card's, not tunable from the command line."""
    ids = K.scene_ids(6, 6)
    g = torch.Generator().manual_seed(2)
    b = 2.0 * torch.randn(SMALL, generator=g)
    pairs = {s: (K.fp8_like(r := torch.randn((40, *SMALL), generator=g) * 2.0), K.fp8_like(0.6 * r + b + 3.0 * torch.randn((40, *SMALL), generator=g))) for s in ids}      # strong noise: after ~ 0.6
    d = str(tmp_path / "t")
    K.write_token_dir(pairs, d)
    res, _ = run_gate(tmp_path, d, expect=12)
    m = res["main"]
    assert m["pass_rise"] and not m["pass_level"] and res["verdict"] == G.V_FAIL, m
    assert G.RISE_MIN == 0.05 if hasattr(G, "RISE_MIN") else (L.RISE_MIN == 0.05 and L.LEVEL_MIN == 0.90)
    assert "--rise-min" not in open(G.__file__, encoding="utf-8").read().split("def main")[1]


def scene_specific_pairs(ids, ticks, shape):
    """real = a scene-specific fixed pattern + noise; the render is unrelated noise. Nothing transfers between scenes, but a map fitted on the scene's OWN fold memorises it."""
    g = torch.Generator().manual_seed(9)
    out = {}
    for sid in ids:
        c = 3.0 * torch.randn(shape, generator=g)
        out[sid] = (K.fp8_like(torch.randn((ticks, *shape), generator=g)), K.fp8_like(c + 0.05 * torch.randn((ticks, *shape), generator=g)))
    return out


def test_same_fold_map_is_flagged_and_would_have_passed_an_unguarded_gate(tmp_path):
    ids = K.scene_ids(1, 1)                                                          # one scene per fold: the other fold's map knows nothing about it, its own fold's map IS it
    d = str(tmp_path / "t")
    K.write_token_dir(scene_specific_pairs(ids, 200, SMALL), d)
    res, fd = run_gate(tmp_path, d, expect=2)
    assert res["verdict"] == G.V_FAIL and res["gate_valid"]                          # honest result: nothing transfers
    reg = res["regression_arm"]
    assert reg["flagged"] and reg["naive"]["would_pass_unguarded"] and reg["naive"]["median_after"] > 0.99, reg


def test_gate_is_broken_when_the_provenance_check_is_disabled_and_invalid_when_main_reads_the_own_fold(tmp_path, monkeypatch):
    ids = K.scene_ids(1, 1)
    d = str(tmp_path / "t")
    K.write_token_dir(scene_specific_pairs(ids, 200, SMALL), d)
    _, fd = run_gate(tmp_path, d, expect=2)
    maps, idx = G.load_maps(fd, SMALL)
    orig = G.score_scene

    def no_provenance(*a, **k):                                                       # MUTANT: the leak flags are never raised
        r = orig(*a, **k)
        r["leak_cross"] = r["leak_same"] = False
        return r
    monkeypatch.setattr(G, "score_scene", no_provenance)
    assert G.gate(G.iter_pairs(d, shape=SMALL), maps, idx, expect_scenes=2)["verdict"] == G.V_BROKEN
    monkeypatch.setattr(G, "score_scene", orig)
    real_arm = G.arm

    def main_reads_own_fold(recs, key, leak_key, *a, **k):                           # MUTANT: the MAIN arm scored with the same-fold map
        return real_arm(recs, "after_same" if key == "after_cross" else key, "leak_same" if key == "after_cross" else leak_key, *a, **k)
    monkeypatch.setattr(G, "arm", main_reads_own_fold)
    res = G.gate(G.iter_pairs(d, shape=SMALL), maps, idx, expect_scenes=2)
    assert res["main"]["pass_rise"] and res["main"]["pass_level"], "the leak should have looked like a pass"
    assert res["verdict"] == G.V_LEAK and not res["gate_valid"], res["verdict"]


def test_gate_reports_fold_lists_that_do_not_contain_the_scene(tmp_path):
    ids = K.scene_ids(3, 3)
    d = str(tmp_path / "t")
    K.write_token_dir(offset_pairs(ids, 70, SMALL), d)
    _, fd = run_gate(tmp_path, d, expect=6)
    maps, idx = G.load_maps(fd, SMALL)
    maps["A"].meta["fold_scenes"] = [s for s in maps["A"].meta["fold_scenes"] if s != ids[0]]
    res = G.gate(G.iter_pairs(d, shape=SMALL), maps, idx, expect_scenes=6)
    assert res["verdict"] == G.V_FOLDS and ids[0] in res["inconsistent_fold_lists"]


def test_gate_refuses_a_tampered_map_and_an_existing_output(tmp_path):
    ids = K.scene_ids(2, 2)
    d = str(tmp_path / "t")
    K.write_token_dir(offset_pairs(ids, 100, SMALL), d)
    _, fd = run_gate(tmp_path, d, expect=4)
    p = f"{fd}/MAP_A.pt"
    b = bytearray(open(p, "rb").read())
    b[len(b) // 2] ^= 0x01
    open(p, "wb").write(bytes(b))
    with pytest.raises(ValueError, match="RALIGN_MAP_HASH_MISMATCH"):
        G.load_maps(fd, SMALL)
    out = tmp_path / "M_ALIGN.json"
    out.write_text("{}")
    assert G.main(["--tokens-dir", d, "--map-dir", fd, "--out", str(out), "--expect-scenes", "4", "--shape", "4,16"]) == 4


def test_the_gate_cosine_is_the_policy_side_cosine(tmp_path):
    import t2a_v8f_encoder as E
    g = torch.Generator().manual_seed(3)
    R, X = torch.randn((5, *SMALL), generator=g), torch.randn((5, *SMALL), generator=g)
    got = G.cosines(R, X)
    ref = np.array([float((R[i].double().flatten() @ X[i].double().flatten()) / (R[i].double().norm() * X[i].double().norm())) for i in range(5)])
    assert np.abs(got - ref).max() < 1e-12 and all(got[i] == E.cosine(R[i], X[i]) for i in range(5))


def test_cli_end_to_end_at_the_cards_shape_40x1024(tmp_path):
    ids = K.scene_ids(24, 24)
    pairs = offset_pairs(ids, 10, L.SHAPE, seed=3)
    d = str(tmp_path / "tokens")
    K.write_token_dir(pairs, d)
    fit_dir, out = str(tmp_path / "fit"), str(tmp_path / "M_ALIGN.json")
    assert F.main(["fit", "--tokens-dir", d, "--out-dir", fit_dir]) == 0                             # 48 scenes, 240 pairs per fold: >= 200 per cell
    assert G.main(["--tokens-dir", d, "--map-dir", fit_dir, "--out", out]) == 0
    r = json.load(open(out))
    assert r["verdict"] == "M-ALIGN-PASS" and r["main"]["n_scenes"] == 48 and r["main"]["n_pairs"] == 480 and r["regression_arm"]["flagged"] and r["gate_valid"]
    assert r["card_sha256"] == L.CARD_SHA256 and set(r["maps"]) == {"A", "B"}
