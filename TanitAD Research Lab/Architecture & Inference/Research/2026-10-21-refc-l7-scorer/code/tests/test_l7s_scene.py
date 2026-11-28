"""Tests for the L7-S scene-token path (scene_bank.py + train_l7.py --mode S).  Every expectation is a LITERAL (token channels 0 / 1 carry the
window's own (clip_index, ws), so a join error is readable as a number the test wrote down, not as an expression over the code under test),
and every guard has a deliberate-regression MUTANT that must turn its check RED.  No optimizer step is taken anywhere in this file; the
``--epochs 0`` dry run constructs the optimizer object and stops, exactly as the PILOT's dry run did.

    python run_isolated.py --stack <stack_min> -m pytest tests/test_l7s_scene.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import refc_scorer as RS  # noqa: E402
import scene_bank as SB  # noqa: E402
import train_l7 as TL  # noqa: E402

N_TOK = {"bev": 6, "agent": 5}
DIMS = {"bev": 8, "agent": 12}


# ------------------------------------------------------------------------------------------------------ fixtures
def tokens(ci, ws, name, seed=0):
    """[n, n_tok, dim] fp16; channel 0 = clip_index, channel 1 = ws (exact in fp16 below 2048), the rest noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(ci), N_TOK[name], DIMS[name])).astype(np.float16)
    a[:, :, 0] = np.asarray(ci)[:, None]
    a[:, :, 1] = np.asarray(ws)[:, None]
    return a


def canonical():
    """3 clips x 4 windows: clip_index 0,0,0,0,1,1,1,1,2,2,2,2 ; ws 7,12,17,22 per clip (the real banks' ws grid: 5 apart, t = ws - 7)."""
    ci = np.repeat(np.arange(3), 4).astype(np.int64)
    ws = np.tile(7 + 5 * np.arange(4), 3).astype(np.int64)
    return ci, ws


def make_dump(root, ci, ws, *, t=None, perm=None, scene_ci=None, scene_ws=None, with_bev=True, with_scene=True, stored=False,
              n_scene_rows=None):
    """Write ``<root>/dump/{fan_extras,l7cap_scene}.npz``.  The scene rows are written in ``perm`` order WITH their own labels (a consistent
    re-ordering); ``scene_ci`` / ``scene_ws`` override the labels (an inconsistent bank)."""
    d = os.path.join(root, "dump")
    os.makedirs(d, exist_ok=True)
    fan = os.path.join(d, "fan_extras.npz")
    np.savez(fan, clip_index=ci, ws=ws, t=(ws - 7) if t is None else t)
    if with_scene:
        p = np.arange(len(ci)) if perm is None else np.asarray(perm)
        sci = (ci[p] if scene_ci is None else scene_ci)
        sws = (ws[p] if scene_ws is None else scene_ws)
        arrs = {"clip_index": sci, "ws": sws, "r7_agent_tokens": tokens(ci[p], ws[p], "agent"),
                "r7_agent_pad": np.zeros((len(p), N_TOK["agent"]), bool)}
        if with_bev:
            arrs["r7_bev_tokens"] = tokens(ci[p], ws[p], "bev")
        if n_scene_rows is not None:
            arrs = {k: v[:n_scene_rows] if k in ("r7_agent_tokens", "r7_bev_tokens") else v for k, v in arrs.items()}
        (np.savez if stored else np.savez_compressed)(os.path.join(d, "l7cap_scene.npz"), **arrs)
    return fan


def fanz_of(ci, ws):
    return {"bank_of_row": np.zeros(len(ci), np.int64), "clip_index": ci, "ws": ws}


def check_batch_is_own_window(sb, ci, ws, rows) -> list:
    """The literal join check: the tokens the scorer is given for fan row r carry channel 0 = ci[r] and channel 1 = ws[r], for BOTH sources."""
    b = sb.batch(np.asarray(rows))
    fails = []
    for name in ("bev", "agent"):
        a = b[name + "_tokens"].numpy()
        for j, r in enumerate(rows):
            if not (np.all(a[j, :, 0] == ci[r]) and np.all(a[j, :, 1] == ws[r])):
                fails.append(f"{name}: fan row {r} (clip {ci[r]}, ws {ws[r]}) got tokens of clip {a[j, 0, 0]:.0f}, ws {a[j, 0, 1]:.0f}")
    return fails


# ------------------------------------------------------------------------------------------------------ unpack
def test_unpack_is_byte_identical_to_the_zip_member_and_content_addressed():
    ci, ws = canonical()
    with tempfile.TemporaryDirectory() as td:
        make_dump(td, ci, ws)
        npz = os.path.join(td, "dump", "l7cap_scene.npz")
        cache = os.path.join(td, "cache")
        p = SB.unpack_member(npz, "r7_agent_tokens", cache)
        with zipfile.ZipFile(npz) as z:
            want = z.read("r7_agent_tokens.npy")
            info = z.getinfo("r7_agent_tokens.npy")
        assert open(p, "rb").read() == want
        assert os.path.basename(p) == f"r7_agent_tokens.{info.CRC:08x}.{info.file_size}.npy"
        assert SB.unpack_member(npz, "r7_agent_tokens", cache) == p and len(os.listdir(cache)) == 1     # a re-run is a cache hit, not a second file


def test_crc_failures_are_not_cached_and_a_corrupt_cache_file_is_replaced():
    ci, ws = canonical()
    with tempfile.TemporaryDirectory() as td:
        make_dump(td, ci, ws, stored=True)
        npz = os.path.join(td, "dump", "l7cap_scene.npz")
        # (a) a flipped DATA byte inside a STORED member: the unpack must raise and leave no cache file that looks complete
        with zipfile.ZipFile(npz) as z:
            i = z.getinfo("r7_agent_tokens.npy")
        raw = bytearray(open(npz, "rb").read())
        # local header = 30 bytes + name + extra; the npy data starts after the 128-byte npy header
        extra = int.from_bytes(raw[i.header_offset + 28:i.header_offset + 30], "little")
        at = i.header_offset + 30 + len(i.filename) + extra + 128 + 7
        raw[at] ^= 0xFF
        bad = os.path.join(td, "bad.npz")
        open(bad, "wb").write(bytes(raw))
        cache = os.path.join(td, "cache")
        with pytest.raises((zipfile.BadZipFile, SystemExit)):
            SB.unpack_member(bad, "r7_agent_tokens", cache)
        assert not [f for f in os.listdir(cache) if f.endswith(".npy")], "a CRC-failed unpack left a cache file"
        # (b) a corrupt cache file of the right size is debris: verify=True re-extracts it
        good = SB.unpack_member(npz, "r7_agent_tokens", cache)
        want = open(good, "rb").read()
        c = bytearray(want)
        c[200] ^= 0x55
        open(good, "wb").write(bytes(c))
        assert open(SB.unpack_member(npz, "r7_agent_tokens", cache, verify=True), "rb").read() == want
        # regression arm: verify=False trusts the size alone (documented), so it WOULD serve the corrupt bytes
        open(good, "wb").write(bytes(c))
        assert open(SB.unpack_member(npz, "r7_agent_tokens", cache, verify=False), "rb").read() == bytes(c)


# ------------------------------------------------------------------------------------------------------ gather + alignment
def test_gather_returns_the_requested_rows_in_the_requested_order():
    ci, ws = canonical()
    with tempfile.TemporaryDirectory() as td:
        fan = make_dump(td, ci, ws)
        sb = SB.attach(fanz_of(ci, ws), [fan], os.path.join(td, "cache"))
        rows = [9, 2, 11, 0, 5]
        assert check_batch_is_own_window(sb, ci, ws, rows) == []
        assert sb.dims == {"bev": 8, "agent": 12}
        b = sb.batch(np.asarray(rows))
        assert b["agent_tokens"].dtype == torch.float32 and b["agent_tokens"].shape == (5, 5, 12)
        assert set(b) <= RS.ALLOWED_INPUT_KEYS, f"the scene batch carries keys the scorer's admissibility list does not know: {sorted(b)}"
        assert b["agent_pad"].dtype == torch.bool and b["agent_pad"].shape == (5, 5) and "bev_pad" not in b   # the capture writes no bev pad
        # mutant: a gather that returns the rows SORTED must be caught by the same check
        class Sorted(SB.SceneBank):
            def batch(self, ii):
                return super().batch(np.sort(np.asarray(ii)))
        mut = Sorted(sb.stores, sb.bank_of_row, sb.scene_row, sb.identical_order)
        assert check_batch_is_own_window(mut, ci, ws, rows) != [], "a sorted gather was not caught"
        sb.close()             # Windows cannot delete a directory that holds an open file


def test_the_join_is_by_key_not_by_position_and_a_positional_join_is_caught():
    ci, ws = canonical()
    perm = np.array([7, 3, 11, 0, 5, 9, 1, 10, 2, 6, 4, 8])           # the scene bank was written in another ROW ORDER, labels moved with it
    with tempfile.TemporaryDirectory() as td:
        fan = make_dump(td, ci, ws, perm=perm)
        sb = SB.attach(fanz_of(ci, ws), [fan], os.path.join(td, "cache"))
        assert sb.identical_order == [False]                           # reported, not hidden
        assert check_batch_is_own_window(sb, ci, ws, list(range(12))) == []
        # mutant: the positional join (scene row = fan row), the bug a re-ordered capture would cause
        mut = SB.SceneBank(sb.stores, sb.bank_of_row, np.arange(12), [True])
        fails = check_batch_is_own_window(mut, ci, ws, list(range(12)))
        assert len(fails) >= 12, f"the positional join read {len(fails)} wrong rows of 24 checks (expected most of them)"
        sb.close()


WHY = {"missing_window": "11 scene windows vs 12 fan windows", "extra_window": "13 scene windows vs 12 fan windows",
       "duplicate_key": "duplicate .clip_index, ws. keys in the scene bank", "shifted_ws": "12 fan windows have no scene row",
       "ws_minus_t_not_constant": "ws - t is not one constant", "no_scene_file": "no l7cap_scene.npz next to",
       "no_bev_source": "lacks .*r7_bev_tokens", "token_rows_ne_keys": "r7_.*_tokens: 11 rows vs 12",
       "target_window_absent": "a training window .* has no scene row"}


@pytest.mark.parametrize("name", ["missing_window", "extra_window", "duplicate_key", "shifted_ws", "ws_minus_t_not_constant",
                                  "no_scene_file", "no_bev_source", "token_rows_ne_keys", "target_window_absent"])
def test_alignment_fails_closed(name):
    ci, ws = canonical()
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "cache")
        fz = fanz_of(ci, ws)
        if name == "missing_window":                                   # the scene bank has 11 of the 12 windows
            keep = np.arange(11)
            fan = make_dump(td, ci, ws, perm=keep, scene_ci=ci[keep], scene_ws=ws[keep])
        elif name == "extra_window":                                   # the scene bank has a 13th window the fan does not
            ci2, ws2 = np.append(ci, 3), np.append(ws, 7)
            fan = make_dump(td, ci2, ws2)
            np.savez(fan, clip_index=ci, ws=ws, t=ws - 7)
        elif name == "duplicate_key":
            sci = ci.copy()
            sws = ws.copy()
            sci[5], sws[5] = sci[4], sws[4]
            fan = make_dump(td, ci, ws, scene_ci=sci, scene_ws=sws)
        elif name == "shifted_ws":                                     # every scene ws is off by one: no key matches
            fan = make_dump(td, ci, ws, scene_ci=ci, scene_ws=ws + 1)
        elif name == "ws_minus_t_not_constant":
            t = ws - 7
            t[3] += 1
            fan = make_dump(td, ci, ws, t=t)
        elif name == "no_scene_file":
            fan = make_dump(td, ci, ws, with_scene=False)
        elif name == "no_bev_source":
            fan = make_dump(td, ci, ws, with_bev=False)
        elif name == "token_rows_ne_keys":
            fan = make_dump(td, ci, ws, n_scene_rows=11)
        else:                                                          # the TARGETS kept a window the scene bank lacks (subset join)
            fan = make_dump(td, ci, ws)
            fz = fanz_of(np.append(ci, 9), np.append(ws, 7))
        with pytest.raises(SystemExit, match=WHY[name]):       # refused for the INTENDED reason, not by an incidental error
            SB.attach(fz, [fan], cache)


def test_two_banks_of_identical_scene_share_one_cache_file_and_rows_span_both():
    ci, ws = canonical()
    with tempfile.TemporaryDirectory() as td:
        f0 = make_dump(os.path.join(td, "s0"), ci, ws)
        f1 = make_dump(os.path.join(td, "s1"), ci, ws)
        cache = os.path.join(td, "cache")
        fz = {"bank_of_row": np.repeat([0, 1], 12), "clip_index": np.tile(ci, 2), "ws": np.tile(ws, 2)}
        sb = SB.attach(fz, [f0, f1], cache)
        assert len([f for f in os.listdir(cache) if f.endswith(".npy")]) == 2          # agent + bev, ONE copy for both banks
        rows = [15, 3, 20, 8]                                           # rows from bank 1, bank 0, bank 1, bank 0, in that order
        assert check_batch_is_own_window(sb, np.tile(ci, 2), np.tile(ws, 2), rows) == []
        sb.close()


def test_all_false_pad_equals_no_pad_so_the_bank_convention_is_read_correctly():
    """The real banks carry r7_agent_pad = all False (0 of 4855 x 300 slots padded; MEASURED).  True = padded in SceneMemory, so an all-False pad
    must give exactly the output of no pad at all."""
    ci, ws = canonical()
    torch.manual_seed(0)
    m = RS.L7Scorer(RS.L7Config(mode="S", source_dims=dict(DIMS)))
    m.eval()
    traj = torch.randn(2, 7, 8, 2)
    v0 = torch.rand(2) * 10
    sc = {"bev": torch.randn(2, 6, 8), "agent": torch.randn(2, 5, 12)}
    with torch.no_grad():
        a = m(traj, v0, scene=sc, scene_pads=None)
        b = m(traj, v0, scene=sc, scene_pads={"agent": torch.zeros(2, 5, dtype=torch.bool)})
        c = m(traj, v0, scene=sc, scene_pads={"agent": torch.tensor([[False, False, True, True, True]] * 2)})
    assert max(float((a[k] - b[k]).abs().max()) for k in a) < 1e-6
    assert max(float((a[k] - c[k]).abs().max()) for k in a) > 1e-5, "regression arm: masking 3 of 5 agent tokens changed nothing"


# ------------------------------------------------------------------------------------------------------ the scene branch is wired
def _grad_report(model, fanz, tg, ii):
    inp, t, keep = TL.to_batch(fanz, tg, ii, "S")
    out = model(**inp)
    loss, _ = RS.pai_loss(out, t, keep, 0.5)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n: (None if p.grad is None else float(p.grad.abs().max())) for n, p in model.named_parameters()}, out


#: the SYNTHETIC bank has no agents and no map, so its nc / ttc / dac masks are empty and those three heads legitimately get no gradient
#: (the real-bank preflight, ``preflight_l7s.py`` G4, requires all 54 parameter tensors to get one)
NO_TARGET_HEADS = ("heads.nc.", "heads.ttc.", "heads.dac.")


def check_scene_branch_gets_gradient(model, fanz, tg, ii) -> list:
    g, _ = _grad_report(model, fanz, tg, ii)
    scene = [n for n in g if n.startswith("memory.") or ".ca." in n]
    return ([f"{n}: grad {g[n]}" for n in scene if not g[n]]
            + [f"{n}: no gradient at all" for n, v in g.items() if v is None and not n.startswith(NO_TARGET_HEADS)])


def test_every_parameter_incl_the_scene_branch_receives_a_gradient_and_the_output_reads_the_scene():
    fanz, tg = TL.synthetic_bank(16, 0)
    fanz["scene"] = SB.synthetic_scene(fanz, DIMS, N_TOK, seed=0)
    torch.manual_seed(0)
    model = RS.L7Scorer(RS.L7Config(mode="S", source_dims=dict(DIMS)))
    ii = np.arange(8)
    assert check_scene_branch_gets_gradient(model, fanz, tg, ii) == []
    g, _ = _grad_report(model, fanz, tg, ii)
    assert len([n for n in g if n.startswith("memory.") or ".ca." in n]) == 16      # 2 proj (w, b) x 2 sources + 2 type_emb + norm (2) + 2 layers x (in_proj w/b + out_proj w/b)
    # the scorer's output depends on the scene: the same candidates with the scene tokens of ANOTHER window score differently
    model.eval()
    with torch.no_grad():
        inp, _, _ = TL.to_batch(fanz, tg, ii, "S")
        a = model(**inp)
        inp2 = dict(inp)
        inp2["scene"] = {k: v.roll(1, 0) for k, v in inp["scene"].items()}
        b = model(**inp2)
    assert max(float((a[k] - b[k]).abs().max()) for k in a) > 1e-4
    # mutant: the memory is bypassed (a constant), the projections stay registered but unused -> both checks must go RED
    model.train()

    class Bypass(torch.nn.Module):
        def __init__(self, real):
            super().__init__()
            self.proj, self.type_emb, self.norm = real.proj, real.type_emb, real.norm

        def forward(self, sources, pads=None):
            B = next(iter(sources.values())).shape[0]
            return torch.zeros(B, sum(v.shape[1] for v in sources.values()), 128), None

    model.memory = Bypass(model.memory)
    assert check_scene_branch_gets_gradient(model, fanz, tg, ii) != [], "a scene-blind scorer passed the gradient check"
    model.eval()
    with torch.no_grad():
        a2, b2 = model(**inp), model(**inp2)
    assert max(float((a2[k] - b2[k]).abs().max()) for k in a2) == 0.0


def test_param_count_at_the_real_capture_widths_is_the_dry_run_literal():
    """(bev 96, agent 384) is the l7cap capture's width.  464,520 is the registered test's literal at agent 256; the 128 extra input
    channels cost 128 x d_model(128) = 16,384 weights -> 480,904, which is also what the real-bank dry run printed."""
    assert RS.n_params(RS.L7Scorer(RS.L7Config(mode="S", source_dims={"bev": 96, "agent": 384}))) == 480904
    assert RS.n_params(RS.L7Scorer(RS.L7Config(mode="S", source_dims=dict(DIMS)))) == 464520 - (88 + 244) * 128    # 422,024


def check_cross_episode(eid, idx, m) -> list:
    """Every val row is scored with the tokens of a row from ANOTHER episode, and the map stays inside the validation slice."""
    inside = set(idx.tolist())
    bad = [int(r) for r in idx if eid[m[r]] == eid[r]]
    out = [int(r) for r in idx if int(m[r]) not in inside]
    return ([f"rows {bad[:5]} map inside their own episode"] if bad else []) + ([f"rows {out[:5]} map outside the val slice"] if out else [])


def test_scene_shuffle_map_is_cross_episode_deterministic_and_a_self_map_is_caught():
    eid = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 5 + ["d"] * 2 + ["z"] * 6)             # 'z' is NOT in the validation slice
    idx = np.where(eid != "z")[0]
    m = TL.cross_episode_map(eid, idx, len(eid))
    assert check_cross_episode(eid, idx, m) == []
    # literal: episode a (rows 0-3) -> b's rows (4,5,6) cycling; b (4-6) -> c (7..11); c (7-11) -> d (12,13) cycling; d (12,13) -> a (0..3)
    assert m[:14].tolist() == [4, 5, 6, 4, 7, 8, 9, 12, 13, 12, 13, 12, 0, 1]
    assert m[14:].tolist() == list(range(14, 20)), "rows outside the slice must be untouched"
    assert np.array_equal(m, TL.cross_episode_map(eid, idx, len(eid)))                    # no RNG
    assert check_cross_episode(eid, idx, np.arange(len(eid))) != [], "an identity map (no shuffle) was not caught"
    with pytest.raises(SystemExit):
        TL.cross_episode_map(eid, np.where(eid == "a")[0], len(eid))


def test_shuffled_scene_evaluation_reads_other_windows_tokens_and_the_targets_stay_put():
    fanz, tg = TL.synthetic_bank(16, 0)
    fanz["scene"] = SB.synthetic_scene(fanz, DIMS, N_TOK, seed=0)
    torch.manual_seed(0)
    model = RS.L7Scorer(RS.L7Config(mode="S", source_dims=dict(DIMS)))
    eid = np.asarray(tg["eid"]).astype(str)
    idx = np.arange(16)
    m = TL.cross_episode_map(eid, idx, 16)
    inp, t, _ = TL.to_batch(fanz, tg, idx, "S", scene_rows=m[idx])
    # literal: the scene tokens handed to the scorer for row r carry the (clip_index, ws) of row m[r] ...
    ci, ws = np.asarray(fanz["clip_index"]), np.asarray(fanz["ws"])
    a = inp["scene"]["agent"].numpy()
    assert all(a[j, 0, 0] == ci[m[j]] and a[j, 0, 1] == ws[m[j]] for j in range(16))
    assert any(ci[m[j]] != ci[j] for j in range(16))
    # ... while the targets are the TRUE rows' (the control moves the scene only)
    _, t0, _ = TL.to_batch(fanz, tg, idx, "S")
    assert torch.equal(t["cost"], t0["cost"])
    r_true = TL.evaluate(model, fanz, tg, idx, "S")
    r_shuf = TL.evaluate(model, fanz, tg, idx, "S", scene_map=m)
    assert r_true["oracle_cost_m"] == r_shuf["oracle_cost_m"] and r_true["own_pick_cost_m"] == r_shuf["own_pick_cost_m"]
    assert r_true["loss"] != r_shuf["loss"]                                              # the scorer's output moved with the scene


class NullOpt:
    """An optimizer that never moves a weight: the training LOOP runs (batches, forward, loss, backward, both evaluations, the log, the
    checkpoint) with ZERO updates, so this file takes no training step anywhere."""
    def __init__(self, *a, **k):
        self.steps = 0

    def zero_grad(self, *a, **k):
        pass

    def step(self, *a, **k):
        self.steps += 1


def test_train_loop_mode_S_with_a_null_optimizer_logs_the_shuffled_read_and_best_pt_loads(monkeypatch):
    import json
    monkeypatch.setattr(torch.optim, "AdamW", NullOpt)
    with tempfile.TemporaryDirectory() as td:
        argv = sys.argv
        try:
            sys.argv = ["train_l7.py", "--synthetic", "160", "--mode", "S", "--epochs", "1", "--out-dir", td]
            assert TL.main() == 0
        finally:
            sys.argv = argv
        log = json.load(open(os.path.join(td, "train_log.json")))
        e = log["epochs"][0]
        assert {"val", "val_scene_shuffled", "train_subset"} <= set(e)
        assert e["val"]["oracle_cost_m"] == e["val_scene_shuffled"]["oracle_cost_m"]         # same windows, same targets
        assert e["val"]["loss"] != e["val_scene_shuffled"]["loss"]                           # the scene moved the scorer's output
        # best.pt round-trips through the package's own loader with its source_dims, and no weight moved from the seed-0 initialisation
        loaded = RS.load_scorer(os.path.join(td, "best.pt"))
        assert loaded.cfg.mode == "S" and loaded.cfg.source_dims == DIMS and RS.n_params(loaded) == 422024
        torch.manual_seed(0)
        fresh = RS.L7Scorer(RS.L7Config(mode="S", source_dims=dict(DIMS)))
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in fresh.state_dict().items())
        # E / F logs keep their schema: no scene-shuffled key
        sys.argv = ["train_l7.py", "--synthetic", "160", "--mode", "E", "--epochs", "1", "--out-dir", os.path.join(td, "E")]
        try:
            assert TL.main() == 0
        finally:
            sys.argv = argv
        assert "val_scene_shuffled" not in json.load(open(os.path.join(td, "E", "train_log.json")))["epochs"][0]


def test_params_cli_default_is_the_real_capture_width(capsys):
    """``python refc_scorer.py --params`` read agent width 256 (not the capture's 384) until 2026-10-22, so it reported 464,520 for L7-S."""
    import json
    import runpy
    argv = sys.argv
    try:
        sys.argv = ["refc_scorer.py", "--params"]
        runpy.run_path(os.path.join(os.path.dirname(HERE), "refc_scorer.py"), run_name="__main__")
    finally:
        sys.argv = argv
    rep = json.loads(capsys.readouterr().out)
    assert rep["L7-S (PAI heads)"] == 480904 and rep["L7-E (PAI heads)"] == 154504 and rep["L7-F (PAI heads)"] == 155400


def test_train_l7_dry_run_mode_S_synthetic_and_the_refusals():
    import json
    with tempfile.TemporaryDirectory() as td:
        sys_argv = sys.argv
        try:
            sys.argv = ["train_l7.py", "--synthetic", "64", "--mode", "S", "--epochs", "0", "--out-dir", td]
            assert TL.main() == 0
        finally:
            sys.argv = sys_argv
        log = json.load(open(os.path.join(td, "dry_run_log.json")))
        assert log["dry_run"] and log["args"]["mode"] == "S" and log["params"] == 422024
        assert not os.path.exists(os.path.join(td, "best.pt")), "a dry run wrote a checkpoint"
        # a real-bank S run without a scene cache must refuse (before it loads anything), not fall back to E
        sys.argv = ["train_l7.py", "--fan-npz", "x/fan_extras.npz", "--targets-npz", "x/t.npz", "--mode", "S", "--epochs", "0", "--out-dir", td]
        try:
            with pytest.raises(SystemExit):
                TL.main()
        finally:
            sys.argv = sys_argv


def test_the_unpack_cli_refuses_an_eval_capture():
    argv = sys.argv
    try:
        sys.argv = ["scene_bank.py", "--npz", "D:/x/eval139_s0/dump/l7cap_scene.npz", "--cache", "C:/nowhere"]
        with pytest.raises(SystemExit, match="REFUSING to unpack"):
            SB._cli()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
