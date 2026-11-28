"""PREREG_V8F_SPEEDSUB_T1 seam B: ``stack/scripts/v8f_t1_speedsub.py`` -- the speed-cell substitution wrapper around the PINNED T1-kin tool. CPU only, synthetic world, tiny model; every test drives the
REAL tool ``main()`` through the wrapper (the actual caller), so a substitution that is built but never reaches O cannot pass.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  CUDA_VISIBLE_DEVICES=-1  pytest -q stack/tests/test_v8f_t1_speedsub.py
Mutation pass: ``SPEEDSUB_WRAPPER=<mutant.py>`` points the whole file at a mutated copy of the wrapper (see ``mutation_speedsub.py``); every mutant must make this file FAIL.

What is pinned, and the control that must read exactly as stated:
  * IDENTITY    ``identity`` is bit-equal to the unpatched tool: every dump array, every tick-row array, the result JSON bytes.            [mutant: identity returns a clone with a changed cell]
  * RESTORE     the three class-level patches are gone after the run; a second install is REFUSED.
  * NON-SPEED   the 55 other code dimensions O READS equal T's own: max |delta| = 0.0 exactly (recorded from the argument of ``run_o``). [mutants: a 6th index; an offset index]
  * CELLS       the cells O reads equal ``mlp_ego_c`` on the recorded ego block, standardised by the checkpoint's cons_mu / cons_sd, <= 1e-6; and they differ from T's own on >= 99 % of (window, tick).
  * REACH       ``egoc`` moves O's PLAN on every window against ``identity`` (the effect reaches the caller's output), the P10 floor arm stays bit-equal.   [mutant: substitution computed but not returned]
  * SHUFFLE     the registered permutation is ``default_rng(20261029).permutation(2399)`` with sha 64785f8a... (SPEEDSUB's banked one) and 0 fixed points; in the run window w reads the egoc run's cells of
                window perm[w] at the SAME tick, exactly; ``egoshufc`` differs from ``egoc`` on >= 95 %.                                  [mutants: wrong seed; tick off by one; unshuffled source]
  * FLOOR       the ``fl`` arm of every mode equals the unpatched tool's, bit for bit.
  * FORWARD_DIAG the single-pass patched ``forward_infer`` equals ``forward_diag(batch, own code with the five cells replaced)`` bit for bit (PREREG: "runs O on the new code as forward_diag does").
  * RECORD      the wrapper's rows are aligned with the tool's own tick rows (win, tick, age_t, v0); a corrupted record is detected.
  * REFUSALS    unknown mode, wrong limit lane, no --dump-ticks, a non-CPU device, a read-out with the wrong sha / missing / without its key / not ego-only, a checkpoint with the wrong sha, a tool with
                the wrong sha, an egoshufc source from another checkpoint or incomplete, an ``o_interface = cut`` checkpoint, a double install.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_eval_speedread as SR  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
SCRIPTS = REPO / "stack" / "scripts"
WRAPPER_PATH = os.environ.get("SPEEDSUB_WRAPPER") or str(SCRIPTS / "v8f_t1_speedsub.py")
REG_PERM_SHA = "64785f8ad0797b02bdf9845cedf6cf425f8759a2993da889cf33146a0e37ab64"      # SPEEDSUB's banked perm_sha256 (SUBREAD.json controls S9), numpy default_rng(20261029).permutation(2399)
CODE_IDX = [46, 56, 57, 58, 59]


def load_wrapper():
    spec = importlib.util.spec_from_file_location("v8f_t1_speedsub_under_test", WRAPPER_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_speedsub_under_test"] = m
    spec.loader.exec_module(m)
    return m


W = load_wrapper()


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_speedsub")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)
    return {"root": root, **w}


def tiny_model(J, seed=0, **cfg_over):
    cfg = M.V8FMiniConfig(**{**TINY, **cfg_over})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head_o[2].weight, std=0.02)
        torch.nn.init.normal_(model.e_code.weight, std=0.5)               # O must actually READ the code, or no substitution could show in the plan
    return model, cfg


def write_ckpt(path, model, cfg):
    torch.save({"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}, path)


def make_readout(path, key="mlp_ego_c", seed=3, ego_only=True, d_in=22):
    """A random ego-only MLP read-out in the real container, with targets in physical m/s (mean 8, sd 4) so it differs visibly from T's own cells."""
    rng = np.random.default_rng(seed)
    net = SR.mlp_build(d_in, 5, SR.MLP_CFG)
    fit = {"state": {k: v.detach().clone() for k, v in net.state_dict().items()}, "ym": np.full(5, 8.0), "ys": np.full(5, 4.0), "best_epoch": 1, "cfg": {"hidden": 256, "dropout": 0.1, "seed": 0}}
    ro = SR.ReadOut("mlp_ego", mlp=fit, ego_only=ego_only, pca_mu=None, pca_V=None, xm=rng.normal(size=d_in).astype(np.float32), xs=(1.0 + rng.random(d_in)).astype(np.float32))
    SR.save_readouts({key: ro}, str(path))
    return SR.load_readouts(str(path))[key]


class Rig:
    def __init__(self, world, tmp):
        self.root, self.tmp = Path(world["root"]), Path(tmp)
        self.J = D.load_join(self.root / "join_eval.npz")
        self.model, self.cfg = tiny_model(self.J)
        self.ckpt = self.tmp / "ck.pt"
        write_ckpt(self.ckpt, self.model, self.cfg)
        self.ckpt_sha = W.sha256_file(self.ckpt)
        self.ro_path = self.tmp / "readout_c.pt"
        self.ro = make_readout(self.ro_path)
        self.ro_sha = W.sha256_file(self.ro_path)
        self.ro_l_path = self.tmp / "readout_l.pt"
        make_readout(self.ro_l_path, key="mlp_ego", seed=4)
        self.ro_l_sha = W.sha256_file(self.ro_l_path)

    def tool_argv(self, label, tag="syn", limit="causal", extra=(), ckpt=None):
        r = self.root
        return ["run", "--ckpt", str(ckpt or self.ckpt), "--tag", tag, "--tracks", str(r / "native_tracks.pt"), "--eval-cache", str(r), "--eval-grid", str(r / "eval_grid_windows.json"),
                "--eval-join", str(r / "join_eval.npz"), "--out-dir", str(self.tmp / label), "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic",
                "--limit", limit, "--dump-ticks", str(self.tmp / label / "ticks"), "--dump-heads", "--batch", "6", "--max-windows", "12", "--n-boot", "20", "--device", "cpu", *extra]

    def wrap_argv(self, mode, label, tag="syn", limit=None, readout=None, readout_sha=None, shuffle_source=None, extra_wrap=(), tool_extra=(), ckpt=None, expect_ckpt_sha=None):
        limit = limit or W.MODE_LIMIT[mode]
        ro, sha = (readout, readout_sha) if readout is not None else ((self.ro_l_path, self.ro_l_sha) if mode == "egol" else (self.ro_path, self.ro_sha))
        a = ["--mode", mode, "--tool-root", str(REPO), "--instr", str(SCRIPTS / "v8f_mini_eval_speedread.py"), "--expect-ckpt-sha", expect_ckpt_sha or self.ckpt_sha]
        if mode in W.SUBSTITUTING:
            a += ["--readout", str(ro), "--readout-sha", sha]
        if shuffle_source:
            a += ["--shuffle-source", str(shuffle_source)]
        return a + list(extra_wrap) + self.tool_argv(label, tag, limit, tool_extra, ckpt)

    def run_wrapped(self, mode, label, **kw):
        rc = W.run(self.wrap_argv(mode, label, **kw))
        return self.tmp / label, rc

    def run_unpatched(self, label, limit="causal", tag="syn"):
        assert T1.main(self.tool_argv(label, tag, limit)) == 0
        return self.tmp / label


def dumps(out, tag="syn"):
    d = out / f"dump_{tag}"
    return {f.name: {k: np.load(f)[k] for k in np.load(f).files} for f in sorted(d.iterdir())}


def ticks(out, tag="syn"):
    z = np.load(out / "ticks" / f"ticks_{tag}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def rec(out, tag="syn"):
    z = np.load(out / "ticks" / f"speedsub_{tag}.npz", allow_pickle=False)
    r = {k: z[k] for k in z.files if k != "meta_json"}
    r["meta"] = json.loads(str(z["meta_json"]))
    return r


def same_json(out_a, out_b):
    """The result JSONs equal as text, but for `ticks_file` (the run directory's own path)."""
    ja, jb = (json.loads((o / "t1v8f_syn.json").read_text()) for o in (out_a, out_b))
    assert ja.get("t1v8f", {}).get("ticks_file") != jb.get("t1v8f", {}).get("ticks_file") or True
    for j in (ja, jb):
        j["t1v8f"].pop("ticks_file")
    sa, sb = json.dumps(ja, sort_keys=True), json.dumps(jb, sort_keys=True)
    assert len(sa) > 1000
    return sa == sb


def same_arrays(a, b):
    assert a.keys() == b.keys() and len(a) >= 3
    for f in a:
        assert set(a[f]) == set(b[f]), f
        for k in a[f]:
            x, y = a[f][k], b[f][k]
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (f, k)


@pytest.fixture(scope="module")
def rig(world, tmp_path_factory):
    return Rig(world, tmp_path_factory.mktemp("speedsub_runs"))


@pytest.fixture(scope="module")
def runs(rig):
    """Every rollout of this file, once."""
    R = {"base": rig.run_unpatched("base"), "base_l": rig.run_unpatched("base_l", limit="logged")}
    for mode in ("identity", "egoc", "identity_l", "egol"):
        out, rc = rig.run_wrapped(mode, mode)
        assert rc == 0, (mode, rc)
        R[mode] = out
    # egoshufc: the registered permutation has 2,399 entries; this world has 12 windows, so the SAME recipe (default_rng(20261029).permutation(n)) is run with n = 12 (--perm-n)
    out, rc = rig.run_wrapped("egoshufc", "egoshufc", shuffle_source=R["egoc"] / "ticks" / "speedsub_syn.npz", extra_wrap=["--perm-n", "12"])
    assert rc == 0
    R["egoshufc"] = out
    return R


# ------------------------------------------------------------------------------------------------ IDENTITY / RESTORE / FLOOR
def test_identity_is_bit_equal_to_the_unpatched_tool(runs):
    same_arrays(dumps(runs["identity"]), dumps(runs["base"]))
    a, b = ticks(runs["identity"]), ticks(runs["base"])
    assert a.keys() == b.keys()
    for k in a:
        if a[k].dtype.kind in "fiub":
            assert a[k].tobytes() == b[k].tobytes(), k
    assert same_json(runs["identity"], runs["base"])


def test_identity_logged_lane_is_bit_equal_to_the_unpatched_tool(runs):
    same_arrays(dumps(runs["identity_l"]), dumps(runs["base_l"]))
    assert same_json(runs["identity_l"], runs["base_l"])


def test_identity_reads_t_s_own_cells_exactly(runs):
    r = rec(runs["identity"])
    assert np.array_equal(r["o_std"], r["own_std"]) and np.array_equal(r["used_std"], r["own_std"])
    assert float(r["nonspeed_dev"].max()) == 0.0 and r["meta"]["nonspeed_dev_max"] == 0.0


def test_the_class_is_restored_and_a_second_install_is_refused(runs, rig):
    V = M.V8FMini
    assert not hasattr(V, "_speedsub_installed")
    assert V.forward_infer.__qualname__.endswith("V8FMini.forward_infer") and V._code_pred.__qualname__.endswith("V8FMini._code_pred") and V.run_o.__qualname__.endswith("V8FMini.run_o")
    sr = SR
    h1 = W.SpeedSubHook(M, sr, "identity")
    with h1:
        with pytest.raises(SystemExit, match="already patched"):
            W.SpeedSubHook(M, sr, "identity").install()
    assert not hasattr(V, "_speedsub_installed")


@pytest.mark.parametrize("mode", ["egoc", "egoshufc", "egol"])
def test_the_floor_arm_is_untouched_in_every_mode(runs, mode):
    base = runs["base_l"] if mode == "egol" else runs["base"]
    a, b = dumps(runs[mode]), dumps(base)
    for f in a:
        assert a[f]["fl"].tobytes() == b[f]["fl"].tobytes() and a[f]["ha0"].tobytes() == b[f]["ha0"].tobytes() and a[f]["g"].tobytes() == b[f]["g"].tobytes(), f


# ------------------------------------------------------------------------------------------------ NON-SPEED / CELLS / REACH
@pytest.mark.parametrize("mode", ["egoc", "egoshufc", "egol"])
def test_the_55_other_dimensions_o_reads_are_t_s_own_exactly(runs, mode):
    r = rec(runs[mode])
    assert float(r["nonspeed_dev"].max()) == 0.0 and r["meta"]["nonspeed_dev_max"] == 0.0
    assert r["meta"]["code_idx"] == CODE_IDX and len(r["win"]) == 12 * 3


def test_the_cells_o_reads_equal_the_readout_on_the_recorded_ego_block(runs, rig):
    r = rec(runs["egoc"])
    m = r["meta"]
    mu, sd = np.asarray(m["cons_mu"]), np.asarray(m["cons_sd"])
    want = ((rig.ro.predict_raw(None, r["ego"]) - mu) / sd).astype(np.float32)
    assert float(np.abs(want - r["used_std"]).max()) <= 1e-6
    assert float(np.abs(want - r["o_std"]).max()) <= 1e-6                     # what O READ, not what the substitution intended
    assert r["ego"].shape[1] == 22 and np.isfinite(r["ego"]).all()


def test_the_recorded_standardisation_is_the_checkpoints_own(runs, rig):
    m = rec(runs["egoc"])["meta"]
    ci = [M.CI[n] for n in W.SPEED_CELLS]
    assert np.array_equal(np.asarray(m["cons_mu"]), rig.model.cons_mu[ci].double().numpy()) and np.array_equal(np.asarray(m["cons_sd"]), rig.model.cons_sd[ci].double().numpy())
    assert m["cons_idx"] == ci and m["ckpt_sha256"] == rig.ckpt_sha and m["readout_sha256"] == rig.ro_sha


@torch.no_grad()
def test_the_non_speed_check_can_read_a_non_zero_value(rig, world):
    """The control that must be able to FAIL: a hook that also edits code dimension 3 (a goal token) is seen by the non-speed deviation, which reads exactly the edit."""
    batch = _batch_from_world(rig, world)

    class Wider(W.SpeedSubHook):
        def _substitute(self, model, code):
            new = super()._substitute(model, code)
            new = new.clone()
            new[:, 3] += 0.25
            return new
    hook = Wider(M, SR, "egoc", rig.ro, n_ticks=3)
    with hook:
        rig.model.forward_infer(batch)
    assert abs(hook.nonspeed_dev_max - 0.25) < 1e-6


def test_the_record_holds_each_window_and_tick_exactly_once_in_the_tool_s_order(runs):
    r = rec(runs["egoc"])
    pairs = list(zip(r["win"].tolist(), r["tick"].tolist()))
    assert len(set(pairs)) == len(pairs) == 36 and sorted(set(r["win"].tolist())) == list(range(12)) and sorted(set(r["tick"].tolist())) == [0, 1, 2]
    assert (np.diff(r["tick"][:36]) >= 0).sum() > 0 and r["win"][:6].tolist() == [0, 1, 2, 3, 4, 5] and r["win"][18:24].tolist() == [6, 7, 8, 9, 10, 11]      # chunk of 6, then its ticks, then the next chunk


def test_the_substitution_moves_the_cells_on_nearly_every_window_and_tick(runs):
    r = rec(runs["egoc"])
    moved = np.abs(r["o_std"] - r["own_std"]).max(1) > 1e-6
    assert moved.mean() >= 0.99


def test_egoc_reaches_the_callers_output(runs):
    """The effect arrives where the tool scores: the executed path ``cl`` and the tick-0 plan ``ol`` differ from identity's on every window; the control (identity vs the unpatched tool) is exactly 0."""
    a, i, b = dumps(runs["egoc"]), dumps(runs["identity"]), dumps(runs["base"])
    d_c = np.concatenate([np.abs(a[f]["cl"] - i[f]["cl"]).reshape(len(a[f]["cl"]), -1).max(1) for f in a])
    d_o = np.concatenate([np.abs(a[f]["ol"] - i[f]["ol"]).reshape(len(a[f]["ol"]), -1).max(1) for f in a])
    d_0 = max(float(np.abs(i[f]["cl"] - b[f]["cl"]).max()) for f in i)
    assert d_0 == 0.0
    assert (d_c > 0).all() and (d_o > 0).all()


def test_egoc_lands_in_the_ticks_file_too(runs):
    """The tool's own tick rows (``plan``) change with the substitution at every tick: the loop is closed through the executed motion."""
    a, i = ticks(runs["egoc"]), ticks(runs["identity"])
    assert (np.abs(a["plan"] - i["plan"]).reshape(len(a["plan"]), -1).max(1) > 0).all()
    assert a["t_cons_pred"].tobytes() != b"" and np.array_equal(a["t_cons_pred"][a["tick"] == 0], i["t_cons_pred"][i["tick"] == 0])      # T's own head output at tick 0 is NOT changed by the substitution


# ------------------------------------------------------------------------------------------------ SHUFFLE
def test_the_registered_permutation():
    p = W.registered_perm()
    assert len(p) == 2399 and W.perm_sha256(p) == REG_PERM_SHA and int((p == np.arange(2399)).sum()) == 0
    assert np.array_equal(p, np.random.default_rng(20261029).permutation(2399))
    assert W.SHUF_SEED == 20261029 and W.SHUF_N == 2399


def test_egoshufc_reads_the_egoc_cells_of_window_perm_w_at_the_same_tick(runs):
    c, s = rec(runs["egoc"]), rec(runs["egoshufc"])
    perm = np.random.default_rng(20261029).permutation(12)
    grid = {(int(w), int(t)): u for w, t, u in zip(c["win"], c["tick"], c["used_std"])}
    for w, t, o, sw in zip(s["win"], s["tick"], s["o_std"], s["src_win"]):
        assert int(sw) == int(perm[int(w)])
        assert np.array_equal(o, grid[(int(perm[int(w)]), int(t))])
    assert s["meta"]["perm_sha256"] == W.perm_sha256(perm) and s["meta"]["perm_seed"] == 20261029 and s["meta"]["perm_n"] == 12


def test_egoshufc_differs_from_egoc_and_from_identity(runs):
    c, s, i = rec(runs["egoc"]), rec(runs["egoshufc"]), rec(runs["identity"])
    perm = np.random.default_rng(20261029).permutation(12)
    t0 = s["tick"] == 0                                                       # tick 0: both runs read the same rows of the same windows, so the cell difference is the shuffle alone
    moved = np.abs(s["o_std"][t0] - c["o_std"][t0]).max(1) > 1e-6
    fixed = perm[s["win"][t0]] == s["win"][t0]                                # the 12-window permutation has fixed points (the registered 2,399 one has none): there the cells are EXACTLY egoc's
    assert moved[~fixed].all() and not moved[fixed].any() and (~fixed).sum() >= 10
    assert (np.abs(s["o_std"] - i["o_std"]).max(1) > 1e-6).mean() >= 0.95


# ------------------------------------------------------------------------------------------------ FORWARD_DIAG equivalence, RECORD alignment
def _batch_from_world(rig, world, n=6):
    """A tick-0 batch of real-format windows built by the tool's own item builder."""
    tracks = D.load_native_tracks(rig.root / "native_tracks.pt")
    J = rig.J
    store = D.ClipStore(rig.root, "eval", lru=8, kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(rig.root / "eval_grid_windows.json", J)[:n]
    proxy = T1._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=rig.cfg.hist, hmax=rig.cfg.hmax, max_age_t=rig.cfg.max_age_t, max_age_s=rig.cfg.max_age_s)
    sts = T1.make_states(store, tracks, J, wins, "deadreckon")
    items = [T1.build_item(ds, proxy, st, 0, 1, 2, "logged", "causal") for st in sts]
    full = D.collate(items)
    return {k: full[k] for k in T1.MODEL_KEYS}


@torch.no_grad()
def test_the_single_pass_hook_equals_forward_diag_on_the_substituted_code(rig, world):
    model = rig.model
    batch = _batch_from_world(rig, world)
    own = model.forward_infer(batch)
    mu = model.cons_mu[[M.CI[n] for n in W.SPEED_CELLS]].double().numpy()
    sd = model.cons_sd[[M.CI[n] for n in W.SPEED_CELLS]].double().numpy()
    new_std = SR.phys_to_std(rig.ro.predict_ego_batch(batch), mu, sd)
    ref = model.forward_diag(batch, SR.substitute(own["code"], new_std, CODE_IDX))
    with W.SpeedSubHook(M, SR, "egoc", rig.ro, n_ticks=3):
        got = model.forward_infer(batch)
    for k in ("traj60", "traj30", "code", "p10"):
        assert got[k].numpy().tobytes() == ref[k].numpy().tobytes(), k
    assert not torch.equal(got["code"], own["code"]) and torch.equal(got["code"][:, [i for i in range(M.CODE_DIM) if i not in CODE_IDX]], own["code"][:, [i for i in range(M.CODE_DIM) if i not in CODE_IDX]])
    assert got["traj60"].numpy().tobytes() != own["traj60"].numpy().tobytes()


def test_the_record_is_aligned_with_the_tool_s_tick_rows(runs):
    for mode in ("identity", "egoc", "egoshufc", "identity_l", "egol"):
        m = rec(runs[mode])["meta"]
        a = m["alignment_with_tool_ticks"]
        assert m["aligned"] and a["win_equal"] and a["tick_equal"] and a["age_t_equal"] and a["v0_equal"], (mode, a)
        assert m["tool_sha256"] == W.TOOL_SHA256 and m["n_ticks"] == 3 and m["n_windows"] == 12


def test_the_alignment_check_detects_a_corrupted_record(runs):
    r, t = rec(runs["egoc"]), ticks(runs["egoc"])
    ok, det = W.check_alignment({k: r[k] for k in ("win", "tick", "age_t", "v0_last")}, t, 3)
    assert ok
    for key, bad in (("win", np.roll(r["win"], 1)), ("tick", np.roll(r["tick"], 1)), ("age_t", r["age_t"] + 1), ("v0_last", r["v0_last"] + 1e-9)):
        arr = {k: r[k] for k in ("win", "tick", "age_t", "v0_last")}
        arr[key] = bad
        assert not W.check_alignment(arr, t, 3)[0], key
    short = {k: r[k][:-1] for k in ("win", "tick", "age_t", "v0_last")}
    assert not W.check_alignment(short, t, 3)[0]


def test_the_logged_lane_egol_uses_the_banked_style_readout(runs, rig):
    m = rec(runs["egol"])["meta"]
    assert m["limit"] == "logged" and m["readout_key"] == "mlp_ego" and m["readout_sha256"] == rig.ro_l_sha
    r = rec(runs["egol"])
    assert (np.abs(r["o_std"] - r["own_std"]).max(1) > 1e-6).mean() >= 0.99


# ------------------------------------------------------------------------------------------------ REFUSALS
def refuse(rig, mode, label, match, **kw):
    with pytest.raises(SystemExit, match=match):
        W.run(rig.wrap_argv(mode, label, **kw))


def test_refusal_unknown_mode(rig):
    with pytest.raises(SystemExit):
        W.run(["--mode", "ego", *rig.tool_argv("x")])


def test_refusal_wrong_limit_lane(rig):
    refuse(rig, "egoc", "r1", "limit", limit="logged")
    refuse(rig, "egol", "r2", "limit", limit="causal")


def test_refusal_no_dump_ticks(rig):
    a = rig.wrap_argv("identity", "r3")
    i = a.index("--dump-ticks")
    del a[i:i + 2]
    with pytest.raises(SystemExit, match="dump-ticks"):
        W.run(a)


def test_refusal_non_cpu_device(rig):
    a = rig.wrap_argv("identity", "r4")
    a[a.index("--device") + 1] = "cuda"
    with pytest.raises(SystemExit, match="CPU"):
        W.run(a)


def test_refusal_readout_sha_missing_key_and_not_ego_only(rig):
    refuse(rig, "egoc", "r5", "pinned", readout_sha="0" * 64, readout=rig.ro_path)
    refuse(rig, "egoc", "r6", "needs --readout", readout=rig.tmp / "nope.pt", readout_sha="0" * 64)
    refuse(rig, "egoc", "r7", "holds", readout=rig.ro_l_path, readout_sha=rig.ro_l_sha)                 # holds 'mlp_ego', egoc reads 'mlp_ego_c'
    bad = rig.tmp / "readout_scene.pt"
    make_readout(bad, ego_only=False)
    refuse(rig, "egoc", "r8", "ego-only", readout=bad, readout_sha=W.sha256_file(bad))
    narrow = rig.tmp / "readout_narrow.pt"
    make_readout(narrow, d_in=12)
    refuse(rig, "egoc", "r9", "ego-only", readout=narrow, readout_sha=W.sha256_file(narrow))


def test_refusal_checkpoint_and_tool_sha(rig):
    refuse(rig, "identity", "r10", "checkpoint", expect_ckpt_sha="1" * 64)
    with pytest.raises(SystemExit, match="T1 tool"):
        W.run(rig.wrap_argv("identity", "r11", extra_wrap=["--tool-sha", "2" * 64]))


def test_refusal_egoshufc_source(rig, runs, tmp_path):
    refuse(rig, "egoshufc", "r12", "shuffle-source", extra_wrap=["--perm-n", "12"])
    src = np.load(runs["egoc"] / "ticks" / "speedsub_syn.npz", allow_pickle=False)
    meta = json.loads(str(src["meta_json"]))
    arrays = {k: src[k] for k in src.files if k != "meta_json"}
    other = dict(meta, ckpt_sha256="3" * 64)
    np.savez(tmp_path / "other_ckpt.npz", meta_json=np.asarray(json.dumps(other)), **arrays)
    refuse(rig, "egoshufc", "r13", "not an egoc rollout of THIS", shuffle_source=tmp_path / "other_ckpt.npz", extra_wrap=["--perm-n", "12"])
    notegoc = dict(meta, mode="identity")
    np.savez(tmp_path / "not_egoc.npz", meta_json=np.asarray(json.dumps(notegoc)), **arrays)
    refuse(rig, "egoshufc", "r14", "not an egoc rollout of THIS", shuffle_source=tmp_path / "not_egoc.npz", extra_wrap=["--perm-n", "12"])
    half = {k: (v[:-1] if getattr(v, "ndim", 0) >= 1 and len(v) == 36 else v) for k, v in arrays.items()}
    np.savez(tmp_path / "half.npz", meta_json=np.asarray(json.dumps(meta)), **half)
    refuse(rig, "egoshufc", "r15", "full grid", shuffle_source=tmp_path / "half.npz", extra_wrap=["--perm-n", "12"])
    refuse(rig, "egoshufc", "r16", "no --max-windows", shuffle_source=runs["egoc"] / "ticks" / "speedsub_syn.npz")          # the registered n = 2399 with --max-windows 12


def test_refusal_o_interface_cut(world, tmp_path):
    R = Rig(world, tmp_path)
    model, cfg = tiny_model(R.J, o_interface="cut")
    write_ckpt(R.ckpt, model, cfg)
    R.ckpt_sha = W.sha256_file(R.ckpt)
    with pytest.raises(SystemExit, match="cut"):
        W.run(R.wrap_argv("egoc", "cut"))
    out, rc = R.run_wrapped("identity", "cut_id")                              # identity does not need O to read the code
    assert rc == 0


@torch.no_grad()
def test_a_hook_without_t_s_code_is_loud(rig, world):
    """If the patched ``_code_pred`` is never reached (a model that builds its code elsewhere), the call fails instead of recording T's own cells as substituted."""
    batch = _batch_from_world(rig, world)
    hook = W.SpeedSubHook(M, SR, "egoc", rig.ro, n_ticks=3)
    with hook:
        orig = hook.orig["_code_pred"]
        M.V8FMini._code_pred = orig                                            # simulate a hook that lost its _code_pred patch
        with pytest.raises(RuntimeError, match="not reached"):
            rig.model.forward_infer(batch)
