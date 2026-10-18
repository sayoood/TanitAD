"""PI ruling D-PI-YAWRATE-1 (2026-10-17): *"the model should get measured ego motion and its history, instead of steering angle which
represents a sort of action, the model should get the measured yaw rate"*.

What these tests pin, and the rule they obey (CLAUDE.md, "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER"): a compliant path that merely
EXISTS proves nothing -- every guard below has a DELIBERATE-REGRESSION arm that re-introduces the steer (or shifts the yaw by one frame) and
must make the guard go RED, and a CONTROL arm (the legacy ``steer`` mode) that must read the leak the compliant mode does not.

1. THE SPELLING.  The yaw rate / kinematics of the compliant path equal ``ego_channels_from_poses``' ``dyaw`` channel (and
   ``kinematic_prior.pose_kin3``) on REAL eval-141 windows to <= 1e-6; a one-frame-shifted spelling FAILS the same comparator; the value at
   frame k never changes when later poses change (a forward difference -- the cache's own ``yaw_rate`` -- does).
2. THE INFORMATION IS RE-EXPRESSED, NOT DELETED.  ``kappa0`` tracks the recorded curvature ``tan(steer)/2.9`` on 12k real eval frames.
3. THE INPUT PATHS NEVER READ STEER.  A twin of every cache clip whose ``steer`` / ``kin3`` / ``p10`` / ``yaw_rate`` are garbage yields
   bit-identical model inputs in ``yaw`` mode (v8F-mini windows and RA-1 clips) and DIFFERENT ones in ``steer`` mode; a loader mutated to
   read ``steer`` again is caught.
4. THE LAUNCH CANNOT SILENTLY BE THE OTHER REPRESENTATION.  Wrong-units prior, missing tracks, a track that is not the clip's, mixed
   representations in one verdict: all REFUSED.
5. THE BANKED P10-yaw ARTIFACTS ARE WHAT THE LOADER COMPUTES.  The loader's P10 trajectory equals the fit script's banked predictions.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytestmark = pytest.mark.skipif(torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1",
                                reason="CPU-only test (set CUDA_VISIBLE_DEVICES=-1)")

from tanitad.data import ra1_pooled as RD  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import ra1_hjepa as R  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from tanitad.models.ego_history import ego_channels_from_poses  # noqa: E402
from tanitad.models.kinematic_prior import pose_kin3  # noqa: E402
import ra1_hjepa_eval as EV  # noqa: E402
import ra1_hjepa_train as TR  # noqa: E402
import build_native_tracks as BT  # noqa: E402

torch.set_num_threads(2)
RA = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
MINI_RAW = RA / "2026-10-17-v8f-mini" / "raw"
EVAL_KIN = MINI_RAW / "eval141_kin.npz"
needs_eval_kin = pytest.mark.skipif(not EVAL_KIN.exists(), reason="eval141_kin.npz not in this checkout")
TINY_FLAGS = ["--d-tok", "4", "--adapter-hidden", "16", "--d1", "16", "--d-up", "16", "--macro-dim", "8", "--enc-hidden", "16",
              "--pred-width", "16", "--pred-depth", "1", "--pred-heads", "2", "--readout-hidden", "16", "--sigreg-proj", "8",
              "--batch", "4", "--warmup", "1", "--log-every", "1", "--ckpt-every", "3"]
HOLD = ["--holdout-n", "2", "--holdout-pool", "6"]
ARM = {"R0": [], "A0": ["--beta", "0.08"], "A1": ["--beta", "0.08", "--gamma", "100"]}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def eval141_tracks():
    z = np.load(EVAL_KIN)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    return [(c, z["poses"][off[i]:off[i + 1]].astype(np.float64), z["actions"][off[i]:off[i + 1]].astype(np.float64))
            for i, c in enumerate(z["clip"].tolist())]


# ================================================================================================ 1. the spelling
def spelling_gap(fn, tracks) -> float:
    """max |fn(track)[j] - dyaw channel of ego_channels_from_poses at raw frame 2j| over every real clip and every grid step."""
    worst = 0.0
    for _c, P, _A in tracks:
        ref = ego_channels_from_poses(torch.as_tensor(P)[None], len(P), dt=0.1)[0, :, 2].numpy()[0::2]
        worst = max(worst, float(np.abs(fn(P) - ref).max()))
    return worst


def shifted_one_frame(P):
    """the deliberate regression: the yaw rate read ONE FRAME LATE (a forward difference at frame k = the backward one at k + 1)."""
    om = np.zeros(len(P))
    om[:-1] = wrap(P[1:, 2] - P[:-1, 2]) / 0.1
    return om[0::2]


@needs_eval_kin
def test_yaw_spelling_equals_ego_channels_dyaw_on_real_windows_and_a_one_frame_shift_fails():
    tr = eval141_tracks()
    assert len(tr) == 141
    assert spelling_gap(lambda P: D.yaw_kin_from_track(P)[1], tr) <= 1e-6                      # the compliant omega
    assert spelling_gap(lambda P: np.concatenate([[0.0], wrap(np.diff(P[:, 2])) / 0.1])[0::2], tr) <= 1e-6   # an independent numpy spelling
    assert spelling_gap(shifted_one_frame, tr) > 1e-3, "the comparator must go RED on a one-frame shift"


@needs_eval_kin
def test_kappa0_kin3_equals_pose_kin3_the_p10_pose_triple():
    for _c, P, _A in eval141_tracks()[:12]:
        kin, _om = D.yaw_kin_from_track(P)
        for j in range(2, len(kin), 7):
            ref = pose_kin3(torch.as_tensor(P)[None, :2 * j + 1], 2 * j + 1, dt=0.1)[0].numpy()
            assert np.abs(kin[j] - ref).max() <= 1e-12, (j, kin[j], ref)
    # window_kin(yaw) is that same triple, on the same frames
    c, P, _A = eval141_tracks()[0]
    ts = np.arange(3, 60, 5)
    kin, _gt, _g = D.window_kin(P, None, ts, "yaw")
    assert np.array_equal(kin, D.yaw_kin_from_track(P)[0][ts])


@needs_eval_kin
def test_yaw_mode_is_causal_and_the_forward_difference_is_not():
    c, P, _A = eval141_tracks()[3]
    kin0, om0 = D.yaw_kin_from_track(P)
    P2 = P.copy()
    P2[2 * 40 + 1:, :] += 1.234                                  # change EVERY pose after frame 80 (and shift the heading too)
    P2[2 * 40 + 1:, 2] += 0.3
    kin1, om1 = D.yaw_kin_from_track(P2)
    assert np.array_equal(kin0[:41], kin1[:41], equal_nan=True) and np.array_equal(om0[:41], om1[:41]), "a value at frame 2j read a later pose"
    assert not np.array_equal(kin0[41:], kin1[41:], equal_nan=True)   # control: the mutation did change the later rows
    # control: the cache's own `yaw_rate` convention (wrapped (yaw[2j+2]-yaw[2j])/0.2) FAILS this test -> it reads the first future pose
    f0, f1 = wrap(P[2 * 40 + 2, 2] - P[2 * 40, 2]), wrap(P2[2 * 40 + 2, 2] - P2[2 * 40, 2])
    assert f0 != f1, "the forward difference at frame 80 must see the future pose we changed"


@needs_eval_kin
def test_window_kin_yaw_never_reads_actions_and_steer_mode_does():
    c, P, A = eval141_tracks()[5]
    ts = np.arange(3, 60, 4)
    k0, g0, _ = D.window_kin(P, None, ts, "yaw")
    k1, g1, _ = D.window_kin(P, np.full_like(A, np.nan), ts, "yaw")
    assert np.array_equal(k0, k1) and np.array_equal(g0, g1)
    s0, _, _ = D.window_kin(P, A, ts, "steer")
    s1, _, _ = D.window_kin(P, A * 0.0 + 0.7, ts, "steer")
    assert not np.array_equal(s0[:, 2], s1[:, 2]), "control: the steer mode reads actions[:, 0]"
    assert np.abs(s0[:, 0] - k0[:, 0]).max() == 0.0             # v0 identical in both modes


@needs_eval_kin
def test_kappa0_retains_the_recorded_curvature_information():
    ks, ts = [], []
    for _c, P, A in eval141_tracks():
        kin, _om = D.yaw_kin_from_track(P)
        k = 2 * np.arange(len(kin))
        m = np.isfinite(kin).all(1) & (kin[:, 0] > 3.0)
        ks.append(kin[m, 2])
        ts.append((np.tan(A[k, 0]) / 2.9)[m])
    ks, ts = np.concatenate(ks), np.concatenate(ts)
    assert len(ks) > 10_000
    assert np.corrcoef(ks, ts)[0, 1] > 0.99, "the measured yaw rate must carry the information the steer channel carried"


# ================================================================================================ 3. the input paths never read steer
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("yawsyn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=3, T=70, seed=0, d_feat=32)
    w["root"] = str(root)
    yaw_prior = M.P10Prior.synthetic(0)
    M.save_p10(yaw_prior, root / "p10_yaw.pt", {"synthetic": True, "units": "pose_kappa"})
    M.save_p10(yaw_prior, root / "p10_steer.pt", {"synthetic": True, "units": "steer"})
    w["prior_yaw"], w["prior_steer"] = str(root / "p10_yaw.pt"), str(root / "p10_steer.pt")
    w["prior"] = M.load_p10(w["prior_yaw"])
    w["tracks_path"] = str(root / "native_tracks.pt")
    w["tracks"] = D.load_native_tracks(w["tracks_path"])
    w["grid_json"] = str(root / "eval_grid_windows.json")
    # THE TWIN: identical cache, but every steer-carrying array (and the forward yaw_rate) is garbage
    twin = Path(tmp_path_factory.mktemp("yawtwin"))
    rng = np.random.default_rng(7)
    for split in ("train", "eval"):
        (twin / split / "clips").mkdir(parents=True)
        (twin / split / "manifest.jsonl").write_text((Path(root) / split / "manifest.jsonl").read_text())
        for f in (Path(root) / split / "clips").glob("*.pt"):
            o = torch.load(f, weights_only=True)
            n = len(o["steer"])
            o["steer"] = torch.as_tensor(rng.normal(0, 0.3, n), dtype=torch.float32)
            o["kin3"] = torch.as_tensor(rng.normal(0, 3.0, (n, 3)), dtype=torch.float32)
            o["p10"] = torch.as_tensor(rng.normal(0, 3.0, (n, 8)), dtype=torch.float32)
            o["yaw_rate"] = torch.as_tensor(rng.normal(0, 1.0, n), dtype=torch.float32)
            torch.save(o, twin / split / "clips" / f.name)
    w["twin"] = str(twin)
    return w


def mini_items(world, cache, kin_mode, store_cls=D.ClipStore):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = store_cls(cache, "train", lru=8, kin_mode=kin_mode, tracks=world["tracks"] if kin_mode == "yaw" else None)
    wins = D.train_windows(store, J, world["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    sel = np.random.default_rng(0).choice(len(wins), 12, replace=False)
    return D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in sel])


def _same(x, y) -> bool:
    return torch.equal(torch.nan_to_num(x.float(), nan=-777.0), torch.nan_to_num(y.float(), nan=-777.0))


def batch_leaks(a: dict, b: dict) -> list:
    """the keys of two collated batches that differ (the leak detector: clean cache vs garbage-steer twin)."""
    bad = []
    for k in a:
        if isinstance(a[k], dict):
            bad += [f"{k}.{kk}" for kk in a[k] if not _same(a[k][kk], b[k][kk])]
        elif not _same(a[k], b[k]):
            bad.append(k)
    return bad


def test_v8f_mini_yaw_windows_ignore_the_steer_and_the_steer_mode_does_not(world):
    clean, twin = mini_items(world, world["root"], "yaw"), mini_items(world, world["twin"], "yaw")
    assert batch_leaks(clean, twin) == [], "kin_mode yaw: a model input depends on the cache's steer / kin3 / p10 / yaw_rate"
    s_clean, s_twin = mini_items(world, world["root"], "steer"), mini_items(world, world["twin"], "steer")
    leaked = batch_leaks(s_clean, s_twin)
    assert {"act", "p10f", "base"} <= set(leaked), f"CONTROL: the legacy mode must read the steer through act/p10f/base, saw {leaked}"
    # and the two modes really are different representations of the same windows
    assert not torch.equal(clean["act"][..., 1], s_clean["act"][..., 1]) and torch.equal(clean["act"][..., 0], s_clean["act"][..., 0])


def test_deliberate_regression_a_loader_that_reads_steer_again_is_caught(world):
    class Leaky(D.ClipStore):
        def get(self, clip):
            d = super().get(clip)
            o = torch.load(self.dir / f"{clip}.pt", weights_only=True)
            d["ch1"] = o["steer"].numpy()                                   # the regression: the recorded steer back into action channel 1
            return d

    clean, twin = mini_items(world, world["root"], "yaw", Leaky), mini_items(world, world["twin"], "yaw", Leaky)
    assert "act" in batch_leaks(clean, twin), "the guard must go RED when the steer is re-introduced"

    class LeakyBase(D.ClipStore):
        def get(self, clip):
            d = super().get(clip)
            o = torch.load(self.dir / f"{clip}.pt", weights_only=True)
            d["base"] = D.kdx_base(o["kin3"].double().numpy())              # the other route: the steer-integrated kd_x base
            return d

    clean, twin = mini_items(world, world["root"], "yaw", LeakyBase), mini_items(world, world["twin"], "yaw", LeakyBase)
    assert "base" in batch_leaks(clean, twin)


def ra1_clip(world, cache, kin_mode, clip="syn-train-001"):
    return RD.load_clip(Path(cache) / "train" / "clips" / f"{clip}.pt", world["prior"], "4x10", kin_mode,
                        world["tracks"][clip] if kin_mode == "yaw" else None)


def test_ra1_yaw_clip_is_bit_identical_under_garbage_steer_and_steer_mode_is_not(world):
    a, b = ra1_clip(world, world["root"], "yaw"), ra1_clip(world, world["twin"], "yaw")
    for k in ("act", "f8n", "p10_traj", "gt", "valid", "kin3"):
        assert _same(a[k], b[k]), f"RA-1 yaw: {k} depends on the cache's steer/kin3/p10/yaw_rate"
    assert torch.equal(a["pool"].float(), b["pool"].float())
    s, t = ra1_clip(world, world["root"], "steer"), ra1_clip(world, world["twin"], "steer")
    assert not torch.equal(s["act"], t["act"]) and not torch.equal(s["f8n"], t["f8n"]), "CONTROL: the legacy mode must read the cache's steer"
    assert torch.equal(a["act"][:, 0], s["act"][:, 0]) and torch.equal(a["act"][:, 2], s["act"][:, 2])       # a and v are the same channels
    assert not torch.equal(a["act"][:, 1], s["act"][:, 1])


def test_deliberate_regression_ra1_loader_that_reads_steer_again_is_caught(world, monkeypatch):
    real = RD.load_clip

    def leaky(path, prior, grid="4x10", kin_mode="steer", track=None):
        d = real(path, prior, grid, kin_mode, track)
        o = torch.load(path, weights_only=True)
        d["act"] = d["act"].clone()
        d["act"][:, 1] = o["steer"].float()                                  # the regression
        return d

    monkeypatch.setattr(RD, "load_clip", leaky)
    a, b = RD.load_clip(Path(world["root"]) / "train/clips/syn-train-001.pt", world["prior"], "4x10", "yaw", world["tracks"]["syn-train-001"]), \
        RD.load_clip(Path(world["twin"]) / "train/clips/syn-train-001.pt", world["prior"], "4x10", "yaw", world["tracks"]["syn-train-001"])
    assert not torch.equal(a["act"], b["act"]), "the guard must go RED when the steer is re-introduced"


def test_ra1_model_loss_is_bit_identical_in_yaw_mode_under_garbage_steer(world):
    cfg = R.RA1Config(arm="A1", d_feat=32, n_tok=40, d_tok=4, adapter_hidden=16, d1=16, d_up=16, macro_dim=8, enc_hidden=16, pred_width=16,
                      pred_depth=1, pred_heads=2, readout_hidden=16, sigreg_proj=8, beta=0.08, gamma=100.0, kin_mode="yaw")
    outs = []
    for cache in (world["root"], world["twin"]):
        bank = RD.ClipBank(cache, "train", world["prior"], "4x10", "yaw", world["tracks"])
        bank.load(world["names"]["train"])
        b = RD.sample_batch(bank, world["names"]["train"], 0, 0, 4, 19)
        am, asd = bank.act_stats()
        torch.manual_seed(0)
        m = R.build_model(cfg, am, asd)
        o = m.forward_batch(b, generator=torch.Generator().manual_seed(0), shuffle_generator=torch.Generator().manual_seed(0))
        outs.append((float(o["total"]), float(o["traj"]), am.clone()))
    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1] and torch.equal(outs[0][2], outs[1][2])
    assert m.act_channels == ("a", "yaw_rate", "v")


# ================================================================================================ 4. refusals
def argv(world, arm, out, mode="yaw", prior=None, tracks=True, extra=()):
    a = ["--arm", arm, "--cache", world["root"], "--p10-prior", prior or (world["prior_yaw"] if mode == "yaw" else world["prior_steer"]),
         "--out", str(out), "--steps", "3", "--kin-mode", mode] + TINY_FLAGS + ARM[arm] + HOLD + list(extra)
    return a + (["--tracks", world["tracks_path"]] if (tracks and mode == "yaw") else [])


@pytest.mark.parametrize("arm", ["R0", "A0", "A1"])
def test_ra1_trainer_runs_the_compliant_mode_and_the_gate_reads_it_back(world, tmp_path, arm):
    assert TR.main(argv(world, arm, tmp_path / arm)) == 0
    gate = json.load(open(tmp_path / arm / "launch_check.json"))
    rows = {r["lever"]: r for r in gate["G_DVB"]}
    assert gate["ok"] and rows["kin_mode_bank"]["built"] == "yaw" and rows["p10_units"]["built"] == "pose_kappa"
    assert rows["act_ch1_name"]["built"] == "yaw_rate" and rows["act_channels"]["ok"], rows
    assert json.load(open(tmp_path / arm / "meta.json"))["kin_mode"] == "yaw"
    assert torch.load(tmp_path / arm / "final.pt", weights_only=False)["cfg"]["kin_mode"] == "yaw"
    if arm != "R0":
        assert gate["G_LIVE"]["idm_ok"] and gate["G_LIVE"]["ok"]


def test_ra1_trainer_default_is_the_compliant_mode(world, tmp_path):
    ap = TR.build_parser()
    assert ap.get_default("kin_mode") == "yaw"
    a = [x for x in argv(world, "R0", tmp_path / "d") if x not in ("--kin-mode", "yaw")]
    assert TR.main(a) == 0
    assert json.load(open(tmp_path / "d" / "meta.json"))["kin_mode"] == "yaw"


def test_ra1_trainer_refuses_a_wrong_units_prior_missing_tracks_and_a_foreign_track(world, tmp_path, capsys):
    assert TR.main(argv(world, "R0", tmp_path / "u", prior=world["prior_steer"])) == 2                 # a steer-unit prior under yaw
    assert "pose_kappa" in capsys.readouterr().out
    assert TR.main(argv(world, "R0", tmp_path / "t", tracks=False)) == 2                               # no --tracks
    assert "--tracks" in capsys.readouterr().out
    assert TR.main(argv(world, "R0", tmp_path / "s", mode="steer", prior=world["prior_yaw"])) == 2     # a yaw prior under steer
    # a track that is not the clip's: swap two clips' tracks -> check_track_matches_grid raises before one number is taken from it
    bad = dict(world["tracks"])
    bad["syn-train-001"], bad["syn-train-002"] = bad["syn-train-002"], bad["syn-train-001"]
    with pytest.raises(ValueError, match="disagrees with the cache pose_grid"):
        RD.load_clip(Path(world["root"]) / "train/clips/syn-train-001.pt", world["prior"], "4x10", "yaw", bad["syn-train-001"])
    with pytest.raises(ValueError, match="needs the native pose tracks"):
        D.ClipStore(world["root"], "train", kin_mode="yaw")
    with pytest.raises(ValueError, match="kin_mode"):
        D.ClipStore(world["root"], "train", kin_mode="steering")


def test_a_legacy_steer_config_serialises_exactly_as_before_the_ruling():
    c = R.RA1Config(arm="R0", d_feat=32)
    assert "kin_mode" not in c.to_json()                                       # a pre-ruling checkpoint's cfg == this cfg's json
    assert R.RA1Config(**{**c.to_json()}).kin_mode == "steer"
    y = R.RA1Config(arm="R0", d_feat=32, kin_mode="yaw")
    assert y.to_json()["kin_mode"] == "yaw" and y.sha12() != c.sha12()
    with pytest.raises(ValueError, match="kin_mode"):
        R.validate_config(R.RA1Config(arm="R0", d_feat=32, kin_mode="steering"))


def test_g_dvb_flags_a_model_built_in_the_other_representation(world):
    cfg = R.RA1Config(arm="A0", d_feat=32, n_tok=40, d_tok=4, adapter_hidden=16, d1=16, d_up=16, macro_dim=8, enc_hidden=16, pred_width=16,
                      pred_depth=1, pred_heads=2, readout_hidden=16, sigreg_proj=8, beta=0.08, kin_mode="yaw")
    m = R.build_model(cfg, torch.zeros(3), torch.ones(3))
    assert all(r["ok"] for r in R.declared_vs_built(cfg, m))
    m.act_channels = R.ACT_CHANNELS_BY_MODE["steer"]
    assert [r["lever"] for r in R.declared_vs_built(cfg, m) if not r["ok"]] == ["act_channels"]


# ================================================================================================ eval + verdict (yaw), mixing refused
def run_eval(world, tmp_path, arm, seed, mode="yaw"):
    out = tmp_path / f"{arm}_{seed}"
    assert TR.main(argv(world, arm, out, mode, extra=["--seed", str(seed)])) == 0
    ev = out / "eval"
    rc = EV.main(["eval", "--ckpt", str(out / "final.pt"), "--eval-cache", world["root"], "--p10-prior",
                  world["prior_yaw"] if mode == "yaw" else world["prior_steer"], "--grid-json", world["grid_json"],
                  "--train-cache", world["root"], "--out", str(ev), "--n-boot", "30", "--fam-boot", "10"] +
                 (["--tracks", world["tracks_path"]] if mode == "yaw" else []))
    assert rc == 0
    return ev


def test_eval_and_verdict_end_to_end_in_the_compliant_mode_and_mixing_is_refused(world, tmp_path):
    evs = {(a, s): run_eval(world, tmp_path, a, s) for a, s in (("A0", 0), ("A0", 1), ("A1", 0), ("A1", 1))}
    r = json.load(open(evs[("A1", 0)] / "eval.json"))
    assert r["kin_mode"] == "yaw" and r["ctl_channel"] == "yaw_rate" and r["p10_units"] == "pose_kappa" and r["instrument_ok"]
    assert r["SENS"]["ctl_channel"] == "yaw_rate" and r["SENS"]["self_swap_control_ok"] and r["SENS"]["self_swap_control_max"] == 0.0
    assert r["SENS"]["channels"]["yaw_rate/k10"]["mean_of_ratios"] > 0.0 and "steer/k10" not in r["SENS"]["channels"]
    assert "yaw_rate_k10" in r["SENS"] and "steer_k10" not in r["SENS"]
    assert "MEASURED yaw rate" in r["K"]["note"]
    runs = [x for (a, s), ev in evs.items() for x in ("--run", f"{a}{':b0.08' if a == 'A0' else ':b0.08g100'}@{s}={ev}")]
    vout = tmp_path / "verdict.json"
    assert EV.main(["verdict", "--out", str(vout), "--n-boot", "30"] + runs) == 0
    v = json.load(open(vout))
    assert v["kin_mode"] == "yaw" and v["ctl_channel"] == "yaw_rate" and v["deciding"] is True
    assert "A1_yaw_rate_k10" in v["SENS"]["per_seed"]["seed0"] and "yaw-rate" not in v["SENS"]["bar"] and "yaw_rate-sensitivity" in v["SENS"]["bar"]
    # a steer-representation run is NON-DECIDING and may never sit in the same verdict as yaw runs
    ev_s = run_eval(world, tmp_path / "steer", "A1", 0, "steer") if (tmp_path / "steer").mkdir() is None else None
    rs = json.load(open(ev_s / "eval.json"))
    assert rs["kin_mode"] == "steer" and rs["ctl_channel"] == "steer" and "steer/k10" in rs["SENS"]["channels"]
    with pytest.raises(SystemExit, match="different kin_mode"):
        EV.main(["verdict", "--out", str(tmp_path / "mixed.json"), "--n-boot", "30", "--run", f"A1:b0.08g100@0={evs[('A1', 0)]}",
                 "--run", f"A1:b0.08g100@1={ev_s}"])
    # a steer-only verdict still works and says it does not decide
    assert EV.main(["verdict", "--out", str(tmp_path / "steer_only.json"), "--n-boot", "30", "--run", f"A1:b0.08g100@0={ev_s}"]) == 0
    vs = json.load(open(tmp_path / "steer_only.json"))
    assert vs["deciding"] is False and "NON-DECIDING" in vs["representation_note"]


# ================================================================================================ v8F-mini model + scripts
def test_v8f_mini_wm_channel_scale_follows_the_mode():
    for mode, scale in (("steer", 0.3), ("yaw", 0.4)):
        m = M.V8FMini(M.V8FMiniConfig(d=32, ff=64, n_q=2, depth_o=1, nb=6, wm="a", hmax=16, kin_mode=mode), M.P10Prior.synthetic(0))
        assert m.wm.ch1_scale == scale == M.CH1_SCALE[mode] and m.built_levers()["wm_ch1_scale"] == scale and m.built_levers()["kin_mode"] == mode
    with pytest.raises(ValueError, match="kin_mode"):
        M.V8FMiniConfig(kin_mode="steering")


def test_p10_units_are_checked_against_the_mode(world):
    assert M.check_p10_units(world["prior_yaw"], "yaw") == "pose_kappa"
    with pytest.raises(ValueError, match="declares units"):
        M.check_p10_units(world["prior_steer"], "yaw")
    with pytest.raises(ValueError, match="declares units"):
        M.check_p10_units(world["prior_yaw"], "steer")
    with pytest.raises(ValueError):
        M.load_p10(world["prior_steer"], "yaw")


# ================================================================================================ 5. the banked artifacts
BANKED = {"540": (RA / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw540", 540), "600": (MINI_RAW / "p10_yaw", 600)}


def eval_pseudo_cache(tmp_path):
    """The 141 eval clips as S4-format clips (pose_grid = the track's even frames; decoys in every steer-carrying field) so the
    ACTUAL loader (`ra1_pooled.load_clip`, yaw mode) produces the P10 trajectories the fit script banked."""
    g = json.load(open(MINI_RAW / "eval_grid_windows.json"))
    root = tmp_path / "pseudo"
    (root / "eval" / "clips").mkdir(parents=True)
    tracks = {}
    for c, P, _A in eval141_tracks():
        T = (len(P) + 1) // 2
        v = torch.as_tensor(P[0::2, 3], dtype=torch.float32)
        torch.save({"pool4x10": torch.zeros(T, 40, 8).to(torch.float8_e4m3fn), "v": v, "a": torch.zeros(T),
                    "pose_grid": torch.as_tensor(P[0::2], dtype=torch.float32), "steer": torch.full((T,), 9.0),
                    "kin3": torch.full((T, 3), 9.0), "p10": torch.full((T, 8), 9.0), "yaw_rate": torch.full((T,), 9.0)},
                   root / "eval" / "clips" / f"{c}.pt")
        tracks[c] = torch.as_tensor(P, dtype=torch.float32).double().numpy()
    return root, tracks, g


@needs_eval_kin
@pytest.mark.parametrize("which", ["540", "600"])
def test_banked_p10_yaw_artifact_equals_what_the_ra1_loader_computes(which, tmp_path):
    d, n = BANKED[which]
    if not (d / "p10_prior.pt").exists():
        pytest.skip(f"{d} not banked in this checkout")
    import hashlib
    rows = [ln.split() for ln in (d / "SHA256SUMS.txt").read_text().splitlines() if ln.strip()]
    assert {fn.lstrip("*") for _sha, fn in rows} == {"p10_prior.pt", "p10_fit.json", "p10_eval_pred6.npy"}
    for sha, fn in rows:
        assert hashlib.sha256((d / fn.lstrip("*")).read_bytes()).hexdigest() == sha, f"{fn}: bytes differ from SHA256SUMS.txt"
    meta = M.p10_meta(d / "p10_prior.pt")
    assert meta["units"] == "pose_kappa" and meta["kin_mode"] == "yaw" and meta["fit_clips"] == n
    prior = M.load_p10(d / "p10_prior.pt", "yaw")
    root, tracks, g = eval_pseudo_cache(tmp_path)
    ref = np.load(d / "p10_eval_pred6.npy")
    got, k = [], 0
    for c in g["order"]:
        cl = RD.load_clip(root / "eval" / "clips" / f"{c}.pt", prior, "4x10", "yaw", tracks[c])
        ts = g["windows"][c]
        got.append(cl["p10_traj"][ts].double().numpy())
        assert bool(cl["valid"][ts].all())
    got = np.concatenate(got)
    assert got.shape == ref.shape
    assert float(np.abs(got - ref).max()) <= 1e-3, "the loader's P10 trajectory differs from the fit script's banked prediction"
    # and the banked fit is the documented one: the compliant floor is within 0.02 m of the steer P10 on eval-141
    fit = json.load(open(d / "p10_fit.json"))
    assert abs(fit["yaw"]["ade6_m"] - fit["steer_control"]["ade6_m"]) < 0.02 and fit["steer_control"]["ade6_m"] == pytest.approx(
        {540: 2.5560, 600: 2.5744}[n], abs=2e-4), "the pipeline control must reproduce the banked steer P10"


# ================================================================================================ the queue
QUEUE = RA / "2026-10-17-refav1-hj-mini" / "code" / "ra1_box2_queue.cmd"


@pytest.mark.skipif(not QUEUE.exists(), reason="queue not in this checkout")
def test_queue_runs_the_compliant_representation_in_its_own_directory():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 50, "the queue must stay CRLF"
    t = raw.decode("ascii")
    lines = t.splitlines()
    assert 'set "KIN=--kin-mode yaw"' in lines and any(l.startswith('set "ARGS=--arm') and "%KIN%" in l for l in lines)
    calls = [l for l in lines if '"%TRAIN%"' in l or '"%EVAL%" eval' in l or '"%RIDGE%"' in l]
    assert len(calls) == 8 and all('--tracks "%TRACKS%"' in l for l in calls), \
        "4 trainer + 2 eval + 2 ridge-probe (PREREG_RA1_RIDGE) call lines, each with --tracks"
    assert sum('--tracks "%TRACKS%"' in l for l in lines) == len(calls), "--tracks appears on call lines only"
    assert 'if not defined RUNS set "RUNS=D:\\tanitad\\runs\\ra1_yaw"' in lines, "never the steer runs' directory (a steer eval.json would 'skip' a compliant job)"
    assert "p10_yaw540" in t and "p10_540" not in t.replace("p10_yaw540", "")
    assert "--kin-mode steer" not in t


@pytest.mark.skipif(os.name != "nt" or not QUEUE.exists(), reason="the queue is a Windows batch file")
def test_queue_dry_run_prints_the_compliant_flags_in_every_argv(tmp_path):
    import subprocess
    env = dict(os.environ, RA1_DRYRUN="1", REPO=str(REPO), RUNS=str(tmp_path / "runs"))
    r = subprocess.run(f'cmd /c ""{QUEUE}" gate"', env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    tr = [ln for ln in r.stdout.splitlines() if "ra1_hjepa_train.py" in ln]
    assert len(tr) == 6 and all("--kin-mode yaw" in ln and "native_tracks.pt" in ln and "p10_yaw540" in ln for ln in tr), tr


# ================================================================================================ the real S4 cache (when this machine has it)
REAL_CACHE = Path("C:/Users/Admin/refav1_hj_cache/test_cpu_fp32")
REAL_TRACKS = Path("C:/Users/Admin/tanitad-data/yawfix/native_tracks.pt")


@pytest.mark.skipif(not (REAL_CACHE.exists() and REAL_TRACKS.exists()),
                    reason="the local real S4 test cache / native_tracks.pt are not on this machine")
def test_real_s4_cache_clips_load_in_the_compliant_mode_and_read_neither_steer_nor_the_forward_yaw_rate():
    tracks = D.load_native_tracks(REAL_TRACKS)
    prior = M.P10Prior.synthetic(0)
    for split, clip in (("train", "8f387977-4086-466d-bba4-b04f8cdc0743"), ("train", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"),
                        ("eval", "dca157a0-20a9-464b-a74c-084b678e7be1")):
        path = REAL_CACHE / split / "clips" / f"{clip}.pt"
        o = torch.load(path, weights_only=True)
        assert D.check_track_matches_grid(tracks[clip], o["pose_grid"].double().numpy(), clip) >= 100
        d = RD.load_clip(path, prior, "4x10", "yaw", tracks[clip])
        om = torch.from_numpy(D.yaw_kin_from_track(tracks[clip])[1].astype(np.float32))
        assert torch.equal(d["act"][:, 1], torch.nan_to_num(om, nan=0.0))
        assert not torch.equal(d["act"][:, 1], o["steer"].float()) and not torch.equal(d["act"][:, 1], o["yaw_rate"].float())
        assert bool(d["valid"].any()) and bool(torch.isfinite(d["p10_traj"][d["valid"]]).all())


def manifest_style(tracks: dict) -> dict:
    """the TRAIN tracks as the v2 manifest builds them: raw frames 0 and 1 do not exist (NaN)."""
    out = {}
    for c, t in tracks.items():
        t = t.copy()
        t[:2] = np.nan
        out[c] = t
    return out


def test_train_style_tracks_with_nan_lead_rows_give_a_finite_action_vector_and_loss(world):
    tr = manifest_style(world["tracks"])
    clip = "syn-train-001"
    cl = RD.load_clip(Path(world["root"]) / "train/clips" / f"{clip}.pt", world["prior"], "4x10", "yaw", tr[clip])
    assert bool(torch.isfinite(cl["act"]).all()), "a NaN in the action vector would NaN every window that contains it"
    assert float(cl["act"][1, 1]) == 0.0 and not bool(cl["valid"][:3].any())      # the undefined step is filled, never `valid`
    store = D.ClipStore(world["root"], "train", lru=4, kin_mode="yaw", tracks=tr)
    d = store.get(clip)
    assert bool(np.isfinite(d["ch1"]).all()) and not np.isfinite(d["kin3"][1]).all() and not np.isfinite(d["p10f"][1]).all()
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    wins = [w for w in D.train_windows(store, J, world["names"]["train"]) if w[1] <= 4][:6]      # windows whose history contains step 1
    assert wins
    b = D.collate([D.V8FMiniWindows(store, J, wins, hmax=10).item(ci, t, 0, 0) for ci, t, _, _ in wins])
    assert bool(torch.isfinite(b["act"]).all())
    m = M.V8FMini(M.V8FMiniConfig(d=32, ff=64, n_q=2, depth_o=1, nb=6, wm="a", hmax=16, kin_mode="yaw"), world["prior"], *M.cons_stats(J))
    assert bool(torch.isfinite(m.wm(torch.randn(len(wins), 16, 40, 32), torch.nn.functional.pad(b["act"], (0, 0, 6, 0))[:, -16:])["total"]))
