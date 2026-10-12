"""Pins for SPEC_WPB_LADDER_A4 (CPU): the VMAX-SHUF identity, the VMAX-XSHUF row and the A4 scorer.

The eval harness is driven END TO END (`main()` of `ladder_eval.py` and of `ladder_eval_xshuf.py`) through a FAKE loader
/ model / dataset that reproduces the eval139 grid's layout -- K windows per episode, episodes in sha order,
`shuffle=False`, batch 8, hence ONE EPISODE PER BATCH -- and a per-clip speed label. The speed treatment is the REAL
`tanitad.train.refcv8_train.speed_input_treatment` (the launch tree's), so the identity below is the real code's.
Expectations are LITERALS, never an expression over the code under test; every reading test has a deliberate-regression
arm that must go RED.
Run:  PYTHONPATH=<tree>/stack;<tree>/taniteval  python -m pytest -q test_ladder_xshuf.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ladder_eval as LE  # noqa: E402  (the chain's harness, unchanged)
import ladder_eval_xshuf as LX  # noqa: E402
import ladder_score as LS  # noqa: E402  (the chain's scorer, unchanged)
import ladder_score_a4 as LS4  # noqa: E402
from tanitad.train.refcv8_train import speed_input_treatment  # noqa: E402  (the REAL eval treatment)

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
OLD_ROWS = ("base", "rc_off", "legal", "rc_shuf", "vmax_off", "vmax_shuf")

# ------------------------------------------------------------------ the fake world --------------------------------- #
#: episode id -> sha12 (sha order a < b < ... differs from id order on purpose) and the per-clip RAW speed (m/s, valid);
#: episode 103 ("eee...") has NO sidecar row, so the dataset emits (0.0, 0.0) for it (the item rule)
SHA = {101: "cccccccccccc", 102: "aaaaaaaaaaaa", 103: "eeeeeeeeeeee", 104: "bbbbbbbbbbbb", 105: "dddddddddddd",
       106: "ffffffffffff"}
SPEED = {102: (10.0, 1.0), 104: (10.0, 1.0), 101: (20.0, 1.0), 105: (30.0, 1.0), 106: (10.0, 1.0)}
T_PER_EP = 10          # 10 candidate windows per episode; window_set keeps K of them


class FakeDS(torch.utils.data.Dataset):
    def __init__(self, speed=None, n2_mode=False):
        self.episodes = [SimpleNamespace(episode_id=e) for e in sorted(SHA)]
        self.index = [(e_i, t) for e_i in range(len(self.episodes)) for t in range(T_PER_EP)]
        self._vis1_sha12 = dict(SHA)
        n = len(self.index)
        self.r8_join = SimpleNamespace(rel=SimpleNamespace(rows={"lat_v7id_a": np.arange(n) % 8,
                                                                 "lon_v7id": np.arange(n) % 6}))
        self._r8_rows = np.arange(n)
        self._spd = dict(SPEED if speed is None else speed)
        self.max_speed_enabled = True
        if n2_mode:              # a per-WINDOW source: no per-clip map -> the harness must pre-pass the loader
            self.r8_speed_mode, self._max_speed_by_sid = "n2", None
        else:
            self.r8_speed_mode, self._max_speed_by_sid = None, self._spd

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e_i, t = self.index[i]
        eid = self.episodes[e_i].episode_id
        v, ok = self._spd.get(eid, (0.0, 0.0))
        fut = torch.zeros(60, 2)
        fut[:, 0] = torch.arange(1, 61, dtype=torch.float32) * 1.0
        return {"pose_last": torch.tensor([0.0, 0.0, 0.0, 9.0 + 0.1 * t]), "future_poses_ext": fut,
                "future_valid_ext": torch.ones(60, dtype=torch.bool), "nav_cmd": torch.tensor(e_i % 3),
                "v_max_ms": torch.tensor(v, dtype=torch.float32), "v_max_valid": torch.tensor(ok, dtype=torch.float32),
                "r8_rc_raw": torch.tensor([float(i), float(e_i), 0.5]), "r8_rc_valid": torch.tensor(True)}


def _fake_modules():
    class SpeedCeilingFilter(torch.nn.Module):
        horizons, tick_s = HORIZONS, 0.1

        def forward(self, cand, v_limit_ms):
            return cand

    v6 = types.ModuleType("tanitad.refs.refcv6_selection")
    v6.SpeedCeilingFilter = SpeedCeilingFilter
    v6.planned_max_speed = lambda cand, horizons, tick_s: cand[..., 0].abs().amax(-1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.filt = SpeedCeilingFilter()
            self.cfg = SimpleNamespace(core=SimpleNamespace(trajectory=SimpleNamespace(horizons=HORIZONS)),
                                       refcv8=SimpleNamespace(speed_input="label"))
            self.r8_enabled = True

        def forward(self, v_max_ms=None, v_max_valid=None, rc=None):
            b = v_max_ms.shape[0]
            noise = torch.rand(b, 8, 2)                    # the SAMPLER: a function of the batch's manual seed
            traj = noise * 0.01
            traj[..., 0] += v_max_ms[:, None] * torch.as_tensor(SLOT_T, dtype=torch.float32)
            if rc is not None:
                traj[..., 1] += rc[:, :1] * 0.1
            vlim = torch.where(v_max_valid > 0, v_max_ms, torch.full_like(v_max_ms, float("inf")))
            self.filt(traj[:, None].expand(b, 3, 8, 2) * torch.tensor([1.0, 1.1, 1.2])[None, :, None, None], vlim)
            return {"traj": traj, "tacv6_lat_logits": traj[:, :, 0].clone(),
                    "tacv6_lon_logits": traj[:, :6, 1].clone()}

    def compute_losses_v3(model, b, device, mode=None):
        """the eval path of r8_before_forward, reduced to the three seams the rows touch (speed, RC, LEGAL)"""
        legal = bool(model._r8_legal_row)
        vm, vv = speed_input_treatment(b["v_max_ms"], b["v_max_valid"], None, training=False,
                                       eval_intervention=model._r8_speed_eval, legal=legal)
        rc = None
        if not model._r8_no_rc:
            rc, valid = b["r8_rc_raw"].float(), b["r8_rc_valid"].bool()
            if model._r8_rc_eval == "shuf":
                rc, valid = torch.roll(rc, 1, 0), torch.roll(valid, 1, 0)
            if legal:
                valid = torch.zeros_like(valid)
            rc = rc * valid[:, None].float()
        model(v_max_ms=vm, v_max_valid=vv, rc=rc)
        return {}

    tr = SimpleNamespace(compute_losses_v3=compute_losses_v3, V3Dataset=object, lan_dataset_class=lambda c: c)
    args = SimpleNamespace(pose_sync_sidecar=None, graft_lan=False, goal_str=False, mode="ref")
    state = {}

    def build_model(config, ckpt, device="cuda", strict=True):
        m = FakeModel()
        return m, m.cfg, args, {"state_dict": {"step": 3000, "missing": [], "unexpected": []}}

    L = types.ModuleType("tanitad.eval.refcv7_loader")
    L.trainer = lambda: tr
    L.load_config = lambda p: {"argv": ["--out", "/x/V-R8/run"]}
    L.build_model = build_model
    L.build_eval_dataset = lambda model, cfg, args_, config, dataset_cls=None: (state["ds"], None, {})
    rb = types.ModuleType("refb_labels")
    rb.waypoint_targets = lambda pl, fut, hz: fut[:, [h - 1 for h in hz], :2]
    return L, v6, rb, state


def run_harness(mod, monkeypatch, out_dir, rows, seeds="0,1", ds=None, k_per_ep=8, batch=8):
    import tanitad.eval
    import tanitad.refs
    L, v6, rb, state = _fake_modules()
    state["ds"] = FakeDS() if ds is None else ds
    monkeypatch.setitem(sys.modules, "tanitad.eval.refcv7_loader", L)
    monkeypatch.setattr(tanitad.eval, "refcv7_loader", L, raising=False)
    monkeypatch.setitem(sys.modules, "tanitad.refs.refcv6_selection", v6)
    monkeypatch.setattr(tanitad.refs, "refcv6_selection", v6, raising=False)
    monkeypatch.setitem(sys.modules, "refb_labels", rb)
    monkeypatch.setattr(sys, "argv", ["ladder_eval", "--run", str(out_dir.parent / "run"), "--expect-step", "3000",
                                      "--rows", rows, "--seeds", seeds, "--out-dir", str(out_dir), "--device", "cpu",
                                      "--workers", "0", "--batch", str(batch), "--k-per-ep", str(k_per_ep)])
    assert mod.main() == 0
    return out_dir


def load(p):
    z = np.load(str(p), allow_pickle=False)
    return {k: z[k] for k in z.files}


def same_arrays(a, b):
    if sorted(a) != sorted(b):
        return False
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if not (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)):
            return False
    return True


# ------------------------------------------------------------------ (a) the defect, reproduced ---------------------- #
def test_DEFECT_old_vmax_shuf_is_an_identity_when_every_batch_is_one_episode(tmp_path, monkeypatch):
    """RED arm of the instrument: on the eval139 layout the chain's VMAX-SHUF row equals `base` on EVERY array."""
    d = run_harness(LE, monkeypatch, tmp_path, "base,vmax_shuf")
    for s in (0, 1):
        b, sh = load(d / f"base_s{s}.npz"), load(d / f"vmax_shuf_s{s}.npz")
        assert same_arrays(b, sh) is True
        assert b["v_fed"].tolist() == [10.0] * 16 + [20.0] * 8 + [30.0] * 8 + [0.0] * 8 + [10.0] * 8


def test_the_roll_itself_is_live_it_is_the_layout_that_kills_it(tmp_path, monkeypatch):
    """CONTROL for the test above: two episodes per batch (K 4, batch 8) and the same roll moves the fed speed -- but
    only at episode seams, and only where the neighbour's value differs: 4 of 24 windows."""
    d = run_harness(LE, monkeypatch, tmp_path, "base,vmax_shuf", seeds="0", k_per_ep=4)
    b, sh = load(d / "base_s0.npz"), load(d / "vmax_shuf_s0.npz")
    assert int((b["v_fed"] != sh["v_fed"]).sum()) == 4
    assert np.flatnonzero(b["v_fed"] != sh["v_fed"]).tolist() == [8, 12, 16, 20]


def test_real_treatment_roll_on_one_clip_batch_is_the_identity():
    v = torch.full((8,), 13.888889, dtype=torch.float32)
    ok = torch.ones(8)
    vm, vv = speed_input_treatment(v, ok, None, training=False, eval_intervention="shuf")
    assert vm.tolist() == v.tolist() and vv.tolist() == [1.0] * 8


# ------------------------------------------------------------------ the donor rule (pure) -------------------------- #
GRID_SHA = [s for s in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc", "dddddddddddd", "eeeeeeeeeeee",
                        "ffffffffffff") for _ in range(8)]
OWN_V = [10.0] * 16 + [20.0] * 8 + [30.0] * 8 + [0.0] * 8 + [10.0] * 8
OWN_OK = [1.0] * 32 + [0.0] * 8 + [1.0] * 8


def test_donor_rule_literal():
    """a,b 10 -> c 20 (b is the same value, skipped); c 20 -> d 30; d 30 -> f 10 (e is UNKNOWN, skipped, counted);
    e unknown -> f 10; f 10 -> cyclic a, b same -> c 20."""
    p = LX.xshuf_plan(GRID_SHA, OWN_V, OWN_OK)
    assert p["donor_v"].tolist() == [20.0] * 16 + [30.0] * 8 + [10.0] * 8 + [10.0] * 8 + [20.0] * 8
    assert p["donor_ok"].tolist() == [1.0] * 48
    assert [str(s) for s in p["donor_sha"][::8]] == ["cccccccccccc", "cccccccccccc", "dddddddddddd", "ffffffffffff",
                                                      "ffffffffffff", "cccccccccccc"]
    assert (p["n_windows"], p["n_episodes"], p["n_changed"], p["n_kept_same"]) == (48, 6, 48, 0)
    assert (p["n_episodes_no_donor"], p["n_donor_candidates_skipped_unknown"]) == (0, 1)
    assert p["per_episode_constant"] is True
    assert p["transitions_kmh"] == {"108.0->36.0": 8, "36.0->72.0": 24, "72.0->108.0": 8, "unknown->36.0": 8}


def test_no_differing_donor_keeps_the_value_and_counts_it():
    p = LX.xshuf_plan(GRID_SHA, [10.0] * 48, [1.0] * 48)
    assert p["donor_v"].tolist() == [10.0] * 48
    assert (p["n_changed"], p["n_kept_same"], p["n_episodes_no_donor"]) == (0, 48, 6)
    assert [str(s) for s in p["donor_sha"][:2]] == ["", ""]


def test_MUT_an_in_episode_roll_is_not_the_donor_rule():
    """deliberate regression: the old rule (roll by one inside each 8-window batch) re-derived on the same grid
    changes NOTHING; the A4 rule changes all 48 -- the two can never be confused."""
    v = np.asarray(OWN_V, np.float32).reshape(6, 8)
    rolled = np.roll(v, 1, axis=1).reshape(-1)
    assert int((rolled != np.asarray(OWN_V, np.float32)).sum()) == 0
    assert LX.xshuf_plan(GRID_SHA, OWN_V, OWN_OK)["n_changed"] == 48


# ------------------------------------------------------------------ (b) the new row, end to end -------------------- #
def test_xshuf_row_feeds_the_donor_speed_and_keeps_the_pairing(tmp_path, monkeypatch):
    d = run_harness(LX, monkeypatch, tmp_path, "base,vmax_xshuf")
    for s in (0, 1):
        b, x = load(d / f"base_s{s}.npz"), load(d / f"vmax_xshuf_s{s}.npz")
        assert x["v_fed"].tolist() == [20.0] * 16 + [30.0] * 8 + [10.0] * 8 + [10.0] * 8 + [20.0] * 8
        assert x["v_valid"].tolist() == [1.0] * 48
        assert b["v_fed"].tolist() == [10.0] * 16 + [20.0] * 8 + [30.0] * 8 + [0.0] * 8 + [10.0] * 8
        # PAIRING: the sampler noise (traj y with the RC term identical) is the SAME draw on every window
        assert np.array_equal(b["traj"][..., 1], x["traj"][..., 1])
        # and the speed term moved by exactly donor - own on the x axis (literal deltas, m/s x slot time)
        dv = np.array([10.0] * 16 + [10.0] * 8 + [-20.0] * 8 + [10.0] * 8 + [10.0] * 8)
        assert np.allclose(x["traj"][..., 0] - b["traj"][..., 0], dv[:, None] * SLOT_T[None], atol=1e-4)
        assert x["win_sha12"].tolist() == b["win_sha12"].tolist() and x["win_t"].tolist() == b["win_t"].tolist()
        assert x["xshuf_donor_sha12"][::8].tolist() == ["cccccccccccc", "cccccccccccc", "dddddddddddd",
                                                         "ffffffffffff", "ffffffffffff", "cccccccccccc"]
        rec = json.loads((d / f"vmax_xshuf_s{s}.json").read_text(encoding="utf-8"))["xshuf"]
        assert (rec["n_changed"], rec["n_kept_same"], rec["n_episodes_no_donor"]) == (48, 0, 0)
        assert rec["n_changed_measured_vs_own_fed"] == 48 and rec["selfcheck"] == "PASS"
        assert rec["source"].startswith("sidecar map") and rec["channel_reached_forward"] is True
        bj = json.loads((d / f"base_s{s}.json").read_text(encoding="utf-8"))
        assert "xshuf" not in bj
        assert bj["window_sha12_t_sha256"] == json.loads((d / f"vmax_xshuf_s{s}.json").read_text(
            encoding="utf-8"))["window_sha12_t_sha256"]


def test_MUT_another_sampler_seed_breaks_the_pairing_check(tmp_path, monkeypatch):
    """the pairing assertion above can go RED: the same row at sampler seed 1 vs base at seed 0 draws other noise."""
    d = run_harness(LX, monkeypatch, tmp_path, "base,vmax_xshuf")
    assert not np.array_equal(load(d / "base_s0.npz")["traj"][..., 1], load(d / "vmax_xshuf_s1.npz")["traj"][..., 1])


def test_per_window_source_uses_the_loader_prepass_and_agrees(tmp_path, monkeypatch):
    d = run_harness(LX, monkeypatch, tmp_path, "vmax_xshuf", seeds="0", ds=FakeDS(n2_mode=True))
    x = load(d / "vmax_xshuf_s0.npz")
    assert x["v_fed"].tolist() == [20.0] * 16 + [30.0] * 8 + [10.0] * 8 + [10.0] * 8 + [20.0] * 8
    rec = json.loads((d / "vmax_xshuf_s0.json").read_text(encoding="utf-8"))["xshuf"]
    assert rec["source"].startswith("pre-pass over the eval loader") and rec["selfcheck"] == "PASS"


def test_MUT_a_plan_built_on_other_windows_is_refused(tmp_path, monkeypatch):
    """the per-batch assertion: if the loader's own speed is not the pre-pass value, nothing is substituted."""
    orig = LX.window_speeds

    def shifted(e_ds, idx):
        v, ok, src = orig(e_ds, idx)
        return np.roll(v, 8), np.roll(ok, 8), src
    monkeypatch.setattr(LX, "window_speeds", shifted)
    with pytest.raises(SystemExit, match="NOT the pre-pass value"):
        run_harness(LX, monkeypatch, tmp_path, "vmax_xshuf", seeds="0")


# ------------------------------------------------------------------ every OLD row is byte-identical ---------------- #
def test_every_existing_row_is_identical_between_the_chain_harness_and_the_A4_copy(tmp_path, monkeypatch):
    rows = ",".join(OLD_ROWS)
    a = run_harness(LE, monkeypatch, tmp_path / "chain", rows)
    b = run_harness(LX, monkeypatch, tmp_path / "a4", rows)
    n = 0
    for r in OLD_ROWS:
        for s in (0, 1):
            za, zb = load(a / f"{r}_s{s}.npz"), load(b / f"{r}_s{s}.npz")
            assert same_arrays(za, zb) is True, (r, s)
            ja = json.loads((a / f"{r}_s{s}.json").read_text(encoding="utf-8"))
            jb = json.loads((b / f"{r}_s{s}.json").read_text(encoding="utf-8"))
            ja.pop("compute")
            jb.pop("compute")
            assert ja == jb, (r, s)
            n += 1
    assert n == 12
    # the rows are not trivially equal to each other (the comparison above can see a difference)
    assert not same_arrays(load(a / "base_s0.npz"), load(a / "rc_off_s0.npz"))
    assert not same_arrays(load(a / "base_s0.npz"), load(a / "vmax_off_s0.npz"))


# ------------------------------------------------------------------ (c) the A4 scorer ------------------------------- #
N_EP, K = 40, 4
W = N_EP * K
V_OWN = np.repeat(np.where(np.arange(N_EP) % 2 == 0, 13.888889, 27.777779), K).astype(np.float32)
V_DONOR = np.repeat(np.where(np.arange(N_EP) % 2 == 0, 27.777779, 13.888889), K).astype(np.float32)


def straight(v, n):
    P = np.zeros((n, 8, 2))
    P[:, :, 0] = v * SLOT_T
    return P


def mkpass(P, gt, v_fed):
    return {"traj": np.asarray(P, np.float32), "gt": np.asarray(gt, np.float32), "gt_valid": np.ones((W, 8), bool),
            "v_lim": np.full(W, np.inf), "p_lat": np.tile(np.eye(8)[0], (W, 1)), "lat_v9": np.zeros(W, np.int64),
            "win_sha12": np.array([f"{i // K:012d}" for i in range(W)]), "v_fed": np.asarray(v_fed, np.float32),
            "v_valid": np.asarray(v_fed > 0, np.float32)}


def write_pass(root, arm, row, seed, z, extra=None):
    d = root / arm
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(d / f"{row}_s{seed}.npz"), **z)
    rec = {"window_sha12_t_sha256": "D"}
    if extra:
        rec.update(extra)
    (d / f"{row}_s{seed}.json").write_text(json.dumps(rec), encoding="utf-8")


def l4_world(tmp_path, xshuf="donor", xshuf_n=W, off_speed=10.0, xshuf_speed=10.0, xshuf_rec=True, base_speed=10.0):
    """GT straight at 10 m/s. V-R8 (and its replicate) drive it exactly; V-VSHUF drives 12 m/s (bar 1 clears); VMAX-OFF
    drives `off_speed`, VMAX-XSHUF `xshuf_speed`; VMAX-SHUF is a COPY of base (the identity). `xshuf` = "donor" feeds
    the other speed value on every window, "identity" copies base's v_fed (the blind row)."""
    w = tmp_path / "W"
    gt = straight(10.0, W)
    base = mkpass(straight(base_speed, W), gt, V_OWN)
    for s in (0, 1):
        for arm, P in (("V0", straight(9.0, W)), ("V0r", straight(9.0, W)), ("V-R8r", straight(base_speed, W)),
                       ("V-VSHUF", straight(12.0, W))):
            write_pass(w / "eval", arm, "base", s, mkpass(P, gt, V_OWN))
        write_pass(w / "eval", "V-R8", "base", s, base)
        write_pass(w / "eval", "V-R8", "vmax_shuf", s, dict(base))
        write_pass(w / "eval", "V-R8", "vmax_off", s, mkpass(straight(off_speed, W), gt, np.zeros(W)))
        vx = V_DONOR if xshuf == "donor" else V_OWN
        write_pass(tmp_path / "a4eval", "V-R8", "vmax_xshuf", s, mkpass(straight(xshuf_speed, W), gt, vx),
                   extra=({"xshuf": {"n_changed": xshuf_n, "selfcheck": "PASS", "donor_map_sha256": "M"}}
                          if xshuf_rec else None))
    (w / "I0.json").write_text(json.dumps({"PASS": True}), encoding="utf-8")
    return w, tmp_path / "a4eval"


def test_A4_L4_clears_with_a_live_control(tmp_path):
    w, x = l4_world(tmp_path)
    res = LS4.score(w, extra_eval=[x])
    l4 = res["rungs"]["L4"]
    assert l4["status"] == "SCORED" and l4["VERDICT"] == "CLEARS"
    assert [c["PASS"] for c in l4["bars"]] == [True, True, True]
    assert res["a4"]["instrument_blind"] is False
    assert res["a4"]["xshuf_control"][0]["n_v_fed_differs_from_base"] == 160
    assert res["a4"]["vmax_shuf"][0]["status"] == "STRUCTURAL-IDENTITY (A4)"
    assert res["a4"]["x3_c3_reported"]["reading"] == "NOT-SEPARATED-BETTER"


def test_MUT_xshuf_row_equal_to_base_is_INSTRUMENT_BLIND_never_a_pass(tmp_path):
    """deliberate regression: the control row carries base's own speed (the A4 defect, re-created) -> RED."""
    w, x = l4_world(tmp_path, xshuf="identity")
    l4 = LS4.score(w, extra_eval=[x])["rungs"]["L4"]
    assert l4["VERDICT"].startswith("INSTRUMENT-BLIND (A4)")
    assert l4["bars"][1]["PASS"] is None


def test_MUT_a_record_that_changed_nothing_is_INSTRUMENT_BLIND(tmp_path):
    w, x = l4_world(tmp_path, xshuf="identity", xshuf_n=0)
    assert LS4.score(w, extra_eval=[x])["rungs"]["L4"]["VERDICT"].startswith("INSTRUMENT-BLIND (A4)")


def test_MUT_a_row_without_its_xshuf_record_is_INSTRUMENT_BLIND(tmp_path):
    w, x = l4_world(tmp_path, xshuf_rec=False)
    assert LS4.score(w, extra_eval=[x])["rungs"]["L4"]["VERDICT"].startswith("INSTRUMENT-BLIND (A4)")


def test_the_A4_bar_can_FAIL_when_another_clips_speed_helps(tmp_path):
    """a live leak: the xshuf row (exact) beats VMAX-OFF (0.05 m/s slow) on every window -> separated better -> bar 2
    FAILS, while bar 3 (off - base = 0.05 <= 0.10) still passes."""
    w, x = l4_world(tmp_path, off_speed=9.95, xshuf_speed=10.0)
    l4 = LS4.score(w, extra_eval=[x])["rungs"]["L4"]
    assert [c["PASS"] for c in l4["bars"]] == [True, False, True]
    assert l4["VERDICT"] == "FAILS"


def test_DEFECT_the_chain_scorer_reads_base_not_a_shuffle__false_pass_and_false_fail(tmp_path):
    """the registered bar 2 compares VMAX-SHUF (== base on this grid) with VMAX-OFF, so it answers "does the KNOWN
    speed beat unknown", not "does ANOTHER CLIP'S speed beat unknown". Both errors, on literal worlds:"""
    # false PASS: another clip's speed (exact) beats unknown (0.05 m/s slow) -- a leak -- while base is no better
    w, x = l4_world(tmp_path / "fp", base_speed=9.95, off_speed=9.95, xshuf_speed=10.0)
    assert LS.score(w)["rungs"]["L4"]["VERDICT"] == "CLEARS"
    assert LS4.score(w, extra_eval=[x])["rungs"]["L4"]["VERDICT"] == "FAILS"
    # false FAIL: another clip's speed reads exactly like unknown (no leak), but base beats unknown
    w, x = l4_world(tmp_path / "ff", off_speed=9.95, xshuf_speed=9.95)
    assert LS.score(w)["rungs"]["L4"]["VERDICT"] == "FAILS"
    assert LS4.score(w, extra_eval=[x])["rungs"]["L4"]["VERDICT"] == "CLEARS"


def test_vmax_shuf_that_is_not_an_identity_is_reported_as_such(tmp_path):
    w, x = l4_world(tmp_path)
    z = load(w / "eval" / "V-R8" / "vmax_shuf_s1.npz")
    z["v_fed"] = V_DONOR
    np.savez_compressed(str(w / "eval" / "V-R8" / "vmax_shuf_s1.npz"), **z)
    a4 = LS4.score(w, extra_eval=[x])["a4"]
    assert a4["vmax_shuf"][0]["status"] == "STRUCTURAL-IDENTITY (A4)"
    assert a4["vmax_shuf"][1]["status"].startswith("NOT-IDENTICAL") and a4["vmax_shuf"][1]["differing"] == ["v_fed"]


def test_a_pass_found_in_two_roots_is_refused(tmp_path):
    w, x = l4_world(tmp_path)
    write_pass(w / "eval", "V-R8", "vmax_xshuf", 0, load(x / "V-R8" / "vmax_xshuf_s0.npz"))
    with pytest.raises(SystemExit, match="found twice"):
        LS4.score(w, extra_eval=[x])


def test_cli_refuses_to_write_the_chains_LADDER_SCORE(tmp_path, monkeypatch):
    w, x = l4_world(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ladder_score_a4", "score", "--w", str(w), "--extra-eval", str(x),
                                      "--out", str(w / "LADDER_SCORE.json")])
    with pytest.raises(SystemExit, match="refusing to write LADDER_SCORE.json"):
        LS4.main()
    assert not (w / "LADDER_SCORE.json").exists()
    monkeypatch.setattr(sys, "argv", ["ladder_score_a4", "score", "--w", str(w), "--extra-eval", str(x),
                                      "--out", str(tmp_path / "o" / "LADDER_SCORE_A4.json")])
    assert LS4.main() == 0
    got = json.loads((tmp_path / "o" / "LADDER_SCORE_A4.json").read_text(encoding="utf-8"))
    assert got["rungs"]["L4"]["VERDICT"] == "CLEARS" and not (w / "LADDER_SCORE.json").exists()


def test_every_other_rung_reads_identically_under_the_A4_scorer(tmp_path):
    """the A4 scorer changes L4 bar 2 only: on a world with the L1 arms AND the L4 arms, every rung but L4 and every
    table is equal to the chain scorer's output."""
    w, x = l4_world(tmp_path)
    gt = straight(10.0, W)
    for s in (0, 1):
        for arm in ("V-R8d",):
            write_pass(w / "eval", arm, "base", s, mkpass(straight(9.0, W), gt, V_OWN))
        for r in ("legal", "rc_off", "rc_shuf"):
            write_pass(w / "eval", "V-R8", r, s, mkpass(gt, gt, V_OWN))
    old = LS._json_safe(LS.score(w))
    new = LS4._json_safe(LS4.score(w, extra_eval=[x]))
    assert old["rungs"]["L1"]["status"] == "SCORED"
    for k in old:
        if k in ("rungs", "passes", "tables"):
            continue
        assert old[k] == new[k], k
    assert {k: v for k, v in old["rungs"].items() if k != "L4"} == {k: v for k, v in new["rungs"].items() if k != "L4"}
    assert set(new["passes"]) - set(old["passes"]) == {"V-R8/vmax_xshuf/s0", "V-R8/vmax_xshuf/s1"}
    assert all(old["tables"][k] == new["tables"][k] for k in old["tables"])
    assert set(new) - set(old) == {"a4"}
