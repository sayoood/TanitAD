"""``--nav-from-v7`` on ``scripts/refc_v3_train.py`` — the nav SOURCE switch.

E-ARCH-NAVSRC-1 (2026-09-02): refcv3's ``nav_cmd`` came from
``refb_labels.nav_command`` (net yaw over the next 15-25 s of FUTURE poses,
valid only with >= 15 s of future), which feeds ``follow``+invalid on 94.6 %
of B1's 19.9 s windows and agrees with the v7.2 ``nav_command`` token on only
65.5 % (MEASURED; package ``TanitAD Research Lab/Architecture & Inference/
Research/2026-09-02-nav-source-agreement/RESULT.md``). The flag feeds the v7.2
token instead — the input refav1 already trains on (``refav1_loader.py``).

Pinned here, one test per discipline:
(a) OFF IDENTITY — the LIVE run resumes through this file: with the flag off,
    ``nav_cmd``/``nav_valid`` equal ``refb_labels.nav_command(poses, t+w-1)``
    on every sampled window, and the sample is DIVERSE (both validities and at
    least two commands appear), so the identity has teeth.
(b) ON — all three tokens land on the POSITION-pinned index
    (``NAV_COMMAND_TOKENS[i]`` -> ``refb.NAV_COMMANDS[i]``) with ``nav_valid``
    True, and every OTHER field (``route_target``/``route_valid`` — the aux
    TARGET — ``goal_tac``, ``future_poses_ext``, ``lat_v7``/``lon_v7``, frames)
    is bit-identical to a flag-OFF twin over the same episodes.
(c) MISSING RECORD — ``NAV_FOLLOW`` + ``nav_valid`` False (refav1_loader's
    convention), counted in the init stats and printed; a join of ZERO clips is
    refused (wrong blob).
(d) THE PIN FIRES — with ``refb.NAV_COMMANDS`` re-ordered, ``enable_nav_from_v7``
    and the preflight refuse; the trainer's local ``NAV_TOKEN_TO_LEGACY`` equals
    ``refav1_loader._NAV_TOKEN_TO_LEGACY``; a manifest without the oracle stamp
    is refused; an unknown / absent token is refused, never defaulted.
(e) ARGPARSE + STAMP — the flag parses (default False); a 1-step smoke
    ``train()`` writes ``config.json`` with ``nav_from_v7`` /
    ``nav_cmd_derivation`` / ``nav_from_v7_stats`` / ``v7_labels`` in BOTH
    states; the mis-specified launches (no ``--v7-labels``; eval on without
    ``--eval-labels``) are refused at start.
CPU-only, synthetic data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refb_labels                                           # noqa: E402
import refc_v3_train as T                                    # noqa: E402
from tanitad.data import v7_labels as v7l                    # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id        # noqa: E402
from tanitad.models.vocab_v7 import NAV_COMMAND_TOKENS       # noqa: E402
from tanitad.refs import refb                                # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

#: the mapping under test, spelled out independently of the trainer's constant
TOK2LEGACY = {"NAV_FOLLOW_ROAD": "follow", "NAV_TURN_L": "left",
              "NAV_TURN_R": "right"}


# ---------------------------------------------------------------- the rig
def _cfg():
    return v3.refc_v3_smoke_config(True)


def _episodes(n, *, min_frames=40, seed=0, ids=None):
    """``_synth_episodes`` as the trainer builds them; ``ids`` replaces the
    string ids with the INTEGER stable ids the v7.2 join keys on."""
    cfg = _cfg()
    eps = T._synth_episodes(n, cfg.core, seed=seed, min_frames=min_frames)
    if ids is not None:
        assert len(ids) == n
        for ep, sid in zip(eps, ids):
            ep.episode_id = sid
    return cfg, eps


def _dataset(cfg, eps):
    return T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                       channels=cfg.core.encoder.in_channels)


def _label(clip_id, nav_token, *, t0_s=2.0, band=(0.0, 4.0), with_nav=True):
    """A real ``V7Label`` (frozen dataclass) with a v7.2-shaped nav record."""
    nav = ({"token": nav_token, "provenance": "ego-future", "oracle": True,
            "args": {"distance_m": 50.0, "time_s": 6.0}} if with_nav else None)
    return v7l.V7Label(
        clip_id=clip_id,
        tac_lat=v7l.HEADS["tac_lat"][0], tac_lon=v7l.HEADS["tac_lon"][0],
        str_action=v7l.HEADS["str_action"][0],
        str_goal=v7l.HEADS["str_goal"][0],
        tac_anchor=None, bands={"tactical_s": list(band)}, t0_s=t0_s,
        horizon={}, audit={}, _oracle={"nav_command": nav})


def _manifest(n, allow=True):
    return v7l.LabelManifest(path="synthetic", md5="0" * 32, n_records=n,
                             schema_version=v7l.EXPECTED_SCHEMA,
                             vocab=v7l.EXPECTED_VOCAB, allow_oracle_nav=allow)


def _windows_of(ds, e_i, k=3):
    """``k`` window indices of episode ``e_i``, spread over its span."""
    idx = [i for i, row in enumerate(ds.index) if int(row[0]) == e_i]
    assert idx, f"episode {e_i} has no windows"
    step = max(1, len(idx) // k)
    return idx[::step][:k]


# ---------------------------------------------------------- (a) OFF identity
def test_flag_off_nav_cmd_is_byte_identical_to_refb_v1():
    """The live run resumes through this class with the flag OFF."""
    cfg, eps = _episodes(3, min_frames=420, seed=1)
    ds = _dataset(cfg, eps)
    assert ds.nav_from_v7 is False and ds.v7_by_sid is None
    n = len(ds)
    picks = sorted(set(list(range(0, n, max(1, n // 24))) + [n - 1]))
    cmds, valids = set(), set()
    for i in picks:
        e_i, t = (int(x) for x in ds.index[i])
        t_last = t + ds.window - 1
        cmd, valid = refb_labels.nav_command(eps[e_i].poses, t_last)
        item = ds[i]
        assert item["nav_cmd"].dtype == torch.long
        assert int(item["nav_cmd"]) == cmd
        assert bool(item["nav_valid"]) == valid
        # the route TARGET is the v2.1 labeler's, on either nav path
        r = refb_labels.route_from_future_v21(eps[e_i].poses, t_last)
        assert int(item["route_target"]) == int(r["route"])
        assert bool(item["route_valid"]) == bool(r["valid"])
        cmds.add(cmd)
        valids.add(valid)
    # the identity must have teeth: both validities and >= 2 commands seen
    assert valids == {True, False}, valids
    assert len(cmds) >= 2, cmds


# ---------------------------------------------------- (b) ON, position pin
def test_flag_on_maps_all_three_tokens_position_pinned_and_touches_nothing_else(
        capsys):
    ids = [1001, 1002, 1003]
    cfg, eps = _episodes(3, min_frames=60, seed=2, ids=ids)
    ref = _dataset(cfg, eps)          # flag-OFF twin over the SAME episodes
    ds = _dataset(cfg, eps)
    labels = {sid: _label(f"clip-{sid}", tok)
              for sid, tok in zip(ids, NAV_COMMAND_TOKENS)}
    ds.v7_by_sid = labels
    ref.v7_by_sid = labels            # same tactical join; ONLY nav differs
    stats = ds.enable_nav_from_v7(_manifest(3))
    assert ds.nav_from_v7 is True and ref.nav_from_v7 is False
    assert stats["n_clips"] == 3 and stats["missing"] == 0
    assert (stats["follow"], stats["left"], stats["right"]) == (1, 1, 1)
    assert stats["derivation"] == T.NAV_FROM_V7_DERIVATION
    assert stats["allow_oracle_nav"] is True
    out = capsys.readouterr().out
    assert "nav_from_v7: follow 1 / left 1 / right 1, missing 0" in out
    for e_i, tok in enumerate(NAV_COMMAND_TOKENS):
        expect = refb.NAV_COMMANDS.index(TOK2LEGACY[tok])
        assert expect == NAV_COMMAND_TOKENS.index(tok)   # the pin itself
        for i in _windows_of(ds, e_i):
            item, twin = ds[i], ref[i]
            assert set(item) == set(twin)
            assert item["nav_cmd"].dtype == torch.long
            assert int(item["nav_cmd"]) == expect
            assert bool(item["nav_valid"]) is True
            for k in twin:
                if k in ("nav_cmd", "nav_valid"):
                    continue
                a, b = item[k], twin[k]
                if torch.is_tensor(a):
                    assert torch.equal(a, b), k
                else:
                    assert a == b, k


# -------------------------------------------------- (c) missing record
def test_clip_without_record_feeds_follow_invalid_and_is_counted(capsys):
    ids = [2001, 2002]
    cfg, eps = _episodes(2, min_frames=60, seed=3, ids=ids)
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {2001: _label("clip-2001", "NAV_TURN_R")}   # 2002: none
    stats = ds.enable_nav_from_v7(_manifest(1))
    assert stats["missing"] == 1 and stats["n_clips"] == 2
    assert (stats["follow"], stats["left"], stats["right"]) == (0, 0, 1)
    assert "missing 1 (of 2 clips" in capsys.readouterr().out
    for i in _windows_of(ds, 1):
        item = ds[i]
        assert int(item["nav_cmd"]) == refb_labels.NAV_FOLLOW == 0
        assert item["nav_cmd"].dtype == torch.long
        assert bool(item["nav_valid"]) is False
        # the tactical join marks the same clip unlabeled — consistent
        assert int(item["lat_v7"]) == v7l.IGNORE_ID
    for i in _windows_of(ds, 0):
        item = ds[i]
        assert int(item["nav_cmd"]) == refb.NAV_COMMANDS.index("right") == 2
        assert bool(item["nav_valid"]) is True


def test_joined_zero_clips_is_refused_as_wrong_blob():
    cfg, eps = _episodes(2, min_frames=60, seed=3, ids=[3001, 3002])
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {9999: _label("clip-9999", "NAV_TURN_L")}
    with pytest.raises(ValueError, match="joined ZERO"):
        ds.enable_nav_from_v7(_manifest(1))
    assert ds.nav_from_v7 is False


# ------------------------------------------------------- (d) the pin fires
def test_alignment_pin_passes_today_and_equals_refav1_loader_pin():
    T.assert_nav_token_alignment()
    from tanitad.data import refav1_loader
    assert T.NAV_TOKEN_TO_LEGACY == refav1_loader._NAV_TOKEN_TO_LEGACY
    assert T.NAV_TOKEN_TO_LEGACY == TOK2LEGACY
    assert tuple(NAV_COMMAND_TOKENS) == ("NAV_FOLLOW_ROAD", "NAV_TURN_L",
                                         "NAV_TURN_R")
    assert tuple(refb.NAV_COMMANDS)[:3] == ("follow", "left", "right")


def test_alignment_pin_fires_when_NAV_COMMANDS_order_differs(monkeypatch):
    monkeypatch.setattr(refb, "NAV_COMMANDS",
                        ("left", "follow", "right", "straight"))
    with pytest.raises(AssertionError, match="alignment broken"):
        T.assert_nav_token_alignment()
    cfg, eps = _episodes(1, min_frames=60, seed=4, ids=[4001])
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {4001: _label("clip-4001", "NAV_FOLLOW_ROAD")}
    with pytest.raises(AssertionError, match="alignment broken"):
        ds.enable_nav_from_v7(_manifest(1))
    assert ds.nav_from_v7 is False    # a refused enable leaves the v1 path on


def test_alignment_pin_fires_when_the_core_tuple_diverges(monkeypatch):
    from tanitad.refs import refc
    monkeypatch.setattr(refc, "NAV_COMMANDS", ("follow", "left", "right"))
    with pytest.raises(AssertionError, match="refc.NAV_COMMANDS"):
        T.assert_nav_token_alignment()


def test_oracle_stamp_and_the_join_are_required():
    cfg, eps = _episodes(1, min_frames=60, seed=5, ids=[5001])
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {5001: _label("clip-5001", "NAV_TURN_L")}
    with pytest.raises(v7l.OracleNavRefused):
        ds.enable_nav_from_v7(_manifest(1, allow=False))
    with pytest.raises(ValueError, match="v7_by_sid is None"):
        _dataset(cfg, eps).enable_nav_from_v7(_manifest(1))


def test_unknown_or_absent_token_is_refused_not_defaulted():
    cfg, eps = _episodes(1, min_frames=60, seed=6, ids=[6001])
    ds = _dataset(cfg, eps)
    ds.v7_by_sid = {6001: _label("clip-6001", "NAV_U_TURN")}
    with pytest.raises(ValueError, match="no legacy mapping"):
        ds.enable_nav_from_v7(_manifest(1))
    ds.v7_by_sid = {6001: _label("clip-6001", None, with_nav=False)}
    with pytest.raises(ValueError, match="no legacy mapping"):
        ds.enable_nav_from_v7(_manifest(1))


# ---------------------------------------------------- (e) argparse + stamp
def _argv(out, *extra):
    return ["--arm", "hier", "--out", str(out), "--smoke",
            "--synth-episodes", "2", "--steps", "1", "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "1",
            *extra]


def test_argparse_flag_parses_and_defaults_off(tmp_path):
    ap = T.build_parser()
    assert ap.parse_args(_argv(tmp_path)).nav_from_v7 is False
    assert ap.parse_args(_argv(tmp_path, "--nav-from-v7")).nav_from_v7 is True


def test_misspecified_switch_is_refused_at_start(tmp_path):
    ap = T.build_parser()
    with pytest.raises(SystemExit, match="needs --v7-labels"):
        T._check_nav_from_v7_args(ap.parse_args(_argv(tmp_path,
                                                      "--nav-from-v7")))
    base = _argv(tmp_path, "--nav-from-v7", "--v7-labels", "x.jsonl.gz")
    with pytest.raises(SystemExit, match="needs --eval-labels"):
        T._check_nav_from_v7_args(ap.parse_args(
            base + ["--eval-cache", "e", "--eval-every", "500"]))
    # eval OFF (eval-every 0) or eval labels supplied: fine
    T._check_nav_from_v7_args(ap.parse_args(base + ["--eval-cache", "e"]))
    T._check_nav_from_v7_args(ap.parse_args(
        base + ["--eval-cache", "e", "--eval-every", "500",
                "--eval-labels", "y"]))
    T._check_nav_from_v7_args(ap.parse_args(_argv(tmp_path)))   # off: no-op
    # and train() itself refuses BEFORE touching data or a device
    with pytest.raises(SystemExit, match="needs --v7-labels"):
        T.train(ap.parse_args(_argv(tmp_path, "--nav-from-v7")))
    assert not (tmp_path / "config.json").exists()


def _write_blob(path: Path, clip_ids, tokens) -> str:
    """A minimal v7.2-shaped label blob ``load_v7_labels`` accepts."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for cid, tok in zip(clip_ids, tokens):
            fh.write(json.dumps({
                "clip_id": cid, "schema_version": v7l.EXPECTED_SCHEMA,
                "vocab": v7l.EXPECTED_VOCAB,
                "a_tac": {"lat": v7l.HEADS["tac_lat"][0],
                          "lon": v7l.HEADS["tac_lon"][0]},
                "a_str": {"token": v7l.HEADS["str_action"][0]},
                "g_str": {"token": v7l.HEADS["str_goal"][0]},
                "g_tac": {}, "bands": {"tactical_s": [0.0, 4.0]},
                "t0_s": 2.0, "horizon": {},
                "nav_command": {"token": tok, "provenance": "ego-future",
                                "oracle": True,
                                "args": {"distance_m": 50.0, "time_s": 6.0}},
            }) + "\n")
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("flag_on", [False, True])
def test_one_step_smoke_train_stamps_the_nav_source(tmp_path, monkeypatch,
                                                    flag_on):
    clip_ids = ["clip-A", "clip-B"]
    sids = [stable_episode_id(c) for c in clip_ids]
    real = T._synth_episodes

    # ⚠️ `clip_ids` accepted and IGNORED on purpose. As of 2026-09-10
    # `_synth_episodes` stamps real join ids natively (it had to: the trainer
    # died on `int('synth-000')`, so no v7-label channel could be smoked at
    # all). This shim predates that and pins the ids EXPLICITLY, which is the
    # stronger assertion for THIS test — so it keeps overriding them, and only
    # needs to tolerate the new keyword the caller now passes.
    def _with_ids(n, cfg, seed=0, min_frames=40, clip_ids=None):
        eps = real(n, cfg, seed=seed, min_frames=min_frames)
        for ep, sid in zip(eps, sids):
            ep.episode_id = sid           # the join key the trainer uses
        return eps

    monkeypatch.setattr(T, "_synth_episodes", _with_ids)
    blob = tmp_path / "s2_labels_v7.2_synth.jsonl.gz"
    md5 = _write_blob(blob, clip_ids, ["NAV_TURN_L", "NAV_TURN_R"])
    out = tmp_path / ("on" if flag_on else "off")
    extra = ["--v7-labels", str(blob)] + (["--nav-from-v7"] if flag_on
                                          else [])
    args = T.build_parser().parse_args(_argv(out, *extra))
    T.train(args)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["nav_from_v7"] is flag_on
    assert cfg["v7_labels"]["md5"] == md5
    assert cfg["v7_labels"]["allow_oracle_nav"] is True
    assert cfg["tac_vocab_version"] == "v7.0"
    if flag_on:
        assert cfg["nav_cmd_derivation"] == T.NAV_FROM_V7_DERIVATION
        assert "oracle" in cfg["nav_cmd_derivation"]
        assert "allow_oracle_nav=True" in cfg["nav_cmd_derivation"]
        st = cfg["nav_from_v7_stats"]["train"]
        assert (st["follow"], st["left"], st["right"], st["missing"]) \
            == (0, 1, 1, 0)
        assert st["label_md5"] == md5
        assert cfg["nav_from_v7_stats"]["eval"] is None   # no eval here
    else:
        assert cfg["nav_cmd_derivation"] == T.NAV_V1_DERIVATION
        assert cfg["nav_cmd_derivation"] == \
            "refb_labels.nav_command (v1, unchanged)"
        assert cfg["nav_from_v7_stats"] is None
    assert json.loads((out / "summary.json").read_text())["done"] is True


def test_preflight_exercises_the_flag_and_the_pin(tmp_path, monkeypatch,
                                                  capsys):
    args = T.build_parser().parse_args(
        ["--arm", "hier", "--out", str(tmp_path), "--smoke", "--preflight",
         "--nav-from-v7", "--v7-labels", "unused-by-preflight.jsonl.gz"])
    assert T.preflight(args) == 0
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setattr(refb, "NAV_COMMANDS",
                        ("left", "follow", "right", "straight"))
    assert T.preflight(args) == 7
    assert "alignment broken" in capsys.readouterr().out
