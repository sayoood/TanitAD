"""The refav1 v7.2 label/nav join — values asserted against hand-built records.

Fixture episodes carry their cache index in every feature element and a
per-episode kappa constant, so every batch row can be mapped back to
(episode, t) and its labels checked against the PLANTED record — never
"shapes look right". Also carries the two MEASUREMENTS the join rule depends
on: the model's range check refuses -100 today (`refa_v1.py:978/:993`), and
``F.cross_entropy``'s default ignore_index averages over valid rows only and
goes NaN when every row is ignored — the exact guard the model-side change
must add before -100 rows can be fed through.
"""
import json
import math

import pytest
import torch
import torch.nn.functional as F

from tanitad.data.refav1_loader import IGNORE_ID, RefAV1Windows

N_TOK, D = 8, 16
T_EP = 201                     # v2ep frames at 10 Hz
T_C = math.ceil(T_EP / 2)      # 101 cache steps at 0.2 s
ACCEL = 0.7

CLIP = {"ep00": "clip-aaaa-0000", "ep01": "clip-bbbb-1111"}

# planted per-episode records. Class indices are the FROZEN v7.0 vocabulary
# positions (vocab_v7.TACTICAL_LAT/LON_ACTIONS_V7, pinned by
# test_vocab_v7_frozen.py): NUDGE_L=4, TURN_R=7; BRAKE_TO=3, ACCELERATE=7.
# nav ids enumerate NAV_COMMAND_TOKENS: NAV_FOLLOW_ROAD=0, NAV_TURN_L=1.
# route ids are refb.ROUTE_CLASSES order: route_left=0, route_straight=1.
REC = {
    "ep00": dict(lat="NUDGE_L", lon="BRAKE_TO", nav="NAV_TURN_L",
                 t0=8.0, band=(2.0, 6.0),
                 lat_id=4, lon_id=3, nav_id=1, route_id=0),
    "ep01": dict(lat="TURN_R", lon="ACCELERATE", nav="NAV_FOLLOW_ROAD",
                 t0=2.0, band=(2.0, 4.0),
                 lat_id=7, lon_id=7, nav_id=0, route_id=1),
}


def _in_band(ep: str, t: int) -> bool:
    r = REC[ep]
    tol = (r["band"][1] - r["band"][0]) / 2.0
    return r["t0"] - tol <= t * 0.2 <= r["t0"] + tol


def _record(clip_id: str, *, lat, lon, nav, t0, band) -> dict:
    return {"schema_version": "s2-geom-v7", "vocab": "v7",
            "clip_id": clip_id, "t0_s": t0,
            "bands": {"operative_s": [0.0, 2.0], "tactical_s": list(band),
                      "strategic_s": [8.0, 30.0]},
            "a_tac": {"lat": lat, "lon": lon},
            "nav_command": {"token": nav, "provenance": "ego-future",
                            "oracle": True,
                            "args": {"distance_m": 10.0, "time_s": 2.0}}}


def _fixture(tmp_path, *, clip_ids=True, label_eps=("ep00", "ep01"),
             mutate=None):
    cache, eps = tmp_path / "cache", tmp_path / "eps"
    cache.mkdir(exist_ok=True), eps.mkdir(exist_ok=True)
    for i, nm in enumerate(("ep00", "ep01")):
        f = torch.arange(T_C, dtype=torch.float16)[:, None, None].expand(
            T_C, N_TOK, D).contiguous()
        torch.save(f, cache / f"{nm}.pt")
        v = 5.0 + ACCEL * 0.1 * torch.arange(T_EP)
        poses = torch.zeros(T_EP, 4)
        poses[:, 3] = v
        actions = torch.zeros(T_EP, 2)
        actions[:, 0] = 0.01 * (i + 1)               # kappa tags the episode
        d = {"poses": poses, "actions": actions, "episode_id": nm}
        if clip_ids:
            d["clip_id"] = CLIP[nm]
        torch.save(d, eps / f"{nm}.v2ep.pt")
    recs = [_record(CLIP[nm], lat=REC[nm]["lat"], lon=REC[nm]["lon"],
                    nav=REC[nm]["nav"], t0=REC[nm]["t0"], band=REC[nm]["band"])
            for nm in label_eps]
    if mutate:
        mutate(recs)
    lp = tmp_path / "labels.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return cache, eps, lp


def _loader(tmp_path, lp=None, nav=None, **kw):
    cache, eps, path = _fixture(tmp_path, **kw.pop("fixture", {}))
    base = dict(op_window=2, op_steps=30, str_dt=3.0, str_ext_steps=2, seed=0)
    base.update(kw)
    return RefAV1Windows(cache, eps,
                         labels_path=(path if lp else None),
                         nav_path=(path if nav else None), **base)


def _rows(b):
    """Map every batch row back to (episode name, t) via the fixture tags."""
    t = b["feats"][:, -1, 0, 0].to(torch.long).tolist()
    ep = ["ep00" if abs(k - 0.01) < 1e-6 else "ep01"
          for k in b["actions"][:, 0, 1].tolist()]
    return list(zip(ep, t))


def test_label_values_match_the_hand_built_records(tmp_path):
    ld = _loader(tmp_path, lp=True, nav=True)
    b = ld.batch(len(ld))                      # one full pass, every window
    for k in ("lat_label", "lon_label", "route_label", "nav_cmd"):
        assert b[k].dtype == torch.long and b[k].shape == (len(ld),)
    n_band = 0
    for i, (ep, t) in enumerate(_rows(b)):
        r = REC[ep]
        if _in_band(ep, t):
            n_band += 1
            assert int(b["lat_label"][i]) == r["lat_id"], (ep, t)
            assert int(b["lon_label"][i]) == r["lon_id"], (ep, t)
            assert int(b["route_label"][i]) == r["route_id"], (ep, t)
        else:
            assert int(b["lat_label"][i]) == IGNORE_ID, (ep, t)
            assert int(b["lon_label"][i]) == IGNORE_ID, (ep, t)
            assert int(b["route_label"][i]) == IGNORE_ID, (ep, t)
        assert int(b["nav_cmd"][i]) == r["nav_id"], (ep, t)   # every window
    # both in-band and out-of-band windows must actually occur, and the
    # in-band count is exact: ep00 t in [30,39] (10), ep01 t in [5,15] (11)
    assert n_band == 21
    assert bool(b["nav_valid"].all())
    assert ld.join_report["labels"]["n_windows_in_band"] == 21


def test_determinism(tmp_path):
    a = _loader(tmp_path, lp=True, nav=True, seed=7).batch(16)
    b = _loader(tmp_path, lp=True, nav=True, seed=7).batch(16)
    for k in ("lat_label", "lon_label", "route_label", "nav_cmd", "nav_valid"):
        assert torch.equal(a[k], b[k]), k


def test_join_miss_is_loud_and_named(tmp_path, capsys):
    ld = _loader(tmp_path, lp=True, nav=True,
                 fixture=dict(label_eps=("ep00",)))
    out = capsys.readouterr().out
    assert "[refav1-labels]" in out and "1 missing" in out and "ep01" in out
    assert ld.join_report["labels"]["missing_episodes"] == ["ep01"]
    assert ld.join_report["nav"]["missing_episodes"] == ["ep01"]
    b = ld.batch(len(ld))
    for i, (ep, t) in enumerate(_rows(b)):
        if ep == "ep01":                      # unlabeled: masked + nav default
            assert int(b["lat_label"][i]) == IGNORE_ID
            assert int(b["route_label"][i]) == IGNORE_ID
            assert int(b["nav_cmd"][i]) == 0
            assert not bool(b["nav_valid"][i])
        else:
            assert bool(b["nav_valid"][i])


def test_zero_join_is_refused(tmp_path):
    def rename_all(recs):
        for j, r in enumerate(recs):
            r["clip_id"] = f"not-in-corpus-{j}"
    with pytest.raises(ValueError, match="ZERO"):
        _loader(tmp_path, lp=True, fixture=dict(mutate=rename_all))


def test_unknown_token_is_refused_by_name(tmp_path):
    def warp(recs):
        recs[0]["a_tac"]["lat"] = "WARP_DRIVE"
    with pytest.raises(ValueError, match="WARP_DRIVE"):
        _loader(tmp_path, lp=True, fixture=dict(mutate=warp))


def test_missing_clip_id_key(tmp_path):
    with pytest.raises(ValueError, match="clip_id"):
        _loader(tmp_path, lp=True, fixture=dict(clip_ids=False))
    # ...but WITHOUT a join request the same corpus still constructs (the
    # pre-join contract) and keeps the None fields.
    ld = _loader(tmp_path, fixture=dict(clip_ids=False))
    b = ld.batch(2)
    assert b["lat_label"] is None and b["nav_cmd"] is None
    assert "nav_valid" not in b


def test_nav_ids_fit_the_model_embedding_and_carry_the_stamp(tmp_path):
    from tanitad.config import StrategicPolicyConfig
    from tanitad.refs.refb import NAV_COMMANDS
    ld = _loader(tmp_path, lp=True, nav=True)
    b = ld.batch(len(ld))
    n_cmd = StrategicPolicyConfig().n_commands
    assert int(b["nav_cmd"].max()) < n_cmd and int(b["nav_cmd"].min()) >= 0
    # the position pin, spot-checked end to end: the planted NAV_TURN_L became
    # id 1, which is exactly NAV_COMMANDS.index("left") — the row the model's
    # nav_emb was sized/ordered by.
    assert REC["ep00"]["nav_id"] == NAV_COMMANDS.index("left") == 1
    assert ld.join_report["nav"]["allow_oracle_nav"] is True
    assert ld.join_report["nav"]["nav_arg_semantics"] == "t0_constant"


# --------------------------------------------------------------------------- #
# the two MEASUREMENTS the -100 contract rests on (brief: measure, not assume)
# --------------------------------------------------------------------------- #
def _tiny_model():
    from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
    from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
    cfg = RefAV1Config(
        tac_vocab_version="v6.0", d_enc=16, d_state=16, n_tokens=8,
        op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
        tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
        str_dt=3.0, str_steps=2, str_dim=8, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1, n_heads=2,
                                            d_ctx=8, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1, n_heads=2,
                                          d_intent=8))
    m = RefAV1(cfg)
    f = torch.randn(3, cfg.op_window, cfg.n_tokens, cfg.d_enc)
    a = torch.randn(3, cfg.op_steps, cfg.a_dim)
    fut = torch.randn(3, cfg.op_steps, cfg.n_tokens, cfg.d_enc)
    return m, f, a, fut


def test_the_model_MASKS_minus100_the_tripwire_flipped_2026_09_01():
    """⭐ This test's first life pinned the DEFECT: the range check refused
    -100, so a mixed labeled/unlabeled batch could not be fed at all. The
    proposed repair landed same-day (validate only ``lbl[lbl != -100]``, skip
    an all-ignored family); this is now the pin of the REPAIRED contract."""
    m, f, a, fut = _tiny_model()
    # mixed batch: accepted, CE over the valid rows only, term in the loss
    out = m(f, a, future_feats=fut,
            lat_label=torch.tensor([0, 1, IGNORE_ID]))
    assert "loss_lat_label" in out
    assert torch.isfinite(out["loss_lat_label"].detach())
    # all-ignored family: SKIPPED, never NaN, and absent from the output
    out2 = m(f, a, future_feats=fut,
             lat_label=torch.full((3,), IGNORE_ID),
             route_label=torch.full((3,), IGNORE_ID))
    assert "loss_lat_label" not in out2 and "loss_route_label" not in out2
    assert torch.isfinite(out2["loss"].detach())
    # genuinely out-of-range VALID rows are still refused by name
    with pytest.raises(ValueError, match="lat_label outside"):
        m(f, a, future_feats=fut, lat_label=torch.tensor([0, 99, IGNORE_ID]))
    with pytest.raises(ValueError, match="route_label outside"):
        m(f, a, future_feats=fut, route_label=torch.tensor([0, IGNORE_ID, 7]))


def test_MEASURED_cross_entropy_default_ignore_index_semantics():
    """What makes -100 SAFE once the range check learns to skip it: the
    default ignore_index averages over valid rows only — and what makes the
    skip NECESSARY: an all-ignored batch is NaN, not zero."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 5, generator=g)
    tgt = torch.tensor([1, IGNORE_ID, 3, IGNORE_ID])
    mixed = F.cross_entropy(logits, tgt)
    only_valid = F.cross_entropy(logits[[0, 2]], tgt[[0, 2]])
    assert torch.allclose(mixed, only_valid)
    assert torch.isnan(
        F.cross_entropy(logits, torch.full((4,), IGNORE_ID)))
