"""D-V7-WIRING — the v7 trainer can read the v7.2 labels, the B1 corpus has a
parity key, and the k-step rollout has a truncation knob (register row
D-V7-READINESS-2026-09-02 §B: "three prerequisites not on the gate").

Every change is flag-gated and DEFAULT = today's behaviour; every guard is
shown able to fail. Nothing here touches a pod: the "real blob" test reads the
local release copy on the dev box and SKIPS elsewhere.

⛔ ONE DOOR, TWO SCHEMAS, SNIFFED FROM THE RECORD (SPEC_V7_LABEL_TRAINER_WIRING
§2): `--s2-labels` dispatches on the first record's `schema_version`, never the
filename. The v1 route is the incumbent `load_s2_labels` call, unchanged. The
v7.2 route goes through `tanitad.data.v7_labels.load_v7_labels` (the STAMP),
lands the strategic ids on the v7 heads through the INCUMBENT join machinery
(`S2WindowSupervision` — reuse the machinery, replace only the rows) and adds
the FACTORED tactical keys on their OWN band (spec §3.1: three bands, one per
layer, differing on every record).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.data import parity  # noqa: E402
from tanitad.data.v7_labels import (  # noqa: E402
    HEADS, tactical_class_ids, window_in_band)
from tanitad.models.v6 import (  # noqa: E402
    GOAL_ARG_SLOTS, STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS, V6Config,
    V6Stack)
from tanitad.models.vocab_v7 import (  # noqa: E402
    NOT_YET_EXTRACTABLE, STRATEGIC_GOAL_TOKENS_V7)
from tanitad.train.intrain_eval import V72, V72_INDEX  # noqa: E402
import s2_labels as SL  # noqa: E402
from s2_labels import IGNORE_ID, NO_LABEL, stable_episode_id  # noqa: E402
from train_v6_staged import (  # noqa: E402
    S2_SCHEMA_V1, S2_SCHEMA_V72, V6LossWeights, V72_ARGS_SUPERVISED,
    V72_TACTICAL_BATCH_KEYS, _S2_ROUTE_TO_ID, _sniff_label_schema,
    _v72_classes, build_parser, dry_run, load_s2_labels_any,
    load_v72_labels_for_trainer, preflight, s2_label_route,
    synthetic_train_batch, v6_loss_step)

# --------------------------------------------------------------------------- #
# facts pinned by this file (MEASURED 2026-09-03, dev box, local release copy)
# --------------------------------------------------------------------------- #
B1_KEY = "physicalai-b1-w120-256x640cyl"
B1_DIGEST_4713 = "e8bfb98e06ebd62edf6566906f0f961b7e6efaf1869ee4aa6646511a9f1509da"
B1_DIGEST_4719 = "a48251e89c7a86032415a55a375d5e5f440360586b21011d7326c7790aa6d342"
B1_MANIFEST_SHA16 = "5feda062a72a32ad"
#: the four keys that existed before B1 — (episode_count, uid_kind,
#: episode_uid_sha256, skip_count). A change here is a change to the SACRED
#: corpus identity and must be a deliberate, separately reviewed edit.
LEGACY_PINS = {
    "physicalai-train-e438721ae894": (
        2376, "epcache_basename",
        "9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386", 24),
    "physicalai-val-0c5f7dac3b11": (600, "epcache_basename", None, 0),
    "physicalai-train-e438721ae894-w120-256x640cyl": (
        2400, "v2ep_clipid",
        "e61a04553df5b9d52a0810be32cf31927bd92644d9d12ada563910b8a0ada4de", 0),
    "physicalai-val-0c5f7dac3b11-w120-256x640cyl": (
        600, "v2ep_clipid",
        "0b176d2e5cb49667d5009366817f948759724e69642e626a47362b93e31da68e", 0),
}
_V72_ROOT = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v72")
needs_release = pytest.mark.skipif(
    not (_V72_ROOT / "s2_labels_v7.2_train.jsonl.gz").exists(),
    reason="local v7.2 release copy not on this box")

#: UUID-shaped clip ids (only their hash matters; nothing gated here)
CIDS = ("0089a096-68be-40df-8097-780bf1ae1c19",
        "00d05901-ed0a-4a43-adca-bdab70d30bfa",
        "0166b2f1-3c66-4d8f-8b8e-2f0c3f0b6b0c")
UNLABELLED = "ffffffff-0000-4000-8000-000000000000"
W, DT, N_STACK = 4, 0.1, 3                    # window, tick, 9 ch -> 3 frames
T0 = 8.0


def _t_now(t: int) -> float:
    """The window's NOW on the RAW clip timeline — the S2WindowSupervision
    expression: (provider index + W - 1 + (n_stack - 1)) * dt."""
    return (t + W - 1 + (N_STACK - 1)) * DT


# --------------------------------------------------------------------------- #
# fixtures — a v7.2 blob (the test_v7_labels record shape + REAL named args)
# --------------------------------------------------------------------------- #
def _rec_v72(cid: str, **over) -> dict:
    r = {"clip_id": cid, "schema_version": "s2-geom-v7", "vocab": "v7",
         "a_tac": {"lat": "LANE_KEEP", "lon": "CRUISE", "truncated": False,
                   "lat_args": {"within_m": 3.0},
                   "lon_args": {"v_target_ms": 9.0},
                   "serves_goals": {"lat_serves": [], "lon_serves": []}},
         "a_str": {"token": "HOLD_MAIN_ROAD", "args": {}},
         "g_str": {"token": "FOLLOW_ROUTE", "args": {}},
         "g_tac": {"anchor": {"goal_x_m": 50.0, "goal_y_m": 1.0,
                              "t_reach_s": 6.0},
                   "goals": {}, "violations": []},
         "bands": {"operative_s": [0, 2], "tactical_s": [2, 6],
                   "strategic_s": [8, 30], "unassigned_manoeuvres": []},
         "t0_s": T0, "horizon": {"available_s": 30.0, "recording_span_s": 30.0},
         "nav_command": {"token": "NAV_FOLLOW_ROAD", "provenance": "ego-future",
                         "oracle": True,
                         "args": {"distance_m": 0.0, "time_s": 0.0}},
         "turn_suppression": None,
         "alpamayo": {"lateral": {"agree": True},
                      "longitudinal": {"agree": True}}}
    r.update(over)
    return r


def _rows() -> list[dict]:
    rows = [_rec_v72(c) for c in CIDS]
    rows[1]["a_tac"]["lon"] = "ACCELERATE"
    rows[2]["a_tac"]["lat"] = "TURN_L"
    rows[2]["a_str"] = {"token": "PREPARE_TURN_L_FOLLOW_ROUTE",
                        "args": {"within_m": 106.5, "by_time_s": 11.1}}
    rows[2]["g_str"] = {"token": "TURN_LEFT_FOLLOW_ROUTE",
                        "args": {"within_m": 106.5, "by_time_s": 11.1}}
    return rows


def _v72_blob(tmp_path, rows, name="s2_labels_v7.2_train.jsonl.gz") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


# --------------------------------------------------------------------------- #
# fixtures — a tiny v1 artifact (the test_v6_s2_loss shape)
# --------------------------------------------------------------------------- #
def _legacy(cid: str) -> int:
    return int.from_bytes(cid.encode()[:4], "big")


def _index_v1(clips, t0=8.0, band=(-2.0, 2.0)):
    out = {"_t0_s": t0, "_valid_window_s": list(band), "clips": {}}
    for cid, ent in clips.items():
        out["clips"][cid] = {
            "label_split": "aug120", "corpus": "test",
            "v2ep_file": f"{cid}.v2ep.pt", "episode_id_legacy": _legacy(cid),
            "episode_id_stable": stable_episode_id(cid), "excluded": False,
            **ent}
    return out


def _block(tok, tokens, slots=()):
    a = [0.0] * GOAL_ARG_SLOTS
    m = [0] * GOAL_ARG_SLOTS
    for s in slots:
        m[s] = 1
        a[s] = 27.3
    return {"token": tok, "token_id": tokens.index(tok), "args": a,
            "arg_mask": m, "provenance": "path",
            "sources": ["engine_a.route_v3"]}


def _rec_v1(cid, g_tok="TURN_LEFT", a_tok="HOLD_CORRIDOR", **over):
    rec = {"schema_version": "s2-strategic-v1", "clip_id": cid, "t0_s": 8.0,
           "g_str": _block(g_tok, STRATEGIC_GOAL_TOKENS, slots=(0,)),
           "a_str": _block(a_tok, STRATEGIC_ACTION_TOKENS, slots=(6,)),
           "valid_window_s": [-2.0, 2.0],
           "disjointness": {"situation_classifier_output_used": False}}
    rec.update(over)
    return rec


def _write_v1(tmp_path, records, index, name="s2_labels_test.jsonl") -> Path:
    d = tmp_path / "labels_v1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "clip_index.json").write_text(json.dumps(index))
    (d / name).write_text("\n".join(json.dumps(r) for r in records))
    return d


# --------------------------------------------------------------------------- #
# fixtures — episodes, index, a tiny v7-vocab stack, argv
# --------------------------------------------------------------------------- #
def _episodes(cids):
    return [types.SimpleNamespace(episode_id=stable_episode_id(c),
                                  frames=torch.zeros(1, 3 * N_STACK, 2, 2))
            for c in cids]


def _index(n_eps: int, step: int = 5):
    return [(e_i, t) for e_i in range(n_eps) for t in range(0, 200, step)]


def _tiny_cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=W,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    base.update(kw)                        # tac_vocab_version defaults to v7.0
    return V6Config(**base)


def _stack(**kw) -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(_tiny_cfg(**kw))


def _args(*argv):
    import argparse
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    return ap.parse_args(list(argv))


def _tiny_argv(tmp_path, stage="S-S", *extra):
    return ["--stage", stage, "--out", str(tmp_path / "out"), "--dry-run",
            "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
            "--patch", "16", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "8",
            "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "2", "--d-tac", "32",
            "--d-str", "16", "--d-goal-embed", "16", "--adapter-hidden", "32",
            "--n-candidates", "3", "--sigreg-slices", "8",
            "--dry-steps", "1", "--dry-batch", "2", "--dry-k", "12",
            *extra]


# =========================================================================== #
# 1. the door: sniff the RECORD, dispatch, refuse the unknown by name
# =========================================================================== #
def test_the_sniff_routes_on_the_record_not_the_name(tmp_path):
    v1 = _write_v1(tmp_path, [_rec_v1(CIDS[0])], _index_v1({CIDS[0]: {}}))
    assert s2_label_route(v1) == "v1"
    # a v7.2 blob wearing a v1-looking NAME still routes v72
    blob = _v72_blob(tmp_path, _rows(), name="s2_labels_aug120.jsonl.gz")
    assert s2_label_route(blob) == "v72"
    s = _sniff_label_schema(blob)
    assert s["schema_version"] == S2_SCHEMA_V72 and s["vocab"] == "v7"
    assert {"a_tac", "a_str", "g_str", "bands", "t0_s"} <= set(s["keys"])
    assert _sniff_label_schema(v1)["schema_version"] == S2_SCHEMA_V1


def test_an_unknown_schema_is_REFUSED_by_name(tmp_path):
    blob = _v72_blob(tmp_path, [_rec_v72(CIDS[0], schema_version="s2-geom-v9")])
    with pytest.raises(SystemExit) as ei:
        s2_label_route(blob)
    msg = str(ei.value)
    assert "s2-geom-v9" in msg and S2_SCHEMA_V1 in msg and S2_SCHEMA_V72 in msg


def test_a_noncanonical_md5_is_REFUSED_unless_allowed_and_printed(tmp_path,
                                                                  capsys):
    blob = _v72_blob(tmp_path, _rows())
    md5 = hashlib.md5(blob.read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as ei:
        load_v72_labels_for_trainer(blob)
    msg = str(ei.value)
    assert md5 in msg and V72["train"]["md5"] in msg and V72["eval"]["md5"] in msg
    assert "--allow-any-labels" in msg
    ls = load_v72_labels_for_trainer(blob, allow_any_labels=True)
    out = capsys.readouterr().out
    assert f"md5={md5}" in out and "NON-CANONICAL" in out
    assert ls.v72["canonical"] is False and ls.v72["md5"] == md5
    assert ls.v72["allow_any_labels"] is True
    assert ls.manifest.allow_oracle_nav is True          # the STAMP


# =========================================================================== #
# 2. the rows: v7 ids on the v7 heads, no arg supervision, the record's band
# =========================================================================== #
def test_v72_rows_carry_v7_ids_zero_arg_supervision_and_the_record_band(
        tmp_path):
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, _rows()),
                                     allow_any_labels=True)
    assert len(ls) == 3 and ls.legacy_ids == {}
    r2 = ls.rows_by_stable[stable_episode_id(CIDS[2])]
    assert r2.g_id == HEADS["str_goal"].index("TURN_LEFT_FOLLOW_ROUTE")
    assert r2.a_id == HEADS["str_action"].index("PREPARE_TURN_L_FOLLOW_ROUTE")
    assert r2.g_token == "TURN_LEFT_FOLLOW_ROUTE" and r2.g_sup and r2.a_sup
    # ⛔ args: NOT supervised — all-zero mask, exactly zero gradient
    assert V72_ARGS_SUPERVISED is False
    for r in ls.rows_by_stable.values():
        assert float(r.g_mask.sum()) == 0.0 and float(r.a_mask.sum()) == 0.0
        assert float(r.g_args.abs().sum()) == 0.0
        # the strategic band is the RECORD's: ±(30-8)/2 = ±11 s around t0
        assert r.t0_s == T0 and r.band == (-11.0, 11.0)
    assert ls.t0_s == T0 and ls.band == (-11.0, 11.0)
    rep = ls.report()
    assert rep["schema_version"] == S2_SCHEMA_V72        # never s2-strategic-v1
    assert rep["v72"]["args_supervised"] is False
    assert rep["v72"]["manifest"]["md5"] == ls.manifest.md5
    assert rep["token_census_records"]["g_str"] == {
        "FOLLOW_ROUTE": 2, "TURN_LEFT_FOLLOW_ROUTE": 1}
    assert "tac_lat" in rep["v72"]["masks"] and "str_goal" in rep["v72"]["masks"]
    assert rep["v72"]["mask_presence"]["ok"] is False    # a 3-record fixture


def test_a_masked_strategic_token_makes_the_family_UNSUPERVISED(tmp_path):
    tok = "EXIT_LEFT_FOLLOW_ROUTE"
    assert tok in NOT_YET_EXTRACTABLE, "fixture premise: the token is masked"
    rows = _rows()
    rows[0]["g_str"] = {"token": tok, "args": {}}
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, rows),
                                     allow_any_labels=True)
    r0 = ls.rows_by_stable[stable_episode_id(CIDS[0])]
    assert r0.g_sup is False and r0.g_id == IGNORE_ID and r0.g_token == NO_LABEL
    assert r0.a_sup is True                                # the OTHER family lives
    assert ls.v72["masked_records"]["str_goal"] == {tok: 1}
    assert ls.report()["token_census_records"]["g_str"][NO_LABEL] == 1


def test_a_token_outside_the_frozen_vocabulary_is_REFUSED(tmp_path):
    rows = _rows()
    rows[1]["a_str"] = {"token": "PREPARE_WARP_DRIVE", "args": {}}
    with pytest.raises(SystemExit) as ei:
        load_v72_labels_for_trainer(_v72_blob(tmp_path, rows),
                                    allow_any_labels=True)
    assert "FROZEN v7 vocabulary" in str(ei.value)


def test_a_drifted_clip_index_beside_the_blob_is_REFUSED(tmp_path):
    """The s2_labels drift guard, kept: an index whose recorded stable id
    disagrees with the function is a silent zero-match and is refused."""
    blob = _v72_blob(tmp_path, _rows())
    idx = {"clips": {c: {"episode_id_stable": stable_episode_id(c) + 1}
                     for c in CIDS}}
    (tmp_path / "clip_index.json").write_text(json.dumps(idx))
    with pytest.raises(SystemExit) as ei:
        load_v72_labels_for_trainer(blob, allow_any_labels=True)
    assert "episode_id_stable" in str(ei.value) and "drift" in str(ei.value)
    # and a CONSISTENT one is cross-checked and recorded
    idx = {"clips": {c: {"episode_id_stable": stable_episode_id(c)}
                     for c in CIDS}, "_clips_without_episode": []}
    (tmp_path / "clip_index.json").write_text(json.dumps(idx))
    ls = load_v72_labels_for_trainer(blob, allow_any_labels=True)
    ci = ls.v72["clip_index"]
    assert ci["checked"] and ci["n_index_clips"] == 3 and ci["stable_id_drift"] == 0
    assert "BY NAME" in ls.v72["resolved"]["index_mode"]


# =========================================================================== #
# 3. the join: the incumbent seven keys + the factored tactical keys,
#    each family on ITS OWN band
# =========================================================================== #
def test_the_join_emits_the_incumbent_keys_PLUS_the_factored_tactical_keys(
        tmp_path):
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, _rows()),
                                     allow_any_labels=True)
    eps = _episodes(CIDS + (UNLABELLED,))
    index = _index(len(eps))
    sup = ls.supervision(eps, window=W, dt=DT, index=index)
    assert isinstance(sup, _v72_classes()["V72WindowSupervision"])
    assert isinstance(sup, SL.S2WindowSupervision)     # the MACHINERY is reused
    assert sup.n_episodes == 4 and sup.n_matched_episodes == 3
    out = sup.batch(list(range(len(index))))
    seven = {"g_str_id", "g_str_args", "g_str_arg_mask", "a_str_id",
             "a_str_args", "a_str_arg_mask", "s2_valid"}
    assert seven <= set(out) and set(V72_TACTICAL_BATCH_KEYS) <= set(out)
    assert out["tac_lat_id"].dtype == torch.long and out["tac_valid"].dtype == torch.bool
    n_tac = n_str = 0
    for j, (e_i, t) in enumerate(index):
        lab = ls.v7_by_stable.get(int(eps[e_i].episode_id))
        t_now = _t_now(t)
        if lab is None:
            assert not out["s2_valid"][j] and not out["tac_valid"][j]
            continue
        # strategic: the record's ±11 s band, evaluated as S2WindowSupervision does
        str_ok = (T0 - 11.0) <= t_now <= (T0 + 11.0)
        assert bool(out["s2_valid"][j]) == str_ok, (e_i, t)
        if str_ok:
            n_str += 1
            assert int(out["g_str_id"][j]) == HEADS["str_goal"].index(lab.str_goal)
            assert int(out["a_str_id"][j]) == HEADS["str_action"].index(lab.str_action)
        # tactical: through the ONE window rule, independently
        tac_ok = window_in_band(lab, t_now)
        assert bool(out["tac_valid"][j]) == tac_ok, (e_i, t)
        if tac_ok:
            n_tac += 1
            lat, lon = tactical_class_ids(lab, t_now)
            assert int(out["tac_lat_id"][j]) == lat == HEADS["tac_lat"].index(lab.tac_lat)
            assert int(out["tac_lon_id"][j]) == lon == HEADS["tac_lon"].index(lab.tac_lon)
        else:
            assert int(out["tac_lat_id"][j]) == IGNORE_ID == int(out["tac_lon_id"][j])
    assert n_tac > 0 and n_str > n_tac                  # both families fire, differently
    rep = sup.report()
    assert rep["n_windows_in_band_tac"] == n_tac == int(out["tac_valid"].sum())
    assert rep["n_windows_in_band"] == n_str == int(out["s2_valid"].sum())
    assert rep["n_windows_supervised"]["tac_lat"] == n_tac
    assert rep["window_token_census"]["tac_lat"]["TURN_L"] > 0    # per family
    assert rep["window_token_census"]["tac_lon"]["ACCELERATE"] > 0
    assert list(rep["tactical_keys"]) == list(V72_TACTICAL_BATCH_KEYS)


def test_per_family_bands_tactical_narrow_strategic_wide(tmp_path):
    """Spec §3.1: ONE band for all families would supervise the tactical heads
    over the strategic horizon. A window 5 s past t0 is strategically valid
    (±11 s) and tactically IGNORED (±2 s)."""
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, _rows()),
                                     allow_any_labels=True)
    eps = _episodes(CIDS)
    t = 125                                              # t_now = 13.0 s
    assert _t_now(t) == pytest.approx(T0 + 5.0)
    sup = ls.supervision(eps, window=W, dt=DT, index=[(0, t)])
    out = sup.batch([0])
    assert bool(out["s2_valid"][0]) and int(out["g_str_id"][0]) >= 0
    assert not bool(out["tac_valid"][0])
    assert int(out["tac_lat_id"][0]) == IGNORE_ID == int(out["tac_lon_id"][0])


def test_a_masked_tactical_token_ignores_that_axis_only(tmp_path):
    tok = "ABORT_LC"
    assert tok in NOT_YET_EXTRACTABLE and tok in HEADS["tac_lat"]
    rows = _rows()
    rows[0]["a_tac"]["lat"] = tok
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, rows),
                                     allow_any_labels=True)
    sup = ls.supervision(_episodes(CIDS), window=W, dt=DT, index=[(0, 75)])
    assert window_in_band(ls.v7_by_stable[stable_episode_id(CIDS[0])], _t_now(75))
    out = sup.batch([0])
    assert bool(out["tac_valid"][0])
    assert int(out["tac_lat_id"][0]) == IGNORE_ID           # masked axis
    assert int(out["tac_lon_id"][0]) == HEADS["tac_lon"].index("CRUISE")
    assert sup.report()["window_token_census"]["tac_lat"] == {NO_LABEL: 1}


# =========================================================================== #
# 4. the v1 route is the incumbent call, untouched
# =========================================================================== #
def test_the_v1_route_calls_the_incumbent_loader_unchanged(tmp_path,
                                                            monkeypatch):
    v1 = _write_v1(tmp_path, [_rec_v1(CIDS[0])], _index_v1({CIDS[0]: {}}))
    calls = []
    sentinel = object()

    def _rec(path, *a, **kw):
        calls.append((path, a, kw))
        return sentinel

    monkeypatch.setattr(SL, "load_s2_labels", _rec)
    import tanitad.data.v7_labels as V7

    def _boom(*a, **kw):
        raise AssertionError("the v1 route must never touch the v7 loader")

    monkeypatch.setattr(V7, "load_v7_labels", _boom)
    assert load_s2_labels_any(v1, allow_any_labels=True, stack=None) is sentinel
    assert calls == [(v1, (), {})]                       # same path, no kwargs
    # and the real loader still loads the same artifact (the sniff read
    # nothing the loader minds)
    monkeypatch.undo()
    assert len(load_s2_labels_any(v1)) == 1


# =========================================================================== #
# 5. the smoke: a v7.2-derived batch through v6_loss_step on a v7 stack
# =========================================================================== #
def test_a_v72_batch_flows_through_v6_loss_step_on_a_v7_stack(tmp_path):
    s = _stack()
    assert tuple(s.vocab_str.tokens) == tuple(STRATEGIC_GOAL_TOKENS_V7)
    ls = load_v72_labels_for_trainer(_v72_blob(tmp_path, _rows()),
                                     allow_any_labels=True, stack=s)
    assert ls.v72["vocab_check"].startswith("stack.vocab")
    eps = _episodes(CIDS)
    index = _index(len(eps))
    sup = ls.supervision(eps, window=W, dt=DT, index=index)
    idx = [i for i, (e_i, t) in enumerate(index)
           if window_in_band(ls.v7_by_stable[int(eps[e_i].episode_id)],
                             _t_now(t))][:4]
    assert len(idx) == 4
    keys = sup.batch(idx)
    assert bool(keys["tac_valid"].all()) and bool(keys["s2_valid"].all())
    b = synthetic_train_batch(s, batch=4, k=12, seed=1)
    b["gt_wp"] = torch.randn(4, 10, 2, generator=torch.Generator().manual_seed(1))
    b |= keys
    L = v6_loss_step(s, b, stage="S-S", weights=V6LossWeights(w_s2_goal=1.0),
                     o1_k=10, o5_k=12)
    assert "s2" in L and bool(torch.isfinite(L["loss"]))
    assert L["log"]["s2_n_valid"] == 4
    assert L["log"]["s2_g_n_valid"] == 4 and L["log"]["s2_a_n_valid"] == 4
    assert L["log"]["s2_g_arg_l1"] is None and L["log"]["s2_a_arg_slots"] == 0
    # per token, per family — and named through the HEAD's (v7) vocabulary
    assert L["log"]["s2_g_tok_counts"] and \
        set(L["log"]["s2_g_tok_counts"]) <= set(STRATEGIC_GOAL_TOKENS_V7)
    # the tactical keys ride along UNREAD — no term consumes them yet
    assert not any(k.startswith("tac_") for k in L["log"])


def test_a_v6_vocab_stack_is_REFUSED_for_v72_labels(tmp_path):
    s6 = _stack(tac_vocab_version="v6.0")
    with pytest.raises(SystemExit) as ei:
        load_v72_labels_for_trainer(_v72_blob(tmp_path, _rows()),
                                    allow_any_labels=True, stack=s6)
    assert "NON-v7 head" in str(ei.value) and "vocab_str" in str(ei.value)


def test_the_v6_ROUTE_TO_index_is_a_MASKED_class_in_v7():
    """`synthetic_s2_batch` still skips `_S2_ROUTE_TO_ID` (a v6 index) on a v7
    head. On the v7 goal vocabulary that index names a DIFFERENT token; the
    skip is harmless ONLY because that token is NOT_YET_EXTRACTABLE and the
    adapter never emits a masked class as valid. A vocab change that unmasks
    it must revisit the synthetic generator — this pins why."""
    assert _S2_ROUTE_TO_ID < len(STRATEGIC_GOAL_TOKENS_V7)
    assert STRATEGIC_GOAL_TOKENS_V7[_S2_ROUTE_TO_ID] in NOT_YET_EXTRACTABLE


def test_the_ROUTE_TO_gate_and_the_log_names_follow_the_heads_vocabulary():
    """MEASURED on the first v7.2 dry-run: the loss named v7 ids through the
    v6 tuples (id 3 -> 'EXIT_LEFT' for 'STOP_AT_FOLLOW_ROUTE') and compared a
    v7 id against the v6 ROUTE_TO index. With the head's own tokens: v7 id 7
    is a class, not ROUTE_TO; v6 id 7 is still refused; a width/name mismatch
    is refused rather than mislabelled; the defaults are the v6 tuples."""
    from train_v6_staged import s2_goal_loss
    from tanitad.models.vocab_v7 import STRATEGIC_ACTION_TOKENS_V7
    g = torch.Generator().manual_seed(0)

    def _batch(gid: int):
        return {"g_str_id": torch.tensor([gid, 0]),
                "g_str_args": torch.zeros(2, GOAL_ARG_SLOTS),
                "g_str_arg_mask": torch.zeros(2, GOAL_ARG_SLOTS),
                "a_str_id": torch.tensor([0, 1]),
                "a_str_args": torch.zeros(2, GOAL_ARG_SLOTS),
                "a_str_arg_mask": torch.zeros(2, GOAL_ARG_SLOTS),
                "s2_valid": torch.tensor([True, True])}

    def _head(width: int):
        return {"logits": torch.randn(2, width, generator=g),
                "args": torch.randn(2, GOAL_ARG_SLOTS, generator=g)}

    v7 = dict(g_tokens=STRATEGIC_GOAL_TOKENS_V7, a_tokens=STRATEGIC_ACTION_TOKENS_V7)
    # v7 head, id 7 valid: a class (LANE_CHANGE_R_FOLLOW_ROUTE), NOT a refusal
    loss, log = s2_goal_loss(_head(8), _head(7), _batch(7), **v7)
    assert torch.isfinite(loss)
    assert log["s2_g_tok_counts"] == {"FOLLOW_ROUTE": 1,
                                      "LANE_CHANGE_R_FOLLOW_ROUTE": 1}
    assert set(log["s2_a_tok_counts"]) <= set(STRATEGIC_ACTION_TOKENS_V7)
    # v6 head (the default names), id 7 valid: ROUTE_TO is still GATED
    with pytest.raises(ValueError, match="ROUTE_TO"):
        s2_goal_loss(_head(len(STRATEGIC_GOAL_TOKENS)),
                     _head(len(STRATEGIC_ACTION_TOKENS)), _batch(7))
    assert STRATEGIC_GOAL_TOKENS.index("ROUTE_TO") == 7 == _S2_ROUTE_TO_ID
    # v7 head with the v6 default names: refused, never mislabelled
    with pytest.raises(ValueError, match="token names"):
        s2_goal_loss(_head(8), _head(7), _batch(0))


# =========================================================================== #
# 6. the dry-run door (spec §7.5: md5 + counts before any GPU is spent)
# =========================================================================== #
def test_dry_run_exercises_the_v72_door_and_records_the_stamp(tmp_path):
    blob = _v72_blob(tmp_path, _rows())
    a = _args(*_tiny_argv(tmp_path, "S-S", "--w-s2-goal", "1",
                          "--s2-labels", str(blob), "--allow-any-labels",
                          "--bptt-truncate", "3"))
    r = dry_run(a)
    s2 = r["s2_labels"]
    assert s2["exercised"] is True and s2["n_records"] == 3
    assert s2["schema_version"] == S2_SCHEMA_V72
    assert s2["v72"]["md5"] == hashlib.md5(blob.read_bytes()).hexdigest()
    assert s2["v72"]["canonical"] is False and s2["v72"]["args_supervised"] is False
    assert s2["v72"]["manifest"]["allow_oracle_nav"] is True
    assert "s2" in r["steps"][0]["terms"]
    # the log names v7 ids with v7 token names (the dry-run's synthetic ids
    # are drawn from the BUILT head's width)
    assert set(r["steps"][0]["s2_g_tok_counts"]) <= set(STRATEGIC_GOAL_TOKENS_V7)
    assert "bptt_truncate" not in r["steps"][0]          # S-S rolls nothing
    out = json.loads((tmp_path / "out" / "dry_run.json").read_text())
    assert out["s2_labels"]["v72"]["md5"] == s2["v72"]["md5"]
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    assert cfg["args"]["bptt_truncate"] == 3 and cfg["args"]["allow_any_labels"]
    # S-W rolls (O5) — the cut is in force there and is logged
    r2 = dry_run(_args(*_tiny_argv(tmp_path / "w", "S-W",
                                   "--bptt-truncate", "3")))
    assert r2["steps"][0]["bptt_truncate"] == 3
    r3 = dry_run(_args(*_tiny_argv(tmp_path / "w0", "S-W")))
    assert "bptt_truncate" not in r3["steps"][0]         # default: unchanged log


@needs_release
def test_the_real_v72_blobs_load_through_the_door():
    """The two canonical blobs (md5-pinned in intrain_eval.V72) load, resolve
    their indexes BY CONTENT beside them, and report the counts the register
    quotes: 4,572 train / 147 eval, 6 eval clips without pixels here."""
    tr = load_v72_labels_for_trainer(_V72_ROOT / "s2_labels_v7.2_train.jsonl.gz")
    assert len(tr) == 4572 and tr.v72["side"] == "train" and tr.v72["canonical"]
    assert tr.v72["md5"] == V72["train"]["md5"]
    ci = tr.v72["clip_index"]
    assert ci["checked"] and ci["n_index_clips"] == 4572
    assert ci["n_clips_without_episode"] == 0 and ci["stable_id_drift"] == 0
    assert "by content" in tr.v72["resolved"]["index_mode"]
    assert hashlib.md5(Path(ci["path"]).read_bytes()).hexdigest() \
        == V72_INDEX["train_index"]["md5"]
    assert tr.v72["masked_records"] == {"str_goal": {}, "str_action": {}}
    assert tr.report()["schema_version"] == S2_SCHEMA_V72
    ev = load_v72_labels_for_trainer(_V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz")
    assert len(ev) == 147 and ev.v72["side"] == "eval"
    assert ev.v72["clip_index"]["n_clips_without_episode"] == 6   # the EVAL6


# =========================================================================== #
# 7. the B1 parity key
# =========================================================================== #
def test_the_B1_key_is_registered_and_the_legacy_keys_are_unchanged():
    man = parity.load_manifest()
    ent = man["corpora"][B1_KEY]
    assert ent["episode_count"] == 4713 and ent["uid_kind"] == parity.V2_UID_KIND
    assert ent["episode_uid_sha256"] == B1_DIGEST_4713
    cm = ent["clip_membership"]
    assert cm["n_clips"] == 4713 and cm["clip_id_sha256_sorted"] == B1_DIGEST_4713
    assert cm["n_clips_released"] == 4719
    assert cm["clip_id_sha256_sorted_released_4719"] == B1_DIGEST_4719
    assert B1_DIGEST_4719.startswith("a48251e89c7a8603")   # == the release's id
    prov = ent["provenance"]
    assert prov["release"]["manifest_sha256_16"] == B1_MANIFEST_SHA16
    assert prov["release"]["manifest_sha256"].startswith(B1_MANIFEST_SHA16)
    assert prov["release"]["n_clips"] == 4719
    ex = prov["exclusion_rule"]
    assert ex["excluded"] == 6 and len(ex["excluded_clip_id_sha256"]) == 6
    assert "--exclude-parity-overlap" in ex["required_build_flags"]
    assert prov["labels"]["train"]["md5"] == V72["train"]["md5"]
    assert prov["labels"]["eval"]["md5"] == V72["eval"]["md5"]
    assert prov["labels"]["train"]["n_records"] == 4572
    assert prov["labels"]["eval"]["n_records"] == 147
    assert prov["geometry"]["frame"]["width"] == 640
    assert prov["derived_from"] is None                   # a NEW domain
    assert "EXPECTED" in prov["verification_status"]
    for key, (n, kind, sha, skips) in LEGACY_PINS.items():
        e = man["corpora"][key]
        assert (e["episode_count"], e["uid_kind"], e.get("episode_uid_sha256"),
                e["skip_count"]) == (n, kind, sha, skips), key
    assert list(man["corpora"]) == [*LEGACY_PINS, B1_KEY]


def test_the_B1_key_resolves_from_the_thor_cache_path_not_from_EVAL6():
    assert parity.corpus_key_of(
        "/home/nvidia/data/physicalai-b1-w120-256x640cyl") == B1_KEY
    assert parity.corpus_key_of(
        "/home/nvidia/data/physicalai-b1-EVAL6-w120-256x640cyl") is None
    # the key must appear in the directory name (register_v2_sibling's rule)
    assert B1_KEY in "/home/nvidia/data/physicalai-b1-w120-256x640cyl"


def test_a_B1_cache_with_the_right_count_but_wrong_membership_is_REFUSED(
        tmp_path):
    """The digest binds: 4,713 files of the wrong clips is refused, and so is
    a one-short cache. (Empty files — the guard reads NAMES, never bytes.)"""
    d = tmp_path / B1_KEY
    d.mkdir()
    for i in range(4713):
        (d / f"fake-{i:05d}.v2ep.pt").touch()
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_v2_parity_cache(d, label="b1", require=True)
    assert "MEMBERSHIP DIFFERS AT THE SAME COUNT" in str(ei.value)
    assert B1_DIGEST_4713 in str(ei.value)
    (d / "fake-04712.v2ep.pt").unlink()
    with pytest.raises(parity.ParityViolation) as ei2:
        parity.assert_v2_parity_cache(d, label="b1", require=True)
    assert "TRUNCATED by 1" in str(ei2.value)


# =========================================================================== #
# 8. the flags: default OFF, refused when inert, plumbed to the rollout
# =========================================================================== #
def test_the_flags_default_off_and_land_in_argparse():
    a = _args("--stage", "S-W", "--out", "unused")
    assert a.bptt_truncate == 0 and a.allow_any_labels is False
    a = _args("--stage", "S-W", "--out", "unused", "--bptt-truncate", "15",
              "--allow-any-labels")
    assert a.bptt_truncate == 15 and a.allow_any_labels is True


def test_preflight_refuses_inert_or_invalid_flags(tmp_path):
    p = preflight(_args(*_tiny_argv(tmp_path, "S-W", "--allow-any-labels")))
    assert any("--allow-any-labels without --s2-labels" in x for x in p)
    p = preflight(_args(*_tiny_argv(tmp_path, "S-W", "--bptt-truncate", "-1")))
    assert any("--bptt-truncate must be >= 0" in x for x in p)
    p = preflight(_args(*_tiny_argv(tmp_path, "S-W", "--bptt-truncate", "20",
                                    "--o5-k", "20")))
    assert any("NEVER fires" in x and "--bptt-truncate 20" in x for x in p)
    p = preflight(_args(*_tiny_argv(tmp_path, "S-W", "--bptt-truncate", "15",
                                    "--o5-k", "20")))
    assert not any("bptt" in x for x in p)


def test_bptt_truncate_reaches_the_rollout_from_v6_loss_step(monkeypatch):
    import tanitad.models.metric_dynamics as MD
    real = MD.rollout_transitions
    calls = []

    def _spy(*a, **kw):
        # (k, bptt): k is the 5th positional argument of rollout_transitions
        calls.append((int(a[4]), int(kw.get("bptt_truncate", 0))))
        return real(*a, **kw)

    monkeypatch.setattr(MD, "rollout_transitions", _spy)
    s = _stack()
    b = synthetic_train_batch(s, batch=4, k=12, seed=1)
    b["gt_wp"] = torch.randn(4, 10, 2, generator=torch.Generator().manual_seed(1))
    O1_K, O5_K, O11_K = 10, 12, 6
    w = V6LossWeights(o11_cf=1.0)                    # O5 + O11 rolls, not one
    kw = dict(stage="S-W", weights=w, o1_k=O1_K, o5_k=O5_K, o11_k=O11_K)
    torch.manual_seed(3)
    L0 = v6_loss_step(s, dict(b), generator=torch.Generator().manual_seed(11),
                      **kw)
    assert calls and {bt for _, bt in calls} == {0}
    assert "bptt_truncate" not in L0["log"]
    n0 = len(calls)
    calls.clear()
    torch.manual_seed(3)
    L4 = v6_loss_step(s, dict(b), generator=torch.Generator().manual_seed(11),
                      bptt_truncate=4, **kw)
    assert len(calls) == n0
    governed = [(k, bt) for k, bt in calls if k in (O5_K, O11_K)]
    o1_rolls = [(k, bt) for k, bt in calls if k == O1_K]
    # EVERY roll v6_loss_step makes itself got it: the O5 factual roll and
    # the O11 counterfactual (o11_negs=1) roll
    assert governed and {bt for _, bt in governed} == {4}, calls
    assert len(governed) == 2, calls
    # ⚠️ MEASURED SCOPE, pinned so it cannot go unnoticed: the O1 response-form
    # rolls (train_stage_a.stage_a_losses, k = o1_k) are NOT governed — that
    # file is outside this change; --help and the register row say so.
    assert o1_rolls and {bt for _, bt in o1_rolls} == {0}, calls
    assert len(governed) + len(o1_rolls) == len(calls), calls
    assert L4["log"]["bptt_truncate"] == 4
    # and the forward value is unchanged by the cut (the property that matters)
    assert float(L4["loss"].detach()) == pytest.approx(float(L0["loss"].detach()))
