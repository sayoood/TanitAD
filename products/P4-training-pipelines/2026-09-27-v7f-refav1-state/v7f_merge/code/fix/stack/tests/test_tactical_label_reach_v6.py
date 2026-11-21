"""R3 / D-TLIGHT-1 for v7F: EVERY tactical label family must REACH the v6-staged loss.

Sibling of ``test_tactical_label_reach.py`` (which pins the label LAYER and the refcv3 closure,
D-TACGOAL-1). This file pins the v6-STAGED trainer (``train_v6_staged.py``) — the v7F line — where,
at c36b6ddd, NO tactical label reached any loss (``v7f_r3/AUDIT.md``): the lat/lon ids rode the
batch unread, the 22-token goal SET (every traffic-light colour) was never projected, and in S-T the
label file was never loaded at all.

PI 2026-09-27 (R3, BINDING): *"All our tactical labels must be used to train the tactical layer to
estimate and choose the right tactical behaviors and goals which MUST condition the operative
planning."*

WHAT IS PINNED, with LITERAL expectations (never an expression over the code under test):

1. default OFF: ``--w-tac-label-all`` 0.0; the flag-off loss is BIT-IDENTICAL to the c36b6ddd
   trainer (content-anchored: git ``c36b6ddd:`` or ``$TANITAD_R3_REF_TRAINER``), with the new keys
   present in the batch or not; a RED<->GREEN swap moves NOTHING when off (D-TLIGHT-1 as measured);
2. ON, through the REAL v7.2 join: each traffic-light colour and every other goal family reaches a
   finite, non-zero-gradient term (goal rows of ``g_tac.type_head``; SPEED_BAND through its
   ``arg_head`` slots 0/1); the lat/lon ids reach their CE; the RED<->GREEN swap MOVES the loss;
3. DELIBERATE REGRESSIONS that must go RED: the join unwired (the term refuses), the goal BCE
   dropped (the reach detector fires), the colour dropped at the join (the detector fires);
4. the preflight refusals, the effective-weight row, and G-DVB (v6) on the BUILT stack, each with a
   mutation arm.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.data import v7_labels as V7L  # noqa: E402
from tanitad.models.v6 import STAGES, V6Config, V6Stack, apply_stage_freeze  # noqa: E402
from tanitad.train import declared_vs_built_v6 as DVB6  # noqa: E402
from s2_labels import stable_episode_id  # noqa: E402
import train_v6_staged as T  # noqa: E402

# --------------------------------------------------------------------------- #
# LITERALS                                                                     #
# --------------------------------------------------------------------------- #
#: the 22 v7 tactical goal tokens, in the FROZEN order (vocab_v7.py:95-113)
GOALS = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD",
         "STOP_POINT", "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE",
         "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R",
         "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW",
         "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
TL = ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW",
      "TRAFFIC_LIGHT_REACT_GREEN")
#: the tokens the v7.2 train blob emits from `vlm-cot` (MEASURED, raw/audit_census_train_blob.txt)
COT = ("CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "GAP_TARGET", "LANE_CHANGE_L", "LANE_CHANGE_R",
       "MERGE", "OVERTAKE_VEHICLE", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R",
       "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_GREEN", "TRAFFIC_LIGHT_REACT_RED",
       "TRAFFIC_LIGHT_REACT_YELLOW", "YIELD")
#: MEASURED on the v7.2 train blob under `measured`: positives, ZERO supervised negatives -> masked
MASKED_MEASURED = ("YIELD", "SPEED_BAND", "CORRIDOR_OFFSET", "GAP_TARGET", "REACT_ON_ONCOMING")
#: ...and under `all` only the constant token stays masked
MASKED_ALL = ("SPEED_BAND",)

CIDS = tuple(f"{i:08x}-68be-40df-8097-780bf1ae1c19" for i in range(9))
W, DT, N_STACK, T0 = 4, 0.1, 3, 8.0
T_IN_BAND = 75            # t_now = (75 + 4 - 1 + 2) * 0.1 = 8.0 s = t0
T_OUT_OF_BAND = 125       # t_now = 13.0 s: strategic band yes (+-11), tactical band no (+-2)

#: (goal set, lat, lon) per clip -- every goal token appears, every set is VALID
#: (vocab_v7.validate_goal_set), and the provenance structure is the train blob's
FIXTURE = (
    (("FOLLOW_LANE", "SPEED_BAND", "TRAFFIC_LIGHT_REACT_RED", "YIELD", "CORRIDOR_OFFSET"),
     "LANE_KEEP", "BRAKE_TO"),
    (("FOLLOW_LANE", "SPEED_BAND", "TRAFFIC_LIGHT_REACT_GREEN", "GAP_TARGET"),
     "LANE_KEEP", "ACCELERATE"),
    (("TURN_L", "SPEED_BAND", "TRAFFIC_LIGHT_REACT_YELLOW", "REACT_ON_ONCOMING"),
     "TURN_L", "ADAPT_SPEED_FOR_CURVE"),
    (("TURN_R", "SPEED_BAND", "TRAFFIC_LIGHT_REACT", "TAKE_EXIT_R"), "TURN_R", "CRUISE"),
    (("STOP_POINT", "SPEED_BAND", "EVADE_IN_CORRIDOR"), "NUDGE_L", "HOLD"),
    (("SPEED_BAND", "OVERTAKE_VEHICLE", "LANE_CHANGE_L"), "NUDGE_R", "FOLLOW"),
    (("SPEED_BAND", "MERGE", "LANE_CHANGE_R"), "LANE_KEEP", "CREEP"),
    (("SPEED_BAND", "TAKE_EXIT_L", "YIELD_FOR_TURN_L"), "TURN_L", "BRAKE_TO"),
    (("FOLLOW_LANE", "SPEED_BAND", "YIELD_FOR_TURN_R"), "LANE_KEEP", "CRUISE"),
)
#: SPEED_BAND (v_lo_ms, v_hi_ms) per clip, LITERAL
SPEED = tuple((2.0 * i, 2.0 * i + 1.5) for i in range(9))


def _goal_meta(tok: str, i: int) -> dict:
    if tok == "SPEED_BAND":
        return {"band_s": [2.0, 6.0], "held": True, "v_lo_ms": SPEED[i][0],
                "v_hi_ms": SPEED[i][1]}
    if tok == "FOLLOW_LANE":
        return {"provenance": "geometry"}
    if tok in COT:
        return {"provenance": "vlm-cot", "disputed": False, "time_basis": "untimed"}
    return {"within_m": 20.0}                        # geometry tokens with no provenance key


def _rec(i: int, goals=None) -> dict:
    gset, lat, lon = FIXTURE[i]
    gset = goals if goals is not None else gset
    return {"clip_id": CIDS[i], "schema_version": "s2-geom-v7", "vocab": "v7",
            "a_tac": {"lat": lat, "lon": lon, "truncated": False,
                      "lat_args": {"within_m": 3.0}, "lon_args": {"v_target_ms": 9.0}},
            "a_str": {"token": "HOLD_MAIN_ROAD", "args": {}},
            "g_str": {"token": "FOLLOW_ROUTE", "args": {}},
            "g_tac": {"anchor": {"goal_x_m": 50.0, "goal_y_m": 1.0, "t_reach_s": 6.0},
                      "goals": {t: _goal_meta(t, i) for t in gset}, "violations": []},
            "bands": {"operative_s": [0, 2], "tactical_s": [2, 6], "strategic_s": [8, 30],
                      "unassigned_manoeuvres": []},
            "t0_s": T0, "horizon": {"available_s": 30.0, "recording_span_s": 30.0},
            "nav_command": {"token": "NAV_FOLLOW_ROAD", "provenance": "ego-future",
                            "oracle": True, "args": {"distance_m": 0.0, "time_s": 0.0}},
            "turn_suppression": None,
            "alpamayo": {"lateral": {"agree": True}, "longitudinal": {"agree": True}}}


def _blob(tmp_path, recs, name="s2_labels_v7.2_train.jsonl.gz") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return p


def _swap_red_green(recs):
    """D-TLIGHT-1's own mutation: every RED <-> GREEN, nothing else."""
    out = json.loads(json.dumps(recs))
    sw = {"TRAFFIC_LIGHT_REACT_RED": "TRAFFIC_LIGHT_REACT_GREEN",
          "TRAFFIC_LIGHT_REACT_GREEN": "TRAFFIC_LIGHT_REACT_RED"}
    for r in out:
        r["g_tac"]["goals"] = {sw.get(k, k): v for k, v in r["g_tac"]["goals"].items()}
    return out


def _episodes():
    return [types.SimpleNamespace(episode_id=stable_episode_id(c),
                                  frames=torch.zeros(1, 3 * N_STACK, 2, 2)) for c in CIDS]


def _cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=W, horizons=(1,),
                                  action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, d_plan_feat=16, emission_hidden=16, n_candidates=3, aux_hidden=16,
        sigreg_slices=8, goal_multilabel=True)
    base.update(kw)                                   # tac_vocab_version defaults to v7.0
    return V6Config(**base)


def _stack(**kw) -> V6Stack:
    torch.manual_seed(0)
    return V6Stack(_cfg(**kw))


def _ns(**kw):
    """The argv side the policy reads (a Namespace, the trainer's own attribute names)."""
    return types.SimpleNamespace(**({"tac_goal_negatives": "measured",
                                     "cot_negative_sidecar": None} | kw))


def _join(tmp_path, stack, recs, *, negatives="measured", enable=True, t=T_IN_BAND):
    """The REAL v7.2 join: load -> supervision -> (R3 enable) -> batch of ONE window per clip at
    raw index ``t``. The join's index always holds an in-band AND an out-of-band window per clip
    (a join with no in-band window is refused at build time -- the term would never fire)."""
    ls = T.load_v72_labels_for_trainer(_blob(tmp_path, recs), allow_any_labels=True,
                                       stack=stack)
    index = [(e_i, tt) for tt in (T_IN_BAND, T_OUT_OF_BAND) for e_i in range(len(CIDS))]
    sup = ls.supervision(_episodes(), window=W, dt=DT, index=index)
    rep = None
    if enable:
        rep = T.build_tac_label_targets(_ns(tac_goal_negatives=negatives), ls, sup, stack)
    return sup.batch([i for i, (_e, tt) in enumerate(index) if tt == t]), rep


def _loss_batch(stack, keys: dict, seed=1) -> dict:
    b = T.synthetic_train_batch(stack, batch=len(CIDS), k=12, seed=seed)
    b["gt_wp"] = torch.randn(len(CIDS), 10, 2, generator=torch.Generator().manual_seed(seed))
    return b | keys


def _step(stack, batch, w=1.0, stage="S-T", **kw):
    torch.manual_seed(3)
    return T.v6_loss_step(stack, batch, stage=stage, weights=T.V6LossWeights(w_tac_label_all=w),
                          o1_k=10, o5_k=12, generator=torch.Generator().manual_seed(11), **kw)


def _goal_rows_reached(stack, term) -> set[str]:
    """Goal tokens whose g_tac.type_head ROW receives a non-zero gradient from ``term``."""
    g = torch.autograd.grad(term, stack.goal_head_tac.type_head.weight, retain_graph=True,
                            allow_unused=True)[0]
    if g is None:
        return set()
    return {GOALS[i] for i in range(len(GOALS)) if float(g[i].abs().sum()) > 0.0}


def _arg_rows_reached(stack, term) -> set[int]:
    g = torch.autograd.grad(term, stack.goal_head_tac.arg_head.weight, retain_graph=True,
                            allow_unused=True)[0]
    return set() if g is None else {i for i in range(g.shape[0]) if float(g[i].abs().sum()) > 0}


def _assert_every_family_reaches(stack, batch, masked: tuple[str, ...]):
    """THE REACH CHECK (the deliberate-regression arms call it and must see it go RED)."""
    L = _step(stack, batch)
    assert "tac_label_all" in L["log"]["terms"], L["log"]["terms"]
    term = L["tac_label_all"]
    assert bool(torch.isfinite(term)) and float(term.detach()) > 0.0
    reached = _goal_rows_reached(stack, term)
    want = set(GOALS) - set(masked)
    assert reached == want, (f"goal reach moved: lost={sorted(want - reached)} "
                             f"extra={sorted(reached - want)}")
    assert {"TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN",
            "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT"} <= reached
    # SPEED_BAND: its token is constant, so it reaches through its ARGS (slots 0/1 only)
    assert _arg_rows_reached(stack, term) == {0, 1}
    return L


# =========================================================================== #
# 1. DEFAULT OFF                                                               #
# =========================================================================== #
def test_the_flags_default_off_and_the_weight_is_stage_gated():
    ap = T.build_parser()
    a = ap.parse_args(["--stage", "S-T", "--out", "unused"])
    assert a.w_tac_label_all == 0.0
    assert a.tac_goal_negatives == "measured"
    assert a.cot_negative_sidecar is None
    assert T.V6LossWeights().w_tac_label_all == 0.0
    on = T.V6LossWeights(w_tac_label_all=1.0)
    assert on.for_stage("S-W").w_tac_label_all == 0.0
    assert on.for_stage("S-T").w_tac_label_all == 1.0
    assert on.for_stage("S-S").w_tac_label_all == 0.0
    assert on.for_stage("S-J").w_tac_label_all == 1.0
    assert T.W_TERM_FLAGS["w_tac_label_all"] == ("--w-tac-label-all", "w_tac_label_all")
    assert "w_tac_label_all" in T.stage_zeroed_terms("S-W")
    assert "w_tac_label_all" in T.stage_zeroed_terms("S-S")
    assert "w_tac_label_all" not in T.stage_zeroed_terms("S-T")


def test_OFF_the_R3_keys_are_not_emitted_and_no_term_or_log_key_appears(tmp_path):
    s = _stack()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], enable=False)
    for k in T.TAC_LABEL_BATCH_KEYS:
        assert k not in keys                           # the incumbent key set, byte-identical
    L = T.v6_loss_step(s, _loss_batch(s, keys), stage="S-T", weights=T.V6LossWeights(),
                       o1_k=10, o5_k=12)
    assert "tac_label_all" not in L and "tac_label_all" not in L["log"]["terms"]
    assert not any(k.startswith("tac_") for k in L["log"])


def _ref_trainer():
    """The c36b6ddd trainer as a module -- CONTENT-anchored (C75: never HEAD).

    `$TANITAD_R3_REF_TRAINER` (a file) wins; else `git show c36b6ddd:...` from the repo."""
    src = None
    env = os.environ.get("TANITAD_R3_REF_TRAINER")
    if env and Path(env).exists():
        src = Path(env).read_bytes()
    else:
        try:
            r = subprocess.run(["git", "show", "c36b6ddd:stack/scripts/train_v6_staged.py"],
                               cwd=_ROOT, capture_output=True, timeout=120)
            if r.returncode == 0 and r.stdout:
                src = r.stdout
        except Exception:
            src = None
    if src is None or b"w_tac_label_all" in src:
        return None
    tmp = Path(tempfile.mkdtemp()) / "train_v6_staged_c36b6ddd.py"
    tmp.write_bytes(src)
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("train_v6_staged_c36b6ddd", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_v6_staged_c36b6ddd"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("with_r3_keys", [False, True])
def test_OFF_the_loss_is_BIT_IDENTICAL_to_the_c36b6ddd_trainer(tmp_path, stage, with_r3_keys):
    """⛔ THE ONE THAT PROTECTS EVERY FLAG-OFF RUN: same stack, same batch, same seeds -- every
    term, every log key, and the global RNG stream equal the c36b6ddd trainer's. With the R3 keys
    PRESENT in the batch too: an OFF term must not read them."""
    old = _ref_trainer()
    if old is None:
        pytest.skip("no c36b6ddd reference: set TANITAD_R3_REF_TRAINER or run inside the repo")
    s = _stack()
    s.eval()
    keys = {}
    if with_r3_keys:
        keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)])
    b = _loss_batch(s, keys)
    kw = dict(stage=stage, o1_k=10, o5_k=12)
    torch.manual_seed(3)
    lo = old.v6_loss_step(s, b, weights=old.V6LossWeights(),
                          generator=torch.Generator().manual_seed(11), **kw)
    rng_old = torch.random.get_rng_state().clone()
    torch.manual_seed(3)
    ln = T.v6_loss_step(s, b, weights=T.V6LossWeights(),
                        generator=torch.Generator().manual_seed(11), **kw)
    rng_new = torch.random.get_rng_state().clone()
    assert torch.equal(lo["loss"], ln["loss"]), f"{stage}: the DEFAULT loss MOVED vs c36b6ddd"
    assert lo["log"]["terms"] == ln["log"]["terms"]
    for t in lo["log"]["terms"]:
        assert torch.equal(lo[t], ln[t]), f"{stage}: term {t} moved"
    assert torch.equal(rng_old, rng_new), "the default path drew a different number of numbers"
    assert set(lo["log"]) == set(ln["log"]), (set(lo["log"]) ^ set(ln["log"]))


def test_NEGATIVE_CONTROL_the_identity_check_can_fail(tmp_path):
    s = _stack()
    s.eval()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)])
    b = _loss_batch(s, keys)
    off = _step(s, b, w=0.0)
    on = _step(s, b, w=1.0)
    assert not torch.equal(off["loss"], on["loss"])
    assert "tac_label_all" in on["log"]["terms"] and "tac_label_all" not in off["log"]["terms"]


def test_OFF_a_RED_GREEN_swap_moves_NOTHING_and_ON_it_MOVES_the_loss(tmp_path):
    """D-TLIGHT-1, measured the way it was found: a RED<->GREEN mutation over the coloured records
    moved 0 supervised records at c36b6ddd. OFF that is still true (the colour is invisible, the
    flag-off trainer is the incumbent); ON the colour reaches the loss."""
    s = _stack()
    s.eval()
    recs = [_rec(i) for i in range(9)]
    k0, _ = _join(tmp_path / "a", s, recs, enable=False)
    k1, _ = _join(tmp_path / "b", s, _swap_red_green(recs), enable=False)
    assert float(_step(s, _loss_batch(s, k0), w=0.0)["loss"].detach()) == \
        float(_step(s, _loss_batch(s, k1), w=0.0)["loss"].detach())
    k0, _ = _join(tmp_path / "c", s, recs)
    k1, _ = _join(tmp_path / "d", s, _swap_red_green(recs))
    red, green = GOALS.index("TRAFFIC_LIGHT_REACT_RED"), GOALS.index("TRAFFIC_LIGHT_REACT_GREEN")
    assert float(k0["tac_goal_y"][0, red]) == 1.0 and float(k1["tac_goal_y"][0, red]) == 0.0
    assert float(k1["tac_goal_y"][0, green]) == 1.0
    l0 = float(_step(s, _loss_batch(s, k0))["tac_label_all"].detach())
    l1 = float(_step(s, _loss_batch(s, k1))["tac_label_all"].detach())
    assert l0 != l1, "the colour swap did not move the R3 term -- the colour is not reaching it"


# =========================================================================== #
# 2. ON -- the join, literal                                                   #
# =========================================================================== #
def test_the_join_projects_the_goal_set_speed_band_args_and_ids_LITERALLY(tmp_path):
    s = _stack()
    keys, rep = _join(tmp_path, s, [_rec(i) for i in range(9)])
    assert set(T.TAC_LABEL_BATCH_KEYS) <= set(keys)
    assert tuple(keys["tac_goal_y"].shape) == (9, 22)
    assert bool(keys["tac_valid"].all())
    ix = GOALS.index
    y0, w0 = keys["tac_goal_y"][0], keys["tac_goal_w"][0]   # clip 0: RED, YIELD, CORRIDOR_OFFSET
    assert [float(y0[ix(t)]) for t in TL] == [0.0, 1.0, 0.0, 0.0]
    assert [float(w0[ix(t)]) for t in TL] == [1.0, 1.0, 1.0, 1.0]   # colours ENTAILED negative
    assert float(y0[ix("YIELD")]) == 1.0 and float(w0[ix("YIELD")]) == 1.0
    assert float(y0[ix("GAP_TARGET")]) == 0.0 and float(w0[ix("GAP_TARGET")]) == 0.0   # CoT: ignored
    assert float(y0[ix("FOLLOW_LANE")]) == 1.0
    assert float(w0[ix("TURN_L")]) == 1.0 and float(y0[ix("TURN_L")]) == 0.0         # geometry
    # the class mask is the MEASURED five (the train blob's, reproduced by the fixture)
    cm = keys["tac_goal_class_mask"]
    assert {GOALS[i] for i in range(22) if float(cm[i]) == 0.0} == set(MASKED_MEASURED)
    assert rep["n_trainable"] == 17 and sorted(rep["masked_why"]) == sorted(MASKED_MEASURED)
    # SPEED_BAND (v_lo, v_hi) -> slots 0/1, nothing else
    assert keys["tac_goal_args"][3, :2].tolist() == [6.0, 7.5]
    assert keys["tac_goal_arg_mask"][3].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rep["join"]["n_joined_records_with_speed_band_args"] == 9
    # the action ids: the v7 HEADS indices of the fixture's tokens
    assert keys["tac_lat_id"].tolist() == [0, 0, 6, 7, 4, 5, 0, 6, 0]
    assert keys["tac_lon_id"].tolist() == [3, 7, 6, 1, 5, 0, 4, 3, 1]


def test_the_join_report_names_the_consumer_only_once_R3_is_enabled(tmp_path):
    """The record must not contradict the loss (MEASURED on the first real-train() smoke:
    config.json said "tactical_consumer: NONE" while the term was in force, because the report
    was snapshotted before the targets were enabled -- train() now re-reads it)."""
    s = _stack()
    ls = T.load_v72_labels_for_trainer(_blob(tmp_path, [_rec(i) for i in range(9)]),
                                       allow_any_labels=True, stack=s)
    index = [(e_i, tt) for tt in (T_IN_BAND, T_OUT_OF_BAND) for e_i in range(len(CIDS))]
    sup = ls.supervision(_episodes(), window=W, dt=DT, index=index)
    before = sup.report()
    assert before["tactical_consumer"].startswith("NONE")
    assert "tac_label_targets" not in before
    T.build_tac_label_targets(_ns(), ls, sup, s)
    after = sup.report()
    assert after["tactical_consumer"].startswith("v6_loss_step term 'tac_label_all'")
    assert after["tac_label_targets"]["negatives"] == "measured"
    assert after["tac_label_targets"]["n_joined_records"] == 9


def test_out_of_band_windows_carry_no_tactical_evidence(tmp_path):
    s = _stack()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], t=T_OUT_OF_BAND)
    assert not bool(keys["tac_valid"].any())
    assert float(keys["tac_goal_w"].sum()) == 0.0 and float(keys["tac_goal_arg_mask"].sum()) == 0.0
    assert keys["tac_lat_id"].tolist() == [-100] * 9


# =========================================================================== #
# 3. ON -- every family REACHES a finite, non-zero-gradient term               #
# =========================================================================== #
def test_ON_all_policy_every_goal_family_and_every_colour_reaches_the_loss(tmp_path):
    s = _stack()
    keys, rep = _join(tmp_path, s, [_rec(i) for i in range(9)], negatives="all")
    assert rep["n_trainable"] == 21 and sorted(rep["masked_why"]) == list(MASKED_ALL)
    L = _assert_every_family_reaches(s, _loss_batch(s, keys), MASKED_ALL)
    lg = L["log"]
    assert lg["tac_lat_tok_counts"] == {"LANE_KEEP": 4, "NUDGE_L": 1, "NUDGE_R": 1,
                                        "TURN_L": 2, "TURN_R": 1}
    assert lg["tac_lon_tok_counts"] == {"ACCELERATE": 1, "ADAPT_SPEED_FOR_CURVE": 1,
                                        "BRAKE_TO": 2, "CREEP": 1, "CRUISE": 2, "FOLLOW": 1,
                                        "HOLD": 1}
    assert lg["tac_goal_pos_counts"]["TRAFFIC_LIGHT_REACT_RED"] == 1
    assert lg["tac_goal_pos_counts"]["TRAFFIC_LIGHT_REACT_GREEN"] == 1
    assert lg["tac_goal_pos_counts"]["TRAFFIC_LIGHT_REACT_YELLOW"] == 1
    assert lg["tac_goal_pos_counts"]["TRAFFIC_LIGHT_REACT"] == 1
    assert lg["tac_speedband_n_slots"] == 18
    for k in ("tac_lat_ce", "tac_lon_ce", "tac_goal_bce", "tac_speedband_l1_ms"):
        assert lg[k] is not None and lg[k] > 0.0
    # the action heads get a gradient from the term
    for h in ("act_head_lat", "act_head_lon"):
        g = torch.autograd.grad(L["tac_label_all"], getattr(s, h).type_head.weight,
                                retain_graph=True)[0]
        assert float(g.abs().sum()) > 0.0, h


def test_ON_measured_policy_colours_reach_and_the_five_masked_rows_are_EXACTLY_zero(tmp_path):
    s = _stack()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], negatives="measured")
    L = _assert_every_family_reaches(s, _loss_batch(s, keys), MASKED_MEASURED)
    g = torch.autograd.grad(L["tac_label_all"], s.goal_head_tac.type_head.weight)[0]
    for t in MASKED_MEASURED:
        assert float(g[GOALS.index(t)].abs().sum()) == 0.0, t


def test_ON_the_gradient_reaches_ONLY_tactical_goal_heads_in_S_T(tmp_path):
    """HEADS ONLY, measured on the real graph under the REAL S-T freeze: the full S-T backward with
    R3 on populates .grad inside layer_tac/planner only, and the R3 term ALONE reaches only the
    three tactical heads (+ the frozen-in-S-T strategic goal path it reads as cond)."""
    s = _stack()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], negatives="all")
    b = _loss_batch(s, keys)
    for p in s.parameters():
        p.requires_grad_(True)
    L = _step(s, b)
    names = [n for n, _ in s.named_parameters()]
    grads = torch.autograd.grad(L["tac_label_all"], [p for _, p in s.named_parameters()],
                                allow_unused=True, retain_graph=True)
    live = {n for n, g in zip(names, grads) if g is not None and float(g.abs().max()) > 0}
    assert {n.split(".")[0] for n in live} == {"goal_head_tac", "act_head_lat", "act_head_lon",
                                               "goal_head_str", "vocab_str"}
    assert not any(n.startswith(("encoder.", "readout.", "adapter_", "predictor_"))
                   for n in live)
    s2 = _stack()
    apply_stage_freeze(s2, "S-T")
    L2 = _step(s2, _loss_batch(s2, keys))
    L2["loss"].backward()
    got = {n for n, p in s2.named_parameters()
           if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert {s2.group_of(n) for n in got} <= {"layer_tac", "planner"}
    assert any(n.startswith("goal_head_tac.type_head") for n in got)


# =========================================================================== #
# 4. DELIBERATE REGRESSIONS -- each must go RED                                #
# =========================================================================== #
def test_RED_the_join_unwired_makes_the_term_REFUSE(tmp_path, monkeypatch):
    s = _stack()
    cls = T._v72_classes()["V72WindowSupervision"]
    monkeypatch.setattr(cls, "enable_tac_label_targets", lambda self, **kw: {})
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)])
    assert "tac_goal_y" not in keys
    with pytest.raises(ValueError, match="missing"):
        _step(s, _loss_batch(s, keys))


def test_RED_the_goal_BCE_dropped_makes_the_reach_check_fire(tmp_path, monkeypatch):
    s = _stack()
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], negatives="all")
    b = _loss_batch(s, keys)
    import tanitad.refs.tac_goal_head as H
    monkeypatch.setattr(H, "tac_goal_loss", lambda logits, y, w, **kw: (logits.sum() * 0.0, 0))
    with pytest.raises(AssertionError, match="goal reach moved"):
        _assert_every_family_reaches(s, b, MASKED_ALL)


def test_RED_the_colour_dropped_at_the_join_makes_the_reach_check_fire(tmp_path, monkeypatch):
    s = _stack()
    real = V7L.tactical_goal_targets
    idx = [GOALS.index(t) for t in TL]

    def _colourblind(label, t_now_s, **kw):
        y, w = real(label, t_now_s, **kw)
        y, w = list(y), list(w)
        for i in idx:
            y[i], w[i] = 0.0, 0.0
        return tuple(y), tuple(w)

    monkeypatch.setattr(V7L, "tactical_goal_targets", _colourblind)
    keys, _ = _join(tmp_path, s, [_rec(i) for i in range(9)], negatives="all")
    with pytest.raises(AssertionError, match="goal reach moved"):
        _assert_every_family_reaches(s, _loss_batch(s, keys), MASKED_ALL)


# =========================================================================== #
# 5. structural refusals in the loss                                           #
# =========================================================================== #
@pytest.mark.parametrize("kw,match", [({"goal_multilabel": False}, "goal_multilabel"),
                                      ({"goal_factored": True}, "goal-factored")])
def test_the_term_refuses_a_head_that_would_not_condition_the_planner(tmp_path, kw, match):
    s = _stack(**kw)
    b = _loss_batch(s, T.synthetic_tac_label_batch(s, len(CIDS), seed=2))
    with pytest.raises(ValueError, match=match):
        _step(s, b)


def test_the_term_refuses_without_the_planner_cut():
    s = _stack(isolate_planner_from_encoder=False)
    b = _loss_batch(s, T.synthetic_tac_label_batch(s, len(CIDS), seed=2))
    with pytest.raises(ValueError, match="TRUNK loss"):
        _step(s, b)


def test_the_term_refuses_a_v6_vocabulary_head():
    s = _stack(tac_vocab_version="v6.0")
    b = _loss_batch(s, T.synthetic_tac_label_batch(s, len(CIDS), seed=2))
    with pytest.raises(ValueError, match="v7.2 label vocabulary"):
        _step(s, b)


# =========================================================================== #
# 6. PREFLIGHT                                                                 #
# =========================================================================== #
def _pf(tmp_path, stage, *extra, dry=True):
    argv = ["--stage", stage, "--out", str(tmp_path / "out"), "--horizons", "1",
            *(["--dry-run"] if dry else ["--v2-cache", str(tmp_path)]), *extra]
    ap = T.build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack")
    return T.preflight(ap.parse_args(argv))


def _hits(problems, needle):
    return [p for p in problems if needle in p]


def test_preflight_refuses_the_term_where_layer_tac_is_frozen(tmp_path):
    for st in ("S-W", "S-S"):
        p = _pf(tmp_path, st, "--w-tac-label-all", "1", "--goal-multilabel")
        assert _hits(p, f"--w-tac-label-all 1.0 in {st}"), p
    assert not _hits(_pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel"),
                     "--w-tac-label-all 1.0 in")


def test_preflight_refuses_the_heads_that_would_not_condition_the_planner(tmp_path):
    assert _hits(_pf(tmp_path, "S-T", "--w-tac-label-all", "1"), "without --goal-multilabel")
    assert _hits(_pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel",
                     "--goal-factored"), "with --goal-factored")
    assert _hits(_pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel",
                     "--tac-vocab-version", "v6.1"), "--tac-vocab-version v6.1")


def test_preflight_label_rules_and_the_SCOPED_s2_relaxation(tmp_path):
    blob = _blob(tmp_path, [_rec(i) for i in range(9)])
    # a REAL run without labels is refused ...
    assert _hits(_pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel", dry=False),
                 "without --s2-labels")
    # ... with the v7.2 blob and w_s2_goal 0 the incumbent "labels with no consumer" rule must
    # NOT fire (R3 IS the consumer) ...
    p = _pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel",
            "--s2-labels", str(blob), "--allow-any-labels")
    assert not _hits(p, "with --w-s2-goal 0"), p
    assert not _hits(p, "--w-tac-label-all"), p
    # ... and the relaxation is SCOPED: without R3 the incumbent refusal still fires
    assert _hits(_pf(tmp_path, "S-T", "--s2-labels", str(blob), "--allow-any-labels"),
                 "with --w-s2-goal 0")


def test_preflight_refuses_a_v1_schema_blob_for_the_term(tmp_path):
    d = tmp_path / "v1"
    d.mkdir()
    (d / "s2_labels_x.jsonl").write_text(json.dumps({"schema_version": "s2-strategic-v1",
                                                     "clip_id": CIDS[0]}))
    p = _pf(tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel", "--s2-labels",
            str(d))
    assert _hits(p, "route v1"), p


def test_preflight_the_policy_flags_are_inert_without_the_term_and_need_each_other(tmp_path):
    assert _hits(_pf(tmp_path, "S-T", "--tac-goal-negatives", "all"),
                 "--tac-goal-negatives all without --w-tac-label-all")
    sc = tmp_path / "sc.json"
    sc.write_text("{}")
    assert _hits(_pf(tmp_path, "S-T", "--cot-negative-sidecar", str(sc)),
                 "without --w-tac-label-all")
    on = ("--w-tac-label-all", "1", "--goal-multilabel")
    assert _hits(_pf(tmp_path, "S-T", *on, "--tac-goal-negatives", "cot-absence-negative"),
                 "without --cot-negative-sidecar")
    assert _hits(_pf(tmp_path, "S-T", *on, "--cot-negative-sidecar", str(sc)),
                 "would be IGNORED")
    assert not _hits(_pf(tmp_path, "S-T", *on, "--tac-goal-negatives", "cot-absence-negative",
                         "--cot-negative-sidecar", str(sc)), "sidecar")


def test_preflight_OFF_adds_no_problem(tmp_path):
    base = _pf(tmp_path, "S-T")
    assert not any("tac-label" in p or "tac-goal-negatives" in p for p in base)


# =========================================================================== #
# 7. G-DVB (v6) -- declared vs BUILT, each check shown able to fail            #
# =========================================================================== #
def _argv_ns(*extra):
    return T.build_parser().parse_args(["--stage", "S-T", "--out", "unused", *extra])


def test_gdvb_v6_a_correct_build_is_clean_and_every_new_flag_is_covered():
    s = _stack()
    a = _argv_ns("--w-tac-label-all", "1", "--goal-multilabel")
    wf = T._weights_from_args(a).for_stage("S-T")
    assert DVB6.check_v6(s, a, weights_in_force=wf, parser=T.build_parser()) == []
    assert DVB6.coverage_v6(T.build_parser()) == []
    assert DVB6.NEW_V6_DESTS == ("w_tac_label_all", "tac_goal_negatives", "cot_negative_sidecar")
    # OFF: nothing built, nothing to mismatch
    a0 = _argv_ns()
    assert DVB6.check_v6(_stack(goal_multilabel=False), a0,
                         weights_in_force=T._weights_from_args(a0).for_stage("S-T")) == []


@pytest.mark.parametrize("build,where", [
    ({"goal_multilabel": False}, "stack.goal_head_tac.multilabel"),
    ({"goal_factored": True}, "stack.goal_head_tac_lat"),
    ({"tac_vocab_version": "v6.0"}, "stack.vocab_tac.tokens"),
    ({"isolate_planner_from_encoder": False}, "stack.cfg.isolate_planner_from_encoder"),
])
def test_gdvb_v6_RED_a_stack_built_otherwise_is_named(build, where):
    s = _stack(**build)
    a = _argv_ns("--w-tac-label-all", "1", "--goal-multilabel")
    bad = DVB6.check_v6(s, a, weights_in_force=T._weights_from_args(a).for_stage("S-T"))
    assert any(m.read_from == where for m in bad), [str(m) for m in bad]
    with pytest.raises(SystemExit, match="G-DVB v6"):
        DVB6.refuse_on_mismatch_v6(s, a,
                                   weights_in_force=T._weights_from_args(a).for_stage("S-T"))


def test_gdvb_v6_RED_the_weight_the_loss_reads_differs_from_argv():
    s = _stack()
    a = _argv_ns("--w-tac-label-all", "1", "--goal-multilabel")
    # S-W's for_stage zeroes it: declared 1.0, read 0.0
    bad = DVB6.check_v6(s, a, weights_in_force=T._weights_from_args(a).for_stage("S-W"))
    assert [m.lever for m in bad] == ["--w-tac-label-all"] and bad[0].built == 0.0
    # OFF in argv but ON in the weights the loss is handed
    a0 = _argv_ns()
    bad = DVB6.check_v6(s, a0, weights_in_force=T.V6LossWeights(w_tac_label_all=1.0))
    assert [m.declared for m in bad] == [0.0]


def test_gdvb_v6_RED_coverage_names_a_new_flag_without_an_entry_and_a_dropped_flag():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-tac-label-all", type=float, default=0.0)
    ap.add_argument("--tac-goal-negatives", default="measured")
    assert DVB6.coverage_v6(ap) == ["cot_negative_sidecar"]          # dropped from the parser
    assert DVB6.coverage_v6(T.build_parser(), dests=DVB6.NEW_V6_DESTS + ("w_brand_new",)) == \
        ["w_brand_new"]                                               # no registry entry
    with pytest.raises(ValueError, match="no check"):
        DVB6.register_v6("x_loss", "loss")
    with pytest.raises(ValueError, match="reason"):
        DVB6.register_v6("x_data", "data")


# =========================================================================== #
# 8. the dry-run smoke exercises the REAL label policy                         #
# =========================================================================== #
def _tiny_argv(tmp_path, stage, *extra):
    return ["--stage", stage, "--out", str(tmp_path / "out"), "--dry-run",
            "--in-channels", "3", "--frame-h", "32", "--frame-w", "32", "--patch", "16",
            "--enc-dim", "32", "--enc-depth", "1", "--enc-heads", "2", "--readout-grid", "4",
            "--readout-dim", "8", "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "--d-tac", "32", "--d-str", "16",
            "--d-goal-embed", "16", "--adapter-hidden", "32", "--n-candidates", "3",
            "--sigreg-slices", "8", "--dry-steps", "2", "--dry-batch", "4", "--dry-k", "12",
            *extra]


def test_dry_run_ON_exercises_the_policy_on_the_real_blob_and_the_term_steps(tmp_path):
    blob = _blob(tmp_path, [_rec(i) for i in range(9)])
    a = T.build_parser().parse_args(_tiny_argv(
        tmp_path, "S-T", "--w-tac-label-all", "1", "--goal-multilabel",
        "--s2-labels", str(blob), "--allow-any-labels"))
    r = T.dry_run(a)
    rep = r["tac_label_all"]
    assert rep["exercised"] is True and rep["negatives"] == "measured"
    assert rep["n_trainable"] == 17 and rep["gdvb_v6"].startswith("PASS")
    for row in r["steps"]:
        assert "tac_label_all" in row["terms"]
        assert row["tac_goal_n_supervised"] > 0
    off = T.dry_run(T.build_parser().parse_args(_tiny_argv(tmp_path / "off", "S-T")))
    assert "tac_label_all" not in off and "tac_label_all" not in off["steps"][0]["terms"]
