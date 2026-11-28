"""Add the DATASET-side coverage for `--w-tac-goal`.

⛔ THE GAP THIS CLOSES IN MY OWN WORK. The gradient probe and the guard test
both INJECT `tac_goal_y`/`tac_goal_w` into the batch by hand, exactly as they
inject `lat_v7`/`lon_v7` — so neither of them ever executes
``V3Dataset.__getitem__``'s new emission block. A term that is proven to reach
a gradient from a hand-made target, and whose real target path is untested, is
the `--wp-index` shape: parsed, stamped, and inert on the only path that
matters.
"""
from __future__ import annotations

import ast
import io
import sys

SRC, DST = sys.argv[1], sys.argv[2]
with io.open(SRC, "rb") as fh:
    _raw = fh.read()
CRLF, LF = _raw.count(b"\r\n"), _raw.count(b"\n")
NEWLINE = "\r\n" if CRLF * 2 > LF else "\n"
print(f"source newline: CRLF={CRLF} LF={LF} -> {NEWLINE!r}")
s = _raw.decode("utf-8").replace("\r\n", "\n")

NEW = '''

# ==========================================================================
# 6 — --w-tac-goal: THE TARGET REACHES THE BATCH FROM THE REAL LOADER
# ==========================================================================
#: ⛔ The tokens the v7.2 blob emits from GEOMETRY, so an ABSENT one is a real
#: negative. Pinned as a LITERAL here exactly as ``load_v7_labels`` pins it
#: from the blob it read -- never imported from the code under test.
_GEOM = frozenset({"FOLLOW_LANE", "SPEED_BAND", "STOP_POINT", "TURN_L",
                   "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"})


def _labelled_ds(tr, monkeypatch, *, enabled: bool,
                 negatives: str = "measured"):
    """A real ``V3Dataset`` over synthetic episodes with a v7.2 join.

    ⭐ Episode 9000 CARRIES a record; episode 9001 deliberately does NOT. The
    second one is the point: a clip with no record must still get both keys, as
    an explicitly all-ignored row. A batch that sometimes carries the key and
    sometimes does not would make the loss-time refusal fire at random instead
    of at launch.
    """
    import torch
    from tanitad.data import v7_labels as v7l
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", _GEOM)
    cfg = v3.refc_v3_smoke_config(True)
    eps = tr._synth_episodes(2, cfg.core, seed=0)
    for i, e in enumerate(eps):
        e.episode_id = str(9000 + i)          # the loader does int(episode_id)
    ds = tr.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                      channels=cfg.core.encoder.in_channels)
    lab = v7l.V7Label(
        clip_id="clip_9000", tac_lat="LANE_KEEP", tac_lon="CRUISE",
        str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
        tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=0.3,
        horizon={}, tac_goals=frozenset({"FOLLOW_LANE", "SPEED_BAND"}),
        tac_goal_meta={"FOLLOW_LANE": {"provenance": "geometry"},
                       "SPEED_BAND": {"provenance": "geometry"}})
    ds.v7_by_sid = {9000: lab}                # 9001 is ABSENT on purpose
    ds.v7_dt = 0.1
    ds.tac_goal_targets = enabled
    ds.tac_goal_negatives = negatives
    first = {e_i: i for i, (e_i, _t) in reversed(list(enumerate(ds.index)))}
    return ds, first, v7l, torch


def test_the_dataset_emits_no_goal_target_when_the_channel_is_OFF(monkeypatch):
    """⛔ THE BIT-IDENTITY PRECONDITION AT THE LOADER. Default OFF must add no
    key at all -- an extra key would change the collated batch of a live
    recipe that never asked for the term."""
    tr = _trainer()
    ds, first, _v7l, _torch = _labelled_ds(tr, monkeypatch, enabled=False)
    item = ds[first[0]]
    assert "tac_goal_y" not in item and "tac_goal_w" not in item, (
        "the loader emitted a goal-set target on an arm that did not ask for "
        "it -- the OFF path is no longer bit-identical")
    # control, same breath: the join IS live, so the absence above is the
    # CHANNEL being off and not the label being missing
    assert "lat_v7" in item, "the v7 join is not live -- the test proves nothing"


def test_the_dataset_emits_the_goal_target_when_the_channel_is_ON(monkeypatch):
    """⭐ The real ``__getitem__`` path -- the one the gradient probes bypass."""
    tr = _trainer()
    ds, first, v7l, torch = _labelled_ds(tr, monkeypatch, enabled=True)
    item = ds[first[0]]
    assert "tac_goal_y" in item and "tac_goal_w" in item, (
        "the loader did NOT emit the goal-set target with the channel on -- "
        "`--w-tac-goal` would refuse at loss time on the real corpus")
    assert tuple(item["tac_goal_y"].shape) == (22,), item["tac_goal_y"].shape
    assert tuple(item["tac_goal_w"].shape) == (22,), item["tac_goal_w"].shape
    assert item["tac_goal_y"].dtype == torch.float32
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}
    # the two tokens this record carries are POSITIVE and supervised
    for tok in ("FOLLOW_LANE", "SPEED_BAND"):
        assert float(item["tac_goal_y"][ix[tok]]) == 1.0, tok
        assert float(item["tac_goal_w"][ix[tok]]) == 1.0, tok
    # a geometry token this record does NOT carry is a supervised NEGATIVE
    assert float(item["tac_goal_y"][ix["TURN_L"]]) == 0.0
    assert float(item["tac_goal_w"][ix["TURN_L"]]) == 1.0
    # ⛔ and a CoT-only token is IGNORED, never supervised as a negative:
    # 3,574 of 4,572 clips were never asked the traffic-light question.
    assert float(item["tac_goal_w"][ix["TRAFFIC_LIGHT_REACT_RED"]]) == 0.0


def test_a_clip_with_NO_record_still_gets_both_keys_all_ignored(monkeypatch):
    """⛔ NEVER A MISSING KEY. Episode 9001 has no v7.2 record; it must still
    carry both tensors, with every cell explicitly ignored."""
    tr = _trainer()
    ds, first, _v7l, _torch = _labelled_ds(tr, monkeypatch, enabled=True)
    item = ds[first[1]]
    assert "tac_goal_y" in item and "tac_goal_w" in item, (
        "a clip with no v7.2 record dropped the key entirely -- the batch "
        "would be ragged and the loss-time refusal would fire at random")
    assert float(item["tac_goal_w"].sum()) == 0.0, (
        "a clip with NO record supervised something -- an unmapped episode "
        "must train nothing, not train a default")
    assert float(item["tac_goal_y"].sum()) == 0.0


def test_the_negatives_policy_is_READ_and_not_hardcoded(monkeypatch):
    """⭐ THE DISCRIMINATING CONTROL for `--tac-goal-negatives`.

    A knob that is parsed and stamped but never reaches the derivation is the
    `--wp-index` defect (3 of 6 knobs inert). Under ``all`` the CoT-only
    traffic-light cell MUST become supervised; under ``measured`` it must not.
    If both read the same, the knob is inert.
    """
    tr = _trainer()
    ds_m, first, v7l, _t = _labelled_ds(tr, monkeypatch, enabled=True,
                                        negatives="measured")
    ds_a, first_a, _v, _t2 = _labelled_ds(tr, monkeypatch, enabled=True,
                                          negatives="all")
    ix = {t: i for i, t in enumerate(v7l.TAC_GOAL_TOKENS)}
    j = ix["TRAFFIC_LIGHT_REACT_RED"]
    w_measured = float(ds_m[first[0]]["tac_goal_w"][j])
    w_all = float(ds_a[first_a[0]]["tac_goal_w"][j])
    assert w_measured == 0.0, (
        f"--tac-goal-negatives=measured supervised a CoT-only cell "
        f"(w={w_measured}); 3,574 of 4,572 clips were never asked the "
        f"traffic-light question, so that is training on no evidence")
    assert w_all == 1.0, (
        f"--tac-goal-negatives=all left the CoT-only cell ignored (w={w_all}) "
        f"-- the knob is PARSED, STAMPED AND INERT, which is the --wp-index "
        f"defect verbatim")
'''

s = s.rstrip("\n") + "\n" + NEW
ast.parse(s)
assert s.count("def test_the_dataset_emits_the_goal_target_when_the_channel_is_ON") == 1

with io.open(DST, "wb") as fh:
    fh.write(s.replace("\n", NEWLINE).encode("utf-8"))
print(f"WROTE {DST} lines={s.count(chr(10))}")
