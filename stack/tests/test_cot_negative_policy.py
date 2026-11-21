"""The 2026-09-16 absence-is-negative policy, PROVEN BY MUTATION.

⛔⛔ EVERY GUARD HERE IS EXERCISED BY REINTRODUCING THE DEFECT IT EXISTS FOR,
never by reading the code and asserting the code is there. An AST census once
read ZERO suspects on both the fixed AND the broken trainer in this programme;
a guard is proven when the failure branch is shown to be REACHABLE, and that
means mutating the artifact and watching the alarm fire.

The table is the deliverable: :data:`MUTATIONS` names, for each guard, the
exact corruption, the exception it must raise, and the phrase the message must
carry. :func:`test_control_unmutated_passes` is its control — if the UNMUTATED
sidecar did not load and validate cleanly, every "raises" below would pass for
the wrong reason and the table would be measuring nothing.
"""
import gzip
import json
import pathlib

import pytest

from tanitad.data.v7_labels import (COT_ABSENCE_POLICY_ID, COT_ABSENCE_RULING,
                                    COT_SIDECAR_DIGEST_ALGO,
                                    COT_SIDECAR_SCHEMA, IGNORE_W,
                                    TAC_GOAL_TOKENS, CotAbsenceNegativeRefused,
                                    TacGoalEmitter,
                                    assert_sidecar_matches_presence,
                                    clip_sha12, cot_backed_tokens,
                                    goal_pos_weight, goal_supervision_census,
                                    load_cot_negative_sidecar, load_v7_labels,
                                    tactical_goal_targets)

COT = "vlm-cot"
GEO = "geometry"


def _rec(clip_id, goals):
    """One record. ``goals`` maps token -> provenance."""
    return {"clip_id": clip_id, "schema_version": "s2-geom-v7", "vocab": "v7",
            "a_tac": {"lat": "LANE_KEEP", "lon": "CRUISE"},
            "a_str": {"token": "HOLD_MAIN_ROAD"},
            "g_str": {"token": "FOLLOW_ROUTE"},
            "g_tac": {"anchor": {"goal_x_m": 50.0, "goal_y_m": 0.0,
                                 "t_reach_s": 6.0},
                      "goals": {t: {"provenance": p, "disputed": False}
                                for t, p in goals.items()}},
            "bands": {"operative_s": [0, 2], "tactical_s": [2, 6],
                      "strategic_s": [8, 30], "unassigned_manoeuvres": []},
            "t0_s": 8.0,
            "horizon": {"available_s": 30.0, "recording_span_s": 30.0},
            "nav_command": {"token": "NAV_FOLLOW_ROAD",
                            "provenance": "ego-future", "oracle": True,
                            "args": {"distance_m": 10.0, "time_s": 1.0}},
            "alpamayo": {"lateral": {"agree": True},
                         "longitudinal": {"agree": True}}}


#: 6 clips. ``YIELD`` and ``TRAFFIC_LIGHT_REACT_RED`` are CoT-backed (so the
#: policy decides them); ``FOLLOW_LANE`` is geometry (so it must be left alone).
ROWS = [
    _rec("clip-a", {"FOLLOW_LANE": GEO, "YIELD": COT}),
    _rec("clip-b", {"FOLLOW_LANE": GEO}),
    _rec("clip-c", {"FOLLOW_LANE": GEO, "TRAFFIC_LIGHT_REACT_RED": COT}),
    _rec("clip-d", {"FOLLOW_LANE": GEO}),
    _rec("clip-e", {"FOLLOW_LANE": GEO, "YIELD": COT,
                    "TRAFFIC_LIGHT_REACT_RED": COT}),
    _rec("clip-f", {"STOP_POINT": GEO}),
]


def _write_blob(tmp_path, rows=None):
    p = tmp_path / "labels.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in (rows or ROWS):
            fh.write(json.dumps(r) + "\n")
    return p


def _sidecar_doc(labels, man, tokens=None):
    toks = tuple(tokens if tokens is not None
                 else [t for t in TAC_GOAL_TOKENS if t in cot_backed_tokens()])
    return {"schema": COT_SIDECAR_SCHEMA,
            "meta": {"policy": COT_ABSENCE_POLICY_ID,
                     "ruling": COT_ABSENCE_RULING,
                     "ruling_widened": "widened the same day",
                     "ruling_date": "2026-09-16", "precedent": "see module",
                     "source_blob_md5": man.md5,
                     "source_blob_records": man.n_records,
                     "digest_algorithm": COT_SIDECAR_DIGEST_ALGO,
                     "tokens": list(toks),
                     "counts_before": {}, "counts_after": {}},
            "clips": {clip_sha12(lb.clip_id):
                      "".join("1" if t in lb.tac_goals else "0" for t in toks)
                      for lb in labels}}


def _write_sidecar(tmp_path, doc, name="sc.json.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return p


# ---------------------------------------------------------------------------
# THE MUTATION TABLE. name -> (what is corrupted, mutator, exception, phrase)
# ⛔ ``stage`` says WHICH guard is being proven: "load" = the sidecar reader,
# "guard" = assert_sidecar_matches_presence, "target" = the loss-target path.
# ---------------------------------------------------------------------------
def _m_blob_md5(doc):
    doc["meta"]["source_blob_md5"] = "0" * 32


def _m_schema(doc):
    doc["schema"] = "tanitad.cot_absence_negative/99"


def _m_policy(doc):
    doc["meta"]["policy"] = "cot-absence-negative/2026-01-01"


def _m_digest_algo(doc):
    doc["meta"]["digest_algorithm"] = "md5(clip_id)"


def _m_row_width(doc):
    k = sorted(doc["clips"])[0]
    doc["clips"][k] = doc["clips"][k] + "0"


def _m_empty(doc):
    doc["clips"] = {}


def _m_drop_clip(doc):
    del doc["clips"][sorted(doc["clips"])[0]]


def _m_extra_clip(doc):
    doc["clips"]["deadbeef0000"] = "0" * len(doc["meta"]["tokens"])


def _m_flip_positive(doc):
    for k, v in doc["clips"].items():
        if "1" in v:
            i = v.index("1")
            doc["clips"][k] = v[:i] + "0" + v[i + 1:]
            return
    raise AssertionError("fixture carries no positive to flip -- the mutation "
                         "would be a no-op and the test would pass vacuously")


def _m_drop_token(doc):
    toks = doc["meta"]["tokens"]
    i = toks.index("YIELD")
    doc["meta"]["tokens"] = toks[:i] + toks[i + 1:]
    doc["clips"] = {k: v[:i] + v[i + 1:] for k, v in doc["clips"].items()}


def _m_add_stale_token(doc):
    doc["meta"]["tokens"] = list(doc["meta"]["tokens"]) + ["MERGE"]
    doc["clips"] = {k: v + "0" for k, v in doc["clips"].items()}


MUTATIONS = [
    # name                   stage    mutator            exception            phrase
    ("sidecar_built_over_another_blob", "load", _m_blob_md5,
     CotAbsenceNegativeRefused, "was built over blob md5"),
    ("schema_tag_changed", "load", _m_schema,
     CotAbsenceNegativeRefused, "declares schema"),
    ("policy_id_changed", "load", _m_policy,
     CotAbsenceNegativeRefused, "carries policy"),
    ("digest_algorithm_undeclared", "load", _m_digest_algo,
     CotAbsenceNegativeRefused, "digest_algorithm"),
    ("row_not_token_wide", "load", _m_row_width,
     CotAbsenceNegativeRefused, "wide"),
    ("empty_policy", "load", _m_empty,
     CotAbsenceNegativeRefused, "empty policy"),
    ("clip_dropped_from_sidecar", "guard", _m_drop_clip,
     AssertionError, "are NOT decided by the sidecar"),
    ("clip_not_in_blob", "guard", _m_extra_clip,
     AssertionError, "are NOT in the blob"),
    ("positive_bit_flipped", "guard", _m_flip_positive,
     AssertionError, "POSITIVES disagree"),
    ("new_cot_token_left_undecided", "guard", _m_drop_token,
     AssertionError, "absent from the sidecar"),
    ("sidecar_token_no_longer_cot_backed", "guard", _m_add_stale_token,
     AssertionError, "no longer vlm-cot-backed"),
]


@pytest.fixture
def loaded(tmp_path):
    blob = _write_blob(tmp_path)
    labels, man = load_v7_labels(blob)
    return tmp_path, labels, man


def test_control_unmutated_passes(loaded):
    """⭐ THE CONTROL. The table below is a list of things that must FAIL; if
    the clean artifact also failed, every one of them would pass for the wrong
    reason. This reads the known value: clean loads, clean validates."""
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    sc, stamped = load_cot_negative_sidecar(p, man)
    rep = assert_sidecar_matches_presence(labels, sc)
    assert rep["n_clips"] == len(labels)
    assert set(sc.tokens) == set(cot_backed_tokens())
    assert "YIELD" in sc.tokens and "FOLLOW_LANE" not in sc.tokens
    # and the STAMP is on the manifest the run will record
    assert stamped.cot_absence_negative["policy"] == COT_ABSENCE_POLICY_ID
    assert stamped.cot_absence_negative["md5"] == sc.md5
    assert stamped.cot_absence_negative["source_blob_md5"] == man.md5
    assert man.cot_absence_negative is None, "the ORIGINAL manifest must not " \
        "be mutated -- an unstamped run must stay unstamped"


@pytest.mark.parametrize("name,stage,mut,exc,phrase", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_is_caught(loaded, name, stage, mut, exc, phrase):
    tmp_path, labels, man = loaded
    doc = _sidecar_doc(labels, man)
    mut(doc)
    p = _write_sidecar(tmp_path, doc, name=f"{name}.json.gz")
    with pytest.raises(exc) as ei:
        sc, _ = load_cot_negative_sidecar(p, man)
        assert_sidecar_matches_presence(labels, sc)
    assert phrase in str(ei.value), f"{name}: guard fired but on the wrong " \
                                    f"complaint: {ei.value}"


def test_policy_without_sidecar_is_refused(loaded):
    """⛔ The policy has NO default. Thousands of unevidenced negatives must
    never be the quiet behaviour of an import."""
    _, labels, _ = loaded
    with pytest.raises(CotAbsenceNegativeRefused) as ei:
        tactical_goal_targets(labels[0], labels[0].t0_s,
                              negatives="cot-absence-negative")
    assert "needs the SIDECAR" in str(ei.value)


def test_sidecar_without_policy_is_refused(loaded):
    """⛔ And the reverse: a stamped manifest whose loss ignored the sidecar is
    a config that lies about what it trained."""
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    sc, _ = load_cot_negative_sidecar(p, man)
    with pytest.raises(CotAbsenceNegativeRefused) as ei:
        tactical_goal_targets(labels[0], labels[0].t0_s, negatives="measured",
                              sidecar=sc)
    assert "would be IGNORED" in str(ei.value)
    with pytest.raises(CotAbsenceNegativeRefused):
        TacGoalEmitter(labels, {0: "clip-a"}, negatives="measured", sidecar=sc)
    with pytest.raises(CotAbsenceNegativeRefused):
        TacGoalEmitter(labels, {0: "clip-a"}, negatives="cot-absence-negative")


def test_uncovered_clip_is_refused_not_defaulted(loaded):
    """⛔ A clip the sidecar never decided must CRASH, not silently fall back:
    a per-clip fallback leaves half the corpus on the old policy and nothing
    says which half."""
    tmp_path, labels, man = loaded
    doc = _sidecar_doc(labels, man)
    del doc["clips"][clip_sha12("clip-a")]
    p = _write_sidecar(tmp_path, doc, name="uncovered.json.gz")
    sc, _ = load_cot_negative_sidecar(p, man)
    target = [lb for lb in labels if lb.clip_id == "clip-a"][0]
    with pytest.raises(CotAbsenceNegativeRefused) as ei:
        tactical_goal_targets(target, target.t0_s,
                              negatives="cot-absence-negative", sidecar=sc)
    assert "does not decide this clip" in str(ei.value)


def test_ignore_state_disappears_for_cot_tokens(loaded):
    """⭐ THE POLICY'S WHOLE EFFECT, asserted on the weights rather than
    described: every CoT token becomes two-state, and the geometry tokens are
    untouched."""
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    sc, _ = load_cot_negative_sidecar(p, man)

    before = goal_supervision_census(labels)
    after = goal_supervision_census(labels, negatives="cot-absence-negative",
                                    sidecar=sc)
    assert before["YIELD"]["ignored"] > 0, "fixture must EXERCISE the ignore " \
        "state, or 'it went to zero' is not a measurement"
    for tok in sc.tokens:
        assert after[tok]["ignored"] == 0
        assert after[tok]["pos"] + after[tok]["neg"] == len(labels)
        assert after[tok]["pos"] == before[tok]["pos"], \
            "positives must be UNCHANGED -- this policy only decides absence"
    for tok in ("FOLLOW_LANE", "STOP_POINT"):
        assert after[tok] == before[tok] or (
            after[tok]["pos"] == before[tok]["pos"]
            and after[tok]["neg"] == before[tok]["neg"]), \
            "a geometry token must not move"

    i = TAC_GOAL_TOKENS.index("YIELD")
    lb = [x for x in labels if "YIELD" not in x.tac_goals][0]
    _, w_old = tactical_goal_targets(lb, lb.t0_s)
    y_new, w_new = tactical_goal_targets(lb, lb.t0_s,
                                         negatives="cot-absence-negative",
                                         sidecar=sc)
    assert w_old[i] == IGNORE_W and w_new[i] == 1.0 and y_new[i] == 0.0


def test_pos_weight_uses_the_existing_path(loaded):
    """The imbalance is now the live problem, and it rides the EXISTING
    ``goal_pos_weight`` path -- a token with no supervised negatives had
    weight 0.0 and now has a real one."""
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    sc, _ = load_cot_negative_sidecar(p, man)
    i = TAC_GOAL_TOKENS.index("YIELD")
    assert goal_pos_weight(labels)[i] == 0.0
    pw = goal_pos_weight(labels, negatives="cot-absence-negative", sidecar=sc)
    assert pw[i] > 0.0


def test_emitter_provenance_carries_the_stamp(loaded):
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    sc, _ = load_cot_negative_sidecar(p, man)
    em = TacGoalEmitter(labels, {i: lb.clip_id for i, lb in enumerate(labels)},
                        negatives="cot-absence-negative", sidecar=sc)
    prov = em.provenance()
    assert prov["tac_goal_negatives"] == "cot-absence-negative"
    assert prov["cot_absence_negative"]["policy"] == COT_ABSENCE_POLICY_ID
    assert prov["cot_absence_negative"]["source_blob_md5"] == man.md5
    assert "YIELD" in prov["supervised_negative_tokens"]
    # the default emitter must stay unstamped
    assert TacGoalEmitter(labels, {0: "clip-a"}).provenance()[
        "cot_absence_negative"] is None


def test_clip_ids_are_not_in_the_clear(loaded):
    """⛔ The sidecar is a COMMITTED artifact: raw clip ids must not be in it."""
    tmp_path, labels, man = loaded
    p = _write_sidecar(tmp_path, _sidecar_doc(labels, man))
    raw = gzip.open(p, "rt", encoding="utf-8").read()
    for lb in labels:
        assert lb.clip_id not in raw
        assert clip_sha12(lb.clip_id) in raw


# ---------------------------------------------------------------------------
# THE TRAINER SEAM. ⛔ Asserted by EXECUTING the parser and the dataset field,
# never by grepping the source for the flag name: a flag that parses and never
# reaches the loss greps identically to one that does.
# ---------------------------------------------------------------------------
def _trainer():
    import importlib.util
    import os
    import sys
    stack = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(stack, "scripts", "refc_v3_train.py")
    if not os.path.exists(py):
        pytest.skip(f"trainer not present at {py}")
    if stack not in sys.path:
        sys.path.insert(0, stack)
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_cotneg", py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_cotneg"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_trainer_opt_in_is_by_name_and_off_by_default():
    m = _trainer()
    ap = m.build_parser()
    base = ["--arm", "hier", "--size", "base", "--v2-cache", "/x", "--out", "/o"]
    a = ap.parse_args(base)
    assert a.cot_negative_sidecar is None
    assert a.tac_goal_negatives == "measured"
    assert m.V3Dataset.cot_negative_sidecar is None, \
        "the OFF path must stay byte-identical: no sidecar on the class default"
    a2 = ap.parse_args(base + ["--tac-goal-negatives", "cot-absence-negative",
                               "--cot-negative-sidecar", "/x/sc.json.gz"])
    assert a2.tac_goal_negatives == "cot-absence-negative"
    assert a2.cot_negative_sidecar == "/x/sc.json.gz"


def test_trainer_dataset_reaches_the_loss_target(tmp_path):
    """⛔ THE SEAM THROUGH THE REAL ``__getitem__``, not through a spy and not
    through a grep. A flag that parses and never reaches the collated batch
    looks identical from the config; the only proof is the tensor.

    The assertion that matters is the CoT cell: ``TRAFFIC_LIGHT_REACT_RED`` is
    weight 0.0 (ignored) under the default policy on exactly this record --
    ``test_tac_goal_trainer_flag`` asserts that -- and must be weight 1.0 with
    the sidecar in place.
    """
    from tanitad.refs import refc_v3 as v3
    tr = _trainer()
    rows = [_rec("clip_9000", {"FOLLOW_LANE": GEO, "YIELD": COT}),
            _rec("clip_9001", {"FOLLOW_LANE": GEO,
                               "TRAFFIC_LIGHT_REACT_RED": COT})]
    blob = _write_blob(tmp_path, rows)
    labels, man = load_v7_labels(blob)
    sc, _ = load_cot_negative_sidecar(
        _write_sidecar(tmp_path, _sidecar_doc(labels, man)), man)

    cfg = v3.refc_v3_smoke_config(True)
    eps = tr._synth_episodes(2, cfg.core, seed=0)
    for i, e in enumerate(eps):
        e.episode_id = str(9000 + i)
    lab = [x for x in labels if x.clip_id == "clip_9000"][0]
    ix = {t: i for i, t in enumerate(TAC_GOAL_TOKENS)}

    def _item(negatives, sidecar):
        ds = tr.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                          channels=cfg.core.encoder.in_channels)
        # the synthetic clip is short; widen the band so the window is in it
        ds.v7_by_sid = {9000: lab.__class__(**{**lab.__dict__,
                                               "bands": {"tactical_s": [0.0, 60.0]},
                                               "t0_s": 0.3})}
        ds.v7_dt, ds.tac_goal_targets = 0.1, True
        ds.tac_goal_negatives, ds.cot_negative_sidecar = negatives, sidecar
        first = {e_i: i for i, (e_i, _t) in reversed(list(enumerate(ds.index)))}
        return ds[first[0]]

    off = _item("measured", None)
    on = _item("cot-absence-negative", sc)
    assert float(off["tac_goal_w"][ix["TRAFFIC_LIGHT_REACT_RED"]]) == 0.0, \
        "the control is broken: this cell is supposed to be IGNORED by default"
    assert float(on["tac_goal_w"][ix["TRAFFIC_LIGHT_REACT_RED"]]) == 1.0, \
        "the sidecar did not reach the collated batch -- the flag is a config " \
        "entry with no loss behind it"
    assert float(on["tac_goal_y"][ix["TRAFFIC_LIGHT_REACT_RED"]]) == 0.0
    assert float(on["tac_goal_w"][ix["FOLLOW_LANE"]]) == 1.0
    assert float(on["tac_goal_y"][ix["FOLLOW_LANE"]]) == 1.0


# ---------------------------------------------------------------------------
# The committed artifact itself, where the corpus is on the box.
# ---------------------------------------------------------------------------
SIDECAR = pathlib.Path(__file__).resolve().parents[2] / (
    "TanitAD Research Lab/Data Engineering/Research/"
    "2026-09-16-flywheel-negatives/cot_absence_negative_v8.0_train.json.gz")
BLOB = pathlib.Path(
    "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Implementation/"
    "incoming/2026-09-10-v8-speed-max-label-release/raw/"
    "s2_labels_v8.0_train.jsonl.gz")
BLOB_MD5 = "fa89ea55dfce68403eb30300e57852ab"


@pytest.mark.skipif(not (SIDECAR.exists() and BLOB.exists()),
                    reason="blob or sidecar not on this box")
def test_shipped_sidecar_matches_the_shipped_blob():
    labels, man = load_v7_labels(BLOB)
    assert man.md5 == BLOB_MD5 and man.n_records == 4572
    sc, stamped = load_cot_negative_sidecar(SIDECAR, man)
    rep = assert_sidecar_matches_presence(labels, sc)
    assert rep["n_clips"] == 4572 and len(sc.tokens) == 15
    after = goal_supervision_census(labels, negatives="cot-absence-negative",
                                    sidecar=sc)
    for tok in TAC_GOAL_TOKENS:
        assert after[tok]["ignored"] == 0, f"{tok} still has an ignore state"
        assert after[tok]["pos"] + after[tok]["neg"] == 4572
    assert after["LANE_CHANGE_R"]["pos"] == 15
    assert after["LANE_CHANGE_R"]["neg"] == 4557
    assert after["TRAFFIC_LIGHT_REACT"]["pos"] == 18
    assert stamped.cot_absence_negative["source_blob_md5"] == BLOB_MD5
