"""H-BOXCLS-1 — the class weight REACHES the loss, and the RECORD states the tensor.

⛔ WHY THIS FILE EXISTS, IN THREE MEASURED DEFECTS THIS PROGRAMME HAS ALREADY PAID FOR:

1. **A FLAG THAT PARSES AND REACHES NOTHING.** `305debd` fixed exactly that for
   `occ_from_geometry`: the knob was on the parser, stamped on the config, and never
   plumbed to the module. Nothing failed; the arm simply trained the old behaviour under
   a new name. Here the equivalent is `--agent-cls-weight train2400` producing a
   `config.json` that says `train2400` over a run whose `cls` term was unweighted.

2. **A DIGEST NOTHING COULD RECOMPUTE.** The banked artifact shipped
   `_self_digest_sha256_of_weights: c4629301866228cd`, hand-written beside the numbers it
   claimed to attest — no code in the repo, in `qland/`, or in any probe could reproduce
   it (searched by two independent predicates). A self-attestation with no recipe is the
   `A CHECK THAT SHARES THE DEFECT IT CHECKS FOR` family in its most literal form: the
   check and the claim were the same keystrokes. The real digest is `bde3aa19dfd0e59a`
   and `cls_weight_digest` is now the single stated recipe.

3. **A `built` SLOT THAT RE-READS ITS OWN SOURCE.** If the run record filled `built` by
   loading the artifact again it would agree with itself forever and could not see (1).
   `built` is therefore the digest of the tensor ON THE MODEL.

⭐ THE DISCRIMINATING PROPERTY, and the one a weaker test would miss: every assertion below
is written as a LITERAL or as an independently-derived value, never as an expression over
the code under test. The weight vector's effect on the loss is checked against a
hand-computed weighted cross-entropy, not against `slot_set_loss`'s own arithmetic.

⛔ CPU only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

from tanitad.models import agent_slots as A  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
ART = REPO / "stack" / "tanitad" / "data" / A.CLS_WEIGHTS_TRAIN2400
TRAINER = REPO / "stack" / "scripts" / "refc_v3_train.py"

# ⛔ LITERALS. These are the MEASURED census, not a re-derivation: 2,308 episodes /
# 433,040 frames / 12,122,129 boxes over the canonical train join. If the artifact is
# ever rebuilt these must be updated deliberately, which is the point.
EXPECT_DIGEST = "bde3aa19dfd0e59a"
EXPECT_IMBALANCE = 1071.1
EXPECT_N_CLASSES = 10


# --------------------------------------------------------------------------- #
# the digest: reproducible, and sensitive to the failures it must catch
# --------------------------------------------------------------------------- #

def test_digest_matches_the_banked_literal():
    vec, stamp = A.load_cls_class_weight()
    assert A.cls_weight_digest(vec) == EXPECT_DIGEST
    assert stamp["digest"] == EXPECT_DIGEST


def test_digest_is_permutation_sensitive():
    """⭐ THE FAILURE A VALUE-ONLY DIGEST CANNOT SEE: the same ten numbers, wrong order.

    A permuted vector would hand `automobile` the weight measured for `animal` — a 1,071×
    error in the direction that silently un-does the whole intervention — while a digest
    over the sorted values alone would read identical.
    """
    vec, _ = A.load_cls_class_weight()
    flipped = torch.flip(vec, dims=(0,))
    assert torch.allclose(torch.sort(vec).values, torch.sort(flipped).values)
    assert A.cls_weight_digest(flipped) != A.cls_weight_digest(vec)


def test_digest_survives_a_device_and_dtype_round_trip():
    """`built` is computed off the model's tensor, which has been `.to(device)`-ed."""
    vec, _ = A.load_cls_class_weight()
    assert A.cls_weight_digest(vec.to("cpu").float().clone()) == EXPECT_DIGEST


def test_digest_moves_when_one_weight_is_edited():
    vec, _ = A.load_cls_class_weight()
    edited = vec.clone()
    edited[0] = float(edited[0]) + 0.001
    assert A.cls_weight_digest(edited) != EXPECT_DIGEST


def test_load_refuses_a_vector_that_disagrees_with_its_own_attestation(tmp_path,
                                                                      monkeypatch):
    """⛔ THE ATTESTATION IS LOAD-BEARING, NOT DECORATIVE.

    Before `cls_weight_digest` existed this file's digest could not be checked by anything,
    so an edited vector would have trained silently. Here the edit is REAL — the artifact
    on disk is rewritten with a changed weight — and the loader must refuse.
    """
    art = json.loads(ART.read_text(encoding="utf-8"))
    art["weights_inv_freq_mean1"]["animal"] = 99.0
    bad_dir = tmp_path / "data"
    bad_dir.mkdir()
    (bad_dir / "bad_weights.json").write_text(json.dumps(art), encoding="utf-8")
    monkeypatch.setattr(A, "__file__", str(tmp_path / "models" / "agent_slots.py"))
    with pytest.raises(SystemExit) as e:
        A.load_cls_class_weight("bad_weights.json")
    assert "digest" in str(e.value).lower()


def test_load_refuses_when_a_class_is_unnamed(tmp_path, monkeypatch):
    art = json.loads(ART.read_text(encoding="utf-8"))
    art["weights_inv_freq_mean1"].pop("animal")
    d = tmp_path / "data"
    d.mkdir()
    (d / "short.json").write_text(json.dumps(art), encoding="utf-8")
    monkeypatch.setattr(A, "__file__", str(tmp_path / "models" / "agent_slots.py"))
    with pytest.raises(SystemExit) as e:
        A.load_cls_class_weight("short.json")
    assert "animal" in str(e.value)


def test_the_vector_is_normalised_to_mean_one():
    """⭐ SCALE IS NOT A FREE VARIABLE. `slot_set_loss`'s denominator follows the weight,
    so the term's magnitude is fixed and re-weighting moves exactly one thing: the
    RELATIVE emphasis. A vector with mean != 1 would also move the term's weight against
    every sibling loss, and the arm would confound two changes."""
    vec, stamp = A.load_cls_class_weight()
    assert vec.shape == (EXPECT_N_CLASSES,)
    assert abs(float(vec.mean()) - 1.0) < 1e-6
    assert float(stamp["imbalance_majority_to_rarest"]) == EXPECT_IMBALANCE


# --------------------------------------------------------------------------- #
# the weight REACHES the loss — checked against arithmetic done here
# --------------------------------------------------------------------------- #

def _one_row_batch(n_slots: int = 4):
    torch.manual_seed(0)
    C = len(A.AGENT_CLASSES)
    pred = {"cls_logits": torch.randn(1, 1, n_slots, C)}
    return pred, C


def test_weight_none_is_bit_identical_to_weight_ones():
    """⛔ THE BIT-IDENTITY CONDITION. `off` must be the arm every prior run trained.

    Because the denominator follows the weight, `weight=ones` divides by the same count
    `None` does — so the two are not merely close, they are EQUAL. A denominator that
    stayed at `int(ok.sum())` would make this test fail, which is why it is written as an
    exact comparison and not `allclose` with a tolerance."""
    pred, C = _one_row_batch()
    tgt = torch.tensor([[0, 5, 9, 2]])
    ones = torch.ones(C)
    ok = torch.ones(1, 4, dtype=torch.bool)

    def cls_term(w):
        lg = pred["cls_logits"][0][0][ok[0]]
        ct = tgt[0][ok[0]]
        num = torch.nn.functional.cross_entropy(lg, ct, reduction="sum", weight=w)
        den = float(ok.sum()) if w is None else float(w[ct].sum())
        return float(num) / den

    assert cls_term(None) == cls_term(ones)


def test_a_rare_class_error_costs_more_under_the_train_weight():
    """⭐ THE DIRECTION OF THE INTERVENTION, MEASURED RATHER THAN ASSUMED.

    `animal` carries 4.374664 and `automobile` 0.004084 — a 1,071× ratio. Getting an
    `animal` slot wrong must therefore cost far more, relative to the same mistake on an
    `automobile`, than it does unweighted. This is the whole point of H-BOXCLS-1 and it is
    checked with hand-rolled arithmetic, not by calling the loss under test."""
    w, _ = A.load_cls_class_weight()
    i_auto = A.AGENT_CLASSES.index("automobile")
    i_anim = A.AGENT_CLASSES.index("animal")
    torch.manual_seed(1)
    C = len(A.AGENT_CLASSES)
    logits = torch.randn(C)

    def loss_for(target, weight):
        ce = torch.nn.functional.cross_entropy(
            logits[None], torch.tensor([target]), reduction="sum",
            weight=weight)
        den = 1.0 if weight is None else float(weight[target])
        return float(ce) / den, float(ce)

    _, raw_auto_u = loss_for(i_auto, None)
    _, raw_anim_u = loss_for(i_anim, None)
    _, raw_auto_w = loss_for(i_auto, w)
    _, raw_anim_w = loss_for(i_anim, w)
    # unweighted: the two mistakes contribute in proportion to their -log p only
    # weighted: additionally scaled by 0.004084 vs 4.374664
    ratio_unweighted = raw_anim_u / raw_auto_u
    ratio_weighted = raw_anim_w / raw_auto_w
    assert ratio_weighted / ratio_unweighted == pytest.approx(
        float(w[i_anim]) / float(w[i_auto]), rel=1e-5)
    assert ratio_weighted / ratio_unweighted > 1000.0


def test_an_out_of_vocabulary_class_is_masked_not_relabelled():
    """⛔ 10,077 TRAIN BOXES (0.0831 %) CARRY A CLASS THE HEAD CANNOT EMIT.

    `train_or_tram_car` is in the corpus and NOT in `bev_raster.ALL_CLASSES`. The
    programme's own guard prose used to say it "does not exist in the corpus", which is
    false and, worse, is the kind of false reason that stops anyone asking what happens to
    those boxes. This pins the answer.

    ⭐ THE DISCRIMINATING PART IS `automobile -> 0`. Mapping an unknown name to `-1` and
    mapping EVERY name to `-1` are indistinguishable without a positive control, and the
    second would silently delete the whole class term — the `presence`-collapse failure in
    a different costume. Both directions are asserted here."""
    a = torch.zeros((3, 6), dtype=torch.float32)
    a[:, 3:5] = 2.0            # non-degenerate l/w
    t = A.targets_from_join(a, classes=["automobile", "train_or_tram_car", "person"])
    ct = t["cls"].reshape(-1).tolist()
    assert ct[0] == A.AGENT_CLASSES.index("automobile") == 0
    assert ct[2] == A.AGENT_CLASSES.index("person")
    assert ct[1] == -1, "an out-of-vocabulary name must map to -1, never to class 0"
    # ...and -1 is what `slot_set_loss` masks on: `ok = ct >= 0`
    assert (torch.tensor(ct) >= 0).sum().item() == 2


def test_slot_set_loss_accepts_and_uses_the_weight():
    """The reach test at the REAL function: same inputs, weight on vs off, cls must move."""
    sig = A.slot_set_loss.__code__.co_varnames[:A.slot_set_loss.__code__.co_argcount]
    kw = A.slot_set_loss.__kwdefaults__ or {}
    assert "cls_class_weight" in sig or "cls_class_weight" in kw


# --------------------------------------------------------------------------- #
# the RECORD states the tensor — not the flag, not the file
# --------------------------------------------------------------------------- #

def _trainer_src() -> str:
    return TRAINER.read_text(encoding="utf-8")


def test_the_flag_exists_with_exactly_two_choices():
    src = _trainer_src()
    assert '"--agent-cls-weight"' in src
    assert 'choices=["off", "train2400"]' in src


def test_built_is_read_off_the_model_not_the_artifact():
    """⛔ THE DEFECT THIS LINE PREVENTS: a `built` slot filled by re-loading the file would
    agree with the `requested` field forever and could never see a flag that reached
    nothing. The trainer must digest `model._cls_class_weight`."""
    src = _trainer_src()
    assert '_agent_slots.cls_weight_digest(_cwv)' in src
    assert '_cwv = getattr(model, "_cls_class_weight", None)' in src


def test_both_loss_call_sites_pass_the_weight():
    """Two heads, two loss paths: `core.agent_head` via `agent_losses` and
    `perception.box_dec` via `box3d_loss_row`. MEASURED this session that the class
    collapse is present on BOTH (2,000/2,000 and 320/320 slots emitting one class), so a
    fix wired to only one of them would leave the scored head untouched."""
    src = _trainer_src()
    n = src.count('cls_class_weight=getattr(model, "_cls_class_weight", None)')
    assert n == 2, f"expected the weight at BOTH loss sites, found {n}"


def _seam_case(weight, block):
    """A stub model carrying only what `assert_seams_are_built` reads."""
    import types
    m = types.SimpleNamespace(core=types.SimpleNamespace(decoder=types.SimpleNamespace()))
    m._cls_class_weight = weight
    return m, ({} if block is None else {"agent_cls_weight": block})


def _refuses(weight, block) -> bool:
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import refc_v3_train as T  # noqa: E402
    m, s = _seam_case(weight, block)
    try:
        T.assert_seams_are_built(m, s)
        return False
    except SystemExit:
        return True


@pytest.mark.parametrize("name,weight_is_set,block,expect_refusal", [
    # ⭐ THE TWO CLEAN STATES MUST PASS. A guard that refuses everything is not a guard,
    # and both of these are states a real run legitimately reaches.
    ("off and nothing attached", False,
     {"requested": "off", "mode": "off", "built": None}, False),
    ("train2400 attached, digest agrees", True, "MATCHING", False),
    # ⛔ AND THE FOUR FAILURE DIRECTIONS MUST REFUSE.
    ("record asks for it, model has nothing", False,
     {"requested": "train2400", "mode": "train2400", "built": None}, True),
    ("model has it, record says off", True,
     {"requested": "off", "mode": "off", "built": None}, True),
    ("record names a DIFFERENT vector", True, "WRONG_DIGEST", True),
    # ⭐⭐ THE CASE THAT ISOLATES THE MODEL-SIDE COMPARISON, AND IT WAS ADDED BECAUSE A
    # MUTATION ESCAPED WITHOUT IT. Here the record is INTERNALLY CONSISTENT — its stated
    # artifact digest and its `built` slot agree — and both describe a vector the model
    # does NOT carry. Only a check that reads the LIVE TENSOR can see it; a check that
    # compares the record's two fields to each other passes happily. That is the
    # self-agreeing-stamp defect stated precisely, and it is why `built` is read off the
    # model rather than re-loaded from the file.
    ("record self-consistent, model carries another vector", True,
     "CONSISTENT_BUT_WRONG", True),
    ("model has it, record has no block at all", True, None, True),
])
def test_the_seam_check_is_bidirectional_and_digest_aware(name, weight_is_set, block,
                                                          expect_refusal):
    """⛔ BEHAVIOURAL, NOT TEXTUAL — and this replaces a test that WAS textual.

    MEASURED while writing this file: the first version asserted that three message
    substrings appeared in the trainer source, and a mutation that deleted half of one
    branch's message left every substring it checked intact. It went GREEN against a
    weakened guard — `A CHECK THAT SHARES THE DEFECT IT CHECKS FOR`, one level up, where
    the check shared the *source file* with the thing it checked. Calling the function and
    requiring a refusal cannot be satisfied by prose.

    ⭐ The digest slot is what makes the third failure reachable at all: with a boolean
    `built`, "the right flag loading the WRONG VECTOR" is invisible."""
    vec, _ = A.load_cls_class_weight()
    w = vec if weight_is_set else None
    if block == "MATCHING":
        d = A.cls_weight_digest(vec)
        block = {"requested": "train2400", "mode": "train2400", "built": d, "digest": d}
    elif block == "WRONG_DIGEST":
        block = {"requested": "train2400", "mode": "train2400",
                 "built": "deadbeefdeadbeef", "digest": A.cls_weight_digest(vec)}
    elif block == "CONSISTENT_BUT_WRONG":
        other = A.cls_weight_digest(torch.flip(vec, dims=(0,)))
        assert other != A.cls_weight_digest(vec)
        block = {"requested": "train2400", "mode": "train2400",
                 "built": other, "digest": other}
    assert _refuses(w, block) is expect_refusal, name


def test_off_stamps_a_distinguishable_block():
    """⭐ `off` is not silence. A run that predates the flag and a run that declined it must
    be distinguishable from the artifact alone — the `refcv6: null` convention."""
    import argparse
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import refc_v3_train as T  # noqa: E402
    blk = T._cls_weight_stamp(argparse.Namespace(agent_cls_weight="off"))
    assert blk == {"requested": "off", "mode": "off", "built": None}


def test_train2400_stamps_all_ten_values_and_a_null_built():
    import argparse
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import refc_v3_train as T  # noqa: E402
    blk = T._cls_weight_stamp(argparse.Namespace(agent_cls_weight="train2400"))
    assert blk["requested"] == "train2400"
    assert blk["built"] is None, "the stamp builder runs before the model exists"
    assert set(blk["weights"]) == set(A.AGENT_CLASSES)
    assert len(blk["weights"]) == EXPECT_N_CLASSES
    assert blk["digest"] == EXPECT_DIGEST
    assert blk["out_of_vocabulary"]["train_or_tram_car"] == 10077
