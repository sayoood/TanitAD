"""An eval split may not contain parity TRAIN clips, and a train corpus may not
contain the DEPLOYED VAL clips — pinned in both directions.

⛔ WHY THIS FILE EXISTS (RETRACTION_LOG C112, MEASURED 2026-08-17/18). The
Alpamayo augmentation corpus was believed disjoint from the parity train split
because it came from a **different source**. Nobody intersected the ids. **201 of
its 4 729 clips are inside ``physicalai-train-e438721ae894``**, and the live v6F
run reads exactly that cache — so an eval split built on the corpus would have
scored the flagship on its own training data, plausibly and wrongly.

⇒ ROOT-CAUSE CLASS: **a non-overlap ASSUMED FROM PROVENANCE rather than COMPUTED
FROM IDS.** The whole point of the code under test is that the question is now
answerable — on any host, without the gated clip list — so the assumption is
never needed again.

⚠️ EVERY TEST HERE MUST BE ABLE TO FAIL (C107: a check that cannot fail is not a
check). Concretely that means:

* the refusal tests feed a **real** contaminated clip id, read at run time out of
  the banked exclusion list — not a synthetic string that would "pass" against an
  empty or broken oracle;
* each refusal test is paired with a **positive control** on a disjoint set, so a
  guard that raised unconditionally would fail here rather than look strict;
* the digest set's self-check is exercised by **tampering with a copy** — a
  self-consistency check nobody ever violates is decoration.

⚠️ Missing artifacts ``pytest.fail``, never ``skip``. A skipped leak test is the
absent check that produced C112, wearing a green suite.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_STACK)
sys.path.insert(0, _STACK)
sys.path.insert(0, os.path.join(_STACK, "scripts"))

from tanitad.data import parity                                  # noqa: E402

_HUB = os.path.join(_REPO, "TanitAD Research Hub")
_PILOT = os.path.join(
    _HUB, "Architecture & Inference", "Implementation", "incoming",
    "2026-08-17-thor-concurrency-pilot")
#: the 201 Alpamayo clips MEASURED to be inside the parity train corpus
EXCLUSION_LIST = os.path.join(
    _PILOT, "alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt")
#: all 4 729 Alpamayo clip ids
ALPAMAYO_IDS = os.path.join(_PILOT, "alpamayo_clip_ids.txt")
#: an ls of the built parity train v2 cache — the source the digest set is minted
#: from, and the reason the mint is reproducible on a box with no pod access
PARITY_LS = os.path.join(_PILOT, "parity_ls.txt")
#: the canonical S2 label artifact (the one corpus consumer that exists today)
S2_LABELS = os.path.join(
    _HUB, "Data Engineering", "Implementation", "incoming",
    "2026-08-16-s2-v1-labels", "review", "labels_v2")
#: the deployed val40, with full clip ids, and its independently banked sha8 twin
VAL40_IDS = os.path.join(
    _HUB, "Architecture & Inference", "Implementation", "incoming",
    "2026-08-18-thor-stranded-rescue", "rescued_beyond_a11", "leadwork",
    "val40_lead_index.json")
VAL40_SHA8 = os.path.join(
    _HUB, "Data Engineering", "Implementation", "incoming",
    "2026-08-04-instrument-durability", "raw", "val40_lead_index_ANON.json")


def _lines(path: str, what: str) -> list[str]:
    if not os.path.exists(path):
        pytest.fail(
            f"{what} is MISSING at {path}. It is banked evidence for a LIVE "
            f"eval leak (RETRACTION_LOG C112); without it this suite cannot "
            f"tell a working exclusion from an absent one, which is exactly "
            f"the state the retraction is about.")
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _json(path: str, what: str) -> dict:
    if not os.path.exists(path):
        pytest.fail(f"{what} is MISSING at {path}.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def contaminated() -> list[str]:
    ids = _lines(EXCLUSION_LIST, "the Alpamayo/parity-train exclusion list")
    assert len(ids) == 201, f"expected 201 banked ids, got {len(ids)}"
    return ids


@pytest.fixture(scope="module")
def alpamayo() -> list[str]:
    ids = _lines(ALPAMAYO_IDS, "the Alpamayo clip-id list")
    assert len(ids) == 4729, f"expected 4729 Alpamayo clip ids, got {len(ids)}"
    return ids


# --------------------------------------------------------------------------- #
# 1. the oracle — committed, self-consistent, and provably the parity split     #
# --------------------------------------------------------------------------- #
def test_the_digest_set_is_committed_and_self_consistent():
    """Nothing below means anything if the oracle is absent or altered."""
    d = parity.load_clip_digests()
    assert d["corpus_key"] == parity.PARITY_TRAIN_KEY
    assert d["n_clips"] == 2400 == len(d["clip_id_digests"])
    assert d["is_full_corpus"] is True
    assert len(parity.parity_train_clip_digests()) == 2400


def test_the_digest_set_reproduces_the_committed_manifest_clip_membership():
    """⭐ THE PROVENANCE CHAIN, end to end and inside the repo.

    ``parity_ls.txt`` (a banked listing of the built parity cache) -> its sorted
    clip-id sha256 == the manifest's ``clip_membership.clip_id_sha256_sorted``
    -> the per-clip digests of those same ids == the committed digest file.

    Without this the digest file is an unverifiable list that DECIDES WHICH
    CLIPS ARE EXCLUDED — the C112 shape one level up: enforcement whose input
    nobody checked."""
    names = _lines(PARITY_LS, "the parity-cache listing")
    ids = sorted({n.split()[-1][: -len(parity.V2_SUFFIX)] for n in names
                  if n.split()[-1].endswith(parity.V2_SUFFIX)})
    assert len(ids) == 2400, f"listing holds {len(ids)} clips, expected 2400"

    cm = parity.clip_membership_of(parity.PARITY_TRAIN_KEY)
    assert cm is not None, "the manifest carries no clip_membership block"
    assert parity.uid_digest(ids) == cm["clip_id_sha256_sorted"], (
        "the banked listing does NOT reproduce the committed corpus digest — "
        "it is not the parity train split, so nothing minted from it is")

    minted = sorted(parity.clip_digest(c) for c in ids)
    assert minted == sorted(parity.load_clip_digests()["clip_id_digests"]), (
        "the committed digest file does not equal the digests of the proven "
        "clip set — re-mint with scripts/make_parity_clip_digests.py")


def test_a_tampered_digest_set_is_refused(tmp_path, contaminated):
    """⚠️ THE SELF-CHECK MUST ACTUALLY FIRE. A short digest file UNDER-excludes
    silently, which is a leak wearing a working guard as a disguise — so drop
    one entry and require a refusal, and require it to be the RIGHT refusal."""
    d = dict(parity.load_clip_digests())
    victim = parity.clip_digest(contaminated[0])
    d["clip_id_digests"] = [x for x in d["clip_id_digests"] if x != victim]
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    parity._CLIP_DIGEST_CACHE.pop(str(p), None)
    with pytest.raises(parity.ParityViolation) as ei:
        parity.load_clip_digests(p)
    msg = str(ei.value)
    assert "digest_of_digests" in msg or "n_clips" in msg
    assert "FILE ALTERED" in msg or "!=" in msg


# --------------------------------------------------------------------------- #
# 2. the refusal — RED without the fix, on a REAL contaminated id               #
# --------------------------------------------------------------------------- #
def test_a_clean_eval_split_passes(alpamayo, contaminated):
    """The positive control. Without it a guard that raised unconditionally
    would pass every other test in this file and look rigorous."""
    clean = [c for c in alpamayo if c not in set(contaminated)][:120]
    rec = parity.assert_eval_clips_disjoint_from_parity_train(
        clean, label="test-clean")
    assert rec["disjoint"] is True
    assert rec["in_parity_train"] == 0
    assert rec["decision_grade"] is True


def test_one_contaminated_clip_refuses_the_whole_eval_split(alpamayo,
                                                            contaminated):
    """⭐ THE TEST THE FIX EXISTS FOR: a 120-clip split that is 119/120 clean is
    still refused. A per-split PASS/FAIL, not a contamination percentage — a
    threshold would have let the 4.3 % Alpamayo case through."""
    split = [c for c in alpamayo if c not in set(contaminated)][:119]
    split.append(contaminated[0])
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_eval_clips_disjoint_from_parity_train(
            split, label="test-poisoned")
    msg = str(ei.value)
    assert "TRAIN-CONTAMINATED EVAL SPLIT" in msg
    assert "LEAK" in msg


def test_the_refusal_never_prints_a_clip_id(alpamayo, contaminated):
    """🔒 Clip ids are gated-confidential PhysicalAI-AV content (parity.py §9).
    A refusal that leaked one would be a confidentiality regression introduced
    BY a safety guard."""
    split = [c for c in alpamayo if c not in set(contaminated)][:5]
    split.append(contaminated[0])
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_eval_clips_disjoint_from_parity_train(
            split, label="test-confidential")
    msg = str(ei.value)
    for cid in split:
        assert cid not in msg, f"the refusal printed a clip id ({cid[:8]}…)"


def test_a_sanctioned_audit_passes_but_is_stamped_not_decision_grade(
        contaminated):
    """Reading train clips IS legitimate for a label audit. What was never
    legitimate is doing it SILENTLY — mirrors ``note_leaky_audit``."""
    rec = parity.assert_eval_clips_disjoint_from_parity_train(
        contaminated[:10], label="test-audit",
        sanctioned_audit="label census over train clips")
    assert rec["disjoint"] is False
    assert rec["decision_grade"] is False
    assert rec["audit_reason"]


def test_filter_removes_exactly_the_contaminated_clips(alpamayo, contaminated):
    kept, dropped, rec = parity.filter_eval_clips(alpamayo, label="test-filter")
    assert dropped == sorted(contaminated)
    assert rec["n_dropped"] == 201
    assert rec["n_kept"] == 4729 - 201 == len(kept)
    parity.assert_eval_clips_disjoint_from_parity_train(kept,
                                                        label="test-filtered")


# --------------------------------------------------------------------------- #
# 3. the exclusion is DERIVED, not a hand-list                                  #
# --------------------------------------------------------------------------- #
def test_the_201_are_derived_from_ids_not_read_from_the_banked_list(alpamayo,
                                                                    contaminated):
    """⛔ C99/C105: a hand-listed set is right until the corpus grows, then
    silently short. The oracle is asked *"is this clip in the parity TRAIN
    split?"*, so it reproduces the banked 201 WITHOUT EVER READING THE BANKED
    LIST — and the same call covers the next 4 472 clips with nothing to
    update."""
    derived = parity.clips_in_parity_train(alpamayo)
    assert derived == sorted(contaminated), (
        f"derived {len(derived)} contaminated clips, banked list has "
        f"{len(contaminated)} — the digest oracle and the banked evidence "
        f"disagree")
    assert len(derived) == 201
    assert abs(len(derived) / len(alpamayo) - 0.042504) < 1e-5


def test_a_clip_outside_both_corpora_is_not_flagged():
    """The oracle must not simply say yes. (Random uuids are not in a 2 400-clip
    set; if this ever fails the digest file is not what it claims.)"""
    strangers = [f"00000000-0000-4000-8000-{i:012d}" for i in range(64)]
    assert parity.clips_in_parity_train(strangers) == []


# --------------------------------------------------------------------------- #
# 4. the LIVE fact — the aug120 label leg is 100 % train, and is used as TRAIN  #
# --------------------------------------------------------------------------- #
def test_the_aug120_label_leg_is_entirely_inside_the_parity_train_split():
    """⛔ THE FACT, in executable form, because prose did not hold it.

    ``label_split`` is a PROVENANCE TAG, not a partition. ``aug120`` reads like
    an independent augmentation corpus and is **201 of 201 inside the corpus the
    flagship trains on**; ``w120val`` is 0 of 600. That is correct as TRAIN
    supervision and catastrophic as a held-out set, and nothing in the name, the
    file or the schema distinguishes the two — which is why it is asserted here
    rather than described somewhere."""
    idx = _json(os.path.join(S2_LABELS, "clip_index.json"),
                "the canonical S2 clip index")
    by: dict[str, list[str]] = {}
    for cid, ent in idx["clips"].items():
        by.setdefault(ent.get("label_split", "?"), []).append(cid)

    assert set(by) == {"aug120", "w120val"}, sorted(by)
    assert len(by["aug120"]) == 201 and len(by["w120val"]) == 600

    aug_in = parity.clips_in_parity_train(by["aug120"])
    assert len(aug_in) == 201, (
        f"the aug120 leg is {len(aug_in)}/201 inside the parity train corpus — "
        f"if this ever drops, the corpus or the digest set moved and the "
        f"train/eval story above must be re-derived, not patched")
    val_in = parity.clips_in_parity_train(by["w120val"])
    assert val_in == [], (
        f"{len(val_in)} w120val clips are in the parity TRAIN corpus — the "
        f"held-out leg is no longer held out")


def test_s2_labels_report_discloses_contamination_into_every_run_config():
    """The disclosure that cannot be forgotten: it rides in ``report()``, which
    ``train_v6_staged`` writes into every run's ``config.json``. A fact nobody
    asked for is exactly the fact that goes unnoticed."""
    pytest.importorskip("torch")
    from s2_labels import load_s2_labels
    rep = load_s2_labels(S2_LABELS).report()
    pc = rep["parity_contamination"]["by_label_split"]
    assert pc["aug120"]["n_in_parity_train"] == 201
    assert pc["aug120"]["usable_as_holdout"] is False
    assert pc["w120val"]["n_in_parity_train"] == 0
    assert pc["w120val"]["usable_as_holdout"] is True


def test_loading_the_labels_as_an_EVAL_set_is_refused():
    """⭐ END TO END THROUGH THE REAL LOADER, ON THE REAL CANONICAL ARTIFACT.
    ``role='train'`` (the incumbent) still loads; ``role='eval'`` refuses and
    names the leg. Both halves are asserted — a loader that refused everything
    would pass the second half alone."""
    pytest.importorskip("torch")
    from s2_labels import load_s2_labels, S2LabelError
    assert len(load_s2_labels(S2_LABELS)) > 0            # role='train' unchanged
    with pytest.raises(S2LabelError) as ei:
        load_s2_labels(S2_LABELS, role="eval")
    msg = str(ei.value)
    assert "aug120" in msg
    assert "not a held-out set" in msg.lower()


# --------------------------------------------------------------------------- #
# 5. the OTHER direction — a train corpus must not swallow the deployed val     #
# --------------------------------------------------------------------------- #
def test_the_deployed_val_digest_set_is_cross_checked_by_a_second_source():
    """A 40-of-600 deployment cannot reproduce the corpus digest, so its proof
    is a SECOND artifact: every episode's independently banked ``clip_sha8``
    equals ``sha256(clip_id)[:8]``. Re-derived here so the weaker proof is
    verified rather than trusted."""
    full = _json(VAL40_IDS, "the val40 lead index (with clip ids)")
    anon = _json(VAL40_SHA8, "the val40 ANON index (with clip_sha8)")
    assert len(full) == len(anon) == 40
    for ep, ent in full.items():
        want = anon[ep]["clip_sha8"]
        got = "clip_" + hashlib.sha256(
            ent["clip_id"].encode("utf-8")).hexdigest()[:8]
        assert got == want, f"{ep}: the two banked artifacts disagree"

    digs = parity.deployed_val_clip_digests()
    assert len(digs) == 40
    assert all(parity.clip_digest(e["clip_id"]) in digs
               for e in full.values())


def test_the_alpamayo_corpus_swallows_6_of_the_40_deployed_val_episodes(
        alpamayo):
    """⭐ MEASURED 2026-08-18, and the more dangerous direction: **6 of the 40
    canonical val episodes (15.0 %) are inside the Alpamayo record set.**

    Blast radius TODAY is zero — no trainer consumes those labels. The trigger
    is the 4 472-clip build: the moment that corpus becomes supervision, 15 % of
    the episode set behind EVERY published open-loop number is in training, and
    no pre-existing guard would notice (§9 checks a cache against its OWN corpus
    digest, and an augmentation corpus is a different corpus by construction)."""
    overlap = parity.clips_in_deployed_val(alpamayo)
    assert len(overlap) == 6, (
        f"{len(overlap)} of the 40 deployed val episodes are in the Alpamayo "
        f"corpus, expected 6 — the corpus or the deployment moved; re-derive "
        f"before changing this number")
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_train_clips_disjoint_from_deployed_val(
            alpamayo, label="test-alpamayo-as-supervision")
    assert "SWALLOWS THE DEPLOYED VAL" in str(ei.value)
    assert "15.0 %" in str(ei.value)


def test_a_train_corpus_with_no_val_clips_passes(alpamayo):
    """Positive control for 10b, and the repair path in one: filtered, the same
    corpus is admissible as supervision."""
    kept, dropped, rec = parity.filter_train_clips(alpamayo, label="test-fix")
    assert len(dropped) == 6 and rec["n_kept"] == 4723
    ok = parity.assert_train_clips_disjoint_from_deployed_val(
        kept, label="test-fixed")
    assert ok["disjoint"] is True


# --------------------------------------------------------------------------- #
# 6. the mint refuses a source that is not the corpus                           #
# --------------------------------------------------------------------------- #
def test_the_mint_refuses_a_clip_set_that_is_not_the_parity_split(alpamayo):
    """THE GENERATION IS THE PROOF (the ``register_v2_geometry_sibling``
    contract). A digest set minted from the wrong clips would authorise the
    wrong exclusions — the failure this whole file defends against, inverted —
    so the generator must refuse before it writes."""
    from make_parity_clip_digests import build
    with pytest.raises(parity.ParityViolation) as ei:
        build(alpamayo, corpus_key=parity.PARITY_TRAIN_KEY,
              source="the Alpamayo corpus (deliberately wrong)")
    assert "REFUSING TO MINT" in str(ei.value)
    assert "MISMATCH" in str(ei.value)
