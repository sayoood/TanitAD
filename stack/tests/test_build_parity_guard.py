"""A corpus may not be BUILT without asking whether it swallows the deployed val.

⛔ WHY THIS FILE EXISTS — it pins the sentence, not the code (RETRACTION_LOG
C112/C113). C113 closed the eval-split direction and ended with an escalation
worded, verbatim:

    "Whoever runs that build must call `parity.filter_train_clips()` first."

**That sentence is the defect.** It is a rule that runs only if the next operator
has read the report — the C108 failure mode (a drift tool that compared the wrong
thing for weeks because the doc that said so was never re-read), pre-registered
instead of discovered. `parity.py` §10/§10b made the question ANSWERABLE; nothing
made it ASKED.

⇒ §10c (`guard_corpus_build`) is the one call the corpus-materialising entry
points make, and this file is what keeps them making it.

MEASURED, and re-derived on every run of this suite (never hard-coded):

* **6 of the 40 canonical val episodes are inside the 4 729-clip Alpamayo record
  set** — 15.0 % of the episode set behind EVERY published open-loop number. The
  6 are obtained here by intersecting the banked Alpamayo id list with the
  COMMITTED per-clip digest oracle, so this suite reproduces C113's headline from
  primary sources rather than quoting it.
* 201 of the same 4 729 are inside `physicalai-train-e438721ae894`.

⚠️ THE POPULATION UNDER TEST IS DERIVED FROM SOURCE, NOT HAND-LISTED (C99/C105
punished hand-listing at two levels). :func:`derive_corpus_writers` walks the AST
of every module under ``stack/`` and asks *"does an artifact-shaped path flow into
a write?"*. A NEW build script therefore lands in the population automatically and
turns this suite RED until it is either gated or classified with a reason.

⚠️ AND THE DERIVATION'S OWN FILTER IS PINNED (C110: "an undercount produced by the
instrument's own filter"). :func:`test_derivation_still_finds_the_known_doors`
fails if a regex edit ever shrinks the population below the doors we know exist —
the failure mode where the guard looks green because the deriver went blind.

⚠️ Missing artifacts ``pytest.fail``, never ``skip``. A skipped leak test is the
absent check that produced C112, wearing a green suite.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
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
#: all 4 729 Alpamayo clip ids — banked evidence, read at run time. This suite
#: adds NO new plaintext copy of any clip id: everything it needs is DERIVED
#: from this file plus the committed digest oracles.
ALPAMAYO_IDS = os.path.join(_PILOT, "alpamayo_clip_ids.txt")


def _lines(path: str, what: str) -> list[str]:
    if not os.path.exists(path):
        pytest.fail(
            f"{what} is MISSING at {path}. It is banked evidence for a LIVE "
            f"build-time leak (RETRACTION_LOG C112/C113); without it this suite "
            f"cannot tell a working gate from an absent one, which is exactly "
            f"the state the retraction is about.")
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def alpamayo_ids() -> list[str]:
    return _lines(ALPAMAYO_IDS, "the Alpamayo clip-id list")


def contaminating_val_ids() -> list[str]:
    """The clips that make the 4 472 build dangerous — DERIVED, never listed.

    This is C113's headline recomputed from primary sources on every run: the
    banked Alpamayo ids ∩ the committed deployed-val digest set."""
    return parity.clips_in_deployed_val(alpamayo_ids())


def clean_ids(n: int = 50) -> list[str]:
    """Alpamayo ids in NEITHER oracle — the positive control.

    Without this, a guard that raised unconditionally would pass every refusal
    test in this file and look strict rather than broken (C107)."""
    bad = set(parity.clips_in_deployed_val(alpamayo_ids())) | \
        set(parity.clips_in_parity_train(alpamayo_ids()))
    return [c for c in alpamayo_ids() if c not in bad][:n]


# --------------------------------------------------------------------------- #
# THE DERIVATION — which modules can materialise a corpus                      #
# --------------------------------------------------------------------------- #
#: filename shapes the pipeline CONSUMES as data. Deliberately concrete: these
#: are the artifacts a downstream stage reads, not "files" in general.
CORPUS_ARTIFACT = re.compile(
    r"\.v2ep\.pt"            # the v2 / w120 compressed episode cache
    r"|ep_\{[^}]*\}\.pt"     # the epcache, f-string form
    r"|ep_%0\d*d\.pt"        # the epcache, %-format form
    r"|/videos/|videos/\{"   # the bridge corpus the label engines eat
    r"|/ego/|ego/\{"
    r"|clips\.json")
#: calls that PUBLISH bytes (``replace``/``rename`` included: the atomic-publish
#: idiom writes to ``.tmp`` and renames, so the rename IS the write).
WRITE_CALLS = {
    "save", "save_episode", "savez", "savez_compressed", "dump", "mimwrite",
    "imwrite", "write_text", "write_bytes", "copyfile", "copy2",
    "upload_folder", "upload_file", "replace", "rename", "open", "to_parquet",
}


def _artifact_path_names(tree: ast.AST) -> set[str]:
    """Locals bound to an artifact-shaped path, two hops.

    Two hops rather than one because the atomic-publish idiom is
    ``f = d / f"ep_{i:05d}.pt"`` then ``tmp = str(f) + ".tmp"`` — a one-hop
    tracker sees the rename target and misses the save."""
    names: set[str] = set()
    for _ in range(2):
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and \
                    n.value is not None:
                refs = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                if CORPUS_ARTIFACT.search(ast.unparse(n.value)) or (refs & names):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def _calls_the_gate(tree: ast.AST) -> bool:
    """A CALL to ``guard_corpus_build`` — not a mention of it.

    ⚠️ This started life as a regex and it was WRONG: two modules matched on the
    word inside a comment explaining why they are exempt. A guard-detector that
    counts prose is how a suite goes green over an ungated build."""
    return any(getattr(n.func, "attr", getattr(n.func, "id", "")) ==
               "guard_corpus_build"
               for n in ast.walk(tree) if isinstance(n, ast.Call))


def derive_corpus_writers(stack_root: str) -> dict[str, dict]:
    """Every module under ``stack/`` that writes a corpus artifact.

    Returns ``{relpath: {"writes": [(lineno, callee)], "gated": bool,
    "mentions": bool}}``.

    ``writes``  — an artifact-shaped path flows into a publishing call. These
                  MUST be gated or classified: they are the ingest doors.
    ``mentions`` — the module merely NAMES an artifact and writes something. The
                  broader census; asserted only as a superset, because a name is
                  not a write.
    """
    out: dict[str, dict] = {}
    for dirpath, dirnames, filenames in os.walk(stack_root):
        dirnames[:] = [d for d in dirnames if d != "tests"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, stack_root).replace(os.sep, "/")
            with open(full, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            mentions = any(
                CORPUS_ARTIFACT.search(ast.unparse(n))
                for n in ast.walk(tree)
                if isinstance(n, (ast.Constant, ast.JoinedStr)))
            if not mentions:
                continue
            artvars = _artifact_path_names(tree)
            writes = []
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                nm = getattr(n.func, "attr", getattr(n.func, "id", ""))
                if nm not in WRITE_CALLS:
                    continue
                if nm == "open":
                    mode = "".join(a.value for a in n.args[1:]
                                   if isinstance(a, ast.Constant)
                                   and isinstance(a.value, str))
                    if not ("w" in mode or "a" in mode):
                        continue
                for a in n.args:
                    refs = {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
                    if CORPUS_ARTIFACT.search(ast.unparse(a)) or (refs & artvars):
                        writes.append((n.lineno, nm))
                        break
            out[rel] = {"writes": sorted(set(writes)), "gated": _calls_the_gate(tree),
                        "mentions": True}
    return out


#: A derived writer that is NOT an ingest door, with the reason. The POPULATION
#: is derived; only the DISPOSITION is reviewed — so a new build script cannot be
#: silently absent, it lands in the derivation and fails until it is classified.
#:
#: ⚠️ Two classes, and the distinction is load-bearing:
#:   CONSUMER   — reads an existing corpus, writes a REPORT/aggregate, not a
#:                per-clip corpus. Its parity exposure is on the EVAL side and is
#:                §10's job (`assert_v2_eval_cache`), not §10c's.
#:   CANNOT_GATE — genuinely materialises a corpus, but its ids are in a
#:                DIFFERENT UID SPACE from the oracle, so a gate here would ask
#:                the question in the wrong vocabulary and answer "0 contaminated"
#:                forever. That is worse than no guard (the `df`-reports-the-
#:                cluster trap), so it is refused and NAMED instead.
NOT_AN_INGEST_DOOR = {
    "scripts/emit_situation_labels.py": (
        "CONSUMER: reads an existing cache dir and writes ONE aggregate .npz of "
        "situation labels. No per-clip corpus is produced."),
    "scripts/eval_flagship_v4.py": (
        "CONSUMER: evaluator. Its writes are result JSON. The contamination "
        "question on this side is §10's (assert_v2_eval_cache), not §10c's."),
    "scripts/eval_v58f.py": (
        "CONSUMER: evaluator, as above."),
    "scripts/run_idm_proof.py": (
        "CONSUMER: analysis; writes a proof artifact over an existing corpus."),
    "scripts/epcache_to_pilot.py": (
        "CANNOT_GATE: ids here are positional `ep_%05d`, not PhysicalAI clip "
        "ids (parity.py:101 — 'a different uid space'). Every oracle lookup "
        "would MISS and report a reassuring zero. The upstream writers "
        "(epcache.build_episodes_cached, rebuild_pai_rolling) ARE gated while "
        "the clip ids still exist; the script says so at run time."),
    "tanitad/lake/view.py": (
        "CANNOT_GATE: lake members key on `episode_id` / `shard_key`; the clip "
        "id is not carried. hydrate_cached copies bytes that were ingested "
        "elsewhere, and the lake path structurally refuses PhysicalAI anyway "
        "(gated-confidential -> PermissionError in assemble_lake_record)."),
}

#: The doors we KNOW exist. This is not the population — it is a floor under the
#: DERIVER, so an edit to CORPUS_ARTIFACT / WRITE_CALLS that quietly narrows the
#: search fails loudly instead of returning a shorter, greener list (C110).
KNOWN_DOORS = (
    "scripts/aug120_pipeline.py",
    "scripts/rebuild_pai_rolling.py",
    "scripts/slice_v2_cache.py",
    "scripts/v2_to_pilot.py",
    "tanitad/data/epcache.py",
)


# --------------------------------------------------------------------------- #
# 1. the derivation                                                            #
# --------------------------------------------------------------------------- #
def test_every_derived_corpus_writer_is_gated_or_classified():
    """The whole point: no corpus writer may be silently ungated."""
    derived = derive_corpus_writers(_STACK)
    doors = {k: v for k, v in derived.items() if v["writes"]}
    assert doors, ("the derivation found NO corpus writers at all — the "
                   "instrument is broken, not the code")
    unclassified = sorted(
        k for k, v in doors.items()
        if not v["gated"] and k not in NOT_AN_INGEST_DOOR)
    assert not unclassified, (
        "these modules WRITE a corpus artifact and neither call "
        "parity.guard_corpus_build nor carry a stated reason:\n  "
        + "\n  ".join(f"{k}: writes at {doors[k]['writes'][:3]}"
                      for k in unclassified)
        + "\n\nA corpus that becomes supervision must be checked against the "
          "deployed val BEFORE it is built (parity.py §10c, RETRACTION_LOG "
          "C112/C113). Either call the gate, or add an entry to "
          "NOT_AN_INGEST_DOOR saying why the question cannot or need not be "
          "asked here.")


def test_derivation_still_finds_the_known_doors():
    """C110 pin: the deriver must not go blind and report a clean population."""
    derived = derive_corpus_writers(_STACK)
    missing = [d for d in KNOWN_DOORS if not derived.get(d, {}).get("writes")]
    assert not missing, (
        f"the derivation no longer sees {missing} as corpus writers. Either the "
        f"module changed shape, or CORPUS_ARTIFACT / WRITE_CALLS was narrowed. "
        f"A shorter population is NOT a cleaner codebase — it is an instrument "
        f"whose own filter produced the undercount (RETRACTION_LOG C110).")


def test_classification_map_has_no_stale_entries():
    """A reason for a module that no longer writes a corpus is doctrine rot."""
    derived = derive_corpus_writers(_STACK)
    stale = [k for k in NOT_AN_INGEST_DOOR
             if not derived.get(k, {}).get("writes")]
    assert not stale, (
        f"NOT_AN_INGEST_DOOR still excuses {stale}, which the derivation no "
        f"longer classifies as a corpus writer. Remove the entry — a stale "
        f"exemption is how an ungated door hides in plain sight.")


def test_the_known_doors_are_actually_gated():
    derived = derive_corpus_writers(_STACK)
    ungated = [d for d in KNOWN_DOORS if not derived.get(d, {}).get("gated")]
    assert not ungated, f"these doors lost their gate: {ungated}"


def test_v2_compressed_is_gated_even_though_the_deriver_cannot_see_its_write():
    """⚠️ THE DERIVER'S KNOWN BLIND SPOT, pinned rather than hidden.

    ``v2_compressed.build`` hands the ``<clip>.v2ep.pt`` path to a helper
    (``build_compressed``) which writes it through a parameter, so no
    artifact-shaped path flows into a write call *in one function*. Static
    derivation cannot see across that boundary, so this door is caught by the
    MENTION census, not by the WRITE rule.

    Saying so is the point: a limitation that is written down and pinned is a
    limitation; one that is not is a hole. The behavioural proof that this
    particular door is closed is
    :func:`test_v2_compressed_build_refuses_before_any_download`."""
    derived = derive_corpus_writers(_STACK)
    row = derived.get("scripts/v2_compressed.py")
    assert row is not None, "v2_compressed.py is not even in the census"
    assert row["gated"], "v2_compressed.py no longer calls the ingest gate"


def test_mention_census_is_a_superset_of_the_write_population():
    derived = derive_corpus_writers(_STACK)
    for k, v in derived.items():
        if v["writes"]:
            assert v["mentions"], f"{k} writes an artifact it never names"


# --------------------------------------------------------------------------- #
# 2. the gate refuses — on REAL contaminated ids, with positive controls       #
# --------------------------------------------------------------------------- #
def test_the_six_val_episodes_are_derived_from_the_oracle():
    """C113's headline, recomputed from primary sources rather than quoted."""
    ids = alpamayo_ids()
    assert len(ids) == 4729, f"banked Alpamayo list is {len(ids)}, expected 4729"
    in_val = parity.clips_in_deployed_val(ids)
    in_train = parity.clips_in_parity_train(ids)
    n_val = len(parity.deployed_val_clip_digests())
    assert len(in_val) == 6, (
        f"{len(in_val)} of the deployed val are inside the Alpamayo record set, "
        f"not 6 (RETRACTION_LOG C113). Either the oracle or the banked list "
        f"changed — resolve which before trusting any build.")
    assert len(in_val) / n_val == pytest.approx(0.15)
    assert len(in_train) == 201, f"expected 201 in parity train, got {len(in_train)}"


def test_undeclared_role_refuses_the_alpamayo_corpus():
    """The 4 472 build, as it would actually be launched: no role declared."""
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(alpamayo_ids(), label="alpamayo-4729")
    assert "INGEST GATE REFUSED" in str(e.value)


def test_positive_control_a_clean_corpus_passes():
    """Without this, an unconditionally-raising gate would pass every test above."""
    ids = clean_ids()
    assert len(ids) >= 20, "not enough clean Alpamayo ids to control with"
    kept, rec = parity.guard_corpus_build(ids, label="clean")
    assert kept == sorted(ids)
    assert rec["disjoint"] and rec["in_deployed_val"] == 0
    assert rec["decision_grade"]


def test_a_single_contaminated_clip_is_enough_to_refuse():
    """No threshold. A percentage gate would have waved C112's 4.3 % through."""
    ids = clean_ids() + contaminating_val_ids()[:1]
    with pytest.raises(parity.ParityViolation):
        parity.guard_corpus_build(ids, label="one-bad-clip")


def test_exclude_mode_drops_exactly_the_contaminated_clips():
    ids = clean_ids() + contaminating_val_ids()
    kept, rec = parity.guard_corpus_build(ids, label="filtered", mode="exclude")
    assert rec["n_dropped"] == len(contaminating_val_ids()) == 6
    assert set(kept) == set(clean_ids())
    assert rec["kept"] == len(kept)
    assert not any(c in kept for c in contaminating_val_ids())


def test_val_role_flips_the_check_to_the_train_split():
    """A held-out corpus SHOULD contain val clips; it may not contain train ones."""
    val_only = contaminating_val_ids()
    kept, rec = parity.guard_corpus_build(val_only, label="the-val-split",
                                          role="val")
    assert kept == sorted(val_only), "role='val' must not refuse its own episodes"
    assert rec["in_deployed_val"] == 6, "the count must still be RECORDED"
    train_ids = parity.clips_in_parity_train(alpamayo_ids())[:5]
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(train_ids, label="leaky-eval-split", role="val")
    assert "INGEST GATE REFUSED" in str(e.value)


def test_train_role_does_not_refuse_the_parity_train_corpus():
    """MEASURED: parity train (2 400) ∩ deployed val (40) = 0, which is what
    makes a default-ON gate safe. A guard that fired on the legitimate build
    would be removed within a week."""
    assert not (parity.parity_train_clip_digests() &
                parity.deployed_val_clip_digests())
    train_ids = parity.clips_in_parity_train(alpamayo_ids())
    kept, rec = parity.guard_corpus_build(train_ids, label="parity-train",
                                          role="train")
    assert kept == sorted(train_ids)
    assert rec["in_parity_train"] == 201 and rec["in_deployed_val"] == 0


def test_audit_requires_a_reason_and_voids_decision_grade():
    ids = clean_ids() + contaminating_val_ids()
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(ids, label="census", role="audit")
    assert "sanctioned_audit" in str(e.value)
    kept, rec = parity.guard_corpus_build(
        ids, label="census", role="audit",
        sanctioned_audit="label census over the val episodes")
    assert kept == sorted(ids), "an audit must not silently shrink the set"
    assert rec["decision_grade"] is False
    assert rec["audit_reason"]


def test_a_typo_in_role_or_mode_refuses_rather_than_weakening_the_check():
    for bad in ("Train", "validation", "eval_split", "none"):
        with pytest.raises(parity.ParityViolation):
            parity.guard_corpus_build(clean_ids(), label="x", role=bad)
    with pytest.raises(parity.ParityViolation):
        parity.guard_corpus_build(clean_ids(), label="x", mode="filter")


def test_the_refusal_never_prints_a_clip_id():
    """🔒 gated-confidential. Counts only, like §9/§10."""
    ids = clean_ids() + contaminating_val_ids()
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(ids, label="secrecy")
    msg = str(e.value)
    leaked = [c for c in ids if c in msg]
    assert not leaked, f"{len(leaked)} clip id(s) appeared in the refusal message"


def test_the_gate_refuses_when_its_oracle_is_missing(tmp_path):
    """⚠️ A guard that no-ops without its oracle is C112 wearing a green suite."""
    absent = tmp_path / "not_minted.json"
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(clean_ids(), label="no-oracle",
                                  val_path=str(absent))
    assert "missing" in str(e.value).lower()
    with pytest.raises(parity.ParityViolation) as e2:
        parity.load_clip_digests(str(absent))
    assert "Mint it with" in str(e2.value), (
        "the refusal must tell the operator how to mint the oracle; a guard "
        "that only says 'no' gets deleted")


def test_require_ingest_gate_reports_both_oracles():
    n = parity.require_ingest_gate("test")
    assert n["parity_train_clips"] == 2400 and n["deployed_val_clips"] == 40


def test_a_truncated_oracle_is_refused_not_silently_under_excluding(tmp_path):
    """A short digest set under-excludes SILENTLY — a leak wearing a guard."""
    d = json.loads(open(parity.DEPLOYED_VAL_DIGESTS_PATH, encoding="utf-8").read())
    tampered = copy.deepcopy(d)
    tampered["clip_id_digests"] = tampered["clip_id_digests"][:-1]
    p = tmp_path / "short.json"
    p.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(parity.ParityViolation) as e:
        parity.guard_corpus_build(clean_ids(), label="tampered",
                                  val_path=str(p))
    assert "NOT SELF-CONSISTENT" in str(e.value)


# --------------------------------------------------------------------------- #
# 3. the gate is WIRED — behaviour, not availability                           #
# --------------------------------------------------------------------------- #
def test_v2_compressed_build_refuses_before_any_download(tmp_path):
    """⭐ The one that matters: the 4 472 build's own entry point, refusing.

    ``build()`` is called with a real selection parquet carrying the deployed-val
    clips. It must raise BEFORE it reaches ``_frame_from_args`` — i.e. before a
    single chunk zip is fetched. C112's launch-path defect died AFTER paying for
    a 536 MB download; a gate that trips late is a gate that costs egress."""
    import pandas                                                  # noqa: PLC0415
    try:
        import v2_compressed                                       # noqa: PLC0415
    except Exception as exc:                                       # noqa: BLE001
        pytest.fail(
            f"scripts/v2_compressed.py could not be imported ({exc!r}). This is "
            f"the module the 4 472 build runs through; if it cannot be imported "
            f"here, this suite cannot prove its gate is wired.")
    sel = tmp_path / "sel.parquet"
    ids = clean_ids(10) + contaminating_val_ids()
    pandas.DataFrame({"clip_id": ids, "chunk": list(range(len(ids)))}
                     ).to_parquet(sel)

    class A:                       # the argparse Namespace build() consumes
        pass
    a = A()
    a.sel, a.out = str(sel), str(tmp_path / "out")
    a.only_clips = ""
    a.corpus_role, a.exclude_parity_overlap, a.sanctioned_audit = "", False, ""
    a.root, a.shard, a.quality, a.codec = str(tmp_path / "root"), "", 90, "jpeg"
    a.projection_mode = "ftheta_crop"
    with pytest.raises(parity.ParityViolation) as e:
        v2_compressed.build(a)
    assert "INGEST GATE REFUSED" in str(e.value)
    assert not os.path.exists(a.out), (
        "build() created its output directory before the gate refused — the "
        "gate must run before ANY side effect")


def test_v2_compressed_exposes_the_three_decision_flags():
    """The refusal names three ways out; all three must exist on the CLI, or the
    message sends an operator to a flag that is not there."""
    with open(os.path.join(_STACK, "scripts", "v2_compressed.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    for flag in ("--corpus-role", "--exclude-parity-overlap",
                 "--sanctioned-audit"):
        assert flag in src, f"{flag} is missing from v2_compressed's CLI"


@pytest.mark.parametrize("mod,before", [
    ("scripts/aug120_pipeline.py", "create_repo"),
])
def test_the_gate_runs_before_the_expensive_step(mod, before):
    """⭐ ORDER IS PART OF THE GUARD. aug120_pipeline pulls ~35 MB/clip; a gate
    after the first batch is a gate that costs a batch of egress to trip."""
    with open(os.path.join(_STACK, *mod.split("/")), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    gate = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "guard_corpus_build"]
    spend = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", "")) == before]
    assert gate, f"{mod} does not call guard_corpus_build"
    assert spend, f"{mod} no longer calls {before} — update this test"
    assert min(gate) < min(spend), (
        f"{mod} calls {before} at line {min(spend)} BEFORE the parity gate at "
        f"line {min(gate)}. The gate must precede the spend.")


def test_epcache_gate_is_scoped_to_physicalai_clip_sources():
    """It must fire for clip dicts and stay silent for path/int sources — a gate
    that answered for comma2k19 would be answering in the wrong uid space."""
    from tanitad.data import epcache                                # noqa: PLC0415
    calls = []

    def _boom(_src):
        calls.append(_src)
        raise RuntimeError("should not reach the builder")

    with pytest.raises(parity.ParityViolation):
        epcache.build_episodes_cached(
            [{"clip_id": c} for c in contaminating_val_ids()],
            _boom, "/tmp/nope", "physicalai-train", {"size": 64})
    assert not calls, "the gate must refuse before the first build_one call"


def test_epcache_gate_is_not_applicable_to_non_clip_sources(tmp_path):
    from tanitad.data import epcache                                # noqa: PLC0415
    from tanitad.data.toy_driving import generate_episode           # noqa: PLC0415
    eps = epcache.build_episodes_cached(
        [0, 1], lambda i: generate_episode(int(i), steps=8, size=32),
        tmp_path, "toy", {"steps": 8})
    assert len(eps) == 2, "a non-clip corpus must build unimpeded"
