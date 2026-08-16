"""Pin the PhysicalAI-AV feature read-set to SOURCE, because the prose number rotted.

WHY THIS TEST EXISTS
====================
``CLAUDE.md`` carries the sentence *"our ingest reads N of 36 features"*. **That N has
now gone stale FOUR times in the same sentence:**

    "2 of 36"  ->  4  (2026-07-26)  ->  5  (2026-08-16)  ->  6  (2026-08-16, this test)

It is a stale absence-claim living inside the rule that warns about stale
absence-claims. The root cause is not carelessness: it is that **a count lived only in
prose, with nothing pinning it to source**, so every time the ingest grew a feature the
number silently became a lie -- and by 2026-08-16 the stale "4" had propagated into
**17 documents** plus a code docstring.

The fourth rot has the same trigger as the 2026-08-03 stale-blocker case swept the same
day: ``obstacle.offline`` landed as a real read (the distance-keeping instrument), and
neither the prose count nor the "UNAVAILABLE" blocker lines were revisited.

THE ANSWER, AND WHY IT IS THREE NUMBERS AND NOT ONE
--------------------------------------------------
The subject "our ingest" was never defined, which is *why* it kept rotting. There are
three distinct read-sets and they legitimately differ:

===========================  =====  ========================================================
layer                        count  features
===========================  =====  ========================================================
``physicalai_r0.py``           2    egomotion, camera_front_wide_120fov
(r0 clip selection)
``physicalai.py``              5    the above + camera_intrinsics, sensor_extrinsics,
(the episode build)                 vehicle_dimensions
program-wide                   6    the above + obstacle.offline (pod-side side-car join;
(incl. the side-car join)           NOT part of the episode build -- see below)
===========================  =====  ========================================================

``obstacle.offline`` is deliberately **outside** the episode build: the join is a
pod-side step (``stack/scripts/build_obstacle_join.py``), which is exactly why
``grep obstacle stack/tanitad/data/physicalai.py`` still returns zero matches. A doc
that says "the episode build reads 5" is correct; a doc that says "our ingest reads 5"
is **wrong as of 2026-08-03** and must say 6 or name the layer.

Evidence class: **MEASURED** (from source, this repo, 2026-08-16) --
``stack/tanitad/data/physicalai.py:232-235`` (the calibration/camera constants, each
consumed at :354, :407, :456), :471-472 (egomotion), and
``stack/scripts/physicalai_r0.py:36-38`` (the r0 templates).

WHAT THIS TEST GUARANTEES
-------------------------
1. The three counts above, **and the exact feature names**, are asserted against the
   live source -- so growing the read-set breaks the build instead of rotting a doc.
2. A **drift detector** (:func:`test_no_undeclared_feature_path_in_episode_build`)
   fails when a *new* HF feature path appears in ``physicalai.py`` that this file does
   not declare. That is the case the previous three rots all were.
3. Every failure message names **the documents to update**, so the fix is mechanical
   rather than a re-derivation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# The documents that carry the count. A failure message that does not tell the  #
# reader WHERE to fix the prose is how this rotted three times before.          #
# --------------------------------------------------------------------------- #
DOCS_CARRYING_THE_COUNT = (
    "CLAUDE.md  (the 'absence found at ONE location' rule -- the canonical statement)",
    "Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md",
    "Project Steering/V6F_PLANNER_DESIGN.md",
    "Project Steering/Gates/flagship-v5-retrain.PREP.md  ('32-of-36', the complement)",
    "stack/tanitad/data/bev_raster.py  (module docstring -- the count also rotted INTO CODE)",
    "TanitAD Research Hub/**/incoming/**  (~11 further dated write-ups; history, "
    "lower blast radius -- fix the four above first)",
)

#: Total features published by the PhysicalAI-AV dataset. PUBLISHED (dataset card);
#: the denominator of every "N of 36" statement in the programme.
PHYSICALAI_TOTAL_FEATURES = 36

#: r0 clip selection -- ``stack/scripts/physicalai_r0.py:36-38``.
R0_SELECTION_FEATURES = frozenset({"egomotion", "camera_front_wide_120fov"})

#: The episode build -- ``stack/tanitad/data/physicalai.py``. The three calibration
#: features are consumed at :354 (intrinsics), :407 (wheelbase), :456 (extrinsics).
EPISODE_BUILD_FEATURES = R0_SELECTION_FEATURES | {
    "camera_intrinsics",
    "sensor_extrinsics",
    "vehicle_dimensions",
}

#: Program-wide, including the pod-side side-car join that feeds the LONGITUDINAL
#: distance-keeping family (landed 2026-08-03).
OBSTACLE_FEATURE = "obstacle.offline"
PROGRAM_WIDE_FEATURES = EPISODE_BUILD_FEATURES | {OBSTACLE_FEATURE}

#: Non-empty read sites for ``obstacle.offline``. Absence at ONE location is not
#: absence, so the assertion below requires the feature in **at least two** of these.
OBSTACLE_READ_SITES = (
    "scripts/build_obstacle_join.py",
    "scripts/lead_state_gate.py",
    "tanitad/data/bev_raster.py",
)

_STACK = Path(__file__).resolve().parents[1]
_REPO = _STACK.parent


def _fix_the_docs(what: str) -> str:
    docs = "\n".join(f"      - {d}" for d in DOCS_CARRYING_THE_COUNT)
    return (
        f"\n\n  {what}\n\n"
        f"  This count is quoted in prose across the programme. It has already gone\n"
        f"  stale FOUR times ('2' -> 4 -> 5 -> 6). If you changed the read-set on\n"
        f"  purpose, update BOTH this test and the documents below in the same commit:\n"
        f"{docs}\n\n"
        f"  State the LAYER with the number ('the episode build reads N'), never the\n"
        f"  bare phrase 'our ingest' -- the undefined subject is what let it rot.\n"
    )


def _source(rel: str) -> str:
    p = _STACK / rel
    assert p.is_file(), f"expected source file missing: {p}"
    return p.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# 1. The counts and the exact names.                                            #
# --------------------------------------------------------------------------- #
def test_episode_build_reads_exactly_five_named_features() -> None:
    """``physicalai.py`` declares -- and consumes -- exactly the 5 named features."""
    from tanitad.data import physicalai as pai

    declared = {
        pai._FRONT_WIDE_CAM,
        pai._CALIB_INTR,
        pai._CALIB_EXTR,
        pai._CALIB_VEH,
        "egomotion",
    }
    assert declared == EPISODE_BUILD_FEATURES, _fix_the_docs(
        f"physicalai.py's feature constants changed.\n"
        f"  expected: {sorted(EPISODE_BUILD_FEATURES)}\n"
        f"  actual:   {sorted(declared)}"
    )
    assert len(EPISODE_BUILD_FEATURES) == 5, _fix_the_docs(
        f"the EPISODE BUILD read-set is now "
        f"{len(EPISODE_BUILD_FEATURES)} of {PHYSICALAI_TOTAL_FEATURES}, not 5."
    )

    src = _source("tanitad/data/physicalai.py")
    # Each calibration constant must be PASSED to the chunk fetcher, not merely
    # defined -- a defined-but-unused constant would inflate the count.
    for const in ("_CALIB_INTR", "_CALIB_EXTR", "_CALIB_VEH"):
        assert re.search(rf"_calib_chunk_path\([^)]*{const}", src), _fix_the_docs(
            f"{const} is declared but never passed to _calib_chunk_path(); it is no "
            f"longer a real read, so the episode-build count drops below 5."
        )
    assert 'root / "labels" / "egomotion"' in src, _fix_the_docs(
        "physicalai.py no longer reads labels/egomotion; the count changed."
    )


def test_r0_selection_reads_exactly_two_features() -> None:
    """``physicalai_r0.py`` (clip selection) reads egomotion + the front-wide camera."""
    src = _source("scripts/physicalai_r0.py")
    found = set()
    if re.search(r'EGO_TMPL\s*=\s*"labels/egomotion/', src):
        found.add("egomotion")
    if re.search(r'CAM_TMPL\s*=\s*\(?\s*"camera/camera_front_wide_120fov/', src):
        found.add("camera_front_wide_120fov")
    assert found == R0_SELECTION_FEATURES, _fix_the_docs(
        f"the r0 SELECTION read-set changed.\n"
        f"  expected: {sorted(R0_SELECTION_FEATURES)}\n"
        f"  actual:   {sorted(found)}\n"
        f"  (The '2 of 36' figure was only ever true of this layer.)"
    )
    assert OBSTACLE_FEATURE not in src, _fix_the_docs(
        "physicalai_r0.py now reads obstacle.offline; the r0 count is no longer 2."
    )


def test_program_wide_readset_is_six_and_includes_obstacle_offline() -> None:
    """``obstacle.offline`` is a REAL read (side-car join), so program-wide is 6.

    This is the read that made "5 of 36" stale on 2026-08-03 -- the same landing
    that made the distance-keeping INTAKE's "until that lands" line stale.
    """
    hits = [rel for rel in OBSTACLE_READ_SITES if OBSTACLE_FEATURE in _source(rel)]
    assert len(hits) >= 2, _fix_the_docs(
        f"obstacle.offline was found in only {len(hits)} of "
        f"{len(OBSTACLE_READ_SITES)} expected read sites ({hits}). Two probes are "
        f"required before concluding either presence or absence. If the side-car "
        f"join was genuinely removed, the program-wide count drops back to 5."
    )
    assert len(PROGRAM_WIDE_FEATURES) == 6, _fix_the_docs(
        f"the PROGRAM-WIDE read-set is now "
        f"{len(PROGRAM_WIDE_FEATURES)} of {PHYSICALAI_TOTAL_FEATURES}, not 6."
    )


def test_obstacle_offline_is_outside_the_episode_build() -> None:
    """The 5-vs-6 split is load-bearing: the join is pod-side, not in the build.

    If this ever fails, the two counts have merged and every doc that distinguishes
    them (including this test's own docstring) is wrong.
    """
    src = _source("tanitad/data/physicalai.py")
    assert "obstacle" not in src, _fix_the_docs(
        "physicalai.py now references obstacle.offline. The episode build has "
        "absorbed the side-car join, so the episode-build count is 6 and the "
        "5-vs-6 distinction this test documents no longer exists."
    )


# --------------------------------------------------------------------------- #
# 2. The drift detector -- the case all three previous rots actually were.       #
# --------------------------------------------------------------------------- #
_FEATURE_PATH_PATTERNS = (
    re.compile(r'labels/([A-Za-z_][A-Za-z0-9_.]*)'),
    re.compile(r'"labels"\s*/\s*"([A-Za-z_][A-Za-z0-9_.]*)"'),
    re.compile(r'calibration/([A-Za-z_][A-Za-z0-9_.]*)'),
)


def _feature_names_in(src: str) -> set[str]:
    """HF feature directory names appearing as path segments in `src`.

    Skips template placeholders (``{kind}``/``<kind>``) and local sidecar CSVs
    (``calibration/physicalai_*.csv``), neither of which is a dataset feature.

    ⚠️ A trailing ``.`` is sentence punctuation from a docstring, NOT part of the
    name -- but an *internal* dot is real (``obstacle.offline``), so strip only the
    trailing one. Getting this wrong reports a phantom undeclared feature.
    """
    out: set[str] = set()
    for pat in _FEATURE_PATH_PATTERNS:
        for name in pat.findall(src):
            name = name.rstrip(".")
            if not name or name.endswith(".csv") or name.startswith("physicalai_"):
                continue
            out.add(name)
    return out


def test_no_undeclared_feature_path_in_episode_build() -> None:
    """A NEW feature path in ``physicalai.py`` fails here, not silently in prose."""
    found = _feature_names_in(_source("tanitad/data/physicalai.py"))
    undeclared = found - EPISODE_BUILD_FEATURES
    assert not undeclared, _fix_the_docs(
        f"physicalai.py references PhysicalAI feature(s) this test does not "
        f"declare: {sorted(undeclared)}.\n"
        f"  The episode-build count is no longer {len(EPISODE_BUILD_FEATURES)}.\n"
        f"  This is EXACTLY the drift that made the prose number stale three times."
    )
    assert found, "drift detector matched nothing — the regexes have gone stale."


# --------------------------------------------------------------------------- #
# 3. The DENOMINATOR. "N of 36" has two halves and only one of them was ever     #
#    checked. 36 is MEASURED in-repo, not merely published.                      #
# --------------------------------------------------------------------------- #
#: MEASURED enumeration of every PhysicalAI-AV feature (2026-07-26 probe package).
_FEATURES_CSV = (
    _REPO / "TanitAD Research Hub" / "Data Engineering" / "Implementation" /
    "incoming" / "2026-07-26-physicalai-feature-probe" / "pai_features.csv"
)


def _published_feature_names() -> set[str]:
    rows = _FEATURES_CSV.read_text(encoding="utf-8").splitlines()
    return {ln.split(",", 1)[0].strip() for ln in rows[1:] if ln.strip()}


def test_denominator_is_36_and_every_feature_we_read_is_real() -> None:
    """The "36" is MEASURED, and our 6 names are genuinely among those 36.

    Pins the *other* half of "N of 36" — the denominator — and catches a typo'd
    feature name, which would otherwise silently shrink the read-set.

    Skips (loudly) on a partial checkout: pods receive ``stack/`` by file-ship and
    may not carry the Research Hub tree. The numerator assertions above still run.
    """
    if not _FEATURES_CSV.is_file():
        pytest.skip(
            f"denominator UNPINNED on this checkout — {_FEATURES_CSV.name} absent. "
            f"The read-set counts above are still asserted; only the '36' is unchecked."
        )
    published = _published_feature_names()
    assert len(published) == PHYSICALAI_TOTAL_FEATURES, _fix_the_docs(
        f"PhysicalAI-AV now publishes {len(published)} features, not "
        f"{PHYSICALAI_TOTAL_FEATURES}. EVERY 'N of 36' statement in the programme "
        f"has the wrong denominator (and 'the other 32' becomes "
        f"{len(published) - len(PROGRAM_WIDE_FEATURES)})."
    )
    bogus = PROGRAM_WIDE_FEATURES - published
    assert not bogus, _fix_the_docs(
        f"we claim to read feature(s) that PhysicalAI-AV does not publish: "
        f"{sorted(bogus)}. Either a name is misspelled here or the dataset changed."
    )


def test_drift_detector_actually_fires() -> None:
    """The guard must be able to FAIL, or it is decoration.

    This programme has repeatedly been burned by guards structurally unable to
    report the answer they are cited for (the C13 class). The drift detector above
    only earns its keep if a newly-added feature path really does trip it, so
    exercise it on a synthetic source containing a feature we do not read.
    """
    synthetic = 'zp = dl(REPO, "labels/lidar_front/lidar_front.chunk_0001.zip")'
    assert "lidar_front" in _feature_names_in(synthetic), (
        "the drift detector did NOT flag an obviously-new feature path — its "
        "regexes have gone stale and it can no longer catch the rot it exists for."
    )
    # ...and it must NOT flag punctuation or the local sidecar CSVs (the two
    # false-positive classes actually hit while writing this test).
    assert _feature_names_in("see calibration/vehicle_dimensions.") == {
        "vehicle_dimensions"
    }
    assert _feature_names_in('Path(root) / "calibration" / "physicalai_wheelbase.csv"') == set()


@pytest.mark.parametrize("rel", ["tanitad/data/physicalai.py", "scripts/physicalai_r0.py"])
def test_no_undeclared_feature_path_program_wide(rel: str) -> None:
    """Nothing in either ingest entry point reads outside the program-wide set."""
    undeclared = _feature_names_in(_source(rel)) - PROGRAM_WIDE_FEATURES
    assert not undeclared, _fix_the_docs(
        f"{rel} references undeclared PhysicalAI feature(s): {sorted(undeclared)}."
    )
