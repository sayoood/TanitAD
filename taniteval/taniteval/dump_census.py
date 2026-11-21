"""The census loader for banked per-window eval dumps (``windows_*.pt``).

⛔ WHY THIS EXISTS (C126, ``Project Steering/RETRACTION_LOG.md``). Two banked
dump pairs are ONE MODEL EACH: ``windows_overfit_refa-dynin-30k.pt`` is
bit-identical to ``windows_refa-dynin-30k.pt`` (same ckpt evaluated twice under
two naming schemes), and ``windows_refc-v12-identity.pt`` equals
``windows_refc-xl-30k.pt`` to GPU re-run epsilon (max |Δpred| 7.6e-06 — it is
the v1.2 experiment's zero-init control over the same frozen decode). The
duplication was documented in prose on 2026-07-26 and then re-counted by every
later census for 23 days, because **censuses re-derive their arm list from
``glob("windows_*.pt")`` — a surface no prose correction can reach**. "27 arms"
was 27 dumps over 25 distinct arms.

The machine-readable fix is ``taniteval/results/dump_exclusions.json`` (beside
the dumps; evidence in ``DUPLICATES.md`` there). THIS module is what makes it
binding: every census gets its dump list through :func:`list_dumps`, which
subtracts the exclusions — or, for a tool that legitimately wants FILES rather
than distinct arms (integrity checks, banking probes), returns everything under
``include_excluded=True`` while still *reporting* the exclusions. A bare
``glob("windows_*.pt")`` in census code is now a defect by convention.

Design rules, each load-bearing:

* **Never silently drop.** The excluded rows are part of the return value
  (:attr:`CensusResult.excluded`), and :meth:`CensusResult.summary` renders
  "N dumps = M distinct arms (K excluded: …)" for reports. A census that hides
  its subtraction just moves the C126 confusion one layer down.
* **A missing exclusions file is LOUD, not fatal** (``exclusions_missing=True``
  and the summary says so): most callers census scratch dirs in tests, and a
  hard failure there would push people back to the bare glob.
* **A STALE exclusion is FATAL** (:class:`StaleExclusionError`): every entry
  carries the sha256 of the bytes it was verified against, and if the on-disk
  file changed, the recorded equality claim is about bytes that no longer
  exist — silently excluding the new content would be *worse* than double
  counting. Re-verify the pair and update ``dump_exclusions.json``.
* **An exclusion only fires when its canonical partner is present** in the same
  directory. If the canonical dump is gone, dropping the excluded one would
  remove the MODEL from the census, not a duplicate of it; the dump is kept and
  the summary carries a note.

No torch, no GPU: names + bytes only, so it is importable everywhere
(including ``tools/`` scripts and pod-side code) at zero cost.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

EXCLUSIONS_NAME = "dump_exclusions.json"
PATTERN = "windows_*.pt"
_PRE = "windows_"


class ExclusionsError(RuntimeError):
    """The exclusions file exists but cannot be trusted (unparseable/invalid)."""


class StaleExclusionError(ExclusionsError):
    """An exclusion's recorded sha256 no longer matches the file on disk.

    The exclusion asserts a measured equality about SPECIFIC bytes. If the
    bytes changed (a re-banked dump, a corrupted pull), that assertion is
    unverified for the new content — excluding it silently could hide a
    genuinely distinct arm. Re-run the pair comparison and update
    ``dump_exclusions.json`` (see ``DUPLICATES.md`` for the protocol)."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_key(dump_name: str) -> str:
    """``windows_<key>.pt`` -> ``<key>`` (other names pass through stemmed)."""
    stem = dump_name[:-len(".pt")] if dump_name.endswith(".pt") else dump_name
    return stem[len(_PRE):] if stem.startswith(_PRE) else stem


@dataclass
class CensusResult:
    """What a census over ``windows_*.pt`` actually found — including what it
    subtracted, so no report can quote the count without the subtraction."""

    results_dir: Path
    #: the census: deduplicated paths (or ALL paths under ``include_excluded``)
    paths: list = field(default_factory=list)
    #: dump filename -> short reason, for every exclusion that FIRED (the
    #: excluded file and its canonical partner are both on disk). Under
    #: ``include_excluded`` these rows are also still in ``paths``.
    excluded: dict = field(default_factory=dict)
    #: ``{"dumps_found": N, "distinct_arms": M}`` — M subtracts fired
    #: exclusions regardless of ``include_excluded`` (it is a fact about the
    #: bank, not about what this caller chose to iterate).
    counts: dict = field(default_factory=dict)
    #: True when ``dump_exclusions.json`` was absent beside the dumps.
    exclusions_missing: bool = False
    #: echo of the call flag, so downstream code can tell a dumps-census
    #: (files) from an arms-census (distinct models).
    include_excluded: bool = False
    #: excluded arm key -> canonical arm key, for EVERY well-formed entry in
    #: the json (whether or not it fired) — the join surface for censuses over
    #: derived artifacts (``driving_<key>.json`` etc.).
    pairs_by_key: dict = field(default_factory=dict)
    #: human-readable oddities (e.g. an exclusion whose canonical is absent).
    notes: list = field(default_factory=list)

    @property
    def arm_keys(self) -> list:
        """Sorted arm keys of ``paths`` — drop-in for the old bare-glob list."""
        return [arm_key(p.name) for p in self.paths]

    @property
    def excluded_arm_keys(self) -> set:
        """Arm keys of the exclusions that fired in this census."""
        return {arm_key(n) for n in self.excluded}

    def summary(self) -> str:
        n = self.counts.get("dumps_found", len(self.paths))
        if self.exclusions_missing:
            return (f"{n} dumps; DISTINCT-ARM COUNT NOT DEDUPLICATED -- "
                    f"{EXCLUSIONS_NAME} missing beside the dumps, so known "
                    f"duplicate banks cannot be subtracted (see "
                    f"taniteval/results/DUPLICATES.md)")
        m = self.counts.get("distinct_arms", n)
        k = len(self.excluded)
        if k:
            pairs = ", ".join(
                f"{name} -> {_PRE}{self.pairs_by_key.get(arm_key(name), '?')}.pt"
                for name in sorted(self.excluded))
            body = f"{n} dumps = {m} distinct arms ({k} excluded: {pairs})"
        else:
            body = f"{n} dumps = {m} distinct arms (0 excluded)"
        if self.include_excluded:
            body += " [include_excluded=True: paths carry ALL dumps]"
        if self.notes:
            body += " [" + "; ".join(self.notes) + "]"
        return body

    def __str__(self) -> str:  # noqa: D105
        return self.summary()


def _load_entries(exc_path: Path) -> list:
    try:
        doc = json.loads(exc_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — any parse failure is the same fact
        raise ExclusionsError(
            f"{exc_path} exists but cannot be parsed ({type(e).__name__}: "
            f"{e}). A census cannot run against an unreadable exclusion "
            f"list — fix the file (schema: excluded_name, canonical_name, "
            f"reason, evidence, sha256).") from e
    entries = doc.get("exclusions")
    if not isinstance(entries, list):
        raise ExclusionsError(
            f"{exc_path} has no 'exclusions' list — wrong schema.")
    for ent in entries:
        if not isinstance(ent, dict) or "excluded_name" not in ent \
                or "canonical_name" not in ent:
            raise ExclusionsError(
                f"{exc_path}: entry without excluded_name/canonical_name: "
                f"{ent!r}")
    return entries


def _validate_hash(res: Path, name: str, recorded, exc_path: Path,
                   role: str) -> None:
    """Fail LOUDLY when a recorded sha256 no longer matches the on-disk file."""
    if not recorded:
        return
    p = res / name
    if not p.exists():
        return
    actual = _sha256(p)
    if actual != recorded:
        raise StaleExclusionError(
            f"STALE EXCLUSION: {exc_path} records {role} {name} at sha256 "
            f"{recorded[:16]}... but the file on disk hashes "
            f"{actual[:16]}... — the bytes the equality claim was measured "
            f"on are gone. Re-verify the pair (DUPLICATES.md protocol) and "
            f"update {EXCLUSIONS_NAME}; refusing to exclude unverified "
            f"content.")


def list_dumps(results_dir, *, include_excluded: bool = False) -> CensusResult:
    """Census the ``windows_*.pt`` dumps under ``results_dir``, exclusion-aware.

    Default (``include_excluded=False``): ``paths`` is the DISTINCT-ARM
    census — known duplicate banks (per ``dump_exclusions.json`` beside the
    dumps) are subtracted, and reported in ``.excluded`` / ``.summary()``.

    ``include_excluded=True``: ``paths`` carries every dump FILE — for tools
    that genuinely operate per-dump (integrity checks, banking probes). The
    exclusions are still validated and reported; ``counts["distinct_arms"]``
    still subtracts them. Callers using this form should say in their output
    that they count dumps, not arms.

    Raises :class:`StaleExclusionError` when an on-disk file no longer matches
    its recorded sha256, and :class:`ExclusionsError` when the exclusions file
    is present but unreadable. A merely ABSENT exclusions file is not an
    error: everything is returned with ``exclusions_missing=True`` and a
    summary line that says so out loud.
    """
    res = Path(results_dir)
    found = sorted(res.glob(PATTERN))
    exc_path = res / EXCLUSIONS_NAME
    if not exc_path.exists():
        return CensusResult(
            results_dir=res, paths=found, excluded={},
            counts={"dumps_found": len(found), "distinct_arms": len(found)},
            exclusions_missing=True, include_excluded=include_excluded)

    entries = _load_entries(exc_path)
    by_name = {p.name: p for p in found}
    pairs_by_key, excluded, notes = {}, {}, []
    for ent in entries:
        ex_name, canon = ent["excluded_name"], ent["canonical_name"]
        pairs_by_key[arm_key(ex_name)] = arm_key(canon)
        _validate_hash(res, ex_name, ent.get("sha256"), exc_path, "excluded")
        _validate_hash(res, canon, ent.get("canonical_sha256"), exc_path,
                       "canonical")
        if ex_name not in by_name:
            continue                      # nothing to subtract here
        if canon not in by_name:
            notes.append(f"{ex_name} kept: canonical {canon} absent, "
                         f"dropping it would remove the arm itself")
            continue
        excluded[ex_name] = ent.get("reason", "excluded (no reason recorded)")

    paths = found if include_excluded else \
        [p for p in found if p.name not in excluded]
    return CensusResult(
        results_dir=res, paths=paths, excluded=excluded,
        counts={"dumps_found": len(found),
                "distinct_arms": len(found) - len(excluded)},
        exclusions_missing=False, include_excluded=include_excluded,
        pairs_by_key=pairs_by_key, notes=notes)


def check_explicit(paths):
    """Duplicate-VALUE guard for tools whose dumps arrive as explicit CLI args
    (``ff_rescore.py --dump LABEL=PATH``) rather than a glob.

    For each passed path, consults ``dump_exclusions.json`` in that path's
    directory. Returns ``(findings, consulted)``:

    * ``findings``: list of dicts with ``kind`` in
      ``{"pair_present", "excluded_passed"}`` plus ``excluded``, ``canonical``
      and ``reason``. ``pair_present`` means BOTH members of a known
      same-model pair were passed — scoring them side by side double-counts
      one model (the exact C126 error; a name-uniqueness check cannot see it).
      ``excluded_passed`` means an excluded dump was passed WITHOUT its
      canonical partner — legitimate (the dump is real), just worth a printed
      note.
    * ``consulted``: the exclusions files that were found and read. Empty
      means no duplicate-VALUE knowledge was available for any passed path —
      callers should say so rather than stay silent.

    sha256s of the passed files that appear in an exclusions entry are
    validated (stale ⇒ :class:`StaleExclusionError`), same as the census path.
    """
    findings, consulted = [], []
    by_dir: dict = {}
    for p in map(Path, paths):
        by_dir.setdefault(p.parent, set()).add(p.name)
    for d, names in sorted(by_dir.items()):
        exc_path = d / EXCLUSIONS_NAME
        if not exc_path.exists():
            continue
        consulted.append(exc_path)
        for ent in _load_entries(exc_path):
            ex_name, canon = ent["excluded_name"], ent["canonical_name"]
            if ex_name not in names:
                continue
            _validate_hash(d, ex_name, ent.get("sha256"), exc_path,
                           "excluded")
            if canon in names:
                _validate_hash(d, canon, ent.get("canonical_sha256"),
                               exc_path, "canonical")
            findings.append({
                "kind": "pair_present" if canon in names else "excluded_passed",
                "excluded": ex_name, "canonical": canon,
                "reason": ent.get("reason", ""),
                "exclusions_file": str(exc_path),
            })
    return findings, consulted
