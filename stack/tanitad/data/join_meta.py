"""Self-describing join sidecars: a digest carries the ARTIFACT it covers.

⛔ **WHY THIS MODULE EXISTS — MEASURED 2026-09-05** (`Project Steering/
Decisions/2026-09-05-mm-decisions.md` §M18, escalated by the DataFlyWheel).
The programme's two ``obstacle.offline`` join sidecars record ``summary.md5``
**over different artifacts**, and neither says which:

======================================  ==========================  ==========
sidecar                                 ``summary.md5`` covers      ``summary.out`` names
======================================  ==========================  ==========
``train2400_agents.jsonl.xz.meta.json`` the **compressed** ``.xz``  ``…/train2400_agents.jsonl.xz``
``val40_agents.jsonl.meta.json``        the **decompressed** file   ``…/val40_agents.jsonl``
``val40_agents.jsonl.xz.meta.json``     the **decompressed** file   ``…/val40_agents.jsonl`` ⚠️
======================================  ==========================  ==========

Re-verified here, not copied: ``md5(train2400_agents.jsonl.xz)`` =
``24cbdca8c3b23aafc2fb17e6bf99cf76`` = the value in the train sidecar, so the
train digest is over the COMPRESSED bytes. The refcv5 density script's C5
check, written against val40, therefore **REFUSES the perfectly good train
join** — a checker that is correct in every line, applied to a file it was
never scoped for.

⚠️ **And the obvious repair — "guess from the filename" — is measurably
wrong.** ``val40_agents.jsonl.xz.meta.json`` sits beside the ``.xz`` and its
own ``summary.out`` names the ``.jsonl``; its ``xz_of_md5`` field holds the
SAME digest as ``summary.md5``, so the extension heuristic gets val40 wrong in
the other direction. A consumer cannot infer the scope. It must be told, or
it must refuse.

⇒ **The rule (M18): a recorded hash carries the ARTIFACT it was taken over —
compressed or decompressed, and the exact filename — or it is not a
verification, it is a number.** A consumer that cannot see that scope will
either refuse a good file (what happened) or, worse, **accept the wrong one
and report success**.

⭐ This is the anchor-units trap in a new costume, so it wears the same
solution rather than a second one: :mod:`tanitad.refs.anchor_meta` makes
builders write ``control_units`` INTO the ``.pt`` and consumers REFUSE a file
that declares nothing. Here builders write :func:`declare` into the sidecar
and :func:`read_digest_scope` REFUSES a sidecar that declares nothing.

Migration, because every sidecar on disk today predates the rule:
:func:`backfill` **measures** which artifact the recorded digest actually
covers (it hashes both forms and reports which one matched) and returns the
declaration to write. It never guesses, and it refuses when neither matches.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

#: schema tag written into every declaration this module makes.
SCHEMA = "tanitad.join_digest_scope/1"

#: the key the declaration lives under, at the TOP LEVEL of the sidecar dict.
BLOCK = "digest_scope"

#: what a digest may be taken over. ``compressed`` = the bytes of the file as
#: it sits on disk (``.xz``); ``decompressed`` = the ``.jsonl`` stream inside.
ARTIFACT_SCOPES = ("compressed", "decompressed")

#: hash algorithms a declaration may name.
ALGOS = ("md5", "sha256")

#: one sentence a refusal can quote, so a reader knows what the rule costs.
INCIDENT = (
    "MEASURED 2026-09-05: train2400's sidecar md5 covers the COMPRESSED .xz "
    "(24cbdca8c3b23aafc2fb17e6bf99cf76, re-verified) while val40's covers the "
    "DECOMPRESSED .jsonl -- and val40's .xz-side sidecar names the .jsonl in "
    "summary.out, so the filename cannot be used to guess. A checker "
    "inherited from val40 REFUSED the perfectly good train join.")


class JoinDigestError(ValueError):
    """Base class: the sidecar's digest cannot be turned into a verification."""


class JoinDigestScopeMissing(JoinDigestError):
    """The sidecar records a digest but does not declare what it covers."""


class JoinDigestMismatch(JoinDigestError):
    """The declared digest does not match the artifact it declares."""


@dataclass(frozen=True)
class DigestScope:
    """A digest that knows what it is about.

    ``filename`` is the **basename** the digest was taken over — the exact
    file, not its directory, because every sidecar on file records an absolute
    build-time path in a scratch directory that no consumer will ever see.
    """

    algo: str
    digest: str
    scope: str
    filename: str
    declared_by: str = "builder"
    note: str | None = None

    def __post_init__(self):
        if self.algo not in ALGOS:
            raise JoinDigestError(f"algo {self.algo!r} not in {ALGOS}")
        if self.scope not in ARTIFACT_SCOPES:
            raise JoinDigestError(
                f"scope {self.scope!r} not in {ARTIFACT_SCOPES}")
        if not self.digest or not str(self.digest).strip():
            raise JoinDigestError("digest is empty")
        if not self.filename or "/" in self.filename or "\\" in self.filename:
            raise JoinDigestError(
                f"filename must be a BASENAME, got {self.filename!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d


# ---------------------------------------------------------------------------
# builders write
# ---------------------------------------------------------------------------

def declare(digest: str, *, scope: str, filename: str, algo: str = "md5",
            declared_by: str = "builder", note: str | None = None) -> dict:
    """The block a BUILDER writes into its sidecar, at the top level.

    >>> declare("abc", scope="compressed",
    ...         filename="train2400_agents.jsonl.xz")["scope"]
    'compressed'
    """
    return DigestScope(algo=algo, digest=str(digest), scope=scope,
                       filename=os.path.basename(str(filename)),
                       declared_by=declared_by, note=note).to_dict()


def attach(meta: dict, digest: str, *, scope: str, filename: str,
           algo: str = "md5", declared_by: str = "builder",
           note: str | None = None) -> dict:
    """Return ``meta`` with the declaration attached under :data:`BLOCK`.

    ⚠️ The existing ``summary.md5`` is left exactly where it is. Nothing that
    reads these sidecars today changes behaviour; the declaration is additive.
    """
    out = dict(meta)
    out[BLOCK] = declare(digest, scope=scope, filename=filename, algo=algo,
                         declared_by=declared_by, note=note)
    return out


# ---------------------------------------------------------------------------
# consumers require
# ---------------------------------------------------------------------------

def read_digest_scope(meta: dict, *, where: str = "<sidecar>") -> DigestScope:
    """Read the declaration, or **REFUSE**.

    ⛔ There is deliberately no fallback. Inferring the scope from
    ``summary.out``'s extension is MEASURED wrong on val40's ``.xz`` sidecar
    (see the module docstring), and a wrong inference does not fail loudly —
    it either rejects a good file or accepts the wrong one and reports success.
    """
    if not isinstance(meta, dict):
        raise JoinDigestScopeMissing(f"{where}: not a JSON object")
    blk = meta.get(BLOCK)
    if not isinstance(blk, dict):
        has = "yes" if _recorded_digest(meta) else "no"
        raise JoinDigestScopeMissing(
            f"{where} declares no `{BLOCK}` block (records a digest: {has}). "
            f"A hash without its artifact scope is not a verification, it is "
            f"a number. {INCIDENT} Fix: run `join_meta.backfill(...)`, which "
            f"MEASURES which artifact the recorded digest covers and writes "
            f"the declaration, or have the builder call `join_meta.attach`.")
    try:
        return DigestScope(algo=str(blk.get("algo", "md5")),
                           digest=str(blk["digest"]),
                           scope=str(blk["scope"]),
                           filename=str(blk["filename"]),
                           declared_by=str(blk.get("declared_by", "builder")),
                           note=blk.get("note"))
    except KeyError as e:
        raise JoinDigestScopeMissing(
            f"{where}: `{BLOCK}` is present but incomplete (missing {e}). "
            f"A partial declaration is not a scope.") from None


def _recorded_digest(meta: dict) -> str | None:
    """The legacy digest, wherever this family of sidecars puts it."""
    s = meta.get("summary")
    if isinstance(s, dict) and s.get("md5"):
        return str(s["md5"])
    for k in ("md5", "sha256"):
        if meta.get(k):
            return str(meta[k])
    return None


# ---------------------------------------------------------------------------
# hashing — the artifact the SCOPE names, never the file that happens to be here
# ---------------------------------------------------------------------------

def _hash_stream(fh: Iterable[bytes] | io.BufferedReader, algo: str) -> str:
    h = hashlib.new(algo)
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
    return h.hexdigest()


def file_digest(path: str | os.PathLike, *, algo: str = "md5",
                scope: str = "compressed") -> str:
    """Digest of ``path`` under ``scope``.

    ``compressed`` hashes the bytes on disk. ``decompressed`` hashes the
    ``.xz`` stream's contents — and for a file that is not ``.xz`` the two are
    the same thing, which is exactly why the scope has to be declared: an
    uncompressed ``.jsonl`` matches BOTH scopes and can never disambiguate a
    sidecar on its own.
    """
    p = Path(path)
    if scope not in ARTIFACT_SCOPES:
        raise JoinDigestError(f"scope {scope!r} not in {ARTIFACT_SCOPES}")
    if scope == "decompressed" and p.suffix == ".xz":
        import lzma
        with lzma.open(p, "rb") as fh:
            return _hash_stream(fh, algo)
    with open(p, "rb") as fh:
        return _hash_stream(fh, algo)


def sidecar_path(join_path: str | os.PathLike) -> Path | None:
    """The sidecar beside ``join_path``, under the names this corpus uses.

    ⚠️ MEASURED: the train join's sidecar is ``<file>.xz.meta.json``, not the
    ``<stem>.meta.json`` the package doc named — a direct fetch of the
    documented name 404s. Both are probed, plus the ``.xz``-stripped form, and
    the ONE that exists is returned. Absence at one name is not absence.
    """
    p = Path(join_path)
    cands = [p.with_name(p.name + ".meta.json")]
    if p.suffix == ".xz":
        q = p.with_suffix("")                       # …/x.jsonl.xz -> …/x.jsonl
        cands += [q.with_name(q.name + ".meta.json"),
                  q.with_suffix(".meta.json")]
    cands.append(p.with_suffix(".meta.json"))
    for c in cands:
        try:
            if c.exists():
                return c
        except OSError:                              # a flapping mount is not absence
            continue
    return None


def verify(join_path: str | os.PathLike, meta: dict, *,
           where: str = "<sidecar>") -> dict:
    """Verify ``join_path`` against ``meta``'s DECLARED digest scope.

    Raises :class:`JoinDigestScopeMissing` when the sidecar declares nothing
    and :class:`JoinDigestMismatch` when the bytes disagree. Returns the
    evidence — the scope, both digests and the file actually hashed — so a
    caller can stamp it into its run record rather than merely printing "ok".
    """
    sc = read_digest_scope(meta, where=where)
    p = Path(join_path)
    if sc.scope == "decompressed" and p.suffix == ".xz":
        expect_name = p.with_suffix("").name          # the .jsonl inside
    else:
        expect_name = p.name
    if sc.filename != expect_name:
        raise JoinDigestMismatch(
            f"{where} declares a {sc.algo} over {sc.filename!r} ({sc.scope}), "
            f"but the file offered is {p.name!r} (whose {sc.scope} artifact "
            f"is {expect_name!r}). The digest is about a DIFFERENT file; "
            f"verifying against it would either refuse a good file or accept "
            f"the wrong one.")
    got = file_digest(p, algo=sc.algo, scope=sc.scope)
    ok = (got.lower() == sc.digest.lower())
    if not ok:
        raise JoinDigestMismatch(
            f"{where}: {sc.algo}({sc.scope} of {p.name}) = {got}, declared "
            f"{sc.digest}. Same scope, different bytes -- this is a real "
            f"integrity failure, not a scope error.")
    return {"verified": True, "algo": sc.algo, "scope": sc.scope,
            "filename": sc.filename, "digest": got,
            "hashed_path": str(p), "declared_by": sc.declared_by}


# ---------------------------------------------------------------------------
# migration — MEASURE the scope of a legacy digest, never guess it
# ---------------------------------------------------------------------------

def backfill(meta: dict, join_path: str | os.PathLike, *,
             algo: str = "md5", declared_by: str = "backfill-MEASURED"
             ) -> tuple[dict, dict]:
    """Give a legacy sidecar its scope, by **measuring** which artifact its
    recorded digest covers.

    Returns ``(meta_with_declaration, evidence)``. Both candidate digests are
    computed and reported, so the evidence says what was ruled out as well as
    what matched.

    ⛔ Refuses when NEITHER candidate matches: that is a real integrity
    failure or the wrong file, and inventing a scope would bury it. ⚠️ Also
    refuses when BOTH match — which happens for an uncompressed ``.jsonl``,
    where the two scopes are the same bytes — because a declaration derived
    from an ambiguous file would then be quoted about the ``.xz`` it does not
    cover. Point it at the compressed artifact, or have the builder declare.
    """
    rec = _recorded_digest(meta)
    if not rec:
        raise JoinDigestError(
            "the sidecar records no digest at all; there is nothing to scope. "
            "Have the builder call `join_meta.attach` with the digest it "
            "computed.")
    p = Path(join_path)
    cand = {s: file_digest(p, algo=algo, scope=s) for s in ARTIFACT_SCOPES}
    hits = [s for s, d in cand.items() if d.lower() == rec.lower()]
    ev = {"recorded": rec, "candidates": cand, "matched": hits,
          "path": str(p), "algo": algo}
    if not hits:
        raise JoinDigestMismatch(
            f"the recorded {algo} {rec} matches NEITHER artifact of {p.name} "
            f"({cand}). That is an integrity failure or the wrong file, and "
            f"a scope must not be invented over it.")
    if len(hits) > 1:
        raise JoinDigestError(
            f"{p.name} is not compressed, so both scopes hash identically "
            f"({rec}) and the recorded digest cannot be attributed. Run the "
            f"backfill against the COMPRESSED artifact, or have the builder "
            f"declare the scope.")
    scope = hits[0]
    fname = (p.with_suffix("").name
             if scope == "decompressed" and p.suffix == ".xz" else p.name)
    out = attach(meta, rec, scope=scope, filename=fname, algo=algo,
                 declared_by=declared_by,
                 note=("scope MEASURED by hashing both artifacts of "
                       f"{p.name}; the other candidate was "
                       f"{cand[[s for s in ARTIFACT_SCOPES if s != scope][0]]}"))
    return out, ev


def summarise(meta: dict) -> dict[str, Any]:
    """A small, JSON-safe view of a sidecar's digest situation, for stamping."""
    try:
        sc = read_digest_scope(meta)
    except JoinDigestError as e:
        return {"declared": False, "reason": str(e)[:400],
                "recorded_digest": _recorded_digest(meta)}
    return {"declared": True, **sc.to_dict()}


# ---------------------------------------------------------------------------
# the one-command migration
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    """``python -m tanitad.data.join_meta <join file> [--write]``

    Reads the sidecar beside the join, MEASURES which artifact its recorded
    digest covers, prints the evidence, and with ``--write`` attaches the
    declaration (a ``.bak`` of the sidecar is kept).
    """
    import argparse
    import json
    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument("join")
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--algo", default="md5", choices=list(ALGOS))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    side = Path(a.sidecar) if a.sidecar else sidecar_path(a.join)
    if side is None:
        print(f"no sidecar found beside {a.join}")
        return 2
    meta = json.loads(Path(side).read_text(encoding="utf-8"))
    try:
        print("already declared:",
              json.dumps(read_digest_scope(meta, where=str(side)).to_dict()))
        return 0
    except JoinDigestScopeMissing:
        pass
    out, ev = backfill(meta, a.join, algo=a.algo)
    print(json.dumps({"sidecar": str(side), "evidence": ev,
                      BLOCK: out[BLOCK]}, indent=1))
    if a.write:
        bak = Path(str(side) + ".bak")
        if not bak.exists():
            bak.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        Path(side).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"WROTE {side} (backup {bak})")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
