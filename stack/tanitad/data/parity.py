"""Parity-corpus INTEGRITY — the content check behind CLAUDE.md §Invariants.

    "Parity is sacred: the canonical train corpus is
     ``physicalai-train-e438721ae894`` (2376 episodes) with skip-hash
     ``f09e44db``. Anything that re-selects episodes breaks cross-arm
     comparability and must be refused."

Until this module existed the ONLY enforcement in any Python trainer was a path
SUBSTRING match (``train_flagship_v4._assert_parity``: ``if PARITY_KEY not in
tc: raise``). That check passes for a *correctly named directory holding the
wrong number of episodes* — which is exactly the failure this program has
already hit: the ``/workspace`` MooseFS quota fills mid-build (``df`` does NOT
show it), the build stops, and the split dir keeps its name. A truncated corpus
then trains silently and every cross-arm comparison off it is void, invisibly.

What this module adds, in ONE place (the codebase already carries a 4×
copy-pasted window class; 13 trainers × a copy-pasted parity check would be
worse):

  1. **count check** — ``len(ep_*.pt) == manifest.episode_count``
  2. **content check** — ``sha256(sorted(episode uids))`` vs the committed
     manifest digest, so a *substituted* or *re-selected* episode set of the
     right size is refused too
  3. **known-leaky split refusal** — ``physicalai-val-f1b378f295ae`` has 78.5 %
     of its populated episodes IN the parity train set (MEASURED, registry
     §Branch-B controls, corrected 2026-07-25). Training or validating against
     it is a leak, not a val.
  4. a **loud, early, actionable** refusal that names expected-vs-actual and the
     first missing/extra ids — raised before any GPU allocation.

Episode identity
----------------
The epcache layout (``tanitad/data/epcache.py``) writes ``ep_%05d.pt`` where the
index is the position of the clip in the ORDERED source list, and ``skip_%05d``
for a clip that failed to build. The index is therefore the canonical, stable
identity of an episode *within a build key* — and the build key itself
(``e438721ae894``) is already a hash of the ordered clip ids + build params
(``epcache.cache_key``). So ``sha256(sorted ep_*.pt basenames))`` pins exactly
"which episode slots are present", which is what truncation and re-selection
change. It does NOT hash episode CONTENT (tensor bytes) — see the manifest's
``limitations`` field and WAVE1_B_REPORT.md.

Two uid spaces, not one (added 2026-07-27)
------------------------------------------
Everything above describes the **raw epcache**. The **v2 compressed** cache is a
flat set of ``<clip_id>.v2ep.pt`` with no positions at all, so ``ep_*.pt`` uids
simply do not exist in it and the checks above cannot be evaluated against one.
That is not academic: the raw epcache at the v5 wide geometry is ~697 GB for the
train split and fits on no host, so **v5's corpus is a v2 cache** — and
``train_flagship4b --v2-cache`` applied NO parity check on that branch at all
(MEASURED, WIDE_FOV_BUILD.md §6). §9 adds the clip-id membership proof, its
registration path and the trainer guard. Read §9's header for exactly what the
v2 path can and cannot prove; it is a shorter list than the raw path's.

Escape hatches, on purpose
--------------------------
Trainers legitimately run on non-parity corpora (toy episodes, comma2k19, the
v2 9 000-clip corpus ``4b7eeeac222d``, the SIDE dynamics encoder). A cache path
that references no registered parity key is NOT an error: the guard prints one
loud ``[parity] NON-PARITY`` line and returns ``parity=False``. Only a caller
that passes ``require=True`` (``train_flagship_v4``, which has always hard-
required parity) refuses in that case. There is deliberately **no environment
variable that disables the content check** — the opt-out is per-call and visible
in the trainer's own argv.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# 1. the registered corpus keys                                                 #
# --------------------------------------------------------------------------- #
PARITY_TRAIN_KEY = "physicalai-train-e438721ae894"
PARITY_VAL_KEY = "physicalai-val-0c5f7dac3b11"
PARITY_SKIP_HASH = "f09e44db"          # 24-corrupt-clip skipset marker
PARITY_TRAIN_EPISODES = 2376
PARITY_VAL_EPISODES = 600

#: Split dirs that are NOT admissible as a val set for a parity-trained arm.
#: ``f1b378f295ae``: 62 of its 79 populated episodes (78.5 %) are IN the parity
#: train set (MEASURED — MODEL_REGISTRY.md §Branch-B "Leakage — CORRECTED
#: 2026-07-25"). The canonical CLEAN val is ``PARITY_VAL_KEY``.
LEAKY_SPLIT_KEYS: dict[str, str] = {
    "physicalai-val-f1b378f295ae":
        "78.5 % of its populated episodes are IN the parity train set "
        "(MEASURED 2026-07-25); the canonical CLEAN val is "
        f"{PARITY_VAL_KEY}",
}

MANIFEST_PATH = Path(__file__).with_name("parity_manifest.json")
MANIFEST_SCHEMA = "tanitad.parity_manifest/1"
EPISODE_GLOB = "ep_*.pt"
_EP_RE = re.compile(r"^ep_(\d+)\.pt$")

#: The v2 COMPRESSED cache is a flat set of ``<clip_id>.v2ep.pt`` — a different
#: uid space from the epcache's positional ``ep_%05d.pt``. See §9.
V2_EPISODE_GLOB = "*.v2ep.pt"
V2_SUFFIX = ".v2ep.pt"
V2_UID_KIND = "v2ep_clipid"
EPCACHE_UID_KIND = "epcache_basename"


class ParityViolation(SystemExit):
    """A refusal to train. Subclasses ``SystemExit`` so it terminates a pod run
    with a non-zero status the supervisor sees, and so existing
    ``except SystemExit`` handling around ``_assert_parity`` keeps working."""


# --------------------------------------------------------------------------- #
# 2. uids + digest                                                              #
# --------------------------------------------------------------------------- #
def episode_uid(p: str | Path) -> str:
    """The stable identity of one cached episode = its ``ep_%05d.pt`` basename.

    Feature caches (``dino_precompute``) and label caches (``v15_prep``) carry
    the SAME basename forward (``o = dst / f.name`` / the ``eids`` list), so one
    uid space covers the raw epcache, the DINO feature dirs and the v15/v16
    label caches."""
    return Path(p).name


def uid_digest(uids: Iterable[str]) -> str:
    """``sha256`` over the newline-joined SORTED uids.

    Canonical serialization is fixed here and nowhere else; changing it
    invalidates every committed manifest, so don't.
    """
    us = sorted(str(u) for u in uids)
    if len(set(us)) != len(us):
        dupes = sorted({u for u in us if us.count(u) > 1})
        raise ValueError(f"duplicate episode uids in the set: {dupes[:5]}")
    return hashlib.sha256("\n".join(us).encode("utf-8")).hexdigest()


def episode_index(uid: str) -> int | None:
    """``ep_01798.pt`` -> ``1798``; ``None`` when the name is not an epcache
    episode file."""
    m = _EP_RE.match(str(uid))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# 3. cache-dir resolution + scanning                                            #
# --------------------------------------------------------------------------- #
def resolve_split_dir(root: str | Path, pattern: str) -> Path:
    """Newest dir under ``root`` matching ``pattern`` — the
    ``train_worldmodel --data cached`` convention shared by ``refb_train
    .load_cached_episodes`` and ``refa_train.load_feature_episodes``.

    Raises ``AssertionError`` (NOT :class:`ParityViolation`) when nothing
    matches, because "you gave me no val dir" is not a parity violation — and
    ``refa_train`` / ``refb_train`` deliberately ``except AssertionError`` around
    their optional val-metrics block. Turning that into a SystemExit would kill a
    finished 30 k run at the metrics write."""
    r = Path(root)
    dirs = sorted(d for d in r.glob(pattern) if d.is_dir())
    assert dirs, f"no cache dir matching {pattern} under {r}"
    return dirs[-1]


def scan_cache_dir(cache_dir: str | Path, pattern: str = EPISODE_GLOB) -> list[str]:
    """Every episode uid physically present in ``cache_dir`` (sorted)."""
    return sorted(p.name for p in Path(cache_dir).glob(pattern))


def scan_skip_markers(cache_dir: str | Path) -> list[int]:
    """The ``skip_%05d`` indices the build recorded — the 24 corrupt clips on a
    healthy parity build. A cache with skip markers but a short ``ep_*.pt`` set
    is truncated; a cache with NEITHER is truncated *and* lost its evidence."""
    out = []
    for p in Path(cache_dir).glob("skip_*"):
        m = re.match(r"^skip_(\d+)$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _V2_HINT(cache_dir: str | Path) -> str:
    """A trailing note appended to a RAW-path refusal when the directory it was
    given is actually a **v2 compressed cache**.

    ⚠️ This exists because the raw guard's refusals were technically correct and
    diagnostically misleading on a v2 dir: pointing ``train_flagship_v4
    --train-cache`` at one produces *"does not reference the canonical corpus"*,
    which sends the reader off to rename a directory — when the real answer is
    that **``train_flagship_v4`` has no v2 support at all** and never should
    acquire one speculatively. The absence is now NAMED at the point of failure
    instead of being latent (see §9 and V2_PARITY_ENFORCEMENT.md §4)."""
    try:
        if not any(Path(cache_dir).glob(V2_EPISODE_GLOB)):
            return ""
    except OSError:                                       # unreadable dir
        return ""
    return (
        f"\n  ⚠️ THIS IS A V2 COMPRESSED CACHE ({V2_EPISODE_GLOB}), not a raw "
        f"epcache."
        f"\n  Raw-epcache trainers (train_flagship_v4 --train-cache, "
        f"train_flagship4b --cache-dirs)"
        f"\n  CANNOT read it: episode identity here is a CLIP ID, not an "
        f"ep_%05d.pt position."
        f"\n  Use:  train_flagship4b.py --v2-cache <dir> --require-parity"
        f"\n  and register the corpus first: scripts/register_v2_sibling.py "
        f"(parity.py §9).")


def corpus_key_of(path: str | Path,
                  manifest_path: str | Path | None = None) -> str | None:
    """The registered corpus key this path references, or ``None``.

    Substring match on the resolved POSIX path — identical to the rule
    ``train_flagship_v4._assert_parity`` has always used, so wiring this in
    never *loosens* an existing check.

    Ties are broken **longest key first**, then lexicographically. That matters
    only once GEOMETRY SIBLINGS exist (§8/§9): a sibling cache legitimately
    lives under a path that also contains its parent's key
    (``…/physicalai-train-e438721ae894/wide120/physicalai-train-w120-<hex>/``),
    and the plain lexicographic rule would resolve it to the PARENT — reporting
    a real cache as the wrong corpus. ⚠️ This cannot change any pre-existing
    resolution, and that is a fact rather than a hope: the three keys registered
    before siblings existed (``physicalai-train-e438721ae894`` 29 chars,
    ``physicalai-val-0c5f7dac3b11`` and ``physicalai-val-f1b378f295ae`` 27 each)
    are pairwise non-overlapping, so on any path where several match, the
    longest-first order and the lexicographic order agree
    (``tests/test_v2_parity.py::test_longest_match_cannot_change_legacy_resolution``).
    """
    s = str(Path(path).resolve()).replace("\\", "/")
    keys = {PARITY_TRAIN_KEY, PARITY_VAL_KEY, *LEAKY_SPLIT_KEYS}
    try:                       # manifest may register more; a missing manifest
        keys |= set(load_manifest(manifest_path).get("corpora", {}))
    except SystemExit:                                    # firewall direction
        pass
    for key in sorted(keys, key=lambda k: (-len(k), k)):
        if key in s:
            return key
    return None


# --------------------------------------------------------------------------- #
# 4. the manifest                                                               #
# --------------------------------------------------------------------------- #
_MANIFEST_CACHE: dict[str, dict] = {}


def load_manifest(path: str | Path | None = None) -> dict:
    """Read (and memoize) the committed episode manifest."""
    p = Path(path) if path else MANIFEST_PATH
    k = str(p)
    if k not in _MANIFEST_CACHE:
        if not p.exists():
            raise ParityViolation(
                f"parity manifest missing: {p}\nRegenerate it with "
                f"scripts/make_parity_manifest.py (see its docstring).")
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("schema") != MANIFEST_SCHEMA:
            raise ParityViolation(
                f"parity manifest {p} has schema {m.get('schema')!r}, expected "
                f"{MANIFEST_SCHEMA!r}")
        _MANIFEST_CACHE[k] = m
    return _MANIFEST_CACHE[k]


def manifest_entry(key: str, path: str | Path | None = None) -> dict | None:
    return load_manifest(path).get("corpora", {}).get(key)


def build_entry(uids: Sequence[str], *, corpus_key: str, split: str,
                skip_indices: Sequence[int] = (), uid_source: str = "recorded",
                provenance: dict | None = None) -> dict:
    """One manifest entry from an OBSERVED uid set (the self-recording path)."""
    us = sorted(str(u) for u in uids)
    return {
        "corpus_key": corpus_key,
        "split": split,
        "episode_count": len(us),
        "uid_kind": "epcache_basename",
        "uid_source": uid_source,
        "episode_uid_sha256": uid_digest(us),
        "skip_indices": sorted(int(i) for i in skip_indices),
        "skip_count": len(skip_indices),
        "provenance": provenance or {},
    }


# --------------------------------------------------------------------------- #
# 5. the check                                                                  #
# --------------------------------------------------------------------------- #
def _fmt(ids: Sequence[str], n: int = 6) -> str:
    if not ids:
        return "(none)"
    head = ", ".join(ids[:n])
    return head if len(ids) <= n else f"{head}, … (+{len(ids) - n} more)"


def _refuse(label: str, key: str, cache_dir, lines: list[str]) -> None:
    raise ParityViolation(
        "\n".join([
            "",
            "=" * 78,
            f"PARITY VIOLATION [{label}] — corpus {key}",
            "=" * 78,
            f"  cache      : {cache_dir}",
            *lines,
            "",
            "  The canonical corpus is SACRED (CLAUDE.md §Invariants). A truncated or",
            "  re-selected episode set breaks cross-arm comparability INVISIBLY — every",
            "  number produced against it is void. Refusing to train.",
            "",
            "  Most common cause: the /workspace MooseFS quota filled mid-build. `df` does",
            "  NOT show that quota — verify with a real dd write test, not df.",
            "  Rebuild:  scripts/rebuild_pai_rolling.py --expect-key e438721ae894 --skip-idx …",
            "  Re-verify: scripts/pod_ops/compute_skipset.py   (hash + count verdict)",
            "  If (and only if) the cache is verified good and the manifest is stale, "
            "re-record",
            "  it with scripts/make_parity_manifest.py --record and commit the diff.",
            "=" * 78,
        ]))


def check_uids(uids: Sequence[str], *, corpus_key: str, label: str,
               cache_dir: str | Path = "<in-memory>", mode: str = "strict",
               manifest_path: str | Path | None = None,
               subset_note: str | None = None) -> dict:
    """Verify an OBSERVED uid set against the committed manifest entry.

    ``mode='strict'``  the uid set must equal the manifest set exactly.
    ``mode='subset'``  the uid set must be a sorted PREFIX of the manifest set —
                       what a deliberate ``[:n]`` episode-subset knob produces
                       (``dino_precompute --train-n``, ``--episodes N``). Still
                       refuses foreign, renumbered or substituted episodes; the
                       shortfall is printed LOUD so a truncation is visible in
                       the log even where it is not fatal.

    ``subset_note``    replaces the default subset warning's tail. The default
                       ("must not be cross-compared with full-corpus arms") is
                       right for a TRAIN subset and wrong for a val DEPLOYMENT,
                       where every arm shares the same 40-episode prefix; see
                       :func:`assert_val_cache`.

    Returns a provenance record. Raises :class:`ParityViolation` otherwise.
    """
    if mode not in ("strict", "subset"):
        raise ValueError(f"mode must be 'strict' or 'subset', got {mode!r}")
    ent = manifest_entry(corpus_key, manifest_path)
    if ent is None:
        _refuse(label, corpus_key, cache_dir, [
            f"  manifest   : NO ENTRY for {corpus_key!r} in "
            f"{Path(manifest_path or MANIFEST_PATH)}",
            "  A registered parity key with no manifest entry cannot be verified.",
        ])
    obs = sorted(str(u) for u in uids)
    n_exp = int(ent["episode_count"])
    exp_digest = ent.get("episode_uid_sha256")
    rec: dict = {
        "corpus_key": corpus_key, "split": ent.get("split"),
        "cache_dir": str(cache_dir), "mode": mode,
        "episodes_expected": n_exp, "episodes_loaded": len(obs),
        "uid_source": ent.get("uid_source"),
        "skip_hash": PARITY_SKIP_HASH,
        "episode_uid_sha256": uid_digest(obs) if obs else None,
        "episode_uid_sha256_expected": exp_digest,
    }

    # -- count -------------------------------------------------------------- #
    if mode == "strict" and len(obs) != n_exp:
        _refuse(label, corpus_key, cache_dir, [
            f"  episodes   : {len(obs)} loaded, {n_exp} expected"
            f"   <-- {'TRUNCATED by %d' % (n_exp - len(obs)) if len(obs) < n_exp else 'EXTRA %d' % (len(obs) - n_exp)}",
            *_diff_lines(obs, ent),
        ])
    if mode == "subset" and len(obs) > n_exp:
        _refuse(label, corpus_key, cache_dir, [
            f"  episodes   : {len(obs)} loaded, at most {n_exp} possible"
            f"   <-- {len(obs) - n_exp} FOREIGN episodes",
            *_diff_lines(obs, ent),
        ])

    # -- content ------------------------------------------------------------ #
    if exp_digest is None:
        rec["content_check"] = "COUNT-ONLY (manifest carries no uid digest)"
        print(f"[parity] {label}: {corpus_key} count OK ({len(obs)}/{n_exp}) — "
              f"NO uid digest committed for this split, so this is a COUNT-ONLY "
              f"check. Upgrade it once on a pod: "
              f"scripts/make_parity_manifest.py --record --split "
              f"{ent.get('split')} --cache-dir <verified cache>", flush=True)
        return rec

    exp_uids = ent.get("episode_uids")
    if mode == "strict":
        got = uid_digest(obs)
        if got != exp_digest:
            _refuse(label, corpus_key, cache_dir, [
                f"  episodes   : {len(obs)} loaded, {n_exp} expected  (count OK)",
                f"  uid sha256 : {got}  loaded",
                f"               {exp_digest}  expected   <-- MISMATCH",
                *_diff_lines(obs, ent),
            ])
        rec["content_check"] = "sha256(sorted uids) MATCHES the committed manifest"
        print(f"[parity] {label}: {corpus_key} VERIFIED — {len(obs)} episodes, "
              f"uid sha256 {got[:12]}… matches the committed manifest "
              f"(skip-hash {PARITY_SKIP_HASH}).", flush=True)
        return rec

    # subset: must be the sorted prefix of the manifest set
    if not exp_uids and len(obs) == n_exp:
        # A FULL set reached through subset mode is still a full set — verify the
        # digest. Without this a manifest recorded digest-only (``build_entry``
        # emits no uid list) would silently degrade every subset-mode caller to
        # count-only even while looking at the entire corpus.
        got = uid_digest(obs)
        if got != exp_digest:
            _refuse(label, corpus_key, cache_dir, [
                f"  episodes   : {len(obs)} loaded, {n_exp} expected  (count OK)",
                f"  uid sha256 : {got}  loaded",
                f"               {exp_digest}  expected   <-- MISMATCH",
                *_diff_lines(obs, ent),
            ])
        rec["content_check"] = ("sha256(sorted uids) MATCHES the committed "
                                "manifest (full set via subset mode)")
        print(f"[parity] {label}: {corpus_key} VERIFIED — {len(obs)} episodes, "
              f"uid sha256 {got[:12]}… matches the committed manifest.",
              flush=True)
        return rec
    if exp_uids:
        prefix = sorted(exp_uids)[:len(obs)]
        if obs != prefix:
            extra = sorted(set(obs) - set(exp_uids))
            _refuse(label, corpus_key, cache_dir, [
                f"  episodes   : {len(obs)} loaded (subset mode, ≤{n_exp} allowed)",
                "  the subset is NOT the canonical sorted prefix of the parity set",
                f"  foreign    : {_fmt(extra)}",
                f"  expected[0:{min(4, len(obs))}] : {_fmt(prefix[:4], 4)}",
            ])
    rec["content_check"] = ("subset prefix of the committed manifest"
                            if exp_uids else "COUNT-ONLY (no uid list committed)")
    if len(obs) < n_exp:
        tail = subset_note or (
            "This is admissible ONLY as a deliberate episode-subset run; it is "
            "NOT strict parity and must not be cross-compared with full-corpus "
            "arms.")
        print(f"[parity] ⚠ {label}: {corpus_key} SUBSET — {len(obs)} of {n_exp} "
              f"episodes ({n_exp - len(obs)} absent). {tail}", flush=True)
    else:
        print(f"[parity] {label}: {corpus_key} VERIFIED (full set via subset mode) "
              f"— {len(obs)} episodes.", flush=True)
    return rec


def _diff_lines(obs: Sequence[str], ent: dict) -> list[str]:
    exp_uids = ent.get("episode_uids")
    if not exp_uids:
        return ["  (manifest carries no uid list for this split — count only)"]
    missing = sorted(set(exp_uids) - set(obs))
    extra = sorted(set(obs) - set(exp_uids))
    return [f"  missing    : {_fmt(missing)}",
            f"  extra      : {_fmt(extra)}"]


# --------------------------------------------------------------------------- #
# 6. the trainer-facing entry points                                            #
# --------------------------------------------------------------------------- #
def assert_parity_corpus(cache_dir: str | Path, *, label: str,
                         require: bool = False, mode: str = "strict",
                         pattern: str = EPISODE_GLOB,
                         manifest_path: str | Path | None = None) -> dict:
    """THE trainer-facing guard. Call it with a split dir BEFORE any GPU work.

    * dir references a registered parity key -> full count + content check.
    * dir references a KNOWN-LEAKY split key -> always refuse.
    * dir references nothing registered      -> ``require=True`` refuses,
      otherwise one loud ``NON-PARITY`` line and ``parity=False``.
    """
    d = Path(cache_dir)
    key = corpus_key_of(d, manifest_path)
    if key in LEAKY_SPLIT_KEYS:
        _refuse(label, key, d, [
            "  LEAKED SPLIT — this directory is not a valid held-out set:",
            f"  {LEAKY_SPLIT_KEYS[key]}",
        ])
    if key is None:
        msg = (f"[parity] ⚠ NON-PARITY corpus for {label}: {d} references no "
               f"registered parity key ({PARITY_TRAIN_KEY} / {PARITY_VAL_KEY}). "
               f"Results off it are NOT cross-arm comparable with the parity arms."
               + _V2_HINT(d))
        if require:
            raise ParityViolation(
                f"PARITY VIOLATION: {label}={str(cache_dir)!r} does not reference "
                f"the canonical corpus {PARITY_TRAIN_KEY}. Any re-selected episode "
                f"set breaks cross-arm comparability and is refused "
                f"(CLAUDE.md §Invariants)." + _V2_HINT(d))
        print(msg, flush=True)
        return {"parity": False, "cache_dir": str(d), "corpus_key": None,
                "label": label}
    uids = scan_cache_dir(d, pattern)
    if not uids:
        _refuse(label, key, d, [
            f"  episodes   : 0 loaded — no {pattern} in this directory",
            "  does the path point at the SPLIT dir (…/physicalai-train-<key>) "
            "and not its parent?",
            *(ln for ln in _V2_HINT(d).split("\n") if ln),
        ])
    rec = check_uids(uids, corpus_key=key, label=label, cache_dir=d, mode=mode,
                     manifest_path=manifest_path)
    rec["parity"] = True
    rec["label"] = label
    rec["skip_markers_present"] = len(scan_skip_markers(d))
    return rec


def guard_split(root: str | Path, pattern: str, *, label: str,
                mode: str = "strict", require: bool = False,
                file_glob: str = EPISODE_GLOB,
                manifest_path: str | Path | None = None) -> tuple[Path, dict]:
    """``resolve_split_dir`` + :func:`assert_parity_corpus` — the form the
    ``*train*`` / ``*val*`` pattern loaders use. Cheap (a glob + one sha256 over
    ~2 400 short strings), so calling it again early in ``train()`` to fail
    before the model reaches the GPU costs nothing."""
    d = resolve_split_dir(root, pattern)
    return d, assert_parity_corpus(d, label=label, require=require, mode=mode,
                                   pattern=file_glob,
                                   manifest_path=manifest_path)


def assert_eids_parity(eids: Sequence[str], *, label: str,
                       corpus_key: str = PARITY_TRAIN_KEY,
                       mode: str = "strict",
                       manifest_path: str | Path | None = None) -> dict:
    """The guard for pipelines that never glob an epcache.

    ``train_flagship_v15`` / ``train_flagship_v16`` consume the ``eids`` list
    baked into the state / pose / label caches by ``v15_prep``. Those eids ARE
    the epcache basenames (``ep_00000.pt``), so the same manifest applies — and
    a label cache MINTED OVER A TRUNCATED EPCACHE carries a short eid list and
    used to train silently, with the three caches agreeing perfectly with each
    other the whole time."""
    rec = check_uids(list(eids), corpus_key=corpus_key, label=label,
                     cache_dir="<label-cache eids>", mode=mode,
                     manifest_path=manifest_path)
    rec["parity"] = True
    rec["label"] = label
    return rec


# --------------------------------------------------------------------------- #
# 7. the VAL side — the evaluator-facing guard                                  #
# --------------------------------------------------------------------------- #
# Wave-1 B closed the TRAIN side. The mirror image is worse: a truncated or
# substituted VAL cache does not void a training run, it produces a
# plausible-looking WRONG ADE — and nothing downstream can detect it, because
# every consumer (leaderboard, registry, gate) receives a number with no way to
# know how many episodes stood behind it.
#
# Two val-specific facts make the train-side guard insufficient on its own:
#
#  1. **The val corpus is DEPLOYED as subsets.** The full build under
#     ``<epcache>/physicalai-val-0c5f7dac3b11`` holds 600 episodes, but the
#     canonical TanitEval deployment on ``tanitad-eval:/root/valdata/`` holds
#     **40** (-> the 881 stride-8 windows every published number is quoted over;
#     MODEL_REGISTRY.md §0.3). A strict ``== 600`` check would refuse every eval
#     this program has ever run. The admissible counts are therefore REGISTERED
#     in the manifest (``known_deployments``) with their evidence, and anything
#     else — e.g. the 12-episode partial deployment that blocked a decision-grade
#     run on pod1 — is refused.
#  2. **``sorted(glob("*val*"))[-1]`` selects the LEAKY split.** Lexicographically
#     ``physicalai-val-0c5f7dac3b11`` < ``physicalai-val-f1b378f295ae``, so the
#     "newest dir wins" convention every evaluator inherited from the trainers
#     picks the 78.5 %-leaked split whenever both are materialised under one
#     root. :func:`resolve_val_dir` replaces it.
VAL_DIR_GLOB = "*val*"


def val_deployments(manifest_path: str | Path | None = None) -> list[dict]:
    """Registered admissible episode counts for the clean val split.

    Each entry is ``{n_episodes, role, where, evidence, evidence_class}``. The
    list is data, not code: adding a deployment is a manifest edit with a cited
    source, which is what keeps this from drifting into folklore."""
    ent = manifest_entry(PARITY_VAL_KEY, manifest_path) or {}
    return list(ent.get("known_deployments") or [])


def resolve_val_dir(root: str | Path, *, label: str,
                    pattern: str = VAL_DIR_GLOB, require_clean: bool = True,
                    manifest_path: str | Path | None = None) -> Path:
    """The val-split resolver — the fixed replacement for
    ``sorted(Path(root).glob("*val*"))[-1]``.

    Selection order:
      1. a dir referencing the registered CLEAN val key wins outright;
      2. otherwise, the newest dir that is not a known-leaky split;
      3. a root offering ONLY leaky splits is refused when ``require_clean``.

    Raises ``AssertionError`` (NOT :class:`ParityViolation`) when nothing matches
    — "you gave me no val dir" is not a parity violation, and several callers
    deliberately ``except AssertionError`` around an optional val block
    (WAVE1_B_REPORT.md §5)."""
    r = Path(root)
    dirs = sorted(d for d in r.glob(pattern) if d.is_dir())
    assert dirs, f"no cache dir matching {pattern} under {r}"
    clean = [d for d in dirs if corpus_key_of(d) == PARITY_VAL_KEY]
    if clean:
        chosen = clean[-1]
        leaky = [d for d in dirs if corpus_key_of(d) in LEAKY_SPLIT_KEYS]
        if leaky:
            print(f"[parity] {label}: {len(leaky)} known-LEAKY val dir(s) present "
                  f"under {r} ({', '.join(d.name for d in leaky)}) — SKIPPED. "
                  f"NOTE: the legacy `sorted(glob('*val*'))[-1]` convention would "
                  f"have selected {sorted(dirs)[-1].name}.", flush=True)
        return chosen
    nonleaky = [d for d in dirs if corpus_key_of(d) not in LEAKY_SPLIT_KEYS]
    if not nonleaky:
        _refuse(label, corpus_key_of(dirs[-1]) or "?", r, [
            "  LEAKED SPLIT — every *val* dir under this root is a known-leaky "
            "split:",
            f"  candidates : {', '.join(d.name for d in dirs)}",
            *(f"  {k}: {v}" for k, v in LEAKY_SPLIT_KEYS.items()),
        ])
    if require_clean and not any(corpus_key_of(d) for d in nonleaky):
        print(f"[parity] ⚠ {label}: no dir under {r} references the registered "
              f"clean val split {PARITY_VAL_KEY}; falling back to "
              f"{nonleaky[-1].name}. Results off it are NOT cross-arm comparable "
              f"with the parity arms.", flush=True)
    return nonleaky[-1]


def assert_val_cache(cache_dir: str | Path, *, label: str,
                     requested: int | None = None, decision_grade: bool = True,
                     pattern: str = EPISODE_GLOB,
                     manifest_path: str | Path | None = None) -> dict:
    """THE evaluator-facing guard. Call it BEFORE loading a single episode.

    Checks, in order:
      * known-**leaky** split -> always refused (never downgradable);
      * cache **absent or empty** -> loud line, ``checked=False``, NO refusal
        (the caller's own ``assert files`` owns that case, and turning it into a
        SystemExit would kill a finished run at its metrics write);
      * registered clean val key -> ``check_uids(mode='subset')`` (count bound +
        the uid digest **when the manifest carries one**, prefix-checked) PLUS
        the **deployment** check: the episode count must be a registered
        deployment (:func:`val_deployments`);
      * unregistered corpus (comma / cosmos / OOD) -> one ``NON-PARITY`` line and
        ``parity=False``; the requested-vs-delivered shortfall still prints;
      * ``requested`` episodes vs episodes actually on disk.

    ``decision_grade=False`` downgrades the deployment/shortfall refusals to loud
    warnings and stamps ``decision_grade: False`` into the returned record, so a
    deliberate partial-val probe stays possible and stays self-labelling.
    """
    d = Path(cache_dir)
    key = corpus_key_of(d)
    if key in LEAKY_SPLIT_KEYS:
        _refuse(label, key, d, [
            "  LEAKED SPLIT — this directory is not a valid held-out set:",
            f"  {LEAKY_SPLIT_KEYS[key]}",
            "  Every ADE, FDE, miss-rate or CI computed against it is "
            "train-contaminated.",
        ])
    uids = scan_cache_dir(d, pattern) if d.is_dir() else []
    if not uids:
        print(f"[parity] ⚠ {label}: no {pattern} under {d} — val integrity "
              f"NOT checked (nothing to check). The caller must refuse an empty "
              f"episode list itself.", flush=True)
        return {"checked": False, "parity": None, "cache_dir": str(d),
                "corpus_key": key, "label": label, "episodes_present": 0,
                "requested_episodes": requested,
                "decision_grade": bool(decision_grade)}

    n = len(uids)
    if key is None:
        print(f"[parity] ⚠ NON-PARITY val corpus for {label}: {d} references no "
              f"registered parity key ({PARITY_VAL_KEY}). {n} episodes present. "
              f"Results off it are NOT cross-arm comparable with the parity "
              f"arms.", flush=True)
        rec: dict = {"checked": True, "parity": False, "cache_dir": str(d),
                     "corpus_key": None, "episodes_present": n,
                     "content_check": "NONE (unregistered corpus)"}
    else:
        rec = check_uids(
            uids, corpus_key=key, label=label, cache_dir=d, mode="subset",
            manifest_path=manifest_path,
            subset_note=("a val DEPLOYMENT is a prefix of the full build; the "
                         "admissible counts are registered in the manifest's "
                         "known_deployments"))
        rec["parity"] = True
        rec["checked"] = True
        rec["episodes_present"] = n
        deps = val_deployments(manifest_path)
        ok_counts = {int(x["n_episodes"]) for x in deps}
        ok_counts.add(int(manifest_entry(key, manifest_path)["episode_count"]))
        rec["registered_deployments"] = sorted(ok_counts)
        rec["deployment"] = next(
            (x for x in deps if int(x["n_episodes"]) == n), None)
        if n not in ok_counts:
            lines = [
                f"  episodes   : {n} present — NOT a registered val deployment",
                f"  registered : {sorted(ok_counts)}",
                *[f"               {x['n_episodes']:>4}  {x['role']}"
                  f"  [{x.get('evidence_class', '?')}]" for x in deps],
                "  A val cache of an unregistered size is either TRUNCATED or a "
                "partial",
                "  deployment. Either way the ADE it produces is not the "
                "published statistic",
                "  and is not cross-arm comparable.",
                "  If this size is a legitimate NEW deployment, register it in "
                "parity_manifest.json",
                "  (corpora." + key + ".known_deployments) WITH its evidence, or "
                "pass",
                "  decision_grade=False for a deliberate, self-labelling partial "
                "probe.",
            ]
            if decision_grade:
                _refuse(label, key, d, lines)
            print(f"[parity] ⚠ {label}: UNREGISTERED val deployment ({n} "
                  f"episodes, registered {sorted(ok_counts)}) — allowed only "
                  f"because decision_grade=False. This number is NOT "
                  f"decision-grade.", flush=True)

    rec.update({"label": label, "requested_episodes": requested,
                "decision_grade": bool(decision_grade)})
    if requested is not None and n < int(requested):
        lines = [
            f"  episodes   : {n} present, {requested} requested"
            f"   <-- SHORT BY {int(requested) - n}",
            "  The evaluator asked for more val episodes than the cache holds "
            "and would have",
            "  silently scored the smaller set — the published number would "
            "name a window",
            "  count that never existed. Refusing.",
        ]
        if key is not None and decision_grade:
            _refuse(label, key, d, lines)
        print(f"[parity] ⚠ {label}: val cache holds {n} episodes but "
              f"{requested} were requested — the run will score {n}. "
              f"NOT the canonical statistic.", flush=True)
        rec["short_of_request"] = True
    if rec.get("parity"):
        print(f"[parity] {label}: {key} val OK — {n} episodes"
              + (f" ({rec['deployment']['role']})" if rec.get("deployment")
                 else "")
              + f", {rec.get('content_check', '?')}.", flush=True)
    return rec


def note_leaky_audit(cache_dir: str | Path, *, label: str, why: str) -> dict:
    """The ONE sanctioned way to touch a known-leaky split: a LABEL / LEAKAGE
    audit that produces no decision-grade number.

    ``route_label_audit`` / ``vlm_route_labels`` / ``vlm_kin_crossval`` /
    ``taniteval.label_overlay`` legitimately read ``f1b378`` — they score route
    LABELS, not a model, and the VLM pass-A artefacts were computed against that
    cache. What was never legitimate is that they did it SILENTLY: the split
    appeared as a default argument and nothing in the run announced it. This
    prints the disclosure and returns a record that is stamped
    ``decision_grade: False``, so an artefact carrying it can never be quoted as
    a held-out number."""
    key = corpus_key_of(cache_dir)
    if key in LEAKY_SPLIT_KEYS:
        print(f"[parity] ⚠ {label}: reading the KNOWN-LEAKY split {key} "
              f"({LEAKY_SPLIT_KEYS[key]}). Sanctioned here because: {why}. "
              f"NOTHING computed in this run is decision-grade.", flush=True)
    return {"checked": False, "corpus_key": key, "cache_dir": str(cache_dir),
            "leaky": key in LEAKY_SPLIT_KEYS, "decision_grade": False,
            "label": label, "audit_reason": why}


def guard_val_split(root: str | Path, *, label: str,
                    requested: int | None = None, decision_grade: bool = True,
                    pattern: str = VAL_DIR_GLOB, file_glob: str = EPISODE_GLOB,
                    manifest_path: str | Path | None = None
                    ) -> tuple[Path, dict]:
    """:func:`resolve_val_dir` + :func:`assert_val_cache` — the form the
    ``sorted(glob("*val*"))[-1]`` evaluators take."""
    d = resolve_val_dir(root, label=label, pattern=pattern,
                        manifest_path=manifest_path)
    return d, assert_val_cache(d, label=label, requested=requested,
                               decision_grade=decision_grade,
                               pattern=file_glob, manifest_path=manifest_path)


# --------------------------------------------------------------------------- #
# 8. GEOMETRY SIBLINGS — a re-cropped corpus of the SAME episodes               #
# --------------------------------------------------------------------------- #
# MEASURED 2026-07-27 (…/incoming/2026-07-27-geometry-configurable/
# selection_verdict_2026-07-27.json): changing the crop/resolution is a
# RE-CACHE, not a re-selection. The ordered source list comes from
# ``discover_r0_clips`` + ``split_clips``, neither of which takes any geometry
# argument; the episode uid is its POSITION in that list; and a build failure
# writes ``skip_%05d`` at the same index regardless of geometry. Only
# ``epcache.cache_key``'s ``params`` moves — so the DIRECTORY NAME changes while
# the episode set does not.
#
# ⚠️ THE OPERATIONAL CONSEQUENCE, which "it is only a re-cache" hides:
# ``corpus_key_of`` substring-matches REGISTERED keys, so a re-cropped cache
# reads as NON-PARITY and every ``require=True`` caller (``train_flagship_v4``)
# REFUSES it. A geometry change is therefore blocked until the new key is
# registered — and the ONLY safe registration is one that PROVES the episode set
# did not move. That is what this function is: it copies the parity entry under
# a new key and refuses if the observed uid digest, count or skip set differ.


def register_geometry_sibling(cache_dir: str | Path, *, new_key: str,
                              geometry: dict, source_key: str = PARITY_TRAIN_KEY,
                              manifest: dict | None = None) -> dict:
    """Build a manifest entry for a RE-CROPPED build of the parity episode set.

    ``cache_dir`` is the freshly built split dir. The entry is minted ONLY if its
    uid set, count and skip indices match the ``source_key`` entry exactly — i.e.
    only if the rebuild really was a re-cache. Any difference means episodes were
    re-selected, added or lost, and that is refused: cross-arm comparability is
    the thing the parity key exists to protect, and a geometry change is allowed
    to move pixels, never membership.

    Returns the new entry (the caller writes it into ``parity_manifest.json`` and
    commits the diff). It records ``geometry`` and ``derived_from`` so the
    sibling can never be mistaken for an independent corpus.
    """
    src = manifest_entry(source_key, None if manifest is None else None)
    if src is None:
        raise ParityViolation(
            f"cannot register a geometry sibling of {source_key!r}: no manifest "
            f"entry for it.")
    obs = scan_cache_dir(cache_dir)
    skips = scan_skip_markers(cache_dir)
    problems = []
    if len(obs) != int(src["episode_count"]):
        problems.append(f"episode count {len(obs)} != {src['episode_count']}")
    if src.get("episode_uid_sha256") and obs:
        got = uid_digest(obs)
        if got != src["episode_uid_sha256"]:
            missing = sorted(set(src.get("episode_uids") or []) - set(obs))
            extra = sorted(set(obs) - set(src.get("episode_uids") or []))
            problems.append(f"uid sha256 {got} != {src['episode_uid_sha256']}")
            if missing or extra:
                problems.append(f"missing {_fmt(missing)} / extra {_fmt(extra)}")
    if sorted(skips) != sorted(int(i) for i in src.get("skip_indices", [])):
        problems.append(f"skip indices {skips} != {src.get('skip_indices')}")
    if problems:
        _refuse("geometry sibling", new_key, cache_dir, [
            "  A re-cropped corpus must contain EXACTLY the parity episode set —",
            "  geometry may change PIXELS, never MEMBERSHIP.",
            *(f"  {p}" for p in problems),
            f"  source     : {source_key}",
        ])
    ent = build_entry(obs, corpus_key=new_key, split=src.get("split", "train"),
                      skip_indices=skips,
                      uid_source=f"re-cache of {source_key} at a new geometry",
                      provenance={
                          "derived_from": source_key,
                          "relation": "geometry sibling — identical episode "
                                      "selection, different canonical frame",
                          "geometry": dict(geometry),
                          "verified": "uid digest, episode count and skip "
                                      "indices matched the source entry exactly",
                      })
    print(f"[parity] geometry sibling VERIFIED: {new_key} holds the SAME "
          f"{len(obs)} episodes and the SAME {len(skips)} skips as "
          f"{source_key} (uid sha256 {ent['episode_uid_sha256'][:12]}…). "
          f"Selection parity is preserved; only the pixels differ.", flush=True)
    return ent


# --------------------------------------------------------------------------- #
# 9. V2 COMPRESSED SIBLINGS — the clip-id uid space                             #
# --------------------------------------------------------------------------- #
# THE HOLE THIS CLOSES (MEASURED by the wide-FOV build stream, 2026-07-27,
# …/incoming/2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md §6):
#
#   ``train_flagship_v4`` has NO v2 support at all, while ``train_flagship4b
#   --v2-cache`` reads v2 and applied NO PARITY CHECK on that branch — its guard
#   lives in ``_cache_split``, which only the ``--cache-dirs`` branch calls. The
#   v5 wide cache was therefore trainable with ZERO parity enforcement.
#
# It is not an oversight that §8's :func:`register_geometry_sibling` could not
# fix it: that function compares ``sha256(sorted ep_*.pt basenames)``, and a v2
# cache HAS no ``ep_*.pt``. The two uid spaces are:
#
#   epcache : ``ep_%05d.pt``          — identity = POSITION in the ordered clip
#                                       list; a build failure leaves ``skip_%05d``
#                                       at the same index, so the positions are
#                                       dense and self-describing.
#   v2      : ``<clip_id>.v2ep.pt``   — identity = the CLIP ID; the directory is
#                                       a flat SET with no positions at all, and
#                                       a decode failure leaves NO marker — the
#                                       file is simply absent.
#
# ⚠️ The re-cache had to be v2: the raw epcache at 120°/256×640 is 293.4 MB per
# episode = ~697 GB for the train split alone (MEASURED, WIDE_FOV_BUILD.md §3)
# and fits on no host in the fleet, against ~95 GB for the v2 PNG build.
#
# WHAT THIS SECTION PROVES, AND WHAT IT CANNOT
# --------------------------------------------
# It proves MEMBERSHIP: the set of clip ids built equals, exactly, the parity
# train split. ⚠️ A COUNT CANNOT DO THIS — drop one clip and add one foreign clip
# and the count is unchanged, which is precisely why the wide-FOV census
# recomputed the corpus key instead of counting clips. Both directions are
# pinned in ``tests/test_v2_parity.py`` (a swapped clip at IDENTICAL count is
# refused, with the reason asserted).
#
# It does NOT prove:
#   * that the PIXELS are right — the geometry is recorded (``_geometry.json``,
#     the entry's ``geometry`` block) and asserted pre-decode by the builder, not
#     re-derived here. A cache built at the wrong FOV with the right clips passes
#     membership and is caught by the geometry record, not by this check;
#   * WHICH clips are missing, in digest-only mode (no ``expect_clips``) — see
#     :func:`verify_v2_membership`;
#   * anything about ORDER. The v2 directory is a set. That costs nothing HERE
#     because for the parity train split the ordered and sorted clip-id digests
#     are IDENTICAL (MEASURED, ``parity_split_meta_2026-07-27.json``: the
#     discovered order already is clip-id order), so the set proof is exactly as
#     strong as an ordered one — but that is a property of this corpus, not a
#     general one, and :func:`verify_v2_membership` records it per run.
#
# 🔒 CONFIDENTIALITY. Clip ids are gated-confidential PhysicalAI-AV content and
# must never leave a pod. Every refusal in this section therefore prints COUNTS
# and DIGESTS only — never an id — which is why it cannot reuse
# :func:`_diff_lines` (that prints uids, safe for positional ``ep_*.pt``, not for
# clip ids). The repo carries only the digests.


def v2_clip_ids(cache_dirs) -> list[str]:
    """Sorted clip ids physically present across one or more v2 cache dirs.

    ``--v2-cache`` is ``nargs="+"`` and the dirs are CONCATENATED by
    ``build_v2_providers``, so the training set is their UNION and the union is
    what must be checked. A clip present in two dirs would be trained on twice
    and is refused here, not silently duplicated."""
    if isinstance(cache_dirs, (str, Path)):
        cache_dirs = [cache_dirs]
    seen: dict[str, list[str]] = {}
    for cd in cache_dirs:
        for p in Path(cd).glob(V2_EPISODE_GLOB):
            seen.setdefault(p.name[:-len(V2_SUFFIX)], []).append(str(cd))
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        raise ParityViolation(
            f"PARITY VIOLATION [v2-cache]: {len(dupes)} clip(s) appear in more "
            f"than one --v2-cache dir. build_v2_providers CONCATENATES the dirs, "
            f"so each duplicate would contribute its windows TWICE and re-weight "
            f"the corpus. Dirs: {sorted({d for v in dupes.values() for d in v})}. "
            f"(clip ids withheld — gated-confidential)")
    return sorted(seen)


def _v2_paths(cache_dirs) -> list[str]:
    if isinstance(cache_dirs, (str, Path)):
        cache_dirs = [cache_dirs]
    return [str(c) for c in cache_dirs]


def _refuse_v2(label: str, key: str, cache_dirs, lines: list[str]) -> None:
    """A v2 refusal. Same shape as :func:`_refuse`, different remediation — and
    🔒 NO ids, ever."""
    raise ParityViolation(
        "\n".join([
            "",
            "=" * 78,
            f"PARITY VIOLATION [{label}] — v2 corpus {key}",
            "=" * 78,
            *(f"  cache      : {c}" for c in _v2_paths(cache_dirs)),
            *lines,
            "",
            "  The canonical corpus is SACRED (CLAUDE.md §Invariants). Geometry may",
            "  change PIXELS, never MEMBERSHIP: a v2 re-cache that drops, adds or",
            "  substitutes a clip is a RE-SELECTION, and every cross-arm number off it",
            "  is void — invisibly, because a wrong cache trains perfectly happily.",
            "",
            "  🔒 clip ids are gated-confidential and are NOT printed. Counts and",
            "  digests above are the whole of the evidence that may leave a pod.",
            "",
            "  Prove membership on the pod (writes the raw proof JSON):",
            "    scripts/register_v2_sibling.py --verify-only --cache <dir> \\",
            "        --expect-clips <parity_train_clips.txt> --out <proof.json>",
            "  Then register + stage the manifest diff:",
            "    scripts/register_v2_sibling.py --cache <dir> --new-key <key> \\",
            "        --expect-clips <parity_train_clips.txt> --write-manifest",
            "=" * 78,
        ]))


def _sibling_candidate_key(cache_dirs) -> str | None:
    """The key a v2 GEOMETRY SIBLING would have been registered under, read off
    the directory name — or ``None`` when the name looks like nothing.

    ⚠️ WHY THIS EXISTS. The v5 runbook is *rebuild -> register -> COMMIT THE
    MANIFEST -> train*, and step 3 is the one that gets skipped: the registration
    runs on a pod, the ``parity_manifest.json`` diff is never staged, and on every
    OTHER host the cache then reads as an unregistered (or, worse, a
    parent-keyed) corpus and ``--require-parity`` refuses to start. The refusal
    was accurate and useless — it said the corpus was not registered, not that a
    specific entry was MISSING from a specific file.

    The rule is deliberately dumb and cannot mis-resolve anything: the directory
    BASENAME is the candidate. ``register_v2_geometry_sibling`` already refuses a
    key that does not appear in the cache path, and ``register_v2_sibling.py``'s
    published runbook renames the dir TO the key, so basename == key by
    construction on the intended path. Nothing branches on the result — it only
    adds lines to a refusal that has already been decided."""
    for d in _v2_paths(cache_dirs):
        name = Path(d).resolve().name
        if name and name not in (".", "/"):
            return name
    return None


def _missing_entry_lines(cache_dirs, matched_key: str | None,
                         manifest_path: str | Path | None = None) -> list[str]:
    """Refusal lines that NAME the manifest entry that is missing, when the
    directory looks like a geometry sibling whose registration never landed."""
    cand = _sibling_candidate_key(cache_dirs)
    mpath = Path(manifest_path or MANIFEST_PATH)
    if not cand:
        return []
    try:
        known = set(load_manifest(manifest_path).get("corpora", {}))
    except SystemExit:                                        # firewall direction
        known = set()
    if cand in known:                                         # nothing is missing
        return []
    parent = (f" (it EXTENDS the registered key {matched_key!r}, which is a RAW "
              f"epcache corpus)" if matched_key and matched_key in cand
              and matched_key != cand else "")
    return [
        "",
        f"  🔴 MISSING MANIFEST ENTRY: {cand!r}{parent}",
        f"     is NOT in {mpath}",
        "     — so this host cannot verify the cache even though the pod that",
        "       built it may have registered it perfectly.",
        "",
        "     This is RUNBOOK STEP 3, the one that gets forgotten:",
        "         git add stack/tanitad/data/parity_manifest.json",
        "     An entry that lives only on the pod makes the cache read",
        "     NON-PARITY on every other host, and --require-parity then refuses",
        "     to start — which is what just happened.",
    ]


def assert_v2_splits_disjoint(train_dirs, val_dirs, *,
                              label: str = "v2 train/val") -> dict:
    """⭐ REFUSE a v2 train cache and v2 val cache that share a clip.

    A DIFFERENT fact from :func:`assert_v2_parity_cache`, which proves each
    directory's membership against the manifest and never compares two of them.
    On the raw path the two splits are separate REGISTERED corpora, so an overlap
    is caught by the digests; on the v2 path they are just two paths a launch
    command supplies, and nothing looked at them together.

    ⚠️ This is the guard the HELD-OUT GATE depends on. A leaked val clip does not
    crash anything — it makes the gate's early-stop read a training episode, so
    the probe reports health while the deployable surface decays. That is
    precisely the ~29.5 GPU-h failure the gate exists to prevent, wearing a
    working gate as a disguise.

    🔒 Counts and digests only — clip ids are gated-confidential."""
    tr = set(v2_clip_ids(train_dirs))
    va = set(v2_clip_ids(val_dirs))
    both = tr & va
    if both:
        raise ParityViolation("\n".join([
            "",
            "=" * 78,
            f"PARITY VIOLATION [{label}] — TRAIN/VAL LEAK",
            "=" * 78,
            *(f"  train      : {d}" for d in _v2_paths(train_dirs)),
            *(f"  val        : {d}" for d in _v2_paths(val_dirs)),
            f"  overlap    : {len(both)} clip(s) appear in BOTH   <-- LEAK",
            f"               ({len(tr)} train, {len(va)} val)",
            "",
            "  The held-out gate probes the val clips to decide when to STOP the",
            "  run. A clip that is also trained on makes that probe report health",
            "  while the deployable surface decays — an early-stop that cannot",
            "  fire is worse than none, because it is believed.",
            "",
            "  🔒 clip ids are gated-confidential and are NOT printed.",
            "=" * 78,
        ]))
    return {"disjoint": True, "train_clips": len(tr), "val_clips": len(va),
            "overlap": 0, "label": label}


def assert_v2_geometry_matches(rec: dict, frame, *, label: str,
                               providers=None, parent=None) -> dict:
    """⭐ BIND THE GEOMETRY to what the trainer verified — the gap §9's header
    names ("nothing hashes PIXELS; a wrong-FOV cache with the right clips
    PASSES").

    Two bindings, and they are NOT equally strong — stated plainly because the
    difference is the whole point:

    1. **SHAPE, read off the cache** (strong). ``providers`` are the objects
       :func:`tanitad.data.v2_dataset.build_v2_providers` returns; their
       ``.frames.shape`` comes from the per-clip ``image_h``/``image_w`` written
       into each payload at BUILD time. Comparing it to the trainer's resolved
       frame catches the exact WIDE_FOV_BUILD.md §7 failure — "omit the flags and
       the trainer builds a 256x256 encoder and is fed 256x640 frames" — and it
       is a property of the bytes on disk, not of a declaration.
    2. **DECLARATION, from the manifest** (weaker, still worth having). The
       registered entry carries the builder's ``_geometry.json``
       (``provenance.geometry``). Comparing ``f_ref`` / ``projection`` to the
       run's frame makes the registration and the run agree on the field of
       view, which SHAPE alone cannot see: 256x640 at f_ref 305.58 (120 deg) and
       256x640 at f_ref 407 (90 deg) have identical shape.

    ⭐ ``parent`` (2026-07-28) is the SUB-FRAME case: the cache is built at
    ``parent`` and the trainer reads a CENTRED SLICE of it (``frame``), via
    ``build_v2_providers(..., frame=…)``. Then the two bindings split cleanly and
    BOTH get stronger:

    * SHAPE is checked against ``frame`` — the raster the trainer actually
      receives. ⭐ This is what makes the slice IMPOSSIBLE TO FORGET: a run that
      declares a sub-frame but whose loader was never told to slice hands back
      ``parent``-shaped providers and this raises. The wiring cannot be
      configured-but-inert, which is the exact defect class this check is here
      for.
    * DECLARATION is checked against ``parent`` — what the builder recorded and
      what is registered. It stays a true statement about the bytes on disk.
    * and ``frame`` must be a real centred slice of ``parent``
      (``calib.subframe_slice``, which refuses a changed focal or projection),
      recorded as ``sliced_from`` with the exact rows/cols.

    ⛔ WHAT THIS STILL DOES NOT DO: it does not hash pixels. A cache whose
    ``_geometry.json`` says 120 deg but whose resampler actually produced 90 deg
    passes both bindings. Only the builder's pre-decode
    ``_assert_geometry_deliverable`` binds the record to the resampler, it runs
    in a different stream hours earlier, and if it is wrong nothing here catches
    it. Said in the code so it cannot be softened in prose."""
    out = {"label": label, "checked_shape": False, "checked_declaration": False,
           "frame": {"height": int(frame.height), "width": int(frame.width),
                     "f_ref": float(frame.f_ref),
                     "projection": str(frame.projection)},
           "pixels_are_not_hashed": (
               "membership proves WHICH CLIPS, shape proves the RASTER SIZE on "
               "disk, the manifest proves what the BUILDER RECORDED. None of "
               "the three proves the resampler produced that field.")}

    declared = frame
    if parent is not None and parent != frame:
        from tanitad.data.calib import subframe_slice
        rs, cs = subframe_slice(parent, frame)     # refuses a non-slice
        declared = parent
        out["sliced_from"] = {
            "parent": {"height": int(parent.height), "width": int(parent.width),
                       "f_ref": float(parent.f_ref),
                       "projection": str(parent.projection)},
            "parent_tag": parent.tag(), "sub_tag": frame.tag(),
            "rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop],
            "note": ("the cache holds the parent; the loader slices it. SHAPE is "
                     "bound to the SUB-frame (so a slice that was configured but "
                     "never applied FAILS here), DECLARATION to the PARENT."),
        }

    if providers:
        shapes = sorted({tuple(int(x) for x in p.frames.shape[-2:])
                         for p in providers})
        out["checked_shape"] = True
        out["cache_frame_shapes"] = [list(s) for s in shapes]
        want = (int(frame.height), int(frame.width))
        if len(shapes) > 1 or shapes[0] != want:
            # ⭐ THE SUB-FRAME CASE: shapes came back as the PARENT, i.e. the
            # sub-frame was declared and the LOADER WAS NEVER TOLD TO SLICE.
            # That is the "verified fix nothing calls" failure, and naming it
            # here is what turns it from a silent wrong-geometry run into a
            # refusal at launch.
            inert = (parent is not None and len(shapes) == 1
                     and shapes[0] == (int(parent.height), int(parent.width)))
            raise ParityViolation("\n".join([
                "",
                "=" * 78,
                f"GEOMETRY VIOLATION [{label}] — "
                + ("the SUB-FRAME WAS DECLARED BUT NEVER APPLIED" if inert else
                   "the cache is not the frame the run declares"),
                "=" * 78,
                f"  run declares : {want[0]}x{want[1]} px, f_ref "
                f"{frame.f_ref:.4f}, {frame.projection}",
                f"  cache holds  : {', '.join(f'{h}x{w}' for h, w in shapes)} px"
                + ("   <-- MIXED GEOMETRIES IN ONE CACHE" if len(shapes) > 1
                   else "   <-- MISMATCH"),
                "",
                *(["  The providers handed back the PARENT raster, so the frame",
                   "  argument never reached the loader: build_v2_providers() was",
                   "  called WITHOUT frame=, or something rebuilt the providers",
                   "  after the geometry was resolved. The run would have trained",
                   "  on the un-sliced frames while its config.json claimed the",
                   f"  sub-frame — pass frame={want[0]}x{want[1]} into",
                   "  tanitad.data.v2_dataset.build_v2_providers().",
                   ""] if inert else []),
                "  The encoder's positional embedding is sized for the DECLARED",
                "  frame. Feeding it a different raster does not crash: it",
                "  trains, and every number off the run is void.",
                "",
                *([] if inert else [
                    "  Pass the cache's own geometry, e.g.:",
                    f"      --frame-h {shapes[0][0]} --frame-w {shapes[0][1]} "
                    f"--frame-hfov <deg> --projection <pinhole|cylindrical>"]),
                "=" * 78,
            ]))

    # The registered block is whatever `register_v2_sibling.py` read out of the
    # cache's `_geometry.json`: `{"frame": {...}, "frame_tag": ..., ...}` on the
    # deployed path, but a FLAT `{"height":…, "width":…}` is equally valid and is
    # what a hand-passed --geometry-json can produce. Accept both rather than
    # silently skipping the check on one of them — a binding that quietly does
    # nothing is the C13 failure this whole function exists to avoid.
    geo = (rec or {}).get("geometry") or {}
    fr = None
    if isinstance(geo, dict):
        fr = geo.get("frame") if isinstance(geo.get("frame"), dict) else geo
    if isinstance(fr, dict) and {"height", "width"} <= set(fr):
        out["checked_declaration"] = True
        out["registered_geometry"] = dict(fr)
        bad = []
        if int(fr["height"]) != int(declared.height) or \
                int(fr["width"]) != int(declared.width):
            bad.append(f"size {fr['height']}x{fr['width']} registered vs "
                       f"{declared.height}x{declared.width} declared"
                       + ("  (the CACHE geometry; the run reads the sub-frame "
                          f"{frame.height}x{frame.width})"
                          if declared is not frame else ""))
        if fr.get("projection") and str(fr["projection"]) != str(declared.projection):
            bad.append(f"projection {fr['projection']!r} registered vs "
                       f"{declared.projection!r} declared")
        if fr.get("f_ref") is not None and \
                abs(float(fr["f_ref"]) - float(declared.f_ref)) > 1e-3:
            bad.append(f"f_ref {float(fr['f_ref']):.4f} registered vs "
                       f"{float(declared.f_ref):.4f} declared  <-- SAME PIXELS, "
                       f"DIFFERENT FIELD OF VIEW")
        if bad:
            raise ParityViolation("\n".join([
                "",
                "=" * 78,
                f"GEOMETRY VIOLATION [{label}] — run vs REGISTERED geometry",
                "=" * 78,
                f"  corpus key : {(rec or {}).get('corpus_key')}",
                *(f"  {b}" for b in bad),
                "",
                "  The manifest entry records the geometry the cache was BUILT",
                "  and REGISTERED at (its _geometry.json). A run that declares a",
                "  different frame is reading those pixels through the wrong",
                "  camera model — membership passes, every metric is void.",
                "=" * 78,
            ]))
    else:
        out["declaration_note"] = (
            "the registered entry carries no geometry.frame block (an "
            "unregistered or pre-geometry corpus) — only the SHAPE binding "
            "applies here")
    return out


def assert_eval_frame_matches_run(run_cfg: dict | None, frame, *, label: str,
                                  cache_frame=None,
                                  flag: str = "--v2-subframe") -> dict:
    """⭐ SCORE THE CHECKPOINT ON THE FRAME IT WAS TRAINED ON — or refuse.

    :func:`assert_v2_geometry_matches` binds the run's declared frame to the
    CACHE. This binds it to the **CHECKPOINT**, and it is a different failure:

        the trainer slices to ``176x624`` and trains 30k steps there; the
        evaluator is handed the same cache, forgets ``--v2-subframe``, reads
        ``256x640`` and publishes an ADE.

    That is the ``ego=`` bug's exact shape — *trained with a capability, scored
    without it* — and it is invisible in every artifact: the cache is right, the
    membership is right, the checkpoint loads, the number is wrong. It is also
    reachable by pure OMISSION, which is why the guard is here and not in a
    runbook.

    ``run_cfg`` is the run's own ``config.json`` (already parsed). The trainer
    writes two geometry blocks and they answer different questions:

    * ``geometry``       — the frame the MODEL saw (``geometry_report(cfg)``,
      i.e. post-``apply_frame``, so it IS the sub-frame on a sliced run);
    * ``geometry_cache`` — the frame the BYTES are, present only when the run
      sliced (``None`` when the model frame and the cache frame are the same).

    ⛔ A checkpoint with no geometry record is NOT silently passed and NOT
    refused: it is reported ``checked=False`` with a loud line, because every
    pre-2026-07-27 arm is in that state and refusing would make the historical
    record unreproducible. The caller decides; the evaluator prints it.

    Returns the check record for the output JSON.
    """
    out = {"label": label, "checked": False,
           "eval_frame": {"height": int(frame.height), "width": int(frame.width),
                          "f_ref": float(frame.f_ref),
                          "projection": str(frame.projection)}}
    geo = (run_cfg or {}).get("geometry")
    if not isinstance(geo, dict) or not {"height", "width"} <= set(geo):
        out["note"] = (
            "the checkpoint's config.json carries no `geometry` block (a "
            "pre-2026-07-27 run, or no sibling config.json was found). The "
            "frame this checkpoint was TRAINED on is therefore UNVERIFIED — "
            "the eval frame above is an assumption, not a match.")
        return out

    from tanitad.data.calib import CanonicalFrame
    trained = CanonicalFrame.from_dict(geo)
    out["checked"] = True
    out["trained_frame"] = trained.to_dict()
    out["trained_frame_tag"] = trained.tag()
    tcache = (run_cfg or {}).get("geometry_cache")
    if isinstance(tcache, dict) and {"height", "width"} <= set(tcache):
        out["trained_cache_frame"] = CanonicalFrame.from_dict(tcache).to_dict()

    if trained != frame:
        sub = f"{trained.height}x{trained.width}"
        # The single most useful line: WHICH flag value reproduces the run.
        fix = ([f"      {flag} {sub}"] if isinstance(tcache, dict) else
               [f"      --frame-h {trained.height} --frame-w {trained.width} "
                f"--projection {trained.projection}"])
        raise ParityViolation("\n".join([
            "",
            "=" * 78,
            f"FRAME VIOLATION [{label}] — SCORING A CHECKPOINT ON A FRAME IT "
            f"WAS NOT TRAINED ON",
            "=" * 78,
            f"  trained on : {trained.height}x{trained.width} px, f_ref "
            f"{trained.f_ref:.4f}, {trained.projection}   ({trained.tag()})",
            f"  eval reads : {frame.height}x{frame.width} px, f_ref "
            f"{frame.f_ref:.4f}, {frame.projection}   ({frame.tag()})",
            "",
            "  This is the `ego=` failure in geometry: the capability was",
            "  present in training and absent at scoring. It does not crash —",
            "  the encoder's positional embedding is sized from this frame, so",
            "  a mismatch either fails the STRICT load with a shape error whose",
            "  cause is three files away, or (same token count, different",
            "  field) produces a plausible ADE off the wrong pixels.",
            "",
            "  Reproduce the run's frame:",
            *fix,
            "",
            "  (read from the checkpoint's own config.json `geometry` block —",
            f"  `geometry_cache` is {'set' if isinstance(tcache, dict) else 'null'}, "
            f"so the run "
            + ("SLICED a larger cache." if isinstance(tcache, dict)
               else "read the cache unsliced."),
            "=" * 78,
        ]))

    if cache_frame is not None and isinstance(tcache, dict):
        tc = CanonicalFrame.from_dict(tcache)
        if tc != cache_frame:
            raise ParityViolation("\n".join([
                "",
                "=" * 78,
                f"FRAME VIOLATION [{label}] — the run sliced a DIFFERENT cache",
                "=" * 78,
                f"  run's cache : {tc.height}x{tc.width}, f_ref {tc.f_ref:.4f}, "
                f"{tc.projection}   ({tc.tag()})",
                f"  eval cache  : {cache_frame.height}x{cache_frame.width}, "
                f"f_ref {cache_frame.f_ref:.4f}, {cache_frame.projection}   "
                f"({cache_frame.tag()})",
                "",
                "  The model frame agrees, so the encoder loads and the number",
                "  looks fine — but the same-size slice was taken out of a",
                "  different field. Pass the geometry the RUN's cache was built",
                "  at (--frame-h/--frame-w/--frame-hfov/--projection).",
                "=" * 78,
            ]))
        out["cache_frame_matches_run"] = True
    print(f"[parity] {label}: frame MATCHES the checkpoint's own config.json — "
          f"trained and scored at {trained.height}x{trained.width} "
          f"({trained.tag()}).", flush=True)
    return out


def clip_membership_of(corpus_key: str = PARITY_TRAIN_KEY,
                       manifest_path: str | Path | None = None) -> dict | None:
    """The CLIP-ID membership record of a raw-epcache corpus entry, or ``None``.

    Distinct from ``episode_count`` on purpose, and the distinction has already
    corrected one brief: the parity train split is **2 400 CLIPS**, of which 24
    fail to decode, leaving **2 376 EPISODES**. ``episode_count`` is 2 376;
    ``clip_membership.n_clips`` is 2 400. A check written against 2 400-as-total
    or against 2 376-as-clips is wrong in opposite directions."""
    ent = manifest_entry(corpus_key, manifest_path) or {}
    cm = ent.get("clip_membership")
    return dict(cm) if cm else None


def verify_v2_membership(cache_dirs, *, label: str = "v2-membership",
                         source_key: str = PARITY_TRAIN_KEY,
                         expect_clips: str | Path | None = None,
                         manifest_path: str | Path | None = None) -> dict:
    """PROVE that a v2 cache holds exactly the ``source_key`` clip split.

    Adapted from ``…/2026-07-28-wide-fov-build/code/verify_v2_parity.py`` (the
    sibling stream's membership proof, whose pass criteria were fixed BEFORE the
    number existed), with three changes, all strengthenings:

    1. **The expected set is bound to the COMMITTED manifest**, not only to the
       export's own sidecar. The original compared the built digest against
       ``train_ids_sha256_sorted`` inside the same ``parity_split_meta.json``
       that shipped beside the clip list — so a self-consistent WRONG pair would
       have verified. Here the supplied list must first reproduce the manifest's
       ``clip_membership.clip_id_sha256_sorted`` before it is trusted as the
       expectation.
    2. **The 24 decode failures are checked by IDENTITY, not by count.** The
       original accepted any ``len(missing) == 24``. The skip indices are
       positions in the ordered clip list, and that list is exactly what
       ``--expect-clips`` supplies, so the *same 24 clips must fail again* — the
       check WIDE_FOV_BUILD.md §8 calls "a strong independent check" but tested
       only by cardinality. A shortfall of 24 made of a DIFFERENT 24 is a
       different episode set and is refused. (Falls back to the count test, and
       says so in the record, if the entry carries no skip indices.)
    3. **It refuses instead of returning a verdict string.** The original wrote
       ``VERDICT: NOT VERIFIED`` into JSON and exited 0 unless there were EXTRA
       clips, so a truncated build did not fail the command.

    ``expect_clips`` is the ordered clip-id list exported from a parity host by
    ``parity_split_export.py`` (which refuses to write unless the host
    reproduces both corpus keys). 🔒 It is gated-confidential and lives only on
    pods. **Without it this degrades to DIGEST-ONLY**: a COMPLETE build still
    proves membership exactly, but an incomplete one can only be refused, not
    diagnosed — the check cannot name what is missing without the list.
    """
    dirs = _v2_paths(cache_dirs)
    built = v2_clip_ids(dirs)
    cm = clip_membership_of(source_key, manifest_path)
    if cm is None:
        _refuse_v2(label, source_key, dirs, [
            f"  manifest   : NO clip_membership block for {source_key!r} in "
            f"{Path(manifest_path or MANIFEST_PATH)}",
            "  A v2 cache is a set of CLIP IDS; without the corpus's committed",
            "  clip-id digest there is nothing to compare it to.",
        ])
    n_exp = int(cm["n_clips"])
    exp_digest = str(cm["clip_id_sha256_sorted"])
    ent = manifest_entry(source_key, manifest_path) or {}
    skip_idx = sorted(int(i) for i in ent.get("skip_indices", []))
    if not built:
        _refuse_v2(label, source_key, dirs, [
            f"  clips      : 0 built — no {V2_EPISODE_GLOB} in these dir(s)",
            "  does the path point at the v2 cache dir itself, not its parent?",
        ])
    got_digest = uid_digest(built)
    rec: dict = {
        "label": label, "cache_dirs": dirs, "source_corpus_key": source_key,
        "uid_kind": V2_UID_KIND,
        "clips_expected": n_exp, "clips_built": len(built),
        "clip_id_sha256_sorted": got_digest,
        "clip_id_sha256_sorted_expected": exp_digest,
        "membership_identical": got_digest == exp_digest,
        "expected_decode_failures": len(skip_idx),
        "order_independent": True,
        "ordered_equals_sorted_for_this_corpus": bool(
            cm.get("ordered_equals_sorted")),
        "expect_clips": str(expect_clips) if expect_clips else None,
    }

    # -- extras are ALWAYS fatal, in either mode ---------------------------- #
    if len(built) > n_exp:
        rec["mode"] = "digest-only" if expect_clips is None else "set-diff"
        _refuse_v2(label, source_key, dirs, [
            f"  clips      : {len(built)} built, {n_exp} in the parity split"
            f"   <-- {len(built) - n_exp} FOREIGN clip(s)",
            f"  clip sha256: {got_digest}  built",
            f"               {exp_digest}  parity split   <-- MISMATCH",
            "  A v2 cache may never hold a clip the parity split does not.",
        ])

    # -- mode A: the exported clip list is available (the strong proof) ------ #
    if expect_clips is not None:
        expect = [ln.strip() for ln in
                  Path(expect_clips).read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
        rec["mode"] = "set-diff (expect_clips supplied)"
        rec["expect_clips_count"] = len(expect)
        rec["expect_clips_sha256_sorted"] = uid_digest(expect)
        rec["expect_clips_is_sorted"] = expect == sorted(expect)
        # (1) the SUPPLIED LIST must itself be the parity split -------------- #
        if uid_digest(expect) != exp_digest or len(expect) != n_exp:
            _refuse_v2(label, source_key, dirs, [
                "  the --expect-clips LIST is not the parity split, so it cannot",
                "  be used as the expectation (a self-consistent wrong pair would",
                "  otherwise verify):",
                f"  list       : {len(expect)} ids, sha256 "
                f"{uid_digest(expect)}",
                f"  manifest   : {n_exp} ids, sha256 {exp_digest}   <-- MISMATCH",
                "  Re-export it with code/parity_split_export.py on a host that",
                "  reproduces BOTH corpus keys (it refuses to write otherwise).",
            ])
        built_s, expect_s = set(built), set(expect)
        missing = sorted(expect_s - built_s)
        extra = sorted(built_s - expect_s)
        rec.update({"missing_count": len(missing), "extra_count": len(extra)})
        if extra:
            _refuse_v2(label, source_key, dirs, [
                f"  clips      : {len(built)} built, {n_exp} in the parity split",
                f"  extra      : {len(extra)}  <-- NOT IN THE PARITY SPLIT",
                f"  missing    : {len(missing)}",
                "  ⚠️ note the counts can MATCH while membership differs — a "
                "swapped",
                "  clip leaves the count untouched. That is why this is a set "
                "diff.",
            ])
        if missing:
            # (2) the shortfall must be EXACTLY the recorded decode failures.  #
            if skip_idx and max(skip_idx) < len(expect):
                expected_missing = sorted(expect[i] for i in skip_idx)
                rec["shortfall_identity_checked"] = True
                rec["shortfall_matches_recorded_skips"] = (
                    missing == expected_missing)
                if missing != expected_missing:
                    n_same = len(set(missing) & set(expected_missing))
                    _refuse_v2(label, source_key, dirs, [
                        f"  clips      : {len(built)} built, {n_exp} in the "
                        f"parity split",
                        f"  missing    : {len(missing)}  (the parity corpus "
                        f"records {len(skip_idx)} decode failures)",
                        f"  of those, {n_same} are the RECORDED failures and "
                        f"{len(missing) - n_same} are NOT",
                        "  A shortfall of the right SIZE made of the WRONG clips is a",
                        "  different episode set. The recorded failures are corrupt",
                        "  source clips and must fail again at any geometry; anything",
                        "  else is a build that lost data.",
                        "  Do not register and do not train — report it.",
                    ])
            else:
                rec["shortfall_identity_checked"] = False
                rec["shortfall_matches_recorded_skips"] = (
                    len(missing) == len(skip_idx))
                if len(missing) != len(skip_idx):
                    _refuse_v2(label, source_key, dirs, [
                        f"  missing    : {len(missing)}, expected at most "
                        f"{len(skip_idx)} decode failures",
                        "  (COUNT-ONLY: this manifest entry carries no skip "
                        "indices, so the",
                        "   identity of the failures could not be checked.)",
                    ])
        else:
            rec["shortfall_identity_checked"] = True
            rec["shortfall_matches_recorded_skips"] = True
        rec["verified"] = True
        rec["content_check"] = (
            "set-diff vs the exported parity clip list: 0 extra, "
            + ("0 missing (COMPLETE build)" if not missing else
               f"{len(missing)} missing == the recorded decode failures"
               + ("" if rec.get("shortfall_identity_checked")
                  else " (BY COUNT ONLY)")))
        print(f"[parity] {label}: v2 MEMBERSHIP VERIFIED — {len(built)} of "
              f"{n_exp} parity clips, 0 foreign, {len(missing)} missing "
              f"({rec['content_check']}).", flush=True)
        return rec

    # -- mode B: digest only (no clip list on this host) --------------------- #
    rec["mode"] = "digest-only (no expect_clips)"
    if got_digest != exp_digest:
        short = n_exp - len(built)
        _refuse_v2(label, source_key, dirs, [
            f"  clips      : {len(built)} built, {n_exp} in the parity split"
            + (f"   <-- SHORT BY {short}" if short > 0 else "   (count OK)"),
            f"  clip sha256: {got_digest}  built",
            f"               {exp_digest}  parity split   <-- MISMATCH",
            "",
            "  DIGEST-ONLY MODE cannot say WHICH clips differ — it has no clip",
            "  list to diff against, only the corpus digest. ⚠️ It therefore also",
            "  cannot accept the 24 legitimate decode failures: any incomplete",
            "  build fails here even when it is correct.",
            "  Re-run on a host holding the exported split, with:",
            "    --expect-clips <parity_train_clips.txt>",
        ])
    rec.update({"missing_count": 0, "extra_count": 0, "verified": True,
                "shortfall_identity_checked": None,
                "content_check": "sha256(sorted clip ids) MATCHES the committed "
                                 "clip_membership digest (COMPLETE build)"})
    print(f"[parity] {label}: v2 MEMBERSHIP VERIFIED (digest-only) — "
          f"{len(built)} clips, sha256 {got_digest[:12]}… matches the committed "
          f"{source_key} clip split.", flush=True)
    return rec


def register_v2_geometry_sibling(cache_dirs, *, new_key: str, geometry: dict,
                                 source_key: str = PARITY_TRAIN_KEY,
                                 expect_clips: str | Path | None = None,
                                 manifest_path: str | Path | None = None) -> dict:
    """Mint a manifest entry for a V2 re-cache of the parity clip split.

    The v2 twin of :func:`register_geometry_sibling`, and it keeps that
    function's contract exactly: **the registration IS the proof** — the entry is
    minted only if :func:`verify_v2_membership` passes, so a key can never exist
    for a cache whose membership was not demonstrated.

    ``new_key`` must appear in the cache DIRECTORY NAME, because
    :func:`corpus_key_of` resolves by path substring and a key nothing resolves
    to is an inert registration — the exact failure mode WIDE_FOV_BUILD.md §6.3
    identified ("on the v2 path the registration would be INERT"). That is
    asserted here rather than left to the runbook.

    Returns the new entry; the caller writes it into ``parity_manifest.json``
    and stages the diff (``scripts/register_v2_sibling.py --write-manifest``).
    """
    dirs = _v2_paths(cache_dirs)
    if new_key in (PARITY_TRAIN_KEY, PARITY_VAL_KEY, *LEAKY_SPLIT_KEYS):
        raise ParityViolation(
            f"refusing to register {new_key!r}: it is an EXISTING registered "
            f"corpus key. A geometry sibling is a NEW corpus with the same "
            f"membership, never an overwrite of the source entry.")
    if not any(new_key in str(Path(d).resolve()).replace("\\", "/")
               for d in dirs):
        raise ParityViolation(
            f"refusing to register {new_key!r}: it does not appear in the cache "
            f"path(s) {dirs}. corpus_key_of() resolves by path substring, so a "
            f"key absent from the directory name would NEVER be found at train "
            f"time — the registration would be INERT and the trainer would read "
            f"the cache as NON-PARITY. Rename the cache dir to contain "
            f"{new_key!r} (or pass the key that is already in it).")
    proof = verify_v2_membership(dirs, label=f"register:{new_key}",
                                 source_key=source_key,
                                 expect_clips=expect_clips,
                                 manifest_path=manifest_path)
    built = v2_clip_ids(dirs)
    src = manifest_entry(source_key, manifest_path) or {}
    ent = {
        "corpus_key": new_key,
        "split": src.get("split", "train"),
        "episode_count": len(built),
        "uid_kind": V2_UID_KIND,
        "uid_source": f"v2 re-cache of {source_key} at a new geometry",
        "episode_uid_sha256": uid_digest(built),
        "skip_indices": [],
        "skip_count": 0,
        "provenance": {
            "derived_from": source_key,
            "relation": "geometry sibling (v2 compressed) — identical clip "
                        "selection, different canonical frame and container",
            "geometry": dict(geometry),
            "membership_proof": proof,
            "verified": proof.get("content_check"),
            "uid_note": "uids here are CLIP IDS, not epcache positions; they are "
                        "gated-confidential and are NOT enumerated in this "
                        "manifest — only their sorted sha256 is.",
        },
    }
    print(f"[parity] v2 geometry sibling VERIFIED: {new_key} holds "
          f"{len(built)} of the {proof['clips_expected']} parity clips "
          f"(clip sha256 {ent['episode_uid_sha256'][:12]}…), 0 foreign. "
          f"Selection parity is preserved; only the pixels and the container "
          f"differ.", flush=True)
    return ent


def assert_v2_parity_cache(cache_dirs, *, label: str, require: bool = False,
                           manifest_path: str | Path | None = None) -> dict:
    """THE trainer-facing guard for ``--v2-cache``. Call BEFORE any GPU work.

    Deliberately mirrors :func:`assert_parity_corpus` decision-for-decision, so
    the two paths cannot drift:

    * dir(s) reference a registered **v2 sibling** key  -> full count + clip-id
      digest check against the committed manifest;
    * dir(s) reference a registered key of the **wrong uid kind** (an epcache
      key on a ``*.v2ep.pt`` directory) -> refuse, naming the kind — never a
      silent pass and never a misleading "no episodes found";
    * dir(s) reference a KNOWN-LEAKY split key         -> always refuse;
    * dir(s) reference nothing registered              -> ``require=True``
      refuses, otherwise one loud ``NON-PARITY`` line and ``parity=False``.

    ⚠️ ``require`` defaults to **False**, which is what preserves the deliberate
    non-parity v2 corpora (``physicalai-v2bal``, 9 000 clips — one of which is
    training as this lands). Enforcement for a parity run is opt-in at the
    trainer's ``--require-parity``, so no existing command changes behaviour.
    """
    dirs = _v2_paths(cache_dirs)
    # FIRST, unconditionally: a clip present in two dirs is a defect whatever
    # the corpus is (the dirs are CONCATENATED, so it trains twice and
    # re-weights the mix), so this must not sit behind the parity branch.
    built = v2_clip_ids(dirs)
    keys = {corpus_key_of(d, manifest_path) for d in dirs}
    named = sorted(k for k in keys if k)
    if len(named) > 1:
        _refuse_v2(label, "/".join(named), dirs, [
            f"  the --v2-cache dirs reference {len(named)} DIFFERENT registered "
            f"corpora: {named}",
            "  They are CONCATENATED into one training set, so the result is",
            "  neither corpus. Refusing.",
        ])
    key = named[0] if named else None
    if key in LEAKY_SPLIT_KEYS:
        _refuse_v2(label, key, dirs, [
            "  LEAKED SPLIT — this directory is not a valid held-out set:",
            f"  {LEAKY_SPLIT_KEYS[key]}",
        ])
    if key is None:
        msg = (f"[parity] ⚠ NON-PARITY v2 corpus for {label}: {dirs} reference "
               f"no registered parity key. Results off it are NOT cross-arm "
               f"comparable with the parity arms.")
        if require:
            raise ParityViolation(
                "\n".join([
                    "",
                    "=" * 78,
                    f"PARITY VIOLATION [{label}] — unregistered v2 cache",
                    "=" * 78,
                    *(f"  cache      : {d}" for d in dirs),
                    "  --require-parity was passed and none of these dirs "
                    "references a",
                    "  registered corpus key. A v2 re-cache of the sacred corpus "
                    "must be",
                    "  REGISTERED before it can be trained on — the registration "
                    "is the",
                    "  proof that geometry changed the pixels and not the "
                    "membership.",
                    "",
                    "    scripts/register_v2_sibling.py --cache <dir> --new-key "
                    "<key> \\",
                    "        --expect-clips <parity_train_clips.txt> "
                    "--write-manifest",
                    "",
                    "  Then stage tanitad/data/parity_manifest.json and make sure "
                    "the cache",
                    "  DIRECTORY NAME contains <key> (corpus_key_of resolves by "
                    "path).",
                    *_missing_entry_lines(dirs, None, manifest_path),
                    "=" * 78,
                ]))
        print(msg, flush=True)
        return {"parity": False, "cache_dirs": dirs, "corpus_key": None,
                "label": label, "uid_kind": V2_UID_KIND, "checked": False,
                "clips_present": len(built)}
    ent = manifest_entry(key, manifest_path)
    if ent is None:
        _refuse_v2(label, key, dirs, [
            f"  manifest   : NO ENTRY for {key!r} in "
            f"{Path(manifest_path or MANIFEST_PATH)}",
            "  A registered parity key with no manifest entry cannot be verified.",
        ])
    if ent.get("uid_kind") != V2_UID_KIND:
        _refuse_v2(label, key, dirs, [
            f"  uid kind   : this directory holds {V2_EPISODE_GLOB} "
            f"(uid_kind {V2_UID_KIND!r}),",
            f"               but {key!r} is registered as "
            f"{ent.get('uid_kind')!r} — a RAW EPCACHE corpus whose episode",
            "               identity is a POSITION, not a clip id.",
            "  The two uid spaces are not comparable, so passing this check would",
            "  prove nothing. Register the re-cache as its OWN v2 sibling key",
            "  instead of borrowing the epcache key's name.",
            *_missing_entry_lines(dirs, key, manifest_path),
        ])
    if not built:
        _refuse_v2(label, key, dirs, [
            f"  clips      : 0 present — no {V2_EPISODE_GLOB} in these dir(s)",
            "  does the path point at the v2 cache dir itself, not its parent?",
        ])
    n_exp = int(ent["episode_count"])
    exp_digest = ent.get("episode_uid_sha256")
    got = uid_digest(built)
    rec = {"parity": True, "checked": True, "label": label, "cache_dirs": dirs,
           "corpus_key": key, "uid_kind": V2_UID_KIND,
           "split": ent.get("split"), "uid_source": ent.get("uid_source"),
           "skip_hash": PARITY_SKIP_HASH,
           "episodes_expected": n_exp, "episodes_loaded": len(built),
           "episode_uid_sha256": got,
           "episode_uid_sha256_expected": exp_digest,
           "derived_from": (ent.get("provenance") or {}).get("derived_from"),
           "geometry": (ent.get("provenance") or {}).get("geometry")}
    if exp_digest is None:
        _refuse_v2(label, key, dirs, [
            "  manifest   : this v2 entry carries NO clip-id digest, so nothing",
            "               about its membership can be checked.",
            "  A v2 entry is minted BY the membership proof "
            "(register_v2_geometry_sibling),",
            "  so a digest-less v2 entry means the manifest was hand-edited.",
        ])
    if len(built) != n_exp or got != exp_digest:
        delta = (f"TRUNCATED by {n_exp - len(built)}" if len(built) < n_exp
                 else f"EXTRA {len(built) - n_exp}" if len(built) > n_exp
                 else "count OK — MEMBERSHIP DIFFERS AT THE SAME COUNT")
        _refuse_v2(label, key, dirs, [
            f"  clips      : {len(built)} present, {n_exp} registered   <-- {delta}",
            f"  clip sha256: {got}  present",
            f"               {exp_digest}  registered   <-- MISMATCH",
            "  ⚠️ a COUNT alone cannot catch this class: drop one clip and add one",
            "  foreign clip and the count is unchanged. The digest is the check.",
        ])
    print(f"[parity] {label}: {key} v2 VERIFIED — {len(built)} clips, clip "
          f"sha256 {got[:12]}… matches the committed manifest "
          f"(sibling of {rec['derived_from']}, skip-hash {PARITY_SKIP_HASH}).",
          flush=True)
    rec["content_check"] = ("sha256(sorted clip ids) MATCHES the committed "
                            "manifest entry")
    return rec


def assert_not_parity(*paths: str | Path, label: str) -> None:
    """The FIREWALL direction, for SIDE models that claim in their own docstring
    never to touch the parity corpus (``train_dynamics_encoder``). Turns that
    prose claim into an assertion."""
    for p in paths:
        if p is None:
            continue
        key = corpus_key_of(p)
        if key in (PARITY_TRAIN_KEY, PARITY_VAL_KEY):
            raise ParityViolation(
                f"PARITY FIREWALL [{label}]: {p} references the WM parity corpus "
                f"{key}. This is a SIDE model — it must never read the parity "
                f"corpus (its own docstring says so). Refusing.")


# --------------------------------------------------------------------------- #
# 10. EVAL-SPLIT CONTAMINATION — a train clip may never be in an eval split     #
# --------------------------------------------------------------------------- #
# THE HOLE THIS CLOSES (MEASURED 2026-08-17/18, RETRACTION_LOG C112):
#
#   The Alpamayo augmentation corpus was treated as disjoint from the parity
#   train split because it came from a DIFFERENT SOURCE. Nobody intersected the
#   ids. 201 of its 4 729 clips are in `physicalai-train-e438721ae894`, and the
#   live v6F run reads exactly that cache — so an eval split built on the
#   corpus scores the flagship on its own training data.
#
# ⇒ ROOT-CAUSE CLASS: **A NON-OVERLAP ASSUMED FROM PROVENANCE RATHER THAN
#   COMPUTED FROM IDS.** Everything below exists to make that assumption
#   unnecessary — and, where it matters, impossible.
#
# ⚠️ WHY §9 COULD NOT ALREADY ANSWER THIS, which is the whole reason for a new
# section. §9 proves a cache's membership against the corpus digest, and
# :func:`assert_v2_splits_disjoint` compares two supplied cache dirs. NEITHER
# can answer *"is this ARBITRARY clip id in the parity train split?"*, because
# the manifest carries only ``clip_id_sha256_sorted`` — a digest of the WHOLE
# SORTED LIST. A whole-list digest is a set identity, not a membership oracle:
# you cannot test one element against it. So the question was unanswerable on
# any host without the gated clip list, and "unanswerable" is exactly the
# condition under which a provenance assumption gets made instead.
#
# 🔒 CONFIDENTIALITY IS PRESERVED. The committed data is
# ``parity_train_clip_digests.json`` — ``sha256(clip_id)`` per clip, minted by
# ``scripts/make_parity_clip_digests.py``, which REFUSES to write unless its
# source reproduces the manifest's committed ``clip_id_sha256_sorted``. It
# answers membership exactly and enumerates nothing. Every refusal below prints
# COUNTS ONLY, like §9's.
#
# ⛔ WHAT THIS IS *NOT*. It is not a list of "the 201 Alpamayo clips". A
# hand-listed offender set is the C99/C105 failure — it is right until the
# corpus grows and then it is silently short. The check is DERIVED: the
# question asked is always *"is this clip in the parity TRAIN split?"*, so the
# next 4 472 Alpamayo clips, a new augmentation corpus, or an OOD set are all
# covered by the same call with no list to update.
#
# ⚠️ SCOPE, stated so it is not over-read: this covers the parity TRAIN split
# only. Overlap with the parity VAL split is NOT a leak (val is held out by
# construction) but it IS a comparability hazard — the same episodes scored
# twice under two names. That check needs a val digest set, which is not minted
# (this host has never held the 600 val clip ids); ``corpus_key=`` is the hook,
# and its absence is named here rather than left to be discovered.

CLIP_DIGESTS_PATH = Path(__file__).with_name("parity_train_clip_digests.json")
DEPLOYED_VAL_DIGESTS_PATH = Path(__file__).with_name(
    "deployed_val40_clip_digests.json")
CLIP_DIGESTS_SCHEMA = "tanitad.parity_clip_digests/1"

_CLIP_DIGEST_CACHE: dict[str, dict] = {}


def clip_digest(clip_id: str) -> str:
    """``sha256`` of ONE clip id. The membership token used by §10.

    Deliberately NOT :func:`uid_digest` — that hashes a whole sorted SET and is
    the thing that could not answer a per-id question. Fixed here and nowhere
    else; changing it invalidates the committed digest file."""
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()


def load_clip_digests(path: str | Path | None = None) -> dict:
    """Read (and memoize) the committed per-clip digest set, self-checked.

    The self-check is not ceremony: this file's ONLY job is to decide what gets
    excluded from an eval set, so a truncated or hand-edited one would silently
    UNDER-exclude — a leak wearing a working guard as a disguise. ``n_clips``
    and ``digest_of_digests`` must both agree with the digests actually present.
    """
    p = Path(path) if path else CLIP_DIGESTS_PATH
    k = str(p)
    if k in _CLIP_DIGEST_CACHE:
        return _CLIP_DIGEST_CACHE[k]
    if not p.exists():
        raise ParityViolation(
            f"parity clip-digest set missing: {p}\n"
            f"Without it no host can answer 'is this clip in the parity TRAIN "
            f"split?', which is the question whose absence produced C112 (a "
            f"non-overlap ASSUMED from provenance). Mint it with:\n"
            f"  python scripts/make_parity_clip_digests.py --from-cache <v2 "
            f"cache> --out {p}\n"
            f"(it refuses to write unless the source reproduces the committed "
            f"clip_id_sha256_sorted).")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") != CLIP_DIGESTS_SCHEMA:
        raise ParityViolation(
            f"{p} has schema {d.get('schema')!r}, expected "
            f"{CLIP_DIGESTS_SCHEMA!r}")
    digs = list(d.get("clip_id_digests") or [])
    problems = []
    if len(digs) != int(d.get("n_clips", -1)):
        problems.append(f"n_clips {d.get('n_clips')} != {len(digs)} digests "
                        f"present")
    if len(set(digs)) != len(digs):
        problems.append(f"{len(digs) - len(set(digs))} duplicate digest(s)")
    if d.get("digest_of_digests") and uid_digest(digs) != d["digest_of_digests"]:
        problems.append(f"digest_of_digests {d['digest_of_digests']} != "
                        f"{uid_digest(digs)} recomputed  <-- FILE ALTERED")
    # ⚠️ ONLY a FULL-corpus set can be compared to the manifest's whole-list
    # digest. A DEPLOYMENT (the 40-of-600 val) is a subset and cannot reproduce
    # it — comparing anyway would refuse every valid deployment, and comparing
    # "when it happens to match" would be a check that never fires. The
    # deployment's proof is its recorded second-source cross-check instead, and
    # a deployment file that claims neither is refused below.
    if d.get("is_full_corpus"):
        ent = manifest_entry(d.get("corpus_key"), None) or {}
        cm = ent.get("clip_membership") or {}
        if cm.get("clip_id_sha256_sorted") and \
                d.get("clip_id_sha256_sorted") != cm["clip_id_sha256_sorted"]:
            problems.append(
                f"clip_id_sha256_sorted {d.get('clip_id_sha256_sorted')} does "
                f"not match the manifest's {cm['clip_id_sha256_sorted']} for "
                f"{d.get('corpus_key')!r}  <-- MINTED FROM A DIFFERENT CORPUS")
    elif not d.get("cross_check_source") or \
            int(d.get("cross_check_episodes") or 0) != len(digs):
        problems.append(
            f"this is a SUBSET set ({d.get('deployment')!r}) and carries no "
            f"complete second-source cross-check "
            f"({d.get('cross_check_episodes')} of {len(digs)} episodes) — a "
            f"subset cannot be proven against the corpus digest, so without "
            f"the cross-check its membership is UNPROVEN")
    if problems:
        raise ParityViolation("\n".join([
            "",
            "=" * 78,
            f"PARITY CLIP-DIGEST SET IS NOT SELF-CONSISTENT — {p}",
            "=" * 78,
            *(f"  {x}" for x in problems),
            "",
            "  This file decides which clips are EXCLUDED from an eval split. A",
            "  short or altered one under-excludes SILENTLY, which is a leak",
            "  wearing a working guard as a disguise. Re-mint it with",
            "  scripts/make_parity_clip_digests.py (it proves the source against",
            "  the committed manifest before writing).",
            "=" * 78,
        ]))
    _CLIP_DIGEST_CACHE[k] = d
    return d


def parity_train_clip_digests(path: str | Path | None = None) -> frozenset[str]:
    """The membership oracle: ``sha256(clip_id)`` for every parity TRAIN clip."""
    return frozenset(load_clip_digests(path)["clip_id_digests"])


def clips_in_parity_train(clip_ids: Iterable[str],
                          path: str | Path | None = None) -> list[str]:
    """Which of ``clip_ids`` are in the parity TRAIN split (sorted).

    Returns the IDS — the caller already holds them, so this discloses nothing
    it did not supply, and a caller that must FIX a split cannot act on a count.
    Everything this module PRINTS stays counts-only."""
    digs = parity_train_clip_digests(path)
    return sorted({str(c) for c in clip_ids if clip_digest(c) in digs})


def assert_eval_clips_disjoint_from_parity_train(
        clip_ids: Iterable[str], *, label: str,
        corpus_key: str = PARITY_TRAIN_KEY,
        path: str | Path | None = None,
        sanctioned_audit: str | None = None) -> dict:
    """⭐ REFUSE an EVAL split that contains a parity TRAIN clip.

    THE point of §10, and the one call an Alpamayo (or any other) eval split
    must not be constructible without. Call it with the clip ids destined for
    the eval/held-out/OOD set, BEFORE any scoring.

    ``sanctioned_audit`` is the ONE way past it, and it is deliberately not a
    boolean: it takes the REASON, mirrors :func:`note_leaky_audit`, prints the
    disclosure and stamps ``decision_grade: False`` into the returned record —
    so an artifact produced under it can never be quoted as a held-out number.
    A label audit or a coverage census over train clips is legitimate; a
    silent one is not.

    🔒 Counts only in every message — clip ids are gated-confidential. Use
    :func:`clips_in_parity_train` in-process when you need to know WHICH.
    """
    ids = [str(c) for c in clip_ids]
    bad = clips_in_parity_train(ids, path)
    n, nb = len(set(ids)), len(bad)
    rec = {"label": label, "corpus_key": corpus_key,
           "eval_clips": n, "in_parity_train": nb,
           "contaminated_frac": (nb / n) if n else 0.0,
           "disjoint": nb == 0,
           "decision_grade": nb == 0 or sanctioned_audit is None,
           "digest_source": str(Path(path) if path else CLIP_DIGESTS_PATH)}
    if not nb:
        print(f"[parity] {label}: eval split is DISJOINT from {corpus_key} — "
              f"{n} clips, 0 in the train split (checked by per-clip sha256, "
              f"not by provenance).", flush=True)
        return rec
    if sanctioned_audit is not None:
        rec["audit_reason"] = sanctioned_audit
        rec["decision_grade"] = False
        print(f"[parity] ⚠ {label}: {nb} of {n} clips "
              f"({100 * nb / n:.1f} %) are IN {corpus_key}. Sanctioned here "
              f"because: {sanctioned_audit}. NOTHING computed over these clips "
              f"is decision-grade — it is a measurement on training data.",
              flush=True)
        return rec
    raise ParityViolation("\n".join([
        "",
        "=" * 78,
        f"PARITY VIOLATION [{label}] — TRAIN-CONTAMINATED EVAL SPLIT",
        "=" * 78,
        f"  eval split : {n} clip(s)",
        f"  in {corpus_key}:",
        f"               {nb} clip(s)  ({100 * nb / n:.1f} %)   <-- LEAK",
        "",
        "  These clips are in the corpus the parity arms TRAIN on. Scoring a",
        "  parity-trained checkpoint on them measures memorisation, not skill,",
        "  and it does not crash: the number is plausible and wrong.",
        "",
        "  This is the REF-A I-JEPA class (~80 % of val inside train, which made",
        "  that arm's val number permanently unusable) — and it is reachable by",
        "  pure OMISSION, because the two corpora have different NAMES and",
        "  different SOURCES. Provenance is not disjointness; ids are.",
        "",
        "  Fix the SPLIT, not the check:",
        "    kept, dropped, rec = parity.filter_eval_clips(ids, label=...)",
        "  or, if reading train clips IS the point (a label audit, a coverage",
        "  census), say so and accept the stamp:",
        "    parity.assert_eval_clips_disjoint_from_parity_train(",
        "        ids, label=..., sanctioned_audit='<why>')",
        "",
        "  🔒 clip ids are gated-confidential and are NOT printed. Call",
        "  parity.clips_in_parity_train(ids) in-process to get them.",
        "=" * 78,
    ]))


def filter_eval_clips(clip_ids: Iterable[str], *, label: str,
                      path: str | Path | None = None
                      ) -> tuple[list[str], list[str], dict]:
    """The sanctioned REMOVAL path: ``(kept, dropped, record)``.

    ``assert_…`` refuses; this one repairs, loudly. Use it where a split is
    being CONSTRUCTED (you own the membership) and the assert where a split is
    being CONSUMED (you must not silently rescore a different set than the
    caller named).

    ⚠️ The record carries ``n_dropped`` and the post-filter count so a report
    can never quote the pre-filter n. That is not pedantry: a split silently
    shrinking under a filter is how a published window count stops matching the
    set it was computed over."""
    ids = sorted({str(c) for c in clip_ids})
    dropped = clips_in_parity_train(ids, path)
    kept = [c for c in ids if c not in set(dropped)]
    rec = {"label": label, "n_in": len(ids), "n_dropped": len(dropped),
           "n_kept": len(kept),
           "dropped_frac": (len(dropped) / len(ids)) if ids else 0.0,
           "rule": f"excluded because present in {PARITY_TRAIN_KEY}",
           "digest_source": str(Path(path) if path else CLIP_DIGESTS_PATH)}
    if dropped:
        print(f"[parity] ⚠ {label}: DROPPED {len(dropped)} of {len(ids)} clips "
              f"({100 * len(dropped) / len(ids):.1f} %) — they are in "
              f"{PARITY_TRAIN_KEY}. The eval split is {len(kept)} clips; quote "
              f"THAT number, never {len(ids)}.", flush=True)
    else:
        print(f"[parity] {label}: {len(ids)} clips, none in "
              f"{PARITY_TRAIN_KEY} — nothing dropped.", flush=True)
    return kept, dropped, rec


# --------------------------------------------------------------------------- #
# 10b. THE OTHER DIRECTION — a TRAIN corpus must not swallow the DEPLOYED VAL   #
# --------------------------------------------------------------------------- #
# ⭐ MEASURED 2026-08-18 while closing C112, and it is the more dangerous half:
#
#   **6 of the 40 canonical val episodes (15.0 %) are inside the Alpamayo
#   4 729-clip record set.**
#
# §10 above asks "does this EVAL split contain TRAIN clips?" — a hazard for a
# split that does not exist yet. This asks the converse, "does this TRAIN /
# augmentation corpus contain the DEPLOYED VAL clips?", and its trigger is
# already scheduled: the moment the Alpamayo corpus becomes supervision (the
# 4 472-clip build), 15 % of the episode set behind EVERY published open-loop
# number — ADE@2s, FDE, miss-rate, the four families, D1/D2/D3 — is inside
# training. Nothing would crash and no existing guard would notice: `parity.py`
# §9 checks a cache against ITS OWN corpus digest, and an augmentation corpus is
# a different corpus by construction.
#
# ⚠️ Blast radius TODAY is ZERO — no trainer consumes the Alpamayo labels (grep
# over `stack/`, `colab/` for a train/loss/dataset path that reads them returns
# nothing, and `V6LossWeights` has no tactical term). This is a guard placed
# BEFORE the failure, which is the only time a guard is cheap.
#
# The digest set is the DEPLOYED 40, not the 600-episode val build, because the
# 40 are what every published statistic was computed over (parity_manifest.json
# `known_deployments`: "canonical TanitEval deployment -> 881 stride-8 windows").
# ⚠️ Its proof is a SECOND-SOURCE cross-check, not the manifest digest — see
# `make_parity_clip_digests.build_deployment` for why a subset cannot have the
# latter, said out loud so the weaker proof is never read as the stronger one.


def deployed_val_clip_digests(path: str | Path | None = None) -> frozenset[str]:
    """``sha256(clip_id)`` for every clip in the canonical 40-episode val
    deployment — the episode set behind the published open-loop statistic."""
    return frozenset(load_clip_digests(path or DEPLOYED_VAL_DIGESTS_PATH)
                     ["clip_id_digests"])


def clips_in_deployed_val(clip_ids: Iterable[str],
                          path: str | Path | None = None) -> list[str]:
    """Which of ``clip_ids`` are in the canonical val deployment (sorted)."""
    digs = deployed_val_clip_digests(path)
    return sorted({str(c) for c in clip_ids if clip_digest(c) in digs})


def assert_train_clips_disjoint_from_deployed_val(
        clip_ids: Iterable[str], *, label: str,
        path: str | Path | None = None,
        sanctioned_audit: str | None = None) -> dict:
    """⭐ REFUSE a TRAIN / augmentation corpus that contains a DEPLOYED VAL clip.

    Call it on the clip ids of anything about to become supervision — a new
    label corpus, an augmentation set, a re-cache, an external dataset join.

    ⚠️ This is NOT :func:`assert_v2_splits_disjoint`. That one needs both a train
    dir and a val dir in hand and can only compare what one launch command
    happened to pass; a label corpus arriving as a parquet, a JSONL or an HF
    dataset never meets a val dir at all. Here the val side is COMMITTED, so the
    question is answerable from the new corpus alone — which is the only form
    that can be asked at ingest time, before anything is built."""
    ids = [str(c) for c in clip_ids]
    bad = clips_in_deployed_val(ids, path)
    n, nb = len(set(ids)), len(bad)
    n_val = len(deployed_val_clip_digests(path))
    rec = {"label": label, "train_clips": n, "in_deployed_val": nb,
           "deployed_val_episodes": n_val,
           "frac_of_val_swallowed": (nb / n_val) if n_val else 0.0,
           "disjoint": nb == 0,
           "decision_grade": nb == 0 or sanctioned_audit is None}
    if not nb:
        print(f"[parity] {label}: DISJOINT from the {n_val}-episode val "
              f"deployment — {n} clips, 0 overlap (checked by per-clip sha256).",
              flush=True)
        return rec
    if sanctioned_audit is not None:
        rec["audit_reason"] = sanctioned_audit
        rec["decision_grade"] = False
        print(f"[parity] ⚠ {label}: {nb} of the {n_val} DEPLOYED VAL episodes "
              f"are in this corpus. Sanctioned because: {sanctioned_audit}. "
              f"NOTHING trained on it may be scored on the canonical val split.",
              flush=True)
        return rec
    raise ParityViolation("\n".join([
        "",
        "=" * 78,
        f"PARITY VIOLATION [{label}] — A TRAIN CORPUS SWALLOWS THE DEPLOYED VAL",
        "=" * 78,
        f"  corpus     : {n} clip(s)",
        f"  overlap    : {nb} of the {n_val} canonical val episodes "
        f"({100 * nb / n_val:.1f} % of the val split)   <-- LEAK",
        "",
        "  Those episodes are the set EVERY published open-loop number is quoted",
        "  over (881 stride-8 windows; parity_manifest known_deployments).",
        "  Training on them does not crash and does not show up in any existing",
        "  check: §9 proves a cache against ITS OWN corpus digest, and an",
        "  augmentation corpus is a different corpus by construction.",
        "",
        "  This is the REF-A I-JEPA failure approached from the training side.",
        "  Exclude the overlap before building:",
        "    kept, dropped, rec = parity.filter_train_clips(ids, label=...)",
        "",
        "  🔒 clip ids are gated-confidential and are NOT printed. Call",
        "  parity.clips_in_deployed_val(ids) in-process to get them.",
        "=" * 78,
    ]))


def filter_train_clips(clip_ids: Iterable[str], *, label: str,
                       path: str | Path | None = None
                       ) -> tuple[list[str], list[str], dict]:
    """The removal path for 10b: drop the deployed-val clips from a corpus that
    is about to become supervision. ``(kept, dropped, record)``."""
    ids = sorted({str(c) for c in clip_ids})
    dropped = clips_in_deployed_val(ids, path)
    kept = [c for c in ids if c not in set(dropped)]
    rec = {"label": label, "n_in": len(ids), "n_dropped": len(dropped),
           "n_kept": len(kept),
           "rule": "excluded because present in the canonical val deployment"}
    if dropped:
        print(f"[parity] ⚠ {label}: DROPPED {len(dropped)} of {len(ids)} clips "
              f"— they are canonical VAL episodes. Training on them would void "
              f"every number quoted over the val split.", flush=True)
    else:
        print(f"[parity] {label}: {len(ids)} clips, none in the val "
              f"deployment — nothing dropped.", flush=True)
    return kept, dropped, rec


def assert_v2_eval_cache(cache_dirs, *, label: str,
                         path: str | Path | None = None,
                         sanctioned_audit: str | None = None) -> dict:
    """:func:`assert_eval_clips_disjoint_from_parity_train` for a v2 cache DIR.

    The evaluator-facing twin of :func:`assert_v2_parity_cache`, and a different
    fact from :func:`assert_v2_splits_disjoint`: that one needs BOTH dirs in
    hand and can only compare what the launch command happened to pass. This one
    needs only the EVAL dir, because the train membership is committed. An
    evaluator handed a single ``--v2-cache`` — which is the normal case — could
    not use the pairwise check at all."""
    rec = assert_eval_clips_disjoint_from_parity_train(
        v2_clip_ids(cache_dirs), label=label, path=path,
        sanctioned_audit=sanctioned_audit)
    rec["cache_dirs"] = _v2_paths(cache_dirs)
    return rec
