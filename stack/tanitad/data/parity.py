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


def corpus_key_of(path: str | Path) -> str | None:
    """The registered corpus key this path references, or ``None``.

    Substring match on the resolved POSIX path — identical to the rule
    ``train_flagship_v4._assert_parity`` has always used, so wiring this in
    never *loosens* an existing check."""
    s = str(Path(path).resolve()).replace("\\", "/")
    keys = {PARITY_TRAIN_KEY, PARITY_VAL_KEY, *LEAKY_SPLIT_KEYS}
    try:                       # manifest may register more; a missing manifest
        keys |= set(load_manifest().get("corpora", {}))   # must not break the
    except SystemExit:                                    # firewall direction
        pass
    for key in sorted(keys):
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
    key = corpus_key_of(d)
    if key in LEAKY_SPLIT_KEYS:
        _refuse(label, key, d, [
            "  LEAKED SPLIT — this directory is not a valid held-out set:",
            f"  {LEAKY_SPLIT_KEYS[key]}",
        ])
    if key is None:
        msg = (f"[parity] ⚠ NON-PARITY corpus for {label}: {d} references no "
               f"registered parity key ({PARITY_TRAIN_KEY} / {PARITY_VAL_KEY}). "
               f"Results off it are NOT cross-arm comparable with the parity arms.")
        if require:
            raise ParityViolation(
                f"PARITY VIOLATION: {label}={str(cache_dir)!r} does not reference "
                f"the canonical corpus {PARITY_TRAIN_KEY}. Any re-selected episode "
                f"set breaks cross-arm comparability and is refused "
                f"(CLAUDE.md §Invariants).")
        print(msg, flush=True)
        return {"parity": False, "cache_dir": str(d), "corpus_key": None,
                "label": label}
    uids = scan_cache_dir(d, pattern)
    if not uids:
        _refuse(label, key, d, [
            f"  episodes   : 0 loaded — no {pattern} in this directory",
            "  does the path point at the SPLIT dir (…/physicalai-train-<key>) "
            "and not its parent?",
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
