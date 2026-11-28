"""nt_guard -- H-NT-DECIDE-1 (PREREG_NT_DECIDE sec. 2 / control K3): the NavSim HELD-OUT guard.

The navtrain fine-tune (arm A1) trains on NavSim token-episodes. Two families of logs must NEVER reach training:

* the **38 selection-scan logs** (``NAVTRAIN_SAMPLE_TOKENS.json`` ``per_log``, ranks 0-37 -- every log the
  D-R8-CKPT-RULE sample scan READ, not only the 30 that contributed its 2,089 tokens): the 10 held-out selection
  reads of A1's snapshots are scored there;
* **every navtest log** (``navtest.yaml`` ``log_names``): the PDMS headline is scored there.

The guard is a FILE (``tanitad.nt_heldout/1``) whose bytes are pinned by sha256 on the trainer argv
(``--nt-heldout-sha256``) so the launch gate binds the exact list, plus a TOKEN TABLE (``--nt-token-table``) that
names the log of every train clip. The trainer REFUSES:

1. a train clip (``<clip_id>.v2ep.pt`` in ``--v2-cache``) whose log is in the held-out set;
2. a train clip the token table does not name (unknown provenance is not "probably fine");
3. a guard file whose sha256 differs from the pinned value, or whose own list fails its internal checks
   (duplicate names, an empty selection list, a stated count that disagrees with the list).

⛔ Checked on the DIRECTORY LISTING before a single provider is built -- the same pattern as
``refuse_eval_clips_in_train`` (filenames are ``<clip_id>.v2ep.pt``; a symlink view lists by link name).

⭐ A named function, so a test calls the REAL refusal, and a mutation (one selection log added to the train
table) must go RED -- ``tests/test_nt_guard.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "tanitad.nt_heldout/1"


class NTGuardError(SystemExit):
    """A refusal (the repo's SystemExit idiom)."""


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_guard(path, *, expect_sha256: str | None = None) -> dict:
    """Read + verify the held-out file. Returns the dict with ``heldout_logs`` as a frozenset."""
    p = str(path)
    if not Path(p).is_file():
        raise NTGuardError(f"[nt-guard] ⛔ held-out file {p!r} does not exist")
    sha = _sha256(p)
    if expect_sha256 is not None and sha != str(expect_sha256).lower():
        raise NTGuardError(f"[nt-guard] ⛔ held-out file {p!r} sha256 {sha} != pinned {expect_sha256} -- the launch "
                           f"binds the exact list; a different file is a different experiment")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("schema") != SCHEMA:
        raise NTGuardError(f"[nt-guard] ⛔ {p!r}: schema {d.get('schema')!r} != {SCHEMA!r}")
    sel = list(d.get("selection_scan_logs") or [])
    nt = list(d.get("navtest_logs") or [])
    held = list(d.get("heldout_logs") or [])
    for nm, lst in (("selection_scan_logs", sel), ("navtest_logs", nt), ("heldout_logs", held)):
        if len(set(lst)) != len(lst):
            raise NTGuardError(f"[nt-guard] ⛔ {p!r}: duplicate names in {nm}")
    if not sel or not nt:
        raise NTGuardError(f"[nt-guard] ⛔ {p!r}: an EMPTY selection-scan ({len(sel)}) or navtest ({len(nt)}) list "
                           f"is a guard that guards nothing")
    if set(held) != set(sel) | set(nt):
        raise NTGuardError(f"[nt-guard] ⛔ {p!r}: heldout_logs != selection_scan_logs | navtest_logs")
    for nm, lst in (("selection_scan_logs", sel), ("navtest_logs", nt)):
        stated = (d.get("counts") or {}).get(nm)
        if stated is not None and int(stated) != len(lst):
            raise NTGuardError(f"[nt-guard] ⛔ {p!r}: counts.{nm} says {stated} but the list holds {len(lst)}")
    out = dict(d)
    out["heldout_logs"] = frozenset(held)
    out["selection_scan_logs"] = tuple(sel)
    out["navtest_logs"] = tuple(nt)
    out["sha256"] = sha
    out["path"] = p
    return out


def load_token_table(path) -> dict:
    """``token_table_nt.json`` (``nt_labels.py``): ``{"rows": [{token, clip_id, log_name, ...}]}`` ->
    ``{clip_id: row}``. REFUSES a duplicate clip id or token."""
    with open(str(path), encoding="utf-8") as fh:
        d = json.load(fh)
    rows = d["rows"] if isinstance(d, dict) else d
    out, toks = {}, set()
    for r in rows:
        cid, tok = str(r["clip_id"]), str(r["token"])
        if cid in out:
            raise NTGuardError(f"[nt-guard] ⛔ token table {path!r}: clip_id {cid} twice")
        if tok in toks:
            raise NTGuardError(f"[nt-guard] ⛔ token table {path!r}: token {tok} twice")
        toks.add(tok)
        out[cid] = r
    return out


def assert_disjoint(log_names, guard: dict, *, where: str) -> dict:
    """REFUSE if any of ``log_names`` is held out. Returns the census."""
    names = [str(x) for x in log_names]
    hit = sorted({n for n in names if n in guard["heldout_logs"]})
    if hit:
        sel = [n for n in hit if n in set(guard["selection_scan_logs"])]
        raise NTGuardError(f"[nt-guard] ⛔ {where}: {len(hit)} HELD-OUT log(s) present ({len(sel)} selection-scan, "
                           f"{len(hit) - len(sel)} navtest), e.g. {hit[:3]}. Training on them voids the held-out "
                           f"selection reads / the navtest score (PREREG_NT_DECIDE sec. 2, K3).")
    return {"n_names": len(names), "n_distinct_logs": len(set(names)), "n_heldout_hits": 0,
            "guard_sha256": guard["sha256"], "n_heldout_logs": len(guard["heldout_logs"])}


def v2_cache_clip_ids(cache_dirs) -> list:
    """Clip ids from the DIRECTORY LISTING (``<clip_id>.v2ep.pt``), no episode opened."""
    dirs = [cache_dirs] if isinstance(cache_dirs, (str, Path)) else list(cache_dirs)
    ids = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.v2ep.pt")):
            ids.append(f.name[: -len(".v2ep.pt")])
    return ids


def refuse_heldout_in_train(cache_dirs, args) -> dict:
    """The trainer entry: ``--nt-heldout`` + ``--nt-heldout-sha256`` + ``--nt-token-table`` (all three or none).

    Returns the census config.json stamps; ``{}`` when the guard is not requested."""
    g, sha, tt = (getattr(args, "nt_heldout", None), getattr(args, "nt_heldout_sha256", None),
                  getattr(args, "nt_token_table", None))
    if not (g or sha or tt):
        return {}
    if not (g and sha and tt):
        raise NTGuardError("[nt-guard] ⛔ --nt-heldout, --nt-heldout-sha256 and --nt-token-table go together: a "
                           "guard without its pinned hash or without the clip->log table cannot refuse anything")
    guard = load_guard(g, expect_sha256=sha)
    table = load_token_table(tt)
    cids = v2_cache_clip_ids(cache_dirs)
    if not cids:
        raise NTGuardError(f"[nt-guard] ⛔ --v2-cache {cache_dirs!r} lists ZERO *.v2ep.pt files")
    unknown = [c for c in cids if c not in table]
    if unknown:
        raise NTGuardError(f"[nt-guard] ⛔ {len(unknown)} of {len(cids)} train clips are NOT in the token table "
                           f"{tt!r} (e.g. {unknown[:3]}): their log cannot be checked against the held-out list, and "
                           f"an unchecked clip is not a clean one")
    logs = [str(table[c]["log_name"]) for c in cids]
    toks = [str(table[c]["token"]) for c in cids]
    cen = assert_disjoint(logs, guard, where=f"--v2-cache {cache_dirs!r}")
    held_tok = set(guard.get("selection_sample_tokens") or ())
    tok_hits = sorted(set(toks) & held_tok)
    if tok_hits:
        raise NTGuardError(f"[nt-guard] ⛔ {len(tok_hits)} train token(s) are selection-sample tokens "
                           f"(e.g. {tok_hits[:3]}) -- a token-level leak the log check should already have caught")
    return dict(cen, n_train_clips=len(cids), n_selection_sample_tokens_checked=len(held_tok),
                token_table=str(tt), token_table_sha256=_sha256(tt), heldout=str(g))
