#!/usr/bin/env python3
"""Mint the PER-CLIP digest set of the parity TRAIN corpus — the data behind
``tanitad.data.parity`` §10 (eval-split contamination refusal).

WHY A DIGEST SET AND NOT THE IDS
--------------------------------
``parity.py`` §9 states the rule this file obeys: *"clip ids are gated-
confidential PhysicalAI-AV content … the repo carries only the digests."* The
committed manifest already carries ONE digest — ``clip_membership.
clip_id_sha256_sorted`` — but that is a digest of the WHOLE SORTED LIST, and a
whole-list digest **cannot answer a per-id membership question**. "Is clip X in
the parity train set?" was therefore unanswerable on any host without the
gated clip list, which is precisely why the Alpamayo non-overlap was ASSUMED
FROM PROVENANCE instead of computed (C112's root-cause class).

This file closes that gap without publishing a single id: it commits
``sha256(clip_id)`` for each of the 2 400 parity train clips. Membership is then
exact — ``sha256(candidate) in digests`` — while the set is not enumerable back
into ids.

⛔ THE GENERATION IS THE PROOF (the ``register_v2_geometry_sibling`` contract).
This script REFUSES to write unless the clip ids it was handed reproduce the
committed ``clip_membership.clip_id_sha256_sorted`` for the corpus **exactly**.
A digest file therefore cannot exist for a wrong, truncated or re-selected clip
set — the same reason a v2 sibling key cannot exist for an unproven cache.

⚠️ IT PROVES MEMBERSHIP, NOTHING ELSE. It says which CLIP IDS the parity train
split holds. It says nothing about pixels, geometry, or episode content — see
``parity.py`` §9's "what this cannot prove" list, which applies verbatim.

SOURCES, in the order you are likely to have one
------------------------------------------------
``--from-cache <dir>``    a built v2 cache (``<clip_id>.v2ep.pt``) — the pod path.
``--from-listing <file>`` a plain ``ls`` of such a dir, one name per line. This is
                          what makes the mint reproducible on a DEV BOX with no
                          pod access: the concurrency pilot banked exactly such a
                          listing (``…/2026-08-17-thor-concurrency-pilot/
                          parity_ls.txt``) and it reproduces the committed digest.
``--from-ids <file>``     the gated ordered clip list (``parity_train_clips.txt``)
                          exported by ``parity_split_export.py``.

Usage
-----
    python scripts/make_parity_clip_digests.py \\
        --from-listing "TanitAD Research Hub/Architecture & Inference/\\
Implementation/incoming/2026-08-17-thor-concurrency-pilot/parity_ls.txt" \\
        --out tanitad/data/parity_train_clip_digests.json

    # verify the committed file without rewriting it
    python scripts/make_parity_clip_digests.py --from-listing <f> --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tanitad.data import parity  # noqa: E402

SCHEMA = "tanitad.parity_clip_digests/1"


def _ids_from_listing(p: Path) -> list[str]:
    """Clip ids out of an ``ls`` of a v2 cache. Non-``.v2ep.pt`` lines (the
    ``_geometry.json`` / ``_v2manifest.pt`` sidecars) are dropped — they are
    not clips and a listing legitimately contains them."""
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        # a listing may be "<name>" or "<mtime> <name>"; take the last field
        if s:
            s = s.split()[-1]
        if s.endswith(parity.V2_SUFFIX):
            out.append(s[: -len(parity.V2_SUFFIX)])
    return out


def _ids_from_cache(d: Path) -> list[str]:
    return parity.v2_clip_ids(d)


def _ids_from_ids(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def build(ids: list[str], *, corpus_key: str, source: str,
          manifest_path: str | Path | None = None) -> dict:
    """The digest entry — minted ONLY if ``ids`` reproduces the committed
    ``clip_membership`` digest for ``corpus_key``."""
    uniq = sorted(set(ids))
    if len(uniq) != len(ids):
        raise parity.ParityViolation(
            f"refusing to mint: the source lists {len(ids)} clip ids but only "
            f"{len(uniq)} are distinct. A duplicated id means the source is not "
            f"a clip SET (🔒 ids withheld).")
    cm = parity.clip_membership_of(corpus_key, manifest_path)
    if cm is None:
        raise parity.ParityViolation(
            f"refusing to mint: {corpus_key!r} has no clip_membership block in "
            f"{Path(manifest_path or parity.MANIFEST_PATH)}. Without the "
            f"committed clip-id digest there is nothing to prove the source "
            f"against, and an unproven digest set is worse than none — it looks "
            f"like enforcement.")
    got = parity.uid_digest(uniq)
    exp = str(cm["clip_id_sha256_sorted"])
    n_exp = int(cm["n_clips"])
    if len(uniq) != n_exp or got != exp:
        raise parity.ParityViolation("\n".join([
            "",
            "=" * 78,
            f"REFUSING TO MINT CLIP DIGESTS — {corpus_key}",
            "=" * 78,
            f"  source     : {source}",
            f"  clips      : {len(uniq)} supplied, {n_exp} registered"
            + ("   <-- COUNT MISMATCH" if len(uniq) != n_exp else "   (count OK)"),
            f"  clip sha256: {got}  supplied",
            f"               {exp}  registered   <-- MISMATCH",
            "",
            "  The generation IS the proof: a digest set minted from anything but",
            "  the registered clip split would silently authorise the wrong",
            "  exclusions — the failure this file exists to prevent, inverted.",
            "  🔒 clip ids are gated-confidential and are NOT printed.",
            "=" * 78,
        ]))
    digs = sorted(parity.clip_digest(c) for c in uniq)
    return {
        "schema": SCHEMA,
        "corpus_key": corpus_key,
        "is_full_corpus": True,
        "split": (parity.manifest_entry(corpus_key, manifest_path)
                  or {}).get("split"),
        "n_clips": len(digs),
        "digest_algorithm": "sha256(clip_id.encode('utf-8')).hexdigest()",
        "clip_id_sha256_sorted": exp,
        "verified_against": (
            "the committed parity_manifest.json clip_membership."
            "clip_id_sha256_sorted — this file could not be written until the "
            "source clip set reproduced it exactly"),
        "digest_of_digests": parity.uid_digest(digs),
        "source": source,
        "confidentiality": (
            "🔒 per-clip sha256 ONLY. The ids are gated-confidential "
            "PhysicalAI-AV content; a digest set answers `is X in the parity "
            "train split?` exactly, and enumerates nothing."),
        "clip_id_digests": digs,
    }


def build_deployment(ids_by_episode: dict[str, str], *, deployment: str,
                     corpus_key: str, source: str,
                     sha8_by_episode: dict[str, str] | None,
                     cross_check_source: str, role: str) -> dict:
    """A digest set for a DEPLOYED SUBSET of a corpus (the 40-episode val).

    ⚠️ WHY THE PROOF IS DIFFERENT, AND SAID SO RATHER THAN GLOSSED. :func:`build`
    proves its source by reproducing the manifest's whole-corpus
    ``clip_id_sha256_sorted``. A 40-of-600 deployment CANNOT reproduce that
    digest — it is a subset — so that proof is unavailable and pretending
    otherwise would be a check that cannot fail.

    The substitute is a SECOND-SOURCE cross-check: ``cross_check`` maps each
    episode to a ``clip_sha8`` banked by a DIFFERENT stream, and every one must
    equal ``sha256(clip_id)[:8]`` of the id being minted. 40 independent 32-bit
    agreements is a real proof of identity, and it is permanently re-runnable
    because both artifacts are in the repo.

    ⛔ Without ``cross_check`` this refuses. An unverified deployment digest set
    would decide exclusions on an unproven membership list — the same shape as
    the thing it is defending against.
    """
    ids = list(ids_by_episode.values())
    uniq = sorted(set(ids))
    if len(uniq) != len(ids):
        raise parity.ParityViolation(
            f"refusing to mint {deployment!r}: {len(ids)} ids, {len(uniq)} "
            f"distinct (🔒 ids withheld).")
    if not sha8_by_episode:
        raise parity.ParityViolation(
            f"refusing to mint {deployment!r} with no second-source "
            f"cross-check. A deployment is a SUBSET, so it cannot reproduce "
            f"the corpus digest, and an unproven digest set that decides "
            f"exclusions is worse than none — it looks like enforcement.")
    missing = sorted(set(ids_by_episode) - set(sha8_by_episode))
    if missing:
        raise parity.ParityViolation(
            f"refusing to mint {deployment!r}: {len(missing)} episode(s) have "
            f"no entry in the cross-check source — a partial proof is not a "
            f"proof (first: {missing[:3]}).")
    checked = mism = 0
    for ep, cid in ids_by_episode.items():
        checked += 1
        if "clip_" + parity.clip_digest(cid)[:8] != str(sha8_by_episode[ep]):
            mism += 1
    if not checked or mism:
        raise parity.ParityViolation(
            f"refusing to mint {deployment!r}: second-source cross-check "
            f"{checked - mism}/{checked} matched ({mism} mismatch(es)). The "
            f"two in-repo artifacts disagree about which clips the deployment "
            f"holds; do not guess which is right.")
    digs = sorted(parity.clip_digest(c) for c in uniq)
    return {
        "schema": SCHEMA,
        "corpus_key": corpus_key,
        "is_full_corpus": False,
        "deployment": deployment,
        "role": role,
        "split": "val",
        "n_clips": len(digs),
        "digest_algorithm": "sha256(clip_id.encode('utf-8')).hexdigest()",
        "clip_id_sha256_sorted": parity.uid_digest(uniq),
        "verified_against": (
            f"a SECOND in-repo source: {checked}/{checked} episodes' banked "
            f"`clip_sha8` equal sha256(clip_id)[:8] of the ids minted here. The "
            f"whole-corpus manifest digest is NOT applicable to a subset and is "
            f"deliberately not claimed."),
        "digest_of_digests": parity.uid_digest(digs),
        "source": source,
        "cross_check_source": cross_check_source,
        "cross_check_episodes": checked,
        "confidentiality": (
            "🔒 per-clip sha256 ONLY — the ids are gated-confidential."),
        "clip_id_digests": digs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("make_parity_clip_digests")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-cache", help="a built v2 cache dir")
    src.add_argument("--from-listing", help="an ls of a v2 cache dir")
    src.add_argument("--from-ids", help="an ordered clip-id list")
    src.add_argument("--from-lead-index",
                     help="a val40_lead_index.json (ep -> {clip_id}); mints the "
                          "DEPLOYED-VAL digest set. Requires --cross-check.")
    ap.add_argument("--cross-check",
                    help="a val40_lead_index_ANON.json (ep -> {clip_sha8}) from "
                         "a different stream — the deployment's proof")
    ap.add_argument("--deployment", default="val40")
    ap.add_argument("--corpus-key", default=parity.PARITY_TRAIN_KEY)
    ap.add_argument("--out", default=None,
                    help="default: tanitad/data/parity_train_clip_digests.json")
    ap.add_argument("--verify-only", action="store_true",
                    help="prove the source and compare to the committed file; "
                         "write nothing")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)

    if a.from_lead_index:
        idx = json.loads(Path(a.from_lead_index).read_text(encoding="utf-8"))
        ids_by_ep = {ep: e["clip_id"] for ep, e in idx.items()}
        sha8 = None
        if a.cross_check:
            cc = json.loads(Path(a.cross_check).read_text(encoding="utf-8"))
            sha8 = {ep: e.get("clip_sha8") for ep, e in cc.items()}
        ent = build_deployment(
            ids_by_ep, deployment=a.deployment,
            corpus_key=parity.PARITY_VAL_KEY,
            source=f"lead index {a.from_lead_index}",
            sha8_by_episode=sha8, cross_check_source=str(a.cross_check),
            role="canonical TanitEval deployment — THE published open-loop "
                 "statistic's episode set")
        default_out = parity.DEPLOYED_VAL_DIGESTS_PATH
    else:
        if a.from_cache:
            ids, source = (_ids_from_cache(Path(a.from_cache)),
                           f"v2 cache {a.from_cache}")
        elif a.from_listing:
            ids, source = (_ids_from_listing(Path(a.from_listing)),
                           f"listing {a.from_listing}")
        else:
            ids, source = (_ids_from_ids(Path(a.from_ids)),
                           f"clip-id list {a.from_ids}")
        ent = build(ids, corpus_key=a.corpus_key, source=source,
                    manifest_path=a.manifest)
        default_out = parity.CLIP_DIGESTS_PATH
    out = Path(a.out) if a.out else default_out
    print(f"[digests] {ent['corpus_key']}"
          + (f" [{ent['deployment']}]" if ent.get("deployment") else "")
          + f": {ent['n_clips']} clips PROVEN "
          + ("against the committed manifest" if ent.get("is_full_corpus")
             else f"against a second source "
                  f"({ent.get('cross_check_episodes')} episodes' sha8)")
          + f" (clip sha256 {ent['clip_id_sha256_sorted'][:12]}…); "
            f"digest_of_digests {ent['digest_of_digests'][:12]}…", flush=True)

    if a.verify_only:
        if not out.exists():
            print(f"[digests] ⚠ no committed file at {out} to compare against — "
                  f"the source is proven, nothing else was checked.", flush=True)
            return 0
        have = json.loads(out.read_text(encoding="utf-8"))
        same = (have.get("digest_of_digests") == ent["digest_of_digests"]
                and have.get("n_clips") == ent["n_clips"])
        print(f"[digests] committed file {out}: "
              f"{'MATCHES' if same else '⛔ DIFFERS FROM'} the proven source "
              f"({have.get('n_clips')} clips, digest_of_digests "
              f"{str(have.get('digest_of_digests'))[:12]}…)", flush=True)
        return 0 if same else 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ent, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[digests] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
