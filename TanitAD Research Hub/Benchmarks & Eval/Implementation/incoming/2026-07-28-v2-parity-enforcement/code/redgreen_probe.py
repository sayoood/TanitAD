"""BOTH-DIRECTIONS PROBE for the v2 parity guard — and the C13 self-check.

Runs the new ``parity.assert_v2_parity_cache`` against one CORRECT cache and
eight deliberately DEFECTIVE ones, and records for each:

  * what a **COUNT-ONLY** check would have concluded, and
  * what the **membership** guard actually concludes, with the refusal's own
    first diagnostic line.

⚠️ The point is the column where the two disagree. A guard that only ever fires
where a count would have fired is a count check wearing a membership check's
name — class C13, "a guard that cannot fail", of which this program has shipped
several. The SWAPPED and RESELECTED rows are the demonstration that this one is
not that.

🔒 All clip ids here are synthetic. No PhysicalAI-AV content is read or written.

    python redgreen_probe.py --out raw/redgreen_2026-07-27.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data import parity                                  # noqa: E402

SRC = "synthetic-train-aaaaaaaaaaaa"
SIB = "synthetic-w120x256cyl-bbbbbbbbbbbb"
CLIPS = [f"{i:02d}cl-{chr(ord('a') + i)}" for i in range(12)]
SKIPS = [3, 7]
GEOM = {"height": 256, "width": 640, "f_ref": 305.5775,
        "projection": "cylindrical"}


def _mk(d: Path, ids) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for c in ids:
        (d / f"{c}{parity.V2_SUFFIX}").touch()
    return d


def _manifest(root: Path) -> Path:
    man = {"schema": parity.MANIFEST_SCHEMA, "corpora": {SRC: {
        "corpus_key": SRC, "split": "train",
        "episode_count": len(CLIPS) - len(SKIPS),
        "uid_kind": parity.EPCACHE_UID_KIND,
        "episode_uid_sha256": parity.uid_digest(
            [f"ep_{i:05d}.pt" for i in range(len(CLIPS)) if i not in SKIPS]),
        "clip_membership": {"n_clips": len(CLIPS),
                            "clip_id_sha256_sorted": parity.uid_digest(CLIPS),
                            "ordered_equals_sorted": CLIPS == sorted(CLIPS),
                            "decode_failures": len(SKIPS)},
        "skip_indices": SKIPS, "skip_count": len(SKIPS)}}}
    p = root / "parity_manifest.json"
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    parity._MANIFEST_CACHE.pop(str(p), None)
    return p


def _register(root: Path, mp: Path, ids, clips_txt: Path) -> Path:
    cache = _mk(root / SIB, ids)
    ent = parity.register_v2_geometry_sibling(
        cache, new_key=SIB, geometry=GEOM, source_key=SRC,
        expect_clips=clips_txt, manifest_path=mp)
    man = json.loads(mp.read_text(encoding="utf-8"))
    man["corpora"][SIB] = ent
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    parity._MANIFEST_CACHE.pop(str(mp), None)
    return cache


def _probe(name: str, cache_dirs, mp: Path, expect_n: int, require=True) -> dict:
    """Run the guard; report it AND what a count-only check would have said."""
    dirs = cache_dirs if isinstance(cache_dirs, list) else [cache_dirs]
    n = sum(len(list(Path(d).glob(parity.V2_EPISODE_GLOB))) for d in dirs)
    row = {"case": name, "clips_present": n, "clips_expected": expect_n,
           "count_only_verdict": ("PASS (a count check sees nothing wrong)"
                                  if n == expect_n else
                                  f"REFUSE (count differs by {n - expect_n:+d})")}
    try:
        rec = parity.assert_v2_parity_cache(dirs, label="probe",
                                            require=require, manifest_path=mp)
        row["guard_verdict"] = ("PASS" if rec.get("parity") else
                                "PASS-THROUGH (NON-PARITY, warned)")
        row["guard_reason"] = rec.get("content_check") or "warn-and-proceed"
    except parity.ParityViolation as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        diag = [l for l in lines
                if any(t in l for t in ("<--", "uid kind", "appear in more",
                                        "DIFFERENT registered", "--require-parity"))]
        row["guard_verdict"] = "REFUSE"
        row["guard_reason"] = (diag[0] if diag else lines[2] if len(lines) > 2
                               else lines[-1])
    row["count_only_would_have_missed_it"] = (
        row["count_only_verdict"].startswith("PASS")
        and row["guard_verdict"] == "REFUSE")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    with TemporaryDirectory() as td:
        root = Path(td)
        mp = _manifest(root)
        clips_txt = root / "clips.txt"
        clips_txt.write_text("\n".join(CLIPS) + "\n", encoding="utf-8")
        good = _register(root, mp, CLIPS, clips_txt)
        N = len(CLIPS)

        rows.append(_probe("CORRECT — the registered parity cache", good, mp, N))

        d = _mk(root / "d_drop" / SIB, CLIPS[:-1])
        rows.append(_probe("DROPPED one clip", d, mp, N))

        d = _mk(root / "d_swap" / SIB, CLIPS[:-1] + ["zz-foreign"])
        rows.append(_probe("SWAPPED one clip (IDENTICAL COUNT)", d, mp, N))

        d = _mk(root / "d_resel" / SIB, [f"re{i:02d}-sel" for i in range(N)])
        rows.append(_probe("RE-SELECTED split (IDENTICAL COUNT)", d, mp, N))

        d = _mk(root / "d_extra" / SIB, CLIPS + ["zz-foreign"])
        rows.append(_probe("EXTRA foreign clip", d, mp, N))

        d = _mk(root / "d_unreg" / "pai_wide120_v2png_train", CLIPS)
        rows.append(_probe("UNREGISTERED cache (the v5 state as built today)",
                           d, mp, N))
        rows.append(_probe("UNREGISTERED cache, WITHOUT --require-parity "
                           "(physicalai-v2bal must still run)", d, mp, N,
                           require=False))

        d = _mk(root / "d_kind" / SRC, CLIPS)
        rows.append(_probe("v2 cache wearing a RAW EPCACHE key", d, mp, N))

        a1 = _mk(root / "d_dup" / f"{SIB}_a", CLIPS[:8])
        a2 = _mk(root / "d_dup" / f"{SIB}_b", CLIPS[6:])
        rows.append(_probe("SAME CLIP in two --v2-cache dirs", [a1, a2], mp, N))

        # the membership PROOF's own strengthening over the sibling script
        sub = {}
        cache = _mk(root / "p_ok", [c for i, c in enumerate(CLIPS)
                                    if i not in SKIPS])
        rec = parity.verify_v2_membership(cache, source_key=SRC,
                                          expect_clips=clips_txt,
                                          manifest_path=mp)
        sub["shortfall_is_the_recorded_decode_failures"] = {
            "verdict": "PASS", "missing": rec["missing_count"],
            "identity_checked": rec["shortfall_identity_checked"]}
        cache = _mk(root / "p_bad", [c for i, c in enumerate(CLIPS)
                                     if i not in (0, 5)])
        try:
            parity.verify_v2_membership(cache, source_key=SRC,
                                        expect_clips=clips_txt,
                                        manifest_path=mp)
            sub["shortfall_right_size_wrong_clips"] = {"verdict": "PASS (!!)"}
        except parity.ParityViolation as e:
            line = [l.strip() for l in str(e).splitlines()
                    if "RECORDED failures" in l]
            sub["shortfall_right_size_wrong_clips"] = {
                "verdict": "REFUSE", "reason": line[0] if line else "",
                "note": "the sibling stream's verify_v2_parity.py accepted ANY "
                        "len(missing)==24; identity is the strengthening"}
        try:
            parity.verify_v2_membership(
                _mk(root / "p_short", [c for i, c in enumerate(CLIPS)
                                       if i not in SKIPS]),
                source_key=SRC, manifest_path=mp)
            sub["digest_only_on_an_incomplete_build"] = {"verdict": "PASS (!!)"}
        except parity.ParityViolation as e:
            sub["digest_only_on_an_incomplete_build"] = {
                "verdict": "REFUSE",
                "note": "⚠️ WHAT THE V2 PATH CANNOT DO: without the exported "
                        "clip list the check cannot tell a legitimate decode "
                        "failure from a lost clip, so a CORRECT incomplete "
                        "build is also refused. --expect-clips is required on "
                        "the pod.",
                "reason": next(l.strip() for l in str(e).splitlines()
                               if "DIGEST-ONLY" in l)}

    out = {
        "what": "both-directions probe of parity.assert_v2_parity_cache / "
                "verify_v2_membership",
        "when": date.today().isoformat(),
        "evidence_class": "MEASURED (synthetic caches, this repo, dev box)",
        "synthetic_corpus": {"n_clips": len(CLIPS),
                             "recorded_decode_failures": SKIPS},
        "cases": rows,
        "membership_proof_cases": sub,
        "c13_self_check": {
            "count_only_would_have_missed": [r["case"] for r in rows
                                             if r["count_only_would_have_missed_it"]],
            "reading": "a guard whose refusals are a SUBSET of what a count "
                       "check already refuses is a count check; these rows are "
                       "the evidence that it is not.",
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
