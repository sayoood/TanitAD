"""A0 — rebuild the CLEAN-124 eval split into two disjoint 62/62 halves.

⛔ WHY 124 AND NOT 139. Two exclusions, measured separately and then INTERSECTED:
  * 11 of the 139 B1 eval clips are inside the parity TRAIN corpus
    (`D-REFCV6-EVAL139-PARITY`), so a held-out read over them is not held out;
  * 4 carry no SAM3 map, so no map/occupancy metric can be computed on them.
They are DISJOINT (`mapless_that_are_also_parity: 0`), so the losses add: 139 - 11 - 4 = 124.
⚠️ The existing splitA(70)/splitB(69) caches were cut from 139 and are NOT usable for a
held-out read as they stand — which is the whole reason this step exists.

⛔ COPIES, NOT LINKS. D: is exFAT: `os.link`/symlinks raise WinError 1. The halves are
independent copies, which is also what makes them safe to hand to two concurrent arms.

⛔ EVERY CHECK IS A POSITIVE ASSERTION ON THE ARTIFACT, never on a count I computed myself:
the guard is re-run on the copied set, each half is re-listed from disk, and the two halves
are intersected.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys

ART = pathlib.Path("D:/Projects/TanitAD-artifacts")
WT = pathlib.Path("C:/Users/Admin/tanitad-wt-bevtac")
SRC = [ART / "v2ep-eval139-416x1024cyl-splitA", ART / "v2ep-eval139-416x1024cyl-splitB"]
MAPS = ART / "sam3-maps-eval"
OUT_A = ART / "v2ep-eval124clean-416x1024cyl-halfA"
OUT_B = ART / "v2ep-eval124clean-416x1024cyl-halfB"
PKG = WT / "TanitAD Research Lab/Data Engineering/Research/2026-09-18-refcv6-eval-parity-clean/raw"
CENSUS = (WT / "TanitAD Research Lab/Architecture & Inference/Research"
          / "2026-09-18-refcv6-devbox-plan/raw/clean_split.json")
MAPFILE = pathlib.Path("C:/Users/Admin/qland/refcv6_eval128_sha12_map.json")

sha12 = lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    clean128 = [l.strip() for l in (PKG / "eval128_clean_sha12.txt").read_text(
        encoding="utf-8").split() if l.strip()]
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    mapless = set(census["mapless_sha12"])
    clean124 = sorted(s for s in clean128 if s not in mapless)
    assert len(clean124) == 124, f"expected 124, got {len(clean124)}"

    m = json.loads(MAPFILE.read_text(encoding="utf-8"))
    by_sha = dict(m["clean_128"])
    clip_of = {}
    for s in clean124:
        cid = by_sha.get(s)
        assert cid, f"no clip_id for {s}"
        # ⛔ RE-DERIVE the hash rather than trust the table: sha12 is
        # sha256(clip_id)[:12], a HASH, and a table can go stale.
        assert sha12(cid) == s, f"sha12 mismatch for {s}: table says {cid}"
        clip_of[s] = cid

    # -- locate every source file and its map, BEFORE copying a single byte --- #
    src_of, missing_cache, missing_map = {}, [], []
    for s, cid in clip_of.items():
        hit = next((d / f"{cid}.v2ep.pt" for d in SRC
                    if (d / f"{cid}.v2ep.pt").is_file()), None)
        if hit is None:
            missing_cache.append(s)
        else:
            src_of[s] = hit
        if not (MAPS / f"{s}.sam3mapgt.npz").is_file():
            missing_map.append(s)
    if missing_cache or missing_map:
        print(f"ZZA0-ABORT missing_cache={len(missing_cache)} "
              f"missing_map={len(missing_map)}ZZ")
        return 2
    print(f"[a0] 124 clips resolved: every one has a cache file AND a SAM3 map")

    # -- the split: sorted-order ALTERNATING, exactly as the 139 split was ----- #
    half_a = clean124[0::2]
    half_b = clean124[1::2]
    assert len(half_a) == 62 and len(half_b) == 62, (len(half_a), len(half_b))
    assert not (set(half_a) & set(half_b)), "halves overlap"

    for out, half in ((OUT_A, half_a), (OUT_B, half_b)):
        out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(half, 1):
            dst = out / src_of[s].name
            if dst.is_file() and dst.stat().st_size == src_of[s].stat().st_size:
                continue
            shutil.copy2(src_of[s], dst)          # ⛔ exFAT: copy, never link
            if i % 20 == 0:
                print(f"[a0] {out.name}: {i}/{len(half)}", flush=True)
        print(f"[a0] {out.name}: {len(half)} clips copied", flush=True)

    # -- POSITIVE verification, re-listed FROM DISK --------------------------- #
    on_disk = {out.name: sorted(p.name.split(".v2ep")[0]
                                for p in out.glob("*.v2ep.pt"))
               for out in (OUT_A, OUT_B)}
    na, nb = len(on_disk[OUT_A.name]), len(on_disk[OUT_B.name])
    inter = set(on_disk[OUT_A.name]) & set(on_disk[OUT_B.name])
    union = set(on_disk[OUT_A.name]) | set(on_disk[OUT_B.name])
    print(f"[a0] on disk: halfA={na} halfB={nb} intersection={len(inter)} "
          f"union={len(union)}")
    if not (na == nb == 62 and not inter and len(union) == 124):
        print("ZZA0-ABORT-DISK-COUNTSZZ")
        return 3

    # -- the parity guard, RE-RUN on the copied set --------------------------- #
    sys.path.insert(0, str(WT / "stack"))
    from tanitad.data import parity
    ids = sorted(clip_of[s] for s in clean124)
    kept, rec = parity.guard_corpus_build(
        ids, label="A0 clean-124 split", role="eval", mode="refuse")
    print(f"[a0] guard(role=eval) PASSED on the 124: kept={len(kept)} "
          f"in_parity_train={rec.get('in_parity_train')} "
          f"in_deployed_val={rec.get('in_deployed_val')}")
    # the discriminating control: the FULL 139 must still be refused
    # ⚠️ 124 + 11 = 135, NOT 139: the 4 map-less clips are not in it, and calling
    # this "the full 139" would be a mislabelled control.
    ctrl = sorted(set(ids) | {by_sha.get(s) or m["excluded"][s]
                              for s in m["excluded"]})
    refused = False
    try:
        parity.guard_corpus_build(
            ctrl, label="A0 control: the 124 PLUS the 11 parity-train clips",
            role="eval", mode="refuse")
    except parity.ParityViolation:
        # ⛔ `ParityViolation` derives from **SystemExit**, NOT Exception — deliberately,
        # so a refusal kills the process rather than being swallowed. ⚠️ That means any
        # `except Exception` around a parity guard is a SILENT NO-OP: a caller who
        # believes it is handling a refusal is not. Caught here BY ITS REAL TYPE because
        # this call is a deliberate control that MUST refuse.
        refused = True
    print(f"[a0] CONTROL: the 124 + the 11 parity-train clips at role=eval "
          f"is refused: {refused}")
    if not refused:
        print("ZZA0-ABORT-CONTROL-DID-NOT-REFUSEZZ")
        return 4

    stamp = {
        "_what": "the CLEAN-124 refcv6 eval split, two disjoint 62/62 halves",
        "_evidence_class": "MEASURED (ours; built and re-verified from disk)",
        "n_total": 124, "n_half_a": na, "n_half_b": nb,
        "derivation": "139 B1 eval clips - 11 inside parity TRAIN - 4 with no SAM3 map",
        "exclusions_are_disjoint": True,
        "split_rule": "sorted sha12, ALTERNATING (the same rule the 139 split used)",
        "parity_gate": {"role": "eval", "mode": "refuse", "kept": len(kept),
                        "in_parity_train": rec.get("in_parity_train"),
                        "in_deployed_val": rec.get("in_deployed_val")},
        "control_124_plus_11_refused": refused,
        "every_clip_has_a_sam3_map": True,
        "copies_not_links": "D: is exFAT; os.link/symlink raise WinError 1",
        "halves_sha12": {"halfA": [sha12(c) for c in on_disk[OUT_A.name]],
                         "halfB": [sha12(c) for c in on_disk[OUT_B.name]]},
    }
    for out in (OUT_A, OUT_B):
        (out / "CLEAN124_MANIFEST.json").write_text(
            json.dumps({**stamp, "this_half": out.name}, indent=1),
            encoding="utf-8", newline="\n")
    pathlib.Path("C:/Users/Admin/qland/a0_clean124_stamp.json").write_text(
        json.dumps(stamp, indent=1), encoding="utf-8", newline="\n")
    print("ZZA0-OK 62/62 disjoint, guard passed, control refusedZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
