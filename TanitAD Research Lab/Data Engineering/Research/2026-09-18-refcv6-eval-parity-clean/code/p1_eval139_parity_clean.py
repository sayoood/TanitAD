"""Produce and PROVE a parity-clean refcv6 eval clip set from the 139 B1 ids.

D-REFCV6-EVAL139-PARITY. The 139 clips of the refcv6 s10.6 coverage rig are NOT
a held-out set: some of them live inside the parity TRAIN corpus
``physicalai-train-e438721ae894``. s10.6 was a COVERAGE pass, so no landed claim
moves -- but any refcv6 arm SCORED on that set as held-out must exclude them.

What this script establishes, in order, and why each step is load-bearing:

  A. the input set (n, uniqueness) -- so the denominator is not taken on trust;
  B. BOTH overlap directions re-derived from the committed per-clip sha256
     oracles (deployed val40, parity train) -- the count in the brief is
     re-measured here, never inherited;
  C. the gate REFUSES the 139 at role="eval" and PASSES them at role=""
     (the two directions are different questions, s10c role table);
  D. the clean list via mode="exclude";
  E. ** THE ROUND TRIP ** -- re-running the gate on the clean list at
     role="eval" must PASS. A count alone is not a proof: a filter that
     produced the right NUMBER by the wrong rule would still be wrong here;
  F. ** THE DISCRIMINATING CONTROL ** -- adding back ANY ONE excluded id must
     make the same gate REFUSE again (all excluded ids tried, one at a time),
     while adding a synthetic id that is in neither oracle must still PASS.
     A filter that cannot be made to fail has not been shown to filter.
  G. the per-half clean n for the two 416x1024 half-caches, because a panel
     that evaluates per half needs its own clean denominator.

CONFIDENTIALITY: clip ids are gated-confidential. Everything written into the
repo is sha12 = sha256(clip_id)[:12] (tanitad.data.semantic_map_gt.sha12).
The sha12 -> clip_id mapping is written OUT of the repo, to --map-out.

Run (dev box):
  set PYTHONPATH=C:\\Users\\Admin\\tanitad-wt-bevtac\\stack
  set PYTHONIOENCODING=utf-8
  python p1_eval139_parity_clean.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tanitad.data import parity
from tanitad.data.semantic_map_gt import sha12

LABEL = "refcv6-eval139"
DEFAULT_IDS = Path("C:/Users/Admin/qland/ids139.txt")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "raw"
DEFAULT_MAP = Path("C:/Users/Admin/qland/refcv6_eval128_sha12_map.json")
HALVES = {
    "splitA": Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl-splitA"),
    "splitB": Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl-splitB"),
    "full": Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"),
}


def _gate(ids, *, role, mode):
    """(ok, kept, record_or_message). Never lets a ParityViolation escape."""
    try:
        kept, rec = parity.guard_corpus_build(ids, label=LABEL, role=role, mode=mode)
        return True, kept, rec
    except parity.ParityViolation as exc:
        return False, None, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--map-out", type=Path, default=DEFAULT_MAP)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    R: dict = {
        "label": LABEL,
        "decision": "D-REFCV6-EVAL139-PARITY",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_class": "MEASURED (this script)",
        "ids_source": str(a.ids),
        "parity_module": parity.__file__,
    }

    # -- A. the input set -------------------------------------------------- #
    raw_lines = [ln.strip() for ln in a.ids.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in raw_lines if ln]
    ids = sorted(set(lines))
    R["input"] = {
        "n_lines_nonempty": len(lines),
        "n_unique": len(ids),
        "duplicates": len(lines) - len(ids),
        "ids_file_sha256": hashlib.sha256(a.ids.read_bytes()).hexdigest(),
    }
    print(f"[A] input: {len(lines)} lines, {len(ids)} unique clip ids")

    # -- oracle provenance (a guard whose oracle is empty is not a guard) --- #
    oracles = parity.require_ingest_gate("p1_eval139_parity_clean")
    R["oracles"] = {
        "parity_train_clips": oracles["parity_train_clips"],
        "deployed_val_clips": oracles["deployed_val_clips"],
        "parity_train_key": parity.PARITY_TRAIN_KEY,
        "train_digest_file": str(parity.CLIP_DIGESTS_PATH),
        "val_digest_file": str(parity.DEPLOYED_VAL_DIGESTS_PATH),
        "membership": "per-clip sha256(clip_id), self-checked by load_clip_digests",
    }
    print(f"[A] oracles: {oracles}")

    # -- B. BOTH directions, re-derived ------------------------------------ #
    in_val = parity.clips_in_deployed_val(ids)
    in_train = parity.clips_in_parity_train(ids)
    try:
        in_v72 = parity.clips_in_v72_eval(ids)
        v72_err = None
    except Exception as exc:                      # oracle may not be on this host
        in_v72, v72_err = [], f"{type(exc).__name__}: {exc}"
    R["overlaps"] = {
        "n_in_deployed_val40": len(in_val),
        "n_in_parity_train": len(in_train),
        "n_in_v72_eval_split": len(in_v72),
        "v72_oracle_error": v72_err,
        "n_clean_expected": len(ids) - len(in_train),
    }
    print(f"[B] in deployed-val40 = {len(in_val)} | in parity-train = "
          f"{len(in_train)} | in v7.2 eval = {len(in_v72)}")

    # -- C. the gate on the FULL 139, both roles --------------------------- #
    ok_eval, _, rec_eval = _gate(ids, role="eval", mode="refuse")
    ok_sup, _, rec_sup = _gate(ids, role="", mode="refuse")
    R["gate_on_139"] = {
        "role_eval_refuse": {
            "passed": ok_eval,
            "expected": "REFUSE (parity TRAIN clips are inside a held-out set)",
            "record": rec_eval if ok_eval else None,
            "violation_excerpt": None if ok_eval else rec_eval.strip().splitlines()[2:7],
        },
        "role_undeclared_refuse": {
            "passed": ok_sup,
            "expected": "PASS (no deployed-val40 episode is inside)",
            "record": rec_sup if ok_sup else None,
        },
    }
    print(f"[C] role='eval' on 139 -> {'PASS' if ok_eval else 'REFUSED'} (want REFUSED)")
    print(f"[C] role=''     on 139 -> {'PASS' if ok_sup else 'REFUSED'} (want PASS)")

    # -- D. the clean list ------------------------------------------------- #
    kept, rec_excl = parity.guard_corpus_build(
        ids, label=LABEL, role="eval", mode="exclude")
    dropped = sorted(set(ids) - set(kept))
    R["exclude"] = {"record": rec_excl, "n_kept": len(kept), "n_dropped": len(dropped)}
    print(f"[D] exclude -> kept {len(kept)}, dropped {len(dropped)}")

    # -- E. THE ROUND TRIP ------------------------------------------------- #
    ok_rt, kept_rt, rec_rt = _gate(kept, role="eval", mode="refuse")
    rt_ok = bool(ok_rt and rec_rt["disjoint"] and rec_rt["n_disqualifying"] == 0
                 and rec_rt["in_parity_train"] == 0 and rec_rt["kept"] == len(kept)
                 and rec_rt["decision_grade"] is True
                 and sorted(kept_rt) == sorted(kept))
    R["round_trip"] = {
        "passed": bool(ok_rt), "assertions_ok": rt_ok, "record": rec_rt if ok_rt else None,
        "claim": "re-running role='eval' on the CLEAN list PASSES with 0 "
                 "disqualifying clips and returns the same ids",
    }
    print(f"[E] ROUND TRIP role='eval' on the clean list -> "
          f"{'PASS' if ok_rt else 'REFUSED'} (want PASS); assertions_ok={rt_ok}")

    # -- F. THE DISCRIMINATING CONTROL ------------------------------------- #
    readd = []
    for cid in dropped:
        ok, _, msg = _gate(list(kept) + [cid], role="eval", mode="refuse")
        readd.append({"sha12": sha12(cid), "gate_passed": ok,
                      "n_in": len(kept) + 1,
                      "outcome": "PASS (BAD)" if ok else "REFUSED (expected)"})
    n_refused = sum(1 for r in readd if not r["gate_passed"])
    # negative control: an id in NEITHER oracle must NOT trip the same gate.
    synth = ["00000000-0000-4000-8000-%012d" % i for i in range(3)]
    synth = [s for s in synth if not parity.clips_in_parity_train([s])
             and not parity.clips_in_deployed_val([s])]
    ok_syn, _, rec_syn = _gate(list(kept) + synth, role="eval", mode="refuse")
    R["control"] = {
        "readd_one_excluded_each": readd,
        "n_readd_trials": len(readd),
        "n_refused": n_refused,
        "all_refused": n_refused == len(readd) and len(readd) > 0,
        "negative_control": {
            "n_synthetic_ids_added": len(synth),
            "gate_passed": ok_syn,
            "expected": "PASS (ids in neither oracle are not a leak)",
            "record": rec_syn if ok_syn else None,
        },
    }
    print(f"[F] control: {n_refused}/{len(readd)} single-readd trials REFUSED "
          f"(want all); +{len(synth)} synthetic ids -> "
          f"{'PASS' if ok_syn else 'REFUSED'} (want PASS)")

    # -- G. the two 416x1024 half-caches ----------------------------------- #
    dropset, keptset, idset = set(dropped), set(kept), set(ids)
    halves = {}
    for name, d in HALVES.items():
        if not d.exists():
            halves[name] = {"dir": str(d), "exists": False}
            continue
        present = parity.v2_clip_ids(d)
        bad = sorted(set(present) & dropset)
        okh, _, rech = _gate(present, role="eval", mode="refuse")
        halves[name] = {
            "dir": str(d), "exists": True,
            "n_clips_present": len(present),
            "n_excluded_present": len(bad),
            "n_clean": len(present) - len(bad),
            "n_present_in_ids139": len(set(present) & idset),
            "n_present_not_in_ids139": len(set(present) - idset),
            "gate_role_eval_passed": okh,
            "gate_record": rech if okh else None,
            "excluded_sha12": [sha12(c) for c in bad],
            "clean_sha12": sorted(sha12(c) for c in set(present) & keptset),
        }
        print(f"[G] {name}: {len(present)} clips, {len(bad)} excluded, "
              f"CLEAN n = {len(present) - len(bad)}")
    a_ids = set(parity.v2_clip_ids(HALVES['splitA'])) if HALVES['splitA'].exists() else set()
    b_ids = set(parity.v2_clip_ids(HALVES['splitB'])) if HALVES['splitB'].exists() else set()
    f_ids = set(parity.v2_clip_ids(HALVES['full'])) if HALVES['full'].exists() else set()
    R["half_caches"] = halves
    R["half_cache_algebra"] = {
        "n_A": len(a_ids), "n_B": len(b_ids), "n_A_inter_B": len(a_ids & b_ids),
        "n_A_union_B": len(a_ids | b_ids), "n_full_dir": len(f_ids),
        "A_union_B_equals_ids139": (a_ids | b_ids) == idset,
        "n_union_minus_ids139": len((a_ids | b_ids) - idset),
        "n_ids139_minus_union": len(idset - (a_ids | b_ids)),
        "n_full_minus_ids139": len(f_ids - idset),
        "n_ids139_minus_full": len(idset - f_ids),
    }
    print(f"[G] algebra: A={len(a_ids)} B={len(b_ids)} A&B={len(a_ids & b_ids)} "
          f"A|B={len(a_ids | b_ids)} full={len(f_ids)}")

    # -- verdict ----------------------------------------------------------- #
    R["verdict"] = {
        "quotable_heldout_n": len(kept),
        "never_quote_n": len(ids),
        "proof_complete": bool(
            (not ok_eval) and ok_sup and rt_ok
            and R["control"]["all_refused"] and ok_syn
            and len(kept) == len(ids) - len(in_train)),
    }

    # -- artifacts --------------------------------------------------------- #
    (a.out / "eval128_clean_sha12.txt").write_text(
        "\n".join(sorted(sha12(c) for c in kept)) + "\n", encoding="utf-8")
    (a.out / "eval139_excluded_sha12.txt").write_text(
        "\n".join(sorted(sha12(c) for c in dropped)) + "\n", encoding="utf-8")
    R["artifacts"] = {
        "clean_sha12_txt": str(a.out / "eval128_clean_sha12.txt"),
        "excluded_sha12_txt": str(a.out / "eval139_excluded_sha12.txt"),
        "sha12_to_clip_id_map_OUT_OF_REPO": str(a.map_out),
        "sha12_rule": "sha256(clip_id).hexdigest()[:12] "
                      "(tanitad.data.semantic_map_gt.sha12)",
    }
    (a.out / "parity_clean_report.json").write_text(
        json.dumps(R, indent=2, sort_keys=False), encoding="utf-8")

    a.map_out.parent.mkdir(parents=True, exist_ok=True)
    a.map_out.write_text(json.dumps({
        "_warning": "GATED-CONFIDENTIAL clip ids. Never commit this file.",
        "label": LABEL, "generated_utc": R["generated_utc"],
        "sha12_rule": "sha256(clip_id).hexdigest()[:12]",
        "clean_128": {sha12(c): c for c in sorted(kept)},
        "excluded": {sha12(c): c for c in sorted(dropped)},
        "splitA_clean": {sha12(c): c for c in sorted(a_ids & keptset)},
        "splitB_clean": {sha12(c): c for c in sorted(b_ids & keptset)},
    }, indent=2), encoding="utf-8")

    print(f"\nVERDICT: quote {len(kept)}, never {len(ids)}. "
          f"proof_complete={R['verdict']['proof_complete']}")
    print(f"report -> {a.out / 'parity_clean_report.json'}")
    print(f"mapping (OUT OF REPO) -> {a.map_out}")
    return 0 if R["verdict"]["proof_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
