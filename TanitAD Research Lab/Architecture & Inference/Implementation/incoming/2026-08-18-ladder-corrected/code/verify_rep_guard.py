#!/usr/bin/env python
"""THE REPRODUCTION GATE for the `ll_rep_*` guard refit.

⛔ WHY A GATE AT ALL. The four `llrepG_*.json` files are produced by TODAY's
`ll1_ladder.py`, which has moved since the banked `ll_rep_*.json` were written
(the guard call and a corrected `_solve` docstring landed in between). If the
producer changed in any way that touches the SOLVE, the guard verdicts I am
about to attach to the banked rows would belong to different rows. The only
admissible proof that they do not is a **field-by-field comparison against the
banked artifact** — C91's rule, applied to my own output: open the file, never
the headline.

⚠️ EXIT CODES ARE NOT EVIDENCE (four instances this session). The verdict is the
JSON this writes, and it names every field compared and every divergence.

Compared per (arm × target × seed): every scalar the banked file carries that is
not part of the guard block, i.e. the entire pre-guard record.
"""
from __future__ import annotations
import json
import pathlib
import sys

ROOT = pathlib.Path(
    "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub/"
    "Architecture & Inference/Implementation/incoming")
BANKED = ROOT / "2026-08-17-latent-linear-ladder" / "raw"
NEW = ROOT / "2026-08-18-ladder-corrected" / "raw" / "rep_guard"
OUT = ROOT / "2026-08-18-ladder-corrected" / "raw" / "rep_guard_gate.json"

ARMS = ["s11250", "nullmatched", "orcdir", "proxyv0"]
# fields introduced BY this re-read; everything else must be bit-identical.
GUARD_ONLY = {"k1_guard", "K1_PASSES_GUARDED"}


def same(a, b) -> bool:
    """⚠️ `NaN != NaN` IS NOT A DIVERGENCE — and my first version of this gate
    reported 24 of them as failures. The `C-V0` arm's `corr_partial_v0` is NaN
    BY CONSTRUCTION (partialling `v0` out of a `v0`-only readout leaves nothing
    to correlate), and Python's float identity says a NaN never equals itself.
    ⇒ A gate that cannot distinguish *"the field changed"* from *"the field is
    a NaN in both files"* manufactures a FAIL. Same class as C95: the check
    itself needed testing in the direction it was not written for."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:          # both NaN
            return True
    return a == b


def main() -> int:
    rows, diffs, missing = [], [], []
    for arm in ARMS:
        b_p, n_p = BANKED / f"ll_rep_{arm}.json", NEW / f"llrepG_{arm}.json"
        if not n_p.exists():
            missing.append(str(n_p))
            continue
        b, n = json.loads(b_p.read_text("utf-8")), json.loads(n_p.read_text("utf-8"))
        for tgt, bt in b["targets"].items():
            nt = n["targets"][tgt]
            for seed, bs in bt["per_seed"].items():
                ns = nt["per_seed"][seed]
                for f, bv in bs.items():
                    if f in GUARD_ONLY:
                        continue
                    nv = ns.get(f, "<ABSENT>")
                    ok = same(bv, nv)
                    rows.append(ok)
                    if not ok:
                        diffs.append({"arm": arm, "target": tgt, "seed": seed,
                                      "field": f, "banked": bv, "new": nv})
            # target-level scalars too (gt_sd, seed ranges, n's)
            for f, bv in bt.items():
                if f == "per_seed":
                    continue
                nv = nt.get(f, "<ABSENT>")
                rows.append(same(bv, nv))
                if not same(bv, nv):
                    diffs.append({"arm": arm, "target": tgt, "seed": "-",
                                  "field": f, "banked": bv, "new": nv})
    out = {
        "_evidence_class": "MEASURED (ours; field-by-field vs the banked artifact)",
        "eval_tier": "T0-DIAGNOSTIC",
        "what": "llrepG_* (guard refit, --fit-mode centred) vs banked ll_rep_* ",
        "arms": ARMS, "fields_compared": len(rows),
        "identical": int(sum(rows)), "divergent": len(diffs),
        "missing_inputs": missing,
        "GATE": "PASS" if (not diffs and not missing) else "FAIL",
        "divergences": diffs[:200],
        "note": ("Only the guard block and a docstring separate today's producer "
                 "from the one that wrote the banked file, so every non-guard "
                 "field must be bit-identical. A single divergence means the "
                 "guard verdicts cannot be attached to the banked rows."),
    }
    OUT.write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "divergences"}, indent=1))
    if diffs:
        print("FIRST DIVERGENCES:", json.dumps(diffs[:5], indent=1))
    return 0 if out["GATE"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
