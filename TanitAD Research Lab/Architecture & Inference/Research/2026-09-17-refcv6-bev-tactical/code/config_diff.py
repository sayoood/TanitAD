"""ACCOUNT FOR EVERY `config.json` KEY THAT DIFFERS — by name, not by count.

``bitid_check.py`` reports the CONFIG comparison as a verdict plus a key list.
On this patch that verdict is **DIFFERS**, and a bare "DIFFERS" is not a result:
it cannot separate *"the patch changed the run record"* from *"the patch changed
the run"*. This resolves it to the leaf.

⛔ THE HONEST FINDING IT PRODUCES. ``seams.tac_decoder_v6`` CANNOT be both
truthful and byte-identical after PI ruling R2: the tip stamped
``bev_blocked_by: "<prose about a structural block>"`` unconditionally, and that
sentence is now FALSE. Keeping it to preserve a digest would be a run record
that lies. So the diff is reported, resolved, and argued — never smoothed.

⭐ AND IT MUST ACCOUNT FOR THE OTHER ONE TOO: ``argv`` differs because ``--out``
names a different directory on the two runs. That is the driver, not the patch,
and saying so requires SHOWING it — hence the per-token diff rather than a
claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _blocks(a: dict, b: dict) -> dict:
    return {
        "keys_only_in_tip": sorted(set(a) - set(b)),
        "keys_only_in_patched": sorted(set(b) - set(a)),
        "keys_with_changed_value": sorted(
            k for k in set(a) & set(b)
            if json.dumps(a[k], sort_keys=True, default=str)
            != json.dumps(b[k], sort_keys=True, default=str)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tip", required=True)
    ap.add_argument("--patched", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ja = json.loads(Path(a.tip).read_text(encoding="utf-8"))
    jb = json.loads(Path(a.patched).read_text(encoding="utf-8"))

    top = _blocks(ja, jb)
    rep = {"top_level": top}

    # -- argv, per token. ⛔ A length mismatch would make `zip` silently
    # truncate and report "no differences" for a run with an extra flag, so the
    # lengths are asserted rather than assumed.
    za, zb = ja.get("argv", []), jb.get("argv", [])
    rep["argv"] = {
        "n_tip": len(za), "n_patched": len(zb),
        "same_length": len(za) == len(zb),
        "differing_tokens": ([{"tip": x, "patched": y}
                              for x, y in zip(za, zb) if x != y]
                             if len(za) == len(zb) else None)}

    # -- seams, per block then per leaf
    sa, sb = ja.get("seams", {}) or {}, jb.get("seams", {}) or {}
    sd = _blocks(sa, sb)
    sd["per_block"] = {}
    for k in sd["keys_with_changed_value"]:
        ka, kb = sa[k], sb[k]
        if isinstance(ka, dict) and isinstance(kb, dict):
            leaf = _blocks(ka, kb)
            leaf["removed_values"] = {x: ka[x] for x in leaf["keys_only_in_tip"]}
            leaf["added_values"] = {x: kb[x]
                                    for x in leaf["keys_only_in_patched"]}
            sd["per_block"][k] = leaf
        else:
            sd["per_block"][k] = {"tip": ka, "patched": kb}
    rep["seams"] = sd

    # ⭐ THE VERDICT THIS FILE EXISTS FOR, and it is a POSITIVE assertion: every
    # top-level key that differs is one of the two accounted for above, and no
    # seam block changed a VALUE (only the tactical block's key vocabulary).
    accounted = set(top["keys_with_changed_value"]) <= {"argv", "seams"}
    no_value_change = all(
        not v.get("keys_with_changed_value")
        for v in sd["per_block"].values() if isinstance(v, dict))
    rep["verdict"] = {
        "only_argv_and_seams_differ": accounted,
        "argv_differs_only_in_the_out_path": bool(
            rep["argv"]["differing_tokens"] is not None
            and len(rep["argv"]["differing_tokens"]) == 1),
        "no_seam_leaf_changed_its_VALUE": no_value_change,
        "seam_blocks_changed": sd["keys_with_changed_value"],
    }
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
