"""BIT-IDENTITY PROOF, on the ARTIFACTS -- never on an exit code.

Compares the tip trainer's run against the patched trainer's, both with the
perception weights at their 0.0 default and the same seed:

  1. metrics.jsonl  -- BYTE identical (sha256 over raw bytes);
  2. ckpt.pt        -- every tensor, element-by-element, bitwise
                       (`torch.equal` on the int view, so a NaN compares equal
                       to a NaN in the same slot and -0.0 does NOT compare
                       equal to +0.0);
  3. config.json    -- no `refcv6_perception` block, and no other key changed;
  4. metrics keys   -- no `map*` / `box3d*` / `ga_*` key anywhere.

⛔ It asserts the SHAPE of every comparison first: a missing file, an empty
digest or a zero-tensor count is reported INCONCLUSIVE, never MATCH. Both sides
come through the same filesystem, so "both empty" is a failure mode that looks
exactly like agreement.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

def _bytes(t):
    """Raw byte view, shape-agnostic: a 0-dim tensor cannot be `.view`ed, so
    it is reshaped to [1] first. ⛔ Byte-level, never `torch.equal` on floats
    -- that says NaN != NaN (a false DIFFERS) and +0.0 == -0.0 (a false MATCH),
    and this comparison must be wrong in neither direction."""
    t = t.detach().contiguous().reshape(-1)
    return t.view(torch.uint8) if t.is_floating_point() else t


ROOT = Path(sys.argv[1])
A, B = ROOT / "tip", ROOT / "patched"
rep = {"root": str(ROOT), "tip": str(A), "patched": str(B)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- 1. metrics.jsonl bytes -------------------------------------------------
ma, mb = A / "metrics.jsonl", B / "metrics.jsonl"
if not (ma.is_file() and mb.is_file()):
    rep["metrics"] = {"verdict": "INCONCLUSIVE", "why": "a metrics.jsonl is missing"}
else:
    ha, hb = sha(ma), sha(mb)
    na, nb = ma.stat().st_size, mb.stat().st_size
    rep["metrics"] = {
        "tip_sha256": ha, "patched_sha256": hb,
        "tip_bytes": na, "patched_bytes": nb,
        "tip_rows": len(ma.read_text(encoding="utf-8").strip().splitlines()),
        "verdict": ("INCONCLUSIVE" if min(na, nb) == 0 or len(ha) != 64
                    else ("BYTE-IDENTICAL" if ha == hb else "DIFFERS"))}
    # the positive control: the key set, and the ABSENCE of the new keys
    keys = set()
    for line in mb.read_text(encoding="utf-8").strip().splitlines():
        keys |= set(json.loads(line))
    bad = sorted(k for k in keys
                 if k.startswith(("map", "box3d", "ga_", "n_map")))
    rep["metrics"]["patched_key_count"] = len(keys)
    rep["metrics"]["perception_keys_present"] = bad
    rep["metrics"]["has_loss_key"] = "loss" in keys      # same-breath control

# --- 2. ckpt.pt tensors -----------------------------------------------------
ca, cb = A / "ckpt.pt", B / "ckpt.pt"
if not (ca.is_file() and cb.is_file()):
    rep["ckpt"] = {"verdict": "INCONCLUSIVE", "why": "a ckpt.pt is missing"}
else:
    sa = torch.load(ca, map_location="cpu", weights_only=False)["model"]
    sb = torch.load(cb, map_location="cpu", weights_only=False)["model"]
    ka, kb = sorted(sa), sorted(sb)
    n_diff = n_cmp = 0
    diffs = []
    for k in ka:
        if k not in sb:
            continue
        x, y = sa[k], sb[k]
        if x.shape != y.shape:
            diffs.append({"key": k, "why": "shape",
                          "tip": list(x.shape), "patched": list(y.shape)})
            n_diff += 1
            continue
        n_cmp += int(x.numel())
        # BITWISE: compare the raw byte views, so NaN == NaN in the same slot
        # and +0.0 != -0.0 -- `torch.equal` on floats would say both are fine.
        bx, by = _bytes(x), _bytes(y)
        if not torch.equal(bx, by):
            d = int((bx != by).sum())
            diffs.append({"key": k, "why": "bytes", "n_bytes_differ": d})
            n_diff += 1
    rep["ckpt"] = {
        "tip_n_tensors": len(ka), "patched_n_tensors": len(kb),
        "keys_only_in_tip": sorted(set(ka) - set(kb)),
        "keys_only_in_patched": sorted(set(kb) - set(ka)),
        "n_elements_compared": n_cmp,
        "n_tensors_bitwise_different": n_diff,
        "diffs": diffs[:10],
        "verdict": ("INCONCLUSIVE" if n_cmp == 0 or len(ka) == 0
                    else ("BITWISE-IDENTICAL" if (n_diff == 0
                                                  and ka == kb) else "DIFFERS")),
        # the DISCRIMINATING control: this comparison must be able to FAIL.
        # Perturb one element of one tensor and re-run the same predicate.
        "mutation_control": None}
    # ⛔ THE MUTATION CONTROL, AND IT MUST BE ABLE TO FAIL.
    # ⚠️ A `+ 1e-7` perturbation is NOT a valid mutation: on a float32 whose
    # magnitude is ~1 it is BELOW the ULP and `x + 1e-7 == x` exactly, so the
    # control reported `detects: false` on a comparison that is perfectly
    # sound -- a check sharing the defect it checks for, caught here.
    # ⇒ Flip ONE BIT of the byte view: representable by construction, on every
    # dtype, and the smallest change the predicate must see. Run over EVERY
    # tensor, so "it works on the one I happened to pick" is not the claim.
    n_detect = n_try = 0
    for k in ka:
        b = _bytes(sa[k]).clone()
        if not b.numel():
            continue
        n_try += 1
        b[0] = b[0] ^ 1
        if not torch.equal(_bytes(sa[k]), b):
            n_detect += 1
    rep["ckpt"]["mutation_control"] = {
        "mutation": "flip the low bit of byte 0 of each tensor",
        "n_tensors_mutated": n_try, "n_detected": n_detect,
        "verdict": ("INCONCLUSIVE" if n_try == 0 else
                    ("DETECTS-EVERY-ONE-BIT-FLIP" if n_detect == n_try
                     else "BLIND-TO-%d-OF-%d" % (n_try - n_detect, n_try)))}

# --- 3. config.json ---------------------------------------------------------
ga, gb = A / "config.json", B / "config.json"
if not (ga.is_file() and gb.is_file()):
    rep["config"] = {"verdict": "INCONCLUSIVE", "why": "a config.json is missing"}
else:
    ja = json.loads(ga.read_text(encoding="utf-8"))
    jb = json.loads(gb.read_text(encoding="utf-8"))
    only_b = sorted(set(jb) - set(ja))
    only_a = sorted(set(ja) - set(jb))
    changed = sorted(k for k in set(ja) & set(jb)
                     if json.dumps(ja[k], sort_keys=True, default=str)
                     != json.dumps(jb[k], sort_keys=True, default=str))
    rep["config"] = {
        "keys_only_in_patched": only_b, "keys_only_in_tip": only_a,
        "keys_with_changed_value": changed,
        "refcv6_perception_in_patched": jb.get("refcv6_perception", "<absent>"),
        "trunk_pretrained_tip": ja.get("trunk_pretrained"),
        "trunk_pretrained_patched": jb.get("trunk_pretrained"),
        "n_keys_tip": len(ja), "n_keys_patched": len(jb),
        "verdict": ("INCONCLUSIVE" if not ja or not jb else
                    ("SAME-KEYS-SAME-VALUES"
                     if not only_a and not only_b and not changed
                     else ("ONLY-refcv6_perception-NULL-ADDED"
                           if only_b == ["refcv6_perception"] and not only_a
                           and not changed
                           and jb.get("refcv6_perception") is None
                           else "DIFFERS")))}

# --- 4. THE DISCRIMINATING CONTROL: the same trainer, run twice ------------
# ⭐ Without it "metrics.jsonl DIFFERS" is unreadable: it cannot separate a
# code change from a wall-clock field. A field that differs between tip and
# tip2 is not evidence about the patch.
C = ROOT / "tip2"


def _fields(p: Path):
    return [json.loads(x) for x in
            p.read_text(encoding="utf-8").strip().splitlines()]


def _fdiff(p, q):
    out = set()
    for ra, rb in zip(_fields(p), _fields(q)):
        out |= {k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k)}
    return sorted(out)


if (C / "metrics.jsonl").is_file() and ma.is_file() and mb.is_file():
    same = _fdiff(ma, C / "metrics.jsonl")
    patch = _fdiff(ma, mb)
    rep["metrics_field_control"] = {
        "tip_vs_tip2_fields_differing": same,
        "tip_vs_patched_fields_differing": patch,
        "n_fields_per_row": len(_fields(ma)[0]),
        "verdict": ("IDENTICAL-MODULO-WALLCLOCK"
                    if patch and set(patch) <= set(same)
                    else ("IDENTICAL" if not patch else "DIFFERS"))}
else:
    rep["metrics_field_control"] = {"verdict": "INCONCLUSIVE",
                                    "why": "tip2 control run missing"}

print(json.dumps(rep, indent=1))
