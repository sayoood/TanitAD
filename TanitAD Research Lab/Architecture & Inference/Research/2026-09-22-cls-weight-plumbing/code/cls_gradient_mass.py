"""What does inverse-frequency weighting ACTUALLY do to the `cls` term? An exact answer, no GPU.

⛔ WHY THIS EXISTS. `H-BOXCLS-1`'s RESULT names an open question and leaves it: *"whether full
inverse frequency is the right strength is itself an open question -- a milder exponent is the
obvious next arm and is not pre-registered here."* RULE ZERO says a turn that names the next
lever and does not run it is unfinished. This runs the part that needs no GPU, and it turns out
to be the decisive part.

⭐ THE STRUCTURAL FACT FIRST, BECAUSE IT NARROWS THE QUESTION TO ONE VARIABLE.
For a single slot with target class `c`, weighted cross-entropy is `w[c] * CE`, so

    d/d(logits) of (w[c] * CE)  =  w[c] * (softmax - onehot)

-- the weight is a SCALAR multiplying that slot's whole gradient. ⇒ **weighting changes no
slot's gradient DIRECTION whatsoever.** It changes only how much each slot counts against the
others in the sum. And because `slot_set_loss`'s denominator FOLLOWS the weight, the term's
total scale is fixed too. So the entire intervention is a redistribution of GRADIENT MASS across
classes, and that redistribution is computable EXACTLY from the census counts alone -- no model,
no checkpoint, no forward pass, no RNG.

⚠️ WHAT THIS CANNOT SAY. It does not say the re-weighting fixes the collapse; that needs a
trained arm and the PI's GPU call. It says exactly what the lever DOES, which is the input to
deciding whether to spend the GPU, and it gives the milder-exponent family an exact
parametrisation instead of a hunch.

⛔ CPU only, read-only, reads the banked census.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Projects/TanitAD/stack")

from tanitad.models.agent_slots import AGENT_CLASSES, load_cls_class_weight   # noqa: E402

ART = pathlib.Path("D:/Projects/TanitAD/stack/tanitad/data/agent_cls_weights_train2400.json")


def mass_shares(counts: dict, weights: dict) -> dict:
    """Each class's share of the class term's total gradient mass.

    Slot `i` with target `c` contributes `w[c]` to the numerator's weight; there are
    `counts[c]` such slots, so class `c` contributes `counts[c] * w[c]` of the total.
    ⚠️ This is the share of the WEIGHTED SUM, which is what the optimiser sees -- it is not
    the share of the *loss value*, which additionally depends on each class's own -log p.
    """
    m = {c: counts[c] * weights[c] for c in counts}
    tot = sum(m.values())
    return {c: m[c] / tot for c in counts}


def main() -> int:
    art = json.loads(ART.read_text(encoding="utf-8"))
    counts = {c: int(art["counts"][c]) for c in AGENT_CLASSES}
    vec, stamp = load_cls_class_weight()
    w_inv = {c: float(v) for c, v in zip(AGENT_CLASSES, vec.tolist())}
    w_uni = {c: 1.0 for c in AGENT_CLASSES}

    before = mass_shares(counts, w_uni)
    after = mass_shares(counts, w_inv)

    # --- the identity control: at alpha = 1 every class must land on EXACTLY 1/C ----------- #
    # ⛔ w proportional to 1/count  =>  count * w is CONSTANT  =>  every class gets 1/C.
    # That is an analytic target, not a re-run of the producer's own arithmetic, and it is the
    # kind of check `CLAUDE.md` names as the strongest available discriminator.
    C = len(AGENT_CLASSES)
    max_dev = max(abs(after[c] - 1.0 / C) for c in AGENT_CLASSES)

    # ⚠️ ...AND THE FIRST RUN OF THIS CONTROL FAILED ITS OWN THRESHOLD, WHICH IS THE USEFUL PART.
    # The deviation came back 5.5e-06, not machine precision, because the BANKED vector stores
    # its weights ROUNDED TO 6 dp. The mathematics is exact; the ARTIFACT is not. Separating
    # the two is the whole point -- `CLAUDE.md` carries the sibling case where a label builder
    # rounded its ladder to 4 dp and then verified the shipped buckets AGAINST THAT SAME ROUNDED
    # LADDER, reading "0 % moved" while 57.5 % of the corpus moved against an independently
    # derived ladder. `EXACT` below is derived from the COUNTS, never from the stored weights.
    raw_exact = {c: 1.0 / counts[c] for c in AGENT_CLASSES}
    mean_exact = sum(raw_exact.values()) / C
    w_exact = {c: raw_exact[c] / mean_exact for c in AGENT_CLASSES}
    after_exact = mass_shares(counts, w_exact)
    max_dev_exact = max(abs(after_exact[c] - 1.0 / C) for c in AGENT_CLASSES)

    rows = []
    for c in sorted(AGENT_CLASSES, key=lambda k: -counts[k]):
        rows.append({"class": c, "count": counts[c],
                     "mass_share_unweighted": round(before[c], 6),
                     "mass_share_inv_freq": round(after[c], 6),
                     "fold_change": round(after[c] / before[c], 1)})

    # --- the milder-exponent family, parametrised exactly --------------------------------- #
    # w ∝ count^(-alpha), normalised to mean 1  =>  class mass share ∝ count^(1 - alpha).
    # alpha = 0 is the status quo; alpha = 1 is this arm; anything between is a named arm
    # rather than a hunch. ⭐ THIS IS THE "NEXT LEVER", LEFT BEHIND WITH ITS ARITHMETIC DONE.
    sweep = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        raw = {c: counts[c] ** (-alpha) for c in AGENT_CLASSES}
        mean = sum(raw.values()) / C
        wa = {c: raw[c] / mean for c in AGENT_CLASSES}
        sh = mass_shares(counts, wa)
        sweep.append({
            "alpha": alpha,
            "weight_ratio_majority_to_rarest": round(
                max(wa.values()) / min(wa.values()), 1),
            "mass_share_automobile": round(sh["automobile"], 6),
            "mass_share_animal": round(sh["animal"], 6),
            "animal_fold_vs_alpha0": round(sh["animal"] / before["animal"], 1)})

    res = {
        "_what": ("exactly what inverse-frequency weighting does to the cls term: it "
                  "redistributes GRADIENT MASS across classes and changes no slot's gradient "
                  "direction"),
        "_evidence_class": "MEASURED (ours) / ANALYTIC -- CPU, no checkpoint, no forward pass",
        "_source_counts": "stack/tanitad/data/agent_cls_weights_train2400.json (TRAIN census)",
        "_digest": stamp["digest"],
        "_structural_fact": (
            "d/d(logits)[w[c] * CE] = w[c] * (softmax - onehot): the weight is a scalar on the "
            "slot's whole gradient, so DIRECTION is unchanged and only the slot's relative "
            "contribution moves. The weight-following denominator fixes the total scale."),
        "n_classes": C,
        "identity_control": {
            "_target": ("at alpha=1, count*w is constant, so every class share is EXACTLY 1/C. "
                        "An ANALYTIC target, not a re-run of the producer's own arithmetic."),
            "expected_share": 1.0 / C,
            "max_abs_deviation_EXACT_weights_from_counts": max_dev_exact,
            "max_abs_deviation_BANKED_6dp_weights": max_dev,
            "_reading": (
                "the identity holds to machine precision on weights derived from the counts, "
                "and to ~5.5e-06 on the BANKED vector because it stores 6 dp. That gap is the "
                "ARTIFACT's precision, not a flaw in the mathematics -- and quantifying it is "
                "why the target is derived from counts rather than from the stored weights."),
            "banked_precision_is_negligible_for_this_use": bool(max_dev < 1e-4),
            "PASS": bool(max_dev_exact < 1e-12 and max_dev < 1e-4)},
        "per_class": rows,
        "headline": {
            "automobile_before": round(before["automobile"], 6),
            "automobile_after": round(after["automobile"], 6),
            "animal_before": round(before["animal"], 6),
            "animal_after": round(after["animal"], 6),
            "animal_fold_increase": round(after["animal"] / before["animal"], 1)},
        "alpha_sweep": sweep,
    }
    res["_VERDICT"] = (
        "⭐ THE LEVER IS A PURE REDISTRIBUTION, AND ITS SIZE IS EXACT. Unweighted, `automobile` "
        "carries %.3f%% of the cls term's gradient mass and `animal` carries %.4f%%. At full "
        "inverse frequency every class carries EXACTLY %.1f%% (max deviation %.1e, an identity, "
        "not a fit) -- a %.0fx increase for `animal` and a %.1fx cut for `automobile`. ⛔ No "
        "slot's gradient direction changes; only which slots dominate the sum. ⇒ the "
        "milder-exponent family is parametrised exactly by w ∝ count^(-alpha), mass share ∝ "
        "count^(1-alpha), and alpha is the ONE knob a follow-up arm would move."
        % (100 * before["automobile"], 100 * before["animal"], 100.0 / C, max_dev_exact,
           after["animal"] / before["animal"], before["automobile"] / after["automobile"]))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print()
    print(res["_VERDICT"])
    pathlib.Path("C:/Users/Admin/qland/work/pbox/cls_gradient_mass.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if res["identity_control"]["PASS"] else 4


if __name__ == "__main__":
    sys.exit(main())
