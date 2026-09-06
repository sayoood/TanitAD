"""P4 -- the STRATEGIC token heads. Mutation + control validation.

Executes PREREG.md section 3 (the parts answerable without a tiny-rig arm):
  A1/A3  support + provenance, from the banked labels        (see p4_label_probe)
  R      deliberate regression: SHUFFLED labels MUST land at the
         no-information macro recall 1/K -- if a head trained on shuffled
         labels scores above chance, the evaluation leaks and P4 is VOID
  controls  majority-class control MUST read macro recall exactly 1/K
            gradient reachability: p.grad is NOT None at w = 0.0
            off-vocabulary token MUST raise, never encode to -1
            head-width mismatch MUST raise

⛔ THE MUTATION REQUIREMENT. Guards need mutation, not inspection: an AST
census once read 0 suspects on BOTH the fixed and the broken trainer. So every
refusal below is proven REACHABLE by actually tripping it.

ASCII ONLY in print() -- cp1252 dev box.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from tanitad.data import v7_labels as v7l
from tanitad.refs import refc_strategic as st

D_CTX = 32


def banner(s):
    print("")
    print("-" * 78)
    print(s)
    print("-" * 78)


def main(out_path=None):
    torch.manual_seed(0)
    res = {}
    print("=" * 78)
    print("P4 VALIDATION -- THE 15-TOKEN STRATEGIC VOCABULARY")
    print("=" * 78)
    kg = len(v7l.HEADS["str_goal"])
    ka = len(v7l.HEADS["str_action"])
    print("head widths from v7_labels.HEADS: str_goal=%d  str_action=%d "
          " (total %d tokens)" % (kg, ka, kg + ka))
    assert kg == 8 and ka == 7, "vocabulary drifted from 8+7"

    # ================= CONTROL: the refusals must be REACHABLE ============ #
    banner("CONTROLS -- every refusal proven REACHABLE by tripping it")

    # (1) off-vocabulary token must RAISE, not encode to -1
    try:
        st.encode_targets(["REDUCE_TO_FOLLOW_ROUTE"], "str_action")
        off_ok = False
        print("[CONTROL off-vocab] FAIL -- the off-vocabulary token was "
              "ACCEPTED; 11.24 %% of a_str would be silently deleted")
    except st.OffVocabularyToken as e:
        off_ok = True
        print("[CONTROL off-vocab] PASS -- REDUCE_TO_FOLLOW_ROUTE raises "
              "OffVocabularyToken (reachable)")
        print("    %s" % str(e).split(". ")[0][:110])
    # ...and the audit path returns -1, proving the DANGEROUS branch exists too
    lax = st.encode_targets(["REDUCE_TO_FOLLOW_ROUTE"], "str_action",
                            strict=False)
    print("[CONTROL off-vocab] strict=False returns %d (the silent-delete "
          "behaviour, opt-in and named): %s"
          % (int(lax[0]), "PASS" if int(lax[0]) == -1 else "FAIL"))

    # (2) head-width mismatch must RAISE
    try:
        st.assert_head_matches_vocabulary(torch.zeros(2, 3), "str_goal")
        width_ok = False
        print("[CONTROL width] FAIL -- a 3-wide head passed an 8-token check")
    except st.StrategicVocabMismatch:
        width_ok = True
        print("[CONTROL width] PASS -- a 3-wide head against the 8-token "
              "str_goal vocabulary is REFUSED (reachable)")

    # (3) absent label encodes to -1 and is NOT confused with off-vocabulary
    none_ok = int(st.encode_targets([None], "str_goal")[0]) == -1
    print("[CONTROL absent] a missing label encodes to -1 (the BAND, not a "
          "defect): %s" % ("PASS" if none_ok else "FAIL"))

    # (4) gradient reachability at weight 0.0 -- p.grad must NOT be None
    head = st.StrategicTokenHead(D_CTX)
    ctx = torch.randn(16, D_CTX)
    tg = {"str_goal": torch.randint(0, kg, (16,)),
          "str_action": torch.randint(0, ka, (16,))}
    loss, tele = st.strategic_loss(head(ctx), tg, w=0.0)
    loss.backward()
    grads = {n: (p.grad is not None) for n, p in head.named_parameters()}
    grad_ok = all(grads.values())
    print("[CONTROL grad@w=0] every parameter has a non-None grad at weight "
          "0.0: %s  (%d/%d tensors)"
          % ("PASS" if grad_ok else "FAIL", sum(grads.values()), len(grads)))
    print("    -> `p.grad is None` therefore still means NOT WIRED, never "
          "SWITCHED OFF (42/138 tensors were once mis-diagnosed this way)")

    # (5) the empty band must give a FINITE loss, not NaN
    head.zero_grad()
    empty = {"str_goal": torch.full((16,), -1), "str_action": torch.full((16,), -1)}
    l0, t0 = st.strategic_loss(head(ctx), empty, w=1.0)
    band_ok = bool(torch.isfinite(l0)) and t0["str_goal_n_supervised"] == 0
    print("[CONTROL empty band] all-ignored batch -> loss %.6f finite, "
          "n_supervised=%d: %s"
          % (float(l0.detach()), t0["str_goal_n_supervised"],
             "PASS" if band_ok else "FAIL"))
    print("    -> a zero loss reads as 'nothing was in band' (88.57 %% of "
          "frames), NOT as 'the head is broken' -- the route-head mask trap")

    # ================= CONTROL: majority reads its KNOWN value ============= #
    banner("CONTROL -- the majority-class predictor MUST read macro recall 1/K")
    # reproduce the measured corpus skew: FOLLOW_ROUTE 68.54 %, TURN_L 14.98 %,
    # TURN_R 16.48 %  (n = 801, MEASURED on the banked v7 sample)
    n = 801
    y = torch.cat([torch.zeros(549, dtype=torch.long),
                   torch.ones(120, dtype=torch.long),
                   torch.full((132,), 2, dtype=torch.long)])
    mc = st.majority_control_recall(y, "str_goal")
    print("majority class %s  share %.4f  K=%d  K_supported=%d"
          % (mc["majority_class"], mc["majority_share"], mc["K"],
             mc["K_supported"]))
    print("macro recall over FULL vocab      = %.4f   (MUST be 1/8 = 0.1250)"
          % mc["macro_recall_full_vocab"])
    print("macro recall over SUPPORTED (3)   = %.4f   (MUST be 1/3 = 0.3333)"
          % mc["macro_recall_over_supported"])
    maj_ok = (abs(mc["macro_recall_full_vocab"] - 0.125) < 1e-12
              and abs(mc["macro_recall_over_supported"] - 1 / 3) < 1e-12
              and abs(mc["majority_share"] - 549 / 801) < 1e-12)
    # and prove a CONSTANT head actually scores it
    const_logits = torch.zeros(n, kg)
    const_logits[:, 0] = 10.0                      # always FOLLOW_ROUTE
    pc = st.per_class_recall(const_logits, y, "str_goal")
    print("a CONSTANT head scores: pooled acc %.4f (DO NOT QUOTE) but macro "
          "recall over supported %.4f"
          % (pc["pooled_accuracy_DO_NOT_QUOTE"],
             pc["macro_recall_over_supported"]))
    const_ok = abs(pc["macro_recall_over_supported"] - 1 / 3) < 1e-12
    print("[CONTROL constant head] macro recall == 1/K_supported: %s"
          % ("PASS" if const_ok else "FAIL"))
    print("    -> THE POINT: 0.6854 pooled would read as a working head. "
          "0.3333 macro is the no-information value.")

    # ================= R: the deliberate-regression arm =================== #
    banner("ARM P4-R -- DELIBERATE REGRESSION (shuffled labels)")
    print("A head trained on PERMUTED strategic labels MUST land at the "
          "no-information macro recall. Above chance ==> the evaluation leaks "
          "and every P4 number is VOID.")

    def train_probe(shuffle, seed, steps=400):
        """FIT on one split, SCORE on a HELD-OUT split.

        The first version of this probe fit and scored the SAME 801 rows and
        BOTH arms read macro recall 1.0000 -- a 2-layer MLP memorises 801
        shuffled labels. The gate correctly reported VOID. Scoring a split
        that was never fit is the fix, and it is the same discipline the
        estimator rules impose on lambda: fit ALL hyper-parameters on the fit
        split; the scored split is scored, never tuned on.
        """
        g = torch.Generator().manual_seed(seed)
        cls = torch.randint(0, 3, (n,), generator=g)
        ctx_ = torch.randn(n, D_CTX, generator=g) * 0.3
        ctx_[torch.arange(n), cls] += 3.0            # class-identifying feature
        y_ = cls.clone()
        if shuffle:
            y_ = y_[torch.randperm(n, generator=g)]
        perm = torch.randperm(n, generator=g)
        fit, test = perm[:n // 2], perm[n // 2:]
        h = st.StrategicTokenHead(D_CTX)
        opt = torch.optim.Adam(h.parameters(), lr=3e-3)
        w = torch.zeros(kg)
        w[:3] = 1.0                                  # only supported classes
        for _ in range(steps):
            opt.zero_grad()
            L, _ = st.strategic_loss(
                h(ctx_[fit]), {"str_goal": y_[fit],
                               "str_action": torch.full((fit.numel(),), -1)},
                weights={"str_goal": w}, w=1.0)
            L.backward()
            opt.step()
        return st.per_class_recall(h(ctx_[test])["str_goal"].detach(),
                                   y_[test], "str_goal")

    real = train_probe(False, 1)
    shuf = train_probe(True, 1)
    chance = 1.0 / 3.0
    print("REAL labels     (HELD-OUT): macro recall = %.4f  (n=%d, d=%d)"
          % (real["macro_recall_over_supported"], real["n_supervised"], kg))
    print("SHUFFLED labels (HELD-OUT): macro recall = %.4f  "
          "(no-information value = %.4f)"
          % (shuf["macro_recall_over_supported"], chance))
    leak = shuf["macro_recall_over_supported"] > chance + 0.10
    sep = real["macro_recall_over_supported"] > shuf["macro_recall_over_supported"] + 0.20
    print("GATE VALIDITY ==> %s"
          % ("VOID -- shuffled labels score above chance; the probe leaks"
             if leak else
             "VALID -- shuffled labels land at chance, and the real arm is "
             "separated from it" if sep else
             "VOID -- the real arm is NOT separated from shuffled; the probe "
             "cannot see a learnable signal"))
    reg_ok = (not leak) and sep

    # ================= summary ============================================ #
    banner("P4 SUMMARY")
    allc = off_ok and width_ok and none_ok and grad_ok and band_ok and maj_ok \
        and const_ok
    print("controls (off-vocab refusal / head-width refusal / absent-vs-off "
          "distinction / grad@w=0 / empty band / majority 1/K / constant head)"
          "  ==> %s" % ("ALL PASS" if allc else "FAIL"))
    print("deliberate-regression arm (shuffled labels)  ==> %s"
          % ("VALID" if reg_ok else "VOID"))
    print("")
    print("[!] PART B (does the head LEARN from vision?) is NOT answered here."
          " It needs a tiny-rig arm with a replicate. P4 is "
          "MECHANISM-VALIDATED and CORPUS-BOUNDED.")
    res = {"controls_all_pass": bool(allc), "off_vocab_refused": bool(off_ok),
           "width_refused": bool(width_ok), "absent_encodes_-1": bool(none_ok),
           "grad_not_none_at_w0": bool(grad_ok), "empty_band_finite": bool(band_ok),
           "majority_macro_recall_full_vocab": mc["macro_recall_full_vocab"],
           "majority_macro_recall_supported": mc["macro_recall_over_supported"],
           "constant_head_pooled_acc": pc["pooled_accuracy_DO_NOT_QUOTE"],
           "constant_head_macro_recall": pc["macro_recall_over_supported"],
           "regression_real_macro": real["macro_recall_over_supported"],
           "regression_shuffled_macro": shuf["macro_recall_over_supported"],
           "regression_chance": chance, "regression_valid": bool(reg_ok)}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("[banked] %s" % out_path)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
