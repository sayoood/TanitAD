# ESCALATION — two lines for `stack/scripts/refc_v3_train.py` (H19-STAMP-1)

⛔ **NOT APPLIED BY ME.** `stack/scripts/refc_v3_train.py` was `MM` in
`git status` for the whole of this turn — a sibling has it staged *and* dirty.
Per the brief's contention rule I escalate an exact diff rather than edit it.

⭐ **WHY IT IS STILL NEEDED after the model-side stamp landed.** The runtime
flag (`out["h19_tactical_feed"]`, `refc_v3.py::_hook`) travels with a *forward
pass*. `config.json` is the artifact a reader opens **without running anything**,
and it is the one that made this defect invisible: refcv4b's own `config.json`
(MEASURED, two dev-box copies at 6,416 B, `tac_vocab_version = "v7.0"`) carries
ten `param_breakdown` keys, a `registered_delta`, `effective_weights`,
`goal_provenance`, `provenance_roles` — and **not one H19 field**.

⚠️ **BOTH SITES, NOT ONE.** `refc_v3_train.main` runs `preflight` only under
`--preflight` and otherwise calls `train()` directly. A stamp in one covers one
launch path of two — the sibling D-ROLL-1 finding, same trainer, same night.

---

## Site 1 — `train()`, the `config.json` record (currently line 3687)

```diff
         "image_hw": list(cfg.core.encoder.image_hw()),
         "tac_vocab_version": cfg.tac_vocab_version,
+        # ⭐⭐ H19-STAMP-1 (2026-09-06). `tac_vocab_version` alone does NOT
+        # answer "did the tactical brain feed the anchor prior on this run" —
+        # a reader must know `refc_v3.py::_hook`'s `man5 = None` guard to
+        # derive it, and no artifact carried the derivation. MEASURED: under
+        # a non-kin3 vocabulary the z_tac action heads move the decoder's
+        # maneuver_logits, lat_prior AND anchor_logits by EXACTLY 0.0.
+        # ⚠️ "dropped" names the TACTICAL FEED, never the prior: `refc.py`
+        # falls back to the core's own 5-way, so H19 itself stays live.
+        **v3.h19_prior_stamp(model),
```

## Site 2 — `preflight()`, beside the params line (currently line 2992-2993)

```diff
     bd = v3.param_breakdown_v3(model)
     print(f"[v3-preflight] arm={args.arm} params={bd}")
+    # ⭐ H19-STAMP-1: the same fact on the OTHER launch path. A preflight that
+    # prints the capacity ledger but not which side of the `man5` guard this
+    # build takes lets a v7.0 arm launch looking identical to a kin3 one.
+    _h19 = v3.h19_prior_stamp(model)
+    print(f"[v3-preflight] h19_prior={_h19['h19_prior']} "
+          f"| source={_h19['h19_prior_source']} | graft={_h19['h19_graft']}")
```

---

## What the two sites will emit (MEASURED on the CPU smoke rung)

| build | `h19_prior` | `h19_prior_source` |
|---|---|---|
| `hier`, `kin3` | `applied` | `tactical_z_tac_via_man5` |
| `hier`, `v7.0` | `dropped(tac_vocab_version='v7.0': derive_man5_logprobs is a [B,3]x[B,3] POSITIONAL contract and cannot read a 8-wide head (D-REFCV4-DEFECTA1))` | `core_aux_kin3` |
| flat | `n/a(flat arm: no tactical cascade)` | `core_aux` |

⛔ **The two rows DIFFER — that is the acceptance criterion**, and it is the one
a sibling's `_unverified` manifest string failed the same night by reading
identically on every arm including the real ones.
`stack/tests/test_h19_prior_stamp.py::test_stamp_is_two_sided` asserts it, and
`raw/mutation_proof.txt` shows the assertion **failing** when the stamp is
hardcoded to `applied` (M2), so the check is proven able to fail.

## Risk

`h19_prior_stamp` reads only built objects and returns four JSON-safe scalars.
It cannot raise on a flat arm (that path was found and fixed by its own test —
`self.tac_vocab_version` is assigned inside the *hier* construction block and
does not exist on a flat build). No weight, vocabulary default or head
construction is touched.
