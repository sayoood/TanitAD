# SR-TTC R1 — Master Mind decisions on the implementation's open points (frozen BEFORE any R1 number exists)

**Written 2026-10-18 ~14:30 Berlin, before S0 finished.** At writing no fit-set concordance, no R1 criterion value and no LC
interval had been computed by anyone. S0 bank units had been read for headers only (keys, dtypes, shapes).
This file's sha256 is written beside it and landed before `srttc_r1.py run` is launched.

**Scope.** The registered body (`PREREG_SELRANK_ARM.md`) and Amendments 1–2 are unchanged: the grid, the bars, the gates,
the estimator, the outcome rows and the A2.4 call. This file settles only what the registered text leaves unspecified,
so that `c*` (written once) does not depend on an implementer's default. Each item names the `srttc_r1.py` parameter
that carries it.

| # | open point | decision (frozen) | parameter | why |
|---|---|---|---|---|
| 1 | Epochs, optimiser, batch: named as hyper-parameters in the draft but absent from the registered grid | **Fixed, not tuned:** Adam (β 0.9, 0.999), no weight decay, no schedule, **10 epochs, 256 sets per batch** | `--epochs 10 --batch-sets 256` | The registration note fixes the grid at lr × anchor × K. Adding epochs would change a registered grid, which A2.5 forbids. The fitted object is one linear output row on frozen features. Each config's final-epoch relative train-loss change is REPORTED as a disclosure; no rule reads it. |
| 2 | R1 says "mean out-of-fold contender concordance"; A2.4 defines "pooled out-of-fold contender concordance" precisely (P2's frozen top-8, pairs with differing realised TTC, logit ties 0.5) | **`c*` = argmax over the 9 configs of the MEAN over seeds {0, 1} of the POOLED out-of-fold concordance** | `--criterion pooled_oof --seeds 0,1` | Read the two registered texts as one statistic: "mean" is the mean over the two training seeds (C3), and "out-of-fold contender concordance" is the A2.4 statistic, the same one as the navtest 0.722 reading. The per-fold mean is REPORTED for every config, with `criteria_agree`. It never switches `c*`. |
| 3 | Seeds inside the selection | seeds 0 and 1, averaged (item 2) | `--seeds 0,1` | C3 already requires both seeds downstream. |
| 4 | "(safe, violating)" pairs for non-binary TTC labels | a pair is `(i, j)` with `y_i > y_j` | built in | Identical for binary labels, and it is the A2.4 pair rule. The count of non-binary top-8 entries is reported. |
| 5 | Loss normalisation | pair loss = mean over the minibatch's contrast pairs of softplus(−(z_i − z_j)); anchor = mean BCE-with-logits over all 64 proposals × sets; total = pair + anchor_w × BCE | built in | The registered loss names both terms and the anchor weight grid; a mean keeps the anchor's weight independent of the batch's pair count. |
| 6 | A2.4 interval wholly inside (0, δ = 0.010) satisfies RISING and FLAT | **FLAT** | `--both-rule flat` | A2.4's own rationale for δ: a doubling that buys less than δ is not a lever the confirm could see. Both flags are always reported. |
| 7 | Ties for `c*` | first in grid order (lr ascending, then anchor ascending); tied configs listed | built in | Deterministic; the smallest update wins a tie. |
| 8 | Replacement logs (A2.1 item 7) | only via an explicit `--replacements-file`; never automatic; > 5 % stops | `--replacements-file` | As A2.1 registered. |
| 9 | P2's own row on fit-set out-of-fold pairs | reported and labelled **IN-SAMPLE for P2**; it is not the 0.72 reference of §7 | built in | P2 trained on these labels. |
| 10 | f_max | **100 %** (PI, 2026-10-18), f₀ = 10 %. The Master Mind applies A2.4 item 4 to `lc_result.json`; the code does not decide scale-up | — | The PI's decision, recorded in `PI_DECISION_QUEUE.md` 2026-10-18 ~10:25. |
| 11 | Does a CPU head fit need a launch-gate PASS? | **YES.** R1 is a training run, and the binding rule (PI, 2026-09-26) has no size exemption. `srttc_r1_gate.py` mints a token bound to {landed commit, argv sha256, S0 bank-manifest sha256}, and `launch_r1.ps1` refuses without a matching one. Checks: G-SUITE, G-DVB (argv equals rows 1–10, the grid equals the registered 9), G-LIVE (one real-bank optimiser step: finite loss, non-zero TTC-row gradient, EXACTLY 0.0 elsewhere; no concordance computed), G-BANK (S0_DONE, every unit's sha256, `logs_without_any_token` ≤ 5 %), G-PATH (dev-gate/navtest refusal). Each check has a deliberate-regression arm that must make the gate FAIL. | `srttc_r1_gate.py` | CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS". |

**What this file does NOT do.** It does not read, and was not informed by, any fit-set or dev-gate number. It changes no
registered bar, gate, grid value, outcome row or estimator. If a later reading suggests one of these choices was
unlucky, the choice stands for this arm, and any change is a new pre-registration.
