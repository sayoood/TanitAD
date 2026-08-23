# D-TAC1 — ADVERSARIAL VERIFICATION RECORD (independent re-derivation)

**Verifier:** adversarial reviewer, 2026-08-03. **Subject:** stream `refc-head-fix` (D-TAC1).
**Method:** every MEASURED claim re-derived with an OWN probe against the primary artifact.
The reducer in `adv1_independent_reducer.py` deliberately does **not** import `refc_tactical`
— the lat×lon algebra is re-implemented from the collapse map stated in the report, so a bug
in the stream's module cannot propagate into the check.

**Primary artifacts re-read:** `tanitad-thor:/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt`
(1364×5 logits, 1364×704 pooled, v0, lat/lon/man5/man_banked labels, eid) and the staged repo files.

**Verdict: PARTIAL.** The structural work survives, most of it bit-for-bit. The part that
decides the next GPU-day does not.

---

## Re-derived and CONFIRMED

| claim | my evidence | script |
|---|---|---|
| 5-way confusion matrix, all cells | element-for-element identical; rowsums `[818,174,109,146,117]`, colsums `[1078,165,114,0,7]`, both sum to n=1364 | adv1 |
| acc 0.7581 / macro-recall 0.5313 / lat-readout 0.9348 & 0.8290 / lon τ=0 0.3621 | identical to 4 dp | adv1 |
| τ frontier, all 8 rows | identical to 4 dp (τ=0.75 macroR 0.4708 vs 0.4709, rounding) | adv1 |
| AUC 0.7082 / 0.7294 / 0.7362 | identical, own rank-AUC with tie correction | adv1 |
| 132 / 1364 = 9.68 % label-destroyed | identical | adv1 |
| `man5 == man_banked` = 1.0000 | 0 mismatches / 1364 | adv1 |
| exact lat×lon round-trip | 5.55e-17 (float64, real logits); module round-trip 1.9e-6 (float32); matches the stated algebra to 6e-8 | adv1, adv3 |
| negative controls discriminate | own shuffle, different seed: macroR 0.3221, auc_active 0.4841, acc5 0.4890 | adv1 |
| **self-consistency control** | `collapse(lat,lon)==man5` 0 mismatches/1364; `collapse(window_factored_labels)==v1` **1.0000** over 3000 fuzzed windows (vs v2 only 0.7667 = exactly v1-vs-v2 agreement) | adv2, adv5 |
| capacity **+897** params | 104,191,577 → 104,192,474, +0.000861 %; `param_breakdown` total == `sum(p.numel())` both arms | adv3 |
| flag off ⇒ byte-identical | same 487 state_dict keys, all tensors equal, forward `traj`/`maneuver_logits` maxdiff **0.0** | adv3 |
| `man_prior_tau` is decode-only | Δtraj = Δanchor_logits = Δmaneuver_logits = Δlon_logits = **0.0** at identical weights + one fixed input | adv5 |
| `lon_to_anchor` zero-init + gradient | 0.0 at init; grad 0.511 via `anchor_logits`; `maneuver_to_anchor is None` when factored | adv3, adv4 |
| trainer weights 0.05+0.05 = 0.10 | exact, from source | adv5 |
| **no pod drift** | Thor `refc.py` byte-identical to repo HEAD; Thor `refc_tactical.py`/`refc_tactical_probe.py` md5-identical to the staged blobs | — |
| all 9 artifacts staged | blob hash == worktree hash **and** staged blob *content* verified (`git show :<path>`), not exit codes | — |
| pytest green | **1816 passed, 12 skipped, 2 xfailed** (217 s); 24 tests in `test_refc_tactical.py` | — |

Nuances found while confirming, worth carrying:

- `auc_lon_active` **equals** `auc_steady` exactly (0.7294 real, 0.4933 shuffled) — an algebraic
  identity. It measures "is something longitudinal happening", never direction.
- Through `traj` **no** gradient reaches any anchor-prior graft (argmax selection is detached).
  The old `maneuver_to_anchor` behaves identically, so this is not a regression — but the H19 /
  selection seam is trained **only** by the anchor CE, never by the trajectory L1.
- `update_tactical_prior` at the default momentum 0.99 is still 31 % off the true prior after
  200 steps (0.131 vs 0.100). It needs ≳500 steps before the buffer is usable.

---

## REFUTED / NOT ESTABLISHED

**R1 — "the longitudinal mass lands ENTIRELY in lane_keep" is false as stated.**
Of 263 true-longitudinal windows: 241 (**91.6 %**) → `lane_keep`, **19 (7.2 %) → a LATERAL class**,
3 correct. The word is in `DTAC1_RESULTS.md:25` and baked into `refc_tactical.py`'s docstring.

**R2 — the accelerate frontier is inflated ~3.2× by windows the same report calls irrecoverable.**
MEASURED: at τ=0, **all 11 correct `accelerate` predictions fall on the 132 label-destroyed windows**
(`tau0_correct_accel_total = 11`, `tau0_correct_accel_on_DESTROYED = 11`). On the 1232 windows the
5-way label CAN represent: accelerate recall **0.0000 at τ=0** and **0.0479 at τ=0.5**, against the
published 0.0455 / 0.1529. §1.3 ("no decode rule can ever recover them") and §2.3 ("peaks at 0.153")
are in direct contradiction.

**R3 — PRECISION IS NEVER COMPUTED.** Zero occurrences of "precision" in `DTAC1_RESULTS.md`, the
probe JSON, the pre-registration, and `refc_tactical_probe.py`. MEASURED by me at the recommended
τ=0.5: brake **precision 0.1935** (398 fired, 153 true), accelerate 0.3776, "longitudinally active"
0.4536 while firing on 36.4 % of windows against a 29.0 % true rate. A decision rule whose entire
mechanism is shifting the boundary toward the rare class is evaluated on recall alone.

**R4 — E-A2's negative control was never run, and the instrument does not clear it.**
`pooled_only` on **shuffled labels** → macro-recall **0.3550**, against the published real 0.3833.
Gap +0.028. The stated 0.3333 floor is not this instrument's empirical null.
(`pooled+v0` 0.4346 vs its own null 0.3328 does clear.)

**R5 — every AUC in the E-A2 table is a fold-pooling artifact.** One global AUC is taken over
probabilities concatenated from two differently-calibrated fold models. `v0_only` brake:
**0.3626 pooled vs 0.6094 averaged per fold**. `pooled_only` steady: 0.4839 vs 0.5166. The
qualitative "every class improves with v0" survives the correction
(0.5395→0.5729, 0.5166→0.5815, 0.7185→0.7655); the six published numbers do not.
(The +0.051 delta is seed-stable: 0.0513 across 5 seeds.)

**R6 — the "prior-corrected decode" uses the VAL LABEL MARGINAL as its prior.** I reproduced the
whole 8-point frontier to 4 dp with `prior = [0.1122, 0.7104, 0.1774]` — literally the report's own
`label_marginal`. The open risk flags τ-on-val only. And the frontier is not robust: at fixed τ=0.5
a ±25 % brake-prior perturbation swings brake recall 0.5948 ↔ 0.4183 and **accelerate 0.0496 ↔
0.2769 (5.6×)**.

**R7 — "unpaired CIs overlap, no separation claimed" used the wrong estimator; paired, it separates
both ways.** Paired episode-cluster bootstrap, 39 episodes, B=4000:
lon-accuracy Δ **−0.1090 [−0.1861, −0.0379]**, lon macro-recall Δ **+0.0971 [+0.0404, +0.1575]**.

**R8 — "the turns are CALIBRATED (165/114)" and "lateral readout 0.8290/0.9348" are two different
decodes in one sentence.** Under `lat_readout` the counts are turn_left **153** vs 174 true (−12.1 %)
and turn_right **87** vs 109 (−20.2 %) — under-predicted, not calibrated.

**R9 — "aux-pressure confound removed by construction" is not established.** Nominal weight matched
(0.05+0.05=0.10, confirmed); loss SCALE not: at uniform, 5-way CE = log 5 = 1.6094 vs factored
half-sum = log 3 = 1.0986 — the factored arm carries **68.3 %** of the baseline's tactical loss
magnitude.

**R10 — "0.026 → 0.503" mixes denominators** (0.026 = 3/117 on the 5-way label; 0.503 = 77/153 on
the 3-way lon label). Matched: **0.0327 → 0.5033** (all windows) or **0.026 → 0.4103** (excluding the
132 destroyed). The magnitude claim survives; the quoted pair is not one frame.

**R11 — "thresholds fixed in advance" is SELF-REPORTED, not verifiable.** The prereg is uncommitted
(`A`) and its mtime **13:04:56** is AFTER the probe JSON (12:58:11) and the Thor substrate (12:57).
Appending the execution record destroyed the only mtime evidence. Evidence class INHERITED, not
MEASURED. (The prereg's internal structure — both branches, §6.3 thresholds, a genuinely refuted
registered prediction — is real; only the ordering is uncheckable.)

**R12 — INCOMPLETE BY RULE: the four metric families are absent.** TACTICAL only. No LONGITUDINAL
(target-speed, headway/time-gap/TTC), no LATERAL kinematics (heading, curvature, yaw-rate,
cross-track — `lat_readout` is classification), no STRATEGIC. The two CIs carried are unpaired.

**R13 — `maneuver_decision` still destroys the longitudinal class on turns.**
`collapse(turn_left, brake_stop) = turn_left`, `collapse(turn_right, accelerate) = turn_right`.
The drop-in field a downstream reader would naturally use is unchanged; consumers must switch to
`lon_decision`. Not named as an integration risk.

**R14 — line-number citations were born stale.** `refc_tactical.py` cites `refc.py L925 / L916-922 /
L966 / L97-99` (HEAD numbers). In the file it ships beside, `man_logits = self.maneuver_head(pooled)`
is at **L1115**. `DTAC1_RESULTS.md:46` cites `refc.py:100`, also HEAD.

**R15 — undeclared deviation from the prereg's own consequence.** §6.3's READOUT-limited branch
commits to "F1 is demoted to a nice-to-have"; the shipped `refc_factored_config()` sets
`tactical_speed_input = True` and the next action launches all three levers.

---

## What I retract from my own pass

My first τ-isolation test reported `tau_changes_traj = 0.0039` and looked like a leak. It was
**confounded** — I drew `v0` twice, once per model, and `v0` reaches the decoder condition. Re-run
with one fixed input, the delta is exactly 0.0. Logged here because the same class of error
(two draws where one was intended) is what makes an instrument report a difference that is not there.
