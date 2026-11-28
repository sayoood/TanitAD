# RESULT — the living paper brought current to 2026-09-06 (v1.3 → v1.4)

`Work package: TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-paper-update/`
`Owner: paper agent. Deliverable: Paper/TANITAD_PAPER.md. Zero GPU.`

## 1. What the paper was, and what it is

| | before | after |
|---|---|---|
| version | **v1.3 (2026-08-29)** | **v1.4 (2026-09-06)** |
| last substantive section | §15 (2026-08-29) | **§17** |
| bytes / chars | 488,690 / 481,407 | **552,551 / 544,379** |
| results from 2026-09-04 → 09-06 | **none** | §16.1–§16.7 |
| false-positive doctrine | scattered across §5.6–§5.8 as I11–I16 | **§17, as method, + I17–I20** |

⛔ **Five substantial results had landed in 48 h and none were in the paper.** They are now, each
with what it does **not** establish (§16.8, §17.6).

## 2. The two-key audit — every number, re-read from its artifact

`raw/verify.py` re-derives **124 quotable values** from the banked artifacts and asserts each one
both (a) equals the artifact and (b) appears verbatim in the paper's new sections.
**Result: 122 PASS · 0 NOT-IN-PAPER · 2 descriptive rows** (the DiffusionDrive-V2 loss line and the
two-branch `std_dev_t_add` assertion, which are source predicates rather than numbers and are
confirmed by their own in-script assertions). Full table: `raw/VERIFICATION.txt`.

Artifacts read (⛔ **no number came from a summary, a changelog, a decisions file, or another
agent's prose**):

| § | artifact |
|---|---|
| 16.1 | `…/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json`, `raw/paired_v4b_vs_v3_FULL.json` |
| 16.2 | `…/2026-09-06-refcv4b-navpred/raw/ECHO_INDEX_CORRECTION.json`, `raw/paired_navpred.json`, `raw/navpred_treated_subsets.json` |
| 16.3 | `…/2026-09-05-refav1-cost-geometry/raw/wkappa_dose.txt`, `raw/feas_audit.txt`, `raw/seed_floor.txt`; `…/2026-09-05-refav1-close-the-gaps/raw/frontier.txt` |
| 16.4 | `…/2026-09-06-refav1-perception-probe/raw/probe_results.json`, `raw/motion_probe.json`, `raw/bank_meta.json` |
| 16.5 | `…/2026-09-06-p1-grad-reach/raw/p1_fullcensus.json`, `raw/p1_the_25.json`; source `stack/tanitad/models/predictor.py` |
| 16.6 | `…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/diffusiondrivev2_model_rl.py`; `TanitAD Research Lab/Library/library.json` key `2512.07745` |
| 17.1 | `…/2026-09-05-withheld-bank-panel/raw/panel_report.json`, `raw/VERDICT.md`, `raw/ONEVAR_AUDIT.txt` |

## 3. ⭐ Three figures I was handed did NOT reproduce from their artifacts — the artifact won

1. ⛔ **The replicate false-positive rate is `6 / 42 = 14.3 %`, not `3 / 18 ≈ 17 %`.** Recursing the
   whole `A0b_replicate.paired_vs_A0` tree gives **42 paired cells, 6 separated**. The `3 / 18`
   form does not reproduce at any scoping I could construct; restricted to the seven family rows
   the panel's own `VERDICT.md` tabulates at the 2 s horizon it is **3 / 14 = 21 %**. §17.1 quotes
   6/42 with 3/14 named as the restricted view.
   ⚠️ Separately, `raw/NOISE_FLOOR.md` in that package is a **crashed** artifact — the generating
   script died on a `UnicodeEncodeError` writing `⇒` to a cp1252 stdout — so it carries no numbers
   at all. It cannot be the source of either figure.
2. ⛔ **The anti-echo shuffle rate `0.2264` is superseded.** `refcv3_arm.py` compared a 3-wide
   `ROUTE_CLASSES` index directly against a 4-wide `NAV_COMMANDS` index. Corrected: **0.3370**
   (n = 1,736); the quotable form is the mutually-exclusive subset, **n = 1,311, label 0.7124 vs
   shuffled-nav 0.1739**. §16.2 quotes the corrected forms and records the defect and its
   root-cause class.
3. ⚠️ **`os_navpred`'s CI is `[0.2721, 0.3314]`** in `paired_navpred.json`, not the
   `[0.2744, 0.3318]` circulating in prose. §16.2 quotes the JSON.

## 4. ⛔ The review finding I raised against my own draft, and fixed

**My first draft repeatedly called the v7.2 nav command an "oracle" and framed
`os_navzero − ha0_ext = +0.1054 m` as "the measured size of the deployment gap". That is contrary
to a binding PI ruling** recorded in `Project Steering/VOCABULARY.md` (2026-09-04): the nav command
is **ground truth and a first-class ROUTE input**, `os` (nav fed) is **the deployment-relevant
arm**, `os_navzero` is a **ROBUSTNESS ABLATION**, and the term "oracle" is explicitly forbidden for
it. The wording was corrected in §16.1, §16.2, §16.8, the abstract and the changelog, and the one
nuance the ruling requires — our signal is **noiseless and perfectly timed** where a real router is
coarser — is now stated wherever the ablation is quoted.
⭐ **Root-cause class: a framing inherited from a briefing rather than from the register.** Same
family as quoting a number from prose; the object was a *word*, not a value.

Also fixed in the same pass: an explicit **evidence-class + tier stamp** on every §16 subsection
(§16.2 now records that its artifact stamps `tier_ruling: UNRULED`; §16.4 records that **no**
T-tier applies to a decodability probe); the **two FAILED pre-registered controls** of the
`os_navpred` roll (`C3` by 3 % of its tolerance, `C6` by 9 windows) with their cross-hardware
mechanism, which the draft had omitted; a **provenance stamp** marking §17.4's inherited rows as
`RETRACTION_LOG`-sourced rather than freshly measured; the **closed-loop** phrasing corrected
against `VOCABULARY.md` (this programme *does* hold AlpaSim/NuRec closed-loop numbers — for other
arms); and a rounding error, `true − shuffled = +0.3325 → **+0.3326**` at full precision.

## 4b. ⭐ One MORE landed result was found while auditing, and it is the sharpest of the set

While tracing §16.2's artifacts I found **`os_navflip` already banked and complete** in the same
package (`…/2026-09-06-refcv4b-navpred/raw/flip_analysis.json`, MEASURED, T1, 4,823 windows / 141
episodes, model-free control PASSING). It was **not on the brief's list of five** and it settles
the presence-vs-content question outright, so it was verified and added rather than deferred
(RULE ZERO: a refutation — or here, a partial answer — is a waypoint, not a deliverable).

**Flip left↔right on EVERY commanded window — the token wrong everywhere — and it costs
`+0.0022 m [−0.0006, +0.0052], NOT separated`. Remove the token and it costs `+0.0961 m`.
⇒ 97.7 % of the nav margin is the token's PRESENCE; 2.3 % is its CONTENT.** On the 1,743
flip-treated windows the flip is likewise not separated (+0.0062 [−0.0024, +0.0142]) while
beating the no-token arm by −0.0586 [−0.0791, −0.0410], separated.
⇒ **On this arm the E13 nav edge behaves as a presence-gated bias vector, not as a route-content
channel** — an architecture finding about the conditioning design, larger than the ADE that
measured it, and directly relevant to refcv5's goal-input design (§16.7).

⚠️ **Operational note for the next paper round:** the final edit of this turn had to wait out a
**G: outage window — 400 consecutive failed content reads of `Paper/TANITAD_PAPER.md` over ~2
minutes while its metadata resolved perfectly** (`ls` reported the exact byte count throughout, and
`git` in the same directory reported *"not a git repository"*). The already-staged blob was
unaffected. **The working procedure is a long RETRY LOOP on the SAME file, never a same-breath
control on a different one**, and every write in this package is idempotent and marker-guarded so
a repeat run cannot double-apply.

## 5. ⛔ Deliberately left OUT

| left out | why |
|---|---|
| any refcv5 **result** | none exists — training, ETA ≈ 2026-09-08 06:20 UTC. §16.7 carries the committed bars only. |
| `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` as capability numbers | INVALID on a v₀-conditioned vocabulary; quoted only to show the type error (a "ceiling" +0.9179 m *worse* than the arm it bounds), and stamped T0. |
| any **parity** claim for refcv4b / refcv5 | B1 corpus; the artifact itself says the eval split is not the canonical parity key. |
| the `best` zero-violation figures as a *superlative* | the register's own `D-REFAV1-CG-ZEROVIOL-SCOPE` withdrew "first/only"; four arms reach zero, one **vacuously**. §16.3 states the replication, the vacuity test, the seed-dependence of the sibling arm, and that the zero is bought by **not turning**. |
| the `wk1` / `wk3` / `wk7` sweep rungs | ⚠️ **Not found in any banked raw artifact** — searched by value with a same-breath control that read non-zero. §16.3 reports only the four rungs `wk15/wk151/ccos_argmax/cos_wk` that `wkappa_dose.txt` carries. **WORK ITEM: bank the A2 rungs.** |
| the `best` / `best_seed1` feasibility rows as a raw quote | ⚠️ `feas_audit_all.txt` covers ten arms and **`best` is not among them**; the numbers exist only in the decisions file and the register. §16.3 quotes the register's form and the raw control rows (`g`, `ha0_ext`, `ol`, `ccos_argmax`) from `feas_audit.txt`. **WORK ITEM: bank the `best`/`best_seed1` audit output.** |
| the closing rate's lag-1 autocorrelation `+0.7836` | present only in the package `RESULT.md`; carried in §16.4 stamped **INHERITED**, not MEASURED. |
| DiffusionDrive-V2's headline PDMS scores as a comparison | PUBLISHED, on a corpus and a simulator we do not have; §16.6 says so explicitly. |

## 6. ⚠️ ESCALATION — the register needs a row this agent may not write

`GOALS_AND_CLAIMS.md` is the live register and the constitution requires it to move in the same
turn as any asserted claim. §17 asserts **four new instrument rules, I17–I20**, and §16.3/§16.4
tighten the scope of two existing refav1 claims. **This agent owns the paper file and this package
only** and did not edit the register. ⇒ **Master Mind action required:** register I17–I20 (or rule
that method rows live only in the paper), and attach §16.3's turn-suppression caveat to the
`D-REFAV1-CG-*` family. Escalated here and in the agent's report, not left as a "please merge"
line in a document nobody re-reads.

## 7. Deliverable manifest

| artifact | where it lives |
|---|---|
| **`Paper/TANITAD_PAPER.md`** — v1.4, §16 + §17, abstract, references, changelog | **repo working tree, staged** |
| `…/2026-09-06-paper-update/RESULT.md` (this file) | **repo working tree, staged** |
| `…/2026-09-06-paper-update/raw/verify.py` — the 124-row two-key harness | **repo working tree, staged** |
| `…/2026-09-06-paper-update/raw/VERIFICATION.txt` — its full output | **repo working tree, staged** |
| `raw/land_navflip.py` — the outage-tolerant, idempotent lander for §4b | scratchpad (the edit it applied is in the paper; the script is kept for the procedure, not the result) |
| pre-edit backup of the paper | scratchpad only (`paper_backup_pre_v14.md`); the repo's git history is the durable copy |
