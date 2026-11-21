# ⛔⛔ WITHDRAWN 2026-09-05 — DO NOT RUN. THE ARM ALREADY COMPLETED.

**This SPEC would spend ~8.6 h of Thor re-running an arm that finished 2026-08-30.**

VERIFIED AT SOURCE (`thor:/home/nvidia/v7tiny/emao14_30k_tauramp/config.json`, read
2026-09-05 by the Master Mind): `ema_decay_ramp` = **`"cosine"`**, `o5_target` = `"ema"`,
`ema_decay` 0.996, `steps` 30000, `summary.done` **True**. The run directory holds `ckpt.pt`
(144,111,717 B), `metrics.json`, `stage_gate.json`, `train_log.jsonl`. Its `config.json`,
`summary.json` and `stage_gate.json` are now **banked** under
`.../Implementation/incoming/2026-08-30-t1-first-v7-read/raw/tauramp_*`.

⭐ **WHY THE NEGATIVE LOOKED SOLID — and the lesson worth more than the arm:** the probe searched
for the **FLAG** (`--ema-decay-ramp`) in configs and source, found it only in the instrument, and
concluded "never launched". It never searched the **ARM NAME** (`emao14_30k_tauramp`), which returns
the registry row, three program reports, the retraction log, the prereg's Stage-3 outcome and a
banked 71 KB T1 artifact. ⛔ **SEARCHING FOR THE FLAG IS NOT SEARCHING FOR THE RUN.** A flag lives
in source and config; a run lives under its NAME in registries, reports and artifacts — and
`--save-every` overwrites the run dir to a single file, so the config corpus structurally cannot
answer the question.

See `Project Steering/Decisions/2026-09-05-mm-decisions.md` §M33 and retraction #32.

---

# SPEC — the tau-ramp arm (`D-EMA-ADOPT`'s unmet condition)

**Status: PRE-REGISTERED, NOT RUN. Both outcomes committed below BEFORE any launch.**
ArchInf FlyWheel, 2026-09-05. Registered against `D-EMA-ADOPT` (PI decision 2026-08-29).

⛔ **Scope discipline, stated first.** This arm satisfies a **PI-committed precondition on how
v7f SHIPS**. It is **not** the v7f launch gate — that is `V7_LAUNCH_GATE.md`'s P1/P2 under the
binding PI directive of 2026-08-31. Running this arm does **not** license a v7f launch, and
neither of its outcomes reopens the EMA adoption.

⚠️ **Naming.** The condition is a **tau (τ)** ramp — the EMA teacher's decay. Some briefs say
"lambda-ramp"; lambda is the LDAD weight, a different object. The flag is `--ema-decay-ramp`.

---

## 1. The claim under test

`D-EMA-ADOPT`: v7f ships with `--o5-target ema` for its prediction gain (`E-DEC-69`:
cos_ctr **0.6043 → 0.7524, +24.5 %**; nrmse **0.8288 → 0.7466, −9.9 %**; drift +3.6 %),
**conditional on one tau-ramp arm first** — because **fixed tau = 0.996 is known-suboptimal
against every published recipe** (BYOL and successors ramp tau from a lower base toward 1).

**H1:** a cosine tau ramp beats fixed tau = 0.996 on prediction, with drift not worse.

## 2. Design — ONE variable

| | |
|---|---|
| **Arm** | `emao14_30k` + `--ema-decay-ramp cosine` |
| **Baseline** | **`emao14_30k` as banked** (fixed tau = 0.996). ⛔ Do **not** retrain the baseline; it is the arm `D-T1-V7-READ` measured. |
| **Steps** | **30,000.** ⛔ A 2 k tau read is **inadmissible** — the MM-E1 stage-1 lesson: both 2 k effects were tau-warmup transients. |
| **The one variable** | the tau schedule. Everything else byte-identical; verify by a key-by-key `config.json['argv']` diff, not by intent. |
| **Cost** | **ESTIMATED ~8.6 h on Thor** (basis: `V7_LAUNCH_GATE`'s queue prices its other 30 k v7-tiny arms at 8.6 h; the k=60 arm at 22.6 h). |
| **Instrument** | already built: `--ema-decay-ramp {off,cosine}` (`train_v6_staged.py:9300`), tau-at-step (`:1819`), auditable per-step trace (`:1913`), **24 tests** in `stack/tests/test_ema_tau_ramp.py`. |

**Guards already in the code** (verified at source, not assumed):
* `--ema-decay-ramp cosine` **requires** `--ema-decay-start <= ` final (`:1901`).
* A ramp reaching **1.0 for the whole run is `--o5-target frozen`'s cell** and is **refused**
  (`:1896`) — the ramp cannot silently become a different experiment.
* The ramp is **inert** and warns when `--o5-target` is not `ema` (`:5380`).

## 3. ⛔ COMMITTED OUTCOMES — both written before the run

| outcome | reading | action |
|---|---|---|
| **RAMP WINS** | ramp **>=** fixed on prediction (cos_ctr / nrmse), **and** drift **not worse** | **ship the ramp** as v7f's EMA schedule |
| **RAMP DOES NOT WIN** | anything else — including *ramp better on prediction but worse on drift* | **ship fixed tau = 0.996 as measured** |
| **either** | — | ⛔ **neither outcome reopens the EMA adoption.** That is a settled PI decision. |

⚠️ **The drift clause is not a tiebreak, it is a veto.** `MM-E5` registers the hypothesis that
drift and prediction may be **positively coupled**; if the ramp buys prediction by paying drift,
that is exactly the trade `MM-E5` says we cannot yet price, and the conservative ship is fixed tau.

## 4. ⛔ ADMISSIBILITY — what makes the result quotable

1. **Tier stamp.** Prediction metrics here are **T0-DIAGNOSTIC**, not a capability claim. A T1
   read requires the `taniteval` path and is a separate deliverable.
2. ⭐⭐ **A REPLICATE ARM IS REQUIRED, per `H-ESTIM-SEED-1`.** A separated paired
   episode-cluster CI on a **single-seed** arm is **necessary but NOT sufficient** — the
   estimator resamples episodes with the models held fixed and is **structurally blind to
   training variance**. On this exact tiny rig, `A0b_replicate` (identical flags, identical
   seed, zero levers moved) produced *"separated"* differences on **3 of 18** metrics — a
   **~17 % false-positive rate**. ⇒ **the tau-ramp's effect must be read against the rig's own
   run-to-run noise floor**, or the verdict is not admissible. **Budget 2 arms (~17.2 h), not 1.**
3. **Estimator.** Paired episode-cluster bootstrap (`taniteval/ci.py`). ⛔ Never
   `overlapping_holdout_se` — it biases the point estimate, not merely the interval.
4. **Controls.** The per-step tau trace must be read back from the run and shown to actually
   ramp; a ramp that silently stayed flat is the failure mode this instrument's tests exist for.

## 5. Blockers, named

* ⭐⭐ **Compute is probably AVAILABLE — the "Thor is committed" premise is STALE.**
  `V7_LAUNCH_GATE`'s *"~50 h of committed work on ONE machine"* dates from 2026-08-31;
  `MODEL_REGISTRY.md:2081` records **`refav1-b1-v72-ep3-speed` COMPLETE at step 21,109, eval
  landed 2026-09-04**. Thor also holds the 178 GB B1 epcache and the `emao14_30k` baseline.
  ⚠️ **Correctly scoped: Thor was NOT probed** (out of bounds for the authoring agent), so its
  *current* occupancy is **UNVERIFIED** — what is MEASURED is that the job which committed it
  finished. **Confirm occupancy before launching.**
* ⛔ **The authoring agent could not run it**: brief forbids Thor and `tanitad-refcv3`; the
  dev-box 4060 was measured at **100 % util / 5,078 of 8,188 MiB** (2026-09-05). This is a
  permission boundary, not a capacity finding.
* ⛔ **The baseline lives on Thor** — `emao14_30k`'s T1 raws at
  `/home/nvidia/t1dumps/emao14_30k/t1.json`. Comparing without them means retraining the
  baseline, which doubles the cost and breaks the "do not retrain the baseline" rule above.
* ⚠️ **Priority.** Per this package's §2.3 ranking, arms **A0** (o11 same-clip control) and
  **A1** (command channel) carry measured effect sizes and attack **P1**, the actual gate; the
  tau ramp has **no measured effect size of its own**. If Thor frees for one 8.6 h slot,
  **A0/A1 should take it** — and that is a PI call, not mine.
