# T1 — PRE-REGISTRATION of the LINEAR PROBE on the frozen v1 visual state

**Written 2026-07-26 22:02 local (20:02 UTC), BEFORE any probe weight existed and before any
probe score was computed.** The file timestamp precedes every JSON in `artifacts/`.
Nothing here may be edited after the first held-out number; corrections go into
`SITUATION_SEMANTICS.md` as marked amendments.

**Author:** research engineer (situation-semantics stream).
**Host:** `tanitad-eval` (A40, idle). pod1 (training) and pod3 (control re-adjudication) untouched.
**pod2:** READ-ONLY file pull only — see §6.

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID and no raw content appears anywhere in
this folder; clips carry an integer index only.

---

## 0. The question this test answers, and why it goes first

`H2_CLASSIFIER.md §9 recommendation 5` (escalation #3) states the ladder:

> *"1. a linear (ridge/logistic) probe on the frozen 2048-d state — the lowest-variance reader;
> if this fails, no head rescues the representation."*

The PI's question is whether we have a model that extracts the **semantic** information needed to
decide when to switch on extra cameras. `NOT_T_seen` — *"an agent the encoder CAN see requires
braking ≥ τ\*"* — is the **best-powered visual-semantics target in the program** (1,642 held-out
positives across 101 clips, INHERITED `artifacts/c12_fix.json`) and the only place where anything
in H2 separated from chance. It is also strictly **easier** than any of the three situations the PI
named: it asks only *"is there an obstacle ahead"*, not *"am I in a roundabout"*.

**If the lowest-variance possible reader cannot extract even this from the frozen visual state,
then no head rescues that representation, and building labels for lane change / roundabout /
intersection on top of it is premature.** That is why T1 runs before T2.

---

## 1. Substrate — fixed here, and NOT rebuilt

The probe consumes the **exact frozen features H2's heads consumed**, so the comparison to
H2's `head_img_ego` / `head_img` / `head_ego` is like-for-like and not a re-derivation:

| | |
|---|---|
| features | `pod2:/workspace/h2clf/feats/clip_*.npz` — 520 admitted clips, 2048-d `float16` per episode frame, produced by the committed `h2c_features.py` from **`flagship4b-speedjerk-30k` @ 29999** (the deployed v1), FROZEN |
| bundle | `pod2:/workspace/h2clf/bundle` — the de-identified label bundle from `h2c_prep.py` |
| transport | streamed read-only over ssh, **md5-verified per file on arrival** (§6) |
| target | `NOT_T_seen(t) = 1 iff a_req_seen_res(t) >= tau*`, `tau* = 0.5 m/s^2` — **imported from `h2c_train.py` line-for-line, never re-implemented and never re-swept** |
| unit | the **frame** (1 column), exactly as `h2c_c12fix.py` evaluates it — NOT the (camera, frame) pair |
| split | **TRAIN = the label's DEV chunks, HELDOUT = the label's CONFIRM chunks**, unchanged from H2. Chunk-disjoint ⇒ clip-disjoint ⇒ episode-disjoint |

⚠️ **The three limits H2 recorded are inherited unchanged and bound this run too**: the universe is
582 of the label's 2,320 clips; the frozen encoder saw 283 of the 322 held-out clips during its own
training; `obstacle.offline` is `prov: "autolabel"`.

---

## 2. The probe — the lowest-variance reader, stated exactly

Two independent linear readers, both fit on TRAIN only:

```
RIDGE     : w = argmin ||Xw + b - y_pm||^2 + lam*||w||^2      (y_pm in {-1,+1}), CLOSED FORM
LOGISTIC  : w = argmin BCE(sigmoid(Xw + b), y) + lam*||w||^2   full-batch LBFGS, convex
```

Closed-form ridge exists so that **no optimiser can be blamed for a null**. Both are reported.

**Feature standardisation:** per-dimension mean/std computed on **TRAIN rows only**, never the
held-out side — identical to `h2c_train.py`.

**Representation ladder** (all pre-registered here; PCA is fit on TRAIN rows only):

| # | name | input dim | what it tests |
|---|---|---|---|
| **P1 ⭐ PRIMARY** | `img_t` | 2048 | the frozen state at the label frame — the escalation's literal ask |
| P2 | `img_win_mean` | 2048 | the 8-step (0.8 s) window mean — the head's temporal context, without extra capacity |
| P3 | `img_win_flat` | 16384 | the head's exact input, read linearly |
| P4 | `img_pca{16,64,256}` | 16 / 64 / 256 | **rung 2 of the ladder** — does a low-rank projection expose what the full 2048-d does not? Tests the swamping hypothesis directly |
| P5 | `ego_t` / `ego_win` | 2 / 16 | ⭐ **POSITIVE CONTROL** — `(v, a_pre)` at `t`, and the same 8-step window `head_ego` actually received. **Both** are run so the control matches the head's input exactly *and* in its simplest form |
| P6 | `ego_win+img_pca{16,64,256}` | 32 / 80 / 272 | ⭐ the **low-rank concatenation** H2 recommended instead of raw concatenation |
| P7 | `img_t_SHUFFLED` | 2048 | ⭐ **NEGATIVE CONTROL** — features permuted across clips, label untouched. Must NOT separate |
| P8 | `constant` | 0 | the chance arm; AP == base rate inside every bootstrap draw |

**Hyper-parameter selection:** `lam` over a fixed log grid, chosen by **5-fold CV grouped by CHUNK
inside TRAIN**, maximising CV-AP — the same rule, the same folds and the same statistic
`h2c_train.py` used. **The held-out side is never read during selection.** One `lam` per row of
the ladder, selected independently.

---

## 3. The estimator — named, and named once

**Paired episode-cluster bootstrap**, `B = 2000`, `seed = 0`, resampling **clips** with
replacement, both arms recomputed inside the same draw. Machinery imported from
`taniteval/taniteval/ci.py` (`episode_index`, `_draws`) via the committed
`2026-07-26-h2-classifier/scripts/h2c_stats.py`. **`overlapping_holdout_se` is used nowhere.**

**The above-chance test is the PAIRED ΔAP against a constant score**, not "does the AP interval
clear the full-sample base rate" — because the base rate is itself random under episode
resampling. This is H2's §0-point-2 correction and it is adopted verbatim.

**`separated` ⇔ the 95 % interval of ΔAP excludes 0.**

---

## 4. ⭐ The power ceiling — measured BEFORE the primary is read

H2's primary was UNDERPOWERED and it only knew that because it measured its ceiling first. The
same discipline applies here. **Two floor/ceiling quantities are computed and written to disk
before any P1–P4 score is looked at:**

1. **Instrument sensitivity (the ceiling).** The **ego** probe (P5). H2 MEASURED that an ego-only
   *head* clears chance on this target by **+0.0766 [+0.0506, +0.1353]**. If our linear probe on
   the same 2 channels does **not** reproduce a separated positive effect, the instrument is not
   sensitive enough at this n and **no P1–P4 null may be read as Outcome B**.
2. **Noise floor.** The **shuffled-feature** probe (P7) and the **constant** arm (P8). Their ΔAP
   vs chance gives the empirical detection floor; the upper 95 % bound of P7's ΔAP is the
   **minimum detectable effect (MDE)** this run can distinguish from nothing.

**Reported before the verdict, whatever they say.**

---

## 5. Outcomes — both committed in advance

Evaluated on `NOT_T_seen`, held-out CONFIRM clips, at the CV-selected `lam`, once.

| | condition | consequence |
|---|---|---|
| **A** | **P1 (or any of P2–P4)** has paired ΔAP-vs-chance with a 95 % interval **excluding 0 from above**, *and* the positive control P5 also separates | **The frozen v1 visual state DOES expose this semantics** and H2's null was a head-capacity / swamping failure. ⇒ vision must enter **low-rank, not by concatenation**; report which rank works and recommend rung 3 (fine-tuned trunk) only if the low-rank path saturates. |
| **B** | **no** image-based row separates, **and** the positive control P5 **does** separate | ⛔ **The frozen v1 visual representation does not expose this class of semantics at all.** Say it plainly. Label work on lane change / roundabout / intersection on top of this representation is premature; the representation is the thing to fix. |
| **UNPOWERED** | the positive control P5 does **not** separate | The instrument cannot detect an effect known to be present at this n. **Neither A nor B may be reported.** State UNPOWERED and stop. |

**Binding:** no re-sweep of `tau*`, no post-hoc arm added to the ladder above, no re-reading of the
held-out side after seeing it, no alternative target. Any follow-up is a new pre-registration.

---

## 6. ⚠️ Deviation from the brief, declared in advance

The brief offered *"regenerate on the eval pod with the committed script, ~9.6 GPU-min"*.
**That option does not exist:** `h2c_features.py` reads the decoded episode cache at
`/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894`, and **the eval pod has
no PhysicalAI episode cache at all** (MEASURED — `find` over `/workspace` and `/root` returns
nothing; the dev-box cache exists but is keyed `14231cd29c74` / `bb543bdf7836`, **not** the parity
key `e438721ae894`, so using it would break cross-arm comparability and is refused under the parity
rule).

Therefore the features were **pulled read-only from pod2** (`tar cf - | ssh`, no write, no compute,
no GPU, no python process started on pod2), md5-verified per file on arrival, and all computation
runs on the free eval pod. Before the pull, pod2's state was measured:

```
nvidia-smi : 0 % util, 0 MiB / 46068 MiB, "No running processes found"
ps -eo ... : no trainer, no python job; only sshd/jupyter/nginx/ops-daemon
```

🔴 **This contradicts the brief's premise that pod2 is running a blind-imagination sweep at 97 %
GPU. Escalated in the report, not buried here.**

## 7. Discipline binding this run

- Evidence class **and tier** on every number.
- Estimator named on every interval; resampling unit = **clip cluster**, never the frame.
- **Bi-directional harness validation** (`e1c_selftest` pattern): a **fidelity check** — the loader
  must reproduce H2's published substrate counts exactly (TRAIN 31,032 rows / 836 positives;
  HELDOUT 50,119 rows / 1,642 positives / 101 of 322 positive clips) — **and** a deliberately
  failing input (P7 shuffled, P8 constant).
- **Parity untouched.** Nothing here re-selects training episodes.
- pod1 and pod3 not touched at all; pod2 read-only.
- **`git add` only.** No commit, no push.
