# H2 · the sensor-need CLASSIFIER — PRE-REGISTRATION

**Written 2026-07-26 17:53 local / 15:53 UTC, BEFORE any classifier was trained** — before a single
feature was extracted, before any head weight existed, before any CONFIRM number was computed. The
file timestamp precedes every artifact in `artifacts/` and every line of `scripts/h2c_train.py`.
Nothing here may be edited after the first training result; corrections go into `H2_CLASSIFIER.md`
as **marked amendments**, exactly as the label work did.

**Author:** research engineer (H2 classifier stream). **Host: pod2** (A40, idle — verified
`0 MiB / 0 %` and a real `dd` write at **503 MB/s for 500 MiB**, 2026-07-26 15:43 UTC).
pod1 / pod3 / eval-pod are **not touched**.

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID, no chunk-level raw content and no frame
appears in any artifact in this folder. Clips are identified by an internal integer index only.

---

## 0. What is being tested

Sayed's design: the tactical brain decides **when** an additional camera is worth processing and
invokes it as a *tool*, instead of processing every camera all the time. **The claim is
EFFICIENCY.** Step 1 is **front camera only** (his explicit scoping); PhysicalAI-AV is the
proof-of-concept corpus.

The target is `L2_trigger`, GO-gated on 2026-07-26 (INHERITED,
`…/2026-07-26-h2-label-v2/H2_LABEL_V2_RESULTS.md`): held-out decision-relevance lift
**2.41× [1.3998, 3.7041]**, paired episode-cluster bootstrap B = 2000 over 1,415 clusters,
leave-one-chunk-out 16/16 still exclude 1.0, verdict *"GO — training can start."* **Nobody started
it.** This document starts it.

**What is NOT being tested here:** whether `L2_trigger` is decision-relevant (already answered), and
whether the cross-camera *residual* is separable (it is not, at that n — §5.3 of the label doc). The
scope trained here is the label's **PRIMARY (out-of-crop / front-periphery) scope**, and the
deliverable may not be headlined as "we learned when to switch on the side cameras."

---

## 1. The label being predicted — frozen, not re-tuned

```
target_L(t) = 1  iff  areq_off_L(t) >= tau*   AND  areq_seen(t) < tau*      # cross_left
target_R(t) = 1  iff  areq_off_R(t) >= tau*   AND  areq_seen(t) < tau*      # cross_right
target_any(t) = target_L or target_R
tau* = 0.5 m/s^2      scope = "crop" (outside the 51.4 deg encoder crop)    resolvable-only = True
```

⛔ **τ\* is NOT re-swept.** It is imported verbatim from `l2_label.py` / `l2_dev.json`, where it was
fixed by a power rule that never reads the lift. Re-sweeping it here would be the `L1` error one
level up. The same module `l2_label.py` is imported, not re-implemented, so the classifier provably
trains on the predicate that was gated.

**Per-camera independent Bernoulli, never a softmax over mixed axes** (`H2_SUBSTRATE §C.1`; the
5-way maneuver softmax defect). Two logits.

### 1.1 ⚠️ The C12 decomposition — pre-committed, MANDATORY whatever the outcome

`L2_trigger` is itself a **conjunction**, and C12 (`RETRACTION_LOG.md`, this workstream's own
retraction) is binding: *when a composite AND-label returns a null you have learned nothing about
which conjunct failed.* Therefore the two conjuncts are trained and scored **as separate heads,
unconditionally**, not only if the primary nulls:

| conjunct | predicate | prior expectation, stated in advance |
|---|---|---|
| **T_off** | `areq_off >= tau*` — an agent the encoder canNOT see requires braking | **HARD.** This is the genuinely extrapolative half. |
| **T_seen** | `areq_seen < tau*` — nothing the encoder CAN see requires braking | **EASY.** It is a statement about the visible scene. **HYPOTHESIS:** if the composite works only because `T_seen` is easy, the head has learned "the road ahead is clear", not "there is something off to the side" — and that is a different capability from the one H2 claims. |

**This 2×2 is the diagnosis licence.** A composite result reported without it is inadmissible.

---

## 2. The substrate — what exists, and its honest limits

| | |
|---|---|
| Label table | 26 PhysicalAI-AV chunks with `obstacle.offline` + calibration + `egomotion` — the L2 builder output, rebuilt by `l2_build.py` in ~5.5 CPU-min (MEASURED, label stream) |
| Front-camera features | the **frozen flagship-v1 encoder** applied to the cached 9-channel 256 px episodes on pod2 |
| Encoder | `flagship4b-speedjerk-30k` @ step 29999, `pod2:/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt` (INHERITED, `MODEL_REGISTRY.md §1.2` — **the deployed v1**, NOT `flagship4b-phase0-30k`, which is the no-speed ablation control) |
| Feature | `WorldModel.encode(frames)` → the **2048-d spatial-grid readout state** the world model itself consumes (`readout.SpatialGridReadout`, 8×8 grid × 32). Encoder + readout **frozen**; no gradient reaches them. This is a **head, not a new backbone.** |
| Episode ↔ clip join | **MEASURED, proven, not assumed:** `episode_id = int.from_bytes(clip_id[:4])`; reconstructing `discover_r0_clips → sorted → split_clips(val_frac=0.2, seed=0)` reproduces **600/600 val** and **2376/2376 train** episode ids, and the 24 skips land exactly on 1798…1941. (`artifacts/join_proof.json`) |
| **Universe** | **582 clips** = the 26 L2-labelled chunks ∩ the 3000-clip pod2 parity episode cache. **88,139 frames.** |

### 2.1 Three limits stated in advance, because they bound what the answer can mean

1. ⚠️ **The universe is 582 clips, not the label's 2,320.** Only clips that are *both* L2-labelled
   *and* already decoded into the pod2 episode cache can be used; the rest would need a ~52 GB gated
   camera re-download plus a full re-decode. **This is a power ceiling, and it is stated before the
   result, not after it.**
2. ⚠️ **The frozen encoder saw 459 of these 582 clips during its own training** (MEASURED: CONFIRM
   283 seen / 76 unseen; DEV 176 / 47). The head's held-out split is clip-disjoint and
   chunk-disjoint, but the *features* on most held-out clips come from an encoder that has seen
   those pixels. **A sensitivity restricted to the 76 encoder-unseen CONFIRM clips is mandatory and
   will be reported whatever it says.**
3. ⚠️ **`obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels.** `prov: "autolabel"`.
   Systematic misses of small/distant agents attenuate anything measured here.

### 2.2 Time alignment — declared as a risk, with its own acceptance test

The L2 grid is `arange(t0, t1, 100 ms)` on the **egomotion/obstacle** clock; the episode grid is
`linspace(t_first, t_last, n)` on the **camera** clock, minus the 2 frames the D-015 stack consumes.
They are not the same index. Alignment is therefore recovered **per clip** by maximising the
normalised cross-correlation of the two ego-speed series (both corpora carry `v` at 10 Hz) over an
integer lag.

**Pre-registered acceptance test:** a clip enters the study only if its best-lag Pearson correlation
is **≥ 0.99**. The distribution of achieved correlations and the number of clips dropped are
published whatever they are. If **> 10 %** of clips fail, the alignment method is declared unfit and
the run is reported as **BLOCKED** rather than rescued.

---

## 3. The split — episode-disjoint, and it inherits the label's own discipline

| | chunks | clips | frames | trigger⁺ frames | trigger⁺ clips |
|---|---|---|---|---|---|
| **TRAIN** = the label's **DEV** chunks | 10 | **223** | 33,884 | **169** (0.499 %) | 25 |
| **HELD-OUT** = the label's **CONFIRM** chunks | 16 | **359** | 54,255 | **308** (0.568 %) | 36 |

(MEASURED before training, `artifacts/universe.json`.)

**Why this split and not a fresh one:** chunk-disjoint ⇒ clip-disjoint ⇒ episode-disjoint, and it
reuses the label work's own boundary, so the held-out side is the side on which τ\* had never been
looked at. Any re-cut would put threshold-selection chunks into my test set.

**Model selection and the operating point are chosen by 5-fold GROUPED cross-validation *inside*
TRAIN, grouped by chunk** (folds = the sorted DEV chunk list dealt round-robin). The held-out
CONFIRM side is read **once**, at the end, at the already-fixed operating point.

⚠️ **Power, stated in advance.** 36 trigger-positive held-out clusters is **below the label work's
own ≥ 40 bar**, and 169 training positives is a small positive set for any learned head. **A "not
separated" here is therefore UNPOWERED, not refuted**, and must be reported with that word — the
program has measured CI half-widths shrinking ×2.8–3.9 going 40 → 600 episodes, and one verdict
flipping tie → separated on power alone.

---

## 4. The arms — all four honest baselines, pre-committed

| arm | inputs | what it settles |
|---|---|---|
| **`head_img_ego`** ⭐ PRIMARY | 8-step window of frozen 2048-d encoder states **+** ego (v, trailing 0.5 s accel) | the deployable configuration — v1 receives its own speed by design (`action_dim = 3`) |
| `head_img` | encoder states only | is there signal in the **image** at all |
| `head_ego` | ego only, same head capacity | how much of the primary is just ego state, **learned** |
| **`heur_ego`** ⭐ the baseline that decides the verdict | ego only, **not learned** — best (v, a_pre) threshold pair on a DEV grid | Outcome B's discriminator |
| `random@rate` | none | fires at the primary's **matched** rate, seeded, 200 draws |
| `always` / `never` | none | the two trivial endpoints of the efficiency axis |

`heur_ego`'s grid is fitted on TRAIN out-of-fold only, and it is fitted to **maximise the same
objective the head optimises** (out-of-fold average precision) so it is the *strongest* non-learned
opponent, not a straw man.

---

## 5. The operating point — a BUDGET rule that never reads a test metric

> This clause exists for the same reason τ\*'s power rule does. An operating point chosen at the
> argmax of a held-out F1 is a held-out sweep wearing a hat.

```
theta*(B) = the threshold at which the OUT-OF-FOLD firing rate on TRAIN equals the
            compute budget B (fraction of frames on which a second camera is run)
PRIMARY BUDGET  B* = 5.0 % of frames        [committed here, before any score exists]
CURVE           B in {0.5, 1, 2, 5, 10, 20} %   — reported in full, descriptive
```

`theta*` is fixed on TRAIN and applied to CONFIRM **unchanged**; the *realised* CONFIRM firing rate
is then a measured outcome (and a real test of calibration transfer), never a target.

**Why 5 %:** it is ~9× the label's own 0.555 % base rate, so it gives an imprecise classifier room to
recall real events while still leaving ~95 % of frames single-camera. It is a **budget**, chosen from
the efficiency goal, and it is computed from a rate — it cannot see precision, recall or AP.

---

## 6. What is measured, and with what estimator

**Every interval: paired episode-cluster bootstrap, `taniteval/taniteval/ci.py` (`episode_index`,
`_draws`), B = 2000, seed 0, resampling CLIPS with replacement, both arms recomputed inside the same
draw. `overlapping_holdout_se` is used NOWHERE.** The unit of resampling is the **clip (episode
cluster)**, never the frame — this label is frame-level and frames inside a 20 s clip are strongly
dependent, so it matters more here than anywhere.

**Per corpus, never pooled.** One corpus is in play (PhysicalAI-AV); no number is averaged with any
other corpus.

1. **Discrimination:** average precision (AP = area under precision–recall, `sklearn`-free trapezoid
   over the realised PR points) **against the base rate**, never accuracy. Also ROC-AUC, secondary.
2. **At `theta*(B*)`:** precision, recall, firing rate, precision-lift over base rate — each with a
   cluster-bootstrap CI, plus the **paired** Δ against `heur_ego` and `random@rate`.
3. **The efficiency trade-off curve:** for every budget in the grid, (firing rate, recall of
   `L2_trigger`, recall of the behavioural slice `L2_label`, measured compute saving). A curve, not
   a point.
4. **Compute, MEASURED not assumed:** per-frame wall-clock and FLOP-proxy of (a) one encoder pass,
   (b) the head itself. The head's own cost is **subtracted** — a gate that costs what it saves is
   worthless, and the arithmetic must show that.
5. **The C12 2×2:** `T_off` and `T_seen` scored separately, same protocol.
6. **Sensitivity:** the 76 encoder-unseen CONFIRM clips.

---

## 7. ⭐ OUTCOMES, COMMITTED IN ADVANCE

| | condition (held-out CONFIRM, at `theta*(B* = 5 %)` fixed on TRAIN) | verdict |
|---|---|---|
| **A — the efficiency claim gets its first evidence** | `head_img_ego` beats **both** `random@rate` **and** `heur_ego` on **recall at matched firing rate**, with the paired episode-cluster bootstrap CI on **both** deltas excluding 0 — **and** AP separated from `heur_ego` | the front-camera classifier identifies escalation-worthy situations. Report the measured saving with its interval and the missed-event cost. |
| **B — a real and useful negative** | it does not separate from `heur_ego` (whether or not it beats random) | **Say so plainly.** If it beats random but not `heur_ego`, **the signal is in the EGO STATE, not the image** — which redirects the MoE design: the gate should be a 2-input kinematic rule, not a vision head, and H2's premise needs re-scoping. |
| **UNDERPOWERED** | neither delta's CI excludes 0 **and** the CI half-width exceeds the point estimate | report as **UNPOWERED**, name the n, and do **not** call it B. |

⛔ **Binding:** no re-sweep of τ, no second operating-point rule, no post-hoc arm, no "marginal but
directionally encouraging". **A marginal result is Outcome B.** If the primary nulls, the C12 2×2 is
reported as the diagnosis and no third label or third head is invented inside this task.

---

## 8. Discipline binding this run

- Evidence class on **every** number, with the artifact path.
- **pod2 only.** pod1 (v2corpus training), pod3 (YouTube harvest, round 7) and the eval pod (P1
  envelope agent) are not touched. One job per pod.
- `PYTHONPATH=/workspace/TanitAD/stack`. Kill by **explicit PID** — never `pkill -f` / `pgrep -f`.
- Never judge pod disk with `df`; a real `dd` write test only.
- **Parity is untouched** — nothing here re-selects training episodes, and the flagship encoder is
  read-only.
- A trainer log is **not** a result. Only held-out eval output is quotable (C1); and a monotonically
  improving training loss licenses nothing about held-out behaviour (C11).
- **STAGE, NEVER PUSH.** `git add` only.
