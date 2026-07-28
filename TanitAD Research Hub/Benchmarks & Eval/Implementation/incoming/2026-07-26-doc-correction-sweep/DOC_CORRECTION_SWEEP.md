# Documentation half of the estimator correction — sweep record

**Date:** 2026-07-26 · **Scope:** the program's quotable surfaces (registry, paper, overview, ledger,
leaderboard, pointer sidecar) · **Compute:** dev-box CPU only. No pod, no GPU, no network. pod1/pod2/pod3
were not contacted. · **Nothing staged, committed or pushed** (`git add` deliberately withheld per brief).

**Inputs, all read in full before editing:**
`…/incoming/2026-07-25-jack-blast-radius/` (`JACK_BLAST_RADIUS.md`, `jack_recompute.json`,
`jack_hierarchy_recompute.json`, `recompute_jack_fullset.py`) ·
`…/incoming/2026-07-25-published-number-provenance/PUBLISHED_NUMBER_PROVENANCE.md` ·
`Project Steering/RETRACTION_LOG.md`.

**Evidence-class legend:** MEASURED (ours + artifact) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## 0. What changed, in one table

| # | File | What | Class |
|---|---|---|---|
| T1 | `Project Steering/MODEL_REGISTRY.md` §6 | rank table **re-emitted** on full-set means + episode-cluster bootstrap; legacy column kept and labelled; **13 inline drift pointers** added; narrowing band widened | MEASURED |
| T2 | `MODEL_REGISTRY.md` §1.4b | v1.6/v1 partition caveat — **mechanism corrected** (conclusion kept) | MEASURED |
| T3 | `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` H26/:628 + H18/:110 | retracted `ctx→tactical +0.044` swept; H18 corrected **upward** | MEASURED |
| T4 | `Paper/TANITAD_PAPER.md` | four flagged inconsistencies fixed (CTRV · param count · 4th tick · I-JEPA leak) | MEASURED |
| +  | `MODEL_REGISTRY.md` §0.3 + §1.2 + §1.3 | definitional statistic line de-retracted; the 4th decision tick given a registry row; floor comparison made like-for-like | MEASURED |
| +  | `Project Steering/PROGRAM_OVERVIEW.md` §5.1 | mirrored rank table brought into line with the re-emitted §6 | MEASURED |
| +  | `Paper/TANITAD_PAPER.md` §7.6 + abstract | **newly-landed** 07-25 C1 retraction (v4-fromscratch ≈0.48) swept — found by the extended linter | MEASURED |
| T5 | `tools/registry_pointers.jsonl`, `Benchmarks & Eval/LEADERBOARD.md` | 3 sidecar pointers migrated to inline; one documented `lint-ok` | — |

`Project Steering/GATE_PROTOCOL.md`, `stack/scripts/run_gate.py`, the `2026-07-25-h2-*` and
`2026-07-25-e1b-*` incoming trees and `Mission Plan.md` were **not touched** (sibling-owned / PI-owned).
`taniteval/` was read-only.

---

## T1 — `MODEL_REGISTRY.md` §6, the cross-arm rank table

### The defect

The whole table was the deprecated **split-mean** (`overlapping_holdout_se`'s `mean` = mean of 8
overlapping random 20 % episode-holdouts), including FDE and miss@2m. The header block said the
statistic's *interval* was defective; it did not say the *central value* was.

### The re-emission

**Primary column is now the full-set mean over all 881 windows with its episode-cluster bootstrap CI95**
(`taniteval/ci.py`, B = 2000, unit = val episode). Ordering claims come from the paired bootstrap. The
legacy figure is retained in a final column explicitly headed
`legacy_split_mean ± overlapping_holdout_se (DEPRECATED)` — **retained, not deleted**, so every published
number stays traceable.

**Values were not re-derived by hand.** They come from `jack_recompute.json` (27 arms recomputed from the
raw `windows_*.pt` dumps) and were cross-checked against the committed `taniteval/results/driving_*.json`,
which already carry `estimator: "episode_cluster_bootstrap"` — **14/14 arms agree to 4 dp on ADE, FDE and
miss.** Those JSONs are now wired in as **13 inline `src:` drift pointers**, so the table is
machine-checked by `tools/registry_lint.py` from here on (16 pointers total, all resolving).

### Before / after — every row

| Arm | ADE before (split-mean) | ADE after (full-set) | FDE before → after | miss before → after |
|---|---:|---:|---|---|
| flagship-30k (v1) | 0.4522 ± 0.0312 | **0.4271** [0.3675, 0.4871] | 0.9437 → 0.9075 | 0.0602 → 0.0454 |
| refc-xl-30k | 0.458 ± 0.057 | **0.4714** [0.3896, 0.5556] | 0.972 → 1.0061 | 0.146 → 0.1419 |
| refc-base-30k | 0.4523 ± 0.0497 | **0.4728** [0.3835, 0.5699] | 0.954 → 1.0031 | 0.135 → 0.1419 |
| refc-small-30k | 0.5007 ± 0.0671 | **0.5261** [0.4295, 0.6262] | 1.045 → 1.1115 | 0.171 → 0.1714 |
| refb-v2-30k | 0.5921 ± 0.0685 | **0.5913** [0.4766, 0.7131] | 1.2305 → 1.2434 | 0.2025 → 0.2066 |
| flagship-speed (19 k) | 0.6277 ± 0.0551 | **0.6152** [0.5422, 0.6951] | 1.3173 → 1.3168 | 0.1799 → 0.1669 |
| refb-v2-20k | 0.6462 ± 0.0548 | **0.6435** [0.5410, 0.7516] | 1.3050 → 1.3218 | 0.2132 → 0.2157 |
| refb-10k | 0.8255 ± 0.0992 | **0.8372** [0.6753, 1.0218] | 1.6714 → 1.6964 | 0.2641 → 0.2679 |
| **CV floor** | 0.8248 | **0.8377** [0.6234, 1.0716] | 1.7081 → **1.7406** | — → 0.3042 |
| refb | 0.8682 ± 0.0817 | **0.8629** [0.6928, 1.0385] | 1.7341 → 1.7351 | 0.3343 → 0.3178 |
| planner_p2 | 0.893 ± 0.114 | 🟥 **not recomputable** | — | — |
| flagship-v3enc-10k | *"🟥 not evaluated"* | **1.9654** [1.6556, 2.2859] | — → 3.6084 | — → 0.6901 |
| refa-dinov2 | 2.1322 ± 0.1821 | **2.1675** [1.9081, 2.4212] | 3.2619 → 3.2803 | 0.6245 → 0.6129 |
| flagship-nospeed | 2.9176 ± 0.3558 | **3.0175** [2.5450, 3.5444] | 4.9395 → 5.0282 | 0.7395 → 0.7423 |
| refa-dynin-30k | 2.9196 ± 0.3937 | **3.0471** [2.4984, 3.6878] | 4.5832 → 4.7642 | 0.7246 → 0.7412 |
| flagship-v2-6k | 6.179 ± 1.2845 | **5.9396** [4.3273, 7.6249] | 12.7015 → 12.4011 | 0.8407 → 0.8524 |

Plus a footnote row for arms recomputed but not ranked (v1.6 0.4375, v4.1 0.8522, v4.2 0.9869, the REF-C
v1.2 family, the REF-A ladder).

### Ranking deltas — **10 of 27 positions move**

Reproduced from `jack_recompute.json` by sorting the 27 recomputed arms on the published split-mean and on
the corrected full-set mean:

| corrected rank | arm | legacy rank | move |
|---:|---|---:|---:|
| 2 | `flagship-v16-ab-ft` | 8 | **−6** |
| 4 | `refc-v12` | 6 | −2 |
| 5 | `refc-v12-identity` | 4 | +1 |
| 6 | `refc-xl-30k` | 5 | +1 |
| 7 | `refc-base-30k` | 2 | **+5** |
| 8 | `refc-xl-live` | 7 | +1 |
| 10 | `refb-v2-30k` | 11 | −1 |
| 11 | `refc-xl` | 10 | +1 |
| 15 | `flagship-v4.1-10k` | 16 | −1 |
| 16 | `refb` | 15 | +1 |

The other 17 arms hold position. *(Caveat, stated because it flatters the count: `refc-v12-identity` and
`refc-xl-30k` are the same checkpoint under two keys and are numerically identical, so one of those two
±1 moves is an artifact of tie-ordering. **9 of 27 are substantive.** `refa-dynin-30k` and
`overfit_refa-dynin-30k` are likewise duplicates but both hold.)*

**Three of the order changes land inside the registry's own table**, and each is stated above it so no
reader thinks rows were shuffled:

1. **REF-C-XL ↔ REF-C-base swap, with a sign flip.** Legacy base 0.4523 ahead of XL 0.4577; full-set XL
   0.4714 ahead of base 0.4728. Paired delta `+0.0054` → **−0.0013 [−0.0316, +0.0281]**. **Not
   separated either way — the 1= three-way tie stands.**
2. **`refb-10k` crosses the CV floor.** Legacy 0.8255 vs CV 0.8248 = ✗; full-set 0.8372 vs CV 0.8377 = ✅.
   Paired test **not separated** ⇒ the honest verdict is **TIE**. This is the exact cause of the standing
   §6-vs-`LEADERBOARD` contradiction: the arm sits *between* the two circulating CV floors.
3. **v3enc enters the table** at 1.9654, replacing a "🟥 not evaluated" cell that contradicted §1.4's own
   RESTART verdict. Ranks renumbered (old 3–12 → new 5–15).

### One finding that was not in either input report

**The three trivial bars were never split-means, and nobody noticed the mismatch.** MEASURED from the
emitter: `bench.py:485-511` (`kinematic_floor` → `best_of_3_ade_0_2s`) and `bench.py:558` (`ctrv_ade`)
both take a plain `.mean()` over all 881 windows. **So every legacy "clears the floor" verdict in this
program compared a *split-mean arm* against a *full-set bar*.** Re-checked like-for-like:

* v1 **survives** (0.4271 vs 0.5005 / 0.523 / 0.5735 / 0.8377) — as do REF-C-XL and REF-C-base.
* **REF-C-small no longer clears the CTRV oracle**: 0.5261 vs 0.523 (its legacy 0.5007 did). New.

The band `1.28–2.06× (10 arms)` was widened to **`1.107–3.100×, median 1.499×` (27 arms)** in §6 and in
`PROGRAM_OVERVIEW`, with the old band named as under-sampled rather than wrong.

---

## T2 — `MODEL_REGISTRY.md` §1.4b: the wrong mechanism

**Before (wrong):** *"`split_by_episode` **hashes the id values**, so the 8-split `heldout` numbers come
from **different random partitions across the two families**."*

**After (MEASURED, two independent probes):**

* **Code.** `stack/tanitad/eval/gates.py:139-152` takes `sorted(set(int(e)))` and passes it to
  `stack/tanitad/instruments/checks.py:49-58` (`i3_episode_split`), which calls
  `torch.randperm(len(episode_ids))`. It permutes **positions in the sorted list**; it never touches the
  values. An order-preserving relabelling therefore yields the **identical** partition.
* **Data.** Ran `split_by_episode` on both dumps' real `eid` arrays for seeds 0–7:
  `windows_flagship-30k.pt` (labels 0–39) and `windows_flagship-v16-ab-ft.pt` (real ids, e.g. 808464434)
  are **both order-preserving w.r.t. file order**, and the returned val index lists are
  **identical for all 8/8 seeds** (176/881 windows each). `gt` and `cv` are byte-identical (max diff 0.0).

**The conclusion is unchanged and now correctly attributed:** those two `heldout` means still must not be
compared — because **the estimator itself is biased**, −6.67 % to +11.69 % per arm, *within* a family as
well as across. v1.6 is the program's extreme case: split-mean 0.4886 vs full-set 0.4375 (**+11.69 %**)
against v1's +5.88 %, so the legacy Δ read `+0.0364` where the true Δ is `+0.0104` (**×3.5**).

**Why this correction was load-bearing:** as written, the old text implied same-family split-means *are*
safe to compare. They are not. The note is marked inline with its date and its root-cause class (*a
plausible mechanism inferred from a correct observation and never read off the code* — the `df`/quota and
`step_s` class in `CLAUDE.md`).

---

## T3 — `HYPOTHESIS_LEDGER.md`: the retracted seam swept

| site | before | after |
|---|---|---|
| `:128` (H26 row) | Status **Partially**; *"**1 of 3 seams load-bearing** (ctx→tactical **+0.044** CI-sep)"*; last-retested 2026-07-18 | Status **Untested (re-opened)**; *"**0 of 3**, not 1 of 3"*; the +0.044 struck through and retracted with **+0.0148 (×2.97)** and the three failed gates; last-retested 2026-07-25 |
| `:628` (07-18 timeline) | *"0/3 -> **1/3 — ctx->tactical FLIPPED to LOAD-BEARING**… delta +0.044 CI-sep"* | struck through, **"RETRACTED 2026-07-25 — the count stays 0/3"**, with the numbers and the one-sided-`separated` reading note |
| `:110` (H18 row) | *"grounding dominance **grew Δ 2.70 m**"* | **Δ +2.9568 m**, corrected **upward**, 8.65× widening needed to un-separate |
| `:642` (same timeline entry) | *"H18 grounding dominance GREW (delta 2.70m)"* | **→ +2.9568 m**, named as the leg that **survives and strengthens** |

**Gate arithmetic carried verbatim:** 0.0148 < MIN_ACC 0.02 · 0.0050 < MIN_COS 0.01 · 0.0437 < MIN_ADE_M
0.05 ⇒ **fails all three on the point estimate alone**, so no interval widening is needed to reject it.

**Framing carried, mirroring `TANITAD_PAPER.md:793-813` word-for-word in substance:** this **withdraws a
published claim and is NOT evidence against the hierarchy.** All three seams were measured under
PC1-violated conditions (`route_target = _NAV_TO_ROUTE[nav_cmd]` ⇒ `route_skill = 0.0` **by
construction**) and PC2-violated conditions (the scored 0.45 m path bypasses the hierarchy). A null from
an instrument that cannot see the effect is a **missing experiment, not a negative result** — so H26's
falsifier has *not* fired, and the row now says so. The reading note about the **one-sided**
`separated` predicate (a naive two-sided port flips the *harmful* intent seam to load-bearing) is
preserved at `:628`.

---

## T4 — `Paper/TANITAD_PAPER.md`, the four flagged inconsistencies

### (a) CTRV — 0.544 (abstract, §7.2) vs 0.523 (§7.3, §7.4 table, registry)

**Verdict: 0.523 is the one to quote; 0.544 is superseded.** Reasons, in order of weight:

1. `MODEL_REGISTRY.md:64` — the only quotable source for model facts — has carried **0.523** since
   2026-07-18, and §7.3, the OOD table (`PhysicalAI … floor 0.523`) and every downstream document use it.
2. `0.544` traces to `Project Steering/FLEET_REVIEW_2026-07-17.md:17,50` — a **2026-07-17 reading**, one
   day older, superseded by the 30 k round.
3. The oracle is a **full-set mean by construction** (`bench.py:558`), so the discrepancy is *not* the
   estimator defect.

**Fixed:** abstract `0.544 → 0.523` (with a pointer to §7.2); §7.2 `0.544 → 0.523`; the derived
`"stands 0.084 above it at 19 k"` → **0.092 on the full set** (0.6152 vs 0.523); the §7.2 forward
reference `"crosses — 0.452 ± 0.031"` → full-set 0.4271; changelog v0.3 entry annotated. A reconciliation
block names the supersession and, separately, protects the **0.545** two sentences away — that is the
*comma2k19* CTRV (`results_openloop_l2.json → comma_highway.L2_pointwise.ctrv.2s`), a **different
corpus**, which is precisely what makes it a replication rather than a restatement.

⚠️ **Not resolved:** *why* the canonical-val CTRV moved 0.544 → 0.523 between 07-17 and 07-18. TanitEval
did not enter git until 2026-07-20 (`a91bef8`), so no code timeline exists for either measurement. Stated
as unresolved inline rather than papered over. **Probe that settles it:** re-run `kinematic_floor` on the
canonical val and persist the CTRV per-window array (≈0 GPU; emitter `bench.py:485-511`/`:558`).

### (b) Model size — 261 M vs 262.8 M vs the measured 263.44 M

**Verdict: 263,442,838 (`total_model`) / 277,404,073 (`trainable`) is the measurement**
(`MODEL_REGISTRY §1.2`). **261 M is D-008's *design budget***, not a measurement. **262.8 M is a third
thing entirely** — see (c).

**Fixed:** abstract *"(261 M parameters)"* → **"(263.4 M parameters measured … the design budget every
arm is matched to is 261 M, D-008)"**; §3.1 *"The instantiated budget is 261 M"* → design budget vs
**instantiated 263,442,838**, with an explicit warning that the per-module split printed there is the
*allocation* and differs from the measured counts (encoder 87,121,280 measured vs 99.5 M allocated);
§6.3 training config → `263.4 M measured (261 M design budget)`. The remaining `~261 M` uses at §4/§5 are
**correct as written** — they describe budget-matching across arms, not a measurement, and were left.

### (c) The fourth decision-tick value, 14.331 ms — **TRACED, not unsourced**

Found at
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json`
(`exp: sc14-P2-real-tms-cnce-log-replay`, `latency.decision_tick_p50_ms = 14.331`, = encode 9.273 +
select_K9 5.058 — matches the paper digit-for-digit). Generator `real_telemetry_tms_cnce.py`.

**Conditions recovered, none of which had been published:** RTX 4060 · **fp32 eager** (line 109 is
`WorldModel(base250cam_config()).to(device).eval()` — no autocast, no CUDA graph) · comma2k19 val,
**n = 30 episodes** · log-replay · architecture **`base250cam`, `params_billions` 0.2628 = 262.8 M,
instantiated fresh**.

**Two corrections follow, and the second is the substantive one:**

* The paper said *"measured on the **deployed** 262.8 M architecture."* **Neither half is right.** The
  deployed flagship is **263.4 M**; the measured object is a fresh `base250cam` WorldModel. Latency and
  CNCE's parameter term are weight-independent, so it is a valid **architecture** read — it is *not* a
  read of the deployed checkpoint. That 262.8 M also happens to equal REF-B's count
  (`MODEL_REGISTRY:1481`) is a collision, not a source — exactly why an unlabelled param count is
  unquotable.
* **Against its own fp32 sibling the gap is architecture config, not regression:** 14.331 (base250cam) vs
  17.75 (step-6,500 model), same tick definition, same GPU, same corpus. The registry's other values are
  11.16 ms (fp16+CUDA-graph) and 17.75 ms (fp32); §7.4 line 572 carries 15.1/17.2 ms p50/p95.

**Fixed:** §7.10 conditions block added; the changelog entry re-attributed; **and `MODEL_REGISTRY §1.2`
now carries the row**, so the paper's number has a registry home. The paper states plainly that the
program retracted one latency figure for exactly this defect and then published a second one the same way.

### (d) ⭐ I-JEPA beats DINOv2 — leak disclosure added

The comparison (`fwd-ADE 3.194 vs 3.796 at 15 k`, §7.2) is repeated with an **overfitting** disclosure but
no **leak** disclosure, while `MODEL_REGISTRY §2.2` (line 844) records
*"Canonical val **80 % LEAKED** into its train set → guard excludes"* and carries it as open risk **R8**
(`:1556`). A registry-marked-**UNUSABLE** number was in the paper.

**Fixed:** an inline 🟥 disclosure at §7.2 stating the leak, quoting the registry, and **withdrawing the
claim in both directions** pending re-evaluation on the clean `f1b378` val; plus the changelog entry at
v0.3. Explicitly scoped: this does **not** touch §7.1's REF-A finding, which rests on the clean canonical
arms (`refa-dynin-30k` 3.0471, `refa-dinov2` 2.1675).

> ⛔ **CORRECTION 2026-07-28 (C48) — the remedy above named the disease.** The superseded wording is
> kept verbatim, with its date, because a sweep whose history is erased cannot be audited. **`f1b378`
> is not the clean val: 62 of its 80 episodes (77.5 %) are bit-identical to parity-train** by content
> (sha256 of raw `poses` **and** `frames_u8`), and it has been **hard-refused in code since 07-23**
> (`data.list_val_episodes(..., allow_leaky=False)` raises) — so this sweep prescribed a split the
> harness itself rejects. The clean target is **`physicalai-val-0c5f7dac3b11`** (0/40 and 0/600 by
> content). ⚠️ **This sweep inherited the phrase from `taniteval/registry.py`'s `note=` field rather
> than checking the artifact** — the same one-line source also propagated into `MODEL_REGISTRY §2.2`
> and `Paper/TANITAD_PAPER.md` §7.2, which is how ONE wrong note became FOUR documents. All four are
> now corrected; the registry source was fixed at `registry.py:85-97`. ⇒ **When a document names a
> specific artifact as the fix, verify the artifact, not the sentence.**

---

## Out-of-brief finding, surfaced by the extended linter — a retraction from *today* standing in the paper

While driving T5 to clean, `registry_lint` (now 51 loaded claims — the log grew twice mid-session) flagged
`RETRACTION_LOG` 07-25 class **C1**: *"flagship-v4-fromscratch val ade@2s ~0.48 and descending → v1's
0.427 with 2/3 of training left."* It was standing **in the paper's abstract** (`≈0.48 at 40 % of
training`), in §7.6 (`0.531 → 0.4825 → 0.4788`) and at §7.6's close (*"whether the co-evolved arm ends
above v1's 0.427 is open"*).

**Corrected in all three**, using the retraction's own numbers: the first decision-grade eval at step
15,000 (episode-cluster bootstrap B=2000, harness pinned by recomputing v1 to 0.4271 exactly) reads
**ADE@2s 0.5839 [0.4962, 0.6821]**, paired **Δ +0.1568 [+0.0630, +0.2504] — CI-separated *behind* v1**,
while CI-separated *ahead* of both trivial floors. The ≈0.48 was the trainer's **dense-20** statistic; the
v1-comparable **4-gate-waypoint** reduction on the same forward pass is 0.5839. **A metric NAME is not a
metric DEFINITION.** §7.6's architectural conclusion (coupling failure was a warm-start artifact) is
untouched — only the level. Flagged here because the v4 line is another agent's territory: **the edits are
in the paper only**, and the orchestrator should confirm them with whoever owns the arm.

---

## T5 — linter

`tools/registry_lint.py` now scans 6 documents. Actions taken:

* **3 sidecar pointers migrated to inline.** The rank-1= rows used `near: "full-set"`, which existed only
  because the full-set mean sat in a parenthetical beside a split-mean headline. After the re-emission
  that literal is gone, so the `near` filter would have found **no candidate number and reported DRIFT
  because the document got more correct**. They are now inline on the rows, joined by 10 more (13 total on
  the table). `tools/registry_pointers.jsonl` keeps the two remaining rows plus a comment recording why.
* **1 documented `lint-ok`** in `LEADERBOARD.md:417` — line 418 *quotes the C7-retracted claim in the act
  of reversing it*; the block **is** the retraction notice. (`reversed` is not in the linter's
  `RETRACTION_MARKERS`; `lint-ok` is the designed per-line escape and is cheaper than widening a
  marker list that governs six documents.)
* **1 documented `lint-ok`** in `TANITAD_PAPER.md:903` — *"roughly twice v1's 0.427"* is **v4.1's** ratio
  to the deployed full-set number, not the retracted v4-fromscratch claim (which is retracted 40 lines
  below).
* **A bug my own prose introduced and the linter caught:** writing the literal `<!-- src: … -->` inside an
  explanatory sentence made `POINTER_RE` parse the ellipsis as a pointer spec → `[ERROR] malformed
  pointer`. Reworded. Worth knowing: **the linter parses pointer syntax in prose, not just in table rows.**

### Final output

```
registry_lint: 6 file(s), 16 pointer(s), 51 retracted claim(s) loaded

[warn]  Project Steering/MODEL_REGISTRY.md:1388: header matches retracted-claim vocabulary,
        but every word is house boilerplate (likely a plain section title): "ref c closed loop"
        matches retracted claim: "flagship v1 beats REF-C closed-loop / drives where REF-C collides"
            rare tokens in the match: (none)
            lines [1388]: ### 4.4 REF-C CLOSED-LOOP - AlpaSim NuRec suite (n = 12) - **RECONSTRUCTION-OOD CONFOUNDED** - MEASURED 202

RESULT: PASS (0 error(s), 1 warning(s))
```

**0 errors, 1 warning — and that warning is the known true negative the brief instructed me to leave**
(§4.4's plain section title, which the linter self-labels as house boilerplate). `--strict` still FAILs on
it by design. `--self-test` PASSes all 5 red/green falsifiers; `pytest -q tools/tests` **127 passed**.

All **16 pointers resolve**, which means the 13 numbers I transcribed into the rank table were
machine-verified against their raw JSONs rather than eyeballed.

---

## What I could NOT resolve, and the probe that would settle each

1. **Why canonical-val CTRV moved 0.544 (07-17) → 0.523 (07-18).** No code timeline exists before
   `a91bef8` (2026-07-20). Flagged inline in §7.2 rather than hidden. *Probe:* re-run `kinematic_floor` on
   the canonical val and persist the CTRV per-window array — ≈0 GPU.
2. **`planner_p2`'s 0.893 ± 0.114 stays legacy** and is now printed as 🟥 **NOT RECOMPUTABLE** in the
   table. It carries **D-033, the v3 pivot**. *Probe:* migrate `planner_p2.py` to `ci.py` (add `full_set`
   + `paired_episode_cluster_bootstrap`) and re-run one arm, ~1 GPU-hour. **This is an integration
   escalation, not a doc note** — it is the last unmigrated module in the harness.
3. **Hierarchy-seam INTERVALS** are still un-recomputable (the panel persists no per-window arrays), so
   the ledger and paper carry corrected *point estimates* with no corrected CI. Rejection does not depend
   on them (all three floors fail on the point estimate alone). *Probe:* re-run `taniteval.hierarchy` on
   the three arms with the migrated code.
4. **`HYPOTHESIS_LEDGER.md:601`** — a 19 k-era entry says the **nospeed@22k** panel *"replicates (1/3
   seams, kappa 0.583)"*. That is a **different arm and a different artifact**, not covered by
   `jack_hierarchy_recompute.json`, so I could not correct it and did not guess. Given the ×2.97–×3.28
   biases measured on the other two panels it is **likely** also a 0/3. *Probe:* run
   `recompute_hierarchy_seams.py` against `hierarchy_flagship-nospeed*.json` if it exists, else re-run the
   panel.
5. **`plan_flagship-30k` (tactical head, 3.38 m)** has no windows dump, so the "head is a lossy readout"
   comparison has legacy statistics on *both* sides. The ratio is ~8×, far beyond the ≤11.7 % bias, so the
   reading survives — now stated with that caveat in both the registry and the overview rather than as a
   clean number. *Probe:* persist a windows dump on the next `plan` pass.
6. **`CLAUDE.md` still quotes `1.28–2.06×`** (the "Never quote an interval without its estimator" rule).
   I did **not** edit it — it is the project's working-agreements file and outside this brief.
   **Orchestrator action:** widen to `1.107–3.100×, median 1.499× (27 arms)`. One line.
7. **`GATE_PROTOCOL.md:96-97`** still publishes *"v1 reached v3enc's value at step **450** → ~**12×**;
   v2's was ~30×"*, a field `MODEL_REGISTRY.md:400-426` declares **void**. **Reported, not edited** — a
   sibling agent owns that file this session. It is a *binding standing protocol* quoting a void
   statistic; the same edit would clear provenance-report finding **F4/S3**.

---

## Deliverable manifest

| artifact | location | exists elsewhere? |
|---|---|---|
| `DOC_CORRECTION_SWEEP.md` (this file) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-doc-correction-sweep/` | **ONE PLACE ONLY** |
| re-emitted §6 rank table + §0.3/§1.2/§1.3/§1.4b corrections | `repo:Project Steering/MODEL_REGISTRY.md` (working tree, **unstaged**) | — |
| four paper fixes + the C1 sweep | `repo:Paper/TANITAD_PAPER.md` (working tree, **unstaged**) | — |
| H26/H18 corrections | `repo:TanitAD Research Hub/HYPOTHESIS_LEDGER.md` (working tree, **unstaged**) | — |
| mirrored rank table | `repo:Project Steering/PROGRAM_OVERVIEW.md` (working tree, **unstaged**) | — |
| `lint-ok` marker | `repo:Benchmarks & Eval/LEADERBOARD.md` (working tree, **unstaged**) | — |
| pointer migration | `repo:tools/registry_pointers.jsonl` (working tree, **unstaged**) | — |

**Nothing was staged, committed or pushed** — per the brief, and per `CLAUDE.md`'s git-hygiene rule, since
`git status --short` shows concurrent sibling edits to `GATE_PROTOCOL.md`, `run_gate.py`,
`gate_emitters.py` and the E1b tree in the same working tree. The orchestrator should check for foreign
entries before staging.

**Read-only guarantee:** `taniteval/` was never written to (`ci.py` and `gates.py` imported by path with
`sys.dont_write_bytecode = True`); no pod was contacted; no GPU was used; the excluded files
(`GATE_PROTOCOL.md`, `run_gate.py`, the `2026-07-25-h2-*` / `2026-07-25-e1b-*` trees,
`Mission Plan.md`) were not modified.
