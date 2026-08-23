# REGISTRY ESTIMATOR SWEEP — landing the corrected G1/G4 values, and what the sweep found on the way

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start:** `7ff8037`
**Compute:** CPU / doc only. No GPU, no model loaded, no pod contacted, Thor untouched.

---

## 0. THE ANSWER — what this pass changed, in one screen

The corrected G1/G4 values existed (`JACK_IN_GATES.md`, 2026-08-16) but **the registry was still
publishing the banned ones**. They are now landed, with the superseded number kept visible beside every
one. Three things were found doing it that were not in the brief:

| # | finding | class |
|---|---|---|
| ⭐ **1** | **A THIRD verdict on the P2 path was never re-decided — and unlike G1/G4 its flip IS REACHABLE.** `planner_beats_cv` is banned on *both* sides; flipping it needs a **+6.59 %** estimator error against a **measured local upper edge of +5.877 %** and a programme-wide **+11.69 %**. **UNDECIDED**, not "no flip". | a verdict inventory taken from the *headline* rather than from the artifact |
| ⭐ **2** | **All THREE closed-loop baseline numbers were banned, not just the threshold.** The brief named 1.6852 → 1.7318. Also banned: **FDE 3.5296 → 3.6190** and **divergence 0.2216 → 0.2350**. All three moved the *same* way — **the v1 closed-loop failure was UNDERSTATED**, so every P2-vs-head margin *widens*. | correcting a bar but not its siblings |
| ⭐ **3** | **The same paired delta is published twice in the registry, 800 lines apart, with intervals differing 3.31×** — REF-A vs flagship: `[2.447, 2.798]` (banned) at §2 vs `[2.0945, 3.2570]` (decision-grade) at §6, the narrow one unlabelled. **3.31× is above the top of the programme-wide 1.107–3.100× band.** | a paired delta — the statistic with the ×−4.15 sign-flip history |

**Nothing flips.** G1 and G4 stand, and D-033 / the v3 pivot stand. But the correction is not cosmetic:
**one of the numbers it moves is the safety-shaped one** (divergence, +20.3 %), and **one of the bars it
moves is in the paper's pre-registered gate list**.

---

## 1. VERIFIED, NOT INHERITED — the banned estimator, from the artifact's own mouth

The brief said *"`planner_p2.py` decided `G1_pass` with the banned estimator"* and told me to verify.
Verified two independent ways, without re-reading the migrated source (another agent holds that file):

1. **The published artifact declares it.**
   `…/2026-07-26-closedloop-artifact-rerun/_pod_pulled/planner_p2_flagship-30k.json` →
   `protocol.ci = "8-split episode jackknife"` — the exact retracted label that `taniteval/ci.py:5-27`
   documents as **neither a jackknife nor a valid SE**. The estimator names itself in the record.
2. **Every deciding number reproduces.** `JACK_IN_GATES.md` §2 recomputes the banned estimator from
   banked windows and gets the published values **bit-exactly at 4 dp**, including the *model-free* CV
   baseline (0.8248 ± 0.1035) — which is what proves the banked windows are the same objects the
   2026-07-19 gate ran on, rather than a re-derivation that happens to agree.

⭐ **And the verdict inventory is larger than the brief's.** Enumerating every boolean in that artifact
(MEASURED, this pass) returns **five**, not two:

| verdict | estimator | re-decided by JACK_IN_GATES? |
|---|---|---|
| `open_loop.G1_pass` | banned | ✅ partial — no flip |
| `open_loop.G1_head_minus_planner_ade2s.separated` | banned (paired) | ✅ — no flip |
| `closed_loop.G4_pass` | banned, **both sides** | ✅ full — no flip |
| ⛔ **`open_loop.planner_beats_cv`** | **banned, both sides** | ❌ **NO — and its flip is reachable (§3)** |
| `weight_sensitivity.beats_head_all` | ✅ **not banned** — raw point values | n/a — stands (but see §5) |

---

## 2. WHAT LANDED IN THE REGISTRY

Located **by content**, not by line number — the brief's `:2482` / `:2716` / `:283` had already gone
stale again between the brief being written and this pass starting (the registry moved 3338 → 3369 lines
under an unrelated commit, `eca7106`).

| what | where it landed | superseded value kept visible? |
|---|---|---|
| §5 P2 results table — rebuilt with a decision-grade column **and** a `superseded (BANNED)` column | `MODEL_REGISTRY.md` §5 | ✅ every cell |
| ⭐ the **PAIRED G4** row — `−0.7375 [−0.9362, −0.5295]`, p(δ>0)=0.0000, n=221/20 | §5, its own table row | ✅ banned paired −0.6873 ± 0.2191 kept |
| §1.2 closed-loop line — **the 2026-08-03 "estimator NOT STATED" flag is answered and CLOSED** | §1.2 | ✅ full 3-row before/after table |
| leaderboard row 10 (P2) + the `plan_flagship-30k` head row | §6 | ✅ |
| "Two readings that matter" #2 — upgraded to decision-grade on **both** sides | §6 | ✅ |
| D-033 decision-log row | §8 | ✅ all six superseded figures listed inline |
| REF-A dyn-in table + D-A5 (the 3.31× interval, §0 finding 3) | §2, §8 | ✅ |
| the reversed **"Cite 0.628 (heldout)"** instruction | §1.3 | ✅ |
| **Alpamayo-2 augmentation row** (new §11.1a, taken from the coordinator) | §11 | n/a — new |

### 2.1 The G4 correction, as landed

| row | BANNED | **decision-grade** | point shift | CI too narrow by |
|---|---|---|---|---|
| planner `closed_bike_ade2s` | 1.0375 ± 0.2023 | **0.9799** [0.7456, 1.2312] | +5.88 % | 1.200× |
| planner `closed_bike_fde2s` | 2.1940 ± 0.4552 | **2.0583** [1.5463, 2.6134] | +6.59 % | 1.172× |
| planner `divergence_gt5m` | 0.0871 ± 0.0460 | **0.0724** [0.0225, 0.1409] | **+20.30 %** | 1.287× |
| **baseline** ADE (the threshold) | 1.6852 ± 0.0977 | **1.7318** [1.5707, 1.9070] | **−2.69 %** | **1.722×** |
| ⭐ **baseline FDE** *(not in the brief)* | 3.5296 ± 0.2548 | **3.6190** [3.2453, 4.0215] | **−2.47 %** | **1.523×** |
| ⭐ **baseline divergence** *(not in the brief)* | 0.2216 ± 0.0431 | **0.2350** [0.1680, 0.3027] | **−5.70 %** | **1.564×** |

⇒ Corrected comparisons: **42.9 % less drift** (paired, the quotable form) · **43.1 % closer** FDE ·
**3.2× fewer** divergences (was 2.5×). **Every margin widens**, because the planner rows shifted down
and the baseline rows shifted up.

⚠️ **Scope, stated per row rather than silently:** the planner is 221 win / 20 ep; the baseline ADE
threshold and the FDE/divergence baselines are 881 win / 40 ep. **Only the ADE has a paired form**
(head **1.7174** on the *same* 221 windows). The FDE and divergence comparisons are **unpaired and
scope-mismatched** and are labelled as such in the registry — I did **not** manufacture paired numbers
for them, because nobody has computed them.

### 2.2 The caveats, carried with the numbers rather than footnoted

Landed **inline in §5, §1.2, §8/D-033** — not as a footnote — exactly as briefed:

* point estimates move **−6.9 % to +6.8 %, bidirectional *within a single artifact*** (head −6.9 %,
  operative +5.9 %) — which is why "how big is the correction" has no single answer;
* intervals were **1.17×–2.17× too narrow**;
* the **divergence rate — the safety-shaped number — moves +20.3 %** (8.7 % → 7.2 %);
* the **G4 threshold itself was a legacy `heldout` mean, 2.69 % low** ⇒ **the old gate was HARDER than
  the honest one**;
* the banned statistic gave **7 of 40 val episodes weight exactly 0** — **C73**, a *wrong-population*
  defect, not a precision one.

---

## 3. ⭐ THE THIRD VERDICT — `planner_beats_cv` is UNDECIDED, and this one could actually flip

MEASURED this pass. Both sides of `planner_beats_cv` are banned: planner **0.8929**, CV **0.8248**.
The CV floor's decision-grade value is **0.8377** — *higher*, which moves the comparison **toward** the
planner.

For the verdict to flip to "beats CV", the planner's corrected mean must fall below **0.8377**:

```
required error = (0.8929 − 0.8377) / 0.8377 = +6.59 %
```

Set against the measured envelopes:

| envelope | value | verdict |
|---|---|---|
| this window set, this split structure (3 arms) | **−6.909 % … +5.877 %** | +6.59 % is **1.12× the upper edge** |
| programme-wide, 27 arms | **−6.67 % … +11.69 %** | +6.59 % sits **comfortably inside** |

⇒ **This is not "no flip". This is UNDECIDED**, and it is a materially different situation from G1,
which needs **−73.6 %** — an error ~11× larger than anything ever measured. The same **~400 s of GPU**
that closes G1's planner arm closes this one, because it is the *same* missing dump.

**Landed as:** registry §5 (a dedicated block), the §6 leaderboard row's `Beats CV` cell changed from
`✗` to ⚠️ **undecided**, `PROGRAM_OVERVIEW.md`, and `LEADERBOARD.md`.

---

## 4. ⭐ THE 3.31× PAIRED INTERVAL — the same delta, published twice, in one document

Found in the wider heldout-vs-`full_set` sweep (priority item 4), not in the brief.

| site | Δ (flagship − REF-A) | interval | width |
|---|---|---|---|
| `MODEL_REGISTRY.md` §2 (REF-A dyn-in) | +2.62 | **[2.447, 2.798]** ⛔ banned | 0.351 |
| `MODEL_REGISTRY.md` §6 (cross-arm) | +2.6200 | **[2.0945, 3.2570]** ✅ decision-grade | 1.1625 |

**Widening factor 3.31×** — *above the top of the programme-wide 1.107–3.100× band* — and it is a
**paired delta**, the statistic on which the 2026-07-25 blast radius measured errors up to **×−4.15
including a sign flip**. The narrow one carried **no estimator label**, so a reader comparing the two
sections would have had no way to tell which to trust.

✅ **The H4 verdict is untouched and in fact strengthens**: on `full_set` means the gap widens
**2.467 → 2.620 m**, and — a clean internal consistency check — the `full_set` difference
(3.0471 − 0.4271 = **2.6200**) reproduces the published paired point estimate **exactly**.

Also corrected in the same table: **both columns were `heldout` split-means**, making the right-hand
column a **cross-arm comparison of two split-means**, which §4.1b of the registry explicitly rules
invalid. Only the ADE@2s row had a `full_set` companion; the other five rows have **no decision-grade
value published anywhere** — flagged in place, not fixed (it needs a §6-style re-emission).

---

## 5. THE FULL SITE INVENTORY — unbounded, and the false-positive problem quantified

**Method.** `os.walk` from the repo root, **no depth limit** (C69: an absence claimed from
`find -maxdepth 4` on files at depth 6 cost 21 days). **4,939 files scanned, max depth reached 8.**

⚠️ **A bare numeric regex is unusable here, and this is the same trap in a fourth costume.** Matching the
banned *numbers* alone returned **229 files** — overwhelmingly floats that coincide (`0.0871` as an
arbitrary value in an unrelated results JSON, `1.8201` containing `0.8201`, a CI bound `[0.8929, …]` on
a completely different quantity). Requiring a **co-occurring context token** (`G1|G4|P2|planner|
closed-loop|tactical head|divergence|CEM|jack|holdout|REF-A|paired`) on the same line cuts it to
**76 lines in 23 files**. *This is the `pgrep -f` / polling-monitor self-match family again: the
discriminator has to be disjoint from the thing being searched.*

### 5.1 LIVE prose — publishes a banned number as a current claim

| site | number | status |
|---|---|---|
| `Project Steering/MODEL_REGISTRY.md` §1.2, §5, §6, §8 | the full P2/G1/G4 set | ✅ **FIXED, both values visible** |
| `Project Steering/PROGRAM_OVERVIEW.md:323` | *0.893 ± 0.114* + ⛔ the **REFUTED** *"not recomputable (no raw JSON, no windows dump)"* | ✅ **FIXED** |
| `Project Steering/PROGRAM_OVERVIEW.md:447` | *+2.257 ± 0.329*, "38 % less" | ✅ **FIXED** — replaced by the paired form |
| `Benchmarks & Eval/LEADERBOARD.md:152` | *0.4522 → 1.685, 22.2 %* | ✅ **FIXED** |
| `Benchmarks & Eval/LEADERBOARD.md:185` | *0.893 ± 0.114*, `✗ Beats CV` | ✅ **FIXED** |
| `Benchmarks & Eval/LEADERBOARD.md:707` | *"P2's ADE survives only under the deprecated estimator"* + "no window dump" | ✅ **FIXED** — partially refuted |
| ⭐ `Paper/TANITAD_PAPER.md:2847,2850` | **G4 bar 1.69 m · G5 bar 0.452 m · head 3.38 m** — in the **pre-registered gate list** | ✅ **FIXED** |

⭐ **The paper site is the one that matters most going forward.** It is not a *result* quoting a stale
number — it is a **gate definition**, so every future arm would have been held to a bar inherited from a
defective statistic. And because the old bar is **2.69 % lower**, that bar is **harder** than the
measurement justifies.

### 5.2 NOT fixed — named, with the reason (escalations, §7)

| site | number | why not fixed here |
|---|---|---|
| `Project Steering/360_REVIEW_2026-07-20.md:38,137,219,314,738` | the full set incl. `[2.447, 2.798]` | **dated review** — an archival snapshot of what was believed on 2026-07-20; rewriting it rewrites history |
| `TanitAD Research Hub/Architecture & Inference/V35_DESIGN.md:90,91` | *1.685 ± 0.098, 22.2 %*; the P2 framing | live design doc, another stream's file |
| `…/Architecture & Inference/ARCHITECTURE_WIRING_COMPARISON.md:212` | *3.38 (3.150 in the P2 pass)* | ditto |
| `…/Architecture & Inference/V4_FLAGSHIP_DESIGN.md:339` | *3.1501* | ditto |
| `…/Benchmarks & Eval/TANITEVAL_V2_METRIC_SUITE.md:53` | *0.4522 → 1.685 ± 0.098; 22.2 %* — **cites MODEL_REGISTRY §1.2, which I just corrected** | ditto — but it now contradicts its own source |
| `…/Production & Optimization/Research/2026-07-20-orin-thor-deployment-and-inference-levers.md:545` | *0.452 → 1.685, 22.2 %* | dated research doc |
| `…/Architecture & Inference/Research/2026-07-19-p2-planner-over-v1.md:33-81` | the whole P2 results block | **the original P2 report** — correctly archival |
| `…/Architecture & Inference/Research/2026-07-19-refa-deep-analysis.md:67` | `[2.447, 2.798]` | dated research doc |
| dated `incoming/` packages (`2026-07-19-alpasim-closedloop-v1`, `2026-07-25-*`, `2026-07-26-*`, `2026-08-03-*`, `2026-08-16-jack-in-gates`) | various | ✅ **correctly archival** — these are the *record* of what was published; several already print both estimators |

⛔ **One of those archival packages contains a three-week-old unactioned instruction.**
`…/2026-07-26-closedloop-artifact-rerun/CLOSEDLOOP_RERUN.md:389` lists divergence `22.2 %` as
**"HEADLINE, quoted in ~10 docs … → 23.5 % … needs edit"**. **It was written 2026-07-26 and not done.**
That is C70 exactly — a correction that lived in a document nobody re-read. This pass closes it for the
registry, the leaderboard, the overview and the paper; §5.2 is the remainder.

---

## 6. THE WIDER `heldout` vs `full_set` SWEEP (priority item 4)

The registry is mostly disciplined — §0.3 states the rule and most rows print both. The exceptions:

| site | finding | action |
|---|---|---|
| ⭐ §1.3 *"Cite 0.628 (heldout)"* | **a standing INSTRUCTION to prefer the banned statistic** over the `full_set` value printed next to it — backwards under §0.3's own rule | ✅ **FIXED** — now *"Cite 0.6152 [0.5422, 0.6951]"*; the note's real contribution (that 0.640 is derived arithmetic, never a measured mean) kept |
| ⭐ §2 REF-A dyn-in table | both columns `heldout`; right-hand column is a **cross-arm split-mean comparison** (§4.1b forbids); paired interval 3.31× too narrow | ✅ interval + labels **FIXED**; the five rows with no `full_set` anywhere **flagged, not fixed** |
| §2 REF-A ablation ladder (`Result (ADE@2s heldout)`) | a whole column in `heldout` | **flagged** — a §6-style re-emission, out of scope here |
| §4 `legacy heldout ±` row | already labelled DEPRECATED | ✅ correct as-is |
| §1.1/§1.5/§3 rows | print both forms | ✅ correct as-is |

*The sharpest instance of the class in the whole document is the §1.3 one — not a stale number, but a
**standing instruction to quote the banned one**.*

---

## 7. ⚠️ ESCALATIONS — sequencing for the orchestrator, not applied unilaterally

1. ⭐ **`planner_beats_cv` is UNDECIDED and its flip is reachable (+6.59 % vs a +5.877 % local /
   +11.69 % programme-wide envelope).** It is the same missing dump as G1's planner arm ⇒ **one ~400 s
   GPU job settles both**, after Thor's 336 M run. Until then it must not be quoted in either direction.
2. **The one open measurement, unchanged from JACK_IN_GATES §8.3:** re-run `collect_openloop` **dumping
   `plan_wp`/`head_wp` per-window**. `…/2026-08-16-jack-in-gates/code/recompute_g1_g4.py` needs no
   changes. ⚠️ **Seed the CEM first** (see 4) or the re-drive is not reproducible.
3. **`planner_p2.py`'s CEM is UNSEEDED** — stamped on the P2 rows as a property of the numbers, per the
   brief. Measured drift 0.019 %, but *measured is not bounded*: that is one observation, not a bound.
   The file is held by another agent; **the fix is theirs to make, and it should land before the
   re-drive in 2**, otherwise the re-drive inherits the same unbounded component.
4. **§5.2's eight un-fixed live/dated sites.** `TANITEVAL_V2_METRIC_SUITE.md:53` is the urgent one — it
   **cites MODEL_REGISTRY §1.2 and now contradicts it**.
5. **`RETRACTION_LOG.md` entry owed** (not written — another agent holds that file). Two classes:
   *(a)* **a correction that was written down and never applied** — `CLOSEDLOOP_RERUN.md:389` said
   *"22.2 % … needs edit"* on 2026-07-26 and it was still unedited on 2026-08-17 (**C70**, and the
   sibling of C73's own root cause); *(b)* **a verdict inventory taken from the headline instead of the
   artifact** — G1/G4 were re-decided while `planner_beats_cv`, a third banned verdict in the *same
   JSON*, went unnoticed. **The durable fix for (b) is mechanical: enumerate every boolean in an
   artifact before declaring its verdicts re-decided.**
6. **§11.2 stale text**, relayed from the coordinator and **flagged in place, not fixed**:
   `goal_evidence: grounded` is retired, the geometric lane-change gate is removed, and aug120 was
   re-fused.

---

## 8. DELIVERABLES

| artifact | repo path | state |
|---|---|---|
| this writeup | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-16-registry-estimator-sweep/REGISTRY_ESTIMATOR_SWEEP.md` | staged |
| registry — §1.2, §1.3, §2, §5, §6, §8 corrections + **new §11.1a** (Alpamayo-2) | `Project Steering/MODEL_REGISTRY.md` | staged |
| overview — P2 row + the P2 narrative | `Project Steering/PROGRAM_OVERVIEW.md` | staged |
| leaderboard — the open⊥closed footnote, P2 row, known-gaps block | `Benchmarks & Eval/LEADERBOARD.md` | staged |
| paper — **the pre-registered G4/G5 gate bars** | `Paper/TANITAD_PAPER.md` | staged |

**Evidence class.** Every number in §§1–6 is **MEASURED** — either recomputed by me this pass from the
banked artifacts (`planner_p2_flagship-30k.json`, `closedloop_flagship-30k.CORRECTED.json`,
`g1_g4_both_estimators.json`) or MEASURED by the 2026-08-16 jack-in-gates pass and re-checked against
those artifacts here. The site inventory is **MEASURED** (this pass, unbounded walk, 4,939 files, max
depth 8). The 27-arm blast-radius envelope and the ×−4.15 sign-flip figure are **INHERITED** from
`CLAUDE.md` / `JACK_BLAST_RADIUS.md` and used only as context, never as a decision input. §11.1a's
Alpamayo figures are **INHERITED** from the three named owning packages and marked as such *in the
registry row itself*; the only things I verified there are the arithmetic identities
(257 + 4,472 = 4,729; 1,418 × 1.30 GB = 1.84 TB) and that all three cited paths resolve.

**No test was loosened, skipped or deleted. This pass touched no file under `stack/` or `taniteval/`.**
