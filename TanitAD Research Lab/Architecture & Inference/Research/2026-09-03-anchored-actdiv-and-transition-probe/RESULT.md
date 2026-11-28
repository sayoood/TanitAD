<title>RESULT — anchored actdiv (GS-8), the transition-level probe rung (GS-9), and refav1's lateral insensitivity (2026-09-03)</title>

# RESULT — E-BE-GS89-0903

`TanitAD Research Lab · Architecture & Inference · Benchmarks & Evals FlyWheel · 2026-09-03`
`Tier: T0-DIAGNOSTIC on every number below — world-model probes, never driving performance.`
`Device: dev-box CPU for EVERY reading (the RTX 4060 carried the paired T1 read all night and was`
`never touched). Evidence class: MEASURED (ours) unless marked; every number has a raw path in §7.`
`SPEC.md was written before the runs; the thresholds quoted here are the ones committed there.`

⛔ **INTEGRATION NEEDED (headline for the coordinator):** (1) the two tools + two test files are NEW
and staged, not wired into any runner — `mm_e19_read.py`'s actdiv stage still calls the banked
`actdiv_local.py`; (2) the **banked `actdiv` instrument has a lift discrepancy** (§2.1) that the
registry rows inherit — a decision is needed on which lift the MM-E19/MM-E10 ratios are quoted
under; (3) three of the five requested v7 checkpoints (`o11p30k`, `splitp30k`, `postrain30k_freeze`)
are NOT on this box and were skipped, never approximated — they need a Thor pull (not by me:
Thor was off-limits tonight); (4) two proposed register rows + one amendment (§6) await adoption.

---

## 0. The answers, one line each

| question | answer | class |
|---|---|---|
| **GS-8** Is the v7 predictor's action response STRUCTURED once anchored on its own zero-action prediction? | **YES — on all four local 30k arms the anchored response is SEPARATED (F_sep 7–136× above its permutation null), PERFECTLY LINEAR on the κ axis (cos(+L,−L) = −1.00 at every level, Spearman 0.985), and monotone on the accel axis — but the accel response ROTATES at ≥ 2σ on the three postrain-recipe arms (cos(+L,−L) turns positive), so they read SEPARATED-NONMONOTONE; `rdw8p30k` stays sign-consistent and reads STRUCTURED-WEAK. None is MATERIAL: the per-dim response at 2σ is 0.3–1.8 % of the scene std (bar 5.95 %).** | MEASURED |
| **GS-8 instrument** | The banked `actdiv` scripts feed `v_first/30`; the trainer trained on `v_last/10`. Under the trainer's lift the legacy ratio on `postrain30k` reads **0.0273**, not 0.0059 — because the legacy roll moves the SPEED-STATE channel with the actions. The anchored read holds v fixed and is immune to this. | MEASURED |
| **GS-9** Does the encoder displacement Δz carry the ego transition; does the PREDICTOR's Δẑ add anything over z_t; does the action reach the transition? | *pending — §4* | — |
| **refav1** Is the step-1,000 predictor insensitive to κ but not to a (H-REFAV1-LAT-INSENSITIVE)? | *pending — §5* | — |

---

## 1. What was run, on what, and what was not

| tool | checkpoints | population | n | device |
|---|---|---|---|---|
| `taniteval/tools/actdiv_anchored.py` (v7 family) | `postrain30k` (md5 `a58585883c27…`, registry §13.9), `k8clip05p30k` (md5 `2d744d6d2faa…`, ⚠️ no registry row — the k8 package's raw JSON is its primary), plus the unrequested local `k60clip05p30k` (`7e3c776d…`) and `rdw8p30k` (`6e382ebe…`) | the banked `actdiv` windows, reproduced VERBATIM: 24 sorted clips of `physicalai-val-0c5f7dac3b11-w120-256x640cyl` (local, 24/24 md5-listed), first 60 frames, W = 6 at `range(0, n, n // 5)` | **144 windows/arm** | CPU |
| same, `--actdiv-compat` | same four | same | 144 | CPU |
| `taniteval/tools/transition_probe.py` | `postrain30k`, `k8clip05p30k` | `physicalai-val130-heldout` (129 clips, local), first 100 frames, dataset-consistent rows, split BY CLIP | *§4* | CPU |
| `actdiv_anchored.py --family refav1` | `refav1_eval_slice\ckpt\ckpt.pt` (fp32 incumbent, step 1,000, cfg from `ckpt['cfg']`), `refav1_eval_slice\ckpt_ep2\ckpt.pt` (clean epoch, step 1,000, `config.json` beside it) | the T1 read's own windows: 20 eval-slice episodes, `k_loader = 30`, stride 10 | **140 windows** | CPU |

**Not run, and why (two probes each — a name search and a listing of every `ckpt*.pt` > 50 MB
under `C:\Users\Admin`):** `o11p30k`, `splitp30k`, `postrain30k_freeze` are not on this box; they
live on Thor (`/home/nvidia/v7tiny/<arm>/ckpt.pt` per the k8 package's manifest), which this task
was forbidden to contact. The tools are ready for them (one CLI invocation each). ⚠️ GS-10's
mechanism hypothesis therefore stays OPEN — it is the o11 reading that would test it.

**Registry conflicts found (rule 4):** `MODEL_REGISTRY.md` has NO row for `k8clip05p30k` and NO
row for `o11p30k` (0 hits each; `k8clip05` 0 hits). Their model facts here come from raw JSON
(`…/2026-09-01-mm-e19-k8-attribution/raw/mm_e19_read_step30000_K8.json`) and from the
`V7_LAUNCH_GATE.md` P2 block respectively, and are marked as such.

---

## 2. GS-8 — anchored action-divergence on the v7 arms

### 2.1 ⛔ The instrument fact that must travel with every banked `actdiv` number

Both banked scripts (`actdiv_thor.py`, `actdiv_local.py`) hardcode **`SPEED_SCALE = 30.0`** and feed
the speed at the window's **FIRST** position; the trainer's `_lift3` (`scripts/train_v6_staged.py:3604`)
divides by **`flagship_v15.SPEED_SCALE = 10.0`** and feeds **`pose_last[:, 3]`** — the LAST position
(`train_v6_staged.py:5717`). The E-DEC-59 helper `v7tiny_g2.py` imports the trainer's constant, so
this is specific to `actdiv`. MEASURED consequence, same 144 windows, same code path, only the lift
differing (raw: `actdiv_anchored_v7_trainerlift.json` vs `…_actdivcompat.json`):

| arm | legacy `ratio_action_over_scene` h=1, banked lift (v_first/30) | same, TRAINER lift (v_last/10) | banked reference |
|---|---|---|---|
| `postrain30k` | *§7 compat JSON (reproduction check)* | **0.027338** (action 0.00631 / scene 0.23099) | 0.005947 (Thor, MM-E10; dev-box 0.005947, k8 package) |
| `k8clip05p30k` | *idem* | **0.014359** (0.00305 / 0.21240) | 0.006165 (k8 package) |
| `k60clip05p30k` | *idem* | **0.007742** (0.00283 / 0.36503) | 0.002983 (MM-E19 read) |
| `rdw8p30k` | *idem* | **0.026363** (0.00503 / 0.19096) | — |

Why the lift moves it 3–5×: the legacy instrument ROLLS the whole 3-channel action across windows,
so its "action variants" also swap the SPEED-STATE channel (a state, not an action); at v/10 that
channel varies 3× more than at v/30. ⇒ the banked ratio is partly a speed-sensitivity read. The
anchored read below replaces only the action channels and keeps each window's own v — which is
why it can be quoted without this caveat. ⚠️ Which lift the MM-E10/MM-E19 rows should be quoted
under is a decision for the coordinator; both numbers are banked here.

### 2.2 The anchored read (h = 1, the only trained head; `replace = last`, 20 candidates + zero)

Controls on every arm: **C0 identity = 0.0 exactly; zero-model = 0.0 exactly; output-scale
invariant; scene spread 0.19–0.36 (C1 far from 0)** ⇒ the panel is admissible.

| arm | F_sep | null median / p95 | **F/p95** | κ: ρ · cos(+,−) at 0.25…3σ | a: ρ · cos(+,−) at 0.25…3σ | rel-mag @2σ (per-dim RMS / scene std) | **verdict (SPEC §4)** |
|---|---|---|---|---|---|---|---|
| `postrain30k` | 16.39 | 0.91 / 1.05 | **15.6** | 0.985 · −1.00 −1.00 −1.00 −1.00 −1.00 | 0.985 · −0.98 −0.92 −0.70 −0.17 **+0.23** | 0.0087 | **SEPARATED-NONMONOTONE** |
| `k8clip05p30k` | 21.87 | 0.86 / 1.01 | **21.6** | 0.985 · −1.00 ×5 | 0.985 · −0.96 −0.86 −0.50 **+0.17 +0.50** | 0.0138 | **SEPARATED-NONMONOTONE** |
| `k60clip05p30k` | 7.19 | 0.91 / 1.05 | **6.8** | 0.985 · −1.00 ×5 | 0.985 · −0.96 −0.83 −0.43 **+0.26 +0.57** | 0.0032 | **SEPARATED-NONMONOTONE** |
| `rdw8p30k` | 136.1 | 0.99 / 1.28 | **106.3** | 0.985 · −1.00 −1.00 −1.00 −0.99 −0.99 | 0.985 · −1.00 −0.99 −0.95 −0.80 −0.62 | 0.0181 | **STRUCTURED-WEAK** |

Per-level magnitudes (per-dim RMS of the mean displacement over the per-dim scene std; the
κ axis is the corpus' `atan(L·κ)` proxy, σ = 0.0043 on these windows; the a axis σ = 0.52 m/s²):

| arm | κ @1σ | κ @3σ | a @1σ | a @3σ | κ/a at equal σ |
|---|---|---|---|---|---|
| `postrain30k` | 0.00145 | 0.00435 | 0.00647 | 0.0240 | 0.22 |
| `k8clip05p30k` | 0.00122 | 0.00366 | 0.0101 | 0.0447 | 0.12 |
| `k60clip05p30k` | 0.00022 | 0.00066 | 0.00224 | 0.0110 | 0.10 |
| `rdw8p30k` | 0.00074 | 0.00223 | 0.0176 | 0.0512 | 0.04 |

**What the structure says.** On every arm the response to κ is a straight line through the origin
(norm ∝ |level| to three digits; +L and −L exactly antiparallel) — the predictor's κ path is a
LINEAR map with a small gain. The response to a is monotone in magnitude but, on the three
postrain-recipe arms, its DIRECTION rotates as |a| grows until +a and −a share a common component
(cos > 0 at 2–3σ): a "something is happening" direction that does not distinguish braking from
accelerating. `rdw8p30k` (the older recipe) keeps both axes antiparallel. Neither reading is the
LeWM picture (means at the origin, overlapping): the means are well separated relative to the
within-candidate scatter. And none is MATERIAL: at 2σ the response is 0.3–1.8 % of a scene change
against the programme's 5.95 % bar, i.e. the same magnitude story the banked ratio told — now with
the structure visible.

Robustness (raw JSON): under `replace = all` (every window position gets the candidate) F/p95
falls to 10.8 / 17.8 / 5.3 / 70.1 with the same ρ and the same accel sign rotation; under input
rescale z ← 0.5z / 2z the verdict class is unchanged on all four arms (`verdict_invariant: true`).
The REALISED read (each window under its own action, quartile bins) does NOT separate on the
postrain-recipe arms (F 1.1–4.0 against shuffled-action floors 1.4–3.9) — the realised actions on
these windows are too small and too scene-correlated for quartile bins; only `rdw8p30k`'s accel
bins separate (36.4 vs 14.7 shuffled). The grid read is the one to quote.

**Reading against SPEC §4 and what it does to P2:** `H-GS8-1` is **SUPPORTED with a qualification
the pre-registration named**: the anchored response is structured — on κ perfectly — but on a it
is monotone-in-norm and NOT sign-consistent at large |a|, which the committed table classes as
SEPARATED-NONMONOTONE. P2 is therefore a GAIN problem on the κ path (a clean linear map, gain
~10⁻³ of the scene) and a GEOMETRY problem on the a path (a response that stops distinguishing
sign). "The predictor does not use its actions" is too coarse: it uses them consistently, tiny,
and — for acceleration — without preserving sign beyond ~1σ.

---

## 3. Units note (recorded before any verdict was quoted)

The first tool version divided the L2 norm of the 2048-d mean displacement by the per-dim scene
std — √2048 ≈ 45× too large — and would have read `rdw8p30k` as SENSITIVE. Caught on the first
real read, corrected to the per-dim RMS (SPEC §11), verdicts re-derived from the stored unscaled
norms, and the v7 passes re-run with the corrected tool (§7 lists both JSONs; the numbers other
than `rel_to_scene`/`material` are identical between runs, as they must be for a deterministic
forward with a fixed seed).

---

## 4. GS-9 — the transition-level rung (pending the CPU pass; filled in §4 of the final version)

*(The panel runs on 129 held-out clips; results are inserted here when `transition_probe_v7.json`
lands. Pre-registered readings R1–R4 in SPEC §7.)*

---

## 5. refav1 — H-REFAV1-LAT-INSENSITIVE (pending; filled in §5 of the final version)

---

## 6. Proposed register rows (for `GOALS_AND_CLAIMS.md`; not written by me)

*(drafted after §4/§5 land)*

## 7. Raw artifacts

*(final manifest at the end)*
