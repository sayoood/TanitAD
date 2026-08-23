# ADVERSARIAL VERIFICATION — stream `scenario-classification`

**Verdict: the report is ~80 % sound and independently reproducible, but its single loudest operational claim (the "HARD BLOCKER") is FALSE, its top escalation is FALSE as stated, and it inherited two load-bearing errors from the plan it was auditing without catching them.**

Everything below is MEASURED by me today (2026-08-02) unless marked. Absolute paths given. Repo working tree was untouched (`git status --short` = only pre-existing `.claude/settings.local.json`, `.claude/hooks/`).

Note on paths: the report writes `…\incoming\…`. There is no top-level `incoming/`. The real root is
`G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub\Architecture & Inference\Implementation\incoming\` (a *second* `TanitAD Research Hub\Implementation\incoming` exists and is empty of these). Abbreviated `HUB\` below.

---

## PART 1 — REFUTATIONS OF THE REPORT

### ⛔ R1. "No per-window camera-head scores are banked anywhere in the repo" is **FALSE**. The report committed the exact error it was policing.

`HUB\2026-07-26-situation-classifier\artifacts\heldout_frames.npz` (39,027,615 B, in repo since 2026-07-27) contains **per-frame held-out scores for all 10 arms**:

```
clip_cluster (308973,) int32   situations (3,) <U12   arms (10,) <U19
y (308973,3) uint8             valid (308973,3) uint8  ego (308973,3) float32
head_img_ego / head_img / head_ego / head_img_ego_concat / head_priv / head_img_shuf
ridge_img_ego / ridge_img / ridge_ego / ridge_img_shuf      each (308973,3) float32
```

Plus `checkpoints\head_{img,ego,img_ego,img_ego_concat,priv,img_shuf}.pt` and `pca.npz`, and `artifacts\train_summary.json` (which the report also said does not exist — true for **v2**, false for **gen-1**).

**I then executed the supposedly-blocked analysis, at 0 GPU, in seconds** (labels, valid mask, scores, and the `clip_cluster` ids the binding estimator needs are all present; `taniteval.ci.paired_episode_cluster_bootstrap` imports and is callable):

| gen-1 held-out AP | head_img | img_shuf | head_ego | ridge_img | img/null | ridge/null |
|---|---|---|---|---|---|---|
| lane_change | 0.03741 | 0.01715 | 0.08699 | 0.03405 | **2.18×** | 1.77× |
| roundabout | 0.00721 | 0.00328 | 0.02328 | 0.01056 | **2.20×** | 3.80× |
| **intersection** | **0.07955** | 0.03304 | 0.13494 | 0.07767 | **2.41×** | 2.47× |

(1,610 clip clusters; scorable 252,826 / 258,540 / 249,480; base rates 0.01725 / 0.00279 / 0.03054.)

The report probed only `*sitclf*` and v2 paths. **"Absence found at ONE location is not absence" — the rule it was enforcing.** The v2 heads *are* pod3-only; that is a narrower and true claim.

### ⛔ R2. "The 2026-07-29 promotion **silently** dropped the cross half" is FALSE as to "silently" — and this breaks the report's #1 escalation and its L0a.

`G:\...\stack\scripts\emit_situation_labels.py:3-9`, verbatim:

```
PI direction 2026-07-29: *"the classifier should detect the situation of lane change not
necessarily based on objects — only on situational labels, so you don't need objects. You need the
detection of lane changes and an intersection label."*

⇒ This produces exactly those two labels and nothing else. No object features, no `obstacle.offline`,
no detections.
```

The drop is an **explicit, documented, PI-directed scope decision recorded in the emitter's own docstring** — not a refactor regression. Consequences:
- Report Escalation #1 ("Non-ego signal … lost in a refactor. Highest-value cheap fix") is **wrong** and must not be filed as a defect.
- **L0a ("restore the dropped cross half … ⭐ do first") contradicts a standing PI directive** and requires PI sign-off. It also re-selects the positive set, so it breaks comparability with every v2 number.

### ⛔ R3. **NEW — the plan's §2.1 misstates its own evidence, and the report did not catch it.** This is L1's entire justification.

`HUB\2026-08-02-camera-situation-detection-plan\CAMERA_SITUATION_PLAN.md:119-121`:
> "under a **linear ridge probe the camera reaches 0.836 of ego, where the neural head puts it at 0.549.** A *linear* probe beating the *neural* head means the signal is present in the features and the head is the problem."

**ridge_img 0.04522 does not beat head_img 0.04869** — the neural head is *higher* (`SITCLF_V2_RESULT.md:79` vs `:27`). 0.836/0.549 are camera **÷ ego ratios**, not camera-vs-camera. No linear probe beat any neural head on the camera arm.

Worse, `SITCLF_V2_RESULT.md:103` says verbatim: *"Do not cite 'the vision pathway is under-trained' as a finding."* The plan's line 121 does exactly that.

### ⛔ R4. **NEW — cross-generation number comparison, in both documents.** Same error class as the v1 full-40 / 19-episode rule in my brief.

Plan `:52-54` and the report's §2.B both justify striking the `situations.py:19-22` docstring with "img-only **0.0376** → v2 **0.04869**". These are different label generations — `SITCLF_V2_RESULT.md:66-68` forbids it explicitly (*"⛔ NOT comparable to the gen-1 numbers … Compare within this table only"*): gen-1 = 3 situations, detector gen 1; v2 = 2 situations, detector gen 2.

**The admissible refutation needs no cross-generation step.** From gen-1's own artifact `train_summary.json`: `head_img` 0.037628 vs `head_img_shuf` 0.016562 = **2.27× its own null**. The docstring's numbers are *accurate* (0.0697/0.0525/0.0376/0.0166/0.1838 reproduce to 4 dp against `train_summary.json`: 0.06968/0.052505/0.037628/0.016562/0.183778). **Its defect is its conclusion, not its arithmetic** — it conflates "vision < ego" with "vision adds nothing", refutable from its own two numbers.

*(Direct check of the brief's named trap: v1's 0.4271 and the 19-episode 0.393 appear in neither document. Not violated literally; violated in class.)*

### ⛔ R5. **NEW — plan §3's "cannot answer the second-camera question" is refuted by a banked instrument.**

Plan `:161`: *"PhysicalAI-AV front-wide cannot answer 'what would a second camera add'."*

`HUB\2026-07-26-situation-classifier\artifacts\sc_results.json` → `camera_need`, per-camera lift with episode-cluster bootstrap CIs:

| intersection (n_clips=68) | lift | CI95 | separated |
|---|---|---|---|
| `camera_cross_left_120fov` | 1.100 | [1.002, 1.208] | **YES** |
| `camera_cross_right_120fov` | **1.283** | [1.157, 1.448] | **YES** |
| `camera_rear_left_70fov` | 0.928 | [0.843, 0.999] | no |
| `camera_rear_tele_30fov` | 0.752 | [0.664, 0.821] | no |
| `any_off_front` | 1.009 | [0.970, 1.045] | no |

(lane_change: only `any_off_front` separates, 1.120 [1.023, 1.234]; roundabout n=9, cross-120s separate.)

⚠️ Fair caveat: `camera_need` measures **agent-visibility lift**, not classifier-AP gain — a necessary-condition instrument, not the ablation. But the absolute "cannot answer" is false, and it points specifically at the **cross-120 cameras**, which the plan does not use.

### ⚠️ R6. "Strictly causal" is overstated — including in the source comments.

MEASURED: `np.gradient` is a **central** difference, so `situations.py:105` `omega = np.gradient(psi, DT)` makes `omega[t]` read `psi[t+1]`. Injecting a yaw step at t=25 gives nonzero `omega_pre` from **t=24**:

```
psi step at t=25; np.gradient nonzero at t = [24, 25]
omega_pre nonzero from t = 24  =>  leaks 1 frame = 0.1 s of future
```

So `situations.py:109` (*"STRICTLY CAUSAL"*) and `sc_build_labels.py:163` (same phrase) are both slightly wrong, and the report repeated them. **The §2.C refutation survives** — 0.1 s against a 3.0 s anticipation lead cannot reconstruct the rule — but the qualifier is mandatory.

### ⚠️ R7. The report's evidence for the ego-channel contents was a comment, presented as MEASURED.

It cited `sc_train_v2.py:37-38` — an `EGO_SCALE` variable name. The authoritative in-repo source is
`HUB\2026-07-26-situation-classifier\scripts\sc_build_labels.py:164`:
```python
packs[f"c{k}_ego"] = np.stack([K["v"], K["alon_pre"], K["omega_pre"]], 1).astype(np.float32)
```
corroborated by `heldout_frames.npz['ego'].shape == (308973, 3)`. Chain to v2 confirmed: `merge_v2.py:20` (`out[ge]=g[ge]`) copies gen-1's ego arrays into the v2 bundle. **Now genuinely MEASURED** — and the plan's *"It receives `[x, y, yaw, v]`"* is false on channel count as well as content.

### ⛔ R8. **NEW — both documents attach "tautology" to the wrong head.**

`sc_build_labels.py:238-253`, `_privileged(K)`, docstring verbatim: *"C-POS ONLY. The FUTURE 3 s summary the labels are literally built from"* — returns `[dpsi, lat, lon, kint, kmax, v[j]-v]` with `j = min(t+30, T-1)`. **`head_priv` (0.23765 CV; 0.49233 held-out on intersection) is the near-tautology arm.** `head_ego` is not.

### ⚠️ R9. Absence probe for a labelling tool missed a second app.

Report listed only `resim/static`. Also present: `stack\tanitad\scena\static\` + `stack\scripts\scena_app.py` (9,723 B FastAPI single-port SPA) — and `stack\scripts\resim_app.py` (6,227 B). **Conclusion survives** (scena is a scenario-*database* browser, per its README), but scena is the better skeleton: it already ships a server.

### ⚠️ R10–R14. Smaller corrections
- **R10.** `sc_cross_index.csv.gz` medians: report's 0.0018 m / 0.9958 are over **all 480** rows. Admitted-only (450): **0.0016 m / 0.9959**.
- **R11.** Line drift in `stack\scripts\vlm_route_labels.py`: `to_pil` is **304-307** (not 303-306), `pick_frames` **285-301** (not 285-300), `--stride` **382** (not 383). `vlm_kin_crossval.py:82` and `:61-66` are **exact**.
- **R12.** `vlm_kin_crossval.py:56` globs `*.json`; the banked pass-A is `.jsonl`. The "~80 % of L0's scoring join" needs a format adapter.
- **R13.** `SITCLF_V2_RESULT.md:66` attaches the **neural** shuffle (0.0166) to the **ridge** row; the ridge null is `ridge_img_shuf` **0.014425**.
- **R14.** `Project Steering\MODEL_REGISTRY.md` (2,092 lines) contains **zero** matches for `sitclf|situation|junction|head_img`. Every number in this stream is quotable only from raw artifacts — both documents under review are prose. Needs a PI ruling: registry gap, or probes are out of registry scope.

### ⚠️ R15. **NEW — urgency the report understated.** `stack\experiments\pod-rescue-20260802\` (dated **today**) swept `pod2/ pod3/ eval/ newpod/` and contains **no sitclf artifacts**. A pod3 rescue is in progress and `/workspace/sitclf/` was missed by it. Not "single-disk" — *single-disk on a pod under active rescue that skipped it.*

---

## PART 2 — WHAT I CONFIRMED (independently reproduced)

**§2.A identity — reproduced EXACTLY.** Script: `C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\verify_sitclf.py`, over 400 `ep_*.pt` in `C:\Users\Admin\tanitad-data\physicalai\_epcache\physicalai-train-14231cd29c74`:

```
identity detect_intersection(K,cross=None) == merge(turns − roundabout) : 400/400
cross events emitted (xe): 0
intersection 222 · raw turns 239 · roundabout 40 · lane_change 59
episodes with intersection 202 = 50.5 %
base_rate intersection 0.09839 (5,846/59,415)   lane_change 0.02317 (1,510/65,168)
ep cache: frames_u8 (199,9,256,256) uint8 · poses (199,4) · maneuvers (199,) int64
```
Every figure matches the report. `emit_situation_labels.py:58` (`ix,_turns,_x = detect_intersection(K)`) and `situations.py:293-297` confirmed verbatim at the stated lines. Local caches: **400** train + **100** val `ep_*.pt`.

**§2.D VLM — reproduced EXACTLY** from `HUB\..\Data Engineering\Implementation\incoming\2026-07-21-vlm-production-semantic\legacy_pod3_passA.jsonl` (400 records / 80 episodes, `nvidia/Cosmos-Reason2-8B`, `vlmroute-2026-07-20-a`):
```
P(junction visible)                  = 88/400  = 0.2200
P(ROUTE=straight | junction visible) = 4/88    = 0.0455
P(junction visible | ROUTE=straight) = 4/254   = 0.0157
P(junction visible | ROUTE=turn)     = 84/146  = 0.5753     odds ratio = 84.7
junction==False & road_geometry=='junction'    = 0 / 400
```
**And the contamination argument is stronger than the report made it**: `n_future_frames` ∈ {1,2,3,4}, `future_frame_times_s = [2.0, 5.0, 10.0, 15.0]` — future frames *were* shown (only the numeric track was withheld: `future_track_given=False` on all 400). `vlm_route_labels.py:194-195` — *"From the FUTURE FRAMES, determine where the vehicle actually goes"* — with `sees_junction_ahead` emitted inside that same JSON block. `--val` default is `physicalai-val-f1b378f295ae` with `parity.note_leaky_audit` at `:391-392`. **The report's HYPOTHESIS that this field is not clean scene truth is well-founded; I raise it to well-supported.**

**Everything else checked and confirmed:** all `SITCLF_V2_RESULT.md` figures at the cited lines (23-28, 77-80, 86-87, 99-103, 128, 156, 159); `situations.py:19-22` and `:29-32` verbatim; `sc_train_v2.py:36-38`; the causal window `off = torch.arange(-(WIN-1), 1)` at `:171`, `:178`, `:200`; V4 exact from `artifacts\label_validation.json` (0.0803 / 0.03326 / 2,939 / 3,428 / **2.415× [1.057, 7.931]** / separated / paired episode-cluster bootstrap B=2000); `sc_cross_index` admitted 450/480, `cross_frac` 0.0162 & 81/480 = 16.9 %, `perp_frac` 0.2383 & 343/480 = 71.5 %; `SITUATION_CLASSIFIER.md:147-149`, `:214-225`, `:823-824`, `:130-133`; `taniteval\taniteval\four_families.py` `tactical:206` / `strategic:254`; `ci.py:261 paired_episode_cluster_bootstrap`; `blind_baseline.py:152 balanced_accuracy`, `:141 VERDICTS` incl. `REFUSED`; `resim\static\app.js` 51,700 B. Param arithmetic self-consistent (3×8+1=25, 16×8+1=129, 129+24=153).

---

## PART 3 — NEW FINDINGS NEITHER DOCUMENT HAS

**N1. The camera's signal is strongest exactly where the plan says the target is unlearnable.** Per-situation held-out AP (R1 table): camera clears its null on all three, and its **best** situation is `intersection` (0.07955, **2.41×**) — the one the plan calls *"unlearnable by construction"*. Both docs only ever quote the pooled CV mean, which understates it. **This is a partial, 0-GPU counter-signal to plan §0** available today. It does not settle the question (still the ego-derived label), but it inverts the expected direction.

**N2. The "capacity, not information" pattern replicates in gen-1** — camera/ego linear 0.035231/0.040413 = **0.872**, neural 0.037628/0.06968 = **0.540**, against v2's 0.836/0.549. A free independent replication across label generations that no one has claimed.

**N3. A resolution/FOV confound L0 cannot separate.** `situations.py:34`: the model gets the **256 px / 51.4° front crop**; `vlm_route_labels.py:304-307` BICUBIC-**upsamples** 256→448. So (a) human gold labellers would view upsampled 256 px at 51.4° HFOV — cross-streets are largely out of frame and a 60-80 m junction may be unresolvable **to the human too**, making the plan's `>80 m` bucket unreliable; (b) part of `head_img`'s low AP may be FOV/resolution rather than the label. Plan §0 (label) and plan §2.3 (resolution) are confounded, and L0 as designed cannot tell them apart.

**N4. Four families.** The plan gives **no** family treatment at all. The report correctly routes sitclf to TACTICAL — but `four_families.py:166 _decision_family` expects decoded decisions with `pred_key`/`gt_key` against a `win` dict; **there is no sitclf adapter**, so L0c's "report via `four_families.py`" is not executable as written. That is a work item (CLAUDE.md: "a missing metric is a work item, not an excuse"), not a blocker.

---

## PART 4 — CORRECTED REPORT

### 4.1 Findings that stand

1. **The v2 `intersection` label is a pure ego function — an exact identity, not an approximation.** MEASURED 400/400, `xe=0` (`verify_sitclf.py`; `emit_situation_labels.py:58`; `situations.py:293-297`). `intersection` ≡ *ego executed a tight quantised quarter-turn (45-135°, ≤6 s, R≤25 m) that was not a roundabout*.
2. **This was a documented PI decision, not a regression** (R2). Gen-1 did use the cross union (`SITUATION_CLASSIFIER.md:147-149`); the PI directed objects out on 2026-07-29.
3. **`head_ego` is NOT a tautology.** It receives 3 causal-up-to-0.1 s channels `[v, alon_pre, omega_pre]` (`sc_build_labels.py:164`; `heldout_frames.npz['ego']` is (N,**3**)) over a strictly causal 8-step window (`sc_train_v2.py:171`). The label needs 3 s of future the head never sees. **Plan §0's escalation beyond C60 must be retracted.** The tautology arm is `head_priv` (R8).
4. **Plan §0's *label* argument survives** — but on the camera-side mechanism alone, and that mechanism is now **doubly under pressure**: by the VLM probe (though contaminated) and by N1 (uncontaminated).
5. **`situations.py:19-22` is a stale absence-claim** inside a file every downstream stream imports. Refute it **within gen-1** (2.27× its own null, R4), never by the cross-generation arithmetic both documents used.
6. **The banked `sees_junction_ahead` is not clean scene truth** (R-confirmed, strengthened). **Route A cannot reuse the existing prompt.**
7. `obstacle.offline` cross-traffic is real but sparse (V4 2.415× [1.057, 7.931]; 92 % of tight turns have no crossing agent). `perp_present` (0.2383 frame rate, 71.5 % of clips) is ~15× denser and is a good **sampling stratifier / auxiliary target**, never ground truth.
8. Four cache keys in play (`e438721ae894` v2 labels / `f1b378f295ae` VLM-leaky / `14231cd29c74` + `bb543bdf7836` local); base rate spans 0.02816 → 0.09839. **AP is not comparable across them.**

### 4.2 Corrected build list

**Executable at 0 GPU, today, in-repo (report said blocked):** gen-1 per-frame held-out scores for all 10 arms + `y` + `valid` + `clip_cluster` (`heldout_frames.npz`); gen-1 checkpoints + `pca.npz`; `train_summary.json`; `sc_results.json` (incl. `camera_need`); `label_validation.json`; `paired_episode_cluster_bootstrap`; `blind_baseline`; 500 local episodes with imagery.

**Genuinely missing:** a labelling UI (confirmed absent at 4 probes — filename scan, all `.html`, `taniteval/`, `stack/scripts/`; `resim` and `scena` are viewers); a clean scene-truth prompt/rubric; a sitclf→`four_families` adapter; a `.jsonl` adapter for `vlm_kin_crossval.py`; **v2** per-window scores (pod3-only, and R15-urgent).

### 4.3 Corrected plan

- **L0a — DROP as written.** Restoring `cross` contradicts a standing PI directive and breaks v2 comparability. Re-file as a **question to the PI**, not a defect.
- **L0-zero (NEW) — ⭐ do first, 0 GPU, hours.** Publish the gen-1 **per-situation** held-out table (R1/N1) with **paired episode-cluster bootstrap CIs** over the 1,610 clusters, plus the `blind_baseline` balanced-accuracy verdict. Everything needed is in the repo. This is the cheapest discriminating step in the whole stream and it partially pre-empts L0.
- **L0b — gold set, revised.** Keep n≈400, stratified on `perp_present` × turn/straight × day/night; current frame only, no future frames, no route question; written rubric; **double-label ≥80 and publish Cohen's κ**. **Add:** record labeller confidence per distance bucket and **report the `>80 m` bucket separately** — at 256 px / 51.4° it may be unresolvable to the human (N3).
- **L0c — adjudicate.** Score gold vs (i) ego-derived label and (ii) the **gen-1 camera head** (in-repo, 0 GPU). The v2 head needs pod3. Report per-family; state LONGITUDINAL/LATERAL **N/A for a classification target, with the reason and n**; build the TACTICAL adapter (N4).
- **L1 — re-justify before spending.** Its stated rationale is refuted (R3). It may still be right; it needs a new argument.
- **Camera-count — partially already answered** (R5). Use `camera_need`; the cross-120s are the candidates.

### 4.4 Pre-registered outcomes

Keep the report's five, with these changes: **#3's threshold cannot be read off the contaminated VLM** — it must come from the gold set only. **Add #6:** *if the gen-1 per-situation table (L0-zero) shows camera AP highest on `intersection` with CI excluding the null* — as the point estimates already do — *plan §0's "unlearnable by construction" framing is weakened independent of any gold set, and §0 must be re-argued before L1-L4 spend.*

### 4.5 Escalations (revised)

1. ⛔ **Plan §2.1 (lines 118-121) misstates its own source and violates an explicit "do not cite" hedge** (R3). Correct before L1 is funded. `RETRACTION_LOG.md` class: *misread evidence / ratio-vs-level confusion*.
2. ⛔ **Plan §0's `head_ego`-tautology claim is refuted** (R7/R8). Class: *overstated-circularity*.
3. ⛔ **Cross-generation comparison in both documents** (R4). Class: *comparing numbers computed on different populations* — the `0.4271`-vs-`0.393` class.
4. ⛔ **Plan §3's "cannot answer the second-camera question" is refuted by a banked instrument** (R5).
5. ⚠️ **URGENT — v2 sitclf heads/bundles are on pod3, which is under rescue TODAY, and the rescue missed them** (R15).
6. ⚠️ `situations.py:19-22` stale absence-claim (refute within gen-1).
7. ⚠️ `SITCLF_V2_RESULT.md:66` mismatched null (R13); `:138` stale (the trainer **is** in-repo).
8. ⚠️ **`MODEL_REGISTRY.md` has no sitclf entry** (R14) — PI ruling needed.
9. ⚠️ `situations.py:109` / `sc_build_labels.py:163` "STRICTLY CAUSAL" is off by one frame (R6).

### 4.6 UNVERIFIED (flagged, not deleted)

- nuScenes/Waymo junction polygons — **PUBLISHED, not re-verified** by the plan, the report, or me. Route B rests on it.
- No map/GNSS in PhysicalAI-AV — **INHERITED** from CLAUDE.md's five-probe settlement; I did not re-probe.
- Cosmos ungated/commercial-OK — **INHERITED** (2026-07-20 byte-pull); not re-probed.
- Two-rig `cy` (543/755, ~215 px) — **INHERITED**; not re-probed. Moot for the ep cache (already cropped), live only if L0 re-crops from raw.
- v2's own numbers are unreproducible in-repo: `cv_log_lines.txt` is a log, and no v2 `train_summary.json` exists (`SITCLF_V2_RESULT.md:128`). **Every v2 figure is INHERITED-from-log until pod3 is pulled.** Gen-1, by contrast, is fully re-derivable in-repo — which is why L0-zero should be run on gen-1.