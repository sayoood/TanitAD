# COMMS — E-REFC-EGO-1: decisions asked, handovers, integration status

*Architecture & Inference FlyWheel · 2026-09-03. Owner: Research Lab literature stream.
⛔ **Nothing here is "please merge" buried in a README.** Each item names a target, an owner
and what happens if it is not picked up — an orthogonality instrument once sat unmerged for
10 days because the request lived in a doc nobody re-read.*

---

## 1. ⛔ ESCALATED — needs a decision or an owner

### C-1. E11 must be RE-SCOPED, not deleted — and the design doc must say why
**Target:** `refc_v3.py:213` (`# No ego state into any goal head (edge E11, REFUSED)`),
`REFC_V3_DESIGN.md:174`. **Owner: refcv4 design (Master Mind / the sibling code stream).**

E11 currently reads as an absolute prohibition justified by the vision-only rule. Two things
changed today:
1. **The PI reinterpreted the rule as anti-echo, not anti-ego-input.**
2. **This WP establishes that E11 is stricter than every planner in the published
   comparator set** — TCP, VAD, VADv2, Hydra-MDP, PARA-Drive and NAVSIM-TransFuser all feed
   ego status to the planner, and **none** withholds it from a goal or planner head by
   design (`RESULT.md` §5).

**The literature's line is drawn at the SCENE ENCODER, not at the planner** (`RESULT.md`
§5.1): the only closed-loop measurement of harm is CARLA-TransFuser's **DS 56.68 → 45.35**
when the velocity is summed into the backbone at all four stages — and refcv4's placement
(measurement encoder → decoder condition; the trunk never sees `v0`) is the **protected**
one.

**Ask:** restate E11 as a **boundary with a citation** — *"ego state never enters the visual
trunk (lib `2205.15997` Tab. 10, `2511.13079`); its entry into the planner condition is
permitted and instrumented (D1–D4)"* — rather than as a blanket refusal. ⚠️ **If E11 stays
absolute, refcv4 cannot consume `a_long` or `yaw_rate` in any head that produces a goal, and
the PI's 2026-09-03 instruction cannot be implemented.**

### C-2. The refcv4 gate must NOT be scored on ADE
**Target:** whatever SPEC pre-registers refcv4's success criteria. **Owner: whoever writes
that SPEC — this must be settled before the arm is launched, not after.**

Three independent primaries say a working anti-echo guard *lowers* displacement error:
- PlanTF: **OLS 88.55 → 87.07 (−1.48)** while **R-CLS 74.79 → 80.59 (+5.80)**.
- CADET: suppressing genuinely spurious reliance moves open-loop metrics by **< 0.1 m** —
  *"displacement error does not register a change in causal reliance"*.
- BEV-Planner: adding a map aux moved L2 **0.55 → 0.96** while CCR improved **4.26 → 2.60**.

**Ask:** the gate reads the **four metric families, manoeuvre-stratified, plus the E-ECHO-1
sensitivity ratio.** An ADE-only gate will reject the working guard. This is the
`EVAL_DOCTRINE` tier rule and the four-families rule pointing the same way, with external
corroboration.

### C-3. `refc.py`'s docstring must be corrected — it launders an unswept hyper-parameter
**Target:** `stack/tanitad/refs/refc.py:10-11`. **Owner: the sibling code stream (it already
owns `H-ARCH-EGOZERO-1` in the same file).**

The docstring says the TCP-C stack is kept verbatim including *"the measurement encoder with
per-sample ego-dropout"*. The **encoder** is TCP's; the **dropout is not** — TCP contains no
dropout of any kind (two probes, `RESULT.md` §2). A reader infers that `ego_dropout = 0.5`
is inherited, validated practice. **It is a TanitAD invention that nobody has swept**, and
the two papers that did sweep it landed on **0.5** and **0.75** with materially different
trade-offs. Same root-cause class as the registry's un-refined-anchor correction: a true
statement phrased so a reader infers a false provenance.

---

## 2. → HANDOVERS (work this WP identified but does not own)

| # | item | to | why it matters |
|---|---|---|---|
| **H-1** | **The rate for `H-ARCH-EGOZERO-1` now has external evidence.** PlanTF swept **0/0.25/0.50/0.75** (best closed-loop at 0.75); DRAMA landed on **0.5** for ego with **0.1** for perception features. Both also carry the **presence-bit** argument (GRU-D). | `2026-09-03-ego-zero-collision` (owner of the fix) | that WP measured the collision (22.5 : 1 withheld-to-real, a 23.5× train/eval meaning shift) but had no external prior for the rate. **The panel should sweep, not assume 0.5.** |
| **H-2** | **E-ECHO-2 (the CV-triviality + manoeuvre census) is 0 GPU and gates everything else.** Two published constructions to reproduce: NAVSIM's CV-PDMS-0.8 filter, and **PARA-Drive's label-only "targeted" split** (drop `keep forward` → 686 keyframes). | Benchmarks & Eval FlyWheel | if ≳80 % of our 881 val windows are CV-trivial, **every number on this split is dominated by the trivial subset** and no anti-echo method is measurable. This re-scopes existing results, not just future ones. |
| **H-3** | **Add D1/D2/D3/D7 to `taniteval`** — blank-image sensitivity, ego-perturbation response, endpoint-gradient norm, manoeuvre stratification. Three of the four need no training compute. | Benchmarks & Eval FlyWheel / `TanitAD_BenchmarkCriteria` | the criteria registry currently has no echo instrument at all. Published reference values exist for D1/D2 (`RESULT.md` §7), so the instrument can be *validated against a known value* rather than trusted. |
| **H-4** | **PARA-Drive's command-only floor (Coll 5.88 / L2 4.66) is the published form of our "constant-only control that must read the no-information value".** | criteria registry | we require this control by rule; here is a driving-domain precedent with numbers, which makes it easier to specify correctly. |
| **H-5** | **`obstacle.offline` (3D agent tracks, 97.44 % of the corpus) makes a detection aux feasible**; a BEV-segmentation or map aux is **not** (PhysicalAI has no map — pinned by `test_physicalai_feature_readset.py`). And **TCP's image→speed aux head, which we inherited, has never been ablated.** | Data Engineering + refcv4 design | §6.5's conflicted evidence is only resolvable by an ablation, and only the detection/speed variants are buildable on our corpus. |
| **H-6** | **VAD's `ego_lcf_feat` component list is UNRESOLVED at three probes** (`vad_nuscenes_converter.py:532`). It is the closest published analogue of our exact `(v, a_long, yaw_rate)` triple. | next literature slot | one file read settles it. Until then, do not assert what VAD's ego vector contains. |

---

## 3. ✅ INTEGRATION STATUS

| item | status |
|---|---|
| **15 primaries banked** (14 arXiv + **PARA-Drive by `--local` from the CVF PDF**) | ✅ done, `library.json` + regenerated `LIBRARY.md`, all `--cited-by` this RESULT.md |
| ⭐ **PARA-Drive converted PUBLISHED-SECONDARY → PRIMARY** | ✅ its Table 6 and the "targeted" split are now admissible; **the CVF 403 is bypassed with a browser User-Agent** — recorded in `raw/PRIMARY_EXTRACTS.md` §4 so the next agent does not re-hit the wall |
| `kb_add.py --verify` (content, not presence) | run over the whole library after banking |
| KB one-liners appended to `Architecture & Inference/Research/KNOWLEDGE_BASE.md` | ✅ |
| Staged, not committed / not pushed / not on `main` | ✅ per the operating standard |

---

## 4. Claims register — what this WP asserts

⚠️ **This WP asserts NO TanitAD claim.** Every finding is `PUBLISHED` about other people's
models, so `GOALS_AND_CLAIMS.md` gains **priors and instruments, not results**. Three items
belong there as **OPEN** design premises for refcv4, each with its external evidence:

| id | premise | evidence | status |
|---|---|---|---|
| **P-EGO-1** | *"Ego state in the planner condition is admissible; ego state in the visual trunk is not."* | lib `2205.15997` Tab. 10 (**−11.33 DS**), `2511.13079`, `paradrive-cvpr2024` §5, `2312.03031` App. C | **OPEN** — untested on our stack; E-ECHO-3 arm A/B tests it |
| **P-EGO-2** | *"A working ego guard lowers displacement error and raises closed-loop/tactical score."* | lib `2309.10443` Tab. II + VIII, `2606.14438`, `2312.03031` Tab. 3 | **OPEN** — decides how the refcv4 gate is written (C-2) |
| **P-EGO-3** | *"The corpus, not the architecture, sets how large the echo can be."* | lib `2406.15349` §3.1 (**CV 79 → 22**), `2506.04218` Tab. 2 (**1.28× → 4.0×**), `paradrive-cvpr2024` (686-frame targeted split) | **OPEN** — E-ECHO-2 measures it for us, 0 GPU |

---

## 5. ⚠️ Two traps recorded for the next agent in this area

1. **`git ls-tree -r` silently truncates on the G: mount** (CLAUDE.md). Everything in this WP
   was verified by **positive assertion** — `git ls-files --stage <path>` compared against
   `git hash-object <path>` for modified tracked files, `--cached` only for new ones — never
   by a file count and never by an empty result.
2. ⛔⛔ **`git add` ON THIS MOUNT CAN STAGE NUL-FILLED BLOCKS WITH EXIT 0 — AND IT IS
   REPRODUCIBLE, SO A SECOND READ CONFIRMS THE CORRUPTION INSTEAD OF CATCHING IT.**
   MEASURED 2026-09-03/04 while staging this WP: `library.json` was staged as a blob
   containing **147,456 NUL bytes (144 KiB, block-aligned)** while Python read the *same
   file* with **zero NULs** and full JSON validity (355 entries). `git hash-object` on a
   fresh read produced the **identical corrupted sha**, which is why repeating the command
   looks like confirmation. **Three of six text files were hit** — `library.json`,
   `KNOWLEDGE_BASE.md`, `LIBRARY.md` — **and one of the fourteen PDFs.** This is the
   `git ls-tree -r` trap's family, aggravated: the damage is written *into the shared index*,
   where another agent's commit would have carried it.
   ⚠️ **AND THE OBVIOUS CHECK PASSES.** The corruption **preserves the byte count**, so
   `git cat-file -s :<path>` matched the worktree size *exactly*; a marker `grep` on the
   staged blob also passed, because the marker happened to lie outside the zeroed region.
   ⇒ **Size + a marker grep is NOT a content check.** The only admissible check is a full
   **sha256 of `git cat-file blob :<path>` against the worktree bytes**.
   ⇒ **The repair that works — never let git read the file:** read the bytes yourself,
   `git hash-object -w --stdin` from that buffer, then
   `git update-index --add --cacheinfo 100644,<sha>,<path>`. Helper banked at
   `raw/stage_verified.py` (verifies every path, repairs via stdin, prints `OK(add)` /
   `REPAIRED(stdin)` / `STILL_BAD`). **Every path staged by this WP was verified this way.**
3. **`stack/scripts/scoped_commit.py` was LIVELOCKED tonight** (~10–13 min of `read-tree` on
   this mount, and HEAD moves inside that window, so it refuses). `C:\Users\Admin\mm_commit.py`
   is the working route: private index, re-seeds on HEAD movement, compare-and-swap
   `update-ref`.
