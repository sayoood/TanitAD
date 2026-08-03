# D-APPEAR — deliverable manifest

**Stream** appearance-shortcut audit · **Date** 2026-08-03 · **0 pod GPU-h.**
`tanitad-new` (v5f) and `tanitad-pod4` (v1arch) were **not touched** — no ssh, no process, no file.
All compute on the dev box (RTX 4060 for matmuls, CPU `eigh`), `OMP_NUM_THREADS` set on every run.

## Artifacts and where they live

| artifact | repo path | in ONE place only? |
|---|---|---|
| **Main deliverable** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-appearance-shortcut-audit/APPEARANCE_SHORTCUT.md` | no — staged |
| **Pre-registration** (outcomes S/C/P/VOID fixed in advance) | `Project Steering/PREREG_APPEARANCE_SHORTCUT.md` | no — staged |
| ⭐ **P1 verdict** — 15 arms + null, primary ratio, strata, four families | `…/2026-08-03-appearance-shortcut-audit/results_p1_physicalai.json` | no — staged |
| ⭐ **P1 per-window predictions** (gt, eid, centre speed, manoeuvre) | `…/results_p1_physicalai.preds.npz` | no — staged |
| ⭐⭐ **P1b mechanism ladder** (within-clip vs across-clip, both corpora) | `…/results_p1b_mechanism.json` | no — staged |
| **P2 cross-rig** (G1 horizon row, G2/G4 transfer, G3 shift sweep) | `…/results_p2_rig.json` | no — staged |
| **P3 scenario-classifier probe** | `…/results_p3_sitclf.json` | no — staged |
| **P4 latent-screen fleet pass** (4 substrates + oracle + reproduction check) | `…/results_p4_screen.json` | no — staged |
| ⭐ tables emitted FROM the JSON | `…/summarize.py` → `…/raw/summary_tables.txt` | no — staged |
| runners | `…/build_pai_substrate.py`, `run_p1_offhighway.py`, `run_p1b_mechanism.py`, `run_p2_rig.py`, `run_p3_sitclf.py`, `run_p4_screen.py` | no — staged |
| logs, incl. the crashed first P1 pass (kept as evidence) | `…/raw/run_log*.txt`, `…/raw/build_*.txt` | no — staged |
| pre-G3-fix P2 result (kept — the VOID cell is documented, not hidden) | `…/raw/results_p2_rig_beforeG3fix.json` | no — staged |
| ⭐ **INSTRUMENT (reusable)** — the 0-GPU latent screen, promoted from a run directory into the repo | `stack/tanitad/eval/latent_screen.py` | no — staged |
| ⭐ **INSTRUMENT TESTS** — 12 contract tests incl. the load-bearing reject test | `stack/tests/test_latent_screen.py` | no — staged |
| **Retraction entries** (2 new, one new class C16) | `Project Steering/RETRACTION_LOG.md` | no — staged |

### ⚠️ Lives in ONE place (dev box only) — derived caches, rebuildable from the staged runners

| file | size | rebuild |
|---|---|---|
| `C:/Users/Admin/tanitad-data/eval/dappear_pai_substrate.pt` | 492 MB | `python build_pai_substrate.py --n-episodes 240 --stride 6` (**130 s**) |
| `C:/Users/Admin/tanitad-data/eval/dappear_rigA.pt` | ~240 MB | `… --rig a --n-episodes 120 --out …/dappear_rigA.pt` (**125 s**) |
| `C:/Users/Admin/tanitad-data/eval/dappear_rigB.pt` | ~245 MB | `… --rig b --n-episodes 120 --out …/dappear_rigB.pt` (**131 s**) |
| `C:/Users/Admin/tanitad-data/eval/dappear_sitclf_still32.pt` | ~200 MB | rebuilt automatically by `run_p3_sitclf.py` (**105 s**) |

**Not staged and deliberately so:** all four are derived data, deterministic from banked inputs plus
the staged code, minutes to rebuild, and far too large for the repo. Their **inputs** pre-date this
stream and are owned elsewhere: the PhysicalAI episode caches
(`C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-{train-14231cd29c74,val-bb543bdf7836}`),
the sitclf latent bank (`…/eval/sitclf_b4_substrate.npz`, sitclf stream) and the comma2k19 latent +
pixel caches (`…/eval/idm_derived_accel_latents.pt`, `…/eval/dlatent_pixel_substrate.pt`, latent
stream). ⚠️ **None of those four inputs is in git either** — that is a pre-existing exposure this
stream inherited and did not create, and it is flagged in the escalations rather than silently
accepted.

## Verification

* `cd stack && pytest -q` → **2023 passed, 12 skipped, 2 xfailed** (630.9 s). The brief's baseline
  was 1900; the suite grew across several streams today and **12 of the new tests are this
  stream's**.
* Staging verified with `git ls-files --cached` — **33 paths tracked** — never with an `add` exit
  code (the documented silent-no-op trap). `git diff` over those paths is empty, i.e. the index
  matches the working tree.
* ⛔ **I committed nothing, pushed nothing, switched no branch, and touched no pod.**

### ⚠️ The repo advanced under this stream, and part of this work is already in HEAD

HEAD moved from **`59d2097`** (session start) to **`9fd9d61`** *("RETRACTED: the raw parity corpus
was not found on any live machine — Thor was holding it")* while these runs were in flight. That
commit **swept in the files this stream had staged incrementally** — `PREREG_APPEARANCE_SHORTCUT.md`,
`latent_screen.py`, `test_latent_screen.py`, `results_p2_rig.json`, `results_p3_sitclf.json` and the
runners are all in HEAD already; the remaining 15 paths (the final `APPEARANCE_SHORTCUT.md`, this
manifest, `results_p1_physicalai.json` + its `.preds.npz`, `results_p1b_mechanism.json`,
`results_p4_screen.json`, `summary_tables.txt` and the later logs) are staged on top.

This is the **`git commit` commits the ENTIRE INDEX** hazard from the other side: incremental
banking is the right behaviour under the stranded-artifact rule, and it means a sibling's commit can
carry your half-finished work under an unrelated message. **Nothing is lost or stranded** — every
artifact is tracked — but the lineage of `9fd9d61` is wider than its message says, and a reader
looking for D-APPEAR in the log will not find it there.

## Escalations — repeated here because a README is not a channel

The full list with evidence is `APPEARANCE_SHORTCUT.md` §5. The two that block other people:

1. ⭐⭐ **`LATENT_BOTTLENECK.md` §0.0 / RANK 1 must be amended** — the appearance-shortcut claim is
   corpus-specific and OUTCOME C was pre-registered as its withdrawal. **Owner: the latent stream.**
2. ⭐⭐ **Every doc quoting "+0.930 → −2.465" must be corrected to "+0.7863 → −2.4654"** — the +0.9297
   belongs to a different experiment in the same JSON. **Owner: whoever owns `MODEL_REGISTRY.md`.**
3. ⭐ **`GATE_PROTOCOL.md`: adopt `tanitad.eval.latent_screen` as a pre-flight gate.** Now a repo
   instrument with tests, so this is a protocol edit, not an implementation task. **Second stream to
   ask.**
4. ⛔ **LONGITUDINAL distance-keeping / TTC has no instrument on PhysicalAI** because no lead-agent
   channel reaches the episode cache, while `obstacle.offline` exists on 97.44 % of clips.
   **Owner: the data/ingest stream.**

## What is NOT done

* ⛔ **v5f was neither probed nor screened.** No banked v5f latent exists for these clips. This is
  the largest gap in the audit — v5f is the arm the programme is deciding about, and the wide-FOV
  cylindrical substrate is exactly where a geometry-vs-appearance question would bite differently.
  It needs the v5f checkpoint and one GPU pass over the wide cache.
* ⛔ **REF-C's ResNet trunk was neither probed nor screened.** Same reason.
* ⛔ **The cross-rig −2.4654 was NOT attributed.** Shown not to reproduce; geometry and
  head-extrapolation both remain live. Separating them needs their MLP head re-run on this cache.
* ⛔ **The "still frame through the frozen trunk" control was not run** (the sitclf stream's version
  of the control). My still-frame arms are raw pixels; the trunk-mediated version needs a GPU pass
  with `v1_speedjerk_ckpt.pt`. The two bound the same quantity from opposite sides.
* ⚠️ **One encoder throughout.** P4 widens the screen's CORPUS calibration 1 → 5; its ENCODER
  calibration is still n = 1.
* ⚠️ **The PhysicalAI corpus here is the r0 500-clip probe build**, not the canonical
  `physicalai-train-e438721ae894`. **No episode was re-selected for any training arm** — this is a
  read-only probe over a banked cache — but the contrast does not transfer to a training arm without
  re-running on the canonical cache.
