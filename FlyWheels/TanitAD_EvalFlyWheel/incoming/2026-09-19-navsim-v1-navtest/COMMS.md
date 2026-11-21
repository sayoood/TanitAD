# COMMS — W3 (NAVSIM v1.1 `navtest`)

## Decisions MADE in this stream (each reversible; the PI / orchestrator may overrule)

| # | decision | why (evidence) |
|---|---|---|
| D1 | **The v1.1 runtime is the EXISTING C: venv with `PYTHONPATH` → the D: v1.1 tree**, not a new venv on D:. | The venv satisfies v1.1's `requirements.txt` (torch 2.0.1+cpu, numpy 1.23.4, shapely 2.0.7, hydra 1.2.0) and its `nuplan-devkit` is exactly v1.1's pin `ce3c323` (tag `nuplan-devkit-v1.2`). `navsim.__file__` is ASSERTED to be inside the v1.1 tree on every run (control C8) — the venv's own navsim 2.0.0 `.pth` is appended after site-packages, so PYTHONPATH wins, but the wrapper does not trust that. All v1 DATA (cache, frames, bank, export) is on **D:** as the PI ruled. |
| D2 | **E1's loader patch is APPLIED to v1.1** — after proving the defect on a real v1.1 cache (C7: the unpatched loader raises `IndexError`; the patched one maps 20/20 tokens to their own paths) and proving the two function bodies are the same code except the token expression (`tests/test_v1_loader_patch.py`, 5 tests, literal expectations + the defect exercised). | v1.1 `dataloader.py:180` has the identical `cache_path.split("/")[-2]`; nuPlan writes `str(WindowsPath)`. |
| D3 | **STOP is an AGENT (`w3_agents_v1.StopAgent`), not a seam file.** | v1.1's runner hands `requires_scene=False` agents only the `AgentInput`; a zero-plan agent reads nothing at all, which is a stronger declared-input statement than a lookup table, and it costs no Scene build. |
| D4 | **The metric cache is built one log per process** (`--per-log`), then one final devkit pass writes the metadata CSV. | MEASURED 2026-09-20: the devkit's `cache_data` loads ALL 136 logs first (2.1 GB RSS) and a sibling agent's CPU training smoke (33.8 GB committed, system available → **419 MB**) tripped the RAM guard and cost the pass. |
| D5 | **The frame bank stores UNIQUE frames + an index**, not one `.npy` per scene. | exFAT clusters are 1 MiB: a per-scene layout costs ~36 GiB allocated and 24,292 files for ~24 GB of data; navtest windows share up to 3 of 4 frames. **KB1: 20/20 scenes bit-exact vs E2's unmodified builder**, incl. the `src` map, rig key, observed fraction and mean pixel. |
| D6 | **Per-token rows are streamed to JSONL** and the CSV can be rebuilt from them (labelled `csv_reconstructed_from_rows_jsonl`). | The orchestrator's hazard note + the navhard evidence: the v2 runner scored 5,912 scenarios and then died in aggregation. |
| D7 | The interval is **W2's registered log-cluster bootstrap** (`taniteval/adapters/navsim_ci.py`, `PDMS_v1_navtest` → single-stage nanmean, clusters = the 136 logs ≥ the RG-14 floor of 8), with W2's own cluster map `raw/cluster_maps/navtest.json` as the token → log source. | It landed while W3 was running; SPEC §2 pre-registered exactly this ("UNAVAILABLE until W2 registers it"). |

## ⭐ Integration asks (escalated; also in the report headline)

0. ⛔ **W1 — `GpuGapLauncher` on `auto` (and on `cuda` at timeout) returns `"cpu"` WITHOUT the CPU
   gate's condition being evaluated. MEASURED on W3's live A1 launch, 2026-09-20.**
   The orchestrator's arbitration has two halves: *`auto|cuda` → wait for a GPU gap*, and
   *`cpu` → allowed only when **no training process is alive*** (else the named override, recorded).
   `gpu_gap.py:246-250` returns `cpu` on `auto` after ONE probe, and `:251-254` returns `cpu` on
   `cuda` after `max_wait_s`; **neither path consults the second half**, so
   `--device auto --gpu-gap` on a box with a live trainer runs CPU inference beside it — the exact
   case the arbitration refuses. The observable: `[gpu-gap] NO_GAP_CPU {'reasons': ['GPU memory
   used 1906 MiB >= limit 1024 MiB']}` and the model on CPU **1.2 s later**.
   **W3's local close (not an edit to W1's file — you are in code freeze):** on any `cpu` return
   from `acquire()` the bridge re-asks `device_decision` as if `--device cpu` had been typed and
   **refuses** on a no; the accepted decision is recorded in the manifest's `device_gate.cpu_regate`
   alongside `launcher_semantics`. **Ask: fold the equivalent check into the launcher** (a two-line
   `probe_training_pids` re-check before returning `cpu`) so every caller inherits it, or tell W3
   the caller is the right place and it will stay here.
   ⚠️ **Documentation half, for whoever writes the suite's operator page:** the 60 s / 8 h
   re-check belongs to **`cuda`** only. `auto` is a single probe. W3 itself wrote the 8 h behaviour
   into `RESULT.md` for an `auto` run and has corrected it (see below) — the wording invites it.

1. **W1 — the `navsim_v1` plugin is landed in my slot** (`taniteval/taniteval/bench/plugins/navsim_v1.py`,
   `run_benchmark(ctx)` against `BenchContext`). It **REFUSES with the exact build command** when the
   navtest metric cache is absent rather than starting a 4–5 h job inside a CLI call, and it
   **reuses** a PASSed W3 scoring run of the same cache (sha256 recorded, `provenance.reused_from`)
   unless `W3_NAVSIM_V1_REUSE=0`. Please confirm both behaviours are acceptable in the suite's
   acceptance test.
2. **W1 — model arms on `navsim_v1` are REFUSED by design until (a) the navtest frame bank exists
   and (b) a GPU gap opens.** The refusal carries the queue command. When W1's gap launcher is the
   only GPU route, `code/run_bridge_navtest.py --gpu-gap` already acquires through
   `taniteval.bench.gpu_gap.GpuGapLauncher`.
3. **W4 — the published reference table for the leaderboard is in the plugin's
   `provenance.published_reference`** (banked primary `2406.15349` v2, sha256 `d3bc66d3…`, Table 1
   p. 7 and Table 3 p. 9, plus the INHERITED HF navtest LB rows). ⛔ `PDMS_v1_navtest` is its own
   table: never mixed with EPDMS.
4. **⭐ EVERY navtest row needs its STOP floor, exactly as E2 showed for warmup.** On the 20-token
   smoke STOP already scores **61.48** against CV's **37.62** (human 92.12). If that survives the
   full split, **CV (published 20.6) is not a floor on navtest** and a do-nothing plan lands within
   reach of the published Ego-Status-MLP row (65.6 / 66.4) — a protocol property, reported with the
   mechanism (`5/12 = 41.7` is free whenever NC·DAC·TTC = 1), not a model claim.
5. **Orchestrator — two after-A8 chains exist on this box**: E1's navhard chain
   (`2026-09-19-navhard-download/code/navhard_after_a8.sh`, which STOPPED at N1 — the aggregation
   crash) and W3's (`code/after_a8_chain.py`). Mine holds ONE worker, waits for ≥ 4 GB available
   before each step and logs the live training processes it is sharing the box with.
6. **Orchestrator / Master Mind — the box's memory is the binding constraint, not the CPU.** A
   sibling CPU training smoke (`refc_v3_train.py --device cpu --batch 4 --image-hw 416 1024`,
   PID 34760) committed **33.8 GB** and drove system-available memory to **419 MB**, aborting my
   cache pass (RAM guard, exit 3). Per-log caching makes W3 robust to it, but any agent planning a
   CPU forward at that resolution should know it can evict every other job on the box.
7. **PI — the Ego-Status-MLP reproduction (SPEC BAR-W3-3) is BLOCKED on a download approval**
   (relayed as with the PI; W3 has NOT fetched anything and will not until an approval is relayed):
   `autonomousvision/navsim_baselines/ego_status_mlp/ego_status_mlp_seed_{0,1,2}.ckpt`,
   **6,518,594 / 6,518,530 / 6,518,530 B** (HF API listing 2026-09-19, sha256 per file known). They
   are the checkpoints the v1.1 leaderboard row (66.4 ± 0.9) was populated with. ⛔ Not downloaded:
   a download needs an explicit approval, and nothing here trains one.

| D8 | **The suite device gate is called, not re-implemented**: `plugins/navsim_v1.py::device_decision` prefers a named shared gate exported by `taniteval.bench.gpu_gap` and otherwise composes that module's own `probe_training_pids`; `code/run_bridge_navtest.py` calls the SAME function (rc 3 on a refusal). | Orchestrator arbitration 2026-09-20: auto/cuda wait for a gap; explicit cpu only with no training alive, or with `--accept-training-box-load`, whose override and accepted process are recorded in `bench_run.json: device.gpu_gap`. 9 tests, incl. the refusal arm and the "W1 exported a gate" arm. |
| D9 | **Reuse is keyed on `(arm runner, devkit agent class, metric cache)` and the banked CSV is RE-CHECKED before use.** | Ruling 2026-09-20 (a)+(b). ⚠️ W3 did NOT already do both: the key was cache-only and the old verdict was trusted. Now `AGENT_CLASS` gives three distinct values (no arm can inherit another's CSV) and `recheck_csv` re-runs rows/valid/summary-row/token-set/C1 on the file as it is now — a fresh CSV gets the same re-check. ⚠️ v1.1 carries exactly **1** summary row (`average`); the ruling's "3 summary rows" is the v2 shape. |

| D10 | **`PDMS_v1_navtest` summary row = 1 row, token `average`** — MEASURED, for W1's `profiles.summary_row_shape`. | Four real v1.1 CSVs (CV/HUMAN/STOP/CV-replicate) each carry exactly one non-hex token row reading `'average'`, and `run_pdm_score.py:145` assigns that literal. ⇒ W1 can restamp the class from DECLARED to MEASURED; evidence `raw/W1_summary_row_shape_navsim_v1.json`. The full-split CSVs will be re-read the same way when they land. |

## ⛔ W3's fourth self-correction — a PRE-REGISTERED metric that read the blind arm as better (2026-09-21)

**What W3 registered** (SPEC §7c, 08:00:16Z): curvature following = the **median** model/human
net-heading ratio on the 771 curve tokens, with the committed reading *"A4 > A1 + 0.05 ⇒ vision
actively HURTS curvature following."* It read A1 0.1487, A4 0.6664 — so the registered reading is
"vision hurts".
**What is true:** the frames-blind arm turns hard in near-random directions (wrong way 41.6 %,
over-turning 36.8 %) and stays on the road 28 % of the time against A1's 55 %. The ratio is SIGNED;
its median lands inside the blind arm's large positive tail and scores "turns more" as "follows
better". Reported as written, marked REFUTED BY ITS OWN TOKENS; no bar depended on it.
**Class:** *a location statistic over a mixed-sign ratio, registered without a control that a
random-direction arm must fail.* Same family as the probe rule in CLAUDE.md (every panel carries a
control that must read a known value) — the pre-registration had controls for the SMOOTHER and
none for the METRIC.
**Durable fix:** `code/turn_following.py`'s consumers now read the sign decomposition (wrong-way,
over, barely, follows) beside the median, and any future curvature metric is registered with a
random-direction control that must read as NOT following.

## ⛔ W3's third self-correction — the model's "long fast paths" were the REFERENCE's (2026-09-21)

**What W3 wrote** (RESULT.md §3.5, the 200-token reading): *"the median planned progress on a zeroed
token is 32.07 m against 18.81 m … a planner that commits to long fast paths"*.
**What is true:** those are index `[0]` of the hook's `progress_raw_m = [PDM-Closed, agent]`
(`code/navsim_v1_win.py:199`) — the **reference** planner. refcv4b's own plan is **23.21 m vs
14.59 m**, and relative to the reference it is **shorter** on the failing tokens (0.750 vs 0.808).
The over-long-plan reading is **refuted**; the failure is the path's shape at speed (lateral).
**Class:** *an array index quoted without its semantics* — the tensor-without-units family. The
meaning of `[0]` was written on line 199 of my own wrapper and I did not reread it.
**What caught it:** generalising the one-off analysis into a script forced every field to be named,
and the name exposed the index. **Durable fix:** the full-split decomposition names both columns
(`agent_m`, `pdm_closed_m`) and never emits a bare `progress`.

## ⛔ W3's second self-correction — the launcher "waits" (2026-09-20)

**What W3 wrote** (in `RESULT.md`, and relayed to the orchestrator): the queued refcv4b arm is
*"LAUNCHED and WAITING … the launcher re-checks every 60 s for up to 8 h"*.
**What is true:** on `--device auto` W1's launcher takes **ONE probe** and returns `cpu`
(`gpu_gap.py:8-10, 230-256`). It never waited; the run was on CPU 1.2 s after the probe.
**Class:** *a true fact quoted outside its scope* — the 60 s / 8 h re-check is real, and belongs to
`--device cuda`. **How it survived:** the claim was written from the launcher's **docstring header**
(which describes the whole class) instead of its `acquire()` body, and nothing in my own output
contradicted it in words — `NO_GAP_CPU` is not the sentence "I did not wait".
**What caught it:** re-reading the live task output before reporting, and asking why a "waiting"
process was 3.4 cores busy with a model loaded.
**Durable fix:** the bridge now prints the real semantics at the decision point, and the manifest
carries `device_gate.launcher_semantics` with the `file:line`, so the next reader gets the scope
attached to the fact.

## ⛔ W3's own over-claim, corrected (2026-09-20)

**What W3 wrote:** *"the paper TRUNCATES … This is a PRIMARY measurement of [N1]'s convention."*
**What the evidence supports:** the CV **cell** was truncated. The paper's convention is
**UNSETTLED** and W3's own probe says so — three other informative cells read as ROUNDING
(TransFuser 83.9 vs 83.8822, Ego-MLP 66.4 vs 66.3989, LTF ±0.6 vs 0.552) and the one test internal
to the paper reads as ROUNDING too.

⇒ **W4's narrowing is correct and W3 adopts it**: publish the cell-level finding, keep the
paper-level verdict UNSETTLED, print the probe's verdict verbatim with both scopes named. The
measurement did not supersede the probe; it upgraded ONE cell from *print vs an INHERITED
leaderboard value* to *print vs a measured value*, and left the other three unexplained — they may
record a different RUN than the 2026-08-23 leaderboard rather than a different rule. ⚠️ The cell
conclusion also rests on a stated assumption (that the paper's CV row is the same quantity as
W3's), which is well supported but is not itself a measurement.

**Class:** *true but wrong for the reader* — a claim true of one cell, phrased as a claim about the
paper, which then propagated. RESULT.md §3.1 and `raw/print_convention_probe.json`
(`post_measurement_2026_09_20`) now carry both scopes; AMENDMENT A1 stays in force for every other
cell. ⭐ It was caught because W4 checked the phrasing against the ARTIFACT instead of accepting it
from a relay — the right reflex, and the reason the record is correct now rather than after
publication.

⚠️ **And W4's first probe for these artifacts came back EMPTY** because a `find … | head -25`
truncated before reaching them; the second, searching for the VALUES, found everything. *"0 hits"
is a claim about the search, not about the content* — the same family as W3's own `git rev-parse`
exit-128 ambiguity, where "not in HEAD" and "probe failed" were indistinguishable until separated
with `cat-file -e`.

## Hazards this stream hit (for the other streams)

* **The session scratchpad is SHARED between streams** (W2's finding, re-confirmed here: the
  scratchpad holds `apply_w2.py`, `blobs_devkit.tsv`, `add_row_windows.py`, … from at least three
  streams). W3 had written ONE generic name there (`idx_library.json`, used to superset-check the
  shared library index before staging). ⇒ **Re-run under `w3_*` names and re-verified**: the
  staged `library.json` blob `352dffcd…` equals W3's worktree copy, contains all 531 entries of
  the previous staged version (0 lost) and carries W3's citation of `2406.15349`; and all 69
  staged W3 paths re-verified index-blob == worktree-blob (68 VERIFIED + 1 that was a live edit,
  re-staged and verified). Every W3 verification now prints the package id and the path list it
  checked.
* **A sibling's CPU training smoke can evict every other job**: `refc_v3_train.py --device cpu
  --batch 4 --image-hw 416 1024` committed **33.9 GB** and drove system-available memory to
  **419 MB** (§D4).
* **An interface mismatch caught by wiring, not by reading**: W1's shared gate is
  `device_gate(requested: str, *, accept_training_box_load=…, queue_hint=…)`; W3's probe first
  called it with the args NAMESPACE and got `ValueError: device must be auto|cpu|cuda`. Fixed, and
  pinned by a test that asserts the call shape plus one that calls the LIVE gate for all three
  device values — the test that would have caught it.
* **Two heavy NavSim jobs now share this box**: E1's navhard N1 re-run (started 09:10 local) and
  W3's cache chain. Both are single-worker; W3 waits for ≥ 3.5–4 GB available before every step.

| D11 | **SPEC AMENDMENT A1: both printed-precision readings, because the paper's convention is UNSETTLED.** | E1 MEASURED that [N2] truncates; W3's probe on [N1] finds 3 ROUNDING / 1 TRUNCATION over the INHERITED leaderboard cells and ROUNDING on the one test internal to the paper. Amended at 2026-09-20T08:38:24Z with **0 navtest CSVs in raw/** — before the number, never after. ⛔ A cell is REPRODUCED only if BOTH readings match; one-sided agreement is labelled with its convention and the other reading printed beside it. |

## Proposed `GOALS_AND_CLAIMS.md` rows (NOT edited here — shared file, five live streams)

E1 and E2 both deferred this edit to the register owner; W3 does the same and leaves the rows
ready to paste. **Every row below is MEASURED** (updated 2026-09-21 — the STOP-floor row was withheld
until the full split landed, and it has). Tier **T1-family**, loop **OPEN**, background
**non-reactive (logged)** throughout; estimator W2's registered log-cluster bootstrap with the
drive-level sensitivity (`taniteval/adapters/navsim_ci.py`).

* **D-NAVSIM-W3-V11-LOADER** (MEASURED 2026-09-20): NAVSIM **v1.1** (`3e8291b`) carries the same
  Windows `MetricCacheLoader` defect E1 fixed in v2 — `dataloader.py:180` keys the cache by
  `cache_path.split("/")[-2]` while nuPlan writes `str(WindowsPath)`. On a real v1.1 navtest cache
  the unpatched loader raises `IndexError` and E1's separator-only replacement maps 20/20 tokens
  to their own paths. Evidence: `…/2026-09-19-navsim-v1-navtest/raw/C7_loader_smoke20.json`,
  `tests/test_v1_loader_patch.py` (5 tests, the defect exercised).
* **D-NAVSIM-W3-SUMMARY-ROW** (MEASURED 2026-09-20): the `PDMS_v1_navtest` CSV carries **exactly 1
  summary row, token `average`** — four independent devkit CSVs plus the line that writes it
  (`run_pdm_score.py:145`). It is the v1 counterpart of v2's three `extended_pdm_score_*` rows, and
  W1's `profiles.summary_row_shape` can be restamped DECLARED → MEASURED. Evidence:
  `raw/W1_summary_row_shape_navsim_v1.json`.
* **D-NAVSIM-W3-FRAMEBANK** (MEASURED 2026-09-20): the navtest frame bank (unique frames + index,
  one `.npy` per camera shard) reproduces E2's `build_frames.py` **bit-for-bit** on 20/20 scenes —
  per-scene sha256, the full `[4,256,640,3]` array, the `src` mask, rig key, observed fraction and
  mean pixel. Evidence: `raw/KB1_frame_bank_vs_e2.json`.
* **D-NAVSIM-W3-STOPFLOOR-V1** (MEASURED 2026-09-20, full split): on NavSim **v1** navtest
  (12,146 tokens / 136 logs) a do-nothing plan scores **STOP 61.8202** against **CV 20.6517** and
  HUMAN 94.5514; paired `STOP − CV` = **+0.4117**, log [+0.3950, +0.4284], separated at log and
  drive level. Mechanism: a stopped ego keeps NC·DAC·TTC ≈ 1 on a split filtered to scenes the
  human passes, and EP ≡ 1 wherever the best compliant progress is ≤ 5 m (`pdm_scorer.py:165-173`).
  ⇒ **Any navtest-v1 model claim must clear STOP, not CV.** Evidence: `raw/analysis_navtest.json`,
  RESULT §3.3.
* **D-NAVSIM-W3-CV-REPRO** (MEASURED 2026-09-20): the v1.1 harness **reproduces the paper's
  constant-velocity row** (arXiv 2406.15349 Tab. 1 p. 7; library key banked): NC/DAC/TTC/C/EP all
  reproduced at the printed digits; PDMS **20.6517** reproduces the printed 20.6 under TRUNCATION
  and the HF leaderboard's 20.6517 under ROUNDING. ⛔ This settles ONE cell, not the paper's
  convention (still UNSETTLED; AMENDMENT A1). BAR-W3-1 **PASS**. RESULT §3.1.
* **D-NAVSIM-W3-HUMAN-CLOSE** (MEASURED 2026-09-20): HUMAN **94.5514** vs published 94.8 —
  **CLOSE, not reproduced**; NC/DAC/TTC/C exact, the whole gap is EP (86.96 vs 87.5), the one term
  normalised against a computed (cached PDM-Closed) quantity. BAR-W3-2 **CLOSE**. RESULT §3.2.
* **D-NAVSIM-W3-REFCV4B-V1** (MEASURED 2026-09-21, full split): refcv4b-b1-v72-40k, arm
  `A1_ego_cmd` (frames + measured t0 ego state + NavSim driving_command), zero-shot:
  **PDMS 58.9738** (NC 91.35, DAC 73.46, EP 52.37, TTC 82.54, C 98.02). ⛔ **BAR-W3-M1 FAILS**
  (`> max(STOP, CV)`: 58.97 < 61.82; `A1 − STOP` −0.0285, log [−0.0497, −0.0077], drive
  [−0.0567, +0.0029] → not separated under the conjunction); ⛔ **BAR-W3-M2 FAILS** (< 65.6).
  Shape: wins **6,680/12,146** head-to-heads vs STOP (median +4.53) but is zeroed by NC·DAC on
  **3,828** tokens (DAC 3,224) vs STOP's 732; **on the 8,318 it does not zero, 86.11 vs 64.58**.
  **Beats STOP under every STRAIGHT command (63.96 vs 62.97, n = 8,070)**; loses under LEFT (47.24)
  and RIGHT (52.03). Evidence: `raw/BAR_W3_M1.json`, `raw/gate_decomposition_A1.json`,
  `raw/A1_by_driving_command.json`, `raw/artifact_A1_navtest.json` (criteria 0 violations).
* **H-NAVSIM-W3-JITTER** (REFUTED 2026-09-21, pre-registered SPEC §7b at 07:35:12Z): *"refcv4b's
  drivable-area failures are removable plan jitter."* A fixed, parameter-free anchored-cubic
  smoother (controls: seam path exact to float32; smoothed HUMAN −0.51 < 1.0) gives **A1S 58.1502**
  and **raises** DAC zeros 3,224 → 3,277 — pre-registered outcome 3. The lateral error is
  systematic: under a STRAIGHT command on curving road refcv4b turns **0.149×** the human's net
  heading; at LEFT commands it turns the right amount (1.07×) in the wrong place (DAC fail 39.8 %).
  Evidence: `raw/A1S_navtest/`, `raw/gate_decomposition_A1S.json`, `raw/turn_following_A1.json`.
* **D-NAVSIM-W3-SEAMPATH** (MEASURED 2026-09-21): the SEAM scoring path (a precomputed-pose agent,
  fingerprint-checked) reproduces the devkit's own HumanAgent on 200 navtest tokens × 6 columns
  to **6.7 × 10⁻⁸** (float32 seam storage) — the control that the model's 58.97 was scored by the
  same arithmetic as the reference arms. Evidence: `raw/HUMANseam_navtest/`.

## Cross-stream facts W3 measured that others may need

* **v1.1 is NON-REACTIVE**: logged actors replay their recorded futures (`docs/metrics.md`,
  `pdm_score.py` scores 2 proposals against the cached interpolated GT observation) — unlike v2,
  which is IDM-reactive in both stages (E2's correction). A "NavSim" loop stamp must say which.
* **v1.1 has no `pdm_score` column at all** (`PDMResults`): the column trap cannot fire here, but
  the headline is still `score`.
* **The v1.1 runner drops failed tokens from the mean** (`mean(skipna=True)`), so a count guard is
  mandatory — an agent that fails on hard scenes would otherwise be rewarded.
* navtest = **12,146 tokens / 136 logs** (v1.1 yaml, parsed and cross-checked against
  `yaml.safe_load`); all 136 log pickles are present on D:.
