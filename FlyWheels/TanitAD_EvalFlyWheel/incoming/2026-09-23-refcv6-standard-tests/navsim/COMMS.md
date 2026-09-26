# COMMS — refcv6 NavSim suite (EvalFlyWheel stream, run from the Master Mind session, 2026-09-23/24)

## ⭐ Escalations (each also in the report headline)

1. ⛔ **THE LIVE refcv6 RUN DOES NOT EQUALIZE THE BOTTOM ROWS IN ITS TRUNK — `--equalize-bottom-rows
   43` IS SILENTLY DROPPED.** `stack/scripts/refc_v3_train.py::_pin_trainer_cfg` sets
   `cfg.core.encoder.trunk_equalize_bottom_rows` as an AD-HOC attribute (`:381`), and 78 lines later
   (`:459`) rebuilds the encoder config with `dataclasses.replace(enc, image_size=…)` for `--image-hw`
   — which carries only DECLARED fields; `CNNEncoderConfig` declares no such field. MEASURED on the
   model rebuilt from the run's own argv through the trainer's own pin path: `TimmResNetTrunk.cfg.
   equalize_bottom_rows = 0`, `normalise()` zeroes nothing, `equalize_calls` 0. Both files are
   byte-identical at the run commit `287d72e` and at `fe5872f` (blobs `7cbe426a…`, `01c45b2d…`;
   positive control: `trunk_compile: bool` IS a declared field, `trunk_equalize_bottom_rows: int` is
   not). The live run's `train.log` has 0 "equaliz" lines of 138; its `config.json` says
   `"equalize_bottom_rows": 43` — the PERCEPTION stamp, read from `args`, not from the built trunk.
   The LIFT bank DOES get the 43-row mask (`train()` passes it from `args`, `:6829`).
   ⇒ The C26 mitigation (the rig-identifying black strip, 0.899 decodable on eval-139) is **not
   applied to the image the trunk sees** in refcv6-r101-s0, while the record says it is. This is the
   exact class the `CNNEncoderConfig` comment at `refc.py:350-355` warns about ("setting them as
   ad-hoc attributes LOOKED fine … a non-field does not survive the config's own round-trip").
   **Owner: Master Mind / Arch.** Fix = a declared field + a built-trunk stamp + a mutation test;
   the decision whether the live run continues is the PI's. This stream evaluates the model AS
   BUILT (trunk 0, lift 43) and says so on every row.
2. ⚠️ **`taniteval/tools/refcv3_arm.rebuild_perception_branch` builds its `LiftGeometryBank`
   WITHOUT `equalize_bottom_rows`**, while the trainer passes 43 — so an eval through `refcv3_arm`
   lifts the bottom 43 rows as OBSERVED where training marked them unobserved. Small, but it is a
   train/eval mismatch in exactly the rows C26 is about. **Owner: the sibling stream building the
   refcv6 loader / taniteval owner.** (This stream builds its NavSim lift with the trainer's mask.)
   ⭐ The battery stream found the same defect independently (its `refcv6_loader.py` item (1), its
   mutation `m1_no_equalize`) — two probes, two routes, one finding.
2b. ⛔ **`taniteval/tools/refcv3_arm.load_model` BUILDS refcv6 WITH THE WRONG ANCHOR UNITS.**
   `train()` reads the anchor artifact (`_read_anchor_artifact`, `:6726`) BEFORE `_pin_trainer_cfg`, so
   the file's DECLARED `control_units` ('alat') and derivation constants reach the decoder;
   `refcv3_arm.rebuild_config` pins without that read, and refcv6's argv has no
   `--anchor-control-units` (the file declares it) ⇒ `core.decoder.anchor_control_units = 'kappa'`: a
   lateral-ACCELERATION vocabulary rolled as CURVATURE (anchor bank off by up to 108 m; every plan
   changes). MEASURED by control KL against the battery stream's trainer-exact loader (3/3 selections
   differed → after replaying train()'s order: bit-identical, 3/3). **Any refcv6 (or other
   units-declaring-file) eval through `refcv3_arm.load_model` is affected** — this stream's own first
   step-1,000 runs were, and are quarantined (`raw/VOID_wrong_anchor_units/`). **Owner: taniteval /
   whoever maintains `refcv3_arm.py`** — the fix is one call before the pin, plus a refusal when the
   rebuilt units differ from the artifact's (`code/refcv6_bridge.load_refcv6` does both).
3. ⚠️ **The pushed tip `fe5872f` (ev6) lacks eval tooling that D:'s working tree has and this
   stream needed:** `taniteval/adapters/navsim_ci.py` (the settled NavSim estimator), the suite
   `taniteval/taniteval/bench/`, `taniteval/adapters/navsim.py`'s `log_names` clustering, and the
   code of the W3 navtest / W7 navhard / stage-1-extract packages. This stream imports them from
   D: by path. **Integration ask: land them on the branch** — a clean-tree gate would fail today.
4. ⚠️ **E2's `score_arm.py` navhard profile points at an INCOMPLETE metric cache**
   (`C:/Users/Admin/navsim/exp/metric_cache_navhard_two_stage`: per-log folders, NO metadata CSV →
   `IndexError` in the loader, MEASURED). The complete cache (5,912 rows) is
   `C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage` — the one W7 scored with.
   `code/score_arm6.py` uses it.
5. ⚠️ **Max-speed admissibility, surfaced not assumed.** The brief records the PI ruling of
   2026-09-19 as making nuPlan map speed limits admissible for refcv6's max-speed INPUT. The ruling's
   recorded text (memory `max-speed-labels-non-parity-ruling`) speaks of LABELS for a max-speed HEAD
   ("inference stays vision-only"). This stream follows the brief and carries `R6_VMAXOFF` (the
   channel withheld everywhere) so the input's contribution is measured either way. **PI to
   confirm.**
6. ⚠️ **refcv6's max-speed "unknown" input is out of distribution.** The run's train sidecar is
   4,572/4,572 valid, so the model never saw the all-zero one-hot it gets wherever the map has no
   limit (BOS 95 %, SG 100 % of scenes). Stated per split with the coverage.
7. **Compute on this box, 2026-09-23/24 night:** the GPU gate stayed CLOSED (first another session's
   `augment_search.py` shards, then the sibling's in-run-eval reproduction held the card), and the
   box hit a MEMORY EMERGENCY (commit 68.2 of 71.0 GB, free RAM 0.2–1.3 GB) from other sessions'
   REFe jobs + two large smokes. E1's RAM guard correctly aborted a navhard CV scoring at 322/450;
   this stream paused its own navtest bank build and RAM-gated its navhard queue. Everything ran on
   CPU fp32 (1.5–1.7 s/scene with the exact dedup).

8. ⛔ **THE MAX-SPEED CHANNEL WAS TRAINED ON AN EGO-FUTURE ORACLE; NAVSIM CAN ONLY GIVE IT A MAP
   LIMIT — AND ON NAVTEST THE TWO AGREE ON 25 % OF TOKENS.** The run's own `config.json`:
   `max_speed_onehot_v6.provenance = "ego-future (oracle INPUT, ~1.7555 bits)"`; the training value
   is the max of the ego's OWN realised speed over [t0+2, t0+6] s, in a CONTAINING window — so the
   one-hot told the model a lower bound as well as an upper one. `v7_labels.oracle_max_speed` names
   it a *"TRAIN/DEPLOY MISMATCH"*; `refcv6_max_speed.py` records the PI's authorisation *"as a
   declared oracle INPUT standing in for a map/nav set-speed service"*. MEASURED on navtest by this
   stream (label-free, `raw/inputs/vmax_oracle_navtest.json`, SPEC §12): the training definition on
   the 2 Hz log gives bins **30: 75.74 % · 50: 23.50 % · 100: 0.75 %** (training: 38.08 / 34.84 /
   22.18 / 4.90 %); where a map limit exists (6,623 tokens) the map bin is **ABOVE** the oracle's on
   **52.45 %**, equal on 45.81 %; **5,461** tokens have no limit at all (the all-zero row, never seen
   in training). ⇒ only **3,034 / 12,146 = 25.0 %** of navtest tokens receive an input with the
   semantics the model was trained on. SPEC §12 prices it with a PRIVILEGED diagnostic arm
   (`R6_VMAXORACLE`, never a result). **Owner: Master Mind / Arch** (a deployable set-speed source,
   or training on the map-limit semantics, is a model decision).
9. ⛔ **TWO OF refcv6's THREE DECLARED SELECTION SEAMS ARE NOT BUILT IN THE LIVE RUN — ITS OWN
   `config.json` DECLARES ALL THREE.** `config.json` `selection_inputs` lists "nav compliance
   (PARAMETER-FREE geometric predicate, one zero-init gate)" and "max-speed ceiling (ARGMAX FILTER,
   no parameters)"; `refcv6_selection.py`'s docstring: *"Three things reach the ranked score in
   refcv6"*. MEASURED on the model rebuilt from the run's argv through the trainer's pin path:
   `decoder.speed_ceiling_filter = False`, `cfg.core.graft_nav_compliance = False` (only
   `graft_behaviour_sel = True`). No trainer flag sets either (`refc_v3_train.py` has no
   `speed_ceiling` / `nav_compliance` argument; only `tests/test_refcv6_tactical.py` sets the
   filter); the `selection_inputs` list is a STATIC string emitted whenever v6 is on
   (`refc_v3.py:1258-1268`). Behaviourally, at step 1,000: **the max-speed input changes NOTHING in
   the plan — 204/204 warmup plans bit-identical between R6_A1 and R6_VMAXOFF (126 of them with a
   map limit), and 3/3 navtest smoke scenes bit-identical across map / oracle / withheld**
   (`raw/controls/plan_deltas_warmup_s1000.json`); the one-hot reaches the tactical logits (KI) but
   no selection flips. Same class as escalation 1 (a record that says a lever is on while the built
   model does not have it). ⭐ The ceiling filter is parameter-free and acts on the argmax only, so it
   could be switched on AT EVAL without retraining — **a PI/Arch decision, not taken here** (it would
   evaluate a model other than the one trained). **Owner: Master Mind / Arch.**

10. ⚠️ **THE SIBLING BATTERY IS SELF-DEADLOCKED ON ITS OWN GPU GATE (sent to the Master Mind
   at 01:41Z via SendMessage, read-only diagnosis, nothing touched).** `ev6_battery/code/run_battery.py
   ::gate_wait()` calls `gpu_gate.gate()` without `self_pid`, so its own process (PID 30624, which
   still holds a 1,830 MiB CUDA context) is counted as "another python on the card" and the gate
   can never pass (`battery_step1000.log`: "GPU gate WAIT … python_compute: [[30624 …]]" every
   60 s from ~01:39Z; it gives up after 12 h). It also keeps THIS stream's gate closed. Fix (theirs):
   `gate(self_pid=os.getpid())` — `gpu_gate.gate` already takes the argument — or release the CUDA
   context before waiting. Same family as the `pgrep -f` self-match trap in CLAUDE.md.

## Decisions made in this stream (reversible)

| # | decision | why |
|---|---|---|
| D1 | Evaluate refcv6 AS BUILT: trunk equalization 0, lift mask 43 | escalation 1 — an eval that "fixed" the trunk would score a model that was never trained |
| D2 | ST (static t0 history) primary, NT a warmup sensitivity | E2's measured ST ≥ NT for refcv4b; NT needs 3 stored frames |
| D3 | CPU → fp32 trunk, CUDA → as-trained bf16+NHWC; one device per checkpoint | bf16 is emulated on this CPU (98 s vs 9 s/scene MEASURED) |
| D4 | exact-duplicate dedup (eval-only) on every refcv6 arm | MEASURED bit-identical to the trunk's native path (KD, 4/4, max \|Δ\| 0.0); 5× cheaper on a static window |
| D5 | per-scene inference seed `sha256("<base>:<token>")` (common random numbers) | the sampler is stochastic at eval; paired arms then share each scene's eps |
| D6 | road plane = the median vehicle-cuboid bottom over 74 logs (−0.3557 m), one vehicle constant | NavSim's ego origin is at axle height; per-log medians range −0.50…−0.15 (terrain + annotation noise). Second probe on the split it is also applied to: a 21-log navtest sample (every 7th log) reads median **−0.3428 m** (−0.484 … −0.153), **1.3 cm** from the constant (`raw/inputs/road_plane_navtest_logs_sample21.json`) |
| D7 | model-free floors reused across checkpoints (warmup: this stream's reproduced CSVs; navhard: W7's banked run) | they read nothing of refcv6; the harness reproduction is the admissibility evidence |
| D8 | SPEC amendment A1: a history with a DROPPED 2 Hz frame is interpolated at its actual times (guards: t0 = 0, increasing, no extrapolation, gaps ≤ 1.0 s) | the navhard bridge REFUSED stage-1 token #243 (times −2.0/−1.5/−0.5/0); census: warmup 0/220, navhard 2/5,912, navtest 15/12,146; recorded before any navhard/navtest score (`raw/SPEC_PREREG_HASH.txt`) |
| D9 | the loader is `refcv3_arm.load_model` + this package's own NavSim lift, not the battery stream's `refcv6_loader.py` | the brief's rule (reuse the sibling's loader if it exists and reproduces the in-run eval) — it lives outside the repo (`C:/Users/Admin/ev6_battery/`) and was mid-verification; `tests/test_loader_crosscheck.py` (KL) compares the two on the NavSim forward instead of assuming equivalence |
| D10 | SPEC amendment A3: the device/precision unit is the SPLIT (all arms of a split, incl. the seed replicate, on one device), and cross-checkpoint reads with differing devices are held against a new control **KP** (CPU fp32 vs CUDA bf16, same checkpoint + seeds, warmup R6_A1; `code/run_kp.sh`, armed on the GPU gate) | §2 said "per checkpoint" while the runner decides per split; on a shared card a per-checkpoint rule idles a milestone or forces it all onto CPU. Recorded before any corrected refcv6 score (`raw/SPEC_PREREG_HASH.txt`) |
| D11 | the step-1,000 navhard bridge was PAUSED at 138 corrected rows (23:43Z) and resumes after the warmup bridge, RAM-gated, at BELOW-NORMAL priority (`code/run_navhard_resume.sh`) | the box hit 1.7 GB free and E1's guard aborted the KH-nav CV scoring; step-1,000 navhard is pipeline validation that any milestone supersedes, so it yields CPU and RAM to KH-nav, warmup, and the milestones |
| D12 | the SPEC §5 FAIL ladder is AUTOMATIC: the runner runs `code/decompose6.py` after every parse — W7's `decompose.py` imported for the two-stage splits (zero attribution with the unique-zero control, the single-term counterfactual asserted against the devkit's `score`, speed bands, per log) + a per-COMMAND split; navtest per command / speed band on W3's floors | cross-check: on W3's refcv4b navtest CSV it reproduces W3's banked `A1_by_driving_command.json` exactly (LEFT 2,501 / 47.2448 vs STOP 58.7779; STRAIGHT 8,070 / 63.9642 vs 62.9692; RIGHT 1,575 / 52.0292 vs 60.7642) |
| D13 | Master Mind 01:52Z relayed the battery fix and ruled: keep the GPU gate as written (RAM >= 8 GB too), keep RAM headroom, keep running on CPU until the gate opens, state the device on every score | done: `parse6.py` / `parse_navtest6.py` now read each arm's device + precision from its seam manifest into the summary, `report6.py` prints it as a column; the runner's CPU RAM gate raised 5 -> 6 GB so a freshly built model (~2.0-2.4 GB) leaves E1's 3 GB sustained floor intact |
| D14 | 2026-09-26 (Master Mind, PI: commits and pushes are COORDINATED): this stream never runs git — finished files are copied to the D: package, listed in `LANDING_READY.txt`, and the Master Mind lands them. `code/stage_to_repo.py` rewritten copy-only (content-verified, `--reconcile`, `--landing` gate: ≤ 20 MB, no canonical UUIDs). ev6 ↔ D: reconciled by content (415/425 identical; the 10 others = today's edits + a run in progress); one stale VOID duplicate on D: moved into `raw/VOID_wrong_anchor_units/` (not deleted) | the shared D: index holds ~2,800 stale entries; concurrent committers have reverted each other |
| D15 | runner fixes after reviewing the unattended step-5,000 run: (a) a refused CUDA bridge that wrote no row re-runs on CPU (warmup@5000 had been scored on NOTHING); (b) an explicit `--device cuda` waits for the brief's gate before any CUDA work; (c) K0 and KD are parsed separately (`pytest -rA`) and a K0 (determinism) failure refuses CUDA for the split | MEASURED on 09-24: runner gate open at 04:11Z, bridge gate closed 900 s later, 0 seams |
| D16 | `--fetch-final` + `milestone_waiter.sh final`: the FINAL `ckpt.pt` only after `summary.json` reads done: true; md5 on Thor before AND after the copy and on the dev box must agree | the Master Mind's rule for the final model |
| D17 | the precision controls run in their own lane (`code/kp_lane.sh`), gated on the GPU gate, never ahead of a result; KP-navtest (step 5,000, 200 tokens, CUDA vs the banked CPU rows) added as a DIAGNOSTIC | navtest@5000 ran on CPU, navhard@5000 on CUDA: cross-checkpoint reads may mix devices (A3) |
| D18 | the navtest diagnostic (200 tokens) now carries R6_A1_s1 — the navtest inference-seed floor (SPEC §4: "inference variance is read from R6_A1_s1"); step 5,000: +1.26 PDMS [−3.17, +5.78] | the SPEC's navtest arm list had no seed replicate |
| D19 | KH-navtest: `score_navtest6.py --official STOP` scores W3's model-free STOP through this package's driver; REPRODUCED W3's CSV cell for cell (12,146 × 8, |Δ| 0.0) | a split's result is admissible only beside a harness control on that split |
| D20 | every unattended output was REVIEWED before it entered RESULT.md (2026-09-26): count guards, stand-ins, device/precision stamps, seed replicate, harness control per split — all PASS for navhard@5000 and navtest@5000; warmup@5000 was EMPTY (D15a) and is re-run | Master Mind: "treat every unattended output as UNVERIFIED until you have checked it" |
