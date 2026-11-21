# LEADSUP_DEFS: what was built for the supervised lead-vehicle head LEADSUP (2026-11-15)

Base: `C:/Users/Admin/v8proof_c/target/tree_patch` (the TARGET result), copied to `C:/Users/Admin/v8proof_c/leadsup/tree_base` and `.../tree_patch`; the nine sha256 of `spatial/target/PINS.txt` were verified on the copy before any edit. CPU only, nothing trained, no git, no pod.
Paths relative to `stack/`. Tests: `tests/test_v8f_lead_sup.py`. Read hook: `scripts/lead_dump.py` (new, outside the pinned set).

## Where the TRAIN lead labels come from (the question of step 1)

* The label join `join_train.npz` (the chain passes `data/inputs/join_train.npz`; `--join-train` of the trainer, loaded by `build_data` in `v8f_mini_train.py`) holds, per cache row `off[c] + j` (cache step `j` of clip `c`, 0.2 s apart = raw frame `2 j`),
  `lead_valid` (int8, from the v9 release rows) and `lead_gap` (float32 [m], NaN where there is no lead or no lead data). `lead_gap` is filled by `build_label_join` (`v8f_mini_data.py:149-223`) from the lead file `--lead-train` (`lead_train.npz`: keys `sha12`, `k`, `lead`, `gap`).
  A join built without that file carries `lead_gap` NaN everywhere.
* The training batch ALREADY carried `lead_gap_h` (the lead gap of the next 30 steps, for the consistency penalty) and the tick-frame `lab*["lead_valid"]`, but NOT a per-frame lead gap for the frames of the window, and not presence per frame. So a new batch key was needed (decision 1).
* The probe join of the earlier stages (`C:/Users/Admin/lorapod/ship2/join_train.npz`, S0E E2/E3, S1) is the same object (arrays `clip`, `off`, `n`, `lead_valid`, `lead_gap`); the rule "presence = lead_valid; gap = lead_gap where valid, finite, 0 <= gap <= 60 m" is S0E E3.

## Flags (every default = the base tree, bit for bit)

| flag | default | meaning |
|---|---|---|
| `--lead-sup {off,on}` | off | builds the head; the term is computed only when on |
| `--w-lead-sup <float>` | 0.0 | weight of `lead_sup` in the total; the config AND the gate refuse `on` with 0.0, and a weight with `off` |
| `--lead-hist <int>` | 5 | consecutive frames (ending at t) whose level-1 states the head reads; refused with `off` unless 5 |
| `--lead-sup-detach {off,on}` | off | on: the head reads `z1.detach()` (needs `lead-sup on`) |

Config fields `lead_sup, w_lead_sup, lead_hist, lead_sup_detach` (`v8f_mini.py`), `LEADSUP_CFG_DEFAULTS`: written into a checkpoint cfg only when they differ from the default, supplied as defaults when absent (`INIT_CKPT_MATCH` has all four), so every earlier checkpoint loads unchanged and a lead-sup checkpoint carries the fields the eval builder needs.
Needs the WM auxiliary with `wm_state grid` (config + gate refusal). Not built for `--encoder lora` (gate refusal: the labels ride the cached-token dataset).

## The head (`LeadSupHead`, `v8f_mini.py`)

* Built in `WMAux.__init__` as its LAST module, `wm.lead_head`, only when on (no module and no state-dict key otherwise). Because it is built last in the WM, every tensor of the model keeps G's init at the same seed (the head arm = G + the head keys, checked bit for bit by `l8` of `checks/t3_checks.py`). Heads built after the WM in `V8FMini` (level, lead, stop heads) would draw after it; none is on in these arms.
* Input: the level-1 grid states `z1 = wm.z1(adapt)` [B, F, 640], the SAME tensor the stack's loss uses (no second projection pass), of the frames the WM auxiliary ALREADY encodes in a window. The window sampling is unchanged. A window has `hmax` = 16 frames; the head reads `lead_hist` consecutive frames and is applied at EVERY end frame `t = hist - 1 .. F - 1` (12 end frames at hist 5), so each window gives up to 12 supervised readings.
* Per-frame `Linear(640, 64)`, the `hist` embeddings concatenated (hist x 64 = 320), `Linear(320, 128)`, GELU, `Linear(128, 3)`. **Added parameters at hist 5: 41,024 + 41,088 + 387 = 82,499** (640x64+64, 320x128+128, 128x3+3); at hist 3: 41,024 + 24,704 + 387 = 66,115 (arithmetic; hist 5 is read from the built model in `checks/t3_checks.json` l8).
* Outputs (float32): presence logit, gap / 50 m, closing speed / 10 m/s. Inputs are the states only: no ego, nav, limit, code or label enters the head; the labels enter the loss only (the head is a train-time auxiliary; `forward_infer`, the planner and `v8f_t1_eval.py` never call it).
* End frames with fewer than `hist` real frames behind them (clip start, `frame_ok` False) are not counted (decision 3). `hist` <= `hmax` is required (config refusal).
* `detach on`: `res["lead_out"] = head(z1.detach())`; the term then reaches no parameter outside the head, EXACTLY 0.0 (the zero control and the comparison arm). `detach off`: the gradient flows through `wm.proj` and the adapter (and the layer norm / pos of the adapter path).

## Labels (privileged, training only), `lead_sup_label_arrays(J)` in `v8f_mini_data.py`

Computed ONCE from the join arrays when the dataset is made with `lead_sup=True` (the trainer does it from `--lead-sup on`); no extra label file. Per join row:
* `pres` = 1.0 where `lead_valid == 1`, else 0.0 (every row has a presence label; rows outside the release read 0).
* `gap` [m] = `lead_gap` where `lead_valid == 1`, finite, `0 <= gap <= 60` (S0E E3 rule), else NaN.
* `clos` [m/s] = least-squares slope of the gap over the trailing run of labelled frames ending at the row (0.2 s apart), at most 5 frames, a 0.2 s step above 3.0 m breaks the run (the run never crosses a clip), at least 3 frames, else NaN. This is the `hist_run` / `ls_slope` rule of `spatial/objslot/objslot_s0.py` (H_MIN 3; the slope of the GAP: positive = the gap opens, negative = closing in), checked against a literal re-implementation on random data and on planted series (`l1`). The `ls_slope` minimum of 2 frames is raised to 3 by H_MIN, as in OBJSLOT D5.
* A sha256 over the three arrays (bytes, float32, in the order pres, gap, clos) and the counts are recorded: `info["lead_sup_labels"]` (selfcheck.json), checked present by the gate.
* The item adds `lead_pres`, `lead_gapm`, `lead_clos` [F] (rows `off + idc`, the frames of the window); the collate adds them as float32 only when present. The default dataset's item and batch are the base tree's, key for key.

## The term (`lead_sup_terms`, `v8f_mini.py`)

`lead_sup` = BCE(presence logit, `lead_pres`) + Huber(gap output, gap / 50 m; delta 0.1 = 5 m) + Huber(closing output, closing / 10 m/s; delta 0.2 = 2 m/s). Each part is the mean over its own mask: BCE over every counted end frame, the gap Huber over counted end frames with a finite gap label, the closing Huber over counted end frames with a finite closing label. Added to the total by ONE line: `total = total + cfg.w_lead_sup * T["lead_sup"] if cfg.lead_sup == "on" else total`. The WM term `T["wm"]` and the stack loss are untouched (the head's term is outside `T["wm"]`, so `w_wm` does not scale it).
Logged: `lead_sup` (live), `lead_sup_bce`, `lead_sup_gap`, `lead_sup_clos`, the counts `lead_sup_n_pres/_n_gap/_n_clos` (detached). G-LIVE requires the four value keys when on.

## Gate / selfcheck

* `bad_lead_sup(a, cfg, model, data_info)` in `v8f_mini_train.py`, called from `declared_vs_built`: declared on that builds nothing, built but not declared, weight 0 with on, weight / hist / detach with off, hist / detach / weight read back from the BUILT head not equal to the declaration (hist from the read-out layer's input width), non-grid state, lora encoder, a dataset that carries no labels (`info["ds_lead_sup"]`), an unrecorded label hash, a label set with no presence, gap or closing rows.
* `built_levers()` reports `lead_sup, lead_hist, lead_sup_detach, w_lead_sup` (`lead_sup_detach` reads the wiring flag of the built `WMAux`; its behaviour is pinned by the exact-zero control, not by the flag alone).
* Selfcheck strings (`gate_extra.py`, whole floats as ints): `levers.lead_sup`, `levers.w_lead_sup` (`1`, not `1.0`), `levers.lead_hist`, `levers.lead_sup_detach`, and `info.lead_sup_n_labelled` (frames the G-LIVE probe batch counts with a finite gap label, in the loss's own counting: all `hist` frames behind the end frame real). Also in `info`: `lead_sup_n_presence`, `lead_sup_n_closing`, `ds_lead_sup`, `lead_sup_labels` (sha256 and counts).
* The synthetic world carries no lead track (release empty, no lead file): `synthetic_lead_labels(J, seed)` fills one, and ONLY the trainer's `--synthetic` path with `--lead-sup on` calls it. Nothing else of the synthetic world changes.

## Decisions the card did not fix (for the lead to review)

1. **Labels in the data path, not a stage after the collate.** The earlier `--lead-head` looks its label up by `(ci, t)` AFTER the collate (`lead_stage`), one value per window. This head needs a label per FRAME of the window (every end frame), and a lookup by `(ci, t)` cannot give it. So the dataset item carries `lead_pres / lead_gapm / lead_clos` for its 16 frames, computed from the join with no extra file (the join already holds `lead_valid` and `lead_gap`; the closing rule needs only the gap series). Consequence: under the DAgger / ego-perturb / recover stages (not used by these arms) the labels stay those of the logged frames, like `wm_speed`.
2. **Presence label = `lead_valid`** (S0E E3 / S1), not "a lead within 40 m" of the earlier `--lead-head`. Gap labels cover only 0 to 60 m. So presence can be 1 on a frame whose gap label is missing (a lead with no lead-file row): the BCE counts it, the gap Huber does not. The earlier head's labels (`has / near`, 40 m) are not used here.
3. **Every end frame of the window is supervised**, not only the tick frame, and only where all `hist` frames behind it are real (`frame_ok`): 12 end frames at hist 5 and hmax 16. This is the card's "the frames the WM auxiliary already encodes". The frames of a clip start (t < 15) are padded and masked.
4. **Huber deltas** (0.1 in gap / 50 = 5 m; 0.2 in closing / 10 = 2 m/s) are the builder's choice, module constants `LEAD_SUP_HUBER_GAP / _CLOS`, mirroring `LEAD_HUBER_DELTA` of the earlier head.
5. **The three parts are summed with weight 1 each** inside `lead_sup`; no per-part weight (the card gives one weight).
6. **Sign of the closing speed**: slope of the gap, so negative = the gap shrinks (the same sign as `ls_slope` / `c` of OBJSLOT D6, lead speed minus ego speed).
7. **The head sees states from the GRID of the arm only** (`wm_state grid`, 640 values per frame); a mean-state arm is refused.
8. **`lead_sup_detach` readback** is the stored wiring flag, not a probe of the built behaviour; the behaviour is covered by tests (`l4`, mutation `mut_detach_removed` goes red).
9. **RNG**: the head is built last in the WM; with the arms' other heads off, a lead arm has G's init plus the head (the head's own draws come after every G draw).
10. **Time stride of the labels**: the join row step is 0.2 s (raw frame 2 j, 10 Hz source), so the slope uses `DT = 0.2`.

## Not built

* No eval-side use of the head (`v8f_t1_eval.py` is byte-identical and unaware of it; its model rebuild takes the head from the checkpoint cfg). The read on the pod is `lead_dump.py`, run later.
* `--lead-sup on` with `--encoder lora` (refused by the gate); a head on the mean state (refused).
