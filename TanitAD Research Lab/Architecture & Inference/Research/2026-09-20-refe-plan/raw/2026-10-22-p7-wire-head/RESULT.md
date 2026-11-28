# P7-WIRE-HEAD — the drivable-area head in the trunk trainer, with launch-gate coverage

Package: `C:/Users/Admin/par18/p7wire_head/` (the only place anything lives). Tip used: `1664143ba9bfb9ad45aeb095a31a69dd6c5b7c30` (source-of-truth archive). **Nothing was staged, committed or pushed anywhere** (local override of the brief). 0 GPU, no pod, no network write; the dev box's `D:/Projects/TanitAD/data/refe_consistent` and `D:/Projects/TanitAD-artifacts/aux_drivable/label_store` were READ, never written.

## ⭐ Headline — read before landing

1. **`refe/train.py` is edited by two deliverables (this one and P7-WIRE-RESERVE's `--reserve-manifest`).** Merge, do not overwrite. Every changed hunk of mine mentions the head and the test pins that (`S6: every changed hunk ... mentions the head`), so the hunks are local; the unified diff is `code/P7_WIRE_HEAD.diff` (against the tip). The lander must build on the live tip with `base=<blob>`.
2. **`refe_p7` REFUSES today, on purpose.** Three pins do not exist until the reserve filter has run, and are left UNREGISTERED in `refe/launch_gate.py` (never defaulted): `P7_AUX_STORE_PINS` (the training store's manifest sha256 + key count), `P7_BANK_TUPLES` (the `train_grow_cmd_3c_p7` tuple count). WIRE-RESERVE (or the PI) fills them from the filter's report **before step 0**; each is refused by `p7_refusals` AND independently FAILS in the check that consumes it (G-DVB store row, G-CLOCK bank row). `P7_RESERVE_SHA256` is registered: `53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29` (R-A, `reserve_sessions.json`, MEASURED by re-hashing the LF form).
3. **WIRE-RESERVE must also add** `--reserve-manifest` to `refe_p7`'s `data_flags` and the draw's own suite (`ZZRESERVE_PASS 17`) to its `suite`. I did not touch `p6_pod.sh`, `p6_prep.py`, `session_reserve.py`.
4. **G-RESERVE's adapter is UNVERIFIED against the real `session_reserve.audit_inputs`.** The brief gave the signature `(reserve, *, bank_files, metric_cache_dir, label_store, image_index, calib_rows) -> dict with ok`; the ARGUMENT TYPES were not specified. `reserve_audit_inputs(argv)` in `gate_checks.py` is the ONE place that maps the launch argv to them (paths as strings; bank files as a sorted list of `targets_rank*.jsonl` paths; `reserve` = the parsed JSON). Tested with fakes only; when `session_reserve.py` lands, run one real call and adjust that function. Any signature drift FAILS (tested), never passes.
5. **One existing test was edited**: `refe/test_launch_gate.py` pin `len(dests) == 61` → `68` (+7 head dests). Its assertion count is unchanged (`ZZLG_STATIC_PASS 1168` before and after).
6. **Not implemented (SPEC_P7 §1 lists them as "listed, not implemented")**: the `w_aux` gradient-norm band 0.05–0.5 (`aux_grad_ratio`), the label-alignment gate (≥ 95 % on ≥ 64 frames, flip-y / off-by-one mutations), G-LIVE-MEM's "head ≤ 1 GB". The head's activation memory at `--batch 4` on the A40 is **UNVERIFIED** (the P6 deterministic-kernel bound was already 43.70 of 44.43 GiB): the pod's `--gate-mem-metrics` run is where it is measured.

## What was built

| file (repo path under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`) | change |
|---|---|
| `refe/train.py` | the head wired behind `--aux-drv/--w-aux-drv/--aux-drv-tap/--aux-drv-depth/--aux-drv-feed/--aux-drv-labels/--aux-drv-shuffle-seed` (68 parser dests, was 61); head is a SIBLING (own seeded init stream), H1 only (feed `on` refused); P6 behaviour with the flag absent is bit-identical (F1 below) |
| `refe/gate_checks.py` | `DVB_KINDS` +7; `aux_dvb_rows`; recorders (`RecAux`, tap observers, Adam-state post hook); `aux_reach` / `judge_aux_reach` / `judge_aux_live`; G-EVAL + G-CKPT head additions; `check_reserve` + `reserve_audit_inputs` |
| `refe/launch_gate.py` | `CHECKS` + `G-RESERVE`; `PROFILES["refe_p7"]` (clone of `refe_p6`, which is NOT edited); `p7_refusals`; `--gate-reserve-json`; `gate:reserve_json` bound into the token |
| `refe/test_p7_head_wiring.py` (NEW) | static block (what G-SUITE runs) + `--full` + `--mutations` |
| `refe/fixtures/train_pre_p7_head.py.txt` (NEW) | verbatim tip `train.py` (sha256 `6b425ac7…ccd4b3`), the oracle of the default-off bit-identity |
| `refe/test_launch_gate.py` | one pin, 61 → 68 |

`patches/` holds the first-pass patch scripts (provenance only; later edits were made directly: **`code/P7_WIRE_HEAD.diff` is authoritative**).

## Hop trace: flag → loss (file:line in the FINAL `refe/train.py`)

| hop | where | proved by |
|---|---|---|
| argparse registers the 7 flags (all default OFF: `--w-aux-drv 0.0`) | `AUX.add_aux_args(ap)` `train.py:2191` | S3 `EVERY head default builds NOTHING`; census `parser registers…` |
| trainer's own refusals (weight > 0, labels, feed on) | `train.py:1205-1210` | S3 (grep hits the implementation, not only the test); M2b arm |
| built ONLY under `--aux-drv`, from the parsed levers | `train.py:1285-1287` (`AUX.AuxDrivable(AUX.AuxDrivableConfig.from_args(a))`) | census; M5 arm RED |
| tap = scene_proj's 2nd output (`visual_ctx`) | hook `train.py:1288`; live for one forward `1697`; two outputs or RAISE `1709`; `aux.predict(taps[1 if visual else 0])` `1711` | G-DVB row `aux_drv_tap` (independent hooks, data_ptr); M4 arm RED |
| labels: `(sdf int8 [160,128], valid)` as the LAST two batch elements | `TargetBank.aux` `671-676`, coverage `833`, `__getitem__` tail `950`; unpack `1690` | F2; coverage row ≥ 99.9 % |
| loss, masked per frame | `aux.loss(...)` `train.py:1724` | G-DVB `valid mask` row |
| **backward SITE 1** (online deferred) | `tot_ = aux.combine(tot_, P["l_aux"])` `train.py:1862` | census + SITE-1 source mutation RED (the online path needs NAVSIM/shapely: not runnable on this box) |
| **backward SITE 2** (plain) | `loss = aux.combine(loss, l_aux)` `train.py:1880` | M1 arm RED on the REAL gate |
| optimiser + clip | `params += aux_params` `1423`; clip `1886` | M3 arm RED (G-LIVE: 43 of 43 head tensors not held) |
| logged terms | `aux_bce, aux_huber, aux_L, aux_w` `train.py:1915`; `aux` event `1670` | G-LIVE `live_terms`; G-DVB store rows |
| checkpoints: beside `model`, never inside | `_aux_ckpt` `1640`; `ckpt_last` `1651`, snapshot `1967`, `model_final` `2019`; resume `1538` | G-EVAL (planner strict load unchanged), G-CKPT (bit-identical round trip) |
| identity (resume refusal) | `IDENTITY_ARGS += AUX.AUX_IDENTITY_ARGS` `1040`; defaults `1047`; store digest `1098` | F1: a pre-P7 checkpoint's identity resumes (no diff) |

## Launch-gate coverage

**Profile `refe_p7`** = `refe_p6` + (required, registered values) `--aux-drv`, `--w-aux-drv 0.1`, `--aux-drv-tap visual`, `--aux-drv-depth 2`, `--aux-drv-feed off`, `--aux-drv-labels <store>`, `--w-field 0.0` named; `--aux-drv-shuffle-seed` FORBIDDEN; `--aux-drv-labels` a DATA flag (the PASS token fingerprints the store); live terms `+aux_bce, aux_huber`; check `G-RESERVE`; schedule 25 × ⌈92,896/256⌉ = **9,075** steps, `epoch_len` 92,896 (SPEC_P7 §2, PUBLISHED-in-spec arithmetic, re-derived). `refe_p6`'s profile dict digest is unchanged (`80fa4d7e…`, MEASURED before and after).

**G-DVB**, one entry per new lever (`aux_dvb_rows`; every BUILT side is what the loader / head / loss actually received, never the namespace):

| lever | declared = built (observed how) | mutation that turns it RED |
|---|---|---|
| `aux_drv` | an `AuxDrivable` was constructed iff the flag (recording subclass) | M5 flag on, head not built |
| `w_aux_drv` 0.1 | multiplier **at the graph**: a hook on L_aux's own output reads d(loss)/d(L_aux) × accum on every micro-batch; + `aux` event + logged `aux_w` | M1 delete the combine line; `aux_weight_off`, `aux_term_detached`; flag omitted (M2) |
| `aux_drv_tap` | the head's input storage IS scene_proj's 2nd output, which IS the scoring decoder's context | M4 tap scene |
| `aux_drv_depth` | CrossBlocks the built head holds | M6 depth 1 |
| `aux_drv_feed` off | built head holds no feed; the REAL scorer's `score_q_mlp[0].in_features` == 60 (read off the model); no autograd path head → score (G-LIVE) | fixture mutations (width 63) |
| `aux_drv_labels` (data) | TargetBank received it; the profile lists it in `data_flags`; the `aux` event's store sha256 + key count == the profile's REGISTERED pins (no fallback to the full store); coverage ≥ 99.9 % | wrong sha / key count (the full store, reserve keys present) / 99.75 % coverage / unregistered pins: all RED in S5 |
| `aux_drv_shuffle_seed` | TargetBank + `aux` event | forbidden in the profile |
| `w_field` 0.0 | already in the registry; the profile now REQUIRES it named `0.0` | `--w-field 0.1` refused (S3) |

**G-LIVE** (`aux_live_facts`, `judge_aux_live`, `aux_reach`, `judge_aux_reach`): head exists, every tensor requires grad, finite gradient on every optimiser step and non-zero on at least one, **held by the optimiser and holding Adam state after the step** (the `probe_zerograd3` pattern: a registered parameter without state never trained); `aux_bce`, `aux_huber` finite on EVERY logged row, first `aux_bce` = ln 2 ± 0.15, `aux_w` == declared. Reach (an L_aux-only backward on the first real labelled sample): **non-zero** on the head, LoRA, `scene_proj`, `pos3d_mlp`; **EXACTLY 0.0** on `traj_head`, `reg_compress`, `registers`, `queries`, `dec`, `ego_enc`, `cmd_embed` and the scorer; the **weight-0 control reads EXACTLY 0.0 on every group**; no autograd path score → head.
**G-EVAL**: head beside `model` in `model_final.pt`, none inside, planner load bit-identical. **G-CKPT**: head weights + Adam moments round-trip bit for bit through halt + resume (`resume_drops_aux` arm). **G-RESERVE**: reserve file sha256 == the registered draw, then `session_reserve.audit_inputs(reserve, *, bank_files, metric_cache_dir, label_store, image_index, calib_rows)` called with the five inputs read off the LAUNCH argv; `ok` False / absent / truthy-not-True / non-dict / raising / module absent / signature drift all FAIL.

## Evidence

* **Static block** (what the refe_p7 profile's G-SUITE runs; in the after-suite): `ZZP7HEAD_PASS 179` ; the pre-existing `test_launch_gate.py --static --no-banked`: before `ZZLG_STATIC_PASS 1168`, after `ZZLG_STATIC_PASS 1168`.
* **`--full`** (F1 default-off bit-identity vs the frozen tip trainer, with its determinism control; F2 a head arm through the REAL gate: G-DVB, G-LIVE, G-EVAL, G-CKPT all PASS; F3 the four gate-instrument arms): `ZZP7HEAD_PASS 207 | ZZP7HEAD_FULL 207`.
* **`--mutations`** (8 real gate runs on temp copies: 1 control + 7 arms): `ZZP7HEAD_PASS 187`.
* **Full `refe/` suite, before vs after** (same box, same env, same runner `run_suite.sh`; the table is `raw/SUITE_BEFORE_AFTER.md`, also below): 51 test scripts (50 at the tip + the new one). rc 0: before 37 / after 39. Identical rc AND final token on 47; differing: ['test_hnscreen_chain', 'test_rls1_queue', 'test_suite_nesting']; new: ['test_p7_head_wiring']. Sum of the `ZZ…PASS <n>` counts of the green tests: before 8,270 / after 8,486 (the difference is the new test's count).
Red BEFORE (the same env, the tip's own code): ['test_aux_drivable_head', 'test_aux_label_store', 'test_aux_s1_wiring', 'test_carve_feed', 'test_hn_bank', 'test_hn_screen_wiring', 'test_hnscreen_chain', 'test_p6_planner_command', 'test_rl_fan_dump', 'test_rls1_chain', 'test_rls1_queue', 'test_s1aux_chain', 'test_sft4r_chain']. Red AFTER: ['test_aux_drivable_head', 'test_aux_label_store', 'test_aux_s1_wiring', 'test_carve_feed', 'test_hn_bank', 'test_hn_screen_wiring', 'test_p6_planner_command', 'test_rl_fan_dump', 'test_rls1_chain', 'test_rls1_queue', 'test_s1aux_chain', 'test_sft4r_chain']. Causes of the pre-existing reds, read off their own output (ENVIRONMENT of this box / the minimal export, not the change): missing `shapely` / `nuplan` modules, `raw/` artifacts absent from the export, a dev-box lock file touched by a parallel agent during the run, and non-`.py` fixtures rewritten to CRLF by the export (`fixtures/scorer_finetune_pre_aux.py.txt` re-hashes to its pin only after CRLF -> LF: MEASURED).

| test (`refe/`) | BEFORE (tip) | before token | AFTER (this change) | after token |
|---|---|---|---|---|
| `test_aux_drivable_head` | rc 1 | ZZTEST_FAIL  1 | rc 1 | ZZTEST_FAIL  1 |
| `test_aux_g12_bf16` | rc 0 | ZZTEST_SKIP_NO_CUDA | rc 0 | ZZTEST_SKIP_NO_CUDA |
| `test_aux_label_store` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_aux_s1_wiring` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_carve_feed` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_carve_feed_gate` | rc 0 | ZZTEST_PASS 10 | rc 0 | ZZTEST_PASS 10 |
| `test_ctx_cache` | rc 0 | ZZTEST_PASS 15 | rc 0 | ZZTEST_PASS 15 |
| `test_d9_accept` | rc 0 | ZZTEST_PASS 16 | rc 0 | ZZTEST_PASS 16 |
| `test_dump_scores` | rc 0 | ZZTEST_PASS 16 | rc 0 | ZZTEST_PASS 16 |
| `test_field_loss` | rc 0 | ZZTEST_PASS 42 | rc 0 | ZZTEST_PASS 42 |
| `test_flock_rw` | rc 0 | ZZTEST_PASS 59 | rc 0 | ZZTEST_PASS 59 |
| `test_gate_gpu_probes` | rc 0 | ZZTEST_PASS 14 | rc 0 | ZZTEST_PASS 14 |
| `test_gate_suite_prerun` | rc 0 | ZZTEST_PASS 144 | rc 0 | ZZTEST_PASS 144 |
| `test_hf_egress` | rc 0 | ZZTEST_PASS 13 | rc 0 | ZZTEST_PASS 13 |
| `test_hn_bank` | rc 1 | ZZHNBANK_SY | rc 1 | ZZHNBANK_SY |
| `test_hn_content_fp` | rc 0 | ZZTEST_PASS 81 | rc 0 | ZZTEST_PASS 81 |
| `test_hn_ctx_anchor` | rc 0 | ZZTEST_PASS 72 | rc 0 | ZZTEST_PASS 72 |
| `test_hn_screen_wiring` | rc 1 | ZZTEST_FAIL 2 | rc 1 | ZZTEST_FAIL 2 |
| `test_hnscreen_chain` | rc 1 | ZZTEST_FAIL 1 | rc 0 | ZZTEST_PASS 31 |
| `test_hnscreen_ctx_chain` | rc 0 | ZZTEST_PASS 22 | rc 0 | ZZTEST_PASS 22 |
| `test_label_dir_refusal` | rc 0 | ZZTEST_PASS 6 | rc 0 | ZZTEST_PASS 6 |
| `test_launch_gate` | rc 0 | ZZLG_STATIC_PASS 1168 | rc 0 | ZZLG_STATIC_PASS 1168 |
| `test_p2r_chain` | rc 0 | ZZTEST_PASS 139 | rc 0 | ZZTEST_PASS 139 |
| `test_p2rep_chain` | rc 0 | ZZTEST_PASS 353 | rc 0 | ZZTEST_PASS 353 |
| `test_p6_chain` | rc 0 | ZZP6CHAIN_PASS 432 | rc 0 | ZZP6CHAIN_PASS 432 |
| `test_p6_cpu_probe` | rc 0 | ZZTEST_PASS 111 | rc 0 | ZZTEST_PASS 111 |
| `test_p6_levers` | rc 0 | ZZP6_PASS 165 | rc 0 | ZZP6_PASS 165 |
| `test_p6_overlap` | rc 0 | ZZTEST_PASS 54 | rc 0 | ZZTEST_PASS 54 |
| `test_p6_planner_command` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_p7_head_wiring` | - | (no output) | rc 0 | ZZP7HEAD_PASS 179 |
| `test_pdm_label_complete` | rc 0 | ZZTEST_PASS 13 | rc 0 | ZZTEST_PASS 13 |
| `test_r0e_b9b10` | rc 0 | ZZTEST_PASS 3601 | rc 0 | ZZTEST_PASS 3601 |
| `test_rl_epo` | rc 0 | ZZTEST_PASS 42 | rc 0 | ZZTEST_PASS 42 |
| `test_rl_fan_dump` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_rl_partition` | rc 0 | ZZTEST_PASS 20 | rc 0 | ZZTEST_PASS 20 |
| `test_rl_pickcon` | rc 0 | ZZTEST_PASS 20 | rc 0 | ZZTEST_PASS 20 |
| `test_rl_pool_available` | rc 0 | ZZTEST_PASS 10 | rc 0 | ZZTEST_PASS 10 |
| `test_rl_r0c` | rc 0 | ZZTEST_PASS 26 | rc 0 | ZZTEST_PASS 26 |
| `test_rl_reward` | rc 0 | ZZTEST_PASS 31 | rc 0 | ZZTEST_PASS 31 |
| `test_rl_s1b` | rc 0 | ZZTEST_PASS 18 | rc 0 | ZZTEST_PASS 18 |
| `test_rl_safeobj` | rc 0 | ZZTEST_PASS 61 | rc 0 | ZZTEST_PASS 61 |
| `test_rl_terms_label` | rc 0 | ZZTEST_PASS 39 | rc 0 | ZZTEST_PASS 39 |
| `test_rl_tlane_label` | rc 0 | ZZTEST_PASS 36 | rc 0 | ZZTEST_PASS 36 |
| `test_rls1_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_rls1_queue` | rc 1 | ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL | rc 1 | (no ZZ token) |
| `test_rls1e_chain` | rc 0 | ZZTEST_PASS 936 | rc 0 | ZZTEST_PASS 936 |
| `test_s1aux_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_s1c` | rc 0 | ZZTEST_PASS 220 | rc 0 | ZZTEST_PASS 220 |
| `test_sft4r_chain` | rc 1 | (no ZZ token) | rc 1 | (no ZZ token) |
| `test_sft4r_gate` | rc 0 | ZZTEST_PASS 153 | rc 0 | ZZTEST_PASS 153 |
| `test_suite_nesting` | rc 0 | ZZTEST_PASS 112 | rc 0 | ZZTEST_PASS 118 |

## Deliberate-regression arms (each turns the real gate RED; control stays PASS) — `raw/mutation_check.out`

```
  PASS   CONTROL: the unmutated trainer: G-DVB PASS, G-LIVE PASS
  RED    M1 delete the line that adds L_aux to the total loss (plain backward site): G-DVB FAIL (--w-aux-drv: declared 0.1 != built None (the multiplier measured at the graph: d(loss)/d(L_aux) x accum on 0 o); G-LIVE FAIL (only 0 of 43 head tensors received a gradient (the rest have no autograd path))
  RED    M2 default weight 0.0, --w-aux-drv omitted from the argv (profile requires it): G-DVB FAIL (profile refusal: refe_rehearsal_p7 requires --w-aux-drv 0.1; the argv does not carry it); G-LIVE FAIL (profile refusal: refe_rehearsal_p7 requires --w-aux-drv 0.1; the argv does not carry it)
  RED    M2b ... and on a profile WITHOUT that requirement the trainer's own refusal stops it: G-DVB FAIL (the BUILD run did not reach the optimiser: rc=4 ); G-LIVE FAIL (the SMOKE trainer run did not finish: rc=4 )
  RED    M3 drop the head's param group from the optimiser: G-LIVE FAIL (43 of 43 head tensors are NOT held by the optimiser (the head's param group was dropped: it computes a gradien)
  RED    M4 tap scene_ctx instead of visual_ctx: G-DVB FAIL (--aux-drv-tap: declared 'visual' != built 'scene' (the storage the head's input IS, on 4 forwards: scene_proj')
  RED    M5 flag on, head not built: G-DVB FAIL (--aux-drv: declared True != built False (an AuxDrivable is constructed iff --aux-drv))
  RED    M6 build 1 block instead of 2: G-DVB FAIL (--aux-drv-depth: declared 2 != built 1 (AuxDrivable.built_facts(): the CrossBlocks the head holds))
```
(`RED` = the named check FAILs with the named reason; the CONTROL row must PASS. Argument-level arm M2: the omitted `--w-aux-drv` is refused by the profile; M2b shows the trainer's own refusal behind it.)

Also RED in the static block (S6, source mutation of `train.py`, 17 mutations incl. **SITE 1 deleted** and the `--grow` loader; the online path cannot run here) and in the F3 gate-instrument arms (`aux_weight_zero`, `aux_weight_off`, `aux_term_detached` → G-LIVE/G-DVB; `resume_drops_aux` → G-CKPT).

## Corrected stale anchors (reported, not edited in the specs)

| claim | status |
|---|---|
| `SPEC_AUX_DRIVABLE_HEAD.md` §5/§7: "`launch_gate.py` is not on the tip; `eval/REFE_LAUNCH_GATE.md` does not exist yet" | **STALE**: both exist at the tip (`refe/launch_gate.py` with `refe_p6`; `eval/REFE_LAUNCH_GATE.md`) |
| `SPEC_P7.md` §1: `train.py` backward sites "≈ l.1754 / 1769", `IDENTITY_*` "≈ l.996–1006", TargetBank "≈ l.635–653, 906" | tip (before this change) **and** now: SITE 1 `1862`, SITE 2 `1880`, `IDENTITY_ARGS` `1035`, `IDENTITY_DEFAULTS` `1043-1047`, TargetBank tail `671-676`/`950` (table above) |
| `WIRING_PLAN.md` §3 anchors (`:1303`, `:1693`, `:1823`, …) | stale (they predate the online deferred backward and the L_field wiring); use the table above |
| `SPEC_P7.md` §1: "Not in `model_final.pt`" | refined: the head is **beside** `model` (key `aux_drv`), **never inside it**; the planner's strict load of `model` is unchanged (G-EVAL proves it). WIRING_PLAN §3 step 8 asks for the key in all three dicts |

## Two measurements made on the way (MEASURED, read-only)

* **The seed-0 head's initial BCE is not ln 2 for imbalanced labels.** Its common-mode logit is ≈ −0.43 (`raw/scratch/logit_probe.py`; seed 1: +0.05, seed 2: −0.25), independent of the context scale, so BCE at positive share p = 0.1 / 0.2 / 0.3 / 0.5 reads 0.56 / 0.60 / 0.65 / 0.73. The SPEC's band `ln 2 ± 0.15` is therefore a function of the label balance. **The real store's mean positive share is 0.38** (40 random shards, 2,144 labels; `raw/scratch/store_pos_fraction.py`), giving ≈ 0.69, so the band is met on real data; a rehearsal fixture with ~10 % positives (mine, first) reads 0.54 and trips it. The fixture now uses ±11 m corridors (~35 %). If the pod's G-LIVE ever FAILS this row, compare with the store's measured share before suspecting the wiring.
* A DataLoader worker keeps every shard it reads (`max_logs=None`): ≤ ~351 MB compressed per worker for the full store (the module's own docstring, INHERITED); ×2 workers at P6's `--workers 2`.

## Cost (restated, ESTIMATED — no pod was available)

Step time **+0.8 – 1.6 %** (ESTIMATED: FLOP share ×2 for attention, SPEC_P7 §1) up to **+2.1 – 2.4 %** (MEASURED on S1's scorer-only host, +0.51 GB; a different host) ⇒ on P7's 9,075 × 57.0 s = 143.7 h (arithmetic, SPEC_P7) that is **+1.2 to +3.5 h**. **Still to measure on the pod (not done here):** an A/B of the trunk trainer's per-step time on the A40 — the same bank, `--batch 4 --accum 64`, ≥ 200 micro-batches, `--aux-drv --w-aux-drv 0.1` vs its head-off twin — reading `elapsed_s` from the step rows (⚠️ `step_s` in trainer logs is ACCUMULATED over `--log-every`), plus `cuda_max_mem_gb` for the head's memory at batch 4. That run is also G-LIVE-MEM's input.

## Deliverable manifest

| repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`) | location | note |
|---|---|---|
| `refe/train.py` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/train.py` | **single location**; sha256 `ad865dab0ec3…`, 145845 B; replaces tip blob 6b425ac7e72c… |
| `refe/gate_checks.py` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/gate_checks.py` | **single location**; sha256 `e8e004e94f1c…`, 166719 B; replaces tip blob d4d711e01411… |
| `refe/launch_gate.py` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/launch_gate.py` | **single location**; sha256 `b479eded942b…`, 133719 B; replaces tip blob 3eb504c46321… |
| `refe/test_launch_gate.py` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/test_launch_gate.py` | **single location**; sha256 `b566dc55f5c2…`, 189803 B; replaces tip blob 850e14c1fa1b… |
| `refe/test_p7_head_wiring.py` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/test_p7_head_wiring.py` | **single location**; sha256 `3efa00fd6b6b…`, 64222 B; NEW |
| `refe/fixtures/train_pre_p7_head.py.txt` | `C:/Users/Admin/par18/p7wire_head/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/fixtures/train_pre_p7_head.py.txt` | **single location**; sha256 `6b425ac7e72c…`, 134442 B; NEW |
| `P7_WIRE_HEAD.diff` | `C:/Users/Admin/par18/p7wire_head/code/P7_WIRE_HEAD.diff` | **single location**; unified diff against the tip (all 4 changed + 2 new files) |
| `RESULT.md` (this file) | `C:/Users/Admin/par18/p7wire_head/RESULT.md` | **single location** |
| raw outputs: `raw/mutation_check.out`, `raw/p7_head_wiring_full.out`, `raw/SUITE_BEFORE_AFTER.md`, `raw/baseline/`, `raw/after/`, `raw/scratch/` (probes) | `C:/Users/Admin/par18/p7wire_head/raw/` | **single location** |
| patch scripts (provenance), runners | `C:/Users/Admin/par18/p7wire_head/patches/`, `run_suite*.sh`, `run_chain.sh` | **single location** |

Line endings of the staged files (the lander writes blobs; must be LF): train.py CR-lines=0; gate_checks.py CR-lines=0; launch_gate.py CR-lines=0; test_launch_gate.py CR-lines=0; test_p7_head_wiring.py CR-lines=0; fixtures/train_pre_p7_head.py.txt CR-lines=0
**Everything above lives ONLY in this package directory** — nothing is in any git working tree, index, branch or remote. A lost disk loses it; the Master Mind's scoped lander is the step that moves it into the repo.

## Reproduce

```
cd "<package>/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
CUDA_VISIBLE_DEVICES=-1 python test_p7_head_wiring.py             # static: what G-SUITE runs        -> ZZP7HEAD_PASS <n>
CUDA_VISIBLE_DEVICES=-1 python test_p7_head_wiring.py --full      # + F1 bit-identity, F2 real gate, F3 arms (needs the dev box's refe_consistent, READ ONLY)
CUDA_VISIBLE_DEVICES=-1 python test_p7_head_wiring.py --mutations # the arms through the real gate on temp copies
CUDA_VISIBLE_DEVICES=-1 python test_launch_gate.py --static --no-banked   # -> ZZLG_STATIC_PASS 1168
```
