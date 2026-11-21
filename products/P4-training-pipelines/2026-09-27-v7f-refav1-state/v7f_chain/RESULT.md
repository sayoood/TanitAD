# v7F chain — `scripts/v6_chain.py --profile v7f` (PI R1–R6, 2026-09-27)

**2026-09-27, TrainingFlyWheel (P4) sub-agent. CPU only (`CUDA_VISIBLE_DEVICES=-1`), isolated copy
`C:/Users/Admin/v7f_chain` (a byte copy of the integrated v7F merge). STAGED, never committed.**

## Headline

| | |
|---|---|
| **Dry ladder** | `run --dry --profile v7f` — **S-W rc 0 → S-T rc 0, 2 steps, 0 failures, 11.4 s** on the REAL trainer (tiny geometry) with the REAL data inputs (canonical v7.2 train blob, md5 `0ff90213…`, 4,572 records; a max-speed sidecar built over that blob by the real builder). MEASURED (`raw/dry_ladder_v7f.log`, `raw/dry_ladder_v7f.json`). |
| **Gate-rule verdicts** | `launch_gate.profile_argv_rules(PROFILES["v7f"], argv)` on the emitted lines: **S-W 0 refusals / 0 pending; S-T 0 refusals / 1 PI-pending (`--tac-op-cond detached` = PI decision D1)**. Per rule: S-W 14/14 rows PASS; S-T 20/20 rows PASS + D1 = PI-DECISION (§3). MEASURED (`raw/emitted_and_gate.json`). |
| **Trainer preflight on the REAL (non-dry) lines** | **0 refusals on S-W and on S-T** (Thor data paths swapped for local copies of the same files, corpus → an empty dir). MEASURED (`raw/emit_and_gate.log`). |
| **Tests** | `tests/test_v6_chain.py` + new `tests/test_v6_chain_v7f.py`: **106 passed (49 + 57), 0 failed, 58.4 s**. Baseline before the change: 49 passed. MEASURED (`raw/tests_v6_chain_and_v7f.log`, `raw/baseline_test_v6_chain.log`). |
| **Default-profile byte identity** | **22/22 CLI invocations identical** (stdout, stderr, rc, every written file) between the pre-change file (sha256 `918859e7…`) and the patched file (`b1bf626b…`). MEASURED (`raw/byte_identity_default_profile.json`). |
| **Mutation** | 4 source mutations of `v6_chain.py`, each turns the new file RED: M1 delete the emission guard → 17 failed; M2 v7f S-T emits `--tac-goal-cond` → 27 failed; M3 S-T drops `--plan-vmax-cap` → 26 failed; M4 CLI leaves the fan at the v6 default → 7 failed. File restored, sha re-verified. MEASURED (`raw/mutations.json`). |
| **Files the parent must apply** | `stack/scripts/v6_chain.py` (patched) + NEW `stack/tests/test_v6_chain_v7f.py` — via `code/apply_v7f_chain.py` (anchor-based, EOL-preserving) or by copying `code/fix/stack/...`. Nothing else changes. |
| **Flags that differ from the gate's `ST_ARGV`** | **No R-flag differs.** Chain-only: `--w-t1 1.0` (= trainer default), `--max-horizon 60`, `--dump-seam-plan`, the data block (`--v2-val-cache --v2-lru 6 --frame-hfov 120 --projection cylindrical`) and 60 flags CARRIED from S-W's own record (E1). Gate-only: `--log-every 2` (a rehearsal knob). S-W additionally carries `--n-candidates 1`, which the test's `SW_ARGV` strips (R5 requires it). §4 justifies each. |

⚠️ **Three things to escalate** (§6): **(F1)** a MEASURED X3 isolation defect of the v7F merge (the tactical nav path reaches the shared nav embedding) with its lever MEASURED — owner `tanitad/models/nav_conditioning.py`; **(F2)** the ST_ARGV data pairing `nav.jsonl.gz` + `refcv6_speed_max_v8_train.jsonl` must be md5-consistent or S-T dies at `build_vmax_join` AFTER the corpus build — UNVERIFIED on Thor, check before launch; **(F3)** the pre-existing default v6 ladder's non-dry lines carry no `--nav-cond` and the trainer's preflight REFUSES them (MEASURED on S-T) — not changed here (byte identity was required).

## 1. What changed (`stack/scripts/v6_chain.py`, +438 / −10 lines, `raw/v6_chain.diff`)

`--profile {v6,v7f}`, default `v6`. Under **v7f**:

* **ladder = S-W → S-T only.** `--step S-S|S-J` and `--stop-after S-S|S-J` are REFUSED BY NAME quoting R6 (`assert_v7f_step_request`); S-S/S-J are never planned.
* **`tac_goal_cond` forced False** by the CLI; a config with it ON (the `ChainConfig` default) is REFUSED with R6 (V6Config refuses the pair with `--strategic-off`).
* **R5 option B:** `--n-candidates 1` on BOTH stages (a CLI `--n-candidates N≠1` is refused; the dry fan of 3 is overridden); S-T adds `--proposals query --selector none`. `--st-arms` (SEL-1/R5) and `--st-winner` (R6) are refused.
* **R2 nav on both stages** (`--nav-cond --nav-labels`); **R6** `--strategic-off` on both; **R1** `--max-speed-input-v6 --plan-vmax-cap --speed-max-sidecar-v6`, **R3** `--w-tac-label-all 1.0 --goal-multilabel --s2-labels`, **R4** `--tac-op-cond <mode>` on S-T — exactly as in `ST_ARGV`.
* **`--tac-op-cond` is a declared config value** `ChainConfig.tac_op_cond`, default `V7F_TAC_OP_COND_DEFAULT = "detached"`, commented as **PI decision D1 (open)**; `e2e` accepted, `off` refused (R4).
* **The three data paths are REQUIRED** (`--nav-labels`, `--s2-labels`, `--speed-max-sidecar-v6`; refused when absent/blank) and the `~` refusal now covers them (`None` under v6, so v6 is unchanged).
* **Out dirs distinct from v6F:** `v7f-SW-{sw_steps/1000}k`, `v7f-ST-{st_steps/1000}k`; an `--sw-dir v6F-…` is refused (R2: a v6F S-W never built `nav.*`, and S-T may not introduce it).
* **S-W geometry is DECLARED** (`V7F_SW_GEOMETRY` = the gate's ST_ARGV geometry block verbatim: `--in-channels 3 --enc-dim 768 --enc-depth 12 --enc-heads 12 --frame-h 256 --frame-w 640 --horizons 1`, plus the levers `--strategic-off --goal-multilabel --newest-frame-only`). v7f's S-W is a NEW run; without a declared geometry it would build at the trainer defaults — `--horizons 1 2 4`, which the trainer's preflight REFUSES on a fresh run. S-T never types geometry: it CARRIES S-W's own `config.json` (E1). A v7f S-W never carries `--geometry-from` (that would drag an older run's lever values, e.g. `--tac-op-cond off`, into it).
* **No v7f line leaves the chain unchecked:** `trainer_argv` runs `assert_v7f_argv` on EVERY emitted argv (dry or real, `commands`/`next`/`manifests`/`run`) = the launch gate's own `profile_argv_rules(PROFILES["v7f"])` **imported, never re-implemented** + the chain's rows the gate does not encode (R5 fan/proposals/selector/`--w-select`, and the per-stage data paths). Unimportable gate ⇒ refusal. `assert_may_launch` reports the verdict as `v7f_rules`.
* `plan` adds a `v7f` block (requirements, D1, the md5 pairing note) and counts S-W in the wall-clock (it is a new run); `manifests` writes S-W's manifest too.
* **Every existing guard still runs** (geometry carry E1 incl. the full derived diff, seam dump E5, the n-candidates rule, no `--v2-subframe`, tilde refusal, SEL-1 admission, gate precondition, `dry_ckpt.pt` ancestry) — the v7f plan goes through the same `assert_may_launch`/`trainer_argv`.

**The two v6 S-T extras under option B** (justified in `_build_plan_v7f`'s docstring):
* `--plan-wta-eps` **DROPPED** — `v6_loss_step` builds the ε-relaxed loser term only `if n > 1` (`train_v6_staged.py`, the `plan_wta_eps` block ~:5025–5030); at `--n-candidates 1` any ε > 0 is advertised-but-inert. (It is a geometry dest, so S-T carries S-W's recorded default `--plan-wta-eps 0.0`.)
* `--w-t1 1.0` **KEPT** — the t1 term trains `layer_tac`'s dynamics and is in force in the gate's rehearsed S-T (G-LIVE terms `plan/seam/t1`) at the trainer default 1.0; stating it pins the value (E1's rule). Confirmed in force in the dry S-T's effective-weight table: `t1_latent … TRAINS` (MEASURED, `raw/dry_ladder_v7f.log`).

**v6 byte identity** is by construction (all new `ChainConfig` fields appended last and hidden from the v6 `plan` view; every branch keyed on `profile == "v7f"`) and by measurement: 22/22 invocations — `plan` (×7 variants incl. `--dry`, `--tiny`, arms, `--a40 --no-tac-goal-cond`, `--out-json`), `commands` (×7 incl. arms, S-S `--no-seam-dump`, overrides, the E1 refusal in `--dry`, `S-X`, tilde), `admission` (×2), `status`, `next`, `manifests` (3 files hashed), `verify`. A v7f-only flag on a v6 ladder is REFUSED (it would otherwise be inert) — the only v6-visible change, and only for flags that did not exist before.

## 2. The dry ladder (MEASURED, `raw/dry_ladder_v7f.*`)

`python scripts/v6_chain.py run --dry --profile v7f --root <SCRATCH>/dry1 --nav-labels <blob> --s2-labels <blob> --speed-max-sidecar-v6 <sidecar>`

**What the data inputs were — exactly.** The dry trainer uses synthetic batches, but it REALLY reads two of these flags: `--s2-labels` (loaded through `load_s2_labels_any` → md5 pin, v7 vocab check, R3 label policy) and `--speed-max-sidecar-v6` (read by `read_speed_max_sidecar_v6` against `label_md5 = md5(--nav-labels)`); `--nav-labels` itself is only md5'd in a dry run (the synthetic batch carries nav keys). So I used real files: the canonical v7.2 TRAIN blob copied from `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz` (md5 `0ff902130ce76886b8a925eceed9e3a5` = `intrain_eval.V72["train"]`), used as BOTH `--nav-labels` and `--s2-labels`, and a sidecar built over that same blob with the real builder (`scripts/build_refcv6_speed_max_window.py --omit-clip-id`: 4,572 clips, 4,572 with a band, census 1741/1593/1014/224 over {30,50,100,120} km/h, 86 clamped; meta in `raw/sidecar_v72_train.meta.json`).

| stage | rc | gate | what was exercised |
|---|---|---|---|
| S-W `v7f-SW-30k` | 0 | INCONCLUSIVE (`_dry_run`) | built with `nav_cond True, strategic_off True, tac_goal_cond False, n_candidates 1, tac_op_cond off`; X3 isolation **pass** |
| S-T `v7f-ST-10k` | 0 | INCONCLUSIVE (`_dry_run`, advanced only through the recorded override) | precondition exercised; `--init-from dry_ckpt.pt` loaded with **missing 0 / unexpected 0**, introduced exactly `tac_op_port.{bias,weight}`, `vmax_tac.{bias,weight}` (nav carried from S-W); s2 v7.2 blob loaded (**4,572 records, s2-geom-v7, side=train**); R3 policy ran (**17/22 goal classes trainable**, G-DVB v6 PASS); R1 sidecar read (`source_md5` = the blob's md5, 4,572 valid rows); R1b cap smoke **0 plans over the limit**; all 83 optimizer tensors got a gradient; built `nav_cond, strategic_off, n_candidates 1, proposals query, selector none, tac_op_cond detached, max_speed_input_v6, plan_vmax_cap, goal_multilabel` |

The S-T X3 isolation probe reads **pass = False** — finding F1 (§6). Tier: T0/synthetic smoke; **no number here is a driving number.**

**Negative control (MEASURED, `raw/dry_ladder_negative_control.*`):** the same ladder with the sidecar's meta `source_md5` set to the v8.0 blob's (`fa89ea55…`, the banked refcv6 sidecar's source): S-W rc 0, **S-T rc 1** — `[refcv6-vmax] ⛔ the sidecar was built over label blob md5 'fa89ea55…' but this run loaded '0ff90213…'`. The data paths are really read, and the R1 pairing is enforced.

## 3. Gate-rule verdicts on the emitted lines (MEASURED, `raw/emitted_and_gate.json`)

Lines generated with Thor paths mirroring the gate's `ST_ARGV`; S-T's geometry carried from a DERIVED S-W record (the trainer's REAL parser applied to the emitted S-W line — what S-T carries if S-W runs with exactly that line; on Thor the real record wins).

| rule row (launch_gate `PROFILES["v7f"]`) | req | S-W | S-T |
|---|---|---|---|
| `stage_allowed` (S-W, S-T) | R6 | PASS | PASS |
| `required_on_v6` `--nav-cond` | R2 | PASS | PASS |
| `required_on_v6` `--strategic-off` | R6 | PASS | PASS |
| `required_on_v6_stage[S-T]` `--max-speed-input-v6` / `--plan-vmax-cap` / `--speed-max-sidecar-v6` | R1 | n/a | PASS / PASS / PASS |
| `required_on_v6_stage[S-T]` `--goal-multilabel` | R3 | n/a | PASS |
| `required_on_v6_stage[S-T]` `--tac-op-cond` | R4 | n/a | PASS |
| `required_positive_v6_stage[S-T]` `--w-tac-label-all` > 0 | R3 | n/a | PASS (1.0) |
| `forbidden_flags_v6`: predates-nav, control-arm, 3× no-isolate, `--tac-goal-cond`, `--goal-factored` | R2/R6/R3/X3 | 7/7 PASS | 7/7 PASS |
| `forbidden_values_v6` `--selector` ∈ {goal, mlp} | SEL-1 | PASS (absent) | PASS (`none`) |
| `forbidden_values_v6` `--tac-op-cond off` | R4 | PASS (absent) | PASS (`detached`) |
| `forbidden_positive_v6` `--w-s1-multi`, `--w-s2-goal` | R6 | PASS | PASS |
| `pi_pending_values` `--tac-op-cond` | R4 / **D1** | n/a | **PI-DECISION** |
| **`profile_argv_rules`** | | **0 refusals, 0 pending** | **0 refusals, 1 pending (D1)** |
| chain rows (R5 fan / proposals / selector, data paths) | R5 | 0 refusals | 0 refusals |

## 4. The emitted lines, verbatim (Thor paths as in `ST_ARGV`; UNVERIFIED on Thor)

**S-W**
```
mkdir -p /home/nvidia/experiments/v7f-SW-30k && cd /home/nvidia/TanitAD/stack && PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-W --out /home/nvidia/experiments/v7f-SW-30k --steps 30000 --batch 8 --lr 0.0001 --n-candidates 1 --nav-cond --nav-labels /home/nvidia/data/v72/nav.jsonl.gz --strategic-off --goal-multilabel --newest-frame-only --in-channels 3 --enc-dim 768 --enc-depth 12 --enc-heads 12 --frame-h 256 --frame-w 640 --horizons 1 --v2-cache /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --save-every 250 --frame-hfov 120 --projection cylindrical --require-parity > /home/nvidia/experiments/v7f-SW-30k/train.out 2>&1 < /dev/null &
```

**S-T**
```
mkdir -p /home/nvidia/experiments/v7f-ST-10k/seam && cd /home/nvidia/TanitAD/stack && PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python3 -u scripts/train_v6_staged.py --stage S-T --out /home/nvidia/experiments/v7f-ST-10k --steps 10000 --batch 8 --lr 0.0001 --n-candidates 1 --a-max 4.0 --adapter-hidden 512 --anchor-goal none --d-goal-embed 128 --d-str 256 --d-t2-hidden 256 --d-t2-proj 128 --d-tac 512 --diffusion-hidden 256 --diffusion-noise-rho 0.9 --diffusion-sigma-a 2.0 --diffusion-sigma-k 0.1 --diffusion-steps 4 --dt 0.1 --ema-decay 0.996 --ema-decay-ramp off --ema-decay-start 0.99 --enc-depth 12 --enc-dim 768 --enc-grad-checkpoint auto --enc-heads 12 --f-blocks 3 --f-hidden-str 512 --f-hidden-tac 512 --fallback-roll-k 10 --frame-h 256 --frame-w 640 --horizons 1 --in-channels 3 --kappa-max 0.2 --mpc-lr 0.05 --mpc-roll-k 0 --mpc-steps 3 --mpc-topk 2 --mpc-w-consist 0.0 --mpc-w-goal 1.0 --mpc-w-kin 0.1 --n-agent-slots 8 --n-anchors 256 --n-lat-bins 16 --n-registers 4 --n-slot-queries 100 --newest-frame-only --o5-target live --o5-target-crop 0.0 --param-budget 300000000 --patch 16 --plan-steps 60 --plan-wta-eps 0.0 --pred-depth 6 --pred-dim 768 --pred-heads 12 --readout-dim 128 --readout-grid 4 --selector-mlp-hidden 256 --selector-tau-m 1.0 --sigreg-free-dims 0 --sigreg-slices 512 --sigreg-subspaces 1 --slot-depth 3 --slot-heads 8 --slot-hidden 256 --slot-src cells --t2-tau 0.1 --tac-vocab-version v7.0 --uplink stopgrad --w-o14 0.0 --window 6 --init-from /home/nvidia/experiments/v7f-SW-30k/ckpt.pt --prev-gate /home/nvidia/experiments/v7f-SW-30k/stage_gate.json --max-horizon 60 --proposals query --selector none --w-t1 1.0 --nav-cond --nav-labels /home/nvidia/data/v72/nav.jsonl.gz --strategic-off --max-speed-input-v6 --plan-vmax-cap --speed-max-sidecar-v6 /home/nvidia/data/refcv6_speed_max_v8_train.jsonl --w-tac-label-all 1.0 --goal-multilabel --s2-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz --tac-op-cond detached --v2-cache /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl --v2-val-cache /home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl --v2-lru 6 --save-every 250 --frame-hfov 120 --projection cylindrical --require-parity --dump-seam-plan /home/nvidia/experiments/v7f-ST-10k/seam > /home/nvidia/experiments/v7f-ST-10k/train.out 2>&1 < /dev/null &
```

**Discrepancies vs the gate's `ST_ARGV` / `SW_ARGV`** (flag-by-flag diff, `raw/emitted_and_gate.json → diff_vs_gate_argv`):

| flag | chain | gate argv | verdict |
|---|---|---|---|
| all R1–R6 / SEL-1 flags + values, geometry block, `--batch 8 --lr 0.0001 --save-every 250 --require-parity --v2-cache` | = | = | identical (28 flags on S-T, 18 on S-W) |
| `--out`/`--init-from`/`--prev-gate` | `v7f-SW-30k`, `v7f-ST-10k` | `v7f-sw`, `v7f-st` | naming only; the chain's dirs are distinct from v6F by construction |
| S-W `--steps` | 30000 | 10000 | `SW_ARGV` is `ST_ARGV` with `--stage` flipped; 30000 is the chain's S-W constant |
| S-W `--n-candidates 1` | present | stripped by the test's `SW_ARGV` | **REQUIRED (R5)**: without it S-W builds the default fan 8 and S-T's `--init-from` dies on `cand_queries` (the chain's first dry-ladder defect, 2026-08-16). The gate's OWN predecessor builder `_v6_predecessor_argv` keeps it. Keep. |
| `--log-every 2` | absent | present | a rehearsal smoke knob; trainer default 50. Justified. |
| `--w-t1 1.0` | present | absent | same value as the trainer default; kept explicit (§1). |
| `--max-horizon 60` | present | absent (derived) | the v6 S-T's existing windowing choice; = the S-T need at `plan_steps` 60 (`train_v6_staged.py:~7441` derivation, not separately re-measured). |
| `--dump-seam-plan <out>/seam` | present | absent (the gate drops it in rehearsal) | E5 seam instrument, zero extra GPU; S-T only. |
| `--v2-val-cache --v2-lru 6 --frame-hfov 120 --projection cylindrical` | present | absent | the chain's standard real-run data block (Thor constants). |
| 60 carried geometry flags on S-T (`--a-max … --window 6`) | present | absent (= trainer defaults) | E1: carried from S-W's own record; incl. the inert-under-`--selector none` `--selector-tau-m 1.0 --selector-mlp-hidden 256`, `--plan-wta-eps 0.0`, `--w-o14 0.0`. |

## 5. Tests (`stack/tests/test_v6_chain_v7f.py`, 57 tests, LITERAL expectations)

Ladder + literal constants; the S-W and S-T lines flag-by-flag (20 present-with-value + 24 absent on S-W; 34 + 10 on S-T, read through `launch_gate`'s reader, not the chain's); gate rules PASS on both + D1 open; R-flag values = the gate's rehearsed argv (literals); refusals (each data path missing/blank, `~` in each, `tac_goal_cond`, fan 3/8, arms, winner, `off`/unknown mode, `v6F-` S-W dir, profile-only flip names all of them, unknown profile, S-S/S-J by name ×4 CLIs, v7f-only flags on a v6 ladder ×4); **regression arms** that EDIT a planned step and must go RED (S-T with `tac_goal_cond=True` emits `--tac-goal-cond` → caught; 10 missing-R-flag arms; 4 forbidden-value arms; a widened fan; plus a pin that S-T gets its geometry levers back from S-W's record through the E1 carry); a **source mutation** (the guard call deleted → the `--tac-goal-cond` arm escapes); the v6 profile untouched (`plan` config key list literal = the pre-change 30 keys; the v6 S-T line carries none of the v7f flags); `next`/`manifests` under v7f; the **dry ladder executed** on the real inputs and its **md5-pairing negative control**. The dry-ladder tests read the blob from `TANITAD_V72_TRAIN_BLOB` or `<repo>/TanitAD Research Lab/…/s2_labels_v7.2_train.jsonl.gz` (md5-checked) and SKIP with that reason only when neither exists (in the D: repo it exists).

Results (MEASURED): new file **57 passed** (18.4 s); with `test_v6_chain.py` **106 passed** (58.4 s). The 7 other test files that reference `v6_chain` give the **identical** result on the original and the patched chain in this isolated copy — `2 failed, 163 passed, 3 skipped, 14 errors`, the same 16 failed/error ids (environmental: the copy has no `TanitAD Research Lab/` docs and no git history) (`raw/related_ab.json`).

Mutations (`raw/mutations.json`): M1 17 RED · M2 27 RED · M3 26 RED · M4 7 RED; working file restored (sha256 `b1bf626b…` re-verified).

## 6. Findings to escalate (Rule Zero: each with its next lever)

**F1 — X3 isolation defect in the v7F merge, with the fix MEASURED.** After S-T trains the zero-init `nav.layer_proj.tactical` (group `layer_tac`), `assert_isolation`'s tactical-uplink probe reaches the SHARED nav code `nav.embed.weight, nav.arg_proj.weight, nav.arg_proj.bias` (group `predictor_op`, BELOW `layer_tac`): `tactical_to_below = 3`, recorded in the S-T `stage_gate.json` as `X3_isolation pass: false`. Mechanism: `NavConditioner.forward = gate[layer] * layer_proj[layer](embed(tok) + arg_proj(args))` (`tanitad/models/nav_conditioning.py`). Not launch-blocking for S-T (X3 is reported-not-required at S-T, and the shared code is FROZEN there so nothing actually moves), but it is a live forbidden edge the S-T certificate carries and a real leak in any stage that trains both (S-J). **Lever, MEASURED in-process with no file edited** (`raw/x3_nav_lever.*`, same argv, same seed, real `main()`): detach the shared code on the non-operative layers — `code = self.encode(token_id, args); if layer != "operative": code = code.detach()` — ⇒ S-T X3 **FAIL → PASS**, and the S-W AND S-T loss rows are **bit-identical** to the unpatched run (S-T `[38.5746…, 63.2535…]`, t1 `[0.92106…, 0.84963…]`), i.e. zero behaviour change in S-W/S-T. **Owner: the v7F merge / `nav_conditioning.py` (not a file this stream may edit); the pinned assertion in `test_v6_chain_v7f.py` flips in the same change.**

**F2 — the ST_ARGV data pairing must be md5-consistent, and Thor's is UNVERIFIED.** The trainer refuses unless the sidecar meta's `source_md5` == md5(`--nav-labels`) (else `--s2-labels`) — MEASURED by the negative control. The banked refcv6 sidecar (`…/2026-09-17-refcv6-tactical-training/raw/speed_max_window_v6_train.jsonl.meta.json`) records `source_md5 fa89ea55…` (`s2_labels_v8.0_train.jsonl.gz`), while the v7.2 train blob is `0ff90213…`; ST_ARGV pairs `v72/nav.jsonl.gz` with `refcv6_speed_max_v8_train.jsonl`. ⚠️ In a REAL run the refusal fires at `build_vmax_join`, which is called AFTER `build_train_episodes` (the corpus build) — minutes in, not milliseconds; the `--dry-run` of the same line catches it in ~10 s. **Before launch on Thor:** compare the sidecar meta's `source_md5` with `md5sum` of the `--nav-labels` file (or dry-run the exact emitted S-T line); if they differ, rebuild the sidecar over the nav blob with `scripts/build_refcv6_speed_max_window.py --omit-clip-id` (CPU, seconds).

**F3 — pre-existing, v6 profile only (NOT changed: byte identity was required).** The default v6 ladder's non-dry S-T line has no `--nav-cond`; the trainer's preflight REFUSES it (`--nav-cond is REQUIRED (PI directive 2026-08-30)`) — MEASURED on the emitted v6 S-T line (`raw/v6_profile_nav_preflight.log`). (That log's second refusal, `--horizons (1, 2, 4)`, is an artifact of my all-defaults geometry source; on a real S-W record it fires only if that record holds 1 2 4.) The v7f profile is the compliant path.

**Test-harness note (not a chain defect):** with the brief's `PYTHONPATH=<copy>/stack;<copy>` the OUTER `taniteval/` directory becomes a namespace package that shadows the inner one, so an in-process preflight reports `No module named 'taniteval.seam_dump'`; the chain's emitted lines use `<workdir>:<parent>/taniteval`, and with that entry the preflight is clean (MEASURED, §Headline).

## 7. The patch script (`code/apply_v7f_chain.py`)

Template `apply_f7_f8.py`: 16 anchors, each must match exactly once or it refuses and writes nothing. Accepts the base as the CRLF worktree copy (sha256 prefix `918859e75df9c343`) or the LF tip blob (`db2e599d…` — the same content; the D: repo's HEAD blob of `stack/scripts/v6_chain.py`). EOL preserved; the result pinned by its LF-normalised sha256 `657358f1…` (CRLF form `b1bf626b…`); the test file installed in the same EOL (LF-normalised `d84c9631…`, CRLF `4114e2ec…`). **Proven** (`raw/patch_proof.log`): `--check` on the merge copy (read-only, merge file sha unchanged afterwards); apply on a FRESH copy of the merge file → **byte-identical** to the working file and test; apply on the D: HEAD LF blob → equal to the working file LF-normalised, 0 CR; a second apply REFUSES (base no longer the merge's).

usage: `python code/apply_v7f_chain.py --stack <repo>/stack` (add `--check` first).

## 8. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `stack/scripts/v6_chain.py` (patched, CRLF, sha256 `b1bf626b…`) | `worktree:C:/Users/Admin/v7f_chain/stack/scripts/v6_chain.py`; repo:`…/v7f_chain/code/fix/stack/scripts/v6_chain.py` | 2 |
| `stack/tests/test_v6_chain_v7f.py` (NEW, CRLF, `4114e2ec…`) | `worktree:C:/Users/Admin/v7f_chain/stack/tests/`; repo:`…/v7f_chain/code/fix/stack/tests/` | 2 |
| `apply_v7f_chain.py` | repo:`…/v7f_chain/code/apply_v7f_chain.py` | 1 (+ scratch) |
| this `RESULT.md` + `raw/*` (logs, JSON, harness scripts; local paths scrubbed to `<SCRATCH>`/`<V7F_CHAIN>`/`<VENV>`) | repo:`…/v7f_chain/` | 1 |
| dry-ladder run dirs, built sidecar (651,586 B, sid-keyed), label-blob copies | scratch only (reproducible: the builder + the banked blob) | scratch |

Nothing lives only on a pod. `C:/Users/Admin/v7f_merge` was read, never written.
