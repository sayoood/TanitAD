# Wiring plan for the auxiliary drivable-area head (S1 host `scorer_finetune.py`, S2 host `train.py`)

> Written 2026-10-13 (local 08:30), at git tip `37645fcc`. **Nothing here is applied.** Every file named "FROZEN" was only read.
> P = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. The module is `P/refe/aux_drivable_head.py`
> (proof: `P/refe/test_aux_drivable_head.py` -> `ZZTEST_PASS 68`). Spec: `P/SPEC_AUX_DRIVABLE_HEAD.md`.
> Evidence classes: **MEASURED** (ours, artifact named) · **INHERITED** (another doc) · **ESTIMATED** · **UNVERIFIED**.

## 0 · Headline

1. The module is done and wiring-ready: one import, six flags, one container. **All anchors below are valid only while the files hash as
   follows** (taken 2026-10-13 08:30; `git status` shows `gate_checks.py` and `launch_gate.py` as `AM`, i.e. someone else is still editing them):
   `scorer_finetune.py d9e4b2fc…` · `train.py cb2f0f3f…` · `model.py 08d7c216…` · `gate_checks.py 0321544a…` · `launch_gate.py 4247e606…`.
   **First act of the wiring agent: `md5sum` those five; on a mismatch re-derive the line numbers with the `grep -n` patterns in column 3.**
2. **S1 is the cheap rung, and its H1 arm is a probe, not a lever.** `scorer_finetune.py` freezes the whole trunk (`:462-470`) and `sctx`
   carries no graph, so a head attached there can change nothing the scorer sees unless H2 feeds it. S1 therefore reads (i) the perception
   proof (head IoU on held-out logs, from the very runs below) and (ii) H2 against a no-head replicate. The representation lever (H1 proper)
   needs S2 (`train.py`, trunk trainable).
3. **Put the head OUTSIDE the `REFe` module** in both hosts (a sibling object, own optimiser group in S1, own `ckpt` key). Registered as a
   submodule it lands in `ckpt_io.full_state(model)` (`train.py:1823`) and the planner's strict load (`planner.py:354-361`, `ckpt_io.py:138`)
   would refuse `model_final.pt` (G-EVAL). The hook-based tap below needs **no `model.py` edit** (frozen) for H1 and S1; H2 at inference
   (S3) needs one.
4. **Four decisions the wiring agent must NOT make alone** (section 6): the head's learning rate (a 7th flag), the u400 schedule caveat,
   `scorer_finetune.py` has no resume, and the dev-box GPU budget.

## 1 · The module's surface (what the hosts call)

| call | does | where it is proved |
|---|---|---|
| `AUX.add_aux_args(ap)` (`aux_drivable_head.py:110`) | the six flags + `--aux-drv`; **every default disables** (`--w-aux-drv 0.0`, `--aux-drv-feed off`) | A7 |
| `AUX.aux_refusals(a)` (`:121`) | a lever named without `--aux-drv`, labels missing, shuffle with feed on -> refused | A7 |
| `AUX.AuxDrivableConfig.from_args(a)` (`:106`) / `AUX.AuxDrivable(cfg)` (`:309`) | frozen config (G-HYG: typo raises) -> head (own RNG stream) + optional feed | A5, A6 |
| `aux.predict(ctx)` (`:320`) -> `(logit, sdf)` | one tensor in; no ego/goal/nav/map | A3 |
| `aux.loss(logit, sdf, label_i8, valid)` (`:323`) | `L_aux`, `bce`, `huber`; an unlabelled frame is exactly 0 | B8, B9 |
| `aux.combine(l_task, l_aux)` (`:326`) | `l_task + w_aux * l_aux`, **always multiplied** (w=0 -> grads of exactly 0.0, not `None`) | C2, mutant "loss weight ignored" |
| `aux.score(scorer, trajs, sctx, sdf)` (`:332`) | H1: the model's own arithmetic (bit-identical); H2: 3 detached scalars added to `score_q_mlp[0]` | D1-D7 |
| `aux.built_facts()` (`:348`) | what G-DVB compares with the declared config (depth, cell, x0/y0, w, scorer width 60 vs 63, n_params) | A8 |
| `AUX.AuxLabelSource(store, shuffle_seed=, max_logs=)` (`:409`) -> `.labels([(log, token)])` | int8 `[B,160,128]` + `valid`; pinned store sha, per-shard sha, one shard read per log | E1-E12 |
| `AUX.head_metrics(...)` (`:235`) | IoU / acc / SDF-MAE at 2 m: the perception-proof read | B10 |
| `AUX.tap_is(ctx, expected)` (`:355`), `AUX.AUX_IDENTITY_ARGS` (`:69`) | G-DVB tap check; the dests to add to identity | A11 |

H2 is an **additive** 3->256 projection, equal to a 63-wide first layer (`D6`, max |d| 9.5e-7) but leaving `score_q_mlp.0.weight [256,60]`
alone, so **u400 loads strictly** (`D7`); `feed_init="zero"` starts as the base scorer's exact function (`D5`, bit-identical), which keeps
`scorer_finetune`'s G3 ("copies equal model at step 0") true with the feed on.

## 2 · S1 host: `refe/scorer_finetune.py` (FROZEN until the RL-S1 ship) — flag -> attribute -> forward -> loss -> backward

| # | step | anchor today | change | grep pattern to re-find it |
|---|---|---|---|---|
| 1 | import | `:54-58` | `import aux_drivable_head as AUX` | `import train as T` |
| 2 | flags | `build_parser` `:374`, last flag `:428`, `return ap` `:430` | `AUX.add_aux_args(ap)` before `return ap`; **+ a 7th flag `--aux-drv-lr`** (decision D-W1) | `def build_parser`, `--dump-scores` |
| 3 | refusals | `main` `:433`, `epo_refusals(a, ap)` call `:436` | `bad_cli += AUX.aux_refusals(a)` | `bad_cli = epo_refusals` |
| 4 | build head | after `armB = ScorerCopy(model).to(dev)` `:479`, hook `:482-483` | `aux = AUX.AuxDrivable(AUX.AuxDrivableConfig.from_args(a)).to(dev)` if `a.aux_drv`; `src = AUX.AuxLabelSource(a.aux_drv_labels, shuffle_seed=a.aux_drv_shuffle_seed, max_logs=64)` | `armB = ScorerCopy` |
| 5 | **tap** | `cap["sctx"]` is set by the pre-hook on `score_dec[0]` `:483` | **nothing to add**: `sctx` IS `visual_ctx` (`model.py:759`, un-detached or detached per config; the trunk is frozen so it has no graph either way) -> tap = "visual" | `register_forward_pre_hook` |
| 6 | forward | `fwd` `:536`, `sctx = cap["sctx"]` `:541`, `sxB = armB(cxy, sctx)` `:542`, `base(...)` `:544` | after `:541`: `logit, sdf = aux.predict(sctx)`; **H2 only**: `sxB = aux.score(armB, cxy, sctx, sdf)` in place of `:542`; stash `cap["aux"] = (logit, sdf)` (do NOT change `fwd`'s 4-tuple return: it is unpacked at `:630` and `:754`) | `def fwd(batch)` |
| 7 | labels (loader) | `IndexedSubset` `:301-310`, `make_train_set` `:313`, eval loader `:621` | see section 5 | `class IndexedSubset` |
| 8 | **loss** | `lA, lB, ln, ep_t = train_losses(...)` just before `:942` | `l_aux, bce, hub = aux.loss(*cap["aux"], label.to(dev), valid.to(dev))`; `(aux.combine(torch.zeros((), device=dev), l_aux) / a.accum).backward()` **separately** from `:942` (`((lA + lB) / a.accum).backward()` stays byte-identical) | `((lA + lB) / a.accum).backward()` |
| 9 | optimiser | `paramsB = list(armB.parameters())` `:924`, `optA/optB` `:925-926`, loop `for o, ps in ((optA, paramsA), (optB, paramsB))` `:949` | `paramsH = list(aux.head.parameters())`; `optH = AdamW(paramsH, lr=a.aux_drv_lr)`; add `(optH, paramsH)` to the `:949` tuple (**separate grad-clip norm**: a shared clip would let the head change the scorer's update scale and break "H1 == no head"). H2: `aux.feed.proj` joins `paramsB` (it is scorer-side, trained by the score loss) | `optB = torch.optim.AdamW` |
| 10 | log | train row (`row = {"event": "train", ...}` `:958`), `evaluate` `:619` | add `aux_bce`, `aux_huber`, `w_aux`, and per-eval `AUX.head_metrics` on the held-out batches (labels for `ds_ho` too) | `"event": "train"` |
| 11 | save | `save_full` `:720`, `scorer_states_last.pt` `:970-971`, final `:975` | add `"AUX": aux.state_dict()` to the `:970` dict and write `aux_drv_<arm>.pt` + `meta["aux_drv"]` (config + store_sha256) beside `model_sft_B.pt`; H2's planner needs both (S3) | `scorer_states_last` |
| 12 | preflight | gates dict `:742+`, `G11` `:865`, `data` event `:516` | `G12_aux_drv_live_and_identity` (section 4) and `info["aux_drv"] = {**aux.built_facts(), "store_sha256": src.manifest["store_sha256"], "frames": ..., "shuffle_seed": ...}` | `gates["G11_epo` |

Flag -> loss reachability trace (CLAUDE.md "built, tested, unreachable", 2026-09-10/11): `--w-aux-drv` (`add_aux_args`, `aux_drivable_head.py:113`)
-> `AuxDrivableConfig.w_aux` (`:80`) -> `AuxDrivable.w_aux` (`:315`) -> `combine` (`:326`) -> step 8 -> `backward()`. **The preflight must break the
chain at each hop** (delete step 8's line -> G12 reads aux grad 0 -> FAIL; pass `--w-aux-drv` absent -> G-DVB `required_flags` refuses).

## 3 · S2 host: `refe/train.py` (FROZEN) — graft arms and any from-step-0 run

| # | step | anchor today | change |
|---|---|---|---|
| 1 | flags | `build_parser` `:1842`, `--score-w` `:1918`, `return ap` `:1983` | `AUX.add_aux_args(ap)`; + `--aux-drv-lr` only if D-W1 says so |
| 2 | identity | `IDENTITY_ARGS` `:963-967`, `IDENTITY_DEFAULTS` `:970-972`, `run_identity` `:1002-1014`, `_identity_diff` `:1019+` | **append `AUX.AUX_IDENTITY_ARGS` to the tuple AND add their defaults to `IDENTITY_DEFAULTS`** (`aux_drv: False`, `w_aux_drv: 0.0`, `aux_drv_tap: "visual"`, `aux_drv_depth: 2`, `aux_drv_feed: "off"`, `aux_drv_labels: ""`, `aux_drv_shuffle_seed: None`). Without the defaults every pre-existing checkpoint refuses its own resume (`:968-969` comment); without the tuple entries a resume with a different head is accepted |
| 3 | build | `model = REFe(cfg)` `:1126` | `aux = AUX.AuxDrivable(cfg_from_args)` **after** it (own RNG stream, so the trunk's init is bit-identical: tested A5) kept as a SIBLING; `model.scene_proj.register_forward_hook` collecting the **second** call (`visual_ctx`; the first, 64 tokens, is `scene_ctx`; tap "scene" takes the first) |
| 4 | labels | `TargetBank.__getitem__` `:786`, `item = (...)` `:874-878`, `return item + (idx,) if self.online else item` `:880`, loop unpack `:1533-1534` | worker-side lookup `src.labels([(r["log_name"], r["token"])])`, appended **after** the existing tuple; the loop's `ridx = batch[9] if len(batch) > 9` (`:1534`) must become explicit (aux pair at fixed trailing slots) or it will read the label as a row index |
| 5 | forward | `model(...)` `:1544` (on-policy) and `:1547` (plain) | after each: `logit, sdf = aux.predict(ctx_tap)` (inside the autocast block) |
| 6 | **loss** | plain path `loss = (l_traj + l_score) / a.accum` `:1693` + `loss.backward()` `:1694`; **on-policy path `((P["l_traj"] + l_score) / a.accum).backward()` `:1683`** | `loss = aux.combine(l_traj + l_score, l_aux) / a.accum` in **BOTH** (two sites; editing one leaves the other arm head-less: the exact failure of the 2,000-step "V2" arm) |
| 7 | optimiser / clip | `params = [... model.parameters() if p.requires_grad]` `:1303`, `AdamW` `:1308`, clip `:1697` | `params += list(aux.parameters())`; clip list likewise (here the head SHOULD share the trunk's clip: its gradient reaches the LoRA) |
| 8 | checkpoints | partial `:1513-1514`, `:1770-1771`, final `:1823` | add `"aux_drv": aux.state_dict()` + `meta["aux_drv"]` to all three dicts; resume `:1412` reads it back; **Adam state round-trips through `opt` automatically** (G-CKPT mutation: drop the head's optimiser state) |
| 9 | metrics | `row = {...}` `:1712` | `aux_bce`, `aux_huber`, `aux_grad_ratio` (||g_LoRA(L_aux)|| / ||g_LoRA(l_traj + l_score)||, SPEC 2.3, accept 0.05-0.5) |
| 10 | planner | `planner.py:350-361` | H1: nothing (the head is not in `model_final.pt`). H2/S3: build `AuxDrivable` from `meta["aux_drv"]`, load `aux_drv`, and add the 3-scalar term to `REFe.score_trajectories` (`model.py:699`), i.e. the one `model.py` edit this feature ever needs |

## 4 · Launch-gate entries (FROZEN files: `gate_checks.py`, `launch_gate.py`; edit after the ship, with their tests)

| gate | anchor today | entry |
|---|---|---|
| G-HYG | `check_hyg` `gate_checks.py:332`, `SCORER_NAMESPACE_FUNCS` `:568` | add `"aux_refusals"` to the tuple (so `a.aux_drv_*` reads in it count as consumption); the config's frozen dataclass is the `aux_drv_feeed`-typo guard (test A6 is the proof to reuse) |
| G-DVB (scorer) | `DVB_KINDS_SCORER` `gate_checks.py:554-565` | `"aux_drv": "built", "w_aux_drv": "built", "aux_drv_tap": "built", "aux_drv_depth": "built", "aux_drv_feed": "built", "aux_drv_shuffle_seed": "built", "aux_drv_labels": "data"` (+ `"aux_drv_lr": "built"`); `"data"` forces the path into the profile's `data_flags` (`RLS1_DATA` `launch_gate.py:54`) so the token fingerprints it |
| G-DVB rows | `scorer_dvb_rows` `gate_checks.py:650` | rows from the preflight `data` event's `aux_drv` block: `depth`, `cell`, `x0/y0`, `w_aux`, `tap` (**data_ptr equal** via `AUX.tap_is`), `feed` (**scorer width 60 vs 63**), store `sha256` == `AUX.PINNED_STORE_SHA256`, `frames == 103,139`, and the shuffle map a derangement |
| G-DVB (train.py) | `DVB_KINDS` `gate_checks.py:372-384` + `dvb_rows` `:401` | the same seven dests; `test_launch_gate` S5 pins this table equal to the parser dests (55 today) -> the pin moves with it |
| G-LIVE | `check_live` `gate_checks.py:759`, `zero_loss_control` `:737`, profile `live_terms` `launch_gate.py:58,76,99` | `live_terms += ("aux_bce", "aux_huber")`; head params non-zero grad; **`w_aux_drv = 0` -> every gradient exactly 0.0** (control, test C2); initial `aux_bce` ~ ln2 +- 0.15; aux-only backward non-zero on LoRA/`scene_proj`/`pos3d_mlp` and **exactly 0.0 on `traj_head`/`reg_compress`** (S2 only; tests C4); grad-norm ratio reported |
| G-LIVE (alignment, leakage) | preflight (`scorer_finetune.py:742+`) | >= 64 real frames: raster verdict vs `navsim_dac.violations` >= 95 % on shifted rank-0 targets, mutations "flip y" / "off by one frame" must turn it RED; `store keys ∩ navtest tokens = ∅` (MEASURED 0 of 12,146, test E10); coverage >= 99.9 % reported (store: 99.856 %, 149 missing, a missing label contributes exactly 0, test E5) |
| G-EVAL | `check_eval` `gate_checks.py:827` | H1: with the head absent the planner's trajectories and scores are **bit-identical** (test D1 is the unit form); mutation "wire the feed with the flag off" -> FAIL |
| G-CKPT | `check_ckpt` `gate_checks.py:952` | head params + Adam state round-trip; 2 resumed steps bit-identical; mutation: drop the head's optimiser state |
| G-CLOCK | `check_clock` `gate_checks.py:1033` | the label of a micro-batch is looked up by the same `(log_name, token)` as its images; mutation: shift the lookup by one sample -> alignment check FAILS |
| **G12 (new, scorer preflight)** | `SCORER_GATES` `gate_checks.py:570`, trainer `gates[...]` `scorer_finetune.py:865` | `aux_drv_live_and_identity`: (a) with the head on and the feed off, **2 optimiser steps leave every scorer parameter bit-identical to a head-off run** (separate clip, separate optimiser: step 9); (b) feed on + zero init: step-0 scorer output == base; (c) `torch.autograd.grad(score, head.parameters(), allow_unused=True)` all `None`; (d) the `probe_zerograd3.py` pattern at the end: every head parameter has Adam state |
| G-LIVE-MEM | preflight `gpu_mem_gb` `scorer_finetune.py:915` | head adds <= 1 GB at the run's batch; ⚠ on the 8 GB dev box with another agent's job resident this is the first thing that can fail (UNVERIFIED: the head's attention is `nn.MultiheadAttention(need_weights=False)` -> SDPA; on CUDA bf16 that is flash/mem-efficient, on a fallback it materialises ~157 MB per sample per block) |
| profile | `PROFILES` `launch_gate.py:63-105` (`refe_rls1` `:97-105`, `_BASE` `:58`) | new `refe_aux_s1` cloned from `refe_rls1`: `required_flags` **names** `--aux-drv`, `--w-aux-drv 0.1`, `--aux-drv-feed on` (a flag whose default disables it must be in the argv, or the arm did not test it); `forbidden_flags` + `--aux-drv-shuffle-seed` for non-control arms |

## 5 · The loader change (both hosts)

The store key is `(log_name, token)`; bank rows carry both (`train.py:309`, `scorer_finetune.py:91` `key(r)` adds `step, rank`; **the label key must NOT
include them** — goal twins and ranks share one label, SPEC 3.1). Do the lookup **in the DataLoader worker**, not the main process: a dataset wrapper
(`AuxSubset(IndexedSubset)` in S1, an extra element in `TargetBank.__getitem__` in S2) calls `src.labels([(r["log_name"], r["token"])])`
and appends `(label_int8 [160,128], valid bool)`. `AuxLabelSource` is picklable (E12), keeps one shard per log (E4), and `max_logs=64` bounds a worker's cache
to ~19 MB (MEASURED mean shard 295 KB: 351,383,740 B / 1,192). The held-out loader (`scorer_finetune.py:621`) needs the same wrapper or the perception proof has no labels.
**Held-out logs are navtrain logs**, so they ARE in the store (the store is navtrain-only; navtest logs are absent by construction, E10).

## 6 · Decisions for the Master Mind / PI (not mine to take)

| id | question | recommendation |
|---|---|---|
| D-W1 | the head is a fresh 2.44 M-parameter module; `scorer_finetune`'s `--lr 3e-5` (`:388`) is a fine-tune rate and would barely move it in 400 updates | a **7th flag `--aux-drv-lr`** (own warm-up, default 2e-4, the P6 rate), in `AUX_IDENTITY_ARGS` and `DVB_KINDS_SCORER`. Alternative: a two-stage S1 (S1a head-only on frozen u400 features -> perception proof; S1b scorer with the trained head frozen) — cleaner attribution, one extra run |
| D-W2 | `--max-updates 400` (`:927-929`) makes the cosine span 400 updates, whereas the banked **u400 is update 400 of a 2,335-update schedule** (lr there is ~0.94 of peak, MEASURED arithmetic from `:930`: 0.5(1+cos(pi*350/2285))) | all S1 arms use the same `--max-updates 400`; "vs the banked u400" is a **reference row only**, attribution is vs the **no-head replicate** (same schedule, same seeds) |
| D-W3 | `scorer_finetune.py` has **no resume** (`grep -ci resume` = 0; it saves `scorer_states_last.pt` `:970` but nothing reads it) | a 4-12 h run on a GPU shared under a lock protocol that dies with the lock restarts from zero: either add a resume (a change to a frozen file, post-ship) or run S1 in 100-update segments with `--max-updates` and re-launch |
| D-W4 | D: has 44.9 GB free (2026-10-13 08:22) | a frozen-trunk `sctx` cache would be 3.94 MB/set (7680x256 bf16) x 25,600 set-visits = **101 GB (ESTIMATED): does not fit**; do not plan on it unless the PI frees disk or accepts a smaller / pooled cache |

## 7 · S1 arm list (SPEC 5) and the dev-box cost

Init = `model_sft4_B_u400.pt` (md5 `8e382eeb…`, 7-output head: `--lane-head` + teacher-lane labels, `scorer_finetune.py:471-473`). All arms `--max-updates 400 --eval-every 400`,
same `--seed s` per pair so the data order (`g`, `:531`) is shared. **Paired by seed.**

| arm | what | seeds | note |
|---|---|---|---|
| **N** no-head replicate | arm B exactly as today, no `--aux-drv` | 3 (s0,s1,s2) | the run-to-run floor: H-ESTIM-SEED-1 says a separated paired CI is necessary, not sufficient; N-vs-N across seeds IS the floor |
| **H1** head, feed off | `--aux-drv --w-aux-drv 0.1` | **0 extra runs IF G12 (a) holds on CUDA** (scorer bit-identical to N, so the head trains INSIDE the N processes); otherwise 3 paired runs | gives the perception proof (held-out IoU / SDF-MAE per log) for free; see the CUDA caveat in section 8 |
| **H2** head + feed | `--aux-drv --w-aux-drv 0.1 --aux-drv-feed on` | 3 (same s0,s1,s2) | the lever under test; H2-vs-N paired per seed, read against the N-vs-N floor |
| **H2-shuf** (control) | H2 + `--aux-drv-shuffle-seed 7` (derangement of ALL keys, E11) | 1 first, 3 if H2 separates | separates "map information" from "3 extra features / extra capacity" (SPEC 9). Gate: shuffle arm is H1/H2 with a wrong map, so it is the one place `aux_refusals` allows shuffle+feed to be a deliberate control — **change `aux_refusals` accordingly if the PI wants it** (today it REFUSES shuffle with feed on) |
| ref | banked u400 scored by the same held-out + navtest harness | 1 | reference only (D-W2) |

**Cost per seed-run.** There is no banked dev-box SFT-trainer rate (`grep -rl s_per_update` finds only the two pod logs). Built from:
- A40, MEASURED: SFT-1 **11.41 s/update** (`raw/2026-10-04-sft1/sft.jsonl`, last 3 rows 11.4076-11.4088); SFT-3 **11.65 s/update** (`raw/2026-10-04-sft3/sft.log` done row: 2,733 updates, 8.8448 h); SFT-4 ~**14.4 s/update** (INHERITED, `STOP_CONSEQUENCES_2026-10-11.md:19`, no `sft.log` banked). 64 sets/update (defaults `--batch 8 --accum 8`, `:386-387`; `PREREG_RL_S1.md:199` also uses 64).
- 4060 vs A40: **2.475-2.559x slower, ESTIMATED from published spec ratios** (`TRAINING_TIME.md` section "The ESTIMATED term", not measured on this workload).
- ⇒ **28-37 s/update ⇒ 3.1-4.1 h for 400 updates**; two held-out evals (step 0 and update 400, 3,137 sets each, same trunk forward) add **0.8-1.0 h** ⇒ **3.9-5.1 h per seed-run** (ESTIMATED). The head adds 0.53-0.79 % of trunk FLOPs (SPEC 0, MEASURED+analytic): negligible.
- Upper bound from the one MEASURED dev-box inference rate: u400's navtest seam took **1,236 s for the 923-token read = 1.34 s/token** (`eval/RESULT_SFT4.md` "Timing"); x64 sets x 400 updates + the evals = **11.9 h** per run. UNVERIFIED how much of that is JPEG decode and PDM scoring rather than the trunk, so treat 11.9 h as the ceiling, not the estimate.
- **S1 total (N x3, H2 x3, H2-shuf x1 = 7 runs): 27-36 h (ESTIMATED) up to 83 h (ceiling), sequential, on a GPU another agent also holds.** ⚠ Fit on 8 GB is UNVERIFIED (ViT-L, 4 cameras, bf16, frozen, `--batch 8`): the first act is `--preflight`, which prints `gpu_mem_gb` (`:915`).

## 8 · Risks and what is UNVERIFIED

- **Bit-identity on CUDA is UNVERIFIED.** Every bit-identity claim of the module (D1, C2) is proved on the CPU. On CUDA, SDPA backward and atomics can be non-deterministic, and a same-seed rerun is a fresh draw (memory `same-seed-rerun-is-a-fresh-draw`). G12 (a) must therefore run in-process under `torch.use_deterministic_algorithms(True)` (or compare two same-process replays), and the "H1 costs nothing" row of section 7 stands only if it passes.
- **Nothing in this plan has been executed against the trainers.** The module's proofs run on a tiny REFe (vits16, 64x128) and on the real label store; the 2.44 M head at the real context size (7,680 tokens) was run forward only (A2, 2 samples, CPU). Memory and speed on CUDA at batch 8 are UNMEASURED.
- The S1 H2 head is trained *inside* the scorer run, so for the first updates the feed is noise from an untrained head. `feed_init="zero"` makes that harmless at step 0 and the pre-registration should say how many updates before the feed is read (D-W1's two-stage form removes the issue).
- `aux_refusals` currently refuses shuffle + feed (a wrong map into the scorer); the H2-shuf control needs that refusal relaxed deliberately (section 7).
- The 4 other `D:`-resident inputs of S1 (`--targets/--images/--calib/--onpolicy` banks) were not checked here; the pod ran SFT-4 with `/workspace` banks. Whether the dev box holds the on-policy sets for 400 updates (25,600 sets) is UNVERIFIED.

## Master Mind rulings on the open items (2026-10-13, after review; `test_aux_drivable_head.py` re-run: ZZTEST_PASS 68)

| item | ruling |
|---|---|
| head learning rate | **YES, a separate head LR flag** (the 7th flag). Its VALUE is fixed in `eval/PREREG_AUX_DRIVABLE_DRAFT.md` before the first S1 step, never tuned on PDMS. It needs its own G-DVB entry like the other six. |
| u400 as comparator | **Reference row only.** u400 stopped at update 400 of a 2,335-update schedule; attribution is head vs the matched **no-head replicate** from the same init and seeds. |
| no resume in `scorer_finetune.py` | **Accepted for S1** (dev-box runs of ~4-12 h ESTIMATED). A killed run restarts from zero and is logged as a restart; it is never stitched. |
| frozen-trunk context cache (~101 GB) | **Not built**: D: has ~45 GB free. Revisit only if the measured per-seed time exceeds the 11.9 h ceiling. |
| extent mask = `known` AND in-extent | confirmed |
| feed excludes t=0 (matches `field_loss`) | confirmed |
| shuffle control with the feed on | the refusal stays. A shuffle-control arm is added only by a dated prereg amendment that relaxes it deliberately. |

Order: wire into `scorer_finetune.py` (S1 host) first. RL-S1 already runs from its own shipped copy (`a120418`, `/workspace/refe-plan-rls1`), so repo edits cannot reach it; any RL-S1 re-ship must still pass `test_rls1_chain.py` and the launch gate with the head flags default-OFF.

