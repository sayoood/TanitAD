# REF-C medium (`base`, 104.2 M) — the D-030 scaling rung, trained with v2.1 route labels

**Date:** 2026-07-20 · **Status:** ✅ **COMPLETE at 29,999 and EVALUATED 2026-07-21** (outcome below;
full numbers in `Project Steering/MODEL_REGISTRY.md` §4.3 — the registry is the quotable source, this
note is the design record) · **Closes registry gap:** R7 (three-size scaling study never ran) —
partially: this is the **middle** rung.

---

## 0. OUTCOME (2026-07-21) — resolved against §0's pre-registered reading rule

The run finished 04:44 UTC 2026-07-21 at step 29,999 and was evaluated the same morning on
`tanitad-eval` through `taniteval.refc_eval`, on the **canonical 40-episode val build**
(`physicalai-val-0c5f7dac3b11`, n=881), at strict parity with XL — same windows, bit-identical GT and
bit-identical CV baseline in every stratum.

| | base (104.2 M) | XL (251.9 M) | paired Δ, episode-cluster bootstrap B=2000 |
|---|---|---|---|
| ADE@2s full-set | **0.4728** | 0.4714 | +0.0013 [−0.0281, +0.0316] **not separated** |
| FDE@2s | **1.0031** | 1.0061 | −0.0030 [−0.0619, +0.0584] **not separated** |
| miss@2m | **0.1419** | 0.1419 | +0.0000 [−0.0261, +0.0272] **not separated** |
| oracle-in-fan | 0.1914 (128 anchors) | **0.1640** (256) | +0.0275 [+0.0142, +0.0405] separated |
| oracle-in-fan, **matched 128-anchor vocabulary** | **0.1914** | 0.2624 | −0.0710 [−0.0965, −0.0502] separated |
| plan tick p50 fp32 | **21.78 ms** | 44.06 ms | 2.02× faster |

**§0's rule fires on the "level or better ⇒ AMBIGUOUS" branch, exactly as written.** The pre-registered
sentence stands verbatim: *"the 2.4× capacity cut costs little on 2,376 episodes, but a label change of
unknown sign rides along."* One correction to that sentence — the label change's sign is **not**
unknown for the quantity that matters here: on flagship v1.5 the v2.1 labels moved ADE **+0.025 m** but
**oracle −0.058 m**, i.e. they *helped the proposal set* by more than either oracle delta measured
above, and base is the arm that had them. So no oracle-based scaling claim is safe in either direction.

**What the run does establish, cleanly:** encoder scale is **not** the fan lever. base's 128 anchors are
a bit-exact prefix of XL's 256, and over matched first-K prefixes base's oracle is at least as good at
K = 4, 8, 16, 32, 64 and 128; XL's whole oracle advantage arrives with anchors 129–256. Anchors are a
buffer (0 params) and the decoder is ~1.7 ms of base's 21.8 ms tick, so **fan width is nearly free and
encoder width demonstrably bought nothing.** The clean resolution of the *scale* question is still one
control run: XL-with-v2.1, or base-with-v1.

---

## ⚠️ HEADLINE CAVEAT — read before quoting any medium-vs-XL number

**REF-C-XL trained with the OLD route labels; this run uses v2.1. A medium-vs-XL difference therefore
conflates SCALE and LABELS and is NOT a clean scaling measurement.**

Two further precisions that matter for how the comparison is written up:

1. XL's label set is **v1**, not v2. `refc_train.py` constructed `FailLoudWindowDataset` with
   `labels_v2` left at its default `False`, so XL's route aux target was
   `refb_labels.route_target(nav_command(...))` — the **circular** target (derived from the same
   quantity fed to the model as `nav_cmd`) that is also **straight-by-default** on unjudgeable
   windows. Any earlier statement that XL used "v2" labels is wrong; the code path never called
   `route_target_v2`.
2. The confounded term is **small by construction**: the route aux is one of six loss terms and
   carries **weight 0.1** against traj 1.0 / anchor-cls 1.0 / LAW 0.5. For calibration, the
   end-to-end label effect measured on flagship v1.5 was **+0.025 m (not CI-separated)**, though it
   did improve the proposal set (oracle −0.058 m).

**Reading rule agreed in advance:**
- medium **clearly worse** than XL (0.458) even *with* the better labels → strong evidence that
  **capacity matters** on this corpus (the confound works against that conclusion, so it survives).
- medium **level or better** → **ambiguous**. Do not claim a scaling conclusion. The honest read is
  "the 2.4× capacity cut costs little on 2,376 episodes, but a label change of unknown sign rides
  along". The clean resolution is an XL-with-v21 or medium-with-v1 control, which is one more 30k run.

---

## 1. Preset identity and MEASURED parameters

**There is no `medium` preset.** The ~104 M rung is `refc_config()`, selected by **`--config base`**,
named **REF-C-base** in code. Measured by `param_breakdown()` at instantiation on pod3
(`/usr/bin/python3`, torch 2.8.0+cu128) — the registry's "never instantiated, docstring estimate" note
for this preset is now closed:

| Preset | `--config` | encoder | decoder | anchors | imagination | **MEASURED total** |
|---|---|---|---|---|---|---|
| `refc_small_config()` | `small` | bw 64, (3,6,16,6) | d256 / 3L | 64 | off | **54,690,001** |
| `refc_config()` | **`base`** | **bw 88, (3,6,16,6)** | **d384 / 4L / 8 heads** | **128** | off | **104,191,577** ✅ |
| `refc_xl_config()` | `xl` | bw 124, (3,8,20,6) | d512 / 6L | 256 | on | **251,932,584** |

`base` per-module split (this run's `config.json` carries the same table):
encoder **90,458,632** · decoder **8,634,505** · strategic 1,903,680 · law 2,902,720 · aux 274,760 ·
measurement 17,280 · imagination **0** (graft off) · speed 0 (`refc1=False`).

So the ladder the registry advertises as **55 / 104 / 252 M is confirmed by measurement** — the
docstring's "~110 M" for `base` was 5.6 % high; **104.19 M** is the number. Ratio to XL: **2.42×**
capacity, and the cut falls mostly on the encoder (90.5 M vs 199.5 M, 2.21×) plus the whole H15
imagination field (21.0 M → 0, it is XL-only by design).

**GPU fit** (A40, 44.4 GiB, batch 20 / window 8 / 9-ch 256 px, 3 fwd+bwd+Adam iterations):
peak allocated **14.44 GiB**, reserved 18.31 GiB, ~1.29 s/step compute-only. Batch 20 / workers 6 kept
**unchanged from XL** — no memory-driven adjustment was needed.

---

## 2. Labels — v2.1, wired explicitly, ROUTE_UNKNOWN masked (never clamped)

`config.v2_labels` still selects v2, so REF-C's route path was wired **directly** to
`refb_labels.route_from_future_v21`, mirroring how flagship v1.5 minted `route_v21`.

Implementation (`stack/scripts/refc_train.py`, new `--labels {v1,v21}` flag, default `v1` so XL stays
reproducible):

- **`RouteV21Dataset`** subclasses `FailLoudWindowDataset` and overrides **only** the route aux target
  and its validity mask. **`nav_cmd` keeps the v1 derivation** — it is a model INPUT, and changing it
  would add a *second* confound on top of the label one. This mirrors `v15_prep.py`, which likewise
  left the fed command on the v1 path.
- **`use_net_dyaw=False`** (Sayed's ruling 2026-07-20: a wide sweep is ROAD FOLLOWING).
- **`ROUTE_UNKNOWN = 3` is masked out of the route CE**, never clamped to `straight`. The trainer
  additionally *raises* if an UNKNOWN ever survives the mask (`route<3 ⟺ valid=True` is the labeler's
  contract) and, when a batch contains no judgeable window, contributes route loss **0** rather than
  falling back to "train on everything" (that v1 fallback is preserved byte-identically under
  `--labels v1`).
- New logged/serialized keys: `route_acc`, `route_valid_frac` per log row; a `labels` provenance block
  and a `data` block in `config.json`.

**Provenance recorded in the run's `config.json`:**

```json
"labels": {"label_set": "v21",
           "route_derivation": "refb_labels.route_from_future_v21",
           "use_net_dyaw": false,
           "nav_cmd_derivation": "refb_labels.nav_command (v1, unchanged)",
           "route_unknown_handling": "masked out of the route CE (ROUTE_UNKNOWN=3, never clamped)",
           "train_label_stats": {...}}
```

**Measured label distribution over the training corpus** (4,000 sampled windows of 406,099, printed at
launch and stored in `config.json`):

| class | frac | | rule that fired | frac |
|---|---|---|---|---|
| left | 0.1210 | | `road_following` | 0.5645 |
| straight | 0.5645 | | `tight_transient` | 0.2360 |
| right | 0.1150 | | `gray_zone` (unjudgeable) | 0.1103 |
| **UNKNOWN (masked)** | **0.1995** | | `no_arc` (unjudgeable) | 0.0892 |

**80.05 % of windows carry a judgeable route target**, and the 19.95 % that do not are *excluded* from
the CE instead of being trained as `straight`. That is the substantive difference from XL's v1 target,
which was both circular with the fed command and straight-by-default.

---

## 3. Single-variable discipline vs REF-C-XL

| Held identical to XL | Value |
|---|---|
| corpus | `/workspace/pai_epcache/physicalai-train-e438721ae894` — 2,376 eps / **406,099 windows** |
| steps | 30,000 |
| optimizer | **Adam** (not AdamW), lr 1e-4, warmup 2000, cosine |
| loss weights | traj 1.0 · cls 1.0 · law 0.5 · route 0.1 · man 0.1 · speed_cls 0.2 (unused, `refc1=False`) |
| decoder mode | `diffusion` (2 truncated-denoise steps) |
| batch / workers | 20 / 6 |
| anchor build | FPS, horizons (5,10,15,20), pool 200,000, seed 0, same source corpus |

| Deliberately different | XL | medium (`base`) |
|---|---|---|
| scale preset | `xl` (251.9 M) | **`base` (104.2 M)** |
| anchor vocabulary | 256 | **128** |
| H15 imagination graft | ON (21.0 M) | **OFF** (preset design) |
| route label set | v1 (circular, straight-by-default) | **v2.1** (`use_net_dyaw=False`) |

**Anchor parity is exact, not merely "same method".** `refc_anchors_full.pt` (XL) is 256 anchors, so it
cannot load into a 128-anchor decoder. A new vocabulary was built with the same script, source, pool
cap and seed → `/workspace/experiments/refc_anchors_base128.pt`. Verified programmatically:

```
is strict prefix of XL vocabulary: True     # base128.anchors == xl256.anchors[:128]
```

FPS is a deterministic greedy sequence from a seeded first point, so the medium vocabulary is literally
**the first 128 entries of XL's 256** — the strongest available parity for this knob.

**Recovery recipe (de-risks the pod-only anchor file):** the vocabulary is reproducible two ways —
rerun `build_refc_anchors.py … --n-anchors 128 --seed 0` on the same corpus, or, with no corpus at all,
`torch.load("refc_anchors_full.pt")["anchors"][:128]`, which is byte-identical (verified above). The
anchors also travel inside the checkpoint as the `decoder.anchors` buffer, so **eval does not need the
file**.

### The run's `config.json`, as launched (the artifact describes itself)

```
cfg.encoder    {in_channels 9, image_size 256, base_width 88, blocks (3,6,16,6)}   -> feat_dim 704
cfg.decoder    {d 384, n_heads 8, layers 4, ff_mult 4, aux_hidden 384, diffusion_steps 2, noise_std 0.1}
cfg.anchors    {n_anchors 128, pool_size 4096, seed 0}      cfg.strategic {hidden 512, d_ctx 64}
window 8 · ego_dropout 0.5 · hierarchy true · graft_maneuver true · graft_imagination FALSE · refc1 false
optimizer      Adam (DiffusionDrive/TCP), lr 1e-4, warmup 2000, cosine
loss_weights   traj 1.0 · cls 1.0 · law 0.5 · route 0.1 · man 0.1 · speed_cls 0.2 (inactive)
data           /workspace/pai_epcache/physicalai-train-e438721ae894 — 2376 eps / 406,099 windows
```

---

## 4. Launch

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup setsid /usr/bin/python3 \
  scripts/refc_train.py \
  --data-root /workspace/pai_epcache \
  --out /workspace/experiments/refc-diffusion-base-v21-30k \
  --steps 30000 --mode diffusion --config base \
  --anchors /workspace/experiments/refc_anchors_base128.pt \
  --batch 20 --workers 6 --labels v21 \
  > /tmp/refc-base-v21-30k.log 2>&1 < /dev/null &
```

Host **tanitad-pod3** (A40 46 GB, idle before launch), **system `/usr/bin/python3`** — pod3's
`/workspace/venv` was upgraded to transformers 5.14.1 for VLM work and is not used here.
Launched 2026-07-20 ~18:40 CEST. Log at `/tmp/…` on purpose: logs written under `/workspace` get
swallowed when a run dies on a full MooseFS quota.

### Pod drift found and repaired before launch (the brief's warning was correct)

`git` on pod3 sits at an old commit; four files were checked by md5 against the repo working tree:

| file | pod state | action |
|---|---|---|
| `scripts/refb_labels.py` | **`use_net_dyaw` still defaulted to `True`** (pre-ruling) | synced from repo |
| `scripts/refb_train.py` | missing `labels_v2` + milestone archiving | synced |
| `tanitad/config.py` | missing the v2/v3 lever fields (additive, default-off; `TrainConfig` verified identical) | synced |
| `tanitad/train/ckpt_io.py` | **absent** | copied (needed for milestone archiving) |
| `tanitad/refs/refc.py`, `scripts/build_refc_anchors.py` | already byte-identical to repo | untouched |

Backups of the pre-sync pod copies: `tanitad-pod3:/workspace/ops/backup-20260720-refcmed/`.
`__pycache__` cleared after the sync. Note the run passes `use_net_dyaw=False` **explicitly**, so the
stale default could not have silently changed the labels — but the sync removes the trap for the next
agent.

### Milestone checkpoints (new)

`refc_train.py` previously only kept `ckpt.pt` (overwritten every 500 steps), which is why XL has no
clean 5k/15k/20k gates. Milestone archiving (`ckpt_step{5000,15000,20000,30000}.pt`, atomic via
`ckpt_io.atomic_archive`) was ported from `refb_train.py`, so **this run produces the D-030 gate
series**. Cost ≈ 1.25 GB per milestone (≈ 6.3 GB total with `ckpt.pt`).

---

## 5. Early curve — healthy, and **~10 h not 1–2 days**

Startup (2,376 mmap episodes + the label-stats pass) ≈ 4 min. Measured rate **60.1–62.9 s per 50 steps
→ 1.20 s/step** (`step_s` is ACCUMULATED over the `--log-every 50` interval — it is not 60 s per step;
this is the exact metric that produced a false alarm before). `data_s` ≈ 0.0–0.3 s per 50 steps, so the
6 workers fully hide the loader **including** the added per-item v2.1 label derivation.

**30,000 steps ETA ≈ 10.0 h → ~02:40 UTC 2026-07-21.** The brief's "1–2 days" was pessimistic; XL was
~3.4 s/step at 2.4× the parameters.

```
 step   loss     traj    cls     law     route  man    anch_acc route_acc rvf   gnorm    lr
    0  15.496   6.882   6.747   3.278   0.711  1.571   0.00     0.88     0.85   271.33  5.0e-08
  100   8.999   2.187   5.531   2.126   0.772  1.402   0.00     0.67     0.75   185.53  5.1e-06
  200   5.813   1.092   4.404   0.240   0.467  1.496   0.05     0.93     0.70   120.39  1.0e-05
  300   4.714   0.511   3.953   0.085   0.611  1.456   0.25     0.80     0.75    97.14  1.5e-05
  400   4.127   0.532   3.351   0.052   0.631  1.538   0.15     0.75     0.80    99.43  2.0e-05
  500   5.016   0.728   4.047   0.048   0.655  1.517   0.05     0.86     0.70   115.45  2.5e-05
  600   4.077   0.548   3.279   0.068   0.814  1.352   0.20     0.69     0.80   114.70  3.0e-05
  700   4.376   0.495   3.657   0.059   0.578  1.365   0.10     0.76     0.85   111.01  3.5e-05
  800   3.921   0.483   3.183   0.096   0.748  1.315   0.15     0.71     0.70    75.70  4.0e-05
  900   4.761   0.487   4.012   0.093   0.576  1.576   0.10     0.75     0.80    78.56  4.5e-05
```

Reading it:

- **traj-recon L1 6.88 → ~0.49 m by step 300** and flat after — the anchored decoder nails offset
  geometry almost immediately, exactly the XL-era pattern.
- **anchor-cls CE 6.75 → ~3.2–4.0**, i.e. below the 128-class chance CE of **ln 128 = 4.852**, and
  `anchor_acc` **0.00 → 0.05–0.25** against a 1/128 = 0.008 chance rate (batch 20 ⇒ 0.05 granularity,
  hence the noise). Selection is the slow term here too — consistent with the XL finding that
  *selection*, not geometry, is REF-C's hard problem.
- **LAW MSE 3.28 → ~0.06** — the latent aux converges fast.
- **The v2.1 masked route CE is training, not degenerate:** loss 0.71 → ~0.58–0.81 with `route_acc`
  0.69–0.93 on `route_valid_frac` 0.70–0.85 of each batch (matching the 0.8005 corpus figure). No
  `ROUTE_UNKNOWN survived the valid mask` raise — the mask and the labeler contract agree.
- `gnorm` is the **pre-clip** norm and sits ~75–120 against `clip_grad_norm_(…, 1.0)`, so clipping is
  active on every step. That is inherited from the trainer and identical to XL's setup, so it does not
  affect the comparison — recorded here only so nobody re-discovers it as an anomaly.
- LR is still in warmup (2,000 steps) through this whole window.

**Operational checks passed:** first `ckpt.pt` written at step 500, **1,250,838,325 B (1.25 GB)** as
predicted; a 4 GB `dd` probe still succeeded afterwards. Cgroup memory reads 45.5 / 46.6 GiB, which
looks alarming but is **not**: `total_rss` is only **6.2 GiB**, the other 38.5 GiB is reclaimable
`mapped_file` page cache from the mmap'd epcache, and `memory.failcnt` is **0**. Watch `failcnt` and
`total_rss`, not `memory.current`.

---

## 6. Eval plan (when the run reaches 30k) — handoff

Canonical protocol, identical to XL's: **`taniteval`, n = 881 windows, 40 val episodes, 8-split
jackknife**, compared against **REF-C-XL final `0.458 ± 0.057` ADE@2s** (`results/refc-xl-30k.json`).

1. **Move the ckpt** pod3 → `tanitad-eval:/root/models/refc-base-v21-30k/ckpt.pt`. Pods cannot SSH each
   other directly; XL's final was moved over the agent-forwarded path at 18.2 MB/s (≈ 70 s for 1.25 GB).
   The HF relay is the fast fallback. **md5-verify both sides, and only copy after the trainer exits**
   (a live `ckpt.pt` can be torn).
2. **Register** in `taniteval/registry.py` — the loader already resolves presets, so the only new key is
   `config_preset="base"`:
   ```python
   dict(key="refc-base-v21-30k", name="REF-C-base (anchored-diffusion, 104M, v2.1 labels, 30k)",
        family="TanitAD", arch="refc", config_preset="base", mode="diffusion",
        ckpt="/root/models/refc-base-v21-30k/ckpt.pt", config="refc-base",
        encoder="trained ResNet (9ch, 256px, base_width 88)", encoder_frozen=False,
        speed_input=True, action_dim=2, hf=None, anti_collapse="trained encoder",
        note="… 128 FPS anchors = strict prefix of XL's 256 … route labels v2.1 (use_net_dyaw=False), "
             "XL was v1 → scale/label confound …")
   ```
3. **Run**: `python3 -m taniteval.runner run --model refc-base-v21-30k --episodes 40`
   then `python3 -m taniteval.runner ab --a refc-base-v21-30k --b refc-xl-30k`.
4. **Proposal-quality panel** (the lever that actually matters per the XL section):
   `taniteval/plan_fan.py` → **oracle-in-fan** and **`frac_sel_2x_worse`**, against XL's
   selected 0.4714 / oracle 0.1640 / gap 0.3075 / frac_sel_2x_worse 0.454 (full-set corpus figures).
   A 128-anchor fan is half XL's width, so **oracle-in-fan is expected to be worse purely from fan
   width** — that is a coverage effect, not a ranking effect, and must be stated when the two are
   compared.
5. Optional but cheap, and the reason the milestones exist: evaluate `ckpt_step5000/15000/20000` to get
   the medium arm's gate series.

---

## 7. Watcher note (for the loop)

- **Process:** `tanitad-pod3`, `ps -eo pid,args | grep "[r]efc_train"` (bracket trick — never
  `pkill -f refc_train`). Log `tail -f /tmp/refc-base-v21-30k.log`.
- **ETA:** step 0 at 16:38 UTC 2026-07-20, 1.20 s/step ⇒ **30 k at ≈ 02:40 UTC 2026-07-21**. Milestone
  ckpts land at ≈ 18:20 (5 k), 21:38 (15 k), 23:19 (20 k) UTC.
- **Health signals:** `loss` and `traj` falling; `anchor_acc` above 0.008 (chance) and rising;
  `route_valid_frac` ≈ 0.8; `gnorm` finite (pre-clip, ~75–120 is normal here). A
  `ValueError: ROUTE_UNKNOWN survived the valid mask` means the labeler contract broke — that is
  intentional fail-loud, report it, do not patch around it.
- **Memory:** cgroup usage sits pinned near the 46.6 GiB limit **by design** (mmap page cache). Alarm
  only on `memory.failcnt > 0` or `total_rss` climbing past ~10 GiB (`/sys/fs/cgroup/memory/`).
- **Quota:** `/workspace` on pod3 is the shared MooseFS volume. A real 10 GB `dd` write succeeded before
  launch (≥ 10 GB free); `df` lies (it shows the 965 TB cluster). Re-probe with `dd`, never `df`. If it
  tightens, the reclaimable candidates are `/workspace/tmp/ijepa_feats` (49 G) and
  `/workspace/tmp/dino_feats_320` (30 G) — **REF-A feature caches, not mine to delete: ask first.**
- **Restart recipe:** the trainer auto-resumes from `ckpt.pt`; relaunch with the exact command in §4
  (`ssh -f`, `PYTHONPATH=/workspace/TanitAD/stack`, system python3).
- **Do not touch pod1** (flagship v3enc training).

---

## 8. Deliverables

| Artifact | Location | Copies |
|---|---|---|
| `--labels v21` wiring + `RouteV21Dataset` + masked route CE + milestone archiving | `stack/scripts/refc_train.py` (staged) | repo + pod3 |
| This note | `TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md` | repo |
| 128-anchor FPS vocabulary (prefix subset of XL's 256) | `tanitad-pod3:/workspace/experiments/refc_anchors_base128.pt` | ⚠️ **single copy (pod-only)** |
| Run dir (`config.json` w/ label provenance + measured params, `ckpt.pt`, milestones, `metrics.json`) | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/` | pod3 + **final `ckpt.pt`/`config.json`/`metrics.json` mirrored to `tanitad-eval:/root/models/refc-base-30k/`** (md5 `8f10d6f934f4199e11ddc7352e074939`); the 5 k/15 k/20 k milestones remain ⚠️ **pod3-only** |
| Eval artifacts (2026-07-21) — canonical row, per-window predictions, full 128-proposal fan, scale A/B | `taniteval/results/{refc-base-30k.json, windows_refc-base-30k.pt, fan_refc-base-30k.pt, scaleab_refc-base-30k_vs_refc-xl-30k.json, eff_refc-base-30k.json}` | repo + eval pod |
| Scale-A/B driver (coverage-matched oracle + paired bootstrap) | `taniteval/refc_scale_ab.py` | repo + eval pod |
| Training log | `tanitad-pod3:/tmp/refc-base-v21-30k.log` | ⚠️ **single copy, `/tmp`** |
| Pre-sync pod file backups | `tanitad-pod3:/workspace/ops/backup-20260720-refcmed/` | pod-only |
