# H-L7-1 PILOT-S -- RESULT: WITHHELD at all three training seeds (K7 miss); no outcome row is drawn

Filed 2026-10-23 07:15 Berlin (clock read 07:12). Runs on CPU, dev box: ts0 2026-10-22 21:08-21:30, ts1 infrastructure
retry 23:13-23:21, ts2 (the Amendment-1 WITHHELD re-run) 23:21-23:30. PI GO in chat 2026-10-22 ("Train PILOT-S now").
Registered reading: `../../PREREG_L7_SCORER.AMENDMENT-1.md` (sha256 `29dd3e84…`), sec. A2-A3, applied mechanically by
`read_pilot_s.py` (seeds 0 / 1, exactly as registered); the seed-2 re-run is read with the same script's `best_row`.

**Tier:** a selection diagnostic on the training split's held-out episodes (T1 self-action open-loop fans, 980 windows,
14 episodes x 2 inference seeds, `val_rows_sha12 5af23a842e87`) -- NOT driving performance; no sec. 3 bar is scored and
eval139 is never read. **Estimator:** none -- these are point values per training seed; nothing here is a lever claim.

## 1. The committed reading

| seed | run | best epoch (early stop on val loss) | pick (m) | scene-shuffled pick (m) | K7 `comf` AUC (>= 0.95) | finite |
|---|---|---|---|---|---|---|
| 0 | first run | 1 | 2.0163 | 2.9298 | **0.8692 MISS** | yes |
| 1 | infrastructure retry | 1 | 2.1059 | 3.1125 | **0.8713 MISS** | yes |
| 2 | Amendment-1 re-run | 1 | 2.2477 | 3.1376 | **0.8765 MISS** | yes |

References on the same 980 windows: own pick **0.99895 m**, in-fan oracle **0.35874 m**, L7-E **1.5892 / 1.5921 m**,
L7-F **1.071 / 1.163 m** (E/F PILOT, `../../pilot/RESULT_PILOT.md`).

**Outcome: WITHHELD.** A3 draws a row only when K7 holds at BOTH seeds. K7 missed at seeds 0 and 1
(`read_ts0_ts1.json`: `"outcome": "WITHHELD"`, `suspect_seeds [0, 1]`). A3 allows ONE re-run, training seed 2, no
hyper-parameter change; it ran and missed K7 too (`read_ts2_rerun.json`). With three of three seeds training-suspect, no
O1 / O2 / O3 row exists and none is drawn here.

Seed 1's first attempt died at epoch 0 on a 3.3 MiB numpy allocation, the Windows commit limit on a near-full C:
(`L7S_ts1/train.crash1.log`). Its identical-command retry is an INFRASTRUCTURE retry, not the A3 re-run
(`scripts/l7pilot_s_ts1_rerun.sh` says so in its header).

## 2. What the numbers show (DESCRIPTIVE -- not a committed reading, decides nothing)

* **Every seed's pick is worse than all three references**: 2.02-2.25 m vs own 1.00, L7-F 1.07-1.16, L7-E 1.59.
  The quantities A3 would have used are all negative: `g_own` -1.017 / -1.107, `g_E` -0.427 / -0.514 (seeds 0 / 1).
* **The scorer does read the scene**: the scene-shuffled pick is worse than the true scene by 0.91-1.01 m (`h`) at every
  seed, against a noise scale `nu` 0.0896. Control C-S1 reads at all three seeds.
* **It overfits at this bank size.** Validation LOSS is lowest at epoch 1 at every seed, so early stopping keeps epoch 1;
  meanwhile the training-subset pick falls below own pick by epoch 4 at every seed (1.277 / 1.238 / 1.177 m vs own 1.392
  on the training subset) while the validation pick stays 0.83-0.93 m worse than own (1.928 / 1.853 / 1.825 m at epoch 4).
  8,730 training windows from 125 episodes for 480,904 parameters.
* **K7 is not a seed accident**: the `comf` AUC never reaches 0.95 at ANY epoch of ANY seed (max 0.900, seed 2 epoch 4),
  while E and F reached it on the same windows (except F ts1 0.926). Adding the scene branch costs the comfort head about
  0.07 AUC at this bank size.

## 3. Next lever (Rule Zero)

The measured gap is generalisation (training subset beats own pick, held-out does not), so the lever with the largest
measured effect is **data**: the REGISTERED full-bank read (train4572 capture, about 6.2 Thor GPU-h), already queued on
Thor AFTER NT A1 by the PI's choice. S stays a registered arm of that read (PREREG sec. 1); it is not ranked by this
pilot because no row was drawn. **Blocked on:** Thor GPU, held by the NT A1 launch until A1 is training.
Not run here, and needing a registered amendment first (A3 forbids a hyper-parameter change on the re-run): a
regularised S (dropout / weight decay on the scene branch) to test whether K7's miss is capacity. It is cheap (CPU,
about 8 min per seed), but it cannot flip the held-out pick by 0.8 m at this bank size.

## 4. Files

| file | what |
|---|---|
| `L7S_ts{0,1,2}/train_log.json` | the trainer's per-epoch log (val, val_scene_shuffled, train_subset), unmodified |
| `L7S_ts{0,1,2}/train.log` | stdout (`ZZEP` rows, `ZZL7TRAIN`, peak working set / commit) |
| `L7S_ts1/train.crash1.log` | the first ts1 attempt's MemoryError |
| `read_ts0_ts1.json` | `read_pilot_s.py` on seeds 0 / 1 (the registered read) |
| `read_ts2_rerun.json` | seed 2's best row via `read_pilot_s.best_row` + K7 |
| `chain.log`, `scene_cache.log` | the run chain's markers; the scene-cache build |
| `scripts/` | the three run scripts as executed |

Not in git (size): the three `best.pt` -- md5 ts0 `0a0e5163266cf0254643031dc7967655`, ts1
`7658c4dc16b7bc6eec7ae2cfd5b5ed4b`, ts2 `f805284c1c9dbb93fa717e1aedf81883`; on the dev box under
`C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts*/`.
