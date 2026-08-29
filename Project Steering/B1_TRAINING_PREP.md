# B1 TRAINING PREP — the four-consumer training on the Alpamayo-labelled corpus

**PI directives 2026-08-29** (both first-hand/aligned): (1) to MM — "the data fly wheel
will start now the production of the training data … corresponds to the alpamayo labeled
data … app. 26h … 120 front camera, ego data and strategic/tactical labels …
prepare on your side the training." (2) to DataFlyWheel — release the dataset for
**v7f, refcv3, refav1, refd**; the alpamayo + tactical/strategic-labelled data is
MANDATORY to complete the training corpus.

## 0. Corpus identity + comparability (register row D-CORPUS-B1, pending G: recovery)

- New corpus = 4,719 clips (alpamayo ∩ egomotion), ~26 h; identity = the DataFlyWheel's
  MANIFEST corpus hash (sha256 over sorted clip ids) + label blob lineage (121a8d93 at
  directive time).
- ⚠️ NEW COMPARABILITY DOMAIN: arms trained here are never same-data-compared to
  `physicalai-train-e438721ae894` parity arms. The old parity rule stays intact WITHIN
  its domain (tiny ladder, existing arms).
- No training launches before the delivered manifest is banked + verified (md5/sha256,
  content assertions per the decode-into-memmap rule: sample rows, non-zero, means).

## 1. Intake pipeline (MM side, blocking)

1. Pull the HF-private bundle (labels + records + egomotion store + retrieval index +
   MANIFEST + DATACARD); verify per manifest.
2. **Episode-cache build** (the trainers consume `--v2-cache` epcaches, w120 256×640
   cylindrical): build from camera chunks (retrieved by the bundle's range-reader tools,
   ~47 GB) + egomotion. ⚠️ Per-clip `cy` rig handling REQUIRED (two rigs, cy~543/cy~755 —
   geometric-center crop is ~215 px wrong for rig B). Skip-list = manifest exclusions
   only; emit the cache's own skip-hash into the registry row.
3. Build location: Thor (its disk + the training happens there) after `emao14_30k`
   finishes (~17:45); the dev box holds the bundle mirror.

## 2. The canonical label consumer (ONE module, all four trainers)

`stack/tanitad/data/v7_labels.py` (new): loads s2_labels_v7 schema and implements the
D-LABEL-GT conditions ONCE — decided-once-consumed-everywhere, the same pattern that
fixed the two-detector class:
- mask the 8 `NOT_YET_EXTRACTABLE` classes (head-width stays full-vocab; loss-mask);
- strategic-skew weighting;
- `disputed` policy (exposed as a filter arg; default = include with flag);
- `time_basis`/`t_nominal_s` band placement (untimed → band midpoint per PI decision);
- `turn_suppression` records passed through (audit, not training input).
Tests: extend the MM-C1-era s2 suites with a v7-set fixture (the named pending item).

## 3. Per-consumer readiness

| consumer | state | prep items |
|---|---|---|
| **v7f** | recipe settled (§5.1 + R2; EMA pending tonight's read) | epcache; config-E geometry sanity (build smoke at scale); staged plan: S-W trunk first, upper stages on labels via §2 |
| **refc_v3** | v7-vocab heads wired; trainer refuses width mismatch | swap kin3 labels → §2 consumer at `--tac-vocab-version v7.0`; preflight |
| **refa_v1** | v7-vocab heads wired | label join via §2; trainer run-config |
| **refd** | v7-vocab heads wired; **model never trained** | trainer script exists? verify; label join via §2 |

## 4. Gates before each launch

- D-B1-GATE (quantisation) applies to DEPLOYMENT validation, not these training runs —
  unchanged, still blocks the quantised-production step only.
- Per-run: prereg with committed outcomes; verify-ship-by-content; done-marker in the
  same turn; four-family eval plan at T1 with the echo control.
- Schedule: Thor freed ~17:45 → EMA read → epcache build overnight → first launch
  after the PI sees the EMA verdict (it decides the v7f recipe's final flag).
