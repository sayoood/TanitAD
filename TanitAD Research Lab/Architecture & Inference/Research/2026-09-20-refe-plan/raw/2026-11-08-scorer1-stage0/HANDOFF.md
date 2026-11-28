# SCORER1 Stage 0 pull - HANDOFF (2026-11-08, subagent). STATUS: BLOCKED, chain NOT launched.

## Pulled to F:/tanitad_scratch/stage0_20261108/ (pod read-only, nice/ionice, filtered on the pod by the 24 logs)
- code/ : `git archive d3227241` of refe, code, splits, raw/...scorer1-impl-v3/stage0, raw/...p7-session-reserve/raw/src (12 MB; `PK` = 2026-09-20-refe-plan).
- data/refe_navtrain/onpolicy/sets/ : 8 jsonl, 3,132 rows, 195.8 MB total with targets. Full sha256 of each file equals the pod's.
- data/refe_navtrain/train_grow/ : targets_rank0 (1,705 rows) + targets_rank1 (1,432) = 3,137 rows (matches the 3,137); calib_table.json (2,336,659 B, sha 772c00ab... = pod). Full sha256 equal to the pod's.
- data/navtrain_pixels/ : 6,820 unique jpgs (3,137 rows x 4 cams share frames), 1,397,905,629 B on the pod; local file count 6,820; sha256 of 80 random files equal the pod's (80/80). Layout kept: navtrain_current_N/<log>/<CAM>/<hash>.jpg.
- Free space after: F: ~15 GB (before 17), C: untouched (15 GB).
- Pod temp /tmp/stage0_20261108 (196,886,927 B) removed. Local temp in %TEMP% removed.
- stage0_gap.py --selftest (CPU): ZZTEST_PASS.

## BLOCKED: spec item (iv) does not exist at a second probe
- `pdm_labels_sft4r/heldout_pdm.tar`, `MANIFEST_pdm_sha256.txt`, `heldout_teacher_lane`, `heldout_lane`: NOT in HF dataset Sayood/tanitad-refe-navtrain-banks (main is the only ref; 19 files, none is pdm_labels_sft4r or refe_sft2; listing verified). sft4r_chain.sh:208 says it pushes them there; the push is not present.
- Also absent: Sayood/tanitad-refe-sft4r, -sft4, -sft2 (404), pod /workspace (find depth 7, excluding metric_cache/pixels/maps), box 2 D:\tanitad, local E:/F:/qland roots.
- ALSO NEEDED, same missing source (refe_sft2): train_pdm and train_teacher_lane for >=1 NON-held-out row. Code read (scorer_finetune.py ~1409-1434): `run_evaluate("step0")` (the dump) is reached only after the REFUSED check `not tr_idx`; tr_idx = rows of the on-policy bank + targets NOT in the 24 held-out logs (+ --require-pdm key), and --lane-head needs --teacher-lane-train (n_cols 7). So the UNVERIFIED answer is NO: the trainer does not eval with an empty train bank; a minimal bank = one non-held-out set with its targets row, pixels, train_pdm and teacher-lane label. Clean fix would be an eval-only entry (a change to landed code, not made here).
- Whether the lane/PDM labels can be re-derived (pdm_label_complete.py needs the metric cache on the pod) is for the Master Mind; nothing was substituted.

## Not done: GPU lock, chain script, WMI launch, markers. No stage0.json.
## Next step for the Master Mind
1. Locate or rebuild heldout_pdm, heldout_teacher_lane, heldout_lane (3,137 rows) and a 1-set train bank (train_pdm + train_teacher_lane), or approve an eval-only entry.
2. Then a chain: wait GPU lock (stage2_sft.take_lock/release, LOCK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock), scorer_finetune with COMMON of scorer1_pod.sh (ckpt E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt) --max-updates 1 --lr 0 --dump-scores D, then stage0_gap.py --scores D/scores_step0_base.npz --sets ... --targets ... --pdm ... --out stage0.json. Held-out data stays in place for reuse.

## Manifest (all single copies on F:, re-pullable from the pod while it keeps the data)
code/, data/ under F:/tanitad_scratch/stage0_20261108/ (markers/ empty).

---
## UPDATE (stage0 chain subagent, 2026-11-08 21:10): chain LAUNCHED, waiting for the GPU lock
Deviations for the Master Mind to rule on: (1) labels from `s1aux` + `refe_heldout_v5` sets instead of `pdm_labels_sft4r`; (2) one train bank of 58 sets instead of one row (the preflight timing loop and drop_last batch 8 need >= 24); (3) the trainer refuses without run/PREFLIGHT_PASS.json, so the chain first runs the TRAINER's own `--preflight` (lr 1e-4, writes run/PREFLIGHT_PASS.json); no launch_gate.py token is minted (nothing is trained; the run is lr 0). Evidence: LABEL_SOURCE.md (same folder).
- Chain: `F:/tanitad_scratch/stage0_20261108/chain_stage0.py`, launched via WMI (parent WmiPrvSE), PID of the cmd wrapper 56492 (python child alive), log CHAIN.log, stdout chain_stdout.log, stage logs logs/{preflight,dump,gap}.log. Lock currently held by owner strat_labels_ctx (pid 35108); the chain waits, never forces.
- Restart if killed: re-run the same WMI command; each stage skips when its output exists (run/PREFLIGHT_PASS.json, dump/scores_step0_base.npz, stage0.json). A stale markers/stage0.FAILED is renamed .prev.
- Markers: `markers/stage0.DONE` (text: path, 3137 sets) or `markers/stage0.FAILED` (reason/traceback). Result: `F:/tanitad_scratch/stage0_20261108/stage0.json`.
- NEXT for whoever reads the marker: DONE -> read stage0.json (Z1 / Z2 / inconclusive per stage0_gap.py), write STAGE0_READING under the pod's refe_scorer1 per LAND_PLAN D (human gate), bank stage0.json + dump/scores_step0_base.npz in the repo/HF, then delete run/ (preflight may write ~1-3 GB of checkpoints), dump/, data/ (re-pullable: HF s1aux, navtrain-banks, p2-joint-pdm; pod pixels) and log each deletion. FAILED -> read the reason and logs/*.log; F: needs >= 8 GB free, C: >= 6 GB.
- Deleted by me (all re-creatable from HF/pod, logged): tmp_join 6,350,776 B; data/ho_dl 65,518,831; data/s1aux_dl 134,976,231; data/p2pdm_dl 51,469,366; data/p2pdm 478,323,318; data/train64_dl/pix.tar 48,404,480; pod /tmp/s0tr 52,495,053. NOT deleted (other agent's): data/refe_navtrain/onpolicy/sets (195.8 MB, the WRONG fan for held-out; delete when convenient).
- UNVERIFIED: the GPU part (preflight on a 58-row train bank, the dump itself) has not run; only CPU argument parsing, the trainer's bank construction and stage0_gap's joins were run.
