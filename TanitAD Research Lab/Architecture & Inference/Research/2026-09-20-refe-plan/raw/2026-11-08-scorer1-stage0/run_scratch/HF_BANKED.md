# SCORER1 Stage 0 run scratch, banked 2026-11-12 (Master Mind)

The small files of the Stage-0 chain (code, logs, markers, run records) from `F:/tanitad_scratch/stage0_20261108/`. The result itself was landed in 97b95a47: `STAGE0_RESULT.md`, `stage0.json`, `CHAIN.log`, `dump.log`. The pull provenance is in `scorer1_impl_20261108/stage0_pull/HANDOFF.md` and `LABEL_SOURCE.md`.

## Score dumps (on HF, not in git)
- Private repo `Sayood/tanitad-refe-scorer1p6`, path `evidence/2026-11-08-scorer1-stage0/`.
- Files: `scores_step0_A.npz`, `scores_step0_B.npz`, `scores_step0_base.npz`.
- Each is 11,997,626 B, with sha256 `37cf26b4930709f2ccdacf08528b234ddc6834e2e13903367aa017657caacc64`. The remote LFS sha256 was verified.
- The three are byte-identical. That is expected: they are step-0 scores, before any update, so arms A and B equal the base model (warm-start identity).

## Not banked (deleted 2026-11-12)
- `dump/partial_step0.npz`: the resumable mid-run checkpoint of a finished dump.
- `scorer_finetune.before_resumable.py`: a `.before_*` backup; the change landed in 3fc16530.
- `code/`: a copy of the refe tree, re-creatable from git.
- `data/` (pulled copies; sources re-verified 2026-11-12):
  - pod `/workspace/data/refe_navtrain/{onpolicy/sets,train_grow}` and `navtrain_pixels` (present);
  - HF dataset `Sayood/tanitad-refe-navtrain-banks` `refe_heldout_v5.tar.zst` (present, 65,517,846 B);
  - HF `Sayood/tanitad-refe-s1aux` `labels/s1_{heldout_pdm,heldout_lane,heldout_teacher_lane,train_teacher_lane}.tar` (present, sha256 prefixes c7bea660 / 606024c1 / abac4ea4 / 08ba5f01 match the recorded manifest);
  - `train_bank/`, derived by `build_bank.py` (banked here).
