# 2026-10-17 H-JEPA literature package (stream S7) — banked primaries + T2 note
Library = `TanitAD Research Lab/Library` (`python tools/kb_add.py --verify`). Citing plan: `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`.
* `2609.34085` AD-E2E-JEPA — frozen DINOv3 ViT-L + SIGReg learnable projector, front-camera only; NAVSIM v2 **navtest stage-1** EPDMS 67.3/72.9 (84.1/86.5 w/o safety); IL 80.2 -> 85.4.
* `2609.26299` ForeDrive — JEPA latent world model coupled asymmetrically to a DiT planner; stop-gradient trains the predictor on forecasting loss only.
* `2502.14819` PLDM (Sobal et al.) — latent dynamics from reward-free offline data; per H-JEPA ref [73] combines IDM + variance-covariance regularisation.
* `2211.10831` Sobal et al. (H-JEPA ref [72]) — JEPA fails with episode-constant distractors (slow features); the case H-JEPA's IDM term answers.
* `2412.01718` HUGSIM (extra, for T2) — 3DGS closed-loop gym simulator; HD-Score; waypoints or (delta, a-dot) in, LQR + bicycle.
* `alpasim-egodriver-proto`, `alpasim-tutorial`, `alpasim-readme`, `hugsim-readme` (extra, local snapshots 2026-10-17) — AlpaSim driver contract, run recipe, Apache-2.0; HUGSIM repo usage.
* Already banked, tag `v8f-plan` + citation added only: `2608.20974` WA-JEPA, `2511.08544` LeJEPA, `2601.03460` FROST-Drive, `2411.04983` DINO-WM, `2506.09985` V-JEPA 2, `2603.19312` LeWM (ref [53]), `2604.03208` HWM (ref [92]).
* `T2_FEASIBILITY.md` — closed-loop options for the refcv family; recommendation: reuse the in-house gsplat harness for a first read, AlpaSim native for the G1 claim, skip HUGSIM.
