# HANDOFF — T2-A (S8). State at 2026-10-17; read this, then act.

**State:** the CPU dry path and the box-2 kit are DONE and tested. No GPU run has happened; no T2 number exists. Read `RESULT.md` (verdict + manifest), then `RUN_ON_BOX2.md`.

## Exact next step (needs box 2, a Windows PC with an 8 GB NVIDIA GPU)
1. Copy `C:/Users/Admin/t2a_scratch/box2_kit/` (2.96 GB (≈ 2,958,400,000 B; `KIT_MANIFEST.json` lists every file exactly)) to box 2. Rebuild if gone: `python code/make_kit.py <dest> --assets --control --md5` (needs the `E:` sources listed in `INVENTORY.md` §2 and the git object `fec3a0dc`).
2. On box 2 (x64 Native Tools prompt, venv with CUDA torch, numpy, msgpack, `gsplat==1.5.3`, ninja): `python t2a\preflight_box2.py --kit . --out preflight.json`. 0 HARD failures ⇒ go. If gsplat will not build on Windows, run the same kit in WSL2 Ubuntu (BLOCKERS B2). If "(c) fit in VRAM" fails, STOP and report (B3) — do not change render flags for some arms only.
3. `python t2a\run_on_box2.py --kit . --out results --scenes 00040136 --arms cl_ha0,cl_p10_t0` (smoke; `cl_ha0` must print `[ok]` = exact-zero steer/accel), then the full grid without `--arms` (add `--with-control` for the refc-base repro control). Resumable. Spawn a fresh agent for scoring; do not hold one open for the run.
4. Bring back `results/`, `preflight.json` to a new `incoming/<date>-t2a-closedloop-box2/raw/`. Read it with the gates in `RUN_ON_BOX2.md` §6 — committed before any number exists. A floor that beats refcv7 on a family is a FAIL on that family; then Rule Zero: the four next levers are listed there, in order.

## For the orchestrator (integration — do not leave this in a README)
- LAND: `stack/experiments/alpasim-gsplat/closedloop_drive.py` (unstaged patch, +35/-1, inert for historical arms: 29 old tests pass), `taniteval/taniteval/floors_kin.py` (UNTRACKED), and this package directory.
- The working-tree branch `agent/arch-inf-20260803` CANNOT build refcv7; the kit pins `fec3a0dc`.
- The 2.96 GB kit exists only under `C:/Users/Admin/t2a_scratch/box2_kit/` (C: only).

## Open questions (0-GPU, any agent)
- Nav equivalence: the harness feeds nav derived from the logged route; refcv7 trained on the v7.2 token. Equivalence UNVERIFIED (BLOCKERS B5).
- Wire `xodr_map.py` into the metrics to get an off-road family (B8).
- **Discriminating ceiling control (B5, ≈ 7 min CPU, background it):** does a SATISFIABLE binding ceiling move the emitted refcv7 plan on `fec3a0dc` (SPEC 26.1 says no) and on `0fcb1c1c` (says yes)? The 5 m/s control used so far has no compliant candidate, so the empty-survivor fallback makes it uninformative (`refcv6_selection.py:187-192`). Needs a stub world at a speed just under a bin edge (limits 8.33 / 13.89 / 27.78 / 33.33 m/s). The test and the `tree_r4/` are in place; `STUB_V` in `stub_gsplat_renderer.py` is the knob.
- Run the full 3-tag slow test once (`T2A_SLOW=1`, ≈ 11 min): its components passed separately.

Constraints that stay: CPU-only here (`CUDA_VISIBLE_DEVICES=-1`, assert no CUDA; the RTX 4060 belongs to another session); no contact with Thor; staged data < 15 GB on `C:/Users/Admin/`; STRATEGIC family reported PAUSED (PI item 25); every number carries tier T2 and the scope sentence *the loop is closed on perception, not on interaction*.
