# MM rulings on the Option B read chain (2026-11-18 ~01:35Z, before any fine-tune checkpoint exists)

Build `BUILD_READCHAIN.md`. Read side `../read/` (landed 0669ec8f). Card `../AMEND_OPTB.md` (frozen).

1. **The loading route is accepted.** The planner's model loader (`wmact_lib.load_model`, ladder-tree config) refuses a fine-tune checkpoint (`TypeError: unexpected keyword 'wm_gamma_levels'`), MEASURED on a real FT-format checkpoint. `optb_route.py` loads that checkpoint strictly from the optb tree (= 4c396aff) and keeps the probe sha256 check. Two things show the route changes no number:
   - R35_s0 rollouts are bit-equal between the ladder tree and the route (`test_route.py`, 9 passed);
   - a wrong declared sha, a tampered probe and an unknown basename all refuse.
2. **BAND-MET is the card's meaning.** PREREG_V8F_ACTRSP line 92 defines BAND-MET as IA-0 skill against Copy-Last at the L1 2 s, L2 6 s and L3 12 s band ends, read on every arm. "Lost at a level" means R35 met the level and the arm does not. The chain computes this from the ladder read, with the rule of `verdict_actsens.py:87-91`.
3. **One score-windows file for both chassis is accepted.** MEASURED by the MM: `actsens/read/out/C7_R35_s0/win/windows.npz` and `lonplan/score_windows_s1/windows.npz` are equal on `eid`, `now`, `A`, `v0` and `donor` (994 windows each). They differ only in `z` (each chassis' own latents, which the planner takes from its own chassis) and in `meta`. So the decide step scores the same windows at both chassis. The chassis-1 planner reads its own file, as the score chain of record does.
4. **Launch.** The MM launches `optb_readchain.cmd` by inline WMI (CreateFlags 0x08000200, `< NUL`); `powershell -File` is blocked by execution policy on this box. The chain is launched after the box-2 chain starts. It polls for banks by itself and is waited on by its sentinel files only.
5. **Land.** The code, tests and docs are landed. `fixture/` (a 128 MB FT-format checkpoint, re-creatable by `make_ft_ckpt.py`) and `tmp/` (the dry-run trees) are not landed. They are deleted after the dry run is read and its result is recorded in `BUILD_READCHAIN.md`.
