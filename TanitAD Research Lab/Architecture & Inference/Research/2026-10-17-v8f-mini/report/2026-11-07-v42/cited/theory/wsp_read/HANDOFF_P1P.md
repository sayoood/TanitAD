# HANDOFF_P1P (P1′ agent, 2026-11-07). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/`. DONE; nothing staged, committed or pushed.

Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I` (never bare `python`), `CUDA_VISIBLE_DEVICES=-1`. Launch long jobs as a direct background command, not `nohup ... &` (the nohup form left two duplicate processes here). The first refit fold took 557 s (cold read of `V_pmove.npy` on F:), later folds 20 s.

## Verdict
**P1′ is NOT EVALUABLE as pre-registered.** After the window filter (v0 in [3, 8), a_human > 0.5) there are 168 windows from 45 episodes; the CONFIDENT group (posterior ≥ 0.8) holds **24** (floor 30), UNSURE **75**, at both seeds (the groups do not depend on the seed). Not padded, no threshold moved.

Descriptive only (below the floor): median ρ CONFIDENT 0.856 / 0.960 (s0 / s1), UNSURE 0.491 / 0.523; Δ +0.365 [0.213, 0.534] and +0.437 [0.240, 0.585], both separated. Direction as predicted, support too thin to read.

## Spec and controls
* `SPEC_P1P.md` sha `73640c67...` at 2026-11-06T23:16:43Z, before any eval posterior or ρ. `SPEC_P1P_AMEND_1.md`: controls N2/N3 were worded as verdict strings that the group-size gate makes unmeetable; read on the statistic instead (both pass). No criterion change.
* Probe: P-MOVE (only population overlapping v0 3–8), label GO, V+E linear, five fold models refit by a byte copy of `tp_lib.py`; posterior = mean of the five `predict_proba`.
* Pass: C4 unit tests; R1 refit reproduces the banked PCA and out-of-fold scores (≤ 6e-7); R2 all-train model gives eval AUC 0.7475 = banked; R3 eval V/E/GO/v0 bit-equal to the banked eval features; N1 shuffle (1 of 20, 0 of 20 separated).
* Probe AUC on eval-141: 0.750 [0.673, 0.825] (P-MOVE range), **0.782 [0.703, 0.853] in the band v0 3–6**; not at chance, so P1′ can test T1′ in principle.
* 50 of the 168 windows have v0 in [6, 8) (outside the probe's training range): 2 land in CONFIDENT.

## Sensitivities (not the verdict)
S1 in-domain [3, 6): 22 / 51 NOT EVALUABLE. S2 `plan_raw`: 24 / 75 NOT EVALUABLE. S3 all-train probe: 24 / 76 NOT EVALUABLE. **S4 GO24 posterior: 49 / 41, Δ +0.271 [0.144, 0.422] / +0.373 [0.201, 0.550], HOLDS at both seeds** (a different probe label; not promoted).

## Next (the Master Mind's call; nothing here needs compute)
1. Pre-register a dated variant with its own sha before reading it: the GO24 posterior as the go/stay probe, or a lower CONFIDENT threshold with a committed floor. S4 shows the first is evaluable.
2. Or add windows: a second held-out set with cached `pool4x10`, logged history and future, then a CPU forward of C0C at tick 0. UNVERIFIED that none exists beyond the stores read here.
3. Integrate: `WSP_READ.md` section 5 and `WSP_READ.json` (via `make_wsp_read_json.py`) still show P1′ NOT EVALUATED; update the row from `P1P_RESULT.md`. Not edited here.

## Files (hashes in `P1P_FILES.sha256`; each exists in this one place only)
`SPEC_P1P.md`, `SPEC_P1P.sha256`, `SPEC_P1P_AMEND_1.md`, `p1p_tp_lib.py`, `p1p_probe_refit.py`, `p1p_read.py`, `p1p_finalize.py`, `P1P_PROBE_MODELS.npz` (64 MB), `P1P_REFIT_R1.json`, `P1P_PERWINDOW.npz`, `P1P_RESULT.json`, `P1P_RESULT.md`, logs. Rerun: `p1p_probe_refit.py` (~12 min cold), `p1p_read.py` (~20 s), `p1p_finalize.py`.
