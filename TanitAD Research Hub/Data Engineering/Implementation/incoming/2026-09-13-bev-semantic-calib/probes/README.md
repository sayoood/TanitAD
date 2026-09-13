# Probes — the throwaway scripts behind quoted numbers

Every number quoted in `BEV_SEMANTIC_CALIB.md` Part 3, in the `R-2026-09-13-horizon`
retraction, and in the docstrings of `flow_scale.py` / `lag_scale.py` came from one of
these. They are banked because **a measurement whose script lives only in one agent's
scratch directory is not reproducible, and this programme has paid for that repeatedly.**

They are deliberately NOT polished into modules. Each answers one question and stops;
promoting them would invite them to be re-run as if they were instruments.

| script | the question | what it returned |
|---|---|---|
| `realprof.py` | does the BEV lag correlation peak at the true displacement on real ink? | **No.** Monotone decay from `r = 0.211` at lag 0; `r = 0.062` at the truth. This is why `lag_scale.py` refuses. |
| `dvrow.py` | does tracked row flow follow the ground-plane model, and where does it stop? | Rows 580–830 follow it (730–780: 27.46 measured vs 27.27 predicted); below 830 flow collapses to ~0 while the model says 59–135 px, and the track rate falls to 15–20 %. **Fixed the row band at 0.77 H.** |
| `rowfit2.py` | horizon + `f·h` from per-point `A = D q q'/(q'−q)` | zero-trend horizon **439.3 px**, `f·h` **2664**, frame-cluster bootstrap [432.8, 446.0] / [2533, 2818]. |
| `bias.py` | is the fit driven by LK selection (slow points survive)? | **Refuted.** Track rate 61.8 → 24.8 → 8.8 % across 1/2/3-frame steps; horizon holds 423.7 / 432.5 / 426.6. |
| `delta.py` | is there a per-pair constant row offset (pitch jitter / EIS)? | **Not supported.** De-meaning made the cost *worse* (1.348 vs 1.340) and gave a degenerate 120 px horizon. |
| `hzslab.py` | which horizon makes lane width range-independent? | **~485 px** (far/near 1.045), vs **1.415 at 523.4**. Half of the retraction's evidence. |
| `lat.py` / `lat2.py` | residual lateral offset, and can it be split from yaw by range? | Midpoint **+0.190 m** [+0.115, +0.220]; the range split is **internally inconsistent** (a constant `lateral` change did not shift slabs equally), so it is not quoted. |
| `yawlat.py` | same split, via whole-profile correlation that identifies no peaks | Locks in only **14–18 of 150 frames** — the far field lacks ink. The measured case for a stronger segmenter. |
| `verify.py` | does BEV agreement — which never saw the flow data — rank the answer first? | **Yes, +59.4 %** over shipped vs +11.1 % for the earlier "best". Output in `../raw/final_verification.txt`. |

Run them with `PYTHONPATH=<stack>/tanitad/data` and the `trajlib` alias registered — see the
header of any one of them.
