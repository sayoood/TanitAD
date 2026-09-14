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

## Part 10 — the on-video instruments (2026-09-14)

| script | question | result |
|---|---|---|
| `overlay_far.py` | on the DELIVERED video, how far is the corridor from the paint, in metres, out to the far field? | yaw −6.80 drift **0.21°**; yaw −7.80 drift **1.10°** (0.79 m at 40 m) ⇒ §47's refit WITHDRAWN |
| `lane_width_far.py` | camera height and horizon from lane width alone — **no focal length** | h = 1.586 m at the adopted horizon; separation histogram shows a 1-lane cluster and a 2-lane cluster at exactly 2× |
| `horizon_joint.py` | the flow horizon (438) and the paint horizon (470) disagree — settle it | joint zero **448**, and the LENS excludes the paint-only answer (it needs crop 0.93×) |
| `road_vp.py` | per-frame vanishing point ⇒ how much is EIS? | ⛔ **DID NOT COMPLETE** — 35 min, no output; HoughLinesP on dilated ridge masks is too slow. Banked unrun; the pooling argument in its docstring is still worth keeping. |
| `joint_fit.py` | joint (f·h, horizon, yaw, lateral) against the paint | ⛔ measured WORSE on the video; its objective uses the prediction-centred window that §49 grades at 22 % |

⛔ **Three association designs failed in this directory and all three are kept in
`overlay_far.py` with their measurements**, because each looked correct when written:
a per-range window centred on the prediction (absorbs 78 % of a known 1°), a local
tracker with a relative threshold (always finds a candidate, never reports a miss,
random-walks), and the working one (RANSAC line, global threshold, prediction used
ONCE as an identity gate — recovers 89 % of a known 1°).
