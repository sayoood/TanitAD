# C1 read: SEARCH on P2's own scorer, B-REST (banked by the Master Mind, 2026-10-20)

**Outcome: `O1_CONFIRMED`** under the sealed `PREREG_SEARCH_P2_CONFIRM.md` (sha256 `658ba157…`), chain `spc_launch.py` (`ZZSPC_DONE O1_CONFIRMED`, `RESULT_SPC.json`).

**Stamps (from the chain):**
- MEASURED. SELECTION tier, label-only (the PDM label-only scorer), so it is **NOT T1 and NOT navtest**.
- SEEN-SCENE: all of B-REST is inside M-SCENE (M-PDM overlap 0). TOKEN-FRESH to the search; LOG-SHARED with Step A and carve B (the same 24 logs), so it is NOT a fresh-log test.
- ONE training seed (P2-s0), no head substitution (H-ESTIM-SEED-1 is open).
- The interval is the paired log-cluster bootstrap (B = 10,000, seed 20260927). It answers "another draw of LOGS" only. The 3-search-seed spread is reported beside it.
- Read alone, never pooled with carve B. STRATEGIC is PAUSED.

| quantity | value |
|---|---|
| tokens / logs | 905 / 24 (B-REST bundle sha256 `7b7dfff3…`) |
| dPDMS, search output vs P2 pick | **+0.5575 [+0.1058, +0.9253] points** |
| per search seed | +0.632, +0.535, +0.505 (range 0.128) |
| safety: dNC / dDAC / dTTC / dC | 0.000 / +0.331 [0.000, 0.734] / +0.258 [-0.310, +0.767] / 0.000 |
| dEP (LONGITUDINAL) | +0.506 [+0.048, +0.949] |
| dDDC (LATERAL) | 0.000 |
| moved share `inert.frac_moved` (ADE > 0.5 m vs pick) | **17.3 %** (per seed 17.5 / 16.7 / 17.8 %; n 2,715), not INERT |
| TACTICAL proxy: command-class change share | 0.66 % (HYPOTHESIS proxy) |

**The C2 licence (`MM_RULING_C2_LICENCE.md` §2, sealed before C1 produced anything) reads MET on all three terms:**
1. O1.
2. Every safety point is at or above −0.25.
3. `frac_moved` 17.3 % ≥ 5 %.

The other §2.5 prerequisites still stand before any navtest token:
- the sealed C2 amendment;
- the deployment-target latency, MEASURED or named UNMEASURED;
- the carve-B fixture reading IDENTICAL on the final navpipe code.

The SEARCH-DOSE arm (O4) runs on the same bundle under `AMENDMENT_DOSE_R1.md`.

**Files:**
- `RESULT_SPC.json`, `chain_state.json`, `cid.json`;
- the merged search and label files (`search_C1_P2HEAD.npz`, `search_CID_P2HEAD.npz`, `labels_C1_P2HEAD.npz`);
- the per-shard search outputs (`search_C1/`);
- the C1 G-REACH token (`gate_C1/`);
- the chain's stage logs (`logs/`), plus `chain_spc.log`.

The B-REST bundle itself (3.4 GB: `bundle.npz` + `ctx.npy`) is NOT in git. It lives on the dev box at `C:/Users/Admin/spc_c1/B_REST/bundle/` (bundle sha256 `7b7dfff38c1f380810e61a3a2720dc9c8377df7455d867790c3f0398dd5af968`).
