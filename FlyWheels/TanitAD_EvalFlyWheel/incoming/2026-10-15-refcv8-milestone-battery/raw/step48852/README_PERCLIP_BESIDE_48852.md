# refcv8 step 48,852 on the PER-CLIP feed: BESIDE, NEVER BINDING (chain_final_r8, finished 2026-10-21 08:43 Berlin)

Every number in this drop was read with the PER-CLIP nav feed (one command per eval clip, no nav arguments).
That feed is superseded (R8-EVAL-NAV-FEED-1, retraction R57; PI 2026-10-20 option a). The binding 48,852 verdict
is the PER-FRAME read: SHIP-R8 PASS, `2026-10-20-refcv8-perframe-final/raw/ship_r8/SHIP_R8_step48852ema_pf.json` (75ce238).

What is here (tier T1 self-action open loop on logged eval139 frames; one training seed; paired episode-cluster bootstrap B 2000):
* `step48852ema/`, `step48852/` -- the EMA and raw batteries on the per-clip feed; `READ_FINAL.md` / `FINAL_INDEX.md` summarise.
  `READ_40K.*` in `step48852/` is the same summary under the summariser's hard-coded name (it reads 48,852, not 40k).
* `ship_r8/SHIP_R8_step48852ema_unc.*` -- SHIP-R8 on the per-clip tree with the per-clip table b208c9c1: FAIL on clause 5
  (S6 TAC_traj_lon_correct separated worse than refcv7 at both seeds). Superseded feed AND superseded table: decides nothing.
* `kdraw/`, `combo_ens_retime/` -- K-draw ensemble and combo reads: both VOID (controls C-K1 / C-K2 failed).
* `a0k0_*/A0K0_*_final48852.md` -- the headroom reports regenerated at 48,852 (the tip's A0K0_*.md are the earlier versions, unchanged).
* `heading_drift/`, `retime_trainfit/`, `retime_deploy/`, `clamp_posthoc/`, `compliant_reread/` -- per-clip lane outputs.
Clip ids: none (scanned CLEAN; npz carry integer episode indices only).
