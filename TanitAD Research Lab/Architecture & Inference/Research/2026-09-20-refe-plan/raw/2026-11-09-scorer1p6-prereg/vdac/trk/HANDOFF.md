# HANDOFF: G0-TRK tracked-footprint dump (A2 sec.1/3), written 2026-11-11 ~09:55 (+0100)

State: smoke (20 tokens) PASSED (d0==table 1280/1280, corner rebuild diff 0.0, DET bit-identical, LIVE 100% disp>0.01 m, d0!=d1 3% on those 20; see smoke_verify/). Full dump running DETACHED (CPU only, 2 shards, low priority, OMP=1; GPU untouched).
Rate ~0.55 s/token/shard (~95 tok/min/shard); 6073 tokens/shard -> ETA ~11:00 (+0100) on 2026-11-11.
NAVSIM: F:/Archive/devbox-C/... (archive moved from D: to F:, both probed); PYTHONPATH navsim-3e8291bf...; venv C:/Users/Admin/navsim-crun/venv.

Processes (own PIDs only): launcher cmd 25348 (shard 0), 18652 (shard 1); venv python 62436 / 49472 (+ children 17004 / 59864 are the python launcher stubs). Kill only these explicitly if needed.

Check completion (no model turns): 
  F:/tanitad_scratch/vdac_trk/run/full/DONE_0of2 and DONE_1of2 exist, and last line of run/shard{0,1}.log is `ZZTRK_DONE shard i/2 ... d0_table_mismatch=0`.
Parts are resumable (chunks of 250; rerun run_shard{i}.cmd via WMI `cmd /c ... < NUL` skips finished parts). Output ~1.6 GB in run/full (F: had 13 GB free).

Next (fresh agent, once both DONE exist):
 1. DET dump (independent): in NAVSIM venv + PYTHONPATH + OMP_NUM_THREADS=1:
    python trk/trk_dump.py --limit 100 --shard 0/1 --out F:/tanitad_scratch/vdac_trk/run/det
 2. python trk/trk_verify.py --full F:/tanitad_scratch/vdac_trk/run/full --det F:/tanitad_scratch/vdac_trk/run/det --out trk/final
    -> trk_verify.json/.md: G0-TRK (777,344), G0-TRK-DET, C-TRK-LIVE, sha256 of full tracked-corner array. Non-PASS => INVALID until explained.
 3. Bank run/full parts (verify by sha256 into trk/manifest) before deleting; scratch is the only copy until then. Write manifest sha256 per part.
 Never compute picks/PDMS/gain/veto on navtest (ledger).

Notes: table 'sub' col 1 = DAC (identity-verified); proposals_p6final.npz tokens == table tokens (asserted). Corners stored only for the first 20 tokens per chunk (global frame, NAVSIM's own) for the rebuild check; otherwise states (x,y,heading float64) + vp + origin; rebuild with trk_dump.rebuild_corners.

Deletions logged: F:/tanitad_scratch/vdac_trk/run/smoke (6.5 MB), .../smoke_det (6.5 MB): re-creatable smoke dumps (20 tokens), verified results kept in trk/smoke_verify.
Retained (running job / only copy): run/full, run/*.cmd, run/shard*.log.
