# HANDOFF_READ (N1DC D1-HO read, 2026-11-20, read-builder agent)
State: `n1dc_ho_read.py` BUILT and REPLAYED: REPLAY_N1DW.json = GREEN at w2 and w4 (102 rows each; accuracy rows EXACT, continuous R1 cells to 1e-12 because dev-box vs pod differ at 1e-16); mutant REPLAY_N1DW_MUTANT.json = RED. Replay dumps deleted. The REAL read is NOT run (no N1DC dump exists). Nothing staged, committed or pushed. No GPU. Pod: only tar + pscp of existing files; its tmp dir removed.

## What it computes (card 2eae7964 s3, s4; pure functions copied unchanged from n1dw_ho_read.py)
Per executor (hb = T1-kin/hb0.2, decides; old = T1-kin/old, beside): D1-HO TAC-LAT N1DC - N1D per seed (paired episode-cluster bootstrap B 2000 seed 0), bar = max(0.0383, F_HO, |N1D draw|, |N1DC draw|); N1DC - N1R; 26 R1 cells (LON/LAT; dk cells n = 0) vs N1R and N1D; Z-USE (N1DC minus `_zuse`) and the D-channel zero twin, TAC-LAT and cells; T0 tick-0 plan TAC-LAT (`ol` of the same dumps, the method of n1dwt0t1_read.py; no separate pass needed) with N1D re-derived as control (0.6266/0.6349, K-T0). Controls K1, K-CONST, K-RE (hb vs RESULT_N1DW_D1HO_HB.json; old vs RESULT_NAVRBD.json), K-T0; any failure sets status INVALID. Outcome labels D1-HO part; guard part prints PENDING-GUARD. Z-EVAL held-out: NOT-YET (needs the zero-history mask). eval-141, K-CLIP, STRATEGIC are not here.

## Pull (MM; pod = training, tar + pscp only; `source C:/Users/Admin/lorapod/podenv.sh`)
    pssh 'mkdir -p /root/strat1/n1dc_readtmp && cd /root/strat1/n1dc/out && tar cf /root/strat1/n1dc_readtmp/n1dc.tar --exclude="ticks_*" --exclude="log_*" --exclude="DONE_*" hb0.2 hbnone'
    mkdir -p F:/TanitAD-scratch/n1dc/read/in/n1dc && cd F:/TanitAD-scratch/n1dc/read/in/n1dc && pscp "$POD_TARGET:/root/strat1/n1dc_readtmp/n1dc.tar" . && tar --force-local -xf n1dc.tar && rm n1dc.tar
    pssh 'rm -f /root/strat1/n1dc_readtmp/n1dc.tar; rmdir /root/strat1/n1dc_readtmp'      # log both deletions
Expected layout `in/n1dc/{hb0.2,hbnone}/t1[_zero]/dump_N1DC-HO-N1DC_s{0,1}[_zero|_zuse]_m10_causal_heads` (994 ep*.npz per seed pair per dir). The `_zuse` dump is looked up in `t1`, then `t1_zuse`, then `t1_zero`; the `_zero` twin in `t1_zero`. About 100 MB per executor; check `(Get-PSDrive F).Free` first (F: had ~5.9 GB after this build; `read/tree` 330 MB and `read/in` 330 MB are re-creatable).

## Run (dev box, about 4 min; commit headroom >= 5.5 GB; one pytest/job at a time)
    cd F:/TanitAD-scratch/n1dc/read && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -u n1dc_ho_read.py --out out_real
Defaults: tree `./tree` (git archive of 134f5da1, blobs 96839329/528276c5/b7e44c3e/25190ad6 verified at start), inputs `./in`, banked `./in/banked`. Result `out_real/RESULT_N1DC_D1HO.json` + `READ_N1DC_D1HO.txt`; exit 1 = a control problem. Env alternatives: N1DC_TREE, N1DC_IN, N1DC_BANKED, N1DC_OUT.

## Left
1. N1DC dumps (hb0.2 and hbnone, real, zero twin, zuse) once the chain finishes; pull; run. 2. The eval-141 guard, K-CLIP and the S1-slot view are separate reads. 3. Copy this folder's code into the repo through the lander (the MM lands). 4. Delete `read/in` and `read/tree` when the read is banked (log them).
