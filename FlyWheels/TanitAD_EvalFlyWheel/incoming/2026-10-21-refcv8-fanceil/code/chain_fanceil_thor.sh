#!/bin/bash
# chain_fanceil_thor.sh -- after the s1 EMA fan roll: s0 EMA (os-only, a FRESH inference draw) then theta_DEC s1 (os-only).
# Each step waits (blocking flock, 1 h) for the Thor GPU lock the previous step holds. Markers ZZCH<NAME>ZZ in chain.log.
F=/home/nvidia/r8ns/fanceil; L=$F/out/chain.log
echo "ZZCHSTARTZZ $(date -Is) pid $$" >> $L
sleep 30
flock -w 3600 /home/nvidia/refcv7_post/thor_gpu.lock bash $F/code/fanceil/run_fanroll_thor2.sh 0 --os-only; echo "ZZCHS0ZZ rc=$? $(date -Is)" >> $L
[ "$(md5sum /home/nvidia/r8ns/navtrain_sel/wiseft/ckpt_mod2/ckpt_wiseft_mod2_DEC.pt | cut -c1-32)" = dad4cb5b5fb316922b4e9a7e85cee8f8 ] || { echo "ZZCHDECMD5BADZZ" >> $L; exit 2; }
CKPT=/home/nvidia/r8ns/navtrain_sel/wiseft/ckpt_mod2/ckpt_wiseft_mod2_DEC.pt TAG=dec_s1 flock -w 3600 /home/nvidia/refcv7_post/thor_gpu.lock bash $F/code/fanceil/run_fanroll_thor2.sh 1 --os-only; echo "ZZCHDECZZ rc=$? $(date -Is)" >> $L
echo "ZZCHENDZZ $(date -Is)" >> $L
