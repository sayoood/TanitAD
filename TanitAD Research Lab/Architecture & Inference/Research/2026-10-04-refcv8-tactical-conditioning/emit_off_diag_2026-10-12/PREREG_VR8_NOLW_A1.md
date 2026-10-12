# PREREG_VR8_NOLW amendment A1: PI approval, how the arm is run, and its measured cost

Written 2026-10-12 ~19:10Z by the Master Mind, before the arm has started (it waits for the ladder). **The readout
(P1, P2, the three outcomes) is UNCHANGED.** This amendment records execution facts only.

1. **PI ruling, verbatim (2026-10-12):** *"yes, run V-R8-noLW after the ladder"*. It is a 12th tiny-rig run, beyond
   PI-R8-LADDER-11, and it runs only after the ladder chain has finished.
2. **How it runs.** `noLW_chain.sh` on Thor (`/home/nvidia/refcv8_ladder/noLW/code/`) = `ladder_chain.sh` lines 1-196
   VERBATIM (the ladder's functions: GPU lock, MemAvailable check, chunked training, `ladder_eval.py`, strip to
   `model_final.pt` + md5, `PULL_READY`; source md5 `f71a25fd`) + `code/noLW/noLW_main.sh` (md5 `86655cbf`; the
   assembled script is md5 `74504dfa`). The main waits for the ladder chain's PID to exit, then starts ONLY if the
   chain's last `ZZSTARTZZ` segment holds `ZZALLDONEZZ` and no `ZZSTOPZZ`. Its own `W` is `…/refcv8_ladder/noLW`,
   never the ladder's. Before arming, all three branches were dry-run in a throwaway dir (`DRY=1`): it refuses on an
   unfinished ladder (rc 4), refuses on a ladder `ZZSTOPZZ` (rc 4), and on a clean finish runs train → `base` eval →
   `ZZNOLW-ALLDONE`, ignoring a `ZZSTOPZZ` from an older segment. Armed 2026-10-12T18:57:41Z (`ZZNOLW-WAIT chain pid
   857680`).
3. **Argv.** The run file `raw/V-R8-noLW.thor-run.argv.json` (argv_sha256 `5ca38d3c…`) differs from the
   pre-registered `raw/V-R8-noLW.argv.json` (`dc4c1fa0…`) ONLY in `--out`, which now points outside the ladder's `W`
   so the ladder's I-0 and scorer can never see it. Against V-R8: 2 of 189 positions differ (index 182, the
   weight `1.0 → 0.0`; index 188, `--out`).
4. **Cost, MEASURED** from the ladder's `chain.log` (chunk 1 start → `ZZDONE`): V-R8 **4 h 24 min**, V-R8d **4 h
   25 min**, V0r **4 h 04 min**; one `base` eval about **11 min**. So about **4.6 h of Thor**. The PREREG's "about
   4.5 h" stands. (A 3.3 h figure from the instantaneous step rate was wrong: it ignored chunk restarts and the
   gradient-share readings.)
5. **One read ADDED, none replaced.** Thor also evaluates `base` at batch 8, both sampler seeds (the chain's own eval
   path). That gives a same-batch comparison with V-R8's and V-R8r's Thor dumps, reported as a second read. P1 and P2
   stay on the dev-box batch-4 `--save-cands` capture, as registered.
