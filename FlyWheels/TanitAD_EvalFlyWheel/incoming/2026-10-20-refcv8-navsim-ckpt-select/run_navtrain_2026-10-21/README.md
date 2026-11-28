# refcv8 checkpoint selection on NavSim navtrain -- the executed run (2026-10-21 06:51-08:53 Berlin, Thor)

Verdict **DEFAULT_EMA_48852** under D-R8-CKPT-RULE + AMENDMENT 1 (`navtrain_sel/DECISION.json`). NavSim v1 open-loop PDMS,
split navtrain, sample FIXED n = 2,089 tokens / 30 logs (all Las Vegas), sorted-token sha256 `9dd68698…`; rolls Thor CUDA,
scoring Thor CPU. Independent cross-check `xcheck_decision.json`. POST-HOC (never deciding): `DIAG_RT_vs_LEGAL_POSTHOC.json`.
Off-bank (dev box `C:/Users/Admin/refcv8_pf/navsim_ckpt/run/`, sha256 in `OFFBANK.sha256`): the bridge `rows_*.jsonl` and the
scorer `*_hooks.json`. Large inputs stay on Thor `/home/nvidia/r8ns/` (downloads, trainval pickles, frame bank, metric cache).
Three zero-byte files (a cache rows file and two empty stderr/stdout captures) are omitted: git landing refuses empty files.
