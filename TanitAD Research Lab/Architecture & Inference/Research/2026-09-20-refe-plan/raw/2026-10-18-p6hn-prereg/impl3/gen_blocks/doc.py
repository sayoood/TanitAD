
P6-HN-S (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md, checklist item 3) -- BOTH DEFAULT OFF, the OFF path is bit-identical to this trainer without them (refe/test_hn_screen_wiring.py I3):
  --hn-bank PATH  arm B ONLY: += L_hn = hn_bank.hn_term (the masked two-call term over [fan ; hn], 1/(B*K*6) per element, six PDM components, no lane target).  Arm A is the BCE-only control.
  --ctx-cache DIR the frozen trunk's visual_ctx comes from the cache (ctx_cache.CtxCache), not from the trunk: A scores through model.score_trajectories, B / the base through their own scorer calls, on one tensor.
  The preflight adds gates G14 (hn_live_and_identity) and G15 (ctx_cache_identity); the data event carries `hn` / `ctx_cache` blocks of BUILT facts for G-DVB.
