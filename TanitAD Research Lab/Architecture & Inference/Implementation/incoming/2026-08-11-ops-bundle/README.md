# Ops bundle 2026-08-11 — the pod-fleet toolkit, banked

**Why this exists:** the operating standard's rule 3 (an artifact on one disk is not done) —
and the pod ops bundle was stranded once before. Everything here ran in production during the
v5.8f campaign (2026-08-09..11). No secrets: connection URLs/tokens live in git-ignored
side files (`gotty_url.txt`, pod HF token at `/root/.cache/huggingface/token`), read in
place, never embedded.

## Transport layer (the only reliable pod I/O paths — see CLAUDE.md traps)

- `gxm.py` — hand-rolled gotty/WebSocket PTY client through the agent proxy (ALPN pinned
  to http/1.1; TLS verification ON). THE ops workhorse: every pod command this campaign
  went through it. Reads base URL from `GOTTY_URL_FILE` (default `gotty_url.txt`), which
  is NOT committed.
- `parse_frames.py` / `merge_frames.py` — the `@@NNN …##` framed-pull protocol for
  corruption-tolerant PULLS over the PTY (lines drop in bursts and get +1-char corruption;
  per-line refetch + later-file-wins merge is what actually holds). For whole files, the
  compact variant is xz+b64 chunks with the same framing and full-file retry-until-valid.
- `p5_upload.py` / `p4_pull.py` — the HF shard relay for multi-GB pod→pod transfers when
  the direct SSH mappings are dead (measured 2026-08-11: they were, both directions):
  2000 MB tar shards + per-shard md5 + `MANIFEST.json` uploaded LAST; the receiving side
  polls for the manifest, verifies EVERYTHING, and refuses partial states.

## Chain scripts (self-sequencing GPU queues; each verify-gates its code before launch)

- `w4r_chain.sh` — W4 unicycle-head refit on the stage-A trunk → W7-w4r re-rank (pod5).
- `i4a_chain.sh` — I4a imagination-ablation triplet (SUPERSEDED by the no-git `i4a2` form
  after the pods-have-no-git-credentials finding; kept as the record of the git-sync
  mistake the CLAUDE.md trap now forbids).
- `p8c_chain.sh` / `p4p8c_chain.sh` — P8 attempt-2 occupancy retrain (pod5 / pod4 twins;
  the pod5 run doubles as a same-seed cross-pod reproducibility check).
- `p4stage.sh` / `p4relay_chain.sh` — pod4 corpus staging: direct-SSH attempt (dead
  mappings, kept as the measurement record) and the HF-relay + P1-rerun chain that
  actually ran.

## Instruments

- `rescore_v58f.py` — decision-grade episode-cluster bootstrap rescore of the banked
  v5.8f windows (taniteval.ci + selgap) — the §1.14 CI source.
- `render_v5f_bev.py` / `render_v5f_fanfull.py` — the delivered v5f video generators
  (BEV pane with GT+selected+fan; full-256 plan-fan in taniteval/plan_fan.py conventions).
