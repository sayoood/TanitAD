# refcv8 EMA-48,852 navtest read: RESULT (A-R8-NS1, the one registered read; PI D-R8-CKPT (b))

N1 FAILED, N4 FAILED (see GOALS_AND_CLAIMS "A-R8-NS1 RESULT" and MODEL_REGISTRY §16.4); N2 / N3 pending (navhard / warmup).
`bars/` = navsim_bars8 output; `gate/R8_GATE_NAVTEST.json` = the pre-registered gate test (G_CMD, G_ARC both WORK);
`gate_control_navtrain/` = the gate scorer's known-value controls on navtrain (run before the read); `full/` = seams, manifests,
scorer CSVs / counts / bindings. Per-token rows (`bridge_navtest/rows_*.jsonl`, `*_rows.jsonl`, `*_hooks.json`, `*.calls.jsonl`)
are OFF-BANK: sha256 in `OFFBANK.sha256`; copies at dev box `C:/Users/Admin/refcv8_pf/navtest_read/raw/full/` and Thor
`/home/nvidia/r8ns/navtest_read/full/`. NavSim tokens / log names only; no PhysicalAI clip id (scanned CLEAN).
