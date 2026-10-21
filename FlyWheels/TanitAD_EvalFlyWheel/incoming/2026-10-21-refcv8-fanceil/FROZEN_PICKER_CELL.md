# FROZEN before the seed-0 fan exists -- 2026-10-21 19:51:51 +0200 (Berlin)

Post-hoc lever found on the inference-seed-1 fan (PICKERS_R8_s1.json, chosen on episode-half A, read on half B):
MBR inside the core sel_score top-K, cell **sel_score, K=32, T=1.0**, restricted to SPEED-CAP-PROXY-compliant candidates
(slot-level planned max speed <= v_max_ms + 0.5 m/s where v_max_valid) -> half-B 2 s delta vs the model's own pick
-0.0337 m [-0.0427, -0.0249].

FROZEN TEST on the seed-0 fan (a FRESH DDIM draw of the same checkpoint, rolled on Thor; the cell is NOT re-chosen,
flag: --frozen-cell sel_score,32,1.0): the replicate PASSES if the FULL-SET paired (mbr_chosenA_capproxy - own) at 2 s is
separated AND < 0; it FAILS otherwise. Both outcomes are reported as written. Same rule for "base" (the pre-graft core pick).
This is exploratory evidence for a refcv9 / eval-rule PI decision, not a registered claim and not a SHIP-R8 bar input.
