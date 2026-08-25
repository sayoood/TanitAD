# The queue — so an idle box always has a next item

⛔ WHY THIS FILE EXISTS. Twice today a box went idle because a job finished and
NOTHING WAS QUEUED BEHIND IT. The idle sentinel makes that loud; this file makes
it answerable in one step instead of requiring a fresh decision each time.
Ordered, each with the MEASUREMENT that motivates it — never a guess.

## Thor (GPU, ~8 h each)
1. **`o3p30k`** — RUNNING. Masked-cell prediction at 30k parity. Motivated by
   E-DEC-40: the recipe is O5+O6 only and O5 is satisfiable by DRIFT, so nothing
   rewards scene content. O3 was tested ONLY at 2k, the regime E-DEC-21 showed is
   uninformative.
2. **`dinofrozen30k`** — frozen DINOv3 trunk + our predictor. Motivated by: frozen
   DINOv3 beats our TRAINED trunk on 4 of 5 aggregate spatial targets
   (`n_free_cols` +0.4857 vs our +0.2080). If an encoder we did not train carries
   more, our parameter budget is in the wrong place.
   ⚠️ PI-level architecture call — flag before launching.
3. **`o2p30k`** — near-field term at 30k parity. Same argument as O3: only ever
   tested at 2k.
4. **`o3o2p30k`** — both, if either alone moves the spatial panel.

## Dev box (probe, ~40 min each)
1. **physics + envpred on `ro128p30k`** — RUNNING.
2. **`splitp30k` vs `ro128p30k` on the same spatial panel** — splitp30k is still
   the best content carrier (+0.1220 held-out) and has never been compared to the
   readout arm on identical rows.
3. **E-DEC-40 on `splitp30k`** — its drift r is +0.1993 vs `rdw8p30k`'s +0.6570.
   ⭐ FAR less self-predictable. Does its residual carry MORE than noise? If yes,
   that arm's latent is qualitatively different and the drift-floor objective is
   back on the table FOR THAT ARM.
4. **The `nrmse` census re-read on `ro128p30k` + `o3p30k`** when they land, with
   `nrmse_SHUFFLED` beside it, per MODEL_REGISTRY 13.0c.

## Standing rule
⛔ When a job finishes, START THE NEXT ITEM IN THE SAME TURN, then report. The
report is the last 10 % of a turn, never the whole turn.
