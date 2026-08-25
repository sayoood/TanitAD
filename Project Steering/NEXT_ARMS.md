# The queue — so an idle box always has a next item

⛔ WHY THIS FILE EXISTS. Twice in one day a box went idle because a job finished
and NOTHING WAS QUEUED BEHIND IT. The idle sentinel makes that loud; this file
makes it answerable in one step instead of requiring a fresh decision each time.
Ordered, each with the MEASUREMENT that motivates it — never a guess.

**Rewritten 2026-08-26 00:20.** ⚠️ **The previous version had rotted in the way
this programme keeps rediscovering: its Thor item 2 was motivated by a claim
C156 RETRACTED** (*"frozen DINOv3 beats our TRAINED trunk on 4 of 5 spatial
targets"* — the retraction is that I omitted `n_agents`, the target we WIN at
+0.1220 vs +0.0998, and recommended replacing our encoder with the teacher we
already distil from). Three more items named arms that had since finished or been
aborted. ⇒ **A queue is a claim about what is worth doing; it rots exactly as fast
as the claims under it.** Re-derive it whenever a load-bearing claim moves.

---

## Thor (GPU, ~8 h each)

1. **`postrain30k`** — **RUNNING**, ~step 13,600/30,000, ETA ~04:30 Europe/Berlin.
   Gate-A flags + `--init-from` the DINOv3-distilled checkpoint.
   ⇒ **When it lands: score it AND launch item 2 in the SAME turn.**

2. ⭐⭐⭐ **`o13p30k`** — **THE NEXT ARM. O13-EGO at 30k parity.**
   Motivated by the strongest pair of measurements in the campaign:
   **E-DEC-48b** (the action's marginal contribution to the future SCENE is
   **−0.1678, t −3.50**, against a positive control at t 8.5–14.3 — nine
   objectives asked for information the data does not contain) and **E-DEC-50**
   (the action DOES determine the ego's own dynamics: Δspeed **t 2.56**, Δyaw
   **t 4.57**, identity control **+0.9337, t 23.74**).
   **Ready to launch:** implemented, 9 unit tests, 2-arm wiring smoke passing,
   pre-registered (`PREREG_O13_EGO_DYNAMICS.md`, four outcomes + a step-12,800
   abort criterion fixed before the outcome is known), and **staged on Thor at
   `/home/nvidia/staging/train_v6_staged_O13.py`, md5
   `ed82d89f41a14e66c40aa0e3a64826d6` verified identical to local.**
   ⛔ **Swap the staged trainer in only AFTER `postrain30k` writes its
   done-marker** — its supervisor relaunches from the trainer path.

3. 🔶 **`o12p30k`** — ActSWM's frozen readout. **DEMOTED, NOT CANCELLED.** It
   would create action-discriminative structure in a space E-DEC-48b measured to
   have no action information. ⚠️ **It is NOT the fallback if O13 fails** — the
   pre-registration commits a REFUTED O13 to **interventional data**, a PI
   provisioning decision, not to a tenth objective on this corpus.

4. ⛔ **`dinofrozen30k` — REMOVED, its motivation was RETRACTED (C156).**
   Kept here as a named removal rather than deleted, because a silently vanished
   queue item is indistinguishable from one nobody got to.

5. ⛔ **`o3p30k` / `o2p30k` / `o3o2p30k` — REMOVED.** O3 was run and **aborted at
   step 20,400 on its own pre-committed criterion**. E-DEC-48b now explains the
   whole family as a class: these all move the SCENE latent.

## Dev box (probe, ~40 min each)

1. **O13 feasibility pilot + matched w=0 control** — **RUNNING** (3,000 steps
   each, 24 eps, NON-PARITY). ⚠️ Answers *"does o13 cost prediction accuracy?"*,
   **not** *"does it generalise?"* — O11 also showed a positive excess while
   degrading `o5` by 18.7 %, and only a matched arm separates the two.
2. **`egostate.py` on `postrain30k`** when it lands — the pre-registered O13 read
   uses this instrument, so the incumbent's numbers on the newest arm are needed
   as the comparison baseline.
3. **E-DEC-40 on `splitp30k`** — its drift r is +0.1993 vs `rdw8p30k`'s +0.6570,
   and it is now the only arm whose latent carries the ego's Δspeed (**t 2.50 /
   2.05 / 2.10, replicated three times, three code paths**). ⭐ Does its residual
   carry more than noise? If yes that arm is qualitatively different.
4. **The `nrmse` census re-read** on the new arms, with `nrmse_SHUFFLED` beside
   it, per MODEL_REGISTRY 13.0c.

## Standing rule

⛔ When a job finishes, **START THE NEXT ITEM IN THE SAME TURN**, then report. The
report is the last 10 % of a turn, never the whole turn.
