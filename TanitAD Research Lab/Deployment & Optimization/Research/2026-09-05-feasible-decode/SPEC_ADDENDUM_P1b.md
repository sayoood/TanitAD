# SPEC ADDENDUM — P1b, the successor lever, pre-registered BEFORE it ran

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/SPEC_ADDENDUM_P1b.md`
2026-09-05 · Architecture & Inference FlyWheel · ZERO GPU.

⛔ **P1 FAILED as committed** (`raw/progress_rank_fix.json`): P1-C1 PASSED
(ρ(progress_v2, `peak_g`) **−0.5897 [−0.6707, −0.5002]**, from **+0.2900**) and **P1-C2
FAILED** — on 160 straight windows the argmax-`progress_v2` candidate covers **18.672 m**
against the argmax-`progress_v1` candidate's **26.216 m**, a ratio of **0.7123** below the
committed **0.95**. The verdict is reported as written. This addendum is the successor,
executed in the same run, with its criteria fixed before it ran.

⚠️ **The criterion is NOT relaxed and the reference is NOT changed.** P1b-C2 keeps the
identical form and the identical 0.95 bar against the identical reference arm. *(It is
worth recording what the failure looks like on inspection — argmax-`progress_v1` covers
26.216 m where holding `v0` covers 17.229 m, i.e. v1's favourite candidate is travelling
52 % further than maintaining speed over 2 s, which is the violating behaviour the fix
exists to stop rewarding. That observation is a DIAGNOSIS. It does not move the bar, and
the bar is not moved.)*

---

## 1. Diagnosis: why `progress_v2` under-shot

`progress_v2 = min(r, 1.0) · w(ex)` **ties** every candidate at or above the v0-hold
reference, so the tie is broken **only** by `w(ex)` — which always prefers the least
aggressive candidate available. The cap did the intended job (it removed the reward for
exceeding `v0`) and then did one job too many: it removed the ability to prefer a
*legitimately faster* candidate over a merely feasible one.

⭐ **And P1 measured the cap's contribution separately**, so this is attribution and not a
story: `prog_v2_caponly` (the cap, no envelope discount) reads ρ(peak_g) **+0.1294** — the
cap alone more than halves the correlation, purely by creating ties.

## 2. The successor lever: `progress_v3` — progress on the path the car can ACTUALLY drive

```
progress_v3(traj, ctx) = clamp( along( project_feasible(traj, v0, mu=0.7) ) / ref, -1, 1.5 )
```

with `project_feasible` the P2 module (`tanitad.refs.feasible_decode`) and `ref` unchanged
from the stock term. In words: **credit a candidate with the distance it would cover
after the friction-feasible projection, not with the distance it drew.**

Three reasons it is the right successor rather than another shape of `w`:

1. ⭐ **It has NO new hyper-parameter.** `w(ex) = 1/(1+ex/ex_s)` needed `ex_s`; this needs
   nothing that is not already fixed by the scorer's own constants.
2. ⭐ **It is the IDENTITY on feasible candidates.** `project_feasible` is a fixed point on
   any path already inside the envelope (pinned: `test_roundtrip_on_an_already_feasible
   _path_is_exact`), so ranking among legitimate candidates is **not compressed at all** —
   which is exactly the property the 2026-08-29 per-window-reference decision was
   protecting and `progress_v2` sacrificed.
3. ⭐ **It unifies P1 and P2 into one mechanism.** The reward and the decode then agree on
   what "possible" means, instead of a reward term and a decode constraint tuned against
   each other.

⛔ **Its degenerate, stated:** "drive as fast as the friction envelope allows, through
everything" — still hackable alone, still checked by `collision` + `headway`, exactly as
the stock term's is. It is **not** less hackable than `progress`; it is hackable in a way
the vehicle can survive.

## 3. Committed criteria — identical in form to P1's

| id | criterion | reading |
|---|---|---|
| **P1b-C1** | ρ(`progress_v3`, `peak_g`) **< +0.05**, per-window Spearman, episode-cluster bootstrap, `n_boot = 4000`, `seed = 11` | PASS / FAIL |
| **P1b-C2** | on straight windows, mean along-track displacement of the argmax-`progress_v3` candidate **≥ 0.95 ×** that of argmax-`progress_v1` **AND ≥ the human's own** | PASS / FAIL |
| **P1b-C3** | reported, not gating: ρ against `ttc_below` / `contact` / `envelope`; and the composed DEFAULT reward with `progress_v3` substituted | reported |

⛔ **Both must pass.** If P1b-C2 fails, the honest verdict is that **no reshaping of a
single scalar progress term clears both bars on this fan**, and the successor is
structural — `progress` and `headway` split into a longitudinal *pair* — which would be
scoped, not started, in this package.

## 4. The second window draw — a REPLICATION, committed as such

P1 ran on the 240-window draw (`fan_bank_base_240w.npz`, md5
`ccecf0ade75631e315e6b9ae78d83478`, 121 episodes) which carries **no ground truth**, so its
P1-C2 human floor used the `ha0` v0-hold substitute. P1b re-runs the **whole panel** on the
independent 400-window draw (`kingate_bank_drawA_v1_selscore.npz`, md5
`9bda7715a6571186935243b242c2ab74`, 70 episodes) which **does** carry `gt4`, so:

* the human floor in C2 is the **human's own** along-track displacement, as originally
  written, not a substitute;
* the 240 w numbers get an independent replication on a different draw. ⛔ **A sign flip or
  a bar crossing between the two draws is reported as a failure of stability, not averaged
  away.**

⚠️ **C-OBJECT on the 400 w draw is a DIFFERENT control** and must be stated as such: that
npz carries no banked `c_*` component columns to reproduce. Its object assertion is instead
that `raw/mu_frontier.py` reproduced the sibling's independently written
`displacement_frontier.py` λ-table **row for row** on the same file — a different script,
a different author, the same object — plus the npz md5 and the checkpoint's
`core.decoder.anchors` key.
