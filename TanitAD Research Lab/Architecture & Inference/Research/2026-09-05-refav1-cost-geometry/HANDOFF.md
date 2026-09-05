# HANDOFF — the refav1 cost-geometry ladder, and how to finish it

*Written 2026-09-05 while arms were still on the dev-box 4060, so a fresh context can
land them without re-deriving anything. Read `RESULT.md` and
`PREREG_COST_GEOMETRY.md` first; both outcomes for every arm are already committed
there.*

## 1. What is running, and where

All arms run on the dev-box RTX 4060 (8 GB), `OMP_NUM_THREADS=6`, from the off-Drive
clone `C:/Users/Admin/tanitad-wt` with `PYTHONPATH=<clone>/stack;<clone>/taniteval`.
**Two arms at a time is the proven ceiling** (a third has been measured to OOM-risk
at ~200 MiB free). Records land in `C:/Users/Admin/refav1_margin/p4out/rec_<tag>.json`
with a `dump_<tag>/` beside them.

| lane | script (scratchpad) | arm | waits for |
|---|---|---|---|
| A | `queueA.sh` | `wk15` -> `wk151` | (running) |
| B2 | `queueB2.sh` | `ccosh_w000` | the `cos_wk` PIDs (done) |
| D | `queueD.sh` | `l3ladder` | `ZZQUEUEA-DONE` |
| E | `queueE.sh` | `kamm07` | `ZZQUEUEB2-DONE` |
| F | `queueF.sh` | `combined` | `ZZQUEUED-DONE` |

Each lane polls for the **ABSENCE** of `python.exe` processes whose command line
names the arm tool — never for a success marker, and never with a shell whose own
command line contains the tool name (that self-match bit once in this session and
killed nothing important only by luck).

## 2. How to finish it — one command

```
bash <scratchpad>/finalize.sh
```

It is idempotent and it **names the arms whose record is absent** rather than
silently thinning the panel. It writes, into the scratchpad:

* `four_family_all.txt` — the four families per arm, with each arm's
  `(metric, W_JERK, W_KAPPA, W_VEND)` printed above it
* `kappa_by_goal_all.txt` — realised curvature by DECODED goal token and by GT
* `kappa_quantisation_all.txt` — the distinct realised `max|kappa|` values and the
  EXACTLY-constant-series fraction (the seed-pool diagnostic)
* `cost_scale_all.txt` — the goal term's own decision range per arm
* `feas_audit_all.txt` — the envelope / friction-circle audit at `vmin` 0, 2, 5
* `pd_all.md` / `pd_all.json` — the PAIRED four-family deltas of every arm against
  `ccos_argmax`

⛔ **`feas_audit` at `vmin = 0` is INADMISSIBLE and says so** — the ground-truth
control fails there. Quote only the `v0 >= 2` and `v0 >= 5` blocks.

Copy each into `raw/` in this package and stage it.

## 3. The rules that decide what the numbers mean

1. **Four families or it is incomplete.** ADE alone would have read the shipped
   arm's paradox as progress (§7.1: seven separated family improvements, produced by
   the planner *stopping*).
2. **A separated CI is necessary and NOT sufficient on this rig.** `D-REFAV1-CG-SEEDFLOOR`:
   two arms differing only in `--plan-seed` read `separated` on **4 of 10** paired
   family metrics. The admissible form is *"the lever's delta exceeds the seed pair's
   delta on the same metric"*. The floor is `ade 0.0607`, `fde 0.3439`,
   `cross 0.0710`, `heading 1.1180`; per-arm point drift on `TAC lat kappa` is
   **25.6 %**.
3. **"Beats the floor" must be conjoined with "acts"** (`D-REFAV1-CG-OBJECTIVE-RESTATED`).
   An arm that ties `ha0_ext` by emitting zero controls has not driven. Report the
   lateral-decision kappa and the turn recall beside the ADE, always.
4. **Every arm carries its cost triple and its metric**, or it is not quotable.
5. **The vocabulary is v7.0 / `GOAL_KAPPA_TURN = 0.08` in every arm here.** The seed
   ladder and the Kamm cap are NOT vocabulary changes (`canonical_controls` and the
   goal field are untouched), so these arms are comparable window-for-window.
   `--goal-kappa-levels` / `--goal-kappa-turn` arms would NOT be.

## 4. The next experiments, in order, if the ladder is not enough

1. **A seed replicate of the winning arm** (`--plan-seed 1`, everything else
   identical). Rule 2 makes this the cheapest thing that converts a promising arm
   into a quotable one, and the tactical family needs it most (25.6 % seed drift).
2. **A Kamm sweep** — `--kamm-mu` in {0.4, 0.7, 1.0}. `mu` is a road property, not a
   tuning knob, so the sweep is a sensitivity statement, not a search.
3. **`ccosh` + `kamm-mu` + the seed ladder at a FINER bottom rung** (0.001 = R 1000 m)
   if `l3ladder` shows the pool binding but the rungs too coarse.
4. **Only then** the goal head (`L5`). The evidence is still that a *perfect* goal
   makes this planner worse (`D-REFAV1-ORACLE-SAT-FULL`, ADE −1.2181 separated,
   20x the seed floor), so a sharper head feeds a mechanism that already fails on a
   perfect input.

## 5. Two decisions that are NOT this agent's to make

Three levers are implemented, pinned, and **OFF by default** — every default is
bit-identical to the pre-2026-09-05 planner:

* `--cost-metric ccosh` (the hold branch; `COST_METRICS` states in the source that
  this belongs to the PI / Master Mind)
* `--seed-kappa-ladder` (the iCEM iteration-0 candidate set)
* `--kamm-mu` (the friction-circle curvature cap)

**Whether any becomes refav1's default is a programme decision.** The arms in this
package measure them; they do not authorise them.

## 6. Housekeeping this turn exposed

* `C:/Users/Admin/tanitad-wt` was **stale** for the sibling stream's feasible-decode
  work (`stack/tanitad/refs/feasible_decode.py` absent, `refc.py` an older blob).
  Both were synced from the repo here and the affected tests then passed
  (**59 passed**). A dev-box clone gets no `pod_currency_audit.py` run — it should.
* `run_oracle.sh` kept a **second shell** alive after its subshell was killed, and
  quietly started `oracle_s1` two hours later. *"Killed X" is not evidence the
  pipeline stopped* — verify the pipeline, not the PID.
* The arm records and dumps were single-copy off-repo; they are now banked under
  `raw/arms/` (90 files, 2.4 MB). **Bank the remaining arms the same way.**
