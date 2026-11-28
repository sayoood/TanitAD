# RESULT_NAVAUDIT -- why the route-text nav misses the turns v9 names, and a wider announcement cut

Evidence class: MEASURED (ours; banked T1-kin dumps, the pinned text table `navtok_table.npz`, zero training, CPU). Tier: T1-kin tick 0 (the window's NOW frame). Estimator: episode-cluster bootstrap (`taniteval.ci`), B 2,000, seed 0, 'another draw of EPISODES' only. Artifact: `F:/TanitAD-scratch/strat1/navaudit/out/RESULT_NAVAUDIT.json`. Builder sha256 `fdbe4d2368348485` (the landed `build_navtext_nav.py`, imported, not edited).

Windows: 197 labelled TURN_L / TURN_R windows (30 episodes). Pinned text names the correct turn at NOW on 112, v9 on 178; **75 windows in 19 episodes are named by v9 only** (the audited set).

## 1. Audit of the windows where only v9 names the turn

Rules (first match wins; `navaudit.py` `classify`): MATCH-FAILED = table flag at NOW is not 0, or a sided row whose countdown could not be computed (`build_navtext_nav.py:80-81`, `:93-94`); ANNOUNCED-LATE-CUT = flag 0, the table side at NOW is the labelled side, token FOLLOW because countdown > `d_ann = max(30, 6 v)` (`:100-101`); ANNOUNCED-LATE-ROW = no side at NOW and a flag-0 row with the labelled side within the next 8 s (40 steps of 0.2 s); ROUTE-LACKS-TURN = no side at NOW and none in the next 8 s; OTHER = the table side at NOW is the other side.

| class | windows (of 75) |
|---|---|
| MATCH-FAILED | 3 |
| ANNOUNCED-LATE-CUT | 11 |
| ANNOUNCED-LATE-ROW | 0 |
| ROUTE-LACKS-TURN | 55 |
| OTHER | 6 |

* Concentration: the 5 episodes with most audited windows hold 0.573 of the 75. Per-episode table in the json (`audit_by_episode`).
* ROUTE-LACKS-TURN: 10 episodes; in 9 of them (50 of the 55 windows) the text table holds NO turn-sided row at ANY step of the clip (episodes 31, 43, 52, 69, 91, 94, 102, 126, 135). The text is silent for the whole clip, not late.
* Of those 9 text-silent episodes, 6 are roundabout clips (RB_* / ROTARY rows on more than half of the flag-0 steps, which the builder maps to none by rule 1, `:6`; episodes 31, 52, 69, 102, 126, 135; 43 windows) and 3 hold CONTINUE rows only (episodes 43, 91, 94; 7 windows).
* Flag-0 step class groups in those episodes (counts over the clip): ep 31: {'ROTARY': 79, 'CONTINUE': 21}; ep 43: {'CONTINUE': 100}; ep 52: {'CONTINUE': 13, 'RB': 87}; ep 69: {'RB': 99, 'ROTARY': 1}; ep 91: {'CONTINUE': 100}; ep 94: {'CONTINUE': 100}; ep 102: {'RB': 65, 'CONTINUE': 35}; ep 106: {'CONTINUE': 58, 'TURN': 44}; ep 126: {'RB': 67, 'CONTINUE': 33}; ep 135: {'RB': 71, 'CONTINUE': 29}.
* ROUTE-LACKS windows with a class the builder maps to none (KEEP / MERGE / RAMP / CONTINUE_SLIGHT ...) whose name ends in the labelled side within 8 s: 0 of 55.
* ANNOUNCED-LATE-CUT: countdown at NOW median 57.2 m, max 73.8 m (n 11); with a text turn within 100 m: 11, within 200 m: 11.
* ANNOUNCED-LATE-ROW with the later row inside the label's 2-8 s band: 0 (of 0).
* v9 (ego-derived turn distance, a label-side quantity) at NOW: 75 of 75 within 100 m, 75 within 200 m. The turns are close; the text has no row for them.
* MATCH-FAILED: flag values in the eval table `{'0': 13794, '1': 2, '2': 100, '3': 341}` (flag meaning beyond 'not 0 = no usable row' is UNVERIFIED here).
* Upper bound: if the cut were removed entirely (any distance), the text table holds the labelled side at NOW on 124 of 197 windows.

## 2. The wide variant (new file, same builder code path)

`navtext_nav_wide.npz`, sha256 `32abf5fcfebbfb86897b9ded8c58cc589c74dc71ccc691ff230c9055013f7866`, 1085860 bytes. Pre-registered (`PREREG_NAVAUDIT.md`): `d_ann = max(60 m, 8 s x v)`, two module constants of the builder changed by the wrapper `build_variant`, nothing else.

* train: same wrapper at 30 / 6 reproduces the pinned arrays {'train_nav': True, 'train_nav_d': True, 'train_rowclamp': True}; wide vs pinned: 14,437 steps changed, equal to the expected set (flag 0, sided, countdown between the two cuts) True, ch3 / ch4 / nav_d identical True, every change FOLLOW to side True.
* eval: same wrapper at 30 / 6 reproduces the pinned arrays {'eval_nav': True, 'eval_nav_d': True, 'eval_rowclamp': True}; wide vs pinned: 355 steps changed, equal to the expected set (flag 0, sided, countdown between the two cuts) True, ch3 / ch4 / nav_d identical True, every change FOLLOW to side True.
* No ego-future read (source: the builder reads the table, `J['now_s']` and track rows up to row 2j: arc at NOW and at `now - age`, speed at row 2j; `:75-100`). Probe: 200 eval and 200 train steps sampled, track rows after 2j replaced by garbage (sd 1e6): 0 / 0 differ (23 / 27 carried a side token). A mutant reading 10 rows ahead goes RED in `test_navaudit.py`.

## 3. Score on the 197 turn windows (landed navecho3 functions, unchanged)

| feed | names the correct turn | names the other side |
|---|---|---|
| pinned text (30 m, 6 s) | 0.569 [0.393, 0.725] (112/197) | 0 |
| **wide text (60 m, 8 s)** | 0.629 [0.437, 0.803] (124/197) | **3** |
| v9 | 178/197 | 0 |

* wide minus pinned, paired over the same windows: +0.0609 [+0.0269, +0.0973] sep; gained 12 windows, lost 0.
* v9 minus wide: +0.2741 [+0.0670, +0.4922] sep.
* Where wide and v9 both name it (114 windows, 17 episodes): text distance median 20.4 m vs v9 13.2 m (text longer on 109, shorter on 5; mean difference +8.3591 [+4.6701, +11.0866] sep); text countdown inside the 2-8 s band for 0.818, v9 0.527.
* Side of the wrong-side windows: 3 windows, all in episode 139 (table row names TURN_R at a countdown of 50-56 m while the label is TURN_L; the pinned cut hid them). Over all 2,399 windows the wide feed carries a turn token on 289 windows (pinned 223, v9 395); on labelled windows whose label is not a turn, wide carries a token on 125 (pinned 87).

Nav-only out-of-fold classifier (navecho2 HGB, 5 channels, episode-grouped 5-fold), macro-F1; paired wide minus pinned:

| windows | n labelled | pinned | wide | wide minus pinned (paired) | TURN_L recall pinned / wide | TURN_R recall pinned / wide |
|---|---|---|---|---|---|---|
| ALL | 2107 | 0.594 | 0.581 | -0.0133 [-0.0394, +0.0090] not sep | 0.404 / 0.340 | 0.600 / 0.613 |
| INF_ANY_pinned_mask | 371 | 0.507 | 0.511 | +0.0044 [-0.0174, +0.0301] not sep | 0.326 / 0.326 | 0.504 / 0.518 |
| INF_ANY_wide_mask | 400 | 0.501 | 0.509 | +0.0080 [-0.0154, +0.0344] not sep | 0.326 / 0.304 | 0.521 / 0.521 |

## 4. Committed bar

Rule: >= 160 of 197 correct, 0 wrong-side, no ego-future read. Measured: correct 124 of 197, wrong side 3, no future read True. **Bar met: False.**

Controls: K-FEED (the pinned table at NOW equals the N1 dump's nav_in) True; K-REPRO / K-DIFF problems list: [].

Limits: 197 windows in 30 episodes; the audited 75 in 19 episodes. The bootstrap answers another draw of episodes only. The route-text table is the only text source read here; the GPS / OSM route itself is not re-read.
