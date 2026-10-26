# T1 S/T read: fact sheet (@81aa3e36; [S]=source, [M]=measured by me on banked files, CPU)
E=stack/scripts/v8f_mini_eval.py D=stack/tanitad/data/v8f_mini_data.py C=taniteval/taniteval/v8f_consistency.py T1=stack/scripts/v8f_t1_eval.py M8=stack/tanitad/models/v8f_mini.py P=TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini G=Project Steering/GOALS_AND_CLAIMS.md

**1. S.** Columns (x m, y m, psi rad) [D:134-146] at ARC 30/50/80 m of the ego's own future path (sigma-8 m smoothed, not times) [D:50; v9 SPEC.md:337]; about 2.9/4.9/7.8 s at median 10.3 m/s [M]. GT = v9 eval139 `rc_A30/A50/A80_{x,y,psi,valid}` [D:211], NOW row, NOW frame [D:735]. T0 metric `checkpoint_metrics` [E:110] -> `checkpoint_errors` [C:294]: `strategic.pos_err_m@{30,50,80}`, `head_err_deg@{30,50,80}`, `n_valid`.
**Clash:** T0 scores held outputs (ages a_t=i%3, a_s=a_t+(i//3)%5 [E:293]) against NOW labels; trainer uses row t-age [M8:849-851]; NOTES.md:22 uses j-age. Gap = "staleness artefact", 0.8-0.9 m per m driven [P/raw/2026-10-25-sfresh/SFRESH_RESULT.md:58].

**2. T.** `tactical_metrics` [E:121-143] reads X has, lat_cls, lon_cls, goal_y, goal_w [D:732-740] from v9 eval139 [D:214]. Lat 5 (LANE_KEEP, TURN_L/R, LANE_CHANGE_L/R) [D:56]; lon 7 (HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP) [D:57]; goals 22 [D:52]. Names: `tactical.{lat,lon}.macro_f1`, `goal_macro_f1`. Labels exist at EVERY cache row [D:192-215; M: 13,896/14,237 rows]; only X samples grid rows [D:725]. Eval lat support just 1910/47/150 [M]. Grid t0 = 3,7,...,71 (stride 4), 2399 windows [M]; cache step 0.2 s [D:49].

**3. Ticks.** m=10 gives j = t0, t0+10, t0+20 [T1:399-410], (age_t, age_s) = (0,0), (1,4), (2,2) [T1:136-143]. S label rows t0, t0+6, t0+18; T rows t0, t0+9, t0+18; same join, no clip-end loss (max row 91, min T 100) [M]. 34 windows (2 clips lack v9 rows) lose everything. Valid [M]: RC@30/50/80 2322/2319/2294, 2321/2313/2290, 2313/2298/2275; T lat/lon/goal 2107/2240/2343, 2056/2099/2314, 2005/1836/2271. Only plan-GT clips (`gt_ok` 2399/2117/1694, [T1:317]). Tick 0 = T0 `fresh`. Raw `s_ckpt` is in the EXECUTED frame, labels in the LOGGED frame; equal only at tick 0.

**4. Baselines.** S: `nav_only_checkpoints` [C:253] (P10 path + R=12 m turn; needs dead-reckoned nav_d, arc not dumped [T1:219]); pos 1.9909/4.7297/10.9951 m, head 7.64/10.78/16.83 deg, deterministic [P/raw/full/eval/floors_report.json `nav_only_baseline`]. T: majority F1 lat .3170, lon .0707, goal 0 [eval JSON `raw.tactical`]. `v8f_kscore.py` constants are plan-ADE only.

**5. Banked M2b** (HACC post-maps the plan only [M8:773-793]); P/raw/full/eval/eval_M2b_s{0,1}.json, P/raw/2026-10-25-sfresh/eval/eval_M2b_s{0,1}-sfresh-fresh.json. s0 ; s1:
| | pos m @30/50/80 | head deg @30/50/80 | T lat/lon/goal F1 |
|---|---|---|---|
| fresh | .977/2.296/5.887 ; .929/2.204/5.896 | 3.67/6.06/12.12 ; 3.44/6.00/12.52 | .8352/.5062/.2620 ; .8476/.5135/.2790 |
| cycle | 7.023/8.012/11.206 ; 6.980/7.938/11.214 | 4.20/6.69/13.05 ; 4.04/6.72/13.60 | .8157/.4992/.2618 ; .8292/.5095/.2621 |
Seed floor |s0-s1|, fresh [M]: pos .048/.093/.009, head .23/.059/.40, lat .0124, lon .0073, goal .0170; ADE6 .1207.

**6. Register.** No row calls S wrong. Only text: code stub "see H-MINI-STRAT-1 (checkpoint error)" [E:94]. STRAT-1 [G:16845]: clause 1 "FAIL ... comparability is UNRESOLVED", then "STALENESS ARTEFACT ... PASSES on fresh ages"; rest OPEN. Strategic un-paused for v8F [G:16840]. The gap you fill: "STRATEGIC and the model's own T code: NOT COMPUTED" [P/RESULT_V8F_T1KIN_HACC.md:27].
