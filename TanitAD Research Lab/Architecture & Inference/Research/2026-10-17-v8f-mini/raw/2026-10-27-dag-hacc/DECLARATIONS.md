# DECLARATIONS (PREREG_V8F_DAG_HACC executor), written BEFORE any DAG plan on W_fit, any DAG fit, bake or rollout existed

Interpretations the PREREG text leaves open. Fixed here, in writing, before the first DAG number. None is changed after a number is seen.
The M2b-side controls (C-H1, C-H2) were the only things run before this file (read-only inspection of the repo export and of the checkpoint structure).

* **D1 (lambda0's reference weight).** HACC's lambda0 = (position term)/(D2 term) "at HBLEND's banked w". HBLEND's w is the output of ONE recipe
  (`v8f_hblend.fit_weights`: closed form per step and axis, clip to [0,1], then isotonic) applied to the arm's own plans. The PREREG changes
  "only the plans (DAG's in place of M2b's)", so the reference weight for DAG is that same recipe applied to DAG's OWN T0 plans on W_fit.
  M2b's banked weights are NOT used for DAG (that would put M2b's plans into DAG's fit). The literal alternative (M2b's banked w as the
  reference) is computed BESIDE, as a sensitivity on lambda*, and decides nothing.
* **D2 (how DAG's T0 plans on W_fit are produced).** Exactly HBLEND's `cmd_fit` route with only the checkpoint changed: the 141 TRAIN clips of
  `fit_clips.json` (k = 32), `train_windows(hist, start)` of the checkpoint's own config (9,582 windows expected), ages = the cycle schedule
  (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = index in the window list), `forward_infer` (p_gt 0, T's own code, no label read),
  tau := traj30, P := p10, y := gt. The plan-production route is checked on M2b first: it must reproduce HBLEND's banked
  `fit_windows_M2b_s{0,1}.npz` (traj30, p30, gt, ages) before it is used for DAG (control C-X1, added; it touches no bar).
* **D3 (the code that runs).** `v8f_hacc.py`, `v8f_hblend.py`, `v8f_hspec.py`, `accel_profile.py` exported from tip 2a416ca7. Only their hard-coded
  scratchpad path constants are re-pointed (a patch script with asserted substitution counts; diff in `code/PATCHES.md`). No arithmetic is edited.
* **D4 (the bake script).** `bake_hacc.py` pins M2b's source checkpoint sha256, the M2b weight tables and the banked M2b HACC plan. Those pins are
  M2b-specific and cannot hold for DAG. The bake MECHANISM (`V8FMini.load_hacc_weights`, then `torch.save({**ck, "model": state_dict, "hacc": prov})`) is
  run unchanged; only the pins are replaced by DAG's (source sha256, DAG weight table, hash of the weights JSON). The M2b-banked-HACC agreement check (its step 5)
  has no DAG analogue and is replaced by C-H3 / C-H4. Reported as a deviation in the result.
* **D5 (C-H3's windows).** "24 eval windows" = the first 24 windows of eval-141 in the harness's own order, plus ALL 2,399 beside.
* **D6 (the logged-limit dumps).** Read A's `t1kf_read_a1` reader consumes three dump configs per seed (decisive, `--limit logged`, strict). The PREREG
  names two decisive rollouts and STRICT beside; the `--limit logged` rollouts (same flags, `--limit logged`) are added BESIDE so the unchanged reader can run.
  They decide nothing.
