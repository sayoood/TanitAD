# PREREG_NAVAUDIT (written before the wide nav is scored; 2026-11-17)

Variant: the landed builder (`build_navtext_nav.py`, as in `F:/TanitAD-scratch/strat1/build/tree/stack/scripts/`) with ONLY `D_ANN_MIN` 30 -> 60 and `D_ANN_PER_MS` 6 -> 8, i.e. `d_ann = max(60 m, 8 s x v)`. Nothing else is touched.
Proof of that: the same wrapper at 30 / 6 reproduces the pinned `navtext_nav.npz` arrays (train and eval) exactly; wide vs pinned differ only in the one-hot token at flag-0 side steps whose countdown lies between the two cuts; ch3, ch4, nav_d identical.
Scored: on the 197 TURN windows (T1-kin tick 0, 141-episode grid): wide text names the correct turn at NOW (count), names the other side (count), distance and time-to-turn vs v9; nav-only out-of-fold HGB (navecho2) on the 5 channels, wide vs pinned text, paired episode-cluster bootstrap B 2,000 seed 0.
Bar for a retrain proposal to the PI: wide text names the correct turn on >= 160 of 197 windows AND 0 wrong-side namings AND reads no ego-future field.
Otherwise: report the next lever, no proposal.
