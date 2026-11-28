# PREREG_NAVRB (written before any number)

Question: can the route text give the exit side of a roundabout, so a text nav could name the turn on the 43 roundabout windows the v9 nav names?

Windows: the ROUTE-LACKS-TURN windows of the 6 roundabout episodes of RESULT_NAVAUDIT (43). Label: the v9 label side (TURN_L / TURN_R), a label-side quantity.

Route-only rule R1 (fixed here, right-hand traffic convention, nothing fitted): the latest router row at or before the window's NOW step that carries a roundabout exit number n (`At roundabout, take exit n`, `navtok.py:32`); n = 1 -> R, n = 3 -> L, any other n or no row -> no side. It reads table rows only, never the track or any row after NOW.
Sensitivities (reported, do not set the reading): R1 on the current row only; the same convention with router rows up to 8 s ahead (not causal at inference); constant R and constant L (context for the bar).

Readings:
* RB-SIDE-FROM-ROUTE: R1 names the v9 side on >= 90 % of the 43 windows, with 0 ego-future reads (a future-read mutant is RED).
* RB-STATE-ONLY: otherwise, and a roundabout state (class RB_* or ROTARY_EXIT, flag 0) is the router row at NOW on >= 90 % of the 43.
* RB-ABSENT: otherwise.
Interval: episode-cluster bootstrap (6 episodes: descriptive only). Evidence class MEASURED, CPU, zero training.
