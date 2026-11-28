# NT A1 zero-gradient gate FAIL -- offenders vs the launch gate's pre-registered dead groups (MM, 2026-10-26 ~20:28 Berlin)

* `zerograd_gate.json` -- the supervisor's post-run gate record, copied read-only from Thor `/home/nvidia/r8ns/nt/runs/nt-a1-s0/` (verdict FAIL, 21 offenders, every `allowed_by` null).
* `zg_vs_glive.py` -- matches each offender against `stack/scripts/launch_gate.py` (tip b8b4f08) profile `refcv8-nt`'s
  `live_dead_admitted` groups whose flag is in the run's own argv (`config.json`, not shipped: 5 MB): `core.decoder.r8_rc_to_cond`
  (`--r8-no-rc`, refcv8 profile, PI Q-A 2026-10-11) and the 12 `_NT_MAP_ONLY_GROUPS` (`--map-min-coverage`, value 0.0, K4).
* `zg_vs_glive.json` -- result: **20 / 21 offenders (106,952 / 106,953 values) are inside a group admitted dead BEFORE launch**;
  the one outside is `core.decoder.r8_rc_gate` (1 scalar). `stack/tanitad/refs/refc.py:3887-3891` applies it only under
  `if r8.get('rc') is not None`, so `--r8-no-rc` makes it dead by the same construction; G-LIVE groups by module, and a bare
  Parameter on `core.decoder` sits in a group that does train, so the launch gate could not list it.

Cause of the FAIL: the supervisor's `ZEROGRAD_ALLOW` was never filled from the launch gate's admitted list. Two instruments, two
allow-lists, not reconciled. NT A1 is the first refcv8-family run under the post-run gate; `refcv8-r101-s0`'s `summary.json` carries no
`zerograd_gate` record. The map-only fine-tune is not exposed: `mapft_prelaunch.sh:344-345` refuses `fire` unless a zero-gradient
dress rehearsal on the smoke checkpoint PASSes.
