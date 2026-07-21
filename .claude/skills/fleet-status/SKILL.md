---
name: fleet-status
description: One-shot live status of the whole TanitAD compute fleet (pod1 training, pod2 arms, local, workflows)
---

Produce a compact live fleet status.

## 1. Run the probe — do NOT hand-write greps

```
python tools/fleet_probe.py            # add --json for machine-readable
```

`tools/fleet_probe.py` **discovers** trainers from the live process table and
their logs from the launcher's stdout redirect + mtime. Exit code: `0` all
GREEN, `1` any AMBER, `2` any RED. It prints the table you should report.
Whole-fleet wall-clock ~10 s (measured 2026-07-21).

**This replaced a set of hardcoded greps** (`p0-sB01-realmix.log`,
`arm_base.log`, `arm_kstep.log`, `pgrep -fc train_worldmode[l]`) that pointed at
runs which had been renamed. A grep that matches nothing prints nothing, so the
monitor reported "no anomaly" while GPUs sat idle behind dead trainers — **four
times**, most recently 2026-07-20 05:01 UTC (2 of 4 GPUs down, probe at 04:55
clean). If you find yourself typing a run name or a log path into this file,
that is the bug coming back: fix the probe's discovery instead.

Rules the probe enforces, and you must too:
- **Absence of evidence is an alarm.** A job whose log cannot be found is AMBER
  ("liveness UNVERIFIED"), never GREEN.
- Report the probe's findings verbatim; do not soften an AMBER into "looks fine".
- Never `pgrep -f` / `pkill -f` a trainer name — it self-matches your own ssh
  command and kills your session (CLAUDE.md). The probe never does this.

## 2. Add what the probe cannot see

3. **Local:** CARLA process state (`Get-Process CarlaUE4* -ErrorAction SilentlyContinue`), 4060 free.
4. **Background:** TaskList for running workflows/monitors; CronList for the
   drumbeat + report crons (flag if missing — they expire after 7 days).

## 3. Close with

Anything anomalous in **bold** (every RED/AMBER finding, duplicate trainers,
dead watchdog, missing crons) and the single next scheduled milestone. If the
probe exits non-zero, the fleet status is not "green with notes".
