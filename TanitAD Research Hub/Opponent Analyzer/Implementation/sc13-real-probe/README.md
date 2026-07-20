# SC-13 real-window anticipation probe (Opponent Analyzer, run #4)

**Not an intake package.** This is the *experiment* behind
`Research/2026-08-07-opponent-sweep-w5.md` §1, archived so the numbers are reproducible. Nothing here
is proposed for `stack/`. (If Benchmarks & Eval adopts `D = CV_fwd − pred_fwd` as a monitor feature —
recommended — that goes through a normal intake package, authored by them.)

## What it answers

SC-13's numbers were a **design oracle**. This asks whether **our own checkpoint**, on **real held-out
windows**, shows any anticipation of an upcoming deceleration — and whether that beats (a) a
detection-free **reactive** floor and (b) a **vision-blind** control.

## Run it

On the eval pod (`ssh tanitad-eval`):

```bash
cd /root/taniteval
PYTHONPATH=/root/taniteval:/root/TanitAD/stack python3 sc13_real_probe.py \
    --model flagship-30k --episodes 40 --stride 2 \
    --out /root/taniteval/results/sc13_flagship30k.json
# then the confound control (reads the saved *_windows.pt, no model re-run):
PYTHONPATH=/root/taniteval:/root/TanitAD/stack python3 sc13_speedmatch.py \
    /root/taniteval/results/sc13_flagship30k_windows.pt
```

`--model` takes any key from `taniteval.registry.MODELS`; `--val` switches corpus (e.g.
`/root/valdata/comma2k19-val-76b6e94a97a1`). Runtime ~5–6 min for 40 PhysicalAI episodes on the A40.
`sc13_real_probe.py` writes `<out>` **and** `<out stem>_windows.pt` — the raw per-anchor substrate, so
every re-analysis (new labels, new thresholds, new controls) is free.

## Results in `results/`

- `sc13_flagship30k.json` — raw AUROCs + bootstrap CIs + anchor counts, flagship-30k / PhysicalAI val.
- `sc13_flagship30k_speedmatched.json` — the same, with the speed confound removed by per-event
  ±1 m/s matching and by v0-stratification. **Read this one for the verdict.**

## Read the numbers correctly

- **`informed` LEAKS.** It feeds the true future actions, i.e. the braking command itself. It exists
  only as an upper bound. The first version of this probe reported `informed` as the result and scored
  AUROC 1.00; that is leakage, not anticipation. **`held` is the claim.**
- **Use the speed-matched numbers.** Braking anchors sit at median 8.94 m/s and cruise controls at
  17.34 m/s; the raw AUROC is partly a speed effect.
- **`blind` is the control that matters.** It is the difference between "our world model anticipates"
  and "ego kinematics anticipate". At n=23 BRAKE_FAR events the gap (+0.07) is inside the bootstrap CI
  — **not** a resolved result. Do not quote `held` without `blind` next to it.
- The `blind` arm uses a **constant mean frame**, which is far off-manifold and may *understate*
  vision. A shuffled-real-frame and a temporally-frozen control are queued for run #5.
