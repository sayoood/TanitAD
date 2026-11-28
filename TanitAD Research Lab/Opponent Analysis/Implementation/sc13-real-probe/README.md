# SC-13 real-window anticipation probe (Opponent Analyzer, runs #4 → #5)

> ## ⛔ READ FIRST — this probe is RETIRED as an anticipation test (run #5, 2026-08-02)
>
> **Run #5 answered the question and the answer is not the one SC-13 was built to argue.** Use
> `sc13_probe_v5.py` + `sc13_analyse_v5.py`; the run-#4 pair below is kept for provenance only.
>
> **Verdict.** Pre-registered falsifier **F-A (volume) did NOT fire** — at stride 1
> (**6,444 anchors, n=44 events / 15 episodes**, double run #4) the effect held at
> `held − reactive` **+0.281**, episode-cluster CI **[+0.009, +0.562]**. So run #4's in-domain positive
> was **not** small-n noise. **But the SURVIVAL condition was NOT met:** `held − blind` +0.064
> CI [−0.019, +0.162] and `held − shuffled` +0.102 CI [−0.011, +0.245] **both include 0** ⇒
> **vision attribution is NOT established.**
>
> **Why, in one table** (speed-matched AUROC, BRAKE_FAR, stride 1):
>
> | arm | sees | matched | share of the gap over `reactive` |
> |---|---|---:|---:|
> | `reactive` | no model | 0.455 | — |
> | `shuffled` | a **different episode's** real window | 0.634 | **≈64 %** |
> | `frozen` | own last real frame ×8, **no motion** | 0.723 | ≈32 % |
> | `held` | real scene + motion | **0.736** | ≈5 % |
>
> ⇒ the signal is a **static-frame + ego-kinematic** property, **not** a rolled-forward consequence.
> Vision still matters for *accuracy* (2 s ADE held 1.186 m vs shuffled 1.321 vs CV 1.743) — it is just
> not what makes this detector work. `held` also **exceeds `gt_oracle` (0.620)**, i.e. it beats the
> score computed from the true future trajectory, which is a property of the `CV − pred` construction
> rather than of foresight.
>
> **Two instrument lessons worth more than the result.**
> 1. **The estimator decided the verdict.** On run #4's **anchor-level** bootstrap both vision
>    differences EXCLUDE 0 and this would read "confirmed". 44 events live in **15 episodes** and are
>    not 44 independent facts. `sc13_analyse_v5.py` reports the **episode-cluster** bootstrap as
>    decision-grade and keeps the anchor-level one only for comparability. Same class as the
>    `overlapping_holdout_se` retraction in `CLAUDE.md` — caught **before** publication this time.
> 2. **The mean-frame `blind` control was the weakest one.** Being off-manifold, it *understated*
>    nothing — it scored **higher** (0.672) than the on-manifold `shuffled` control (0.634). A control
>    that breaks the encoder is not a control that removes the information.
>
> **What still has value:** `D = CV_fwd − pred_fwd` as a cheap label-free **monitor feature** — it does
> beat a naive deceleration floor in-domain with a CI-separated margin. It must **not** be labelled
> vision- or imagination-driven, its edge over a plain ego-kinematic feature is **unproven**, and it
> needs run #4's competence guard (it was unreliable on comma2k19, where the model loses to CV).
>
> **What would actually settle the H15 claim:** the **closed loop** — not more open-loop anchors.

## Run #5 files

- `sc13_probe_v5.py` — 5 rollout arms (`informed`/`held`/`blind`/`shuffled`/`frozen`), stride 1, records
  each anchor's **start index and episode** so any earlier stride is recoverable as an exact subset.
- `sc13_analyse_v5.py` — speed-matched + stratified AUROCs, **episode-cluster** and anchor-level
  bootstraps, **paired** difference CIs, and the pre-registered verdict computed in-script.
- `results/sc13_v1_stride1_analysis_{all,stride2}.json` — `all` is the stride-1 result; **`stride2` is
  the exact re-derivation of run #4's anchor set**, which reproduced it to three decimals
  (held 0.723 / blind 0.653 / reactive 0.434 / informed 0.680 / gt_oracle 0.633).
- `results/sc13_v1_stride1_windows.pt` (1.3 MB) — **the raw substrate, banked in-repo.** Run #4's
  substrate was left on the eval pod and was **lost when the pod was re-provisioned**; every future
  re-analysis off this file is free and survives the next re-provision.

```bash
# eval pod; note the stack lives at /workspace/TanitAD (NOT /root/TanitAD) after re-provisioning
PYTHONPATH=/workspace/TanitAD/taniteval:/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
  python3 sc13_probe_v5.py --model flagship-30k --ckpt /workspace/v1_modelonly.pt \
    --episodes 40 --stride 1 --out /workspace/sc13v5/sc13_v1_stride1.json
python3 sc13_analyse_v5.py /workspace/sc13v5/sc13_v1_stride1_windows.pt --boot 2000
```

`--ckpt` overrides the registry path because the registry still points at `/root/models/…`, which **no
longer exists** on the re-provisioned pod. Runtime ~18 min for 40 episodes at stride 1 on the A40.

---

# Provenance: the original run-#4 probe

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
