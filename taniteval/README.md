# TanitEval

The single evaluation harness for the TanitAD driving world-models. One CLI runs
every axis on a named model from the registry, reproducibly, and writes
self-describing JSON that the leaderboard and A/B tooling read.

> **Source of truth.** Model facts (params, status, results) come from
> `Project Steering/MODEL_REGISTRY.md` and the raw `results/*.json` this tool
> writes — never a summary. The registry of *evaluatable arms* is
> `taniteval/registry.py` (`MODELS`), the only place a `--model` key is defined.

---

## The one command

```bash
python -m taniteval.runner <subcommand> [--model KEY] [--episodes N]
```

`runner.py` is **the** entrypoint. Every eval axis is a subcommand; there is no
second script to remember. The standard decision-grade eval of one arm is:

```bash
python -m taniteval.runner run       --model flagship-30k   # open-loop + beyond-ADE + efficiency (inline)
python -m taniteval.runner closedloop --model flagship-30k  # imagination-in-the-loop
python -m taniteval.runner report                           # render the dashboard
```

`run` already emits, in one `results/<key>.json`: open-loop ADE/FDE/miss, the
**beyond-ADE TanitEval-v2 tier-0 `driving` suite** (cruise/transient, along-vs-cross
split, progress, path geometry, heading-by-curvature, kinematic strata — each with
a decision-grade CI and a paired test vs the CV and hold-v0 floors), and a cheap
**efficiency** panel (latency / FLOPs / memory / params). Closed-loop is separate
because it integrates a bicycle model through the planner in the loop
(open-loop ADE does **not** predict closed-loop — measured 0.45 m → 1.69 m on v1).

### Subcommands

| command | what it measures | writes |
|---|---|---|
| `run` / `run-all` | open-loop ADE/FDE/miss + inline `driving` + `efficiency` | `results/<key>.json`, `results/windows_<key>.pt` |
| `driving` / `driving-all` | beyond-ADE tier-0 suite, recomputed offline (CPU, no GPU) from stored windows | `results/driving_<key>.json` |
| `closedloop` / `closedloop-all` | imagination-in-the-loop rollout + compounding-error + stability | `results/closedloop_<key>.json` |
| `closedloop-report` | render the closed-loop markdown from those JSONs | `results/CLOSEDLOOP_REPORT.md` |
| `ab --a K1 --b K2` | paired A/B on the same windows | `results/ab_K1_vs_K2.json` |
| `imagination` / `imag-all` | vision × action ablation, latent fidelity | `results/imag_<key>.json` |
| `hierarchy` / `hier-all` | H26 operative→tactical→strategic cascade (4-brain arms) | `results/hier_<key>.json` |
| `generalize` / `gen-all` | cross-corpus (comma / cosmos / OOD) | `results/gen_<key>.json` |
| `pathspeed` / `pathspeed-all` | decoupled longitudinal/lateral planning quality | `results/pathspeed_<key>.json` |
| `efficiency` / `eff-all` | full precision-sweep + throughput (the deployment axis) | `results/eff_<key>.json` |
| `regression [--update-golden]` | golden-value guard over stored results | `results/golden.json` |
| `report` | render the HTML leaderboard/dashboard | `results/dashboard.html` |

Example arms (`taniteval/registry.py`): `flagship-30k` (v1 FINAL), `flagship-nospeed`
(no-speed ablation control — **not** v1), `refa-dinov2`, `refa-dynin-30k`,
`refb-v2-30k`, `refc-xl-30k`, `refc-base-30k`. 17 arms total.

---

## Install / environment

Use the project venv (`C:/Users/Admin/venvs/tanitad`, py3.13 + torch cu128) or the
pod's `/workspace/venv`. No `pip install` of `taniteval` itself — it runs off
`PYTHONPATH`.

**Dev box** — three paths on `PYTHONPATH` (the `tanitad` package, the
`driving_diagnostic` helper, the `taniteval` package parent):

```bash
export PYTHONPATH="<repo>/stack;<repo>/stack/scripts;<repo>/taniteval"
python -m taniteval.runner run --model flagship-30k
```

**Pod** — the modules self-insert `/root/taniteval` + `/root/TanitAD/stack` +
`/root/TanitAD/stack/scripts`; just:

```bash
cd /root/taniteval && python -m taniteval.runner run --model flagship-30k
```

Running evals needs a GPU + the val episode cache (`/root/valdata/...`); on the
dev box you can import, unit-test, and offline-recompute `driving` from stored
windows, but not run a fresh model rollout.

> **Never eval on a training pod** (adds GPU/RAM load — an eval OOM-killed the
> flagship once) and never on the reserved gate pod. Use the dev box or a free pod.

---

## Production invariants (why numbers here are trustworthy)

1. **Val split is pinned and the leaky one is refused.** The only decision-grade
   physicalai split is the CLEAN held-out `physicalai-val-0c5f7dac3b11`
   (`data.CLEAN_VAL`). The `physicalai-val-f1b378f295ae` split leaks **~78 %** of
   its episodes into the parity train corpus `e438721ae894`;
   `data.list_val_episodes` — the chokepoint every decision-grade command routes
   through — **raises** on it by default. Plus a per-model **leakage guard** drops
   any val episode in that model's own train-ids and refuses a number on < 8
   clean episodes. Parity is sacred: nothing here re-selects canonical episodes.

2. **Intervals use the decision-grade estimator.** New claims use the
   **episode-cluster bootstrap** over the 40 val episodes
   (`ci.episode_cluster_bootstrap`; paired arms → `ci.paired_episode_cluster_bootstrap`),
   which the `driving` panel emits. The old `bench` `±ci95` is
   `overlapping_holdout_se` — **DEPRECATED, 1.28–2.06× too narrow, not a
   jackknife** — kept only so historically published intervals stay reproducible.
   Never quote it for a decision. See `ci.py` for the full rationale.

3. **The viz standard is the default.** `corpus_overlay.py` ("THE STANDARD")
   renders camera projection **+** metric BEV inset **+** decoded tactical
   maneuver **+** strategic route/goal HUD **+** per-frame ADE together, per
   Sayed's standing preference; BEV-only is the fallback when camera calibration
   is unrecoverable.

---

## Tests

```bash
python -m pytest taniteval/tests/ -q      # 153 passing, ~17 s, CPU-only
```

`taniteval/conftest.py` bootstraps `PYTHONPATH` from its own location, so the
suite runs green on the dev box **and** the pod with no preset env. (`tests/run_all.py`
is the older bespoke runner; pytest is canonical.)
