# DELIVERABLE MANIFEST — 2026-07-27 imagination/perception manifold research

**Stream:** Architecture & Inference — research. **Date:** 2026-07-27 (Europe/Berlin).
**Everything below is in the repo working tree and STAGED (`git add`). Nothing was committed. Nothing was
pushed. No branch was switched.**
**Deliverable path:** `TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-imagination-perception-manifold/`

## 1. Artifacts

| artifact | what it is | where it lives |
|---|---|---|
| `PRE_REGISTRATION.md` | ⭐ written before any fix-evaluation literature was read, with an honest disclosure of exactly what preceded it. Fixes the defect definition, the evidence rules, **the four falsifiers F1–F4 for "not fixable without a retrain"**, the ranking rule, and a declared confound in my own reading | **repo** |
| `MANIFOLD_MISMATCH_RESEARCH.md` | ⭐⭐ the main report — verdict · pre-registration adjudicated · **§2 the zero-compute F4 result and the no-speed attributing ablation** · §3 is-it-named · **§4 the per-architecture posterior-vs-prior decode table** · §5 ten ranked fixes · §6 five experiments with pre-registered falsifiers · §7 horizon mismatch · §8 driving world models · §9 limitations · §10 amendments | **repo** |
| `CITATIONS.md` | the citation table with a **verification-tier column that distinguishes what I fetched myself from what came via a search extract or a delegated agent**, plus **§D "what I searched for and did NOT find"** and **§F the delegated-search provenance with its failures preserved** | **repo** |
| `MANIFEST.md` | this file | **repo** |

**Nothing produced by this stream lives in only one place.** No pod file, no worktree file, no
scratchpad-only artifact. Four markdown files, all in the repo, all staged.

## 2. ⛔ What was NOT produced, declared rather than discovered later

* **No code.** The five experiments in §6 are **designed, not implemented**. X1 needs a latent-caching
  script that does not exist yet.
* **No new measurement of the model.** Per `PRE_REGISTRATION.md` §5 this stream launches nothing. Every
  number is read from a committed artifact, a committed training log, or the source code.
* **No pod contact.** pod1 (training), pod2 (owed controls), pod3 (classifier build) and the eval pod
  (trafficsim) were **never contacted**. No GPU was used. Total compute: one 12 MB `torch.load` on the dev
  box CPU and some JSON parsing.
* **No registry edit.** §3 of the escalations below asks for one; I did not make it.

## 3. 🔴 ESCALATIONS — raised here, in the headline, not written into a README

**E1. `grounding.invdyn[*]` is a real-pair decoder that already exists in every 4-brain checkpoint, and its
error has been logged 620× per run since the first flagship run — unread.** `g_op_mid_de_m` vs
`g_op_fwd_ade_m` is a **perfectly paired** real-vs-imagined decode contrast on the same training batch.
**It says the real-pair path plateaus at 1.0129 m while the imagined path reaches 0.0304 m, and that the
gap is manufactured by training (2.36× → 33.3×).** ⛔ **This must not sit in a document. It either goes into
`MODEL_REGISTRY.md` as a v1 row, or it is explicitly declined.** It is free to verify — §6 X4 re-runs the
band analysis over every committed log in ~5 CPU-minutes and **should be run on v4 before v4 inherits any
conclusion from v1.**

**E2. A registry/leaderboard decision is owed, and this stream sharpens one already raised.**
`BLIND_IMAGINATION.md` §7.1 E2 asked whether the headline metric keeps the `op` decoder or moves to the
calibrated one. §7 of this report says **both answers are wrong**: the design four independent SOTA
forecasting systems converged on is **a bank of horizon-specialised readouts selected by lead time**
(GraphCast recommends it; FuXi and Pangu-Weather ship it), and **we already own the bank**. The crossover
is already measured. ⇒ **the decision to make is not "which single readout" but "what selection rule",
and it costs zero GPU.** (Candidate **C8**, §5.)

**E3. Candidate C1 is ~10 lines and ≈0 % step time, and it must be decided BEFORE the next flagship run
starts — not after.** Symmetric/mixed-source supervision of the metric head needs `fut_states`, which
`grounding_losses` **already computes** for term (a), and needs no extra encode and no extra rollout. It is
the TD-MPC2 pattern (Eq. 3) applied one level up, and it is the only fix in §5 that addresses the defect at
its locus at negligible cost. ⛔ **If a run launches without it, the fix waits a full run cycle.**
⚠️ **Conditional on X1** (§6.1): if X1 shows the metric ego-motion is not in the latent, C1 will not help
and **C3 (encoder-level ego-motion supervision) is the item instead** — which is why X1 is worth 20 GPU-min
before any run.

**E4. The `MODEL_REGISTRY` claim that our metric readout is a perception decoder needs qualifying.**
`HYPOTHESIS` with three MEASURED supports: it may be an **action-integrator** reading the injected `v0`.
⛔ **Do not enter this in the registry until X2 (§6, ~20 GPU-min) runs the surgical test** — but do not let
the peek/duty-cycle/re-anchoring streams keep building on the assumption either.

**E5. Publication claim, if the program wants one.** Valdi ([arXiv:2607.00917](https://arxiv.org/abs/2607.00917))
asserts our exact design choice is safe and never measures it; **no driving world model has a metric head
trained exclusively on imagination-rollout pairs**; and the occupancy family reports 2.0–3.9× recon-vs-
forecast gaps without ever disentangling decoder-transfer from future uncertainty — which our
matched-timestep instrument does. ⇒ **there is a paper-sized, defensible, already-measured contribution
here.** Raised as a program decision, not acted on.

## 4. Integration this work needs

| what | who owns it | why it will otherwise be missed |
|---|---|---|
| the F4 real-vs-imagined row → `Project Steering/MODEL_REGISTRY.md` (v1, and v4 after X4) | registry owner | it is a *property of the deployed decoder* that every grounded number in the program inherits |
| the C8 selection rule → `taniteval/rollout.py::collect` + both `canary_rollout`s | eval-harness owner | it changes the headline metric's definition and cannot be done silently |
| C1 → `stack/tanitad/models/metric_dynamics.py::grounding_losses` | trainer owner | ⛔ **must land before the next run starts** |
| X1/X2 → a new eval-only script | whoever next has pod2 or the eval pod idle | ~20 GPU-min each; X1 gates C1 vs C3 |

## 5. Reproduction — everything in this report, end to end, on the dev box

*(no GPU, no pod; `python` = `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` for the torch step only)*

```
# §2.2 — the F4 result and its trend (base python is enough; stdlib json only)
python -c "import json,statistics; rows=[json.loads(l) for l in open('taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl') if l.strip()]; \
sel=[r for r in rows if 28000<=r['step']<31000]; \
print('op_mid(REAL) %.4f  op_fwd(IMAG) %.4f  n=%d'%(statistics.mean(r['g_op_mid_de_m'] for r in sel), statistics.mean(r['g_op_fwd_ade_m'] for r in sel), len(sel)))"

# §2.3 — the attributing ablation (same command, nospeed-phase0_train_log.jsonl, band 19000-21000)
# §2.3 — the config diff proving only 3 fields differ:
#        flatten both *_config.json and compare keys (see the report's §2.3 for the result)

# §2.2 — the mean-speed baseline (needs torch)
"C:/Users/Admin/venvs/tanitad/Scripts/python.exe" -c "import torch; \
d=torch.load('TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt',map_location='cpu',weights_only=False); \
s=d['speed'].double(); print('n=%d mean=%.4f std=%.4f'%(s.numel(),s.mean(),s.std()))"
```

**Expected:** `op_mid(REAL) 1.0129  op_fwd(IMAG) 0.0304  n=41` · `n=599 mean=12.8997 std=9.7985`.

## 6. ⚠️ Git staging note for whoever commits this

`git status --short` with a **scoped pathspec** showed only one of the three files as staged, which looks
like a staging failure. It is not: **`git ls-files --stage -- <dir>` confirms all four are in the index.**
Verify with `git ls-files --stage`, not scoped `git status`, before concluding anything is missing.
`core.fsmonitor` and `core.untrackedCache` are both already `false`; no `.gitignore` rule matches
(`git check-ignore` exits 1).
