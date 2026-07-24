# TanitAD — working agreements

Sub-300M hierarchical 4-brain latent world model for autonomous driving. PI: Sayed.

## Source of truth (this rule exists because prose lied to us)

**`Project Steering/MODEL_REGISTRY.md` is the ONLY quotable source for model facts** —
architecture, params, training args, parity key, results, status. Any number in any report
cites the registry or the **raw eval JSON**, never a summary, changelog, or weekly report.

Three errors propagated for days because they were copied from prose:
- `flagship4b-phase0-30k` is the **no-speed ablation control** (2.918 m), NOT the deployed v1
  (that is **`flagship4b-speedjerk-30k`**, 0.452 m). The HF repo name invites this inversion.
- "REF-B v2 died at 22,600" — it did not; `metrics.json` says step **29999**. The *log* went stale.
- REF-A **I-JEPA's val number is unusable**: ~80% of val leaked into its train set.

If a doc and the registry disagree, the registry wins and the doc gets fixed.

**Never quote a learning-curve exponent bare.** Any exponent carries its **fit window, R² and n**,
or it is not admissible — and it may never decide a restart. Below **R² 0.80** there is no quotable
exponent at all; use the matched-step ratio. Never extrapolate more than **2×** beyond the fitted
range, and never compare exponents fit over different windows. *(Measured 2026-07-20: the same
`g_op_fwd_ade_m` log gives −0.387/−0.505/−0.564/−0.621/−0.738 depending on the window, all at
R² 0.09–0.58; v1's reference "−0.84" is the 1500–7500 window at R² 0.541, and on matched windows v1
and v3enc are statistically indistinguishable.)* Restart/continue decisions follow
`Project Steering/GATE_PROTOCOL.md` via `stack/scripts/run_gate.py`.

**Never quote an interval without its estimator.** The block historically labelled *"8-split
episode-disjoint jackknife"* is neither a jackknife nor a valid SE — it is
`overlapping_holdout_se`, and it is **1.28–2.06× too narrow** (measured across 10 arms). The
decision-grade interval is the **episode-cluster bootstrap** over the 40 val episodes
(`taniteval/ci.py`); for two arms on the same windows use the **paired** version, never a
combination in quadrature.

## Briefing a subagent — the contract

Every subagent brief MUST carry the preamble in
`Project Steering/AGENT_OPERATING_STANDARD.md`. Its three binding rules:

1. **Stage, never push.** Agents `git add` their deliverables into the working tree and
   **never commit and never push**. They must NOT leave work only on a pod or only in a
   worktree. *(The old "commit nothing" default stranded REF-B v2's architecture, the entire
   TanitEval harness, the pod ops bundle, and 486 lines of TanitResim — each on a single disk.)*
2. **End with a deliverable manifest** — every artifact and **where it lives**
   (repo path / pod:path / worktree). Stranding must be visible in the report, not discovered
   in an audit months later.
3. **Escalate integration, don't write "please merge" into a doc.** An orthogonality instrument
   sat unmerged for **10 days** because the request lived in a README nobody re-read.

## Traps preflight (each of these has cost hours more than once)

- **`pgrep -f <trainer>` / `pkill -f <trainer>` self-matches your own ssh command** and kills your
  session — returns empty output and looks like nothing happened. Kill by **explicit PID**.
- **`PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED** on pods or trainers die with
  `ModuleNotFound: tanitad`. `cd` alone is not enough.
- **Never judge pod disk with `df`.** It reports the 965 TB cluster and hides the per-pod MooseFS
  quota. Use a real `dd` write test. A full quota killed the flagship mid-checkpoint.
- **`step_s` in trainer logs is ACCUMULATED over `--log-every`** (÷50), not per-step. This has
  caused false "training is 430 s/step" alarms.
- **Moving multi-GB files between pods:** pods cannot SSH each other, and the dev-box relay is
  ~1 MB/s. Push → HF from the source pod (~118 MB/s), then pull. Verify md5.
- **A RunPod volume resize stops the pod and reassigns its SSH port** (`Connection refused`, not
  `timed out`). The working key is `~/.ssh/tanitad_pod`, not the console's `id_ed25519`.
- **Verify before alarming.** Check the metric's definition and take multiple samples first;
  several "outages" were measurement artifacts.

## Git hygiene — the mistake that has now happened twice

**`git commit` and `git commit --amend` both commit the ENTIRE INDEX, not the files you
just `git add`ed.** When several agents stage work concurrently — the normal state here —
a "quick commit of my thing" silently sweeps in a sibling's half-finished code under the
wrong message. This has happened twice in one session (`60265d3` swallowed the eval
tooling; `3d41bd0` swallowed REF-C v1.2's in-progress rescorer).

**Rule: when the index contains other agents' work, commit with an explicit pathspec and
do NOT follow it with `--amend`:**
```
git commit -F <msgfile> -- <path1> <path2>        # pathspec form, no amend afterwards
```
Check `git status --short` for foreign staged entries FIRST. If a long message is needed,
write it to a file and pass `-F`, because the `--only ... && --amend` pattern re-opens the
whole index and defeats the pathspec.

## Operating standard — raised by Sayed 2026-07-21

The program's pace goes up, and so does the bar. Five rules, each with the failure that earned it.

**1. State the evidence class or don't state the claim.** Every number carries
`MEASURED (ours + artifact path)` · `PUBLISHED (cited)` · `INHERITED (another agent/doc, NOT
re-verified)` · `ESTIMATED` · `HYPOTHESIS`. **A claim that decides a GPU-day must be MEASURED or
PUBLISHED — never INHERITED.** *(2026-07-21 alone: five retractions, every one from quoting a
faster-moving source than the harness. "v1.6 is best-in-program" was a **trainer log**, ~10 % optimistic
vs `eval_*.py`. Trainer val watches a curve; only eval output is quotable.)*

**2. Absence found at ONE location is not absence.** Before writing "X does not exist", probe a second
path, a second name, and the tool that owns the fact. *(Cost this session: the Vulkan ICD is in
`/etc/vulkan/icd.d/`, not `/usr/share/` → "our pods cannot render" stood for **12 days** and blocked
AlpaSim + CARLA. `ps -C python3` returns EMPTY for a healthy job because pods run
`/workspace/venv/bin/python` → a near-miss "the VLM job is dead". `obstacle.offline` — 3D agent tracks
on **96.90 %** of our corpus — was declared non-existent for days; our ingest reads 2 of 36 features.)*

**3. Finish before you start. An artifact on one disk or in one agent's context is NOT done.**
Definition of done = **in the repo, staged, with its provenance**. *(LAL-v2 anticipation: implemented,
tested, **unmerged 12 days**. An orthogonality instrument: **10 days**. TanitEval, REF-B v2's
architecture, the pod ops bundle — each stranded on a single disk.)*

**4. Retractions are the learning mechanism — log the ROOT-CAUSE CLASS, not just the correction.**
`Project Steering/RETRACTION_LOG.md` is append-only and **must be read before asserting in a known
class**. A retraction with no class taught nobody anything.

**5. Aim above the published state of the art, and settle conflicts with experiments, not deference.**
When ambition meets inconvenient evidence, the answer is the **cheapest discriminating experiment**,
pre-registered with **both outcomes committed in advance** — not a scoped-down goal. *(The "strategic
choice is a ~2 % lever" refusal was **confounded**: REF-C evaluates with `nav_cmd=None`, so a decoder
that never had a working route input learned the marginal. I nearly designed the hierarchy away on it.)*

**Orchestration.** Parallel streams are the default, but: every brief carries a **priority order** so a
killed agent still yields value; agents **bank incrementally** rather than holding a final synthesis;
and **fan-out is capped** — uncontrolled sub-spawning exhausted the weekly API budget on 2026-07-21 and
cost three agents' work.

## Invariants

- **`Keys.txt` is git-ignored — NEVER commit it.** Read tokens in place
  (`grep -oE 'hf_[A-Za-z0-9]+'`); never copy, print, or write them to args.
- **Agents never commit to `main`** and never edit `Project Steering/Mission Plan.md`.
- **Parity is sacred:** the canonical train corpus is `physicalai-train-e438721ae894`
  (2376 episodes) with skip-hash `f09e44db`. Anything that re-selects episodes breaks
  cross-arm comparability and must be refused.
- **Never add GPU/RAM load to a pod that is training**, and never eval on a training pod.
- Full suite lives at `stack/` — `pytest -q` must stay green before any commit.
