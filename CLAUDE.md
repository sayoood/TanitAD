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

## Invariants

- **`Keys.txt` is git-ignored — NEVER commit it.** Read tokens in place
  (`grep -oE 'hf_[A-Za-z0-9]+'`); never copy, print, or write them to args.
- **Agents never commit to `main`** and never edit `Project Steering/Mission Plan.md`.
- **Parity is sacred:** the canonical train corpus is `physicalai-train-e438721ae894`
  (2376 episodes) with skip-hash `f09e44db`. Anything that re-selects episodes breaks
  cross-arm comparability and must be refused.
- **Never add GPU/RAM load to a pod that is training**, and never eval on a training pod.
- Full suite lives at `stack/` — `pytest -q` must stay green before any commit.
