# Agent Operating Standard — v1 (2026-07-20)

*Adopted after the 2026-07-20 repo audit, which found that the program's dominant failure mode
was not bad work — it was **good work stranded outside git**. This standard exists to make that
structurally impossible. Binding for every subagent brief.*

## Why this exists — the evidence

| what was stranded | where it lived | consequence if lost |
|---|---|---|
| REF-B v2 architecture (`arch_v2`, `anchored_tactical`, `yaw_input`) | one pod disk | our **3rd-best arm** (0.592) unbuildable |
| **The entire TanitEval harness** incl. the P2 planner | one pod disk | **every headline number** unreproducible, conclusions left in the paper |
| Pod ops bundle (`supervise_run.sh`, `pod_boot_hook.sh`, …) | one local branch, never pushed | the documented pod recovery path |
| TanitResim maneuver strip (486 lines) | an uncommitted worktree | a worktree prune would have destroyed it |
| Atomic-archive fix | an unmerged branch | a **live bug** silently corrupting gate checkpoints |
| Orthogonality/isotropy instrument | an unmerged branch, "please merge" in a README | **10 days** of H26 instrumentation idle |

The root cause was our own default instruction — *"commit nothing"* — which protected the repo
and reliably stranded the work. This standard replaces it.

---

## THE PREAMBLE — paste verbatim into every subagent brief

```
## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish.
   Do NOT `git commit`, do NOT `git push`, do NOT switch branches. Never leave work that
   took real effort living ONLY on a pod or ONLY in a worktree — copy it into the repo and
   stage it. If you believe something should not be staged, say why in your report.

2. END WITH A DELIVERABLE MANIFEST — a table of every artifact you produced and WHERE it
   lives: `repo:<path>` / `<pod>:<path>` / `worktree:<name>`. Mark anything that exists in
   only ONE place. This table is not optional.

3. ESCALATE INTEGRATION. If your work needs merging, wiring in, or a decision, say so in
   your report's headline. Do NOT write "please merge this" into a README — that has been
   missed for 10 days before.

4. QUOTE ONLY PRIMARY SOURCES. Model facts come from `Project Steering/MODEL_REGISTRY.md`
   or raw eval JSON — never from a summary, changelog, or weekly report. If a doc conflicts
   with the registry, the registry wins; report the conflict.

5. FAIL LOUD, REPORT HONESTLY. If something is uncertain, mark it UNVERIFIED rather than
   guessing. A flagged gap is far better than a confident wrong answer. If you could not do
   part of the task, say so plainly — do not quietly narrow the scope.

## Traps preflight (each has cost hours more than once)
- `pgrep -f`/`pkill -f <trainer>` SELF-MATCHES your ssh command and kills your own session.
  Kill by explicit PID only.
- `PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED on pods (`cd` alone is not enough).
- NEVER judge pod disk with `df` (it shows the 965TB cluster, hiding the per-pod MooseFS
  quota). Use a real `dd` write test.
- `step_s` in trainer logs is ACCUMULATED over `--log-every`, not per-step.
- Multi-GB pod→pod transfer: pods can't SSH each other; the dev-box relay is ~1MB/s. Push
  → HF from the source pod (~118MB/s), then pull, then md5-verify.
- Keys.txt is git-ignored and must NEVER be committed; read tokens in place, never print them.
- Never add GPU/RAM load to a pod that is training; never eval on a training pod.
- Parity is sacred: `physicalai-train-e438721ae894` (2376 eps). Refuse anything that
  re-selects episodes.
```

---

## Standing cadence

- **Weekly audit pair.** Run the *model-registry* and *repo-triage* agents every week. In one
  run they found a live bug, 4 reproducibility gaps, 3 lineage errors and an 80% val leak —
  the highest return of any agents in the program. They are cheap relative to what they catch.
- **Nightly pod drift check.** `stack/scripts/pod_git_drift.py` reports any code that exists on
  a pod but not in git. **A pod is not storage:** nothing that took more than an hour to produce
  may live only on a pod.
- **Registry refresh** whenever a model version is created, retired, or re-measured.

## Reviewing agent output — the checklist

1. Does the manifest show anything living in only ONE place? → rescue it before anything else.
2. Are the numbers traceable to the registry or raw JSON?
3. Did it flag gaps as UNVERIFIED rather than guessing?
4. Does it need integration that would otherwise sit unread?
5. Does the full suite (`cd stack && pytest -q`) still pass?
