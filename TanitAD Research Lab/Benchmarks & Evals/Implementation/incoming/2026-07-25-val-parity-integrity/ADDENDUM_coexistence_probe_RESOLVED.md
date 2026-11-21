# ADDENDUM — the one open question in VAL_PARITY_REPORT.md is now RESOLVED

**Date:** 2026-07-25 · **Run by:** orchestrator · **Evidence class: MEASURED** (read-only `ls -d` on all
four pods, ~1 s, no GPU, no load added to the three live trainings).

## The question

`VAL_PARITY_REPORT.md` §blast-radius closed truncation and the leaky-split-in-ADE-path, but explicitly
refused to guess one thing:

> *"**NOT ruled out, and I will not guess:** whether both val dirs ever co-existed under one epcache root
> — the precondition for the `[-1]` bug to have actually fired. … **Settles it:** `ls -d <each epcache
> root>/*val*` on pods 1/2/3 + eval."*

That refusal was correct, and the probe it named is exactly what was run.

## The measurement

| pod | root | val dirs found |
|---|---|---|
| `tanitad-pod` (pod1) | `/workspace/data/physicalai_phase0/_epcache` | **0** |
| `tanitad-pod` (pod1) | `/root/valdata` | **1** — `physicalai-val-0c5f7dac3b11` ✅ clean |
| `tanitad-pod2` | `/workspace/data/physicalai_phase0/_epcache` | **1** — `physicalai-val-0c5f7dac3b11` ✅ clean |
| `tanitad-pod3` | `/workspace/pai_epcache` | **1** — `physicalai-val-f1b378f295ae` 🟥 **leaky (only)** |
| `tanitad-eval` | `/root/valdata` | **4** — `comma2k19-val-76b6e94a97a1`, `cosmos-val-a7a8527ba14e`, `cosmos-val-e8f3cef4976b`, `physicalai-val-0c5f7dac3b11` |

## Verdict

**The `sorted(glob("*val*"))[-1]` lexicographic-max defect could NOT have silently swapped a clean
PhysicalAI val for the leaked one — because the two never co-exist under any root.** Per-root:

1. **pod1, pod2 — clean only.** One PhysicalAI val dir, and it is `0c5f7dac3b11`. The resolver has no
   leaked alternative to prefer. **Not exposed.**
2. **eval pod — 4 val dirs, but only ONE is PhysicalAI, and it is the clean one.** Note the lexicographic
   order works *in our favour* here by luck, not design: `comma2k19… < cosmos-val-a7… < cosmos-val-e8… <
   physicalai-val-0c…`, so `[-1]` resolves to the **clean PhysicalAI** split. ⚠️ **This is a latent
   hazard, not a safe design** — adding any corpus whose name sorts after `physicalai` (e.g. `waymo-…`,
   `zenseact-…`) would have silently re-pointed every bare-glob evaluator at a different corpus entirely.
   The guard landed in this workstream removes the dependence on that accident.
3. **pod3 — the leaked split is the ONLY PhysicalAI val present.** So anything run on pod3 through a bare
   glob resolver received `f1b378f295ae`. This is **not a new exposure**: it is exactly the already-known
   and already-documented case — `MODEL_REGISTRY.md` §Branch-B records its `*_val` **R²** as computed on
   the leaky split (corrected earlier today at registry lines 1737–1747), and the sibling's git-history
   probe independently showed the **`taniteval` ADE path never used it** (`git log --all -S f1b378f295ae
   -- taniteval/` hits only `label_overlay.py`, a video renderer; `runner.VAL` was the clean split in the
   harness's first commit).

**Net:** the residual risk the report flagged is **closed**. No committed ADE is exposed to a silent
clean→leaky swap. The pod3 leaked-split usage is confined to the Branch-B / IDM-probe **R²** results that
are already flagged in the registry, and the newly-wired guard now makes the split explicit rather than
glob-resolved everywhere.

**Kept as a standing hazard note:** the eval pod's correct resolution today is an *alphabetical accident*.
Never restore a bare-glob val resolver.
