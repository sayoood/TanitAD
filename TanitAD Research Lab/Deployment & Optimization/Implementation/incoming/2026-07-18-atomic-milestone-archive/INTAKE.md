# INTAKE — atomic milestone-checkpoint archive (kill-mid-copy → silent-corrupt gate ckpt)

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-18-atomic-milestone-archive/`
- **Author agent / date:** Production & Optimization (Saturday, run #5), 2026-07-18
- **Proposed target:** new `stack/tanitad/train/ckpt_io.py` + 3 one-line call-site swaps
- **Hypothesis / WP served:** ops-fragility (F-5/F-6/F-7); protects the gate protocol
  (D1/D2/D3 milestone evals) that the whole 3-arm bake-off decision depends on.

## What & why (≤10 lines)

Compliance review #3 (`stack/scripts/` + training loop) found a **live** ops-fragility bug.
All three pod trainers archive gate milestones with a **non-atomic** copy straight to the
final path — `train_flagship4b.py:337`, `refb_train.py:358`, `refa_train_plus.py:540` — each
guarded by `if step >= m and not arch.exists(): shutil.copy2(ckpt, arch)`. A kill *during*
`copy2` (the documented pod2 self-kill / eval-OOM 2026-07-16 / Errno122-quota-full history)
leaves `ckpt_step{m}.pt` **truncated but existing**. On the next save the guard sees
`arch.exists()` → **never re-archives**, so the corrupt milestone silently stands; the gate
protocol later `torch.load`s it → crash or garbage D1/D2/D3 numbers. Same silent-corrupt class
the atomic *resume* write (`tmp.replace(ckpt)`) already guards — the archive path was missed.
**Fix:** `atomic_archive` copies to a `.partial` sidecar then `os.replace` (atomic rename); a
kill mid-copy leaves only `.partial`, the final path never appears half-written, and the
`not arch.exists()` guard self-heals on the next save. `archive_milestones()` is the drop-in.

> ⏹ **CLOSED 2026-08-16 — THE BUG IS FIXED. The "live ops-fragility bug" framing above is HISTORY;
> do not re-open it.** The ORCHESTRATOR VERDICT block below was never filled in, which is the only
> reason this reads as outstanding — the code landed regardless.
> Evidence (MEASURED, tip working tree):
> - `stack/tanitad/train/ckpt_io.py:23` → `def atomic_archive(src, dst) -> Path` — the module was
>   created at the proposed path.
> - **All three named call sites were swapped**, and each carries a comment naming the failure mode:
>   `stack/scripts/train_flagship4b.py:619-620`, `stack/scripts/refb_train.py:426-427`,
>   `stack/experiments/reset-speed4b/refa_train_plus.py:541-542`. A repo-wide grep finds **no
>   surviving bare `shutil.copy2` milestone archive** in any trainer.
> - **The Risk section's open question is ANSWERED: `refc_train.py` does archive, and it was swapped
>   too** — `stack/scripts/refc_train.py:678-679`. So was a trainer that did not exist when this was
>   written: `stack/scripts/train_flagship_v4.py:669-673`.
> - `stack/tests/test_ckpt_io.py` is in the suite that must stay green (it exercises `atomic_archive`
>   at `:23`, `:43`, `:70`, `:73` — including the crash-then-reheal sequence).
>
> ⚠️ **One deviation from the proposal, stated so nobody hunts for it:** the integration adopted
> `atomic_archive` **directly** at each call site, keeping each trainer's own
> `for m in MILESTONES: if step >= m and not arch.exists():` loop. The proposed
> **`archive_milestones()` wrapper was NOT adopted** — `grep -rn "archive_milestones" stack/` returns
> zero hits. Behaviourally equivalent (the atomic write is the fix), but the milestone loop is still
> duplicated across five trainers, so a future milestone-policy change must touch five files.
> Swept by the 2026-08-16 stale-blocker sweep.

## Evidence & tests

- Tests: `tests/test_atomic_milestone_archive.py` — **4 passed in 1.58 s** (CPU torch, no GPU).
  - `test_current_copy2_corrupt_milestone_survives_guard` — **reproduces the live bug**: a
    half-written `ckpt_step5000.pt` is unloadable (`torch.load` raises) and the current guard
    adopts it forever (skips re-archive).
  - `test_atomic_archive_no_partial_at_final_on_crash` — a simulated `OSError(122)` mid-copy
    leaves **only** `.partial`; the final milestone path does not exist.
  - `test_atomic_archive_reheals_after_crash` — after a crash leaves `.partial`, the next save
    re-archives a clean, loadable milestone (no manual cleanup).
  - `test_archive_milestones_roundtrip_and_idempotent` — normal path is loadable + not re-copied.
- The three call sites were verified verbatim on the current mainline (`fcbab02`); the buggy
  guard logic is inlined in test #1 so the witness tracks the real code.

## Risk & rollback

- Blast radius: adds one new module (`ckpt_io.py`); swaps the 3 inline archive loops for
  `archive_milestones(ckpt_path, step)`. Behaviour-identical on the happy path (same filenames,
  same idempotence); only the crash path changes (corruption-safe). No API/training-math change.
- `refc_train.py` uses the same pattern — swap it too if it archives (confirm at integration).
- Rollback: revert the 3 call sites to the inline `shutil.copy2` loop and delete `ckpt_io.py`.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
