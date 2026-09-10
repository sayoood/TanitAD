# APPLICATION RECORD — the `agent_gt` forward plumbing is APPLIED and PINNED

**Date** 2026-09-10 · **Agent** Architecture & Inference FlyWheel · **Branch**
`agent/arch-inf-20260803` · **Tier** N/A (a wiring proof, not an eval)
**Evidence class** MEASURED (ours) unless stamped otherwise. Every number below
carries the artifact that produced it, in this directory.

> ⭐ **Headline.** The patch applies cleanly at **offset 63**. `agent_gt` now
> **provably reaches `RefCV3Model.forward`** under `--agents oracle` — measured on
> a real forward, with the oracle's emitted slot boxes **bitwise equal** to the
> batch's `agent_box`. `--agents off` and `--agents head` are **bitwise
> unchanged** across params, losses **and** gradients. Two deliberate
> corruptions prove each comparison **can** fail. The regression test the banked
> measurement deliberately was not is now shipped: **11 passed** post-patch,
> **11 failed** pre-patch.

---

## 0. Environment

| | |
|---|---|
| box | dev box (Windows 11), CPU only — no GPU was needed |
| python / torch | 3.13.5 / **2.11.0+cu128** (`torch.cuda.is_available() == True`, unused) |
| tree under test | `C:\Users\Admin\tanitad-agentgt` (off-Drive clone of `stack/` + `taniteval/`) |
| pristine baseline | `C:\Users\Admin\tanitad-agentgt-base` (identical copy, patch NOT applied) |
| width | `refc_v3_smoke_config(hier=True)`, `--agent-queries 8`, B=2, N=8 |

⚠️ **Why an off-Drive pair and not the repo tree.** `CLAUDE.md`: G: cannot RUN
the stack (`Errno 22` mid-import), and the repo's own suite carries pre-existing
failures. Every suite number below is therefore reported as a **diff against a
pristine baseline built from the same bytes**, never bare.

---

## 1. Did the patch apply cleanly, and at what offset?

**YES — offset 63, by two independent appliers.**

| probe | rc | result |
|---|---|---|
| `git apply --check -v` (on the G: repo, worktree blob `b1ae8135`) | **0** | `Hunk #1 succeeded at 2168 (offset 63 lines).` |
| `patch --dry-run -p1 --verbose` | **0** | `Using Plan A... Hunk #1 succeeded at 2168 (offset 63 lines).` |
| `patch -p1` (real application, off-Drive tree) | **0** | same hunk, same offset |

The patch file is byte-identical to the banked one
(`md5 9eb35506d8f73bcad439f57b2bc9c6f0`, 3,773 B).

**The applied change is a strict superset of the original**, verified by diff
rather than asserted:

| | |
|---|---|
| lines | 5,590 → **5,640** |
| md5 | `620fc095a3e85b56a87527b29dbcdf10` → `4e9c9834f20d6c50b35c163aae4ab4a8` |
| `diff` | **1 line removed** (the forward call's closing line) · **51 added** |
| `agent_gt` occurrences | **0 → 7** (6 lines) |
| `agent_box` occurrences — ⛔ SAME-BREATH CONTROL | **12 → 16** (never zero) |

⛔ The control matters: on this mount a `0` from a grep is indistinguishable
from a file that could not be read. `agent_box` reading non-zero on **both**
sides is what makes the `agent_gt` `0 → 7` admissible.

---

## 2. Is `agent_gt` provably reaching the forward? — MEASURED on a real forward

`raw/forward_reach_PREPATCH.json` · `raw/forward_reach_POSTPATCH.json`

The probe spies `RefCV3Model.forward`, runs the trainer's own
`compute_losses_v3`, and asks the forward **what it was handed**. It is not a
grep and not an import check.

### PRE-PATCH — the defect, reproduced

```
ValueError: this build is `--agents oracle` but no agent_gt reached the
forward. The oracle's tokens ARE the ground-truth boxes; with none the seam
emits nothing and the arm would read as 'agent tokens do not help' while never
having had any.                                          (refc_v3.py:1405)
```

### POST-PATCH — GREEN, and the assertion has content

| measurement | value |
|---|---|
| `forward` calls | 1 |
| `agent_gt` is None | **False** |
| `agent_gt` keys | `["box", "cls", "rates", "valid", "yaw"]` |
| `agent_gt["box"]` shape | `[2, 8, 4]` |
| `agent_gt["box"] == batch["agent_box"]` | **True (bitwise)** |
| `agent_gt["yaw"] / ["cls"] / ["valid"]` vs batch | **True (bitwise)**, each |
| `out["agent_slots"]["box"] == batch["agent_box"]` | **True (bitwise)** |
| total loss | **111.02839660644531**, finite |
| backward | OK; gradient lands on every `layer.agent_gate` |

⭐ **The third row down is the one that matters.** A non-`None` kwarg would pass
on a branch built from the wrong tensors. `degrade_boxes` returns *"the inputs
unchanged objects"* at `sigma_range_m = 0 / miss_rate = 0`
(`refc_agents.py:404-409`, and the run confirms both are `0.0`), so a
byte-for-byte match of the **emitted slot boxes** proves the batch's ground
truth is what the decoder actually cross-attends — value flow, not arrival.

---

## 3. Are `--agents off` / `--agents head` bitwise unchanged?

**YES — params, losses AND gradients, on both arms.** Same seed (1234), same
config, same batch, identical RNG sequence, base tree vs patched tree.

⛔ **The GT boxes are injected on EVERY arm**, including `off` and `head`.
Injecting them only on the oracle arm would confound *"the branch did not
fire"* with *"the batch was different"*, and these rows would pass for the
wrong reason.

| arm | params | losses | grads | loss (base) | loss (patched) | verdict |
|---|---|---|---|---|---|---|
| `--agents off` | 193 / **0 differ** | 20 / **0** | 153 / **0** | 38.79359436035156 | 38.79359436035156 | **BITWISE_IDENTICAL** |
| `--agents head` | 275 / **0 differ** | 20 / **0** | 235 / **0** | 104.13790893554688 | 104.13790893554688 | **BITWISE_IDENTICAL** |

`raw/parity_agents_off.json` · `raw/parity_agents_head.json`

---

## 4. Did the deliberate corruption prove the comparison CAN fail?

**YES — and it took two corruptions, because they answer different questions.**
WP-B's removability proof was green for a reason unrelated to WP-B; a proof
that cannot fail proves nothing.

### Corruption A — IN-BRANCH VALUE (`batch["agent_box"].to(device) * 2.0`), gate untouched

| side | result | artifact |
|---|---|---|
| ON — oracle value flow | **RED**: `agent_gt.box == batch` **False**, `slot_box == batch` **False** | `raw/corruptionA_value_oracle_probe.json` |
| OFF — `--agents off` | **still BITWISE_IDENTICAL** (0/193, 0/20, 0/153) | `raw/corruptionA_parity_off.json` |
| OFF — `--agents head` | **still BITWISE_IDENTICAL** (0/275, 0/20, 0/235) | `raw/corruptionA_parity_head.json` |

⇒ the ON assertion is **sensitive**, and the corruption does **not** leak
outside the gate.

### Corruption B — THE GATE REMOVED (`if True:`), so the branch fires on every arm

| side | result | artifact |
|---|---|---|
| `--agents off` | **FAILS, hard**: `ValueError: agent_gt was supplied but this build has no agent seam` (`refc_v3.py:1398`) | `raw/corruptionB_gate_removed_agents_off.txt` |
| `--agents head` | ⚠️ **still BITWISE_IDENTICAL** (loss 104.13790893554688) | `raw/corruptionB_parity_head.json` |
| oracle (control) | still GREEN — the mutant is not simply broken | `raw/corruptionB_oracle_probe.json` |

⇒ the **OFF** comparison is sensitive: removing the gate is detected, and
detected as a refusal rather than a drift.

---

## 5. ⛔ A FINDING THE TASK DID NOT ASK FOR — `agent_gt` on a `--agents head` build is SILENTLY DROPPED

**MEASURED, and it is why row 2 of Corruption B reads the way it does.** With
the gate removed the head arm is bitwise identical *not because the trainer's
gate held* — it did not — but because **the model ignores the tensor**:

* `refc.py:3366-3367` — the non-oracle branch builds slots from `fmap` and
  **never reads `agent_gt`**;
* `refc_v3.py:1398` — the reverse guard fires only when `_ag is None or not
  _ag.enable`. On a head build `enable` is **True**, so it does not fire.

⇒ **`--agents head` + `agent_gt` = silently dropped**, which is precisely the
failure the guard's own docstring says it exists to prevent (*"a privileged
tensor that is SILENTLY DROPPED reads as 'agent tokens do not help' in a result
table"*), one branch over from the one that caught this whole defect.

⚠️ **Scope it honestly — it is LATENT, not live.** The applied patch is gated
on `enable AND oracle`, so nothing currently supplies `agent_gt` on the head
path, and no banked head arm is affected. What it does mean:

* the `head` parity row above is **over-determined** — it is identical for two
  independent reasons and therefore **cannot discriminate a correct gate from
  no gate at all**. The **`off`** row is the load-bearing one, and it is stated
  as such in the test's docstring rather than left for a reader to notice;
* the model-side guard has a **third direction it does not cover**. Closing it
  is a one-line change in `refc_v3.py` (`enable and not oracle` ⇒ refuse), but
  that file is not this work package's and is under concurrent edit, so it is
  **ESCALATED, not silently patched**.

---

## 6. What goes RED when the patch is reverted?

`stack/tests/test_refc_v3_agent_gt_reaches_forward.py` — **11 tests**.

| tree | result | wall |
|---|---|---|
| patched | **11 passed** | 2.57 s |
| pristine (pre-patch) | **11 failed** | 6.24 s |

⛔ **And they fail for exactly two mechanisms, both correct** — not
incidentally:

| n | mechanism |
|---|---|
| 4 | `ValueError: this build is --agents oracle but no agent_gt reached the forward` — the defect itself, on a real forward |
| 7 | `AssertionError: the mutation anchor for '<name>' matched 0 times, not 1. The deliberate-regression arm is DISARMED, so this test would pass without proving anything.` |

⭐ The second row is the design working. The mutation arms are anchored on
**exact literals from the patched source**; on a tree without the patch they
cannot be armed, and `_mutate` **fails loud** rather than skipping — a
disarmed regression arm that silently passes is the defect this repo has met
four times in one night.

**On the patched tree the reverted mutant runs and is refused**
(`test_REVERTING_THE_PATCH_REPRODUCES_THE_DEFECT`), with a same-breath control
(`test_the_reverted_trainer_still_runs_agents_off`) proving the mutant is not
merely broken for every arm.

`raw/pytest_agent_gt_GREEN_postpatch.txt` · `raw/pytest_agent_gt_RED_prepatch.txt`

---

## 7. Suite — against a pristine-baseline diff, never bare

**Affected surface** = the 70 test files that reference `refc_v3_train`,
`refs.refc`, `refc_agents` or `refc_v3`.

| group | pristine baseline | patched | delta |
|---|---|---|---|
| 66 core files | **21 failed / 1193 passed / 16 skipped** | **21 failed / 1204 passed / 16 skipped** | **+11 passed** = exactly the new tests |
| 4 `taniteval/tools` files | 76 passed / 1 skipped / 19 errors | 76 passed / 1 skipped / 19 errors | none |

⛔ **The failure SET diff is EMPTY** — `diff` of the sorted `FAILED` node ids
between the two trees returns nothing, and likewise for the `ERROR` set.
`raw/suite_failureset_base.txt` vs `raw/suite_failureset_patched.txt`.

The 21 pre-existing failures (identical on both trees, **none introduced by
this work**): `test_fan_safety.py` (8), `test_feasible_decode.py` (5),
`test_refc_v4.py` (5), `test_argparse_help_percent.py` (1),
`test_nav_known_channel.py` (1), `test_refc_tactical.py` (1).

⚠️ The 19 errors are an artifact of the **partial off-Drive tree** (a missing
`<repo>/tools/criteria_check.py`, which was not copied), identical on both
sides. They are not evidence about the patch in either direction.

---

## 8. Traps met while doing this — each cost a round, each is reusable

1. ⛔ **A text-mode Python round-trip rewrote every line ending**, turning a
   one-line mutant into a 5,640-line diff. **`grep -c $'\r'` reported `0`** on
   both files; the **file SIZE** (`339,880 → 345,526` = +1 byte/line) is what
   settled it. Rebuilt in binary mode. *Same family as the `df` / cgroup traps:
   a probe reporting the wrong thing, read as an answer.*
2. ⛔ **`cp` from G: "succeeded" and produced ZERO-BYTE files.** The md5 of
   every copied `taniteval` file was `d41d8cd98f00b204e9800998ecf8427e` — the
   md5 of an empty file — while directory listings and `stat -c%s` on the
   source returned correct sizes. Presence is not content.
3. ⭐ **The same-breath control is what produced the right diagnosis.** The
   first reading was *"`taniteval/tools` is cloud-only and never hydrates"*.
   A control read of `stack/tanitad/refs/refc_agents.py` **in the same command**
   *also* failed ⇒ the **mount** was down, not the subtree. Recovered from the
   local mirror, md5-verified against it.
4. ⚠️ **MSYS `/tmp` is not Windows `C:\tmp`.** A bash-written `/tmp/x.json` is
   invisible to the venv's Windows python; three reads failed with
   `FileNotFoundError` that looked like the producer never ran. Fixed by using
   an explicit `C:/Users/...` path for every cross-tool file.
5. ⚠️ **A ~340-line Bash heredoc died at PARSE time** (`unexpected EOF`), the
   documented long-script failure. Authored through a file-writing tool instead.

---

## 8b. CAN THE WP-C GATE ACTUALLY RUN? — checked, not asserted

⭐ A forward that accepts `agent_gt` is not the same claim as *"the gate runs"*.
Three things were read/measured rather than assumed:

1. **The join is enabled on `--agent-join` ALONE, not on `--w-agent > 0`**
   (`refc_v3_train.py:4295`). The gate launches with `--agents oracle
   --agent-join <file> --w-agent 0`, so the batch **does** carry `agent_box`
   and the branch's `SystemExit` cannot fire on it.
2. ⛔ **The EVAL dataset gets the join too** (`:4372`) — this one mattered.
   The in-training eval reuses `compute_losses_v3`, and `SystemExit` derives
   from `BaseException`, so the eval block's `except Exception` **cannot catch
   it**. Had eval lacked the join, the run would have died at the **first eval
   (step 500)** with 8 % of the compute already paid — exactly the
   `t1_eval` class this repo has met before. It does not lack it.
3. **All six gate arms pin cleanly on the patched trainer at REAL width**
   (`base`, not smoke), using `run_gate.sh::base_args()` **verbatim**:

| arm | pin | `agents.enable` / `.oracle` | `cross_agent` | wp_index built | stamp `agents.oracle` |
|---|---|---|---|---|---|
| B0 | OK | True / True | True | **False** | True |
| B1 | OK | True / True | True | **True** | True |
| B0r | OK | True / True | True | **False** | True |
| B1r | OK | True / True | True | **True** | True |
| B1const | OK | True / True | True | **True** | True |
| B1shuf | OK | True / True | True | **True** | True |

`raw/gate_argv_pin_check.json` · `code/gate_argv_check.py`. The `wp_index`
column reads False on exactly the two `--wp-index off` arms and True on the four
`on` arms — the control that shows the pin is reading the flag rather than
returning a constant.

⚠️ **One residual, NOT a blocker, deliberately not fixed here.** There is still
**no PIN-TIME guard for `--agents oracle` without `--agent-join`** — the `head`
path has exactly that guard at `:591`, the oracle path does not. The patch
downgrades the failure from a forward-time `ValueError` deep in the model to a
first-batch `SystemExit` naming the cause, which is a large improvement but is
still after the GPU is allocated. It is **not** a blocker because `run_gate.sh`
passes the join. It was left alone on purpose: adding an unrequested guard would
change this file's blob again and invalidate the parity proof above, and the
brief was explicit that nothing be hand-merged silently. Named as a cheap
follow-up instead.

---

## 9. Verdict

* **The WP-C oracle gate is UNBLOCKED on this defect.** `--agents oracle` now
  has a supplier, the supply is proven bitwise at the seam, and the arm trains
  (finite loss, gradient on the agent gate).
* **No behaviour changed for any arm that does not ask for the oracle**, proven
  bitwise and shown capable of failing.
* **The gap the author left open is closed**: the RED/GREEN measurement is now
  a regression test that runs in CI in 2.57 s.
* ⚠️ **One escalation, not a blocker:** the `--agents head` + `agent_gt` silent
  drop (§5).
