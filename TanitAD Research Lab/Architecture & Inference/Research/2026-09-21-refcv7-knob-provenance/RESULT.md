
<!-- REFCV7-WEIGHTS-WERE-UNSTAMPABLE-TEST-WAS-RED-AT-THE-TIP-2026-09-21 -->

### ⛔ 2026-09-21 — `test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE` was RED AT THE TIP, and the two refcv7 weights were the reason: 36 of 38 agent knobs were verified, `--w-r7-wta` and `--w-r7-scorer` were not

MEASURED by me, CPU only, no GPU
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-21-refcv7-knob-provenance/`).

I hit this failure while landing `6a472d1`, **verified it was pre-existing** by checking the tip's
own `agent_slots.py` out over mine, recorded it — and then carried it forward a second time in
`305debd` without fixing it. This closes it.

**The enumeration came first, because the shipped loop aborts at the FIRST unstampable knob** and
would have hidden a second. A throwaway copy that COLLECTS instead of raising reports exactly two:

| | |
|---|---|
| knobs checked OK | **36** |
| knobs with NO admissible probe value | **2** — `--w-r7-wta`, `--w-r7-scorer` |

⇒ M18's guarantee — *"a run must be able to state every `--agent-*` / `--w-*` knob it trained at"* —
did **not** hold for the two refcv7 weights, and the test that exists to say so had been failing.

**The fix is a SEAM-table row, which is the sanctioned place** (the table's own comment: *"a SEAM
table, not the per-knob VALUE table the docstring forbids"*). ⛔ Every entry satisfies a refusal
that is **correct**; the fix is to satisfy them, never to weaken one. The chain
(`refc_v3_train.py:644-695`) is **four deep**: the weights refuse without `--refcv7`; `--refcv7`
refuses without `--arm hier`, without `--tac-decoder-v6` (which itself refuses without
`--v7-labels`), and without `--trunk timm`. The WTA row adds `--r7-no-select`, because with the
scorer weight still 0 selection from an **untrained** scorer is refused by name. The scorer row
additionally needs `--w-r7-wta > 0`, `--r7-nav-tau-rad > 0`, `--agent-join`, and `--map-gt-root`
**with** `--w-map > 0` — plus the rig extrinsics, because `--map-gt-root` with both perception
weights at 0 fires the **reverse** refusal.

⚠️ **`TACV6_ON` is a separate constant on purpose.** `--w-tac-v6` defaults to **0.0** and
`--tac-decoder-v6` at a zero weight is refused by name. Folding the weight into `TACV6` would put
`--w-tac-v6` **twice** on the `w_tac_v6` row, where argparse silently keeps the **last** occurrence
— so the probe would assert whichever value happened to come second, and a reordering would change
what the test checks without changing what it reports.

⭐⭐ **AND THE GREEN WAS NOT TAKEN ON TRUST — a test can go green by SKIPPING the knob, which is
exactly the defect this programme's own rule names.** Mutation: make `agent_knob_stamp` drop every
dest beginning `w_r7`. Result **RED** (1 failed, 38 passed), trainer restored byte-identical.
⇒ the two knobs are genuinely exercised. **Before the fix that mutation could not have been
caught**, because the test aborted earlier at *"no admissible probe value"* and never reached the
stamp — i.e. the guard was blind precisely where it now bites.

**Suite:** `test_refc_v3_agent_provenance.py` **39 passed, 0 failed** (was 38 passed, 1 failed).
The provenance guard now covers **38 of 38** agent knobs.
