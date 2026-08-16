# S2 LOSS — the strategic-goal supervision term is WIRED: default-off, head-only by MEASURED gradient reach, joined by stable id, every guard proven able to fail

**2026-08-16, A&I stream.** Executes the escalation the label build filed
(`…/2026-08-16-s2-v1-labels/S2_V1_LABELS.md` §8 item 1, spec in
`…/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` §1.2): the S2 term
(`w.s2_goal`), its batch keys, and the `--s2-labels` loader with the
clip-index join now exist in `stack/scripts/train_v6_staged.py` +
`stack/scripts/s2_labels.py`. The label side was FORMAT-COMPLETE and waiting;
this closes the trainer side. Every number below is **MEASURED this run**
(dev box, CPU only, Thor untouched) unless stamped otherwise.

**Headline.** `L_s2 = CE(g_str) + CE(a_str) + masked-L1(args)`, masked to the
`s2_valid` band, in force **only in S-S/S-J** via `for_stage`, default
**0.0 = bit-identical incumbent loss** (proved against a CONTENT-anchored
pre-change trainer, per stage, RNG stream included). Its gradient reach is
**measured as exactly the 16 tensors of `goal_head_str.* + act_head_str.*`**
— zero encoder/readout/predictor/adapter, and the shared vocabulary tables
are **not touched** (the spec's "+ vocab embeddings if applicable" resolves
to *not applicable, measured*). The join consumes the shipped
`clip_index.json` through `stable_episode_id` **only**; the colliding legacy
16-bit id is refused with the collision census in the message. The real
**797-record artifact loads through the real CLI** (`--dry-run` banked under
`raw/dryrun-ss-s2/`) with token censuses equal to the label report's.
Suite: baseline **3396** → combined-tree **3486 passed**; S2's delta is
**+39 passed / +1 pod-only skip**, zero regressions in its lane (the 3
combined-tree failures are a concurrent stream's un-updated pin, attributed
in §3/§6.1).

---

## 1. What was built (and where)

| piece | where | shape |
|---|---|---|
| `V6LossWeights.w_s2_goal` | `train_v6_staged.py` | default **0.0**; `for_stage` zeroes it in S-W **and** S-T (layer_str frozen there — the advertised-but-inert lie `--w-select`/`--w-anchor` already refuse), keeps it in S-S/S-J |
| the term | `s2_goal_loss` + the `v6_loss_step` block | exactly §1.2's formula; CE via `ignore_index=-100` for out-of-band/unlabeled windows; arg L1 = `|pred−label|·arg_mask` averaged over SET slots of VALID windows (a batch with zero valid windows contributes a zero-in-graph term — the O3 `n_masked==0` idiom — and logs `s2_n_valid: 0`) |
| batch keys | `g_str_id/g_str_args/g_str_arg_mask/a_str_id/a_str_args/a_str_arg_mask/s2_valid` | documented in the `v6_loss_step` batch contract; required iff the weight is in force |
| loader + join | **NEW** `stack/scripts/s2_labels.py` (`load_s2_labels`, `S2LabelSet`, `S2WindowSupervision`) | validates every record against the REAL v6 vocabularies (no pinned copy to drift), builds per-episode rows once, assembles per-window tensors in O(batch) |
| launch surface | `--s2-labels <dir-or-jsonl>` + `--w-s2-goal` | 6 preflight refusals (§5); dry-run REALLY exercises the loader when the flag is supplied and the loss path on synthetic keys (the ladder-seam lesson: a verifier that skips the launch's flags cannot catch their failure class) |
| run record | `config.json["s2"]` | label censuses per token + per provenance, join counts, disjointness verdict — the raw material of the S-S gate's goal-provenance audit |
| tests | **NEW** `stack/tests/test_v6_s2_loss.py` | 40 tests: identity, reach, leak, semantics, loader refusals + passing twins, join, preflights, dry-run |

Per-token telemetry is **per family, never pooled**: every log row carries
`s2_g_ce / s2_a_ce / s2_g_arg_l1 / s2_a_arg_l1 / s2_g_acc / s2_a_acc /
s2_g_tok_counts / s2_a_tok_counts / s2_n_valid` (visible in the banked dry
rows, e.g. `"s2_g_tok_counts": {"FOLLOW_MAIN_ROAD": 1, "NONE_ABSTAIN": 1,
"TURN_LEFT": 1}`).

## 2. ⛔ GOAL HEADS ONLY — the binding rule, MEASURED not asserted

The rule (*"labels supervise GOAL/INTERPRETATION HEADS only, never any WM
trunk loss"*, HIERARCHY_VOCABULARY §2, the diagram) holds because
`v6.py:2657-2664` computes `g_str/a_str` from `z_str_p = _cut(z_str, cut)` —
**detached under the default planner cut** — and the S2 term reads those
emitted logits/args and nothing else. Proved the
`test_v6_gstr_port.test_t1_gradient_reaches_…` way, on a real autograd graph:

* **Reach = exactly** `{goal_head_str, act_head_str} × {trunk.0, trunk.2,
  type_head, arg_head} × {weight, bias}` — 16 tensors, all group
  `layer_str`; the test asserts set EQUALITY, so a widened head or a new
  leak both fail by name.
* **The vocab tables get zero gradient** — the heads' logits/args come from
  their own trunk/type_head/arg_head; `vocab_str.encode` sits only on the
  downstream conditioning path (`e_g_str`) this loss never reads. The reach
  detector is itself negative-controlled: a probe loss on `e_g_str` DOES
  light `vocab_str.table.weight`, so "vocab untouched" is a seen absence,
  not a blind one.
* **The leak the guard prevents is real and two-layered** (measured): with
  the planner cut off (uplink still on), a head CE escapes into
  `adapter_str.*` — the strategic layer's WM-side uplink adapter, the very
  tensors the s1 latent loss trains (a label loss on the representation
  path); with BOTH cuts off it reaches the **encoder and readout**. ⇒
  `v6_loss_step` REFUSES `w_s2_goal > 0` on an
  `isolate_planner_from_encoder=False` build, and preflight refuses the
  `--no-isolate-planner` combination — **with no
  `--i-know-this-is-the-control-arm` escape, tested**: a binding rule has no
  control arm.
* Under the S-S freeze, backward of the full loss populates `.grad` only in
  `layer_str` (measured through `apply_stage_freeze`).

Echo/disjointness posture is unchanged from the label build's audit
(`S2_V1_LABELS.md` §6, INHERITED): at inference the heads read only `z_str`
(vision-derived, `d_cond=0`); no label source is a model input. This stream
adds the **trainer-side re-assertion on the bytes it consumes** (§4).

## 3. ⛔ The incumbent is untouched — content-anchored, per stage (C75)

* **Trainer:** `test_default_loss_is_bit_identical_to_the_PRE_S2_trainer`
  resolves the newest committed `train_v6_staged.py` revision **without**
  the `w_s2_goal` marker (never HEAD — C75) and runs old-vs-new
  `v6_loss_step` over the SAME current model, default weights, all four
  stages: **every loss bit-equal, log key set identical, and the global RNG
  stream consumed identically** (the S2 block draws nothing — also pinned
  separately, the C74 discipline). Negative control: turning the term on
  makes the same comparison fail.
* **Model:** this change edits **no `tanitad/models/v6.py` line** (the
  sibling stream owns it this turn; zero escalations needed). The existing
  battery re-ran green in this suite run: default **87,893,449 / 405** and
  config E **336,542,025 / 573** unchanged, per-tensor `torch.equal` against
  their own content anchors (`test_v6_gstr_port.py`,
  `test_v6_factored_goal.py`). The live resume is safe: S2 adds **no
  state_dict key**, so there is also no geometry to carry across stages —
  unlike `--selector`, dropping the flag later breaks nothing.
* **Suite:** baseline (pre-edit tree, this box) **3396 passed / 17
  skipped / 2 xfailed (8:03)**. Post-edit, on the COMBINED tree (three
  streams' concurrent edits, §6.1): **3486 passed / 3 failed / 17 skipped /
  2 xfailed (7:30)**. The S2 contribution is **+39 passed, +1 skip**
  (`stable_episode_id` fallback-vs-canonical pin, torchvision-gated: runs
  on pods, skips on the dev box), verified in isolation (39/1) and
  re-verified with the identity suites after the siblings' latest edits
  (203 passed / 1 skipped). ⚠️ **The 3 failures are NOT S2's** — all in
  `tests/test_v6_stage_init_introduction.py`, whose committed pin
  `assert STAGE_MAY_INTRODUCE["S-T"] == ("cand_score.", "cond_tac_dyn.")`
  fails against the DIFFUSION/MPC stream's in-flight extension
  `+ ("prop_diffusion.", "fallback.")` — the failing literal names their
  two prefixes and no S2 token; S2 touches no allowance and adds no
  state_dict key. Reproduced in isolation and left to that stream's
  in-progress turn (their test/pin update is the fix); whoever commits
  runs the suite green first, per the standing rule.

## 4. The join — stable id only, and every refusal can fire

`clip_index.json` (801 clips) → `stable_episode_id(clip_id)` (blake2b>>1,
63-bit) → the trainer episodes' `.episode_id`, which IS that id on the
production path (`build_train_episodes` → `build_v2_providers`
`stable_ids=True` default). Loader/join refusals, **each with a
proven-able-to-fail test and a passing twin**:

| refusal | why |
|---|---|
| missing `clip_index.json` | unjoinable labels = a term that silently never fires (§1.2's own warning) |
| **legacy 16-bit episode id on ANY episode** | it collides (69/2400 + 7/600, `_legacy_collisions`) — and an id unique among the 801 *labeled* clips can still collide with an *unlabeled* corpus clip the loader cannot see, silently supervising the wrong scene. Stricter than the index's "refuse ambiguous keys" note, deliberately: rebuilding the manifest gets stable ids for free, so there is no admissible reason to join through the legacy id |
| index `episode_id_stable` ≠ recomputed | a drifted hash is a silent zero-match; the cross-check runs over all 801 entries at load |
| **ROUTE_TO record** | mirrors `s2_schema.validate()` (G1 CLOSED 0/31; no categorical channel on `vocab_str`); ALSO refused per-batch inside `s2_goal_loss`, so a hand-built batch cannot smuggle it to the head |
| duplicate / unknown / EXCLUDED clip record | ambiguous target / unjoinable / the 4 manufactured abstains must not re-enter |
| arg-mask discipline (shape, 0/1, unset-slot-nonzero), token_id drift, schema_version | the §1.2 record contract, enforced on load |
| **payload-only disjointness scan + per-record stamp** | "situation/sitclf" in the goal payload refuses; the same word in a META field does NOT (tested both ways — the polling-monitor trap: the scan must not match the record's own stamp) |
| zero joined episodes / zero in-band windows with `w > 0` (trainer-side) | refused before the optimiser, after the corpus — a supervision that never fires must not look like a run |

**Band math** (tested at both edges, both stack depths): a window `(e, t)`
is supervised iff `|(t + W − 1 + (n_stack−1))·dt − t0_s| ≤ 2.0` — `t0_s=8.0`
lives on the RAW clip timeline and a v2ep provider drops the first
`n_stack−1` frames (`_scan_meta` `poses[k:]`), read per episode off its own
channel count. At W=6, 9-channel stack, 10 Hz: provider `t ∈ [53, 93]`.

**The real artifact through the real code** (MEASURED): `load_s2_labels`
on `…/2026-08-16-s2-v1-labels/labels/` → **797 records**, g_str census
`FOLLOW 395 · TURN_L 137 · TURN_R 113 · LANE_TARGET 80 · STOP_AT 59 ·
NONE_ABSTAIN 13`, a_str `HOLD 526 · P_STOP 88 · REDUCE_TO 85 · P_LC 80 ·
RESUME 18`, provenance `path` 797/797, 801 index clips / 4 excluded —
**equal to `S2_V1_LABELS.md` §3 (aug120+val summed), independently
re-derived by this loader** and pinned as a skip-if-absent test. The full
CLI path (`main → preflight → dry_run → loader + loss`) is banked:
`raw/dryrun-ss-s2/{dry_run,config,stage_gate}.json` (gate INCONCLUSIVE +
`_dry_run`, as a dry gate must be).

## 5. The launch surface

Preflight refusals (milliseconds, before any corpus/GPU; all tested):
`--w-s2-goal` in S-W/S-T · weight without `--s2-labels` on a real run ·
labels with the weight 0 (unless acked — a deliberate load-only rehearsal) ·
missing labels path (the `--gate-probes` lesson: it would otherwise die
after the corpus build) · negative weight · **`--no-isolate-planner`
combination, unconditionally**. The production S-S line adds exactly:

```
--w-s2-goal 1.0 --s2-labels <…/2026-08-16-s2-v1-labels/labels>
```

⚠️ **Expected supervision density on the real corpus (ESTIMATED, stated so
nobody reads the log wrong):** only the **201 aug120 labels sit in the
train corpus** (the 596 val labels are the eval side's); ≈40 in-band window
starts per labeled clip ⇒ ~8k of ~410k windows (~2 %) carry the term, and
batches with `s2_n_valid: 0` are normal and harmless (zero-in-graph). If
S-S wants denser S2 signal, a label-aware sampler arm is a DECLARED
decision to pre-register — O4 is label-free by design and must not be
quietly bent (parity + attributability).

## 6. Escalations (named lanes)

1. ⚠️ **THREE STREAMS ARE CO-TENANTS IN `train_v6_staged.py` THIS TURN, and
   the staged blob carries all three** — this stream's S2, the diffusion/
   MPC/fallback stream (`STAGE_MAY_INTRODUCE["S-T"] += ("prop_diffusion.",
   "fallback.")`, `--mpc*` launch surface), and the monitoring stream
   (`LayerSpectrumMonitor`, `sigreg_trend_verdict`, `X4_rank_retention`).
   `git add` stages whole files, so co-tenancy in the index is structural,
   not a mistake; said here so the eventual commit message can attribute.
   The hunks are disjoint (no overlapping edit) and the combined tree is
   green: the S2 battery + the identity suites re-ran after the siblings'
   latest edits (16:25) — **203 passed / 1 skipped**.
   ⛔ **The one real hazard: the staged trainer imports
   `LayerSpectrumMonitor`/`sigreg_trend_verdict`, which exist only in the
   working tree's `v6.py` — and `v6.py` is NOT staged** (the monitoring/
   planner streams' in-flight file, deliberately not staged by this stream:
   staging a sibling's mid-edit model file freezes a half-done state). A
   commit of today's index would be **import-broken** until `v6.py` is
   staged by its owner. Whoever commits: verify `git ls-files --stage
   stack/tanitad/models/v6.py` moved past HEAD first.
2. **`v6_chain.py` does not yet carry `--s2-labels/--w-s2-goal` in its S-S
   argv** — deliberately untouched here (the chain is the one place launch
   lines are constructed; adding the flags is a small follow-up in the same
   lane as the `--tac-goal-cond` carry tests). Until then an S-S chain
   launch runs S2-off, which is the safe default, not a wrong run.
3. **The `goal_provenance` promotion trigger is now ARMED**:
   `STAGE_GATE_SPEC["S-S"]` keeps it `reported` with the recorded promotion
   condition "the moment S2 lands" (DIAGRAM_CONFORMANCE.md) — S2 has now
   landed. The audit instrument is still to build (gate-owner lane); this
   trainer already banks its raw material in `config.json["s2"]`
   (provenance census, disjointness verdict, join counts).
4. **PI items unchanged** from the label build: LANE_TARGET video
   spot-check (§5 there), the ±1=left convention ratification,
   KEEP_CORRIDOR fold.
5. **No `v6.py` change was needed for S2** — the diagram's seam
   (`z_str_p` detached under the planner cut) already carries the rule; this
   stream edited no line of the sibling-owned file.

## Deliverable manifest

| artifact | where |
|---|---|
| This report | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-s2-loss/S2_LOSS.md` |
| The loader/join module (NEW) | `stack/scripts/s2_labels.py` |
| The term + weights + CLI + preflights + dry-run/train wiring | `stack/scripts/train_v6_staged.py` |
| The test battery (NEW, 40 tests) | `stack/tests/test_v6_s2_loss.py` |
| Banked end-to-end dry-run on the REAL labels (CLI path) | `…/2026-08-16-s2-loss/raw/dryrun-ss-s2/{dry_run,config,stage_gate}.json` |
| Suite records (pre/post) | `3396/0/17/2` → combined-tree `3486/3/17/2`, the 3 failures a concurrent stream's pin (§3); S2 in isolation `39/0/1` (this box; tails in the session scratchpad, counts restated here) |

Everything staged, nothing committed, nothing pushed (AGENT_OPERATING_STANDARD).

**Evidence classes.** Gradient reach, leak structure, identity-vs-anchor,
loader/join behaviour, real-artifact censuses, suite counts: **MEASURED
(ours; this box, this run — artifacts above)**. The label-side quality
findings (G1 verdict, LANE_TARGET flag, provenance audit): **INHERITED**
from `S2_V1_LABELS.md`/`S2_STRATEGIC_GAP.md`, with the censuses
independently re-derived here and matching. Supervision density on the real
corpus: **ESTIMATED** (window arithmetic; no v2 cache on this box to count
against). `v6.py` seam facts (`_cut`, head inputs, vocab non-participation):
**MEASURED** via the reach probes, not read off comments.
