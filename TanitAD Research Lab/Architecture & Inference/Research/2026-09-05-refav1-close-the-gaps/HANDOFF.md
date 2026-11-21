# HANDOFF — refav1 close-the-gaps (2026-09-06, ~02:45 Europe/Berlin)

## DELIVERABLE MANIFEST — every artifact and where it lives

**All committed to `agent/arch-inf-20260803`, verified by content markers inside the committed
blob with an absent-marker control reading 0 (never by blob-equality alone — see `ARCH-C`).**

| artifact | where | commit |
|---|---|---|
| `w_kappa_by_goal` implementation | **repo** `stack/tanitad/refs/refa_v1.py` | `d290cb9` |
| `--w-kappa-by-goal` CLI + guards + record/sidecar columns | **repo** `taniteval/tools/refav1_arm.py` | `d290cb9` |
| 15 tests (flag-off parity, parser, classifier, mechanism, admissibility) | **repo** `stack/tests/test_refa_v1_goal_kappa_cost.py` | `d290cb9` |
| pre-registration (§6 amendment before any A3 result) | **repo** `…/2026-09-05-refav1-close-the-gaps/SPEC_CLOSE_THE_GAPS.md` | `d290cb9` |
| RESULT (§A–§E) | **repo** `…/RESULT.md` | `d290cb9`, `89b9fd9` |
| frontier + 3 seed-floor pairs | **repo** `…/raw/frontier.py`, `frontier.txt` | `d290cb9` |
| ADE attribution by goal token (5 pairs) | **repo** `…/raw/attribute.py`, `attribute.txt` | `89b9fd9` |
| Kamm non-firing discharge | **repo** `…/raw/kamm_bind.py`, `kamm_bind.txt` | `d73bf05` |
| A3 reader + within-arm control | **repo** `…/raw/a3_read.py`; **Thor** `/home/nvidia/refav1_ctg/a3_read.py` | `89b9fd9` |
| Thor A3 queue | **repo** `…/raw/queueCTG.sh`; **Thor** `/home/nvidia/refav1_ctg/queueCTG.sh` | `d290cb9` |
| Thor A4 queue | **repo** `…/raw/queueCTG2.sh`; **Thor** `/home/nvidia/refav1_ctg/queueCTG2.sh` | `89b9fd9` |
| dev-box queue (A1/A2 + dev-box `gkappa`) | **repo** `…/raw/gap_queue3.py`; running from **scratchpad** | `d290cb9` |
| claims register | **repo** `Project Steering/GOALS_AND_CLAIMS.md` | `d290cb9`, `d3a5251`, `89b9fd9`, `d73bf05` |
| retraction `ARCH-C` + addendum | **repo** `Project Steering/RETRACTION_LOG.md` | `4baa7fe`, `89b9fd9` |

⛔ **NOTHING IS STRANDED.** The only off-repo copies are the isolated code trees
(`C:/Users/Admin/tanitad-ctg`, `/home/nvidia/refav1_ctg/code`), which are byte copies of the
committed files — verified by md5 on both sides.

## LIVE WORK — what is running and what it will produce

| where | arm | state | reads against |
|---|---|---|---|
| **Thor** | `G_gkappa` (`--w-kappa-by-goal 15.11245,0.0`) | RUNNING, ~5/8 eps | `T_wk15` (in-rig) + the **within-arm** LANE_KEEP control |
| **Thor** | `G_gkappa_inv` (`0.0,15.11245`) | RUNNING | ⛔ deliberate regression; must be worse than `G_gkappa` |
| **Thor** | `A4a_gk_kt02`, `A4b_kt02` | QUEUED behind them (3-arm gate) | ⭐ the drive-AND-turn candidate + its attribution arm |
| **dev box** | `kammshift`, `wk1`, `gkappa`, `wk3`, `wk7` | QUEUED, deferring to `ta_queue3.py` | A1 prediction 0.886; A2 knee |

**To read the Thor arms the moment they land:**
```
ssh tanitad-thor 'cd /home/nvidia/refav1_ctg &&
  PYTHONPATH=code/stack:code/taniteval PYTHONIOENCODING=utf-8 \
  /home/nvidia/venvs/tanitad-edge/bin/python a3_read.py out /home/nvidia/refav1_lon/out'
```
It prints the four families with **turn recall beside ADE**, the realised curvature by decoded
goal token, and the **within-arm control** (LANE_KEEP must be identical to `T_wk15`, TURN must
differ). ⛔ **If both columns differ, or neither does, the A3 reading is VOID.**

## THE THREE THINGS THE NEXT CONTEXT MUST NOT RE-DERIVE

1. ⭐⭐ **The turn collapse is a 3.9x UNDER-TURN, not a suppression** — `frac k != 0` stays
   1.0000 on TURN_L; the magnitude falls 0.08000 → 0.02066. Recall only thresholds it.
2. ⛔⛔ **The goal's commanded 0.08 (R 12.5 m) is WORSE THAN DRIVING STRAIGHT on this corpus**
   — TURN_L ADE 1.5043 (obeying) vs `ha0` straight-line 1.4630, and more penalty monotonically
   improves ADE. **So A3 alone recovers recall and worsens ADE, by arithmetic, and that is A3
   correctly obeying a wrong goal.** A4 is the pair that fixes both.
3. ⛔ **The Kamm constraint never fired on the turn windows** (0/13 could bind at v0 2.78 m/s).
   Its "zero turning cost" is a panel-speed property; at 20 m/s the same circle cuts 0.08 to
   0.0172 and would under-turn harder than `wk15`.

## OPEN WORK ITEMS, NAMED

* ⛔ **A highway-speed turn panel does not exist.** Every turning claim here is **low-speed**
  (TURN_L v0 2.851, TURN_R 2.782 m/s). Nothing transfers to highway turning without one.
* ⛔ **No arm in this package has an inference-seed replicate of its own configuration.** The
  ADE bars use the *unconstrained* neighbour's 0.06070 floor as the conservative choice. Turning
  verdicts are safe (recall floor 0.00000 on three pairs).
* ⚠️ **223 phantom staged deletions in the shared index** (223/223 classified PHANTOM: present in
  HEAD *and* non-empty on disk; 0 real). `git reset --` was blocked by a live sibling's
  `index.lock`, which must not be unlinked while git is alive. `mktree_commit.py` is immune, but
  **any pathspec-free commit by another stream would mass-delete.** Flagged, not fixed.
* ⚠️ `distance_keeping` **UNAVAILABLE** (no lead block on this panel) and `strategic`
  **UNAVAILABLE** (no route label) — stated per family with the reason, and both are work items,
  not passes.

## TWO OPS FACTS MEASURED WHILE THE ARMS RAN — both cost someone time tonight

### 1. ⭐ THOR SUSTAINS **SIX** CONCURRENT refav1 ARMS, NOT THREE — throughput +30 %

`queueTHOR.sh` uses `MAXJ=3` and my queues copied it. MEASURED 2026-09-06 00:29–00:31Z when a
sibling stream added three arms on top of my three, taking the box to **six**:

| state | s/episode/arm | arms | episodes/s | note |
|---|---|---|---|---|
| 3 arms | ~620 | 3 | 0.00484 | the assumed ceiling |
| **6 arms** | ~951 | 6 | **0.00631** | **+30 % net throughput** |

**It is not thrashing, and that was checked rather than assumed** — a two-sample CPU-time probe 90 s
apart showed **all six** counters advancing (mine 1492->1519 and 1488->1517; the newest 246->306),
memory **29 GB used of 122** with 93 GB available, GPU **98 %**. ⚠️ Approximate: the 951 s episode
spans the interval in which arms were being added, so it is an upper bound on the slowdown and the
+30 % is therefore a lower bound. ⇒ **`MAXJ=3` / `MAXARMS=3` is conservative and is costing the
programme eval throughput.** Worth re-measuring deliberately before raising it in the queues.

### 2. ⛔ `mktree_commit.py` EXITS 1 ON A SUCCESSFUL COMMIT WHEN THE SUBJECT HAS NON-ASCII

MEASURED on commit `ac16b45`: the tool created the commit, printed
`VERIFIED in HEAD by blob comparison: 5 path(s)`, and **then died** at its last line —
`print(git("log", "--oneline", "-1"))` — with `UnicodeEncodeError: 'charmap' codec` because the
commit **subject** began with `⛔⛔` and this box is cp1252. **The commit had already landed.**
⇒ A non-zero exit from this tool is **not** evidence the commit failed, and a zero exit was never
evidence it succeeded — which is exactly why the standing rule is to verify by a **content marker
inside the committed blob** with a control that must read zero. That check is what settled it here.
⚠️ **Not fixed by me on purpose:** `stack/scripts/mktree_commit.py` is being actively edited by the
refcv5 stream (`81bc9c8` restores guards it says were deleted), and racing them on one file is how
work disappears. **The fix is one line** — wrap that final `print` (or set
`PYTHONIOENCODING=utf-8` in the tool's own environment). **Escalated here rather than written into
a doc nobody re-reads.**
