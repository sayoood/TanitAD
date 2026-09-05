"""Append the Rung A1 rows to `Project Steering/GOALS_AND_CLAIMS.md`.

⛔ BYTE APPEND WITH READ-BACK — never a str-split/rejoin. `insert_rows.py` was
retired 2026-09-04 for exactly that bug: reading a large UTF-8 register into a
string, splitting it, and rejoining it re-encodes the WHOLE file, so a single
mis-detected encoding or a mount drop mid-write rewrites bytes nobody edited.
Here the existing bytes are never read into memory for writing at all: the block
is appended in "ab" mode and verified by seeking to `size_before` and comparing
the tail byte-for-byte.

Afterwards the sibling rows named in the brief are re-asserted POSITIVELY, so a
truncation cannot pass as success.

Idempotent: refuses if the marker is already present.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD")
REG = REPO / "Project Steering" / "GOALS_AND_CLAIMS.md"
MARKER = "## D-RUNGA1 —"
SENTINELS = ["D-REFCV5-LADDER-1", "D-REFCV5-PLAN-1", "D-V5A-", "D-V5A-CAM3",
             "D-REFAV1-SURFACE-PAIRED", "D-REFCV5-LADDER-2"]

BLOCK = """

## D-RUNGA1 — the REF-C harness reads the whole acceptance bar, and all twelve pre-registered ablations are runnable (2026-09-05, Benchmarks & Eval FlyWheel)

⭐ **CLOSES BOTH HALVES OF `D-REFCV5-LADDER-2`.** Zero GPU; nothing launched; `refcv4b-b1-v72-40k`
untouched on `tanitad-refcv3`. Package:
`TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-05-rung-a1-harness/RESULT.md`.

| id | claim / hypothesis | status | evidence |
|---|---|---|---|
| D-RUNGA1-1 | ⭐ **`ha0_ext` IS NOW AN ARM OF THE REF-C HARNESS, SO "BEAT BOTH `ha` AND `ha0_ext`" IS READABLE ON THAT SURFACE.** It is ported as the **same shared call the refav1 harness makes** — `refav1_arm.hold_ext_controls`, generalised with an explicit `stride` whose **default 2 reproduces refav1's read bit-exactly** (pinned), invoked from `refcv3_arm.py` at the corpus's native 0.1 s tick (`stride=1`, `dt=0.1`, `loader=None`, that argument being read only for its `dt`). ⛔ **Nothing is re-derived in `refcv3_arm.py`** — asserted by `__code__.co_filename`, not by a name that a local copy could shadow. The arm reaches the RECORD, not only the loop: `ARM_TIERS` (**T1**), `ARM_MEANING`, the DEFAULT arm list, the manifest (the shared function's own docstring + the literal call), the action-unit block (the steer→kappa conversion applies to `ha0_ext` as to `ha`), and the paired blocks `paired_os_minus_ha0ext` and `paired_os_navzero_minus_ha0ext` | **SUPPORTED (MEASURED 2026-09-05, harness-level; ⛔ NO model number is claimed here)** | `repo:taniteval/tools/refcv3_arm.py`, `repo:taniteval/tools/refav1_arm.py`; pin `repo:stack/tests/test_refcv3_ha0_ext_shared.py` (**12 passed**) |
| D-RUNGA1-2 | ⛔⛔ **THERE ARE TWO `ha0_ext` KINEMATICS IN THE PROGRAMME AND THEY DISAGREE BY METRES, SO THE CHOICE IS LOAD-BEARING.** `tanitad.eval.echo_gate.ha0_ext` → `refc_v3.kinematic_goal_extrapolation` is the CLOSED FORM in the goal label's frame; `refav1_arm.paths_from_controls` → `refa_v1_plan.unicycle_paths` is the DISCRETE 0.1 s unicycle that `ha`, `ha0` and refav1's own `ha0_ext` are integrated through. MEASURED over 6 s: they agree **EXACTLY (0.000000 m)** on the degenerate straight-constant-speed case and diverge up to **1.862923 m** at (v0 15.0, a0 1.00, k0 −0.020), with 0.460776 / 0.338136 / 0.340827 m on three further plausible states, and **0.540642 m already at 2 s** on the worst — *the same order as the entire model margin* (refcv3 `os` ADE ≈ 0.44 m @2 s). ⇒ calling the closed form in the harness would have produced a REF-C `ha0_ext` **incomparable with refav1's**, and the cross-harness comparison would have measured the integrator instead of the model | **SUPPORTED (MEASURED 2026-09-05, corpus-kinematic tier — ⛔ NOT a driving number, NOT T1)** | pinned by `test_the_two_shared_kinematics_disagree_by_metres_so_the_choice_is_material` in `repo:stack/tests/test_refcv3_ha0_ext_shared.py` |
| D-RUNGA1-3 | ⭐ **ALL TWELVE ABLATION ARMS OF `PREREG_REFCV4B_HIERARCHY_EVAL.md` §3 NOW HAVE A CLI ROUTE.** The eight that had none get `--ablate {gstr_zero, gstr_shuffle, e7_off, e9_off, h19_off, ego_zero, sel_refined, frames_blind}`, plus `--ablate-frames` as its own named flag and `--gstr-bank` / `--gstr-shuffle-seed`; the other four already had routes (`os_navzero`, `os_navshuf`, `--with-navflip`, and FULL needing none). The bijection with §3 is **pinned**, not asserted. Every flag is stamped into the arm's own record: the manifest carries `ablation`, `analyze_refcv3` copies it onto **every** per-arm block, and a dump whose manifest predates the stamp reads **UNKNOWN — never "none"** (*absence of the field is absence of the record*). Two regimes in one `--dump-dir` are refused, and an `ABLATION.txt` marker names the regime | **SUPPORTED (MEASURED 2026-09-05, harness-level)** | `repo:taniteval/tools/refcv3_arm.py::ABLATIONS`; pin `repo:stack/tests/test_refcv3_ablations.py` (**28 passed, 1 skipped**) |
| D-RUNGA1-4 | ⭐ **EVERY ABLATION FLAG CHANGES THE FORWARD PATH, OR THE TOOL REFUSES — A FLAG THAT PARSES AND DOES NOTHING IS WORSE THAN A MISSING ONE.** Pinned by rolling each ablation on a CPU fixture and asserting at least one banked array moves against the FULL roll. Refusals where the switch would be silently inert: `sel_refined` at diffusion `steps == 0` (`refc.py:1596` leaves `refined is conf` BY CONSTRUCTION; `:1630` gates `score_emitted` on `steps > 0`), `ego_zero` on a build fed no ego block, `gstr_shuffle` without a bank, a hier-only edge on a flat build, `h19_off` on a decoder with no anchor-prior head. And `e9_off` **reports itself INERT** when the trained `goal_gate` is exactly 0.0 — a FINDING about the checkpoint (Caveat-B: the zero-init gate never opened), ⛔ never a licence to write *"no effect, therefore the seam is inert at eval"*. The `g_str` injection is **VERIFIED against the model's own emitted `g_str`** on the first window and the roll is REFUSED above 1e-4: a hook registered on the wrong module raises nothing and reports nothing | **SUPPORTED (MEASURED 2026-09-05)** | same pins |
| D-RUNGA1-5 | ⛔⛔ **`PREREG_REFCV4B_HIERARCHY_EVAL.md` §3's REGISTERED `H19-OFF` MECHANISM DOES NOT EXIST ON THE CHECKPOINT IT IS REGISTERED FOR.** §3 specifies `decoder.maneuver_to_anchor = None`. MEASURED 2026-09-05: every REF-C **v3/v4** build — including `refcv4b` — sets `core.factored_maneuver = True` (`refc_v3.py:437`, `:615`), so `refc.py:1214-1221` builds the FACTORED pair `lat_to_anchor` + `lon_to_anchor` and leaves `maneuver_to_anchor` **None**; the literal mechanism would have removed nothing. The switch now removes whichever anchor-prior heads the build actually carries (`refc.py:1572-1581`'s `terms`), records **which**, and refuses only when there are none. ⇒ **the PREREG needs an ERRATUM, not the code a workaround** — and because the prereg's falsifiable object is its **git blob id at staging time**, that erratum must be a NEW separately-staged statement, never a silent edit | **SUPPORTED (MEASURED from source, two paths: the config defaults and the decoder constructor) — ⛔ ESCALATED, NOT ADJUDICATED** | `repo:stack/tanitad/refs/refc_v3.py:437,615`; `repo:stack/tanitad/refs/refc.py:1214-1221,1572-1581`; the correction is carried inside `ABLATIONS["h19_off"]["⛔ prereg_correction"]` |
| D-RUNGA1-6 | ⚠️ **THE PREREG'S OWN §7 ESCALATION LIST IS INCOMPLETE:** it names SEVEN missing switches and **omits the frame-blind DELIBERATE REGRESSION**, §3's last row — the arm the panel's validity rests on, because *a gate that has never been shown to FAIL an image-blind arm certifies nothing* (`H-ECHO-4`, where an ADE-scored gate once passed an echoing arm). ⇒ §3, not §7, is the authoritative list; the tool's registry mirrors §3 and says so in its own comment, and `--ablate-frames` exists as a **named** flag so the arm can be run without remembering an enum member. ⭐ Its INTERNAL CONTROL is pinned: `ha`, `ha0` and `ha0_ext` read no frames and must come back **bit-identical** to the FULL roll while `os` moves — if a control moved, the ablation leaked and every margin computed against it would be measuring the leak | **SUPPORTED (MEASURED 2026-09-05; the prereg text was read directly)** | `repo:Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` §3 vs §7; pin `test_frames_blind_moves_the_model_but_not_the_model_free_controls` |
| D-RUNGA1-7 | ⚠️ **A PRE-EXISTING SUITE FAILURE, DEMONSTRATED PRE-EXISTING RATHER THAN ASSERTED.** `stack/tests/test_closedloop_floor.py` fails at COLLECTION with `ImportError: cannot import name 'FLOOR_ARMS' from 'closedloop_drive'` (`stack/experiments/alpasim-gsplat/closedloop_drive.py`). Neither file is touched by this rung, and the identical error reproduces with the **UNPATCHED** copies of `refcv3_arm.py` / `refav1_arm.py` swapped back in (md5-verified restored afterwards). ⇒ a WORK ITEM for whoever owns the AlpaSim closed-loop driver, not a regression here | **MEASURED 2026-09-05 (off-Drive mirror, CPU)** | package `.../2026-09-05-rung-a1-harness/RESULT.md` §4.1 |
| D-RUNGA1-8 | ⚠️ **THE PANEL HAS A RUN-ORDER DEPENDENCY THIS RUNG DOES NOT REMOVE:** `gstr_shuffle` needs a banked **FULL** dump on the same episodes AND the same window grid, because the registered mechanism permutes `g_str` **ACROSS WINDOWS** and this harness's batch rows are the nav CONDITIONINGS of one window — permuting them would permute nav, not windows (the `--with-navzero` defect in a new costume). The tool refuses loudly without `--gstr-bank`, but "roll FULL first" is a runbook fact and belongs in the panel's launch script | **OPEN — a runbook item for the post-training panel** | `repo:taniteval/tools/refcv3_arm.py::load_gstr_bank` |

⚠️ **HONEST SCOPE, carried with every row above.** This rung is **harness work**. Every number in it
is a **source fact** (an absence with a same-breath control, a line reference read from the worktree)
or a **kinematic fact** (the integrator divergence, corpus-kinematic tier). ⛔ **There is no T0 or T1
model number here, and none of these rows says anything about whether refcv4b clears the bar** — they
say only that the bar can now be READ and that the twelve arms can now be RUN. The acceptance-bar
claims of `D-REFCV5-PLAN-9` and `D-REFCV4-*` remain exactly as open as they were.

⛔ **Two conflicts are RECORDED AND NOT ADJUDICATED** (`D-RUNGA1-2` vs `D-REFCV5-LADDER-2`'s wording
*"must call the same `echo_gate.ha0_ext`"*, and `D-RUNGA1-5`'s prereg erratum). Both are Master Mind
decisions. No document was silently edited to make them agree.
"""


def main():
    for attempt in range(20):
        try:
            raw = REG.read_bytes()
            if not raw:
                raise OSError("empty read")
            if MARKER in raw.decode("utf-8"):
                print("ALREADY APPLIED — marker present; refusing to duplicate")
                return 0
            size_before = len(raw)
            del raw                       # never rewrite what we did not edit
            data = BLOCK.encode("utf-8")
            with open(REG, "ab") as fh:   # ⛔ APPEND BYTES. No split, no rejoin.
                fh.write(data)
                fh.flush()
            # ---- READ-BACK: the tail, byte for byte -------------------------
            with open(REG, "rb") as fh:
                fh.seek(size_before)
                tail = fh.read()
            if tail != data:
                raise SystemExit(
                    f"⛔ READ-BACK MISMATCH: appended {len(data)} bytes, tail "
                    f"reads {len(tail)}. The file may be damaged — inspect "
                    f"before retrying, do NOT append again.")
            after = REG.read_bytes().decode("utf-8")
            missing = [s for s in SENTINELS if s not in after]
            if missing:
                raise SystemExit(f"⛔ SENTINEL ROWS LOST: {missing}")
            print(f"APPENDED {len(data)} bytes; {size_before} -> "
                  f"{size_before + len(data)}; read-back OK; sentinels "
                  f"{SENTINELS} all present")
            return 0
        except OSError as ex:
            print(f"  [attempt {attempt}] {type(ex).__name__} {ex}", flush=True)
            time.sleep(12)
    raise SystemExit("the mount never let the append through")


if __name__ == "__main__":
    sys.exit(main())
