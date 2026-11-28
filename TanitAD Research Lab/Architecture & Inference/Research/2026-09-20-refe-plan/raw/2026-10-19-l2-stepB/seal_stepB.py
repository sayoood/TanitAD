#!/usr/bin/env python3
"""Seal Step B: build PREREG_L2_STEPB.md from the draft + the MM ruling (Revision 1, VERBATIM), pin everything, write the .sha256 and SEALED_AT.txt, then prove the seal guard accepts it (and refuses a one-byte edit).
Run ONCE, after the code is final (the pins bind the code's sha256).  Every edit of the draft is an exact-string replacement that must hit exactly once (no silent miss).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe seal_stepB.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

PKG = "C:/Users/Admin/par18/l2stepb"
sys.path.insert(0, f"{PKG}/code")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import stepB  # noqa: E402
import stepB_chain as C  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def lf(path: str) -> str:
    return open(path, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")


def sub1(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"edit target hit {n} times (must be exactly 1): {old[:90]!r}")
    return text.replace(old, new)


def slice_replace(text: str, start: str, end: str, new: str) -> str:
    """replace text[start .. end) (start inclusive, end exclusive) by `new`; both markers must be unique"""
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"slice markers must be unique: {start[:60]!r} ({text.count(start)}), {end[:60]!r} ({text.count(end)})")
    a, b = text.index(start), text.index(end)
    if b <= a:
        raise SystemExit(f"slice end before start: {start[:40]!r}")
    return text[:a] + new + text[b:]


def run_tests() -> tuple:
    r = subprocess.run([PY, "-m", "pytest", "test_stepB.py", "test_stepB_chain.py", "-q", "-p", "no:cacheprovider"], cwd=f"{PKG}/code", capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    tail = r.stdout.strip().splitlines()[-1]
    m = re.search(r"(\d+) passed", tail)
    if r.returncode != 0 or not m or "failed" in tail or "error" in tail:
        raise SystemExit(f"tests are not green, no seal: {tail}")
    open(f"{PKG}/logs/pytest_test_stepB_sealed.log", "w", encoding="utf-8", newline="\n").write(r.stdout)
    return int(m.group(1)), tail


def main() -> int:
    dry = len(sys.argv) > 2 and sys.argv[1] == "--dry-run"                      # --dry-run <path>: build the text, run its self-checks, write it to <path>, seal NOTHING
    if not dry and os.path.exists(f"{PKG}/PREREG_L2_STEPB.sha256"):
        raise SystemExit("already sealed: a sealed text is never rewritten (a change goes into a sealed amendment)")
    draft, ruling = lf(f"{PKG}/PREREG_L2_STEPB_DRAFT.md"), lf(f"{PKG}/MM_RULING_STEPB.md")
    cfg = C.Cfg()
    n_pass, pytail = (0, "dry run") if dry else run_tests()
    fresh_sha = lf(cfg.fresh_sha).split()[0]
    pins = C.collect_pins(cfg, fresh_prereg_sha256=fresh_sha)
    pins["ruling_sha256"] = hashlib.sha256(ruling.encode()).hexdigest()
    pins["draft_sha256"] = hashlib.sha256(draft.encode()).hexdigest()
    short = lambda k: pins[k][:8]  # noqa: E731

    t = draft
    # ---------------------------------------------------------------- A. title, intro, Revision 1 (verbatim), the map
    intro = f"""# PREREG L2-STEPB (SEALED): L1 Step B, the CEM search, on P2's scorer with the L2 EP head

Status: SEALED 2026-10-19 (the exact time is in `SEALED_AT.txt`; this text's sha256 is in `PREREG_L2_STEPB.sha256`; the detached chain `code/stepB_chain.py` refuses to run without both, section 11)

Written 2026-10-19 by the L2-STEPB-BUILD agent as a draft (`PREREG_L2_STEPB_DRAFT.md`, sha256 `{short('draft_sha256')}…`), amended by the Master Mind's ruling (**Revision 1, below, verbatim**) and SEALED before any Step B search ran.
**No Step B search has been run, no search output has been labelled, no dPDMS has been read, and no carve-B label or score was read by this agent: nothing in this file is an outcome.**
It is an amendment to `PREREG_L1_SEARCH_ADMISSION.md` (sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`, section 4 O1 and section 5), L1 Amendment 1 (sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff`, C8)
and `PREREG_L2_OFFPROP.md` (sealed, sha256 `3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983`, outcome L2-ADMIT); none of the three is edited. It stands beside `PREREG_L2_FRESH.md` (sealed 2026-10-19 16:25, sha256 `{short('fresh_prereg_sha256')}…`), which defines carve B.
Evidence classes: MEASURED (ours, artifact named), INHERITED (another package, not re-verified), ESTIMATED, HYPOTHESIS.

## Revision 1 (MM ruling, pre-seal)
The Master Mind's ruling on the draft, folded in VERBATIM (file `MM_RULING_STEPB.md`, sha256 `{pins['ruling_sha256']}`). **Where the body below (the draft, edited only where the ruling says so, each edit tagged `[Rev1 #n]` or `[Rev1 D0]`) and the ruling differ, THE RULING PREVAILS.**

````markdown
{ruling.rstrip()}
````

### How the ruling is applied to the body (a map, not a second source)
| ruling | applied in |
|---|---|
| Decision 0: a diagnostic on carve B now; licensing UNCHANGED; stamps | section 0, section 1 (data), section 7 (stamps), section 10 (licensing), section 11 (chain) |
| 1 units (PDMS points) | section 3 |
| 2 seed-spread term | section 3 |
| 3 worst of the two head seeds | section 3 |
| 4 P2HEAD control kept | section 1 |
| 5 NOT CLEARED => L2-BRANCH; O4 => raised-M arm once, then CLOSED with a NAMED next lever | section 3 table; `analyse_stepB.NEXT_ARM` (printed by the chain) |
| 6 start and baseline = P2's pick | section 0 |
| 7 clips stay at the hull | section 2 |
| 8 E7 narrowed | section 5 |
| 9 EP-ADMISSION on BOTH banks, built only after an O1 | section 6 |
| 10 TRADE-restricted arm built ON a TRADE | section 3 table |
| the chain | section 11 |

"""
    t = slice_replace(t, "# PREREG L2-STEPB (DRAFT, NOT ADOPTED)", "## 0. When this launches, and what it is", intro)

    # ---------------------------------------------------------------- B. section 0
    s0 = """## 0. When this runs, and what it is
- **[Rev1 D0] Step B runs NOW, as a pre-registered DIAGNOSTIC on carve B; it licenses nothing alone.** The draft's precondition (Step A O1 first) assumed Step B was the costly stage; it is not (MEASURED, section 8: 1.8-2.7 CPU-h for three arms, no GPU) and it is the only read of the actual endpoint (the true PDMS of the search output against P2's pick).
  It launches once the L2-FRESH B chain prints `CHAIN_DONE_NO_READ` with its gates holding and BOTH amendments are sealed (section 11). **Licensing is UNCHANGED:** EP-ADMISSION and the shared navtest look require a Step A O1 (READ 1 or READ 2) AND a Step B O1 on BOTH head seeds. Every outcome of Step B is reported whatever Step A reads, so no result is a forking path.
- **What it is:** L1's Step B (`tts_cem.cem_search`, landed, frozen, sha-pinned) run against P2's scorer with ONLY the EP row of `score_head` replaced by the L2 head file. An INFERENCE-time search on CPU: no optimiser, no loss, no gradient.
- **[Rev1 #6] Search start and baseline = P2's own pick** (`bundle['pick']`, the banked argmax of P2's aggregate), not the L2-scorer's in-fan argmax. The L2 head was validated (A1) on perturbations OF P2's pick, so searching around that pick keeps the search inside the domain the head was admitted on. The endpoint is therefore the effect of the SEARCH on top of P2's pick.
  The in-fan re-ranking by the L2 head is read separately and descriptively from banked labels (section 3, "decomposition"); a search output replaces the pick only if the L2 scorer scores it above P2's pick inside the original 64-fan (`tts_cem.cem_search`, final re-score).

"""
    t = slice_replace(t, "## 0. When this launches, and what it is", "## 1. Arms and data (fixed now)", s0)

    # ---------------------------------------------------------------- C. section 1: P2HEAD tag, the data clause
    t = sub1(t, "| **P2HEAD** (control) |", "| **P2HEAD** (control; KEPT, [Rev1 #4]) |")
    data = f"""- **Data (the DATA CLAUSE) [Rev1 D0]:** carve B's bundle (same schema as L1's `bundle.npz`: token, log, traj_raw, pick, banked sub / pdms), built by the L2-FRESH chain (`D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/`).
  **Its token list must equal the sealed carve** (400 plan tokens, seed 20261019, token-list sha256 `{pins['carve_sha256']}` = sha256 of the tokens joined by newlines in bundle order = the bundle's `ctx.json` `tokens_sha256`); **a mismatch REFUSES**, in four places: the chain (before any child process),
  the gate (`g_reach_stepb.run_gate`, no stamp is minted), the launch (`stepB.run_search_cli`, the line tagged `@DATACLAUSE`) and the `ctx.json` / `ctx.npy` sha256 check. No CLI flag overrides the value (only a Python caller, i.e. a test, can pass a planted carve's sha256).
  **RED arms, MEASURED before this seal on planted bundles:** one token replaced, one dropped, two swapped (the same SET in another order), one appended and an empty list are each refused (`test_data_clause_red_arms_*`, `test_refuses_a_token_mismatch_*`); Step A's REAL bundle (sha `211e2640…`, 400 tokens, a legitimate bundle that is not the carve) is refused by the gate's default;
  deleting the launch's `@DATACLAUSE` line lets the planted bundle past the data and the launch fails LATER (`test_data_clause_mutation_arm_*`), and replacing the chain's call by a no-op lets the swapped-token bundle run the whole chain (`test_the_data_clause_red_arm_*`): each refusal IS its line. GREEN control: the real carve's token list (a list of tokens, no label) hashes to the pinned value.
  The carve (L2-FRESH, sealed; MEASURED by that package, `raw/fresh_sample.json`): the literal carve (the held-out logs minus the 24) is EMPTY (0 tokens, 0 logs); B = 400 plan tokens outside Step A's 400, inside the SAME 24 logs (1,305 candidates; G0 A1.2 PASS). The L2 head was fitted on Step A's 400 tokens, so on B it is **token-fresh and LOG-SHARED**:
  it has seen neighbouring windows of every log it is searched on, and this is NOT a fresh-log test. Every Step B line is stamped SEEN-SCENE (L1 A1.3) and says so.
"""
    a = t.index("- **Data:** the bundle of the fresh-carve Step A re-run")
    b = t.index("- **Search seeds:**")
    t = t[:a] + data + t[b:]

    # ---------------------------------------------------------------- D. section 2: O4 config pinned now
    t = sub1(t, "It needs its own config file and sha256, written before that arm runs.", f"Its config file is written NOW, before any outcome, and pinned: `stepB_config_O4.json`, sha256 `{pins['config_o4_sha256']}` (LF); the arm needs its own G-REACH-STEPB stamp per head.")

    # ---------------------------------------------------------------- E. section 3: units, table rows, head seeds, seed spread
    t = sub1(t, "- **Units (an interpretation L1 did not state):**", "- **Units [Rev1 #1, RATIFIED as an interpretation recorded before any data; L1 did not state them]:**")
    t = sub1(t, "and +0.30 on [0, 1] is unreachable.", "and +0.30 on [0, 1] is unreachable. In points the safety tolerance -0.25 equals ONE token in 400 losing a binary sub-score: a coherent reading, and the strict one.")
    t = sub1(t, "Only an O1 on BOTH head seeds licenses the single navtest read (after section 6). |", "Only an O1 on BOTH head seeds, TOGETHER WITH a Step A O1 (READ 1 or READ 2) [Rev1 D0], licenses EP-ADMISSION (section 6) and then the shared navtest look. |")
    t = sub1(t, "Endpoint is then descriptive. |", "Endpoint is then descriptive. **Route [Rev1 #5]:** the raised-M widened arm ONCE (M = 256, 32 elites, widen 2.0, clips at the admitted hull; `stepB_config_O4.json`, section 2). If that is ALSO INERT the search is CLOSED for this scorer, and the report must NAME the next lever with what unblocks it: "
                  "L2-BRANCH (unblocked by its own sealed amendment, a `launch_gate.py` PASS token and the dev-box GPU lock) or the P6 trunk (unblocked by the PI's authorisation of the trunk compute). Closing is never the end of the turn. |")
    t = sub1(t, "**Not built** (it needs a modified elite selection plus its own identity and reach checks; built only if a TRADE occurs). |", "**Built ON a TRADE, not now [Rev1 #10]** (it needs a modified elite selection plus its own identity and reach checks). |")
    t = sub1(t, "L1 names no next arm for this row; proposed here: the EP-led search lever is closed for this scorer and the next arm is L2-BRANCH (last scorer-branch block trainable), already committed by the L2 prereg for L2-PARTIAL. |",
             "**L2-BRANCH [Rev1 #5, RATIFIED]:** the EP-led search lever is closed for this scorer and the next arm is L2-BRANCH (last scorer-branch block trainable), the arm the L2 prereg committed for L2-PARTIAL. L1 names no next arm for this row. |")
    t = sub1(t, "- **Two head seeds:** the combined outcome", "- **Two head seeds [Rev1 #3, RATIFIED]:** the combined outcome")
    t = sub1(t, "- **Seed-spread term:**", "- **Seed-spread term [Rev1 #2, RATIFIED]:**")

    # ---------------------------------------------------------------- F. section 5: re-run per head, E7
    t = sub1(t, "It must be re-run for each real head file (about 1 min).", "It must be re-run for each real head file (about 1 min): **the chain re-runs it on EACH real head (s0, s1) before that head's arm, and a stamp minted on the test head does not carry over [Rev1 #8].** "
                                                                         "Re-run after the Revision 1 edits on the TEST head and Step A's bundle (explicit planted-carve override; the default refuses that bundle): PASS, 8 of 8 mutation arms RED (`raw/gate_testhead_rev1/`).")
    t = sub1(t, "- **E7, stated:**", "- **E7, stated (NARROWED, with the reason recorded) [Rev1 #8, RATIFIED]:**")
    a = t.index("  **Where `launch_gate.py` DOES apply:**")
    b = t.index("\n\n## 6.")
    e7 = ("  **Where `launch_gate.py` DOES apply:** any arm that TRAINS with an L2 head (L2-BRANCH and beyond, a REFe retrain). The MM's phrase \"Step B and beyond\" now reads **\"any arm that TRAINS with an L2 head (L2-BRANCH and beyond)\"**; "
          "BACKLOG `L2-STEPB-GATE` is resolved by G-REACH-STEPB. The reason, recorded: `launch_gate.py` governs TRAINING launches (the PI rule of 2026-09-26, \"if we are starting a training session\"); Step B is an inference-time search with no optimiser, loss or gradient, "
          "and a profile whose G-LIVE, G-CKPT and G-CLOCK have no object would certify less than a gate built on the real risk. The right gate is G-REACH-STEPB with its sha-bound PASS stamp (head, checkpoint, config, bundle, code) re-asserted inside the launch.")
    t = t[:a] + e7 + t[b:]

    # ---------------------------------------------------------------- G. section 6: both banks
    a = t.index("- **Open (for the Master Mind):**")
    b = t.index("\n\n## 7.")
    s6 = ("- **EP-ADMISSION's object: BOTH banks, and ADMIT needs both to admit [Rev1 #9, RULED].** The thresholds, estimator and rules stay frozen; only the candidate block (appended per read) names the objects. "
          "(i) The frozen literal object: an in-fan bank of \"P2 + the L2 EP row\" (head seeds 0 / 1 = the candidate's two seeds), via a head-file input to `build_candidate_bank.py` (`l2_scorer.install_ep_head`). "
          "(ii) A search-output bank in the same schema, built from Step B's labelled outputs: that is what the deployed system picks. **Both are built ONLY after a Step B O1 (not now); the candidate block is written before either bank is read.** "
          "Order: Step A O1 (READ 1 or READ 2) AND a Step B O1 on both head seeds, then EP-ADMISSION on both banks, then the shared navtest look.")
    t = t[:a] + s6 + t[b:]

    # ---------------------------------------------------------------- H. section 7: stamps
    t = sub1(t, "SEEN-SCENE (L1 A1.3, and section 1 above);", "SEEN-SCENE (L1 A1.3, and section 1 above); the head is **TOKEN-FRESH and LOG-SHARED on carve B** (400 tokens disjoint from the 400 it was fitted on, inside the SAME 24 logs): NOT a fresh-log test [Rev1 D0];")

    # ---------------------------------------------------------------- I. section 9: the ten decisions, ruled
    s9 = """## 9. The ten decisions (Revision 1: all RULED; the ruling's text prevails)
| # | decision | ruling | applied in |
|---|---|---|---|
| 1 | units of the L1 bar | PDMS POINTS (100 x [0, 1]); RATIFIED as an interpretation recorded before any data | section 3 |
| 2 | seed-spread term | dPDMS point > (max - min) of the 3 per-search-seed points; RATIFIED | section 3 |
| 3 | two head seeds | the WORST of the two; RATIFIED | section 3 |
| 4 | P2HEAD control | KEPT (descriptive attribution; it costs one arm) | section 1 |
| 5 | NOT CLEARED / O4 | NOT CLEARED => L2-BRANCH; O4 INERT => the raised-M widened arm once; if also INERT the search is CLOSED for this scorer and the report NAMES the next lever (L2-BRANCH or the P6 trunk) with what unblocks it | section 3 table, `analyse_stepB.NEXT_ARM` |
| 6 | start and baseline | P2's pick; RATIFIED. The in-fan re-ranking stays descriptive | section 0 |
| 7 | clips under O4's widening | stay at the hull; RATIFIED | section 2 |
| 8 | E7 | NARROWED, with the reason recorded: G-REACH-STEPB, not `launch_gate.py`, governs an inference-time search | section 5 |
| 9 | EP-ADMISSION's object | BOTH banks; ADMIT needs both; built ONLY after a Step B O1 | section 6 |
| 10 | TRADE-restricted arm | built ON a TRADE, not now | section 3 table |
"""
    t = slice_replace(t, "## 9. Decisions this draft needs from the Master Mind", "## 10. Not licensed", s9 + "\n")

    # ---------------------------------------------------------------- J. section 10
    t = sub1(t, "A Step B O1 licenses EP-ADMISSION and then the shared navtest look, nothing more.", "**Step B licenses nothing alone [Rev1 D0]:** a Step B O1 on BOTH head seeds, together with a Step A O1 (READ 1 or READ 2), licenses EP-ADMISSION on both banks and then the shared navtest look, nothing more. "
                                                                                                        "Every Step B outcome is reported whatever Step A reads; this chain does not read Step A.")

    t = sub1(t, "`pytest test_stepB.py` (36 tests) green, bound by code sha.", "`pytest test_stepB.py test_stepB_chain.py` green (counts in section 11), bound by code sha.")

    # ---------------------------------------------------------------- K. section 11: chain, pins, seal
    code_pins = [k for k in pins if k.startswith("code:")]
    s11 = f"""

## 11. The chain, the pins and the seal [Rev1: the chain]
**The chain** (`code/stepB_chain.py`, ONE detached CPU process: `CUDA_VISIBLE_DEVICES=-1`, threads capped at 4 (label step 1), no lock, no GPU, no pod; below-normal priority; markers ZZSTEPB_* at the START of their lines, children's output in `run/logs/`; resumable: valid stamps, shards, merged files and label files are skipped):
(a) WAIT until the sealed text and its `.sha256` hold (the seal guard), L2-FRESH's chain printed `CHAIN_DONE_NO_READ`, L2-FRESH's own `PREREG_L2_FRESH.sha256` exists and names the same carve and heads, and L2-FRESH's gates hold (`ZZBUNDLE_OK 400`, labels G1 identity and its control, G2a / G2c / G3, cross-device pick agreement >= 0.99, C-FEAT); a FAIL marker of that chain, a dead chain with no marker, or a timeout STOPS this chain.
(b) the DATA CLAUSE (section 1) and G-REACH-STEPB on EACH real head (s0, s1): both PASS stamps exist BEFORE any search, else `ZZSTEPB_STOP`.
(c) arms L2-s0, L2-s1, P2HEAD x 3 search seeds. (d) the labels (`label_stepB.py`: the planted identity at the labeller, exit 3 = STOP). (e) the registered analysis (`analyse_stepB.py`): the four families (STRATEGIC PAUSED), O1 / TRADE / NOT CLEARED / O4, worst of the two head seeds, the stamps, the NAMED next arm. (f) `ZZSTEPB_DONE <combined outcome>`.
**Seal guard** (`stepB_chain.Chain.check_seal`, run at the start AND again after the wait; REFUSES, exit 5): the text's sha256 differs from `PREREG_L2_STEPB.sha256`; the marker `Status: SEALED` is absent; a pin below is missing or differs from the file or constant the chain will use (code files by LF sha256, L1's four files, both configs, both head files and `final_heads.json`, the carve pin vs `stepB.SEALED_CARVE_SHA256`);
an operative code file is not pinned. Editing any pinned code file after this seal makes the chain refuse; a change goes into a sealed amendment.
**Outputs** (CPU, small, on C:): `run/` beside this text: stamps, shards, merged search files, labels, `RESULT_STEPB.json`, `chain_state.json`, logs.
**MEASURED before this seal (planted inputs only; nothing of carve B or Step B was read):** `pytest code/test_stepB.py code/test_stepB_chain.py`: **{n_pass} passed** (`{pytail}`; `logs/pytest_test_stepB_sealed.log`); the chain on the REAL inputs without a seal read `ZZSTEPB_STOP SEAL_MISSING`, exit 5.

**Pins (machine-readable: the chain parses this block; `make_stage.py` is excluded because it only copies files for landing and never runs in the chain):**
{C.render_pins(pins)}

*Code files pinned: {', '.join(k[5:] for k in code_pins)}.*
"""
    t = t.rstrip("\n") + "\n" + s11
    if "Status: DRAFT" in t or ruling.rstrip() not in t or t.count("Status: SEALED") < 1:
        raise SystemExit("sealed text self-check failed (draft marker present, ruling not verbatim, or no SEALED marker)")
    if "DRAFT, NOT ADOPTED" in t or "the Master Mind must confirm it in one line" in t:
        raise SystemExit("an unedited draft statement survived")

    if dry:
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
            f.write(t)
        print("ZZDRY", len(t), "chars ->", sys.argv[2])
        return 0
    # ---------------------------------------------------------------- write, record, prove
    body = t.encode("utf-8")
    sha = hashlib.sha256(body).hexdigest()
    with open(cfg.prereg, "wb") as f:
        f.write(body)
    with open(cfg.prereg_sha, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha}  PREREG_L2_STEPB.md\n")
    b_log = (C.read_text(cfg.b_log) or "").strip().splitlines()
    state = os.listdir(cfg.b_dir) if os.path.isdir(cfg.b_dir) else None
    when = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(f"{PKG}/SEALED_AT.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"SEALED_AT {when}\nfile PREREG_L2_STEPB.md\nsha256 (LF bytes) {sha}\n\n")
        f.write("no Step B search, no search label, no dPDMS, and no carve-B label or score existed or was read at the seal.\n")
        f.write(f"state of the L2-FRESH B chain at the seal: chain_B.log last line: {b_log[-1] if b_log else '(empty)'}\n  {cfg.b_dir} holds: {state}\n")
        f.write(f"L2-FRESH seal: PREREG_L2_FRESH.sha256 = {fresh_sha}\n")
        f.write(f"tests before the seal: {n_pass} passed ({pytail})\n\npins (the block inside the sealed text):\n")
        for k, v in pins.items():
            f.write(f"  {k}: {v}\n")
    # the guard accepts the real seal and refuses a one-byte edit (a RED arm on the real files)
    out = []
    ok = C.Chain(cfg, out=out.append).check_seal()
    assert ok["carve_sha256"] == stepB.SEALED_CARVE_SHA256 and out and out[-1].startswith("ZZSTEPB_SEAL_OK " + sha), out
    tmp = tempfile.mkdtemp()
    shutil.copytree(f"{PKG}/raw", f"{tmp}/raw")
    open(f"{tmp}/PREREG_L2_STEPB.sha256", "w").write(f"{sha}  x\n")
    open(f"{tmp}/PREREG_L2_STEPB.md", "wb").write(body + b" ")
    c2 = C.Cfg(pkg=tmp)
    out2 = []
    try:
        C.Chain(c2, out=out2.append).check_seal()
        raise SystemExit("RED ARM FAILED: a one-byte edit of the sealed text was accepted")
    except C.StopChain as e:
        assert e.reason == "SEAL_MISMATCH", e.reason
    shutil.rmtree(tmp)
    print("ZZSEALED", sha, "tests", n_pass, "guard: accepts the real seal, refuses a one-byte edit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
