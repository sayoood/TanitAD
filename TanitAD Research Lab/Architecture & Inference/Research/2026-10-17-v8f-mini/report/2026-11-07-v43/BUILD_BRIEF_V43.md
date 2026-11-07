# Brief: build report v43 (the T2 attribution and the LoRA launch; the theory after six reads)

The PI, 2026-11-07, verbatim: *"dont idle continue autonmously incl. the weight space programming methods and the reasoning update our scientific paoer with al these inputs and related results"*

Read first and follow: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (the preamble).
* ⛔ Never commit, never push; the Master Mind lands. End with a deliverable manifest.
* ⛔ Never delete files: move them aside. LF line endings. Dev-box CPU only, `CUDA_VISIBLE_DEVICES=-1`.
* Read the repository only through `git -C C:/Users/Admin/tanitad-push show <tip>:<path>`, never from a working tree.
* Context budget ≤ 150k. The report is about 1 MB: read it by anchors (`grep -n`, then `sed -n` on a range), never whole. ⛔ Hand off at about 110k with `HANDOFF_V43.md` (three agents today ran past 250k; do not). Reuse v42's scripts by copying and editing anchors; do not re-read the report to understand them.
* ⛔ Banned words anywhere you write: void, superseded, retracted, formerly, "no longer", correction, withdrawn, and "closed loop" / "closed-loop". The patch script scans for them; keep it so.
* ⛔ Never write the pod's IP address, port or pod id. Call it "the PI's GPU pod (RTX 4000 Ada, 20 GB)".
* Never use a bare `"bash"` in a Python subprocess on this box: it reaches WSL. Use `shutil.which("bash")`.

## Tip, base and machinery
* **Tip:** `git -C C:/Users/Admin/tanitad-push ls-remote origin refs/heads/agent/arch-inf-20260803`. Take the live value.
* **Base:** the landed v42, `R/report/2026-11-07-v42/` (`R = TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`). md sha256 `ada20830...`.
* **Machinery:** copy `E:/Projects/TanitAD-artifacts/dagland/report_v42/` (`patch_v42.py`, `audit_v42.py`, `checks_v42.py`, `render_check_v42.js`, `build_v42.sh`, `HANDOFF_V42.md`) into `report_v43/` and adapt them. Adapt; never rewrite:
  * anchored edits;
  * `@@T` table directives, copied verbatim from the sources;
  * the italic quotes asserted verbatim in a source;
  * the numbers audit, which must end at 0 unmatched;
  * the checks, which must all PASS (v42's 33, plus new ones that tie each new headline number to its file);
  * the offline render with 0 ragged tables.

## What v43 adds (each number with its artifact path and evidence class; each driving number with its tier; four families wherever a read reports them, STRATEGIC PAUSED by PI item 25)
Every source is a raw folder on the tip, under `R/raw/`. Find exact files with `git ls-tree`. Each folder's README and RESULT file is its own summary; quote numbers from the RESULT/JSON files, never from a commit message.

1. **§7.45, the T2 attribution, B′ and F′, and the LoRA launch (2026-11-07).**
   * The attribution card `PREREG_T2_ATTRIB.md` (frozen `f7d2cfa1`; `raw/2026-11-07-t2-attrib-prereg/`): why LC − C0C carried three terms, and the two stage-5 arms MC and BPC. The stage-5 driver: `raw/2026-11-07-t2-stage5-driver/`.
   * **MC read, PT-ACTIVE** (`raw/2026-11-07-t2-mc-read/`): C0C − MC separated at both seeds on ADE6, along-track, accel, DK-HW and M1 (C0C crawls MORE). MC itself against P10: NOT-DOMINANT; M1 +0.3493 / +0.2919 against C0C's +0.4073 / +0.4041 (stage-4 read of record). Say it plainly: post-training adds 0.06 / 0.11 to a crawl that is mostly in the base at T2.
   * **BPC read, the route term RT** (`raw/2026-11-07-t2-bpc-read/`, landed before you start): report RT-WITHIN-FLOOR or RT-MATTERS exactly as the card's log entry states it, with the rows it names. The log reads RT-WITHIN-FLOOR, narrowly: at s1 all three rows are separated and below MC's seed floor by small margins. Report both facts.
   * **F′** (`raw/2026-11-07-lora-fprime/`): B′, the identity eval, F′ = 0.12701 m, 2F′ = 0.2540 m, all four families, distance keeping. The pair read and its identity: `raw/2026-11-07-t2-pair-fprime-prep/`.
   * **The LoRA arm L launched** on the PI's GPU pod 2026-11-07 02:32 UTC with a launch-gate PASS. MEASURED step time 45.27 s/step (the gate's timing) and step 50 logged at 2,258 s. About 241 h per seed is ESTIMATED. LC support: `raw/2026-11-07-t2-lc-support/`. Pin 5 of the read (seed-0-only reads, 23 of 23 mutations RED, the WSL-bash incident): `raw/2026-11-07-t2-pin5/`. ⛔ No LoRA driving result exists. Say so.
   * **The T1-kin BLIND read** (`raw/2026-11-07-t1-blind/`, landed before you start): the card `PREREG_T1_BLIND.md` (frozen `9619414a`), its outcome word as written, the gates, and B on M1 and ADE6 at both seeds. Beside it, the render-gap yardstick `gapscale/README.md` in that folder (descriptive, decides nothing). If the folder is absent, write "pre-registered, not yet read" and say what it decides.
2. **§7.46, the theory after six reads (2026-11-07).** Mathematics in the report's style, continuing §7.44.
   * **P1′b** (`raw/2026-11-07-theory-p1pb/`): T1′ falls; the numbers as its result file gives them.
   * **WSP-GAIN READ 2** (`raw/2026-11-07-wsp-gain/`): HARMS, as the frontier predicted.
   * **AUCMAP** (`raw/2026-11-07-aucmap/`): outcome (iii). No gate on the forced-level and floor levers reaches the corner, even at AUC 1.0. The binding predicate is the lever operator, not go/stay discrimination.
   * **T5 and P5** (`raw/2026-11-07-theory-amend3/`, `raw/2026-11-07-p5-dilution/`): T5 says the a0 gain is shrunk by regression dilution of a noisy finite-difference input. The frozen amendment calls a0 a 0.2 s difference. It is a one-frame (0.1 s) difference, as the `RETRACTION_LOG.md` entry of commit a6b854de records. Write 0.1 s and cite that entry. λ 0.968 [0.966, 0.970]: P5 FAILS, so dilution does not explain the shrinkage.
   * **WSP-GAIN-D** (`raw/2026-11-07-wsp-gain-d/`): the edit aimed at the data's own gains reads HARMS. READ 1 MECH-OK; M1 −0.0175 / −0.0172 (share_closed 0.329 / 0.289); nine new separated-worse rows per seed. It is the third edit to read HARMS.
   * **P4** (`raw/2026-11-07-p4-cueclass/`): the cue-class card (frozen `6717a0cf`), pre-registered before any LoRA feature exists, and its baseline.
   * **dk141** (`raw/2026-11-07-dk141/`): the distance-keeping instrument for any eval-141 run, with its identity against the banked reader.
   * **Theory Amendment 4, T6 (render-domain information loss)** (`raw/2026-11-07-t1-blind/cards/THEORY_CRAWL_AMEND_4.md`, sha `016c2950`, written before the BLIND read): the token dependence Δ(D) of the planner, the data-processing consequence at T2, and the rival T6′. P6 is the T1-kin BLIND read. Its outcome VISION-USED keeps T6 and drops T6′. Write the definitions out, as §7.44 does.
   * **The synthesis, written as an argument with each step's evidence:**
     * every admissible downstream lever lies on one chain (P3), and both weight edits harm (WSP-GAIN, WSP-GAIN-D);
     * the corner is unreachable by the lever operator (AUCMAP iii);
     * the input-noise explanation fails (P5);
     * at T2 the crawl sits mostly in the base and does not move when the scene tokens are replaced by their mean (stage-2 BLIND, X2).
     * State what the T1-kin BLIND read adds, by its outcome word.
3. **Abstract, status line, §8.1 / §8.3 / §8.4 / §9 / §10:** updated to these sections. Give each new raw folder one row in the section-10 table. In §8.4, show the next levers with their bars:
   * L's PRIMARY |ADE6(L) − ADE6(B′)| > 0.2540 m;
   * the T2 dominance read of LC by `DRAFT_PREREG_T2_DOM_LORA.md`, with whatever gates its log adds;
   * the render-alignment card `PREREG_T2_RALIGN.md` (frozen `a6c84433`; `raw/2026-11-07-t1-blind/cards/`). It holds: the GAP48 pairs, the scene-fold cross-fit, the per-cell affine map, the M-ALIGN gate (held-out cosine +0.05 and at least 0.90), and its outcomes RA-DOMINANT / RA-MOVES / RA-NULL or RA-HARMS / ALIGN-FAILS-MECH. Its tier is T2-RA, never pooled. If `t2v8f/gap48/` results exist on the tip when you build, report them as descriptive; otherwise write "pre-registered, not yet read".

## Wording rules
* Every number carries its class: [M] / [P] / [E] / [H]. Every driving number carries its tier: T0 / T1-kin / T2.
* A refuted prediction is written as refuted, with its failure condition quoted from the frozen card.
* ⛔ No good-news softening, and no post-hoc finding promoted to a result.
* No DOMINANT exists; there is no scale signal.

## Output
`report_v43/out/V8F_SCIENTIFIC_REPORT.md` and `v8f_report.html`, plus the scripts, `edits_v43.md`, the new section files, `MANIFEST.txt`, and `HANDOFF_V43.md`. The handoff gives the build command, the check counts, and the exact folder to land as `R/report/2026-11-07-v43/`. Hand-back to the Master Mind: ≤ 300 words plus paths.
