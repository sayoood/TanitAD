# P6-HN prereg: notes and open questions for the Master Mind and the PI

Written 2026-10-18 on tip `3a28bc19`. The draft is `PREREG_P6HN.draft.md`; its sha256 is in the `.sha256` file next to it. It is a DRAFT and is not registered.

## Questions (each one blocks only P6-HN)

**Q1. PI: compute, and whether to screen cheaply first. This is the headline.**
- **Cost.** The arm costs 6.8-7.65 A40-days, plus the same again for a conditional replicate (draft §12, ESTIMATED).
- **Why the arm will probably not separate.** The draft's power section (§10, ESTIMATED) sets two numbers side by side:
  - **Detection threshold.** HN-ADOPT needs a point Δ above about 0.3-0.6 PDMS. A lever claim needs a point Δ above about 0.63-0.95.
  - **Expected effect.** The only published scorer-form PDMS effect is +0.22 (DA-WAM Tab. 3, a single run; I re-read it in the banked PDF).
    The gross reach is 0.29-0.94, ESTIMATED.
  - So the likeliest outcome is HN-NOT-PROVEN.
- **The three options:**
  - **(a)** Run the draft as written, after P6 reads.
  - **(b)** First, a scorer-only screen under its own prereg. The trunk is frozen on P6 (or on u400 now). A scorer-only fine-tune is run with
    and without the bank, at seeds 0-2.
    - It is cheaper, and it carries its own replicate.
    - It should be tighter. A scorer change on a fixed fan read R0B at a CI half-width of about 0.30 (INHERITED, `D-REFE-FREEPOINT-1`).
    - Its limit: it answers the narrower, head-only question. HN-BLIND results from it would not transfer to a trunk-trained arm.
  - **(c)** Park P6-HN behind P7.1 or the iPad-form targets. Both have a larger ESTIMATED reach (LAB-RUN-019 §4.3).
- **My recommendation is (b) then (a), and only if (b) shows ZEROS-FALL.** The decision is the PI's.

**Q2. PI: map-derived negatives (C9).**
- These are excluded until ruled: negatives built from lane boundaries, drivable polygons or lane-graph alternatives.
- The draft's negatives are trajectory-derived, and they are labelled with P6's existing training-time PDM labels. That is no new map use.
- If you rule map-derived negatives admissible as **training labels**, they become a separate arm, never folded into this one.

**Q3. Master Mind, design: negatives only, or both labels.**
- The draft commits to the published form: only offsets with PDMS = 0 enter the loss. This follows DA-WAM's dual constraint, verified in the PDF.
- The alternative feeds all 8 offsets, with both labels, as a within-scene contrast. That is the draft's next lever in §11 row 5/6, not this arm.
- Confirm before registration.

**Q4. Master Mind: the floor F.**
- F = 0.33 comes from P2: a warm-start fine-tune, about 21 A40-h, a single draw.
- P6 uses the same F with the same caveat, so the draft inherits it.
- If P6 gets a seed-1 replicate, the draft uses max(0.33, |Δ_seed(P6)|).

**Q5. Master Mind: G-LIVE-LAB on the migrated container.**
- A linear ESTIMATE at 4 label workers gives 10.887 × 4/7 = 6.22 sets/s. That is below the gate's 6.735 bar, though above the 4.49 actually needed.
- P6 will hit this first, under BACKLOG P6-3C's "MEASURE FIRST". P6-HN inherits P6's resolution.

**Q6. Master Mind: order after a P6 NOT PROVEN.**
- PREREG_P6's own next lever for NOT PROVEN is a scorer-only PDM fine-tune on P6.
- P6-HN would compete with it for the A40. The order is your call.
- Under Q1(b), the two could share the frozen-P6 rig as two separate arms, each with its own prereg.

## Design notes the register may want (none of these is a result)

- ⭐ **P6 already trains on fan-mined hard negatives.**
  - P6's scorer learns from online PDM labels of each sample's own 64 proposals (SPEC §3.1-3.2), so the census's fan supply (33.7 %) is
    already inside P6's loss.
  - The HN lever's new content is therefore geometry the fan does not produce: the offset bank. "Mine the fan" is a re-weighting, not a new lever.
  - This reframes what the census measured. It still bounds where the bank is needed most.
- **Both papers' numbers checked against the banked PDFs** (sha256 matches `library.json`):
  - `2608.19085` Tab. 3 matched pair: PDMS 93.46 → 93.68, NC 98.88 → 99.11, DAC 98.58 → 98.88, EP 90.47 → 89.97, TTC 96.33 → 96.81.
  - `2605.19771` Tab. 4: LTF\* 88.7 → 89.7 (DAC 97.0 → 97.9); DiffusionDrive 88.1 → 89.2 (DAC 96.2 → 97.4).
  - LAB-RUN-019's derived zero-proxy cuts are consistent with these rows. The census had marked them "not re-read against the PDFs".
  - This read moves the DA-WAM and BeyondDrive table values from SECONDARY to PRIMARY for this package.
- **Instrument work items the draft names** (not built):
  - the `--hn-bank` trainer path, plus its G-DVB row and the `refe_p6hn` profile;
  - a `refe_p6` refusal of `--hn-bank`;
  - the bank builder (`refe/hn_bank.py`, proposed name);
  - the two-arm joint-draw difference for ZT4's pairwise DAC accuracy, with its identity control (an arm minus itself = 0, CI [0, 0]).

## Nothing was changed in the repo

- No existing repo file was edited, so no base blob sha applies.
- No git commands were run beyond read-only `git show`, `ls-tree` and `grep` on the mirror.
- No pod, GPU or navtest table was touched.
