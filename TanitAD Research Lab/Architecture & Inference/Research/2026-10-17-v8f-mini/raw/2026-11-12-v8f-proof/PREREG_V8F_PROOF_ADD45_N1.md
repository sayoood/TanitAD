# Note N1 to card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD): the union lead file carries no closing speed

Note of the v8F proof study, written by the coordinator fork. Frozen before any tool of the card exists and before any of its runs (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The card still needs a go of the PI. Tier: T0; never driving performance.

## 1. What was learned after the card was frozen (INHERITED from the tree owner session, read from source at cdbff971, nothing executed; not re-verified here)

The lead file of the 10k clips and of the union stores, per frame, a lead flag, the gap, a time gap, the arc length of the lead and the EGO speed (`stack/scripts/build_v9_labels.py`, `lead_for_frame` lines 507 to 538, written by `cmd_lead_pass` lines 1111 to 1140). It stores no closing speed and no identity of the lead, so a closing speed cannot be derived from it. The card assumed a closing-speed field (controls d2 and d5, the read M1, the name `LEAD-READ-SCALES-CLOSING`, the control arms with permuted closing labels).

## 2. Changes (everything not named here stays as the card writes it)

* The probe of both arm types is trained on presence and gap only; the closing part of the loss has weight 0 in every arm, so the two arm types differ in their training clips alone.
* M1 is not read and `LEAD-READ-SCALES-CLOSING` cannot be given. The names on the gap (`LEAD-READ-SCALES-GAP`, `-WORSE-AT-SCALE`, `-FLAT-AT-SCALE`) are unchanged.
* Control arms: the gap labels are permuted in place of the closing labels (`P16CSH_s0`, `P16USH_s0`, full length). d5 becomes: both arms with permuted gap labels read a gap R2 whose interval includes 0 or lies below 0, while their presence AUROC stays at least 0.80 (presence labels are intact, so the arm is live).
* d2 becomes: for 200 corpus clips drawn at random the lead flag and the gap of the union lead file equal those of the corpus lead file at the same frame index (tolerance 1e-4), and the frame index of the lead file is on the clock of the cached tokens (the tree owner session's same-clock check of the lead file passes at shift 0 and fails at its planted shifts; its result path is cited).
* d3 becomes: the corpus arm is a live gap probe: at seeds 0 and 1 the interval of its gap R2 lies above 0 and its presence AUROC is at least 0.85. Beside, naming nothing: its place against the range of the eight `TK16` arms of row 97 (0.513 to 0.791), which were trained with the closing part in the loss.
* d4 on the control arms is read on the presence part.

## 3. What does not change

The evaluation set, the pairing, the bootstrap, the bars of the three names on the gap, the budget, the smoke before the run and what follows from each outcome. The lever set of the integrated candidate needs presence and gap only (the consistency term reads `lead_pres` and `lead_gapm`; the distance rule is carried in its form without the closing speed, card ADD33), so the missing closing speed blocks nothing there.
