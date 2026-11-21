# Search log — 2026-08-28 Data Engineering pass

Method: WebSearch + WebFetch + full-text extraction (PyMuPDF) from banked PDFs.
All numbers quoted in RESULT.md were read from the banked primaries, not from
search snippets.

## Queries run
1. `autonomous driving auto-labeling lane centerline reference ego lateral offset "lane keeping" label pipeline HD map`
   → MATLAB LKA docs, LaneSegNet, CenterLineDet, AutoGraph, lane-graph estimation.
   No published *label calibration protocol* for deliberate offset. (empty for the
   specific question)
2. `weakly supervised temporal action localization video-level labels multiple instance learning THUMOS14 fraction of fully supervised performance`
   → P-MIL 2305.17861 (banked; Tab. 1 extracted).
3. `HowTo100M narrations misaligned "MIL-NCE" temporal alignment network percentage alignable narration timestamps`
   → MIL-NCE 1912.06430, TAN 2204.02968 (both banked; 30 %/15 % + ~50 % numbers
   extracted from PDFs).
4. `CoVLA dataset arxiv auto-labeling rule-based captions trajectory "vision-language-action" Turing lane position`
   → CoVLA 2408.10845 (banked; §3.1.4 rule-based captioning extracted; 2 Hz frames).
5. `CLRerNet arxiv id lane detection LaneIoU CULane F1 state of the art`
   → 2305.08366 (banked; F1 81.43 paper / 81.11–81.55 released).

## Local primaries read
- lib `2511.00088` Alpamayo-R1: pp. 5 (dataset criticism: causal confusion from
  exposing full clip + future events; vague free-form traces), 11 (Tab. 1 closed-set
  decisions incl. In-lane nudge / Out-of-lane nudge), 12 (Tab. 2 critical components
  incl. Lane/lanelines; §4.1 anchoring: "first action taken by the ego vehicle
  immediately after the critical reasoning moment"; §4.2 clip selection).
- Verbatim extraction transcripts: this session's tool log; page numbers cited in
  RESULT.md are from the banked PDFs.

## Register context read
- GOALS_AND_CLAIMS.md D-DATA-GTAC (1563), D-DATA-GTAC-b (1564), D-LABEL-GT (1565),
  D-DATA-ALPA-LAT, D-DATA-COT-HALLUC.

## Papers banked this pass (Data Engineering)
| key | why |
|---|---|
| 2204.02968 | TAN — 30 % alignable / 15 % well-aligned |
| 1912.06430 | MIL-NCE — ~50 % misaligned, MIL mechanism |
| 2305.17861 | P-MIL — weak-vs-full THUMOS14 gap |
| 2408.10845 | CoVLA — rule-based caption pipeline scoped to its references |
| 1807.11546 | BDD-X — timed-annotation ceiling (IoU 0.63) |
| 2305.08366 | CLRerNet — lane-boundary reference instrument |
(2511.00088 Alpamayo-R1 and 2406.15349 NAVSIM were already banked.)
