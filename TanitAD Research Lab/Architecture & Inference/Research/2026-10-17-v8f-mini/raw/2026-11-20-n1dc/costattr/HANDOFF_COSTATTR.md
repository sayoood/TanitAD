# HANDOFF_COSTATTR (2026-11-20)
Card PREREG_V8F_N1DC_COSTATTR sha256 295db8eb (checked). Nothing staged in the repo, nothing committed or pushed.
Done: pulled U ckpts (sha256 equal the pod's: s0 11cfe22a, s1 4074f70f); K-COPY pass (278 model tensors: o_code.weight all zero, 277 equal N1DC); two U eval-141 hb0.2 dumps (N1DC-N1DC_s{0,1}U, rc 0) in dumps141_hb02/; read run.
RESULT: status INVALID, no label printed. K-SAME/K-FL-HB, K1, K-COPY, K-GUARD pass. **K-ADD FAILS** (max abs 0.0492 s0, 0.0091 s1, must be <= 1e-12).
Cause (kadd_diag.json): the card's "window means are linear" holds only on windows finite in all three arms. cell_delta masks each pair to windows finite in BOTH arms; yaw-rate / curvature / dk_hw / dk_tg cells have NaN windows that differ per arm, so P1+P2 != N1DC-N1D on 7 cells at s0 and 6 at s1 (incl. cost cell s1 dk_hw_abs_err_m). On the common-finite mask the identity reads exactly 0.0 on every cell. Bookkeeping flaw of the card wording, not a data fault.
Next step (MM decision, not mine): amend the card (new sha) so P1, P2 and K-ADD use the windows finite in N1DC, N1D and U together, then rerun costattr_read.py (about 2 min, no new dumps). The labels were NOT computed under that reading.
Deleted: tmp/, both pulled ckpts (see F:/TanitAD-scratch/n1dc/DELETIONS.log). Pod source untouched. Keep the U dumps (dumps141_hb02/dump_*U*, ticks/); MM banks them.
