# HANDOFF (STRATFAM, 2026-11-17)
State: instrument, tests (9/9, scipy blocked) and first read P7 vs P6 done; P7 adherence 0.8805201177625123 reproduced exactly. Nothing committed or pushed.
Files (hold = E:/Projects/TanitAD-artifacts/scorer_gym_20261107/strat_family/): land/eval/instruments/{strategic_family.py,test_strategic_family.py}; strat_family_P6_P7.json; STRATFAM_P6_P7.md; INTEGRATION.md.
Run: set REFE_EVAL_DIR=F:/tanitad_scratch/stratfam/pk/eval (copies of the pinned command_adherence, proxy_eval, eval_checkpoint, refe_navtest_seam from tip c937b4c6; on the repo they sit beside the instrument so it is unneeded), TANITAD_REPO=E:/Projects/TanitAD, then C:/Users/Admin/venvs/tanitad/Scripts/python.exe land/eval/instruments/test_strategic_family.py.
Next step (MM): land the two new files via the lander (new files: base NEW; repo path <= 200; run test_text_encoding_is_explicit first), then run `read` on the P5-C C1/C0 seams.
Open: floor 0.8841 quoted from the banked JSON (INHERITED), not recomputed. Scratch F:/tanitad_scratch/stratfam/ (pk/, tmp/, summ.py; about 1 MB) is kept until landing, then delete by literal path; nothing deleted so far.
