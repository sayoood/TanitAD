from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_verdict.py"); s = p.read_text(encoding="utf-8")
a = """         "reported apart (floor beside).", "", f"**committed outcome (prereg s5):** {OUTCOME[word]}", ""]"""
b = """         "reported apart (floor beside).", "",
         f"**amendment:** PREREG_REFCV9_INMODEL_PA_AMENDMENT1 (sha256 {RC.AMEND1_SHA}) -- controls only; every bar, arm, population and estimator "
         "unchanged. Prereg sha256 " + RC.PREREG_SHA + ".", "",
         "**Disclosed (AMENDMENT1 change 5):**", "",
         "* **(a) tree drift (NOT an R9 effect; owned by H-R9-TREEDRIFT, not investigated here).** The tip tree's planner without the R9 head selects "
         "a different path from the fb47a58-rolled bank on 70 / 70 calibration windows (max 7.24 m; z_tac bit-equal; deterministic run-to-run). "
         "So the bars are composed OFFLINE on the BANKED fb47a58 paths (the same path as H2CN_DEPLOY8 / RETIME_DEPLOY8), and the in-forward roll "
         "serves K-OFF / K-EQ only.",
         "* **(b) a DEFECT in the banked R9 code, fixed by one line.** `enable_refcv9` called `torch.manual_seed` inside `fork_rng(devices=[])`, "
         "which re-seeds the CUDA RNG and does not restore it: merely building the head moved the selected path on 70 / 70 calibration windows "
         "(max 8.10 m). Fix: `torch.default_generator.manual_seed` (CPU generator only), pinned in FIX_R9_RNG.sha256; flags-ON == flags-OFF "
         "bit for bit after it (K-OFF below).",
         "* **CONSTZ fit-side flag.** On the stage-2 fit (calibration split, out-of-fold CV objective), R9PF-CONSTZ -- z_tac replaced by its "
         "training mean, i.e. NO vision -- scored BETTER than R9PF: 1.83-1.85 vs 1.95-1.97 over fit seeds 0/1/2 (R9PF_FIT_INFO.json). "
         "K-CONSTZ and the INERT branch decide it on eval, as sealed.", "",
         f"**committed outcome (prereg s5):** {OUTCOME[word]}", ""]"""
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8"); print("verdict patched")
