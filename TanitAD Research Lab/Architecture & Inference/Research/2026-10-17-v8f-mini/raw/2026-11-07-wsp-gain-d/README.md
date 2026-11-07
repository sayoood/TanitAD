# WSP-GAIN-D: the weight edit aimed at the data's own longitudinal gains (2026-11-07)

* **Card:** `PREREG_WSP_GAIN_D.md`, banked frozen in `../2026-11-07-p5-dilution/` (sha256 of the first 5,211 bytes `d6e691a3065fbd7c...`). `PREREG_WSP_GAIN_D_with_log.md` here is the same card with its dated section-6 entries (PIN entries, an erratum and its fix, and the result entry). Its first 5,211 bytes hash to the same value.
* **Verdict, by the card's section 4: HARMS.** `RESULT_WSPD.md` / `.json`, written by `result_d.py`.
  * READ 1 (T0-class, 20,000 TRAIN fit windows): MECH-OK. The edit moves dS/dv0 and dS/da0 to within 10 % of g_data (0.9303, 1.3637) at both seeds.
  * READ 2 (T1-kin, eval-141, paired episode-cluster bootstrap B 2,000, seed 0): M1 D - C0 is -0.0175 [-0.0241, -0.0117] at s0 and -0.0172 [-0.0242, -0.0104] at s1, so share_closed is 0.329 / 0.289. Nine new separated-worse rows appear at each seed.
* **Committed next step:** the data-matched point joins P3's chain beside WSP-GAIN's. No T2 stage runs, and the perception lever (LoRA) stays the route.
* **What is here:**
  * code: `wsp_common.py`, `wsp_measure.py`, `wsp_edit.py`, `fit_gdata.py`, `fit_and_write_d.py`, `read1_d.py`, `wspd_read.py`, `result_d.py`, `dk_beside.sh`, `chain_d.sh`;
  * `tests/`, and `run_mutations.py` with `logs/mutants/`: 14 of 14 mutations RED;
  * `raw/`: fits, READ 1 / READ 2 JSON, per-window arrays, and distance keeping beside (`DK141_TABLE_D.md`);
  * `logs/`.
* **Not here:**
  * The edited checkpoints are at `E:/Projects/TanitAD-artifacts/v8f-wspedit-d/M2b_s{0,1}_wspedit_d/ckpt_last.pt` (sha256 `30bdd83c...` / `8d89c610...`, in `raw/PIN_HASHES_D.txt`).
  * The rollouts are in the agent's scratch on F:.
  * Six zero-byte `*.finished` markers stay on E:. The lander refuses empty files.
