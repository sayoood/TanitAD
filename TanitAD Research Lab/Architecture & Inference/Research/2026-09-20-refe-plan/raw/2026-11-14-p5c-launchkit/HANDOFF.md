# HANDOFF: P5-C launch kit (land-ready), 2026-11-14. CPU only; no git writes, no pod, no GPU, nothing shipped.

Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/`. Tip at build: `99ffcd3a`. Read order: this file, `REBASE.md`, `PROFILE.md`, `RUNBOOK_P5C.md`, `TESTS.md`.

## State
- **Rebase done (0 conflicts).** 16 files in `files/<repo path>` (9 + 7 one-line pin-test edits: see `TESTS.md`; without those 7 the land goes RED); the 5 hold MOD files are merged onto the tip (only `gate_checks.py` and `test_launch_gate.py` had drifted; clean 3-way merges). `diffs/` = tip -> kit.
- **Open item done.** `traj_label_table.py` now has a REQUIRED `--rank-policy all|rank0` (no default) and a sha-bound `--meta-json`; the policy is declared in the table meta and ENFORCED by the one reader (a table without a policy, or whose rank-1 labels differ under `rank0`, is refused at the bank; the checkpoint meta carries `traj_label_rank_policy`). G-DVB reads the table's policy against a profile pin. The meta-json is refused when it is about another label, policy, row count or bank (sha of the bank files).
  - **Finding (MEASURED in `chk_rank_policy`, fixture 5/7):** the MM's worry (a rank0 table + `--require-coverage 1.0` refuses at ~62 %) applies only to a table that OMITS rank-1 rows (92,896 / 150,603 = 61.7 %). The card's own merge (`merged_labels_rank0only.jsonl`) writes rank-1 rows as raw identity labels, so coverage is 100 % and the build passes. A rank-0-only labels file IS refused (tested). So no card-level policy was chosen by me.
- **Profiles done:** `refe_p5c` (C1), `refe_p5c_c0` (C0), `refe_p5c_c0r` (C0r), pins unfilled -> they REFUSE. B-P6 builder exists, not registered. See `PROFILE.md`.
- **Runner patch:** `runner/p5c_pod_label.sh` (passes policy + meta-json, asserts policy in verify, new `PIN_TABLE_MOD`).
- 26 pre-existing profile fingerprints identical tip vs kit (`logs/digest_*.json`, MEASURED); 3 new profiles.

## MM decisions needed
1. **Amendment A2 (card-level).** (a) Runner blob `f9220d1f` and `traj_label_table.py` pin `9592b62c` (A1 Point 2 / POD_SHIP_LIST) move; ship the kit's two files and disclose pre-data. (b) PI 2026-11-12 "P5-R regardless of R4" contradicts A1 Section 0 ("applies ONLY if R4 fires; void if Tc20s passes"). The runner still builds the P5-R table when its gates pass; use `P5C_LABEL_TABLE_SHA256_p5r.txt`. Needs A2 text. Per brief: cards win; I changed nothing in the cards.
2. **C0r naming.** Brief: C0r = matched raw control (maybe identity table). Card: C0 = matched raw-label arm (table ABSENT), C0r = C0 with `--seed 1`, conditional, PI approval. Implemented the card. (Identity table == flag-off bit for bit is proven in `test_traj_label_table.py`, so an identity control is not needed.) The ~57-58 L40-h is the C1+C0 pair; C0r is +28-29 (card 14).
3. **G-RESERVE dropped** from the P5-C profiles (card 10.1 says "every other refe_p7 entry unchanged"): it requires `--aux-drv-labels`, which P5-C does not have. Bank is P7's, pinned by count + content.
4. **Card line 113 names `--scorer-mode`** "as P7"; P7's argv has none and `refe_p6` FORBIDS it. Followed P7's argv.
5. **`--resume`:** card 10.1 says fresh start; P7's ARGS carries `--resume`. Kept it optional in the argv; the gate refuses a `--out` holding `ckpt_last.pt` at MINT time (not at `verify`).
6. **Gate evidence** (G-LIVE-LAB / MEM / EGR files): reuse P7's or re-measure? MEM at least is head-dependent. No `p5c_pod.sh` chain was written; RUNBOOK section 8 is the manual form.
7. **Not built:** G-LIVE (f) warm-start identity, (h) lever isolation on the real model, post-training G-ADAM (card 10.4).
8. **Label pod cores:** `P5CL_NSHARD` must be <= the pod's ~15 CPUs, not 16/64 (RUNBOOK 4).
9. **Land order:** after P7's last arm trains; tip blobs must still match `REBASE.md`; fill the 3 pins (`P5C_BASE_SHA256`, `P5C_LABEL_TABLE_SHA256`, `P5C_RANK_POLICY`); rerun pin-sensitive tests; re-derive `SHIP_MANIFEST.json` and the ship list with `--profile refe_p5c`.

## Tests (details `TESTS.md`)
REGRESSIONS 0 after the 7 pin edits. Env-identical failures on tip: `test_launch_gate.py` full (dev data dir absent), `test_p7_ship_list.py`. **Not run:** ~32 other launch_gate-reading scripts, p6_chain, p6_levers, gate_suite_prerun: C: fell to 6 GB (other sessions) and my driver stopped at the 12 GB floor. MM: run them after the land.
Tip drift: tip moved to `e3fe96ac` during the work; none of the 16 files' blobs changed.

## Deletions (this session's scratch, literal paths)
`C:/Users/Admin/p5ck/` (963 MB: tipA 481, ovB 481, work/m, tmp1-3, res, scripts; logs and scripts copied to `logs/` first). Logged below after the delete.
