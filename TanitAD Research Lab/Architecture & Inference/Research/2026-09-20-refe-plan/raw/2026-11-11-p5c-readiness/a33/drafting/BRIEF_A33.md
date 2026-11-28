# BRIEF: Amendment A3.3: two PI rulings on C1 label run 3 (R-A tolerance 1e-8; the TB-5 row takes its Tc20s label). Agent: Sonnet. MM, 2026-11-16.

Read `Project Steering/AGENT_OPERATING_STANDARD.md` (repo `E:/Projects/TanitAD`) first.
- **Binding:** stage, never commit or push; end with a manifest; escalate integration.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a33/`.
- **Scratch:** `F:/tanitad_scratch/p5c_a33_tmp/`. TMP/TEMP/TMPDIR live there for every test, with `CUDA_VISIBLE_DEVICES=-1`.

## Facts (repo; read with `git -C C:/Users/Admin/tanitad-push show FETCH_HEAD:<path>` after `git fetch -q origin agent/arch-inf-20260803`; `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`)
- **Run 3** (`/workspace/data/p5c_label_a32` on the pod; findings in `PK/raw/2026-11-16-p5c-window/run3/RUN3_FINDINGS.md`): control PASS (A3.2); main FAULT on TB-5 and TB-0r.
- **TB-0r:** R-B passes in full on 16,133 rank-0 rows. R-A (the Tc20s warm start vs the dev box, tol 1e-9) misses at a max of 2.548e-9; 45 rows over 1e-9.
- **TB-5:** exactly 1 of 150,603 rows fails: rank 0, token `d06ff3cbd902…`, RJ5 traj sha `926ffc669830…`, `p5r_S2_yaw_branch` false. Its P5-R tail heading flipped 3.578 rad; its Tc20s label is sane (max |dyaw| 0.550).
- **A3.2:** `PK/raw/2026-11-11-p5c-readiness/PREREG_P5C_A3_2.md`; files in `.../a32/`. Gates `a32/a1/p5c_table_gates.py` (blob `1711a72f`); runner `a32/a1/p5c_pod_label.sh` (blob `8cfe7a72`); tests beside them.
- **The RJ5 path module** is `PK/raw/2026-11-11-p5c-readiness/a3/p5c_rj5_path.py` (blob `409f4481`), with its tests `a3/test_p5c_rj5_path.py`.
- **Exporting blobs:** use `git cat-file blob`. The mirror has `core.autocrlf=true`, so never go through a working tree.

## The PI rulings (2026-11-16, chosen from the MM's options; quote verbatim)
1. **R-A:** *"Bar 1e-8 (Recommended)"*, i.e. "Set R-A to 1e-8, about 4x the largest measured floor (2.55e-9), for TB-0 and TB-0r on all ranks. R-B stays the decision gate on the labels C1 trains on. The distributions are disclosed."
2. **TB-5 row:** *"Its Tc20s label (Recommended)"*, i.e. "Use the same row's Tc20s(RJ5) warm-start label, which passes the sanity check. It is recorded per row as a disclosed fallback; TB-5 then passes 150,603/150,603."

## Deliverable, in priority order
1. **R-A at 1e-8 for TB-0 (control pass, ALL ranks) and TB-0r.**
   - ⛔ TB-1 stays at 1e-9. If TB-0 and TB-1 share one CLI tolerance today, add a TB-0-specific tolerance so TB-1 is unchanged.
   - The control call uses `--tc20s-ranks all` (A3.3 supersedes A3.2's rank scope; the rank-1 disclosure stays in the detail).
   - `p5c_rj5_path.py tb0r` gets a tolerance argument, default = old behaviour; the runner passes 1e-8.
   - Every default must reproduce today's behaviour bit-for-bit.
2. **The TB-5 fallback, narrow by construction.**
   - Before the main gates, a row whose P5-R label fails TB-5's yaw-branch / S2 test takes its own Tc20s label, with a recorded marker.
   - The row then carries provenance `fallback_reason` (for example `tb5_s2_tc20s`) in the C1 provenance sidecar and is counted in the meta.
   - TB-5 must still check finiteness and |dyaw vs traj| < π on that row's NEW label, and must not count it as a P5-R row for the S1/S2/S5 flags.
   - ⛔ **Cap:** the rule applies to AT MOST the ruled row. Pin its identity (token `d06ff3cbd902…`, traj sha `926ffc669830…`; read the full values from run 3's outputs). Any other row failing S2, or more than 1 row, is still a FAULT. Any other row needs a new ruling.
3. **Reuse of run 3's shards.** Make the runner able to run from a COPY of run 3's work dir so only the post-shard stages re-run: rekey, the fallback, repair, gates, tb0r, provenance, table. Prove the reused shard outputs are byte-identical to run 3's (sha256 manifest).
   - If the runner already resumes over done shards, use that and document it.
   - ⛔ Do NOT launch the real run. The MM does that after the freeze.
   - You MAY copy run 3's work dir on the pod to `/workspace/data/p5c_label_a33/` (cp, then sha-verify).
4. **Tests.** Extend the A3.2 gate and runner suites and `a3/test_p5c_rj5_path.py`. Each mutant must go RED:
   - tolerance applied to TB-1;
   - tolerance ignored;
   - fallback applied to an unlisted row;
   - fallback applied to 2 rows;
   - fallback leaves the P5-R label in place;
   - fallback not recorded in provenance;
   - TB-5 skips the new label's dyaw check;
   - the tb0r tolerance ignored.

   All old suites must stay green; list any pin assertion you had to update.
5. **Real-data proof on the pod, in a NEW dir `/workspace/data/p5c_a33_check/`.** Run the new gates, tb0r and fallback against run 3's outputs.
   - Expected: TB-0 PASS at 1e-8 (all ranks, max 1.49e-9); TB-0r PASS (max 2.548e-9); TB-5 PASS 150,603/150,603 with 1 disclosed fallback; every other gate result identical to run 3.
   - Defaults (old tolerances, no fallback) must reproduce run 3's results.
   - ⛔ Write nothing into `/workspace/data/p5c_label*` (except your COPY to `p5c_label_a33`) or into `/workspace/data/p5c_ship`.
6. **`PREREG_P5C_A3_3_DRAFT.md`:** both quotes verbatim, the post-data disclosure, the changes, the pins (blob + sha256, 0 CR bytes), the tests, the proof, and the exact launch command for the reuse run. Also write `HANDOFF.md` (at most 2k tokens).

## Pod access (binding)
- **Commands:** `bash C:/Users/Admin/par18/pd.sh <t> '<cmd>' 2>/dev/null | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<IP>/g; /hf_/d'`.
- **Ship a file:** `pd.sh 60 'cat > /path' < file`, then verify the sha256.
- **Pod python:** `/workspace/venv-teacher/bin/python`, run with `nice -n 19`.
- ⛔ Never print `C:/Users/Admin/freept/pod_direct.txt`, IPs, ports or tokens.
- Use no GPU. Start no long jobs. Kill only your own processes, by PID.

## ⛔ Do not
- Commit or land anything.
- Edit frozen A3/A3.1/A3.2 files in place (new files go in this hold).
- Change any gate beyond the two rulings.
- Delete outside your own scratch (literal paths; log each deletion).
- Exceed about 150k context.

**Report:** at most 300 words, plus a manifest.
