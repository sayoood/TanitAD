# HANDOFF A3-IMPL (2026-11-08) -- selector DONE, bundle_a3 chain RUNNING (detached)

**Done:** `select_suite_tokens.py` (A3 build), `a3_slots.py`, outputs `SUITE_CANDIDATES_A3.json`, `SUITE_PIN_A3.json`, `MAP_FIELDS_A3.json`, `HAND_VERIFY_A3.txt`, legacy regen `SUITE_CANDIDATES.json` / `_ALT.json` (sha256 identical to `../SUITE_CANDIDATES*.json`: 3d669b74... / 5633414d...).
**Running:** `chain_a3.cmd` (launched via WMI, PID 35872, `< NUL`): prepare -> [a0 forward on CPU, ~25 s/token, 27 tokens] + extract (32 tgz, ~30 min) -> sky -> wait for `F:/TanitAD_tsf_bundle2_20261107/a0_forward_a3/DONE` -> assemble.
Log: `F:/TanitAD_tsf_bundle2_20261107/work_build_a3_chain.log` (ends with CHAIN_DONE); a0: `.../a0_forward_a3.out`; per-archive status: `.../work_a3/status_*.json`.
**Next step (fresh agent, only after CHAIN_DONE):**
1. `grep -c . F:/TanitAD_tsf_bundle2_20261107/work_a3/extract_report.json`; check MANIFEST: `python -c "import json;m=json.load(open('F:/TanitAD_tsf_bundle2_20261107/bundle_a3/MANIFEST.json'));print(m['n_tokens'],m['n_images_found'],m['n_images_wanted'],len(m['missing_images']),m['tm2s_frames']['n_images_found'],m['a0_model_inputs']['status' if 'status' in m['a0_model_inputs'] else 'mode'])"` -> expect 27 tokens, 216/216 t0 and 216/216 tm2s images, a0 mode forward-cpu, 27 lines.
2. If the chain died: re-run `chain_a3.cmd` the same way (every stage resumes: extract skips found files; a0 has `--resume` -- add it to the a0 line first).
3. If a0 is too slow, the GPU command (needs the dev-box GPU lock, not taken here): `python -I code/refe_a0_inputs_b2.py --tokens F:/TanitAD_tsf_bundle2_20261107/work_a3/wanted.json --out F:/TanitAD_tsf_bundle2_20261107/a0_forward_a3 --device cuda --forward`.
4. Delete `F:/TanitAD_tsf_bundle2_20261107/work_a3/` only after the bundle MANIFEST sha-verifies (rule 2: re-creatable from the tgz).
**Not in bundle_a3:** fill ranks 6-10 of the new slots (only candidates rank<=5 + the 3 AMENDMENT_5 tokens). Add by editing `candidates()` in `build_bundle_a3.py` and re-running `chain_a3.cmd` (resumable).
