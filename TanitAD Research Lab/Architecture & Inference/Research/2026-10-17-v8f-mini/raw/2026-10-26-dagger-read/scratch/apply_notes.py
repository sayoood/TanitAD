import pathlib
P = pathlib.Path("C:/Users/Admin/dagread/NOTES.md")
s = P.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub("(`fixtures/dagger_smoke/`, copied byte-for-byte from `dg/.../2026-10-26-dagger-build/smoke/`)",
    "(`fixtures/dagger_smoke/`: byte copies of the generator's smoke; since the K-B4/K-B5 refresh (section 9) from `di/.../2026-10-26-dagger-build/smoke_kb45/`, before it from `dg/.../smoke/`)")
sub("`fixtures/dagger_smoke/` (3 real JSON documents), `PREREG_V8F_DAGGER.md` (the LANDED text), `out/` (evidence of THIS pass), `out_pre_a1/` (evidence of the pass before Amendment 1, kept).",
    "`fixtures/dagger_smoke/` (the generator's real smoke: 3 JSON documents + the shard npz, with `*.pre_kb45` backups of the two documents the K-B4/K-B5 build changed), `fixtures/trainer_kb45/WRITE_SITES.txt` (the trainer's write sites of `train_window_list_sha256`, verbatim), "
    "`PREREG_V8F_DAGGER.md` (the LANDED text), `out/` (evidence of THIS pass, the K-B4/K-B5 refresh), `out_pre_kb45/` (evidence of the A1 pass, kept), `out_pre_a1/` (evidence of the pass before Amendment 1, kept).")
sub("**Needs the bank generator to add** (the reader REFUSES, naming the field, until it is there):", "**[CLOSED 2026-10-26, section 9] Needs the bank generator to add** (the reader REFUSES, naming the field, until it is there):")
sub("**Needs the TRAINER to add:**", "**[CLOSED IN SOURCE 2026-10-26, section 9] Needs the TRAINER to add:**")
sub("### The reader's bank controls on the generator's REAL smoke (`out/smoke_audit.out`, `out/smoke_audit.json`; inputs unchanged: `out/smoke_inputs_sha256_before.txt` re-checked)",
    "### The reader's bank controls on the generator's REAL smoke, BEFORE the K-B4/K-B5 build (`out_pre_kb45/smoke_audit.out`, `out_pre_kb45/smoke_audit.json`; inputs unchanged: `out_pre_kb45/smoke_inputs_sha256_before.txt` re-checked; the refreshed table is in section 9)")
sub("| K-B4 / K-B5 | **MISSING** (\"needs the bank generator to add it\") |", "| K-B4 / K-B5 | **MISSING** (\"needs the bank generator to add it\") -- superseded: PASS on the refreshed smoke, section 9 |")
sub("* **E7 (open): the TRAINER must record", "* **E7 (CLOSED IN SOURCE 2026-10-26, section 9): the TRAINER must record")
sub("* **E8 (open): the bank generator must record", "* **E8 (CLOSED 2026-10-26, section 9): the bank generator must record")
i = s.index("## 8. Evidence of this pass")
head, sec8 = s[:i], s[i:]
sec8 = sec8.replace("## 8. Evidence of this pass (written by scratch/finish_a1.sh after the mutation run)", "## 8. Evidence of the A1 pass (previous pass; its files are in `out_pre_kb45/`; written by scratch/finish_a1.sh after the mutation run)")
sec8 = sec8.replace("`out/", "`out_pre_kb45/")
s = head + sec8

SEC9 = """
## 9. The K-B4 / K-B5 + trainer-window-sha refresh (2026-10-26, second pass; reader logic NOT changed)
**What was refreshed.** The bank generator now writes `PROVENANCE.json` `controls` `K-B4` (`PASS`, `cases` for the six names, `run_at`, `n_shards_checked`) and `K-B5` (`PASS`, `max_pose_now_err_m`, `max_rel_transform_err_m`, `tol_m`, `n_records`, `n_pairs`, `n_shards`, per-shard pose-log shas, ...), and the shard provenance carries its own `controls`. Source of the new smoke: `di/.../2026-10-26-dagger-build/smoke_kb45/` (copied OUT, read-only; the integration tree was never written, no script was run inside it). `out/smoke_kb45_inputs_sha256_before.txt` == `_after.txt` (the five files did not move while I read them).
| fixture (`fixtures/dagger_smoke/`) | sha256 old | sha256 new |
|---|---|---|
| `PROVENANCE.json` (+ `.pre_kb45` = the old bytes) | `2fe315f0...91b6` | `6c780d62...ad45` |
| `shard_000of001.provenance.json` (+ `.pre_kb45`) | `45264478...3b5b` | `d9915ee1...63c4` |
| `bank.json` | `c23c35b5...5d13` | **unchanged** (asserted by sha256) |
| `shard_000of001.npz` (new in fixtures; 1.75 MB) | `8250fe28...85e9` (in `dg/`) | **identical bytes** |

Old vs new, field by field (`test_fixtures_are_the_kb45_generator_output_and_the_pre_kb45_documents_are_kept`): every field the reader reads is equal; what moved is `controls` (added), `created_unix`, and the code / path / timing fields. The npz is now IN `fixtures/dagger_smoke/` so the real-smoke test no longer needs `dg/` for the bank (it still needs `dg/smoke_world/join_train.npz` for the smoke's synthetic TRAIN list and skips without it).
**Trainer.** `v8f_mini_train.py` (integration tree) records `train_window_list_sha256` at `:1398` (`info`, `DG.windows_sha256(J["clip"], wins)`) and `:1663` (`summary.json`, TOP LEVEL, directly under `train_sha256` at `:1662`); `:1542` puts the same `info` into `selfcheck.json` under `"info"`. **The reader's `--key-window-sha` default (`train_window_list_sha256`) is exactly that key at the place `find_key` looks (summary.json, bare): no default changed.** The synthetic trainer records already had the key at summary.json top level; they now also carry `selfcheck.json["info"]` (`train_sha256`, `train_window_list_sha256`, `n_clips`, `n_windows`; ORC-0, the older build, none of the window key). `fixtures/trainer_kb45/WRITE_SITES.txt` is a mechanical extract of those four lines (`scratch/extract_trainer_sites.py`; the source file's sha256 is in its header) and `test_the_trainer_records_train_window_list_sha256_where_the_readers_default_key_looks` pins default key == trainer's key == the synthetic summary's key. **The reader reads summary.json (the run's LAST write), not the early `selfcheck.info` copy:** `trainer_window_list_sha_not_recorded` (key removed from summary.json while the info copy stays) still REFUSES.
**Observed in the same trainer source, NOT run:** `levers["dagger_p"]`, `levers["dagger_bank_sha256"]` and the summary totals `dagger_n` / `dagger_eligible` are written under the names the reader's key defaults expect. No trainer RUN output exists, so K-B3.2 / TRAIN-LIST / BANK-DECLARED have still never passed on a real `summary.json`.
**The refreshed smoke audit** (`out/smoke_audit_kb45.out`, `.json`; same arguments as before: M2b sha `601cb435...`, the generator's eval grid sha `659c3b65...`, `dg/smoke_world/join_train.npz`):
| control | result |
|---|---|
| BANK-FILES, K-B0 (804/804), K-B1 (3 chunks, 96 windows), K-B2, K-B3.1, K-B3.3, BANK-ACCOUNTING | PASS (unchanged from the pre-refresh audit) |
| **K-B4** | **PASS** (six cases True; relayed, `verified_by_reader: false`) |
| **K-B5** | **PASS** (`max_pose_now_err_m` 0.0, `max_rel_transform_err_m` 1.5045e-05 <= 1e-4; relayed) |
| BANK-CONFIG | **FAIL, as it must**: the bank is PARTIAL (`partial True, limit_clips 12`) and `common.train n_clips 12 / n_windows 804` vs 4512 / 306566 (a 12-clip smoke is never a bank) |
| K-B3.2 / TRAIN-LIST / BANK-DECLARED | NOT RUN (need the trainer's run records) |

COVERAGE 0.7214 (plumbing only, synthetic features).
**Not enforced (choice left to the Master Mind, E10):** the new records carry fields the reader could cross-check and does not: `K-B5.tol_m` (equals the reader's 1e-4; pinned only against the REAL fixture by the key-structure test, not enforced on a bank), `K-B5.n_records` (equals the bank's `n_records` on the real doc), `K-B5.n_shards` / `K-B4.n_shards_checked` (should equal the bank's shard count), `K-B5.log_pose_sha256_by_shard`. Enforcing them is reader logic and needs mutants; this pass changed no reader logic (only the `--key-window-sha` help text, the module docstring and one comment, and `bank_audit.py`'s NOT RUN text).
"""
s = s.rstrip("\n") + "\n" + SEC9
P.write_text(s, encoding="utf-8", newline="\n")
print("notes ok")
