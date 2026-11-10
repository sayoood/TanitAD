# stopbal suite handoff (agent ended per token rule 6a)
State (patched 2bca1c52 vs parent 5514a684):
- Full-tree export impossible: tree is 13.5 GB per side (TanitAD/ 10.6 GB, FlyWheels/ 2.3 GB) and F: extraction ran ~100 files/s then crawled; aborted. Restricted exports (stack, taniteval, tools, Project Steering; 3,931 files):
  - F:/TanitAD-scratch/stopbal_suite/patched2  DONE (rc=0)
  - F:/TanitAD-scratch/stopbal_suite/parent2   export still running in background (check export2.log for "parent2 rc=0")
- Selection: sel_u2.txt (119 test files = launch_gate readers (72) U test_v8f_dagger_train U all test_v8f_* (50) U test_text_encoding_is_explicit). TanitAD Research Lab/ NOT exported -> tests reading it will error identically on both sides (environmental).
- patched2 pytest running detached: run_side.sh patched2 -> pytest_patched2.log (ends with "DONE rc=N"). Import check import_patched2.txt: tanitad from patched2\stack OK.
- NEXT: when export2.log has "parent2 rc=0" and patched run is DONE, run `run_side.sh parent2` (sequential, not parallel; script drops test_v8f_stopbal on parent since file is new). Then diff failure ids (-rfEs lines) between pytest_patched2.log and pytest_parent2.log -> NEW-FAIL / PRE-EXISTING / FIXED.
- Run is slow: ~46 tests in first hour (F: busy, other sessions at 100% disk). Log is dots only until finish; failures list at the end.
- CLEANUP at end (literal paths): F:/TanitAD-scratch/stopbal_suite/{patched,patched2,parent,parent2,pt_x} (patched = partial aborted junk, ~31k files).
