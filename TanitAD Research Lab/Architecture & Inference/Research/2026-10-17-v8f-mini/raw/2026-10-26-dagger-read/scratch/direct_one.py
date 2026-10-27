import sys
sys.argv = ["x"]
exec(open("C:/Users/Admin/dagread/scratch/direct_run.py", encoding="utf-8").read().split("ok = True")[0])
ok = True
for nm in ("test_the_real_generator_smoke_reads_exactly_as_NOTES_md_reports", "test_bank_audit_on_the_real_kb45_smoke_passes_k_b4_and_k_b5_and_fails_only_the_partial_bank"):
    ok &= call(getattr(T, nm))
print("ALL-PASS" if ok else "SOME-FAILED")
