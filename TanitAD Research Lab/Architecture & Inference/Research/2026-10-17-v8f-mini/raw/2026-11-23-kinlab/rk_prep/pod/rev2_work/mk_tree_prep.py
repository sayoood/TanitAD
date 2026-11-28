import sys
s = open("rk_tree_prep.sh.rev1", encoding="utf-8", newline="").read()
def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
sub("# rk_tree_prep.sh -- CPU, seconds, no GPU / lock / marker. KINLAB (card PREREG_V8F_KINLAB, commit 0ca6f49d).",
    "# rk_tree_prep.sh -- REVISION 2. CPU, seconds, no GPU / lock / marker. KINLAB (card PREREG_V8F_KINLAB + Amendment 1, commit 1a8b11a1).")
sub("+ the six blobs the KINLAB commit adds or changes (rk_delta.tar, git archive of 0ca6f49d: v8f_recovery.py, v8f_dagger.py,\n# v8f_mini_train.py modified; test_v8f_kinlab.py, golden_v8f_dagger_df250a00.py, golden_v8f_recovery_df250a00.py added)",
    "+ the eight blobs the commit differs by (rk_delta.tar, git archive of 1a8b11a1: v8f_recovery.py, v8f_dagger.py,\n# v8f_mini_train.py modified; test_v8f_kinlab.py, test_v8f_kincache.py, golden_v8f_dagger_df250a00.py, golden_v8f_dagger_0ca6f49d.py, golden_v8f_recovery_df250a00.py added)")
sub("(git ls-tree -r 0ca6f49d stack taniteval, in the pod form, PLUS the eval-grid row, as all_blobs_ra.txt)", "(git ls-tree -r 1a8b11a1 stack taniteval, in the pod form, PLUS the eval-grid row, as all_blobs_ra.txt: 3,420 rows)")
sub("The kit files (rk_env.sh, all_blobs_rk.txt, rk_delta.tar, blobcheck.py) are read from this script's own folder; when shipped, that folder is /root/rollhist.", "The kit files (rk_env.sh, all_blobs_rk.txt, rk_delta.tar) are read from this script's own folder (/root/rollhist/kit_rk); blobcheck.py and the RA files from $R.")
sub('"fb1a1ff1674ded47"', '"2a64eb92e4fef2bf"')
sub('"bba9d744362f4d9c"', '"c505d59fddf08dc5"')
sub("$PY $HERE/blobcheck.py", "$PY $R/blobcheck.py", 2)
open("rk_tree_prep.sh", "w", encoding="utf-8", newline="").write(s)
