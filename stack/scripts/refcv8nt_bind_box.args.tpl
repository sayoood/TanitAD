# G-BOX-OVERFIT binding harness argv for the NT launch (profile refcv8-nt), OPTION B 2026-10-22. refcv8_bind_box.args.tpl
# + `--frameset-profile refcv8-nt` (g_box_overfit.FRAMESETS: the NT frame set gbo_frameset_nt.json, its md5 and its C3
# literals; the prereg, the sec. 5 bars, the arms, the optimiser and the schedule are refcv8's) + the NT data roots the
# harness READS (the NT labels and the token-episode cache), so the closure records and the gate re-hashes them.
# The audit dir is prepared by nt_prelaunch.sh bind: the prereg from stack/ops/refcv8_box_audit/raw, the NT addendum
# and the NT frame set from stack/ops/refcv8nt_box_audit/raw, each md5-checked.
stack/scripts/g_box_overfit.py
--launch-argv
{ARGV}
--audit-dir
{OUT}/audit
--out
{OUT}/gbo_binding.json
--device
cuda
--binding
--commit
{COMMIT}
--candidate
BINDING: launch commit {COMMIT} (refcv8-nt; A19 MAIN-only; NT frame set)
--arms
main
--frameset-profile
refcv8-nt
DATA-ROOT /home/nvidia/data
DATA-ROOT /home/nvidia/refcv8_v9labels
DATA-ROOT /home/nvidia/r8ns/nt/labels/a1
DATA-ROOT /home/nvidia/r8ns/nt/v2ep_a1
DATA-ROOT {OUT}/audit
