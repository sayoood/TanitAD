# PREREG_V8F_ROLLHIST, Amendment 6 (FROZEN 2026-11-21 04:56Z by the MM, before any RH or B0 training step or number)

Card `PREREG_V8F_ROLLHIST.md` (781eea87) and Amendments 1 to 5 all stand. Amendment 5 is 5ebf1f05. The commit of record stays **97fe0a56**, and no `stack/` file and no check changes. This amendment changes only the working directory of the study checker's call.

## What happened (MEASURED, pod)
* After Amendment 5 the chain relaunched at 04:52:34Z. The RH_s0 selfcheck passed again.
* The mint (`runs/RH_s0/gate/checks.json`) read **21 of 23 checks PASS**, including C-RH-BANKGEN, C-BANK, G-DAGGER, G-LIVE, G-SUITE and C-RH-PAIR. Two checks failed:
  * **C-MERGE**: "input tenk_manifest differs from the recorded sha256";
  * **C-MD5**: the argv's `--v9-train-md5` 9881b170... is not in the accepted set. Its union entry is null because C-MERGE failed first.
* **The cause is the working directory, not the data.**
  * `/root/dscale10k/merged/merge_record.json` (written 2026-11-16) records `tenk_manifest` as the RELATIVE path `in/manifest_10k.jsonl`. Every other input is absolute.
  * The DSCALE10K arms ran with cwd `/root/dscale10k`, where that path resolves.
  * The ROLLHIST mint does not, so the file is not found.
  * From `/root/dscale10k`, the file's sha256 is f2928333..., equal to the record. All the record's inputs then match.
  * `md5(v9_labels_union.npz)` = 9881b170..., equal to the argv's `--v9-train-md5`.

## The change
* `/root/rollhist/chk_arm_rh.sh` gains one line, `cd $DS` (`/root/dscale10k`), before the `mint` / `run` branch. The copy is kept as `chk_arm_rh.sh.bcwd`, and the new sha256 is b699452b.
* Every argument on that call is absolute, and so is every path in the arm argv (checked: no relative path argument). So the working directory changes only how the record's one relative path resolves.
* The patch tool is `fix_cwd.sh`. It refuses unless the anchor is unique and the diff is one line. It runs `bash -n`, then re-hashes every merge-record input from `/root/dscale10k`, and none differs.

## Relaunch
The same command (`relaunch_rh.sh`, causal_g1x). This is a first read under the failed-control rule: an environment repair before any value, and no check changes.

## Lesson (with Amendments 4 and 5)
Three pre-train refusals in a row came from paths that only the pod's real launch exercises:
* a repo-relative constant;
* a manifest key one level deep;
* a cwd-relative record path.

Before a pod launch, run the arm's own `chk_arm_rh.sh mint` by hand on the pod, on the real bank and data, and read `checks.json` in full.
