# DRY_MINT_PLAN: what the MM runs on the pod BEFORE the RHp3 launch

Why: Amendments 4 to 6 (three pre-train refusals) all came from paths only a pod run exercises. Everything here is CPU, small, and writes only under `/root/rollhist/dry_p03/` and the new `runs/RHp3_s*/argv.json`. **No GPU. The pod is running the N1DH chain (GPU lock held): none of these steps touches the lock, a marker, a process or the GPU.** Pod access: `source C:/Users/Admin/lorapod/podenv.sh`, then `pssh '<cmd>'` and `pscp`. Python on the pod: `/root/venv/bin/python`. All paths below are on the pod.

## 0. Install and verify the staged files (MM copies; nothing else moves)
```
pscp  rh_p03_chain.sh rhp3_arm.sh chk_arm_rhp3.sh rhp3_argv.py rh_p03_marker_watch.sh launch_p03.sh  "$POD_TARGET:/root/rollhist/"
pscp  dscale10k_check_rhp3.py  "$POD_TARGET:/root/rollhist/study/"
pssh 'cd /root/rollhist && sha256sum rh_p03_chain.sh rhp3_arm.sh chk_arm_rhp3.sh rhp3_argv.py rh_p03_marker_watch.sh launch_p03.sh study/dscale10k_check_rhp3.py'    # equal to MANIFEST_SHA256.txt
pssh 'cd /root/rollhist && for f in rh_p03_chain.sh rhp3_arm.sh chk_arm_rhp3.sh rh_p03_marker_watch.sh launch_p03.sh; do bash -n $f && echo "bash -n ok $f"; done'
pssh 'cd /root/rollhist && sha256sum study/dscale10k_check_rh.py'                                              # still 69ff1a0f...: the RH / B0 checker is NOT replaced
```
Expect: sha256 lines equal to the manifest, five `bash -n ok`, and `69ff1a0f962123fd...` for the old checker.

## 1. Render the two argv files (reads RH's argv and the bank; writes `runs/RHp3_s0|1/argv.json` only)
```
pssh 'cd /root/rollhist && source rh_env.sh && RH_READOUT=causal_g1x /root/venv/bin/python rhp3_argv.py render'
```
Expect: two `s<k>: ...argv_sha256=` lines and `ZZ RHP3-RENDER PASS`. It refuses unless RH's `--dagger-p` is 0.1, its bank sha256 equals `sha256(bank.json)`, and the new argv differs from RH's in exactly `--dagger-p` and `--out`.
Cross-check by eye (no python): `pssh 'cd /root/rollhist/runs && for s in 0 1; do diff <(tr -d " \n" < RH_s$s/argv.json | sed "s/,/\n/g") <(tr -d " \n" < RHp3_s$s/argv.json | sed "s/,/\n/g"); done'`. Expect exactly the lines `"0.1"` -> `"0.3"` and `.../runs/RH_s<k>/train"` -> `.../runs/RHp3_s<k>/train"`, for each seed, and nothing else.

## 2. The stock pair check on the real files (same call `rhp3_arm.sh` makes)
```
pssh 'cd /root/rollhist && source rh_env.sh && export_py && for s in 0 1; do sha=$(sha256sum bank/s$s/bank.json | cut -d" " -f1); /root/venv/bin/python $TREE/stack/scripts/rollhist_pair_check.py --rh runs/RHp3_s$s/argv.json --b0 runs/B0_s$s/argv.json --bank-sha256 $sha; done'
```
Expect `ZZ ROLLHIST-PAIR PASS` twice. (The stock check accepts any dose above 0: the dose is pinned by the study checker in step 3, not here.)

## 3. The arm's own mint, dry, on the real bank and data, then read `checks.json` IN FULL
`chk_arm_rhp3.sh` has a dry form: `DRY_GD` (a scratch gate dir whose path contains `/dry_`) and `DRY_SC` (RH's finished selfcheck of the same seed, because the p = 0.3 selfcheck needs the GPU). With either set, `run` is REFUSED and the token is written only into `DRY_GD`.
```
pssh 'mkdir -p /root/rollhist/dry_p03'
pssh 'cd /root/rollhist && RH_READOUT=causal_g1x DRY_GD=/root/rollhist/dry_p03/dry_gate_s0 DRY_SC=/root/rollhist/runs/RH_s0/selfcheck/selfcheck.json bash chk_arm_rhp3.sh mint RHp3_s0 > dry_p03/mint_s0.log 2>&1; echo rc=$?; tail -n 3 dry_p03/mint_s0.log'
pssh 'cd /root/rollhist && RH_READOUT=causal_g1x DRY_GD=/root/rollhist/dry_p03/dry_gate_s1 DRY_SC=/root/rollhist/runs/RH_s1/selfcheck/selfcheck.json bash chk_arm_rhp3.sh mint RHp3_s1 > dry_p03/mint_s1.log 2>&1; echo rc=$?; tail -n 3 dry_p03/mint_s1.log'
```
Expect `rc=0` and `TOKEN-WRITTEN RHp3_s0` (and `_s1`). Then read the record in full:
```
pssh '/root/venv/bin/python - <<EOF
import json
for s in (0, 1):
    d = json.load(open(f"/root/rollhist/dry_p03/dry_gate_s{s}/checks.json"))
    r = json.load(open(f"/root/rollhist/runs/RH_s{s}/gate/checks.json"))
    print(s, len(d), "checks;", "all pass:", all(v["pass"] for v in d.values()), "; same names as RH_s%d:" % s, sorted(d) == sorted(r))
    print({k: v["problems"] for k, v in d.items() if not v["pass"]})
    print("C-ARGV:", json.dumps(d["C-ARGV"])[:200]); print("C-RH-PAIR readout:", d["C-RH-PAIR"].get("readout_token"))
EOF'
```
Expect, per seed: 21 checks (RH_s0's `checks.json` has 21: C-ARGV, C-RH-PAIR, C-MERGE, C-MD5, C-STEPS, C-NCLIPS, C-CHUNK, C-TRAIN-SHA, C-DISJ, C-HOLD, C-MASK, C-JOIN-EQ, G-HYG, G-LIVE, C-BANK, G-DAGGER, C-RH-BANKGEN, G-CLOCK, G-EVAL, G-CKPT, G-SUITE), all pass, the same names as RH's, an empty problems dict, readout `causal_g1x`. Any other reading is a refusal to chase before the launch, not at 04:00.
**What this does not prove:** G-DAGGER at p = 0.3 reads RH's selfcheck (p = 0.1) here. The p = 0.3 selfcheck (`dagger_probe`, every p = 0 control exactly 0.0) is first read at the arm itself; if it refuses, the chain stops before any training step (about 2 min lost).

## 4. Leave the pod as found, log the deletion
The dry gate dirs and logs are own temporary data (re-creatable, small). After step 3 is read, by literal path:
```
pssh 'du -sk /root/rollhist/dry_p03; rm -rf /root/rollhist/dry_p03/dry_gate_s0 /root/rollhist/dry_p03/dry_gate_s1 /root/rollhist/dry_p03/mint_s0.log /root/rollhist/dry_p03/mint_s1.log; rmdir /root/rollhist/dry_p03'
```
Log the path and size in the handoff. `runs/RHp3_s*/argv.json` stay (the arm needs them).

## 5. Launch (MM, after Amendment 7 is frozen)
`bash /root/rollhist/launch_p03.sh` (marker watcher, then `RH_READOUT=causal_g1x rh_p03_chain.sh`, which waits on `/root/gpu_queue.lock` and holds it to its end). Watch `/root/rollhist/rh_p03_chain.log`; its last line is `ZZ ROLLHIST-P03 DONE ZZ` or `ZZ ROLLHIST-P03 FAIL ZZ <why>`. Then `read/rh_read_p03_chain.cmd` on the dev box (WMI launch).
