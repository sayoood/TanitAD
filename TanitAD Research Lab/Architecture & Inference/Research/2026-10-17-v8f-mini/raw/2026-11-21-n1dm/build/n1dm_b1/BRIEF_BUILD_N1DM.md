# Brief: build and pre-register "N1DM", a mode-routed plan head on N1D's recipe (code patch + tests + gate + pod chain + read). Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing; the MM lands, on its own branch.
* No GPU. Do not edit protected files: launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py. Work in a scratch export.
* No secrets, and no IPs, ports or pod ids in any file.
* PI go, first-hand: "Build both". It covers building and pre-registering this ARCHITECTURE change; running needs a second go with the cost.

## Why
* N1DH read **N1DH-COST** (`F:/TanitAD-scratch/n1dc/read_n1dh/MM_VERDICT_N1DH.md`).
  * There is no D1-HO gain. The route head reads the decision at T0 (0.69 / 0.71), but the decoded plan does not follow it (0.62 / 0.64).
  * The card's NO-GAIN diagnosis: the loss is in the decoder's geometry. Its next lever: "a mode-routed plan head (K = 3 residual branches of `head_o`, selected by the posterior)".
* Read the N1DH card `F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DH.md` (f1df05bc) and Amendments 1 to 5. Its sections on `--aux-route`, the conditioning, Z0/Z1, REACH, the mutations, K-LABEL, K-LIM and G-DVB are your template.
* The N1DH build records are your pattern: `F:/TanitAD-scratch/n1dc/build/n1dh_b1/` (HANDOFF_B1.md, mkpatch_n1dh.py, n1dh.patch, run_mutants_n1dh.py, run_regress.sh, regress_cmp.py).
* Code of record: **1f217bf0**, branch `agent/v8f-n1dh-20261120` on the C: mirror `C:/Users/Admin/tanitad-push`. Export with `git -c core.autocrlf=false archive 1f217bf0 stack taniteval` into your own folder.

## The design (MM; flag in the card draft if the source makes any part impossible)
* **A new mode `--aux-route modes`** (the `off` / `ce` / `cond` cases stay bit-for-bit as they are).
  * It is `ce` (route_head CE, `--w-aux-route 0.5`, own-code rows only) plus **K = 3 residual plan branches**, and NO route_emb conditioning. One variable moves against N1DH's `ce` attribution.
  * Branch k is a zero-initialised `Linear(1024 -> D_o)`, where D_o = `head_o`'s output width. They are built LAST, under a forked RNG, after every existing module, in the N1DH pattern.
* **Plan:** `head_o(h) + sum_k p_k * branch_k(h)`.
  * `p = softmax(route_head(h).float(), -1)`, **detached**, so only the CE trains the route head.
  * The selection is soft: one deterministic form at training and at inference, with no argmax.
* **Step 0 is bit-identical to N1D** in forward, outputs and every loss term but the route CE (the branches are zero). This is the Z0/Z1 identity, with an exact-equality control.
* **Admissibility (CLAUDE.md, the goal-input and vision-only rules):** state what the posterior is computed from at inference (O's feature from the arm's own inputs), and why it is not the situation classifier's output and carries no label source.
* **Reachability (CLAUDE.md, built-tested-unreachable rule):**
  * non-zero gradient on every branch after one step, with an exact-0.0 control;
  * a mutation that deletes the branch add must go RED;
  * a mutation that permutes p across rows must change the plan;
  * declared_vs_built and `built_levers` entries;
  * the study checker kind (`strat1_check.py` kind `N1DM`, as N1DH added `N1DH`).
* **Arms:** N1DM_s0 / N1DM_s1, on N1DH's recipe and init (M2b_sK), 3,000 steps, with the argv = N1D's argv plus exactly `--aux-route modes --w-aux-route 0.5`. Deciding cell, guard, Z-POST and controls are as N1DH's. Comparator N1D; N1DH beside.

## Priority order (bank each part as it is done)
1. **`F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DM_DRAFT.md`:**
   * mechanism with `file:line`, and both outcomes committed per row with the next lever of each;
   * controls, a deliberate-regression arm (mutation), cost ESTIMATED from the N1DH chain's measured walls (`/root/strat1/n1dh/n1dh_chain*.log` and `runs/*/train/train_log.jsonl`, read-only on the pod);
   * the "4 metric families" read plan with STRATEGIC in full.
2. **The patch** in `F:/TanitAD-scratch/n1dc/build/n1dm_b1/`:
   * export, `mkpatch_n1dm.py` (anchored, counted edits), new tests `stack/tests/test_v8f_n1dm.py`;
   * every census test that greps `stack/scripts` or the v8f loaders updated and run whole (`test_v8f_n1dc.py`, `test_v8f_n1dh.py` and any other: grep for it). Lesson: a new script once turned an existing census test red;
   * the mutation table, and the regression of the touched test files, base 1f217bf0 vs new.
3. **The pod chain** (`n1dm_chain.sh`, its arm, gsuite and launcher, modelled on `/root/strat1/n1dh/n1dh_*.sh`, read-only pulls):
   * the full pod tree file list, INCLUDING the eval-141 grid (N1DH Amendment 3);
   * check the post-training lines once on a real finished checkpoint, because N1DH Amendment 4 was a missing space in such a line.
4. **The read:** copy `F:/TanitAD-scratch/n1dc/read_n1dh/` tools as `read_n1dm/` with the arm renamed EVERYWHERE (grep for N1DH and N1DC tags; N1DH Amendment 5 was a leftover tag), plus a dry check.

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp "$POD_TARGET:/path" local`)
Read and pull only. Never write there, run no python or pytest there, and never touch a process, lock or marker. Another session's jobs run on the pod.

## Dev box
* CPU only, one python process at a time, each gated on commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, with PYTHONPATH separated by `;`.
* TMP/TEMP go under your folder. Delete your own temp at the end, by literal path, and log it.
* F: must stay >= 6 GB. Never wait more than about 4 minutes in one call.
* Hand off (write `HANDOFF_N1DM.md` and end) if your context passes about 150k.

## Deliverables
* `n1dm_b1/`: patch, tests and logs, mutation table, regression, pod chain, HANDOFF_N1DM.md (at most 800 tokens), MANIFEST_SHA256.txt.
* The draft card.
* `read_n1dm/` with its dry check.

Hand back in at most 250 words, with a deliverable manifest, and say plainly which checks PASSED and which did not run.
