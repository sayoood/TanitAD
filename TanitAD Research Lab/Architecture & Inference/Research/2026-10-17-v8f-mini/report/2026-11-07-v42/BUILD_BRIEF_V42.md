# Brief: build report v42 (T2 re-perception, the LoRA pod, and the theory of the crawl with the weight-space reading)

The PI, 2026-11-07, verbatim: *"dont idle continue autonmously incl. the weight space programming methods and the reasoning update our scientific paoer with al these inputs and related results"*

Read first and follow: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (the preamble).
* ⛔ Never commit, never push; the Master Mind lands. End with a deliverable manifest.
* ⛔ Never delete files: move them aside. LF line endings. Dev-box CPU only, `CUDA_VISIBLE_DEVICES=-1`.
* Read the repository only through `git -C C:/Users/Admin/tanitad-push show <tip>:<path>`, never from a working tree.
* Context budget ≤ 150k. The report is about 850 KB: read it by anchors (`grep -n`, then `sed -n` on a range), never whole. Hand off at about 130k with `HANDOFF_V42.md`.
* ⛔ Banned words anywhere you write: void, superseded, retracted, formerly, "no longer", correction, withdrawn, and "closed loop" / "closed-loop". The patch script scans for them; keep it so.
* ⛔ Never write the pod's IP address, port or pod id. Call it "the PI's GPU pod (RTX 4000 Ada, 20 GB)".

## Tip, base and machinery
* **Tip:** `git -C C:/Users/Admin/tanitad-push ls-remote origin refs/heads/agent/arch-inf-20260803`. Take the live value.
* **Base:** the landed v41, `R/report/2026-11-06-v41/` (`R = TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`).
* **Machinery:** copy `report_v41/` (`patch_v41.py`, `audit_v41.py`, `checks_v41.py`, `render_check_v41.js`, `build_v41.sh`, `HANDOFF_V41.md`, `BUILD_BRIEF_V41.md`) into `report_v42/` and adapt them. Adapt; never rewrite:
  * anchored edits;
  * `@@T` table directives, copied verbatim from the sources;
  * the italic quotes asserted verbatim in a source;
  * the numbers audit, which must end at 0 unmatched;
  * the checks, which must all PASS;
  * the offline render with 0 ragged tables.

## What v42 adds (each number with its artifact path and evidence class; each driving number with its tier; the four families wherever a read reports them, STRATEGIC PAUSED by PI item 25)
1. **§7.42, T2 re-perception on NuRec scenes.** It holds:
   * the tier: the camera rendered at the executed pose, agents replayed at log time, a playback executor; RR, the reindexed logged camera, beside it;
   * stages 1–4;
   * rule R-SCN (14 of 62 scenes refused before any planner ran; 48 read);
   * the crawl read (`PREREG_T2_CRAWL`, frozen `2c4c6557`: CRAWL-PERSISTS, branch a);
   * **the dominance read of record (`PREREG_T2_DOMINANCE`, frozen `38aa9df3`): NOT-DOMINANT at both seeds**, with its failing_terms verbatim, the per-family tables, and the two disclosed invocation errors;
   * the committed next (the perception lever at T2, `DRAFT_PREREG_T2_DOM_LORA.md`).
   * **Sources** (raw folders on the tip; find exact paths with `git ls-tree`):
     * `raw/2026-11-06-t2-crawl-read/` (commit f64a3367);
     * `raw/2026-11-06-t2-stage4-c0c/` (89cbd4da);
     * `raw/2026-11-06-t2-dom-repin/` (939176e7);
     * `raw/2026-11-06-t2-dominance-read/` (ec80b6c2), with `DOMINANCE_T2.md` / `.json` and the card with its log;
     * the T2 cards and the R-SCN amendment, wherever the tip holds them (grep `PREREG_T2_DOMINANCE`, `AMENDMENT_T2_SCENE_REFUSAL`).
2. **§7.43, the LoRA run moves to the PI's GPU pod.** It holds:
   * the venue, MEASURED;
   * the throughput proxy and every measured step time;
   * the frame route (±1 level on 1.377 % of values, Linux against Windows CPU `grid_sample`);
   * the token identity (the device alone 2.11–2.25e-2, the pod frames 2.25–2.63e-2, against τ 1e-3), and therefore the committed B′ route;
   * Amendment 4 (frozen `626aa1e5`) and its log (the gate route of `E7_DECISION.md`);
   * the state of the pipeline at build time.
   * **Sources:** `raw/2026-11-06-lora-pod-amend4/` (701e5a9e) and `raw/2026-11-06-t2-dominance-read/lora_pod/` (the plan and the proxy bench).
   * If `E:/Projects/TanitAD-artifacts/dagland/motion_c/lora_pod/HANDOFF_LORAPOD3.md` exists when you build, add its measured batch-64 timing.
   * ⛔ No LoRA driving result exists. Say so.
3. **§7.44, a theory of the crawl, and a weight-space reading of v8F.** Mathematics in the report's style, with the derivations written out:
   * **T1 / T1′:** the squared-error planner is a conditional mean; the mixture shrinkage â = π(z)·a_go + (1 − π(z))·a_stay; why T1 (a contraction of the speed level) was refuted and T1′ replaced it.
   * **T2:** the data-processing inequality bounds every lever downstream of the features, and the ROC frontier follows from it.
   * **T3:** the scope of LoRA on q / v with a fixed 4×10 pool.
   * **T4:** what a weight edit can and cannot do.
   * **The predictions and their reads:** P1 FAILS, P2 HOLDS (scale-sensitive; the weight-only claims matched 2 of 4), P3 FAILS as written (ceiling arms dominate; the admissible levers form one chain, post-hoc). P1′ is open.
   * **The weight-space-programming method as we applied it [P]:** Takehi, Teoh, Zhang and Khattab, 2026-11-02, a blog post, not peer-reviewed; Library key `takehi2026weightspace`. The protocol: claims from the tensors alone, hashed before any forward pass, then checked by Jacobian and ablation on held-out windows.
   * **What the weights say:** the kinematic base gains 1.00 / 1.90 shrink to 0.81 / 1.27 and 0.76 / 1.11; the −0.35 m/s shift; vision reaches the plan through S and T.
   * **The pre-registered edit WSP-GAIN** (frozen `92d61dcb`): ΔW = α · u · (Δg_a · w_a + Δg_v · w_v)ᵀ.
     * If `motion_c/theory/wsp_read/` or `motion_c/wsp_edit/` (or wherever `HANDOFF_WSPEDIT.md` points) holds its reads when you build, report them as written.
     * Otherwise write "pre-registered, not yet read".
   * **Sources:** `raw/2026-11-06-theory-crawl/` (b17b5492): `THEORY_CRAWL.md`, `THEORY_CRAWL_AMEND_1.md`, `PREREG_WSP_GAIN.md`, `wsp_read/*`.
4. **Abstract, status and the "what is next" paragraph:** updated to these sections. Plus a related-work line for weight-space learning and reading, citing the Library entry, and the reproducibility appendix rows for the new raw folders.

## Wording rules
* Every number carries its class: [M] / [P] / [E] / [H]. Every driving number carries its tier: T0 / T1-kin / T2.
* A refuted prediction is written as refuted, with its failure condition quoted from the frozen note.
* ⛔ No good-news softening, and no post-hoc finding promoted to a result.
* No DOMINANT exists; there is no scale signal.

## Output
`report_v42/out/V8F_SCIENTIFIC_REPORT.md` and `v8f_report.html`, plus the scripts, `edits_v42.md`, the new section files, `MANIFEST.txt`, and `HANDOFF_V42.md`. The handoff gives the build command, the check counts, and the exact folder to land as `R/report/2026-11-07-v42/`.
