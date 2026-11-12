# LIT_A: formal verification, weight-space programming, NN verification (v8 literature pass)

Evidence classes: PRIMARY = I fetched the page/abstract. SECONDARY = search snippet or a third-party write-up only. Nothing was downloaded; no paper PDF was read in full. Two fetches (lean-lang.org/install, leanprover-community linux page) returned no commands, so the Lean install path below is INHERITED/unverified from general knowledge and must be checked on the box.

## Q1. Formal verification with AI assistance

**1.1 Anthropic, "LLMs are making formal software verification feasible at a much larger scale" (anthropic.com/research/formal-verification, dated 22 Oct 2026). PRIMARY (web page; no arXiv id; appendix is a PDF I did not open).**
- Claude proved no-crash properties (no memory-safety error, no failed assertion, any input under 64 MB) for 239 fuzz entry points across 123 OSS-Fuzz libraries. Proofs are at the compiled-binary level (single-threaded static RISC-V, musl, Linux process model), not source.
- Toolchain: Rocq proof checker; Theorem Labs supplied RISC-V and Linux process semantics; Verus used for a Rust Python interpreter; Claude-written x86 semantics for a zlib study (RFC 1950 spec). About 180 M lines of Rocq in total, typical proof about 600 k lines, median binary about 18 k instructions.
- Workflow: take an existing fuzz test, compile, state "never crashes for inputs under 64 MB" as the spec, agent writes proof, Rocq checks it. On a bug: patch code, regenerate spec, re-prove. Patches were penalised and logged to deter reward hacking (e.g. rewriting the program to trivially not crash). Hundreds of bugs found. The Rust Python interpreter matched CPython on 99.95 % of 1.7 M random programs.
- Efficiency: on 36 small tasks Opus 5.5 proved half using about 40 % fewer tokens than Opus 5; agents sharing a workspace used 17 % fewer tokens than solo; full proofs can still cost hundreds of millions of tokens; proof reuse across shared runtime code is large.
- Limits they state: verified is not bug-free (spec and machine model are trusted); musl/static/RISC-V model differs from glibc/x86/ARM in deployment; sanitiser coverage gaps; caller misuse not specified.
- Released: github theorem-labs/oss-verify-proofs, theorem-labs/rocq-linux-process-semantics, a Claude skill `add-project`.
- Use for v8: copy the pattern, not the toolchain: (a) spec written first and frozen, (b) the Lean kernel as the only judge, (c) a log that penalises any edit to the spec or the code under test, (d) mutation of the spec must turn the proof red. Apply it to the small closed-form statements in our code (e.g. the mask/bar identity in the reachability rule), not to the trained network.

**1.2 Zhang, "Towards Formalizing Reinforcement Learning Theory: A Robbins-Siegmund Approach", arXiv 2511.03618 (v1 5 Nov 2025, v2 20 Aug 2026). PRIMARY (abstract page).**
- Lean 4 + Mathlib proof of almost-sure convergence of Q-learning and linear TD with Markovian samples via Robbins-Siegmund. Code: github.com/ShangtongZhang/rl-theory-in-lean. A search summary (SECONDARY) says about 10,000 lines of Lean and that Mathlib's conditional expectation was the main obstacle; I could not confirm the line count on the abstract page.
- Use for v8: the closest existing Lean RL base; use it as the dependency for any stochastic-approximation or TD statement.

**1.3 Performance-difference lemma / simulation lemma in Lean.** One search ("Lean 4 Mathlib performance difference lemma simulation lemma") found NO Lean formalisation of either. That does not establish that none exists; a GitHub code search for "performance_difference" in Lean repos is the cheap second probe. Both are finite-sum identities over a discounted state-visitation measure; on `Fintype` MDPs they are short to prove and a realistic first agent task (identity-level, no measure theory).

**1.4 Provers an agent can call (SECONDARY, snippets).**
- DeepSeek-Prover-V2, arXiv 2504.21801 (Apr 2025): 88.9 % miniF2F-test pass for the 671B model, 49/658 PutnamBench; 7B and 671B weights on Hugging Face; subgoal decomposition from DeepSeek-V3 sketches. Use for v8: only the 7B is realistic on one GPU; prefer a frontier coding agent plus Lean's own tactics (`aesop`, `simp`, `linarith`, `positivity`, `exact?`) over a local prover for our statements.
- Kimina-Prover (Preview): 80.7 % miniF2F at pass@8192 per an aggregator; arXiv id NOT found in my results; weight release not confirmed.
- LeanDojo, AlphaProof: not searched; no claim made.

**1.5 Lean 4 + Mathlib install, cached build, disk size.** Commands are INHERITED from general knowledge (to verify); sizes are from Lean Zulip snippets (SECONDARY).
- Commands: `curl https://elan.lean-lang.org/elan-init.sh -sSf | sh`; `lake new v8proof math` (Mathlib template) or add `require mathlib from git "https://github.com/leanprover-community/mathlib4"` to the lakefile and copy its `lean-toolchain`; then `lake exe cache get` (downloads prebuilt .olean files; never run a full `lake build` of Mathlib).
- Size: Zulip reports Mathlib plus deps about 3.6 to 3.9 GB, a full cache download of 5.1 GB, a partial per-module fetch down to 1.5 GB, and one VS Code starter install at 7 GB. Plan 6 to 8 GB including the toolchain and check `du -sh .lake` after the cache step. Keep ONE Mathlib project and put our files under it; each extra project repeats the cost. Put it on F: (1 KB clusters), not on D: (exFAT, 1 MiB clusters, loose files).

## Q2. "Weight space programming" and its relatives

**2.1 What the phrase refers to (2026).** "Sparks of Weight-Space Programming", Takehi, Teoh, Zhang, Khattab (MIT CSAIL), blog post dated 2 Nov 2026, blog rikiyat.github.io/blog/weight-space-programming/, logs github RikiyaT/lm_understand-logs. No arXiv id; not peer reviewed. I fetched a DataCamp write-up (SECONDARY for the original). Definition: a general-purpose LLM reads a network's raw weights and writes or edits them from a plain-language goal, with no gradient descent.
- One LLM only (GPT-6 Astra), networks up to 0.8 B weights. Reported: identified which game each of 4 PPO policies was trained on (4/4); found every induction head in the tested transformers; editing 3.5 % of a failing Kinetix car-and-ramp policy lifted success from 0 % to 100 % over 1,024 episodes; wrote MinAtar Breakout networks scoring 517.4 vs PPO 15.8 (100 M steps) and 25.3 (500 M), while PPO at 500 M still beat it on Seaquest and Asterix; an edit of 131,072 weights (0.016 %) in MLP layers 15 and 19 of Qwen3.5-0.8B moved MATH-500 from 7.9 % to 48.1 % and GSM8K from 13.8 % to 67.1 % (an outside reproduction read 51.0 % on MATH-500).
- Failures stated: could not tell a sycophantic Gemma 3 270M fine-tune from an honest one from weights alone; edits remove obstacles but cannot add a missing ability (Gemma math 6.8 % to 7.7 %).
- Use for v8: it covers policy networks (PPO, Kinetix) but not vision encoders or world models. Treat as a hypothesis generator. Cheap probe: give a coding agent per-layer statistics of a v8F checkpoint and ask it to find untrained or dead parameter groups, then compare with the exact optimizer-state gate. Never quote these numbers as ours.
- The neighbouring term "weight space learning" is a separate area: survey arXiv 2603.10090 (Mar 2026; SECONDARY) on understanding, representing and generating weights (hypernetworks, hyper-representations).

**2.2 Compiled transformers.**
- Tracr, Lindner et al., arXiv 2301.05062, NeurIPS 2023. SECONDARY. RASP program to transformer weights. READ: ground-truth circuits. CHANGE: by construction, not editing. Cost: none, tiny models. LLM only.
- ALTA, Shaw et al., arXiv 2410.18077 (Oct 2024). SECONDARY. Adds loops, targets Universal Transformers; analysis of "expressible but not learnable". Use for v8: a template for stating in code which latent-dynamics function a block can express before asking training to learn it.
- RASP itself and Learning Transformer Programs: not searched; no claim.

**2.3 Model editing.**
- MEMIT, Meng et al., arXiv 2210.07229 (ICLR 2023): thousands of facts, multi-layer updates (some bibliographies cite 2305.14792; arXiv shows 2210.07229). SECONDARY.
- AlphaEdit, Fang et al., arXiv 2410.02355 (ICLR 2025 oral): null-space projection, one extra line on MEMIT-style code, +36.7 % average over the best baselines (snippet). SECONDARY.
- READ: where a fact lives (causal tracing). CHANGE: localised facts, no retraining, seconds to minutes. LLM only. Use for v8: the null-space idea is reusable for a safe patch of a planner head (edit orthogonal to the span of preserved inputs); no vision/policy use found.

**2.4 Parameter decomposition.** APD (Braun et al., 2025; not fetched) and SPD, Bushnaq, Braun, Sharkey, arXiv 2506.20790 (25 Jun 2025, v2 4 Sep 2025), code github.com/goodfire-ai/spd. SECONDARY. SPD learns a causal-importance function that samples masks over subcomponents; reportedly succeeds where APD failed on toy models; still compute heavy. READ: weight-space mechanisms per datapoint. CHANGE: ablate components. Toy models and small LMs only.

**2.5 SAEs, transcoders, attribution graphs.** Anthropic "Circuit Tracing: Revealing Computational Graphs in Language Models" and "On the Biology of a Large Language Model" (transformer-circuits.pub, Mar 2025). SECONDARY. Cross-layer transcoder replacement model; useful insight on about a quarter of tried prompts; limits: reconstruction error, inactive features, graph complexity. READ activation-level features (not weights). CHANGE by feature steering. Cost: training a transcoder per model. LLMs; no driving use found.

**2.6 Spectral diagnostics.** WeightWatcher / HTSR (Martin and Mahoney; not fetched). SECONDARY. Alpha is the power-law tail exponent of a layer's eigenvalue spectrum; folk thresholds vary by paper (2 to 4, or 2 to 5-6); alpha under 2 is linked to over-fitting. Recent HTSR grokking work: arXiv 2506.04434, 2602.02859. READ: per-layer quality with no data. CHANGE: nothing directly. Cost: one SVD per layer, minutes. Vision models are in scope. Use for v8: run it on every checkpoint as a free post-training gate; quote alpha only with fit window, R squared and n (programme rule).

**2.7 Not searched this pass:** task arithmetic, model merging, weight-space steering, hyper-representations, neural collapse, linear-probe guarantees. No claims made.

## Q3. Verifying networks and learned planners

**3.1 VNN-COMP 2025 report, arXiv 2512.19007 (Kaulen et al., 22 Dec 2025). PRIMARY (abstract) plus SECONDARY (snippet scores).** 8 teams, 16 regular and 9 extended benchmarks; networks as ONNX, specs as VNN-LIB; AWS equal-cost hardware. alpha-beta-CROWN won and was top on all scored benchmarks (1566.9 vs NNV 796.4 under zero tolerance, per a snippet).
- alpha-beta-CROWN / auto_LiRPA: input ONNX (with VNN-LIB) or PyTorch (auto_LiRPA is the PyTorch bound-propagation library, arXiv 2002.12920 from memory, NOT confirmed this pass). Properties: local robustness, output-range and linear input-output specs, GPU bound propagation with branch and bound. Scale: convolutional nets of about 10^5 to 10^6 neurons are typical in the competition (INHERITED, not verified). A 100 M+ parameter encoder is out of reach for complete verification; incomplete bounds on a small head are realistic.
- Marabou 2.0, arXiv 2401.14461 (CAV 2024), SECONDARY. SMT/simplex based, ONNX input, proof production, Python API, more activations. Complete but smaller scale; no 2025 competition result seen.
- NNV: in the 2025 standings; version 3.0 not confirmed; NNV 2.0 is CAV 2023. MATLAB-based reachability.
- Use for v8: verify the small planner head or trajectory decoder (input box around a latent, output box on waypoints), export ONNX with a fixed opset, run alpha-beta-CROWN. Do not attempt the encoder. State the property as a bound on output deviation, not as "safe".

**3.2 Conformal guarantees around learned predictors (all SECONDARY, snippets).**
- Lindemann et al., arXiv 2210.10254: conformal prediction regions on learned trajectory predictors fed into MPC; CARLA pedestrian intersection; no distribution assumption.
- FORCE-OPT, arXiv 2507.22389: calibrated forward-reachable sets from learned trajectory predictors, checked on nuScenes, with an online conservativeness filter.
- SafePath, arXiv 2505.09427: conformal filter on LLM-proposed paths; nuScenes and Highway-env.
- UNISafe, arXiv 2505.00779 (PMLR v305): world-model latent safety filter with a conformal OOD threshold; manipulation, not driving.
- Calibrated safety prediction with world models, arXiv 2508.09346; not driving.
- No paper found that combines a driving world model, conformal guarantees and verification of a learned planner. Coverage is marginal over exchangeable calibration scenes, so it answers the same kind of question as our episode bootstrap, not a per-scene guarantee.
- Use for v8: split-conformal calibration of operative-level ADE residuals on held-out episodes (episode-level exchangeability, per horizon) gives a coverage statement for a safety margin with no model access; cheapest to implement and fits our T1 tier.

**3.3 Not searched:** control barrier functions, shielding, RSS-based filters, any verified driving planner. No claims made; this is the gap to fill next.

## Six most actionable
1. Lean: use Zhang's rl-theory-in-lean (2511.03618) as the base; first agent task is the performance-difference and simulation lemmas on finite MDPs.
2. Anthropic recipe: frozen spec, kernel as sole judge, edits to spec or code penalised and logged, mutation check.
3. One Mathlib project on F:, `lake exe cache get`, budget 6 to 8 GB.
4. alpha-beta-CROWN on the planner head via ONNX only.
5. Split-conformal residual margins per horizon on held-out episodes.
6. WeightWatcher alpha and effective rank as a free checkpoint gate, quoted with window, R squared, n.

## arXiv ids cited
2511.03618
2512.19007
2504.21801
2603.10090
2301.05062
2410.18077
2210.07229
2305.14792
2410.02355
2506.20790
2506.04434
2602.02859
2401.14461
2002.12920
2210.10254
2507.22389
2505.09427
2505.00779
2508.09346
