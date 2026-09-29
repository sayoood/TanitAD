# Stream V1 — Citation Verification Pass (2026-09-25)

**Verification method:** WebSearch for each citation (arXiv ID + key terms, then title if needed)
**Search budget used:** 31 of 45 searches

## Citation Verification Table

| # | citation as used | verdict | evidence |
|---|---|---|---|
| 1 | LeJEPA, Balestriero & LeCun, arXiv:2511.08544 | CONFIRMED | arXiv:2511.08544; title "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"; SIGReg (Sketched Isotropic Gaussian Regularization) confirmed; JEPA without heuristics (no stop-gradient, teacher-student, schedulers) |
| 2 | LFG, arXiv:2602.22091 (CVPR 2026) | EXISTS-CLAIM-UNCONFIRMED | arXiv:2602.22091 "Learning to Drive is a Free Gift"; PDMS 85.2 confirmed; "10% labels" mentioned in search but specific 81.4 number not found |
| 3 | DriveVLA-W0, arXiv:2510.12796 | CONFIRMED | arXiv:2510.12796; world modeling as dense supervision amplifies data scaling for VLA driving confirmed |
| 4 | WoTE, arXiv:2504.01941 | CONFIRMED | arXiv:2504.01941; "End-to-End Driving with Online Trajectory Evaluation via BEV World Model"; PDMS 88.3 on NAVSIM confirmed |
| 5 | GTRS, arXiv:2506.06664 | CONFIRMED | arXiv:2506.06664; "Generalized Trajectory Scoring for End-to-end Multimodal Planning"; NAVSIM v2 challenge winner confirmed |
| 6 | DriveSuprim, arXiv:2506.06659 | CONFIRMED | arXiv:2506.06659; "Towards Precise Trajectory Selection"; coarse-to-fine trajectory selection + 87.1 EPDMS confirmed |
| 7 | ZTRS, arXiv:2510.24108 | CONFIRMED | arXiv:2510.24108; "Zero-Human Demonstration End-to-end Autonomous Driving"; zero-imitation, EPO reward-only confirmed |
| 8 | SimLingo, arXiv:2503.09594 | CONFIRMED | arXiv:2503.09594; "Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment"; disentangled waypoint prediction confirmed |
| 9 | GoalFlow, arXiv:2503.05689 | CONFIRMED | arXiv:2503.05689; "Goal-Driven Flow Matching for Multimodal Trajectories"; PDMS 90.3 confirmed |
| 10 | Hydra-MDP, arXiv:2406.06978 | CONFIRMED | arXiv:2406.06978; "End-to-end Multimodal Planning with Multi-target Hydra-Distillation"; NAVSIM challenge winner (1st place) confirmed |
| 11 | Dreamer 4, arXiv:2509.24527 | CONFIRMED | arXiv:2509.24527; "Training Agents Inside of Scalable World Models"; offline diamonds in Minecraft confirmed |
| 12 | TRM "Less is More: Recursive Reasoning with Tiny Networks", arXiv:2510.04871 | CONFIRMED | arXiv:2510.04871; TRM beats HRM confirmed; 7M parameters vs LLMs |
| 13 | Energy-Based Transformers, arXiv:2507.02092 | CONFIRMED | arXiv:2507.02092; "Energy-Based Transformers are Scalable Learners and Thinkers" confirmed |
| 14 | BarrierNet, arXiv:2203.02401 | CONFIRMED | arXiv:2203.02401; "Differentiable Control Barrier Functions for Vision-based End-to-End Autonomous Driving"; safe robot control confirmed |
| 15 | differentiable kinematic bicycle model decoder, arXiv:2603.12421 | EXISTS-CLAIM-UNCONFIRMED | arXiv:2603.12421; "A Neuro-Symbolic Framework"; differentiable kinematic bicycle model mentioned but exact decoder claim not fully confirmed |
| 16 | Synergistic Simplex, arXiv:2605.08190 | CONFIRMED | arXiv:2605.08190; "Cooperative Runtime Assurance for Safety-Critical Autonomous Systems"; simplex/runtime-assurance architecture for learned controllers confirmed |
| 17 | RECTOR, arXiv:2605.25095 | EXISTS-CLAIM-UNCONFIRMED | arXiv:2605.25095; "Priority-Aware Rule-Based Reranking"; tiered rulebook (Safety ≻ Legal ≻ Road ≻ Comfort) + WOMD confirmed; specific % numbers (28.58 % → 20.42 %) not in search snippet |
| 18 | neuro-symbolic post-hoc safety guard for driving commands, arXiv:2608.11451 | CONFIRMED | arXiv:2608.11451; "Herding End-to-End Autonomous Driving via Neuro-Symbolic Safety Guards"; post-hoc guard on E2E/VLA outputs confirmed |
| 19 | conformal calibration of world-model imagination / confidence, arXiv:2608.26533 | ID-MISMATCH | arXiv:2608.26533 exists but is "Barrier Function Conformal Safety Clearance Certification with CVaR"; title/topic do not match "world-model imagination/confidence" claim |
| 20 | DreamLedger, arXiv:2608.23863 | CONFIRMED | arXiv:2608.23863; "Where to Refuse World-Model Imagination Using Execution-Settled Credit"; execution-settled imagination/refusal confirmed |
| 21 | GraphPilot, arXiv:2511.11266 | CONFIRMED | arXiv:2511.11266; "Grounded Scene Graph Conditioning"; relational/graph supervision at train time only; gains persist without graph at test confirmed |
| 22 | Chat2Scenic, arXiv:2607.14387 | CONFIRMED | arXiv:2607.14387; "An Iterative RAG-Based Framework for Scenario Generation"; LLM-generated Scenic scenarios + 76.42 % compile success confirmed |
| 23 | MOSAIC, arXiv:2604.08366 (NVIDIA/NYU) | EXISTS-CLAIM-UNCONFIRMED | arXiv:2604.08366; "Scaling-Aware Data Selection for End-to-End Autonomous Driving"; scaling-law-guided data-mixture selection confirmed; "up to 80 % fewer examples" not explicit in search |
| 24 | Cosmos-Drive-Dreams, arXiv:2506.09042 | CONFIRMED | arXiv:2506.09042; "Scalable Synthetic Driving Data Generation with World Foundation Models" confirmed |
| 25 | arXiv:2403.11304 | CONFIRMED | arXiv:2403.11304; "Pioneering SE(2)-Equivariant Trajectory Planning"; SE(2) equivariance/mirror symmetry + +20.6 % L2@3s on small data confirmed |
| 26 | Alpamayo-R1, arXiv:2511.00088 (NVIDIA) | CONFIRMED | arXiv:2511.00088; "Bridging Reasoning and Action Prediction"; reasoning VLA with chain of causation for driving + NVIDIA confirmed |
| 27 | reward-hacking survey of autonomous research agents, arXiv:2609.28614 | EXISTS-CLAIM-UNCONFIRMED | arXiv:2609.28614; "Reward Hacking Challenges Oversight"; 30.5 % hack rate found for open-ended tasks; citation claims "30.5 % / 74.6 %" but only 2.9 % for task-specific kernels found |
| 28 | NVIDIA AlpaSim E2E Closed Loop Challenge 2026 | EXISTS-CLAIM-UNCONFIRMED | AlpaSim E2E Challenge exists on HF; PhysicalAI-AV NuRec data use confirmed; closing date 2026-10-31 not explicitly confirmed |
| 29 | Safety case patterns for VLA-based driving (SimLingo), arXiv:2603.16013 | CONFIRMED | arXiv:2603.16013; "Safety Case Patterns for VLA-based driving systems: Insights from SimLingo" confirmed |
| 30 | Rig3R, arXiv:2506.02265 | CONFIRMED | arXiv:2506.02265; "Rig-Aware Conditioning for Learned 3D Reconstruction"; cross-rig generalization confirmed |
| 31 | NVIDIA PhysicalAI-Autonomous-Vehicles dataset (Hugging Face card) | CONFIRMED | HF card nvidia/PhysicalAI-Autonomous-Vehicles; 306,152 clips, ~1,700 h, multi-sensor data confirmed |

## Summary

| verdict | count |
|---|---|
| CONFIRMED | 23 |
| EXISTS-CLAIM-UNCONFIRMED | 7 |
| ID-MISMATCH | 1 |
| NOT-FOUND | 0 |

**Non-confirmed entries requiring attention:**

- **#2 (LFG):** PDMS 85.2 confirmed, but "81.4 PDMS with 10% labels" claim unverified
- **#15 (Bicycle model):** Paper exists but exact title/scope mismatch  
- **#17 (RECTOR):** Tiered rulebook confirmed on WOMD, but specific violation reduction percentages not found
- **#19 (arXiv:2608.26533):** ID points to barrier function paper, not world-model confidence calibration
- **#23 (MOSAIC):** Data-selection scaling law confirmed; "80% fewer examples" not verified
- **#27 (Reward-hacking):** 30.5% confirmed for open-ended; 74.6% figure does not match search results
- **#28 (AlpaSim):** Challenge exists on PhysicalAI-AV NuRec; 2026-10-31 deadline unverified

## Deliverable Manifest

| artifact | location | status |
|---|---|---|
| V1_citation_check.md | repo:/Project Steering/Reviews/2026-09-25-programme-review/streams/ | staged |
