# Progress and Paradigm Shifts in AI-Driven Autonomous Driving: End-to-End Architectures, Latent World Models, and Edge AI

The architecture of autonomous driving (AD) systems is currently undergoing a fundamental transformation. For over a decade, the industry relied on modular, cascaded pipelines in which perception, prediction, and planning operated as isolated subsystems. Although this compartmentalization facilitated software development and the isolated testing of components, it inevitably led to information bottlenecks, cascading error propagation, and an inability to natively optimize the overall system for the ultimate goal of safe vehicle control. Recent breakthroughs have clearly proven that so-called end-to-end (E2E) architectures—defined as fully differentiable neural networks mapping raw, multimodal sensor data directly to planned trajectories or low-level control commands—are significantly superior to modular systems.

By propagating feature representations globally across the entire network and utilizing backpropagation to minimize planning-oriented loss functions, E2E systems can capture complex interactions between road users and long-range spatial dependencies that are typically discarded in modular systems. An outstanding example of this development is the BridgeAD architecture, which integrates historical predictions for the current frame directly into the perception module, while historical predictions and plans for future frames are linked to motion planning. This approach follows the philosophy that the future is a continuous extension of the past, thereby bridging the gap in temporal coherence.

Nevertheless, realizing production-ready E2E autonomy required solving several critical bottlenecks: managing the long-tail distribution of rare traffic events, integrating deductive logic and observability without unacceptable inference latencies, safely validating policies in closed-loop simulations, and compressing massive foundation models for real-time deployment on edge hardware. This report provides an exhaustive analysis of the most successful current frameworks addressing these challenges and synthesizes analytical derivations to detail the current state and future trajectory of AI-driven autonomous mobility.

## Training Data Curation, Filtering, and Solving Long-Tail Problems

The performance of E2E systems is fundamentally limited by the quality, diversity, and scalability of their training data. Real-world driving data exhibits an extreme long-tail distribution; common scenarios with low interaction density dominate the datasets, while safety-critical edge cases are massively underrepresented. Analyses of scaling laws in autonomous driving based on industrial datasets with over 8,000 driving hours show that merely increasing data volume is insufficient to linearly boost performance in closed-loop scenarios if the data distribution is not intelligently managed. Manually annotating rare events is not scalable due to cost.

### Automated Data Engines and Zero-Shot Labeling

To solve the long-tail problem, researchers have developed fully automated data engines, such as the Automatic Data Engine (AIDE). These systems utilize Vision-Language Models (VLMs) and Large Language Models (LLMs) to construct self-improving training loops. The traditional data pipeline is thereby broken down into automated, intelligent subsystems:

First, dense captioning models generate highly detailed semantic descriptions of raw video frames (Issue Finder). These descriptions are matched against the existing label space and model predictions to automatically identify new, out-of-distribution object categories. Subsequently, VLMs act as semantic zero-shot search engines (Data Feeder) to query massive repositories of uncurated sensor logs and specifically isolate frames containing these novel scenarios.

The subsequent auto-labeling phase (Model Updater) uses open-vocabulary object detection architectures to generate bounding boxes for the queried images. Foundation models then perform zero-shot classification on these image crops to assign highly precise pseudo-labels, completely eliminating the need for human annotation. Finally, in a verification phase, LLMs synthesize diverse textual descriptions of potential scenario variations to guide further data queries and continuously safeguard the updated E2E model against regressions.

### Active Curation and Density-Balancing Frameworks

Beyond automated labeling, the exact composition of training batches determines the stability of the E2E policy. The ACID (Active Curation as Implicit Distillation) framework demonstrates that dynamic, online batch selection often outperforms complex knowledge distillation procedures. By continuously selecting training samples that maximize the performance gap between a small student model and a massive reference model, ACID implicitly forces the student model to focus on the hardest, most information-rich long-tail distributions.

Complementary to this, frameworks like Den-TP focus on density-balanced data curation. By analyzing the spatial and temporal density of agents within a scene, these pipelines re-weight training samples to prioritize highly interactive multi-agent environments, which often get lost in standard datasets amidst mundane highway driving. A holistic approach that considers data quality as a central metric is also proposed in a five-layer Vase framework (Data, Data Quality, Task, Application, Goal), proving that the targeted removal of redundancies in multimodal sensor data can significantly improve the performance of downstream detection models.

_Analytical Derivation:_ Data curation is no longer a static preprocessing step but a dynamic, active participant in the training loop. Foundation models have evolved from being the end product of data pipelines to autonomous architects of the datasets themselves. This creates a recursive improvement loop: stronger foundation models provide higher-quality pseudo-labels and more precise curation, which in turn enables the training of even more robust E2E driving policies.

## Cognitive Logic, Observability, and Reasoning in E2E Models

A historical criticism of E2E neural networks is their "black-box" nature. The lack of observability and interpretability poses a massive hurdle for safety-critical certification. To solve this, modern systems integrate Vision-Language-Action (VLA) architectures that provide explicit semantic deductions and human-readable decision rationales. However, directly querying massive VLMs in the primary control loop introduces significant latencies that clash with the millisecond requirements of vehicle operation.

### Distillation of Deductive Logic into Latent Spaces

To resolve the discrepancy between cognitive observability and inference speed, frameworks like VERDI implement an advanced distillation paradigm. Instead of querying a VLM at runtime, the E2E model is trained in parallel with an offline VLM.

During training, ground-truth trajectories and multi-view images are passed to a teacher VLM, which generates structured textual explanations for the perception, prediction, and planning phases using sequential logic prompts. These textual responses are encoded into high-dimensional latent vectors. Simultaneously, the modular E2E driving model processes the same sensor inputs to generate intermediate feature representations. Using so-called Progressive Feature Projectors, the E2E features are projected into the shared latent space and aligned with the VLM's semantic embeddings via cosine similarity optimization.

_Analytical Derivation:_ By enforcing alignment exclusively in the latent space during backpropagation, the E2E model internalizes the structural understanding and physical common sense of the massive VLM. At inference time, the VLM is entirely discarded. This allows the compact E2E model to exhibit human-like semantic deductions and zero-shot generalizations while maintaining a real-time execution speed of over 4.5 FPS.

### Causal Alignment, 3D Metrics, and Chain-of-Causation

Unstructured text generation is often too unconstrained for the strict physical realities of driving. Moreover, classic VLMs primarily rely on 2D images, leading to insufficient metric spatial perception. The LVLDrive (LiDAR-Vision-Language) framework addresses this by augmenting pre-trained VLMs with LiDAR point clouds via a Gradual Fusion Q-Former structure. The incremental injection of 3D features into the VLM's attention mechanism prevents the catastrophic forgetting of linguistic priors while granting the model robust, metric spatial understanding.

On a logical level, the Alpamayo-R1 architecture introduces a causally grounded approach utilizing a highly structured "Chain-of-Causation". Based on a massive vision-language backbone, Alpamayo-R1 strictly constrains semantic generation to:

1. A singular high-level decision (e.g., yield, merge).
    
2. The minimal set of causal factors in the environment directly responsible for this decision.
    
3. A precise, causally linked textual path connecting the observation to action.
    

By eliminating irrelevant environmental descriptions and enforcing strict causal locality (limiting historical context to a 2-second window), Alpamayo-R1 achieves a 132.8% improvement in causal understanding of long-tail events and a 37% increase in consistency between generated text and the physical vehicle maneuver. The model hits an inference latency of 99 ms, proving real-time capability for Level 4 autonomous systems.

Additionally, the concept of a Semantic Observer is being explored, running as an independent autonomy layer at 1-2 Hz alongside the primary control loop to detect semantic anomalies (e.g., a deflated ball on the road vs. a shadow) and orchestrate fail-safe handovers if necessary.

## Latent Space Planning and Efficient Trajectory Decoding

Traditional VLA planners often decode waypoints autoregressively, similar to how LLMs generate text. This sequential token-by-token generation suffers massively from exposure bias—small initial coordinate errors drastically compound over a multi-second trajectory—and is inherently limited by memory bandwidth on edge GPUs. Novel frameworks solve this by decoding continuous trajectories in parallel or using highly structured latent primitives.

### Parallel and Hierarchical Decoding

The ColaVLA framework eliminates autoregressive latency via a unified latent space and a Hierarchical Parallel Planner. Instead of decoding textual logic and waypoints sequentially, a Cognitive Latent Reasoner compresses multimodal scene understanding into highly compact, decision-oriented "meta-action embeddings," requiring only two VLM forward passes. These embeddings act as global priors. The parallel planner then uses a hybrid attention mask to simultaneously decode all temporal scales and multimodal trajectory waypoints in a single forward pass. This ensures causal consistency and eliminates the modality conflict between discrete text tokens and continuous geometric control.

### Structured Motion Primitives and VQ-VAE

The LAMP (Lane-Aligned Motion Primitives) framework moves away from abstracting driving intentions as simple coordinate endpoints. Instead, it employs a Vector Quantized Variational AutoEncoder (VQ-VAE) to learn a discrete codebook of shape-aware trajectory prototypes. These "motion primitives" capture complex spatiotemporal dynamics—such as specific curvature and acceleration profiles—that endpoint-based methods cannot model.

Crucially, LAMP integrates a topology-aware intention selector. Before the decoder finalizes a trajectory, this selector evaluates the discrete queries against a lane topology prior and filters out aggressive, off-road, or physically unreachable intentions. This guarantees that even alternative trajectories with lower probabilities remain structurally feasible, an indispensable requirement for safety-critical emergency planning.

### Decoding Semantic Cost Maps

Directly mapping compressed latent features to trajectory coordinates creates a highly entangled representation where risks and drivability are hard to isolate. The PLAN-S architecture bridges this gap by decoding a style-conditioned, four-channel semantic cost map directly from the latent Bird's-Eye-View (BEV) representation.

These four channels explicitly model dynamic obstacles, static obstacles, off-road areas, and general drivability. Using dual adaptive feature-wise linear modulation (AdaFiLM), these cost maps are dynamically conditioned on the ego vehicle's current kinematic state and a specified driving style preference. On the nuScenes dataset, PLAN-S reduced the collision rate in the 3-second window by a relative 42%.

### Block-Diffusion and Scaffold Speculative Decoding

To overcome diffusion model latency, the Fast-dDrive model introduces block-diffusion combined with Scaffold Speculative Decoding. Recognizing that VLA models for driving often output highly structured data (e.g., JSON schemas), Fast-dDrive freezes structural tokens into a "scaffold." Through a forward pass with block-bidirectional attention, all masked value tokens are drafted simultaneously. A causal, autoregressive head then verifies the block. This enables massive KV-cache reuse and parallelization, resulting in a 12x throughput speedup over autoregressive baselines and achieving state-of-the-art latency profiles for edge hardware.

|**Trajectory Decoding**|**Mechanism**|**Primary Advantage**|**Addressed Limitation**|
|---|---|---|---|
|**Hierarchical Parallel (ColaVLA)**|Single-forward-pass decoding via meta-action embeddings.|Eliminates sequential exposure bias, maximizes GPU parallelism.|High latency and compounding errors of autoregressive generation.|
|**Motion Primitives (LAMP)**|VQ-VAE codebook of trajectory prototypes, filtered by lane topology.|Guarantees physical and topological feasibility of all modes.|Impossible, off-road predictions in alternative trajectories.|
|**Semantic Cost Maps (PLAN-S)**|AdaFiLM-conditioned 4-channel cost maps prior to waypoint selection.|Provides explicit observability of risks and driving style modulation.|Entangled, uninterpretable latent representations.|
|**Block-Diffusion (Fast-dDrive)**|Scaffold Speculative Decoding with KV-cache reuse.|12x inference speedup, causal verification.|Memory-bandwidth bottleneck in diffusion and AR models.|

_Analytical Derivation:_ Injecting an explicit spatial cost map (as in PLAN-S) or a topological filter (as in LAMP) between the latent encoder and trajectory decoder acts as a highly effective regularization mechanism. It forces the network to maintain a strict spatial understanding of physics and risk, ensuring that the final trajectory selection is bounded by geometric reality rather than arbitrary statistical correlations in the training data.

## World Models for Autonomous Driving: Simulation and Reinforcement Fine-Tuning

Latent World Models have established themselves as the definitive mechanism for both self-supervised representation learning and the safe optimization of RL (Reinforcement Learning) policies. A World Model acts as an internal neural simulator that compresses high-dimensional sensor data into a compact state and rolls it forward into the future, conditioned on the ego vehicle's hypothetical actions. Current taxonomies categorize these models by the form of their latent representations (continuous states, discrete tokens, or hybrids) and their structural priors for geometry and semantics.

### Reinforcement Fine-Tuning in Latent Space

Training driving policies via RL in the physical world is dangerous, and rendering complex 3D simulations for millions of RL episodes is computationally exorbitant. Latent World Models solve this by allowing the agent to train and receive rewards entirely within the compressed vector space.

The WorldRFT framework (World Model Planning with Reinforcement Fine-Tuning) leverages this by structuring the planning task into a hierarchical decomposition and applying Group Relative Policy Optimization (GRPO). By Gaussianizing trajectories and applying collision-aware rewards within the latent predictive environment, WorldRFT systematically optimizes the driving policy, reducing collision rates on the nuScenes benchmark by 83% (from 0.30% to 0.05%).

The Think2Drive framework extends this application by using a Latent World Model to train planning in the challenging CARLA-v2 simulator, which includes 39 highly complex corner-case scenarios (like construction zones or merging into flowing traffic). Due to the massive efficiency gains of the latent space, Think2Drive was trained to expert level in just 3 days on a single GPU and became the first documented model to achieve 100% route completion in CARLA-v2.

### Compressing Diffusion Inference for RL

The DreamerAD architecture drastically accelerates latent RL by compressing the diffusion sampling process of video generation models. Standard diffusion world models require up to 100 iterative denoising steps to "imagine" a future state, leading to significant latency (e.g., 2 seconds per frame) that blocks high-frequency RL interactions. DreamerAD employs "Shortcut Forcing," a recursive technique for multi-resolution step compression that reduces diffusion sampling to a single step. Combined with an autoregressive dense reward model operating directly on latent features, this enables an 80-fold acceleration in RL training while maintaining the ability to losslessly decode latent vectors into RGB frames for human verification.

_Analytical Derivation:_ World Models represent a paradigm shift from reactive AI to predictive, imaginative AI. By decoupling the physical environment engine from computationally heavy graphics rendering, latent world models allow autonomous agents to hallucinate millions of counterfactual scenarios. The AI thus learns at machine speed to navigate catastrophic edge cases without ever exposing a physical chassis to risk.

## Efficient Validation in Closed-Loop Simulations

Historically, evaluating E2E AD systems relied on open-loop metrics like Average Displacement Error (ADE). However, open-loop tests are fundamentally flawed for autonomous control; because the model's predicted actions do not influence the future state of the pre-recorded environment, minor deviations do not compound, and interactive responses from other agents are ignored. Leading frameworks therefore strictly enforce evaluation in closed-loop simulations to measure actual safety and error-correction capabilities. For example, HiDrive introduces an advanced benchmark based on Unreal Engine 5, featuring over 330 routes with rare long-tail traffic situations and physically realistic lighting to evaluate advanced moral and legal decisions in closed-loop settings.

### Generator-Scorer Ranking and Pseudo-Simulation

The CLOVER (Closed-Loop Value Estimation and Ranking) framework proves that training solely by imitating individual trajectories leads to a severe evaluation mismatch. Trajectories that safely deviate from human demonstrations are often penalized by standard loss functions, even if they are perfectly valid and safe according to planning-based metrics. CLOVER uses a lightweight generator-scorer architecture. It constructs evaluator-filtered pseudo-expert trajectories to provide set-level coverage supervision. During closed-loop self-distillation, the trajectory scorer is fitted to true environment sub-scores, while the generator is conservatively refined toward teacher-selected, vector-Pareto-optimal targets.

In parallel, the PerlAD framework constructs a data-driven pseudo-simulation entirely in vector space to bridge the gap between static datasets and dynamic execution. By introducing a Prediction World Model that generates reactive behaviors for surrounding agents—conditioned on the ego vehicle's plan—PerlAD allows a hierarchical decoupled planner (combining Imitation Learning for lateral path generation and RL for longitudinal speed optimization) to train efficiently without expensive online rendering. PerlAD outperforms previous E2E RL methods on the Bench2Drive benchmark by 10.29% in Driving Score.

### 3D Gaussian Splatting for Sensor-Realistic Rendering

To close the visual sim-to-real gap for closed-loop testing, the industry has rapidly adopted 3D Gaussian Splatting (3DGS). While Neural Radiance Fields (NeRFs) suffer from prohibitive rendering latencies, 3DGS enables photorealistic novel view synthesis in real time.

The SplatAD architecture pushes this boundary by being the first 3DGS framework to render both camera and LiDAR data in real time. Traditional 3DGS models only model RGB pixels. SplatAD mathematically projects 3D Gaussians into spherical coordinates and intersects them with non-equidistant tiles that perfectly match physical LiDAR beam distributions. Furthermore, it rasterizes depth and features over a non-linear point grid and applies custom algorithms to physically accurately compensate for rolling shutter effects, LiDAR intensity returns, and stochastic ray dropouts. The DriveE2E framework further extends this realism by importing digital twins of real-world intersections into the CARLA simulator based on infrastructure cameras.

To ensure these simulated environments provide diverse long-tail training, the RealityBridge framework enables editable 3DGS environments—such as object insertion or trajectory modification—while applying video generative priors to translate the edited, artificial-looking splats back into artifact-free, real-world camera styles (Sim-to-Real Editing).

## Personalization and V2X Cooperation

As E2E models mature, user acceptance comes into focus, necessitating the integration of individual driving styles. Traditional E2E approaches often converge to a generic mean, leading to uncomfortable or unpredictable behavior for specific users. Frameworks like MAVERIC and StyleDrive implement data-driven personalization approaches by capturing user-specific traits via vision-language models and style-based metrics (like Maximum Mean Discrepancy) and applying them to the planning process, significantly increasing subjective driving safety and comfort.

Concurrently, Vehicle-to-Everything (V2X) communication expands ego vehicles' perception horizons beyond direct line-of-sight. The central challenge here is multi-agent sensor fusion under strict bandwidth constraints. Modern V2X systems therefore use sparse, query-based methods and transformers to efficiently encode cooperative representations and dynamically decide which safety-critical information must be transmitted to avoid communication latencies.

## Edge Deployment: Efficient Inference and Hardware Optimization

An E2E model's theoretical accuracy is irrelevant if it cannot meet the strict latency (30-100 ms), memory, and thermal constraints of in-vehicle edge hardware. Deploying heavy Vision Transformers (ViTs) and massive VLA networks requires rigorous, hardware-aware adaptation.

### Heterogeneous Computing and Transformer Adaptation

Modern edge platforms (e.g., NVIDIA Jetson, DRIVE Orin) feature heterogeneous compute clusters combining standard GPUs with fixed-function accelerators like Deep Learning Accelerators (DLAs) and Optical Flow Accelerators (OFAs). Standard ViTs possess structural incompatibilities that force computations to fall back from the highly efficient DLA to the GPU, causing severe latency spikes.

Recent optimizations resolve this by reshaping tensors, approximating complex error functions (ERF) with hardware-friendly `tanh` operations, and replacing standard Layer Normalization with bounded `tanh` activations. Leveraging sophisticated scheduling frameworks like H-FraDS, which balance execution across dual-DLA and GPU cores simultaneously, adapted Swin Transformers achieve over 125 FPS (under 24 ms latency per frame) with a 2.36x throughput speedup, comfortably exceeding the 30 FPS requirement for urban driving.

### Model Compression and Quantization

To fit massive parameter counts into limited VRAM, aggressive Post-Training Quantization (PTQ) to INT8 or FP8 precision is mandatory. However, directly quantizing large models often causes catastrophic performance degradation (e.g., a 23% mAP drop). Research proves that Knowledge Distillation (KD) is an absolute prerequisite for stable edge deployment; a compact student model (e.g., YOLOv8-S) trained to mimic the unquantized teacher model preserves quantization robustness, maintains high precision, and reduces false alarms by 44% compared to a directly collapsed teacher model.

|**Optimization Technique**|**Target Hardware/Constraint**|**Mechanism**|**Performance Impact**|
|---|---|---|---|
|**LayerNorm to bounded `tanh`**|Fixed-function DLAs|Removes unsupported operations to avoid GPU fallbacks.|>2x throughput speedup; ensures strict sub-30ms latency.|
|**Knowledge Distillation + INT8**|Edge GPUs with low VRAM|Student mimics precision calibration of unquantized teacher.|Prevents catastrophic mAP drop; 44% fewer false alarms.|
|**Scaffold Speculative Decoding**|VLA / Diffusion models (Fast-dDrive)|Auto-accepts JSON scaffolds; drafts/verifies token blocks in parallel.|12x throughput speedup; maximizes KV-cache reuse.|

## Production Readiness, Commercial Deployment, and System Integration

Transitioning these advanced AI architectures from academic benchmarks to commercial deployment reveals a strategic bifurcation in the industry between monolithic E2E models and highly structured "Agentic AI" swarms.

### Globally Scalable, Monolithic E2E Deployment

Companies like WeRide, in partnership with Tier-1 suppliers like Bosch, are proving the viability of single-stage monolithic E2E architectures on a global scale. Their proprietary models integrate perception, prediction, planning, and control into a single neural network. By embedding this software into automotive-grade domain controllers, they are currently conducting Level 2++ validations on public roads in Germany, France, Japan, and China.

The primary advantage of the unified E2E architecture is its rapid adaptability. Because the model relies on global feature propagation rather than hard-coded heuristics, it can quickly generalize to diverse international traffic regulations, varying driving cultures, and highly variable climate conditions without requiring profound modular rewrites.

### Agentic AI and OEM-Agnostic Platforms

In contrast, deployments like the Level 4 robotaxi program by Uber and Autobrains in Munich underscore an alternative, highly modular approach termed "Agentic AI". Instead of processing the entire driving task through a single massive neural network, the Agentic AI architecture breaks the cognitive load down into a swarm of specialized, autonomous software agents.

One AI agent is trained exclusively to evaluate right-of-way logic, another monitors pedestrian trajectories, and another regulates lane-change dynamics. A higher-level heuristic system runs these agents in parallel to finalize decisions in real time. Based on computing platforms like NVIDIA DRIVE Hyperion, this approach requires significantly less computational power (e.g., 1-2 TOPS for specific camera modules).

_Analytical Derivation:_ The Agentic AI approach represents a calculated compromise between the power of deep learning and the stringent safety requirements of European regulators. By compartmentalizing neural logic, developers can isolate errors and provide deterministic guarantees for specific subsystems (e.g., adherence to right-of-way rules), which is substantially harder to certify in a monolithic E2E black box. Furthermore, this approach facilitates an "OEM-agnostic" model; it eliminates the need for bespoke, heavily modified vehicle roofs loaded with unique sensor arrays, allowing production vehicles from established automakers to integrate into autonomous ride-hailing networks with minimal physical modifications.

## Summary and Outlook

The most successful recent advancements in AI-driven autonomous driving share a common direction: the dissolution of rigid, heuristic-based modules in favor of continuous, differentiable latent spaces. However, the sheer scale of foundation models has necessitated profound architectural innovations to make these approaches physically and economically viable.

To solve the long-tail dilemma, automated data engines (AIDE) and active curation frameworks (ACID) have transformed passive datasets into dynamic, self-refining curricula. To integrate observability without fatal latency penalties, frameworks like VERDI and Alpamayo-R1 successfully distill structured causal logic directly into the latent representations of compact inference models. Trajectory decoding has evolved from slow, error-prone autoregressive generation to highly efficient, parallel processes utilizing discrete motion primitives (LAMP) and semantic risk maps (PLAN-S).

Simultaneously, the industry has fundamentally changed how these systems are tested and trained. Latent World Models (DreamerAD, WorldRFT, Think2Drive) now serve as indispensable internal physics engines for rapid, risk-free reinforcement fine-tuning. For final validation, closed-loop neural simulators using advanced 3D Gaussian Splatting (SplatAD) have successfully bridged the visual and physical sim-to-real gap for both camera and LiDAR modalities, while systems like CLOVER effectively eliminate imitation biases in evaluation.

Ultimately, the path to ubiquitous Level 4 autonomy depends entirely on the co-design of hardware and software. Through rigorous speculative decoding, layer normalization adaptations, and distillation-backed quantization, these massive cognitive systems now operate within the strict thermal and temporal confines of in-vehicle edge computers. Whether deployed as monolithic E2E architectures for global scaling or as compartmentalized Agentic AI swarms for regulatory certification—current research emphatically proves that scalable, OEM-agnostic, and globally adaptable autonomous driving is no longer a theoretical pursuit, but an imminent commercial reality.

## References

- [2] https://www.computer.org/csdl/journal/tp/2024/12/10614862/1Z0o0IOChhK
    
- [4] https://arxiv.org/html/2504.00911v2
    
- [5] https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Past_and_Future_End-to-End_Autonomous_Driving_with_Historical_Prediction_CVPR_2025_paper.pdf
    
- [6] https://arxiv.org/html/2507.21610v1
    
- [10] https://github.com/HaoranZhuExplorer/World-Models-Autonomous-Driving-Survey
    
- [12] https://arxiv.org/html/2501.11260v4
    
- [14] https://arxiv.org/html/2603.11093v1
    
- [15] https://arxiv.org/html/2505.15925v4
    
- [16] https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chahe_ReasonDrive_Efficient_Visual_Question_Answering_for_Autonomous_Vehicles_with_Reasoning-Enhanced_CVPRW_2025_paper.html
    
- [17] https://arxiv.org/pdf/2512.24331
    
- [19] https://arxiv.org/abs/2603.14908
    
- [20] https://arxiv.org/html/2605.09972v1
    
- [21] https://arxiv.org/abs/2605.15120
    
- [22] https://papers.nips.cc/paper_files/paper/2025/file/3b3ea354dbda9579b4e134421bab6449-Paper-Conference.pdf
    
- [23] https://arxiv.org/pdf/2509.23922
    
- [24] https://arxiv.org/abs/2605.09972
    
- [25] https://arxiv.org/html/2607.10942v1
    
- [26] https://arxiv.org/pdf/2304.10891
    
- [27] https://arxiv.org/abs/2607.10942
    
- [29] https://arxiv.org/html/2601.03290v1
    
- [30] https://arxiv.org/html/2601.22927v1
    
- [32] https://cvpr.thecvf.com/virtual/2024/poster/29457
    
- [33] https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.pdf
    
- [34] https://arxiv.org/html/2512.22939v3
    
- [35] https://arxiv.org/html/2606.26661v1
    
- [36] https://arxiv.org/html/2606.16274v1
    
- [37] https://arxiv.org/pdf/2606.26661
    
- [38] https://openaccess.thecvf.com/content/CVPR2026/papers/Peng_ColaVLA_Leveraging_Cognitive_Latent_Reasoning_for_Hierarchical_Parallel_Trajectory_Planning_CVPR_2026_paper.pdf
    
- [40] https://arxiv.org/abs/2606.06014
    
- [42] https://www.researchgate.net/publication/386089287_Think2Drive_Efficient_Reinforcement_Learning_by_Thinking_with_Latent_World_Model_for_Autonomous_Driving_in_CARLA-V2
    
- [43] https://thinklab-sjtu.github.io/CornerCaseRepo/
    
- [44] https://arxiv.org/abs/2512.19133
    
- [46] https://openaccess.thecvf.com/content/CVPR2025/papers/Udandarao_Active_Data_Curation_Effectively_Distills_Large-Scale_Multimodal_Models_CVPR_2025_paper.pdf
    
- [47] https://cvpr.thecvf.com/virtual/2025/poster/35213
    
- [48] https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Den-TP_A_Density-Balanced_Data_Curation_and_Evaluation_Framework_for_Trajectory_CVPR_2026_paper.pdf
    
- [49] https://arxiv.org/pdf/2504.04338
    
- [50] https://arxiv.org/html/2506.17346v1
    
- [53] https://rex.libraries.wsu.edu/view/pdfCoverPage?instCode=01ALLIANCE_WSU&filePid=13426258180001842&download=true
    
- [54] https://arxiv.org/pdf/2604.26857
    
- [56] https://www.authorea.com/doi/pdf/10.22541/au.176348756.61222219/v1?download=true
    
- [58] https://arxiv.org/html/2603.11093v1
    
- [60] https://arxiv.org/html/2605.23163v1
    
- [61] https://arxiv.org/html/2511.00088v2
    
- [62] https://arxiv.org/html/2603.28888v1
    
- [63] https://www.mdpi.com/1424-8220/26/9/2870
    
- [64] https://arxiv.org/html/2602.18757v2
    
- [65] https://arxiv.org/html/2602.18757v3
    
- [66] https://cvpr.thecvf.com/virtual/2025/poster/35169
    
- [67] https://arxiv.org/html/2607.08072v2
    
- [69] https://ojs.aaai.org/index.php/AAAI/article/view/42463/46424
    
- [70] https://www.deeplearning.ai/the-batch/nvidias-alpamayo-r1-is-a-robotics-style-reasoning-model-for-autonomous-vehicles
    
- [71] https://introl.com/blog/nvidia-neurips-alpamayo-physical-ai-december-2025
    
- [75] https://arxiv.org/abs/2511.00088
    
- [77] https://openaccess.thecvf.com/content/CVPR2026/html/Peng_ColaVLA_Leveraging_Cognitive_Latent_Reasoning_for_Hierarchical_Parallel_Trajectory_Planning_CVPR_2026_paper.html
    
- [78] https://www.researchgate.net/publication/394606275_SOLVE_Synergy_of_Language-Vision_and_End-to-End_Networks_for_Autonomous_Driving
    
- [84] https://arxiv.org/html/2603.09086v1
    
- [86] https://www.researchgate.net/publication/403154910_Latent-WAM_Latent_World_Action_Modeling_for_End-to-End_Autonomous_Driving
    
- [89] https://arxiv.org/html/2603.24587v1
    
- [90] https://arxiv.org/abs/2605.15120
    
- [92] https://arxiv.org/html/2607.10975
    
- [96] https://www.nasdaq.com/press-release/weride-accelerates-global-expansion-its-end-end-intelligent-driving-solution-2026-0
    
- [98] https://www.quiverquant.com/news/WeRide+Begins+International+Road+Testing+of+One-Stage+End-to-End+Intelligent+Driving+Solution+with+Bosch
    
- [100] https://ir.weride.ai/news-releases/news-release-details/weride-accelerates-global-expansion-its-end-end-intelligent
    
- [101] https://www.finanznachrichten.de/nachrichten-2026-07/69007483-weride-inc-weride-accelerates-global-expansion-of-its-end-to-end-intelligent-driving-solution-399.htm
    
- [106] https://www.emergentmind.com/topics/alpamayo-r1-ar1
    
- [108] https://arxiv.org/html/2511.00088v1
    
- [110] https://introl.com/blog/nvidia-neurips-alpamayo-physical-ai-december-2025
    
- [111] https://chatpaper.com/paper/205905
    
- [112] https://openaccess.thecvf.com/content/CVPR2026/papers/Peng_ColaVLA_Leveraging_Cognitive_Latent_Reasoning_for_Hierarchical_Parallel_Trajectory_Planning_CVPR_2026_paper.pdf
    
- [120] https://arxiv.org/html/2505.15925v4
    
- [121] https://www.themoonlight.io/en/review/verdi-vlm-embedded-reasoning-for-autonomous-driving
    
- [124] https://arxiv.org/html/2605.15120v1
    
- [131] https://openreview.net/forum?id=7piDzw05kh
    
- [132] https://arxiv.org/html/2403.17373v1
    
- [133] https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_AIDE_An_Automatic_Data_Engine_for_Object_Detection_in_Autonomous_CVPR_2024_paper.pdf
    
- [136] https://arxiv.org/html/2603.24587v1
    
- [137] https://thinklab-sjtu.github.io/CornerCaseRepo/
    
- [138] https://ojs.aaai.org/index.php/AAAI/article/view/38149
    
- [141] https://arxiv.org/abs/2603.24587
    
- [142] https://arxiv.org/html/2606.26661v1
    
- [146] https://www.researchgate.net/figure/LAMP-improves-the-reliability-of-multimodal-trajectory-predictions-a-A-VQ-VAE-provides_fig1_408106176
    
- [148] https://arxiv.org/html/2605.23163v1
    
- [149] https://arxiv.org/abs/2605.23163
    
- [150] https://www.researchgate.net/scientific-contributions/Jin-Wang-2226451384
    
- [154] https://arxiv.org/html/2606.03909v1
    
- [155] https://cvpr.thecvf.com/virtual/2025/poster/35165
    
- [156] https://arxiv.org/html/2606.16278v2
    
- [158] https://research.zenseact.com/publications/splatad/
    
- [160] https://arxiv.org/html/2606.06014v1
    
- [161] https://www.researchgate.net/publication/406039432_PLAN-S_Bridging_Planning_with_Latent_Style_Dynamics_for_Autonomous_Driving_World_Models
    
- [166] https://arxiv.org/html/2603.14908v1
    
- [168] https://arxiv.org/abs/2603.14908
    
- [170] https://www.researchgate.net/publication/402848648_PerlAD_Towards_Enhanced_Closed-loop_End-to-end_Autonomous_Driving_with_Pseudo-simulation-based_Reinforcement_Learning
    
- [173] https://www.researchgate.net/publication/402631032_WorldRFT_Latent_World_Model_Planning_with_Reinforcement_Fine-Tuning_for_Autonomous_Driving
    
- [178] https://research.chalmers.se/publication/551971/file/551971_Fulltext.pdf
    
- [179] https://research.zenseact.com/publications/splatad/
    
- [184] https://www.jns.org/news/israel-news/uber-partners-with-israeli-autobrains-and-nvidia-on-munich-robotaxi-program
    
- [185] https://vision-mobility.de/en/news/uber-and-autobrains-want-to-test-robo-taxis-in-munich-391869.html
    
- [186] https://autobrains.ai/
    
- [189] https://www.springerprofessional.de/automated-driving/mobility-concepts/plans-for-a-robotaxi-program-in-munich/52826570