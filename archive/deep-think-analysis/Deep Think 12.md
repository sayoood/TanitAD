

To operationalize **Hypothesis 13** for the TanitAD stack, we must resolve a fundamental dilemma in End-to-End (E2E) autonomous driving: **Interpretability vs. Policy Performance**.

If an E2E network is forced to optimize for human visual priors (like drawing perfect lane lines or 3D bounding boxes), it suffers from "representation dilution," wasting capacity on task-irrelevant background details and degrading raw driving control.

To achieve transparent explainability with **zero degradation** to the primary driving loop, the Extraction Heads must be designed as mathematically isolated **"Read-Only Probes."** Below is the exact architectural blueprint, the stop-gradient mechanics, the multi-task loss formulation, and the design of the Behavioral Extraction Head.

### 1. The Information Firewall: Architecture & Stop-Gradients

We bifurcate the computational graph immediately after the main spatiotemporal encoder, creating two parallel flows: the **Primary Control Loop** and the **Explainability Sub-Network**.

1. **The Primary Loop:** The backbone processes sensor inputs into a highly compressed, abstract latent tensor, $\mathbf{Z}_{tactical}$. The driving policy maps this directly to an action: $a^* = \pi(\mathbf{Z}_{tactical})$.
    
2. **The Stop-Gradient Junction:** We duplicate the latent space and mathematically sever the computational graph:
    
    $$ \mathbf{Z}_{exp} = \text{stop_gradient}(\mathbf{Z}_{tactical}) $$
    
    _(In PyTorch: `Z_exp = Z_tactical.detach()`; in JAX: `jax.lax.stop_gradient(Z_tactical)`)_
    
3. **Shared Explainability Adapter (SEA):** Because $\mathbf{Z}_{tactical}$ is optimized for pure control, its features are highly non-linear. $\mathbf{Z}_{exp}$ is passed through a lightweight SEA (e.g., a 3-layer Transformer Encoder or ConvNeXt block) to realign these abstract features into a semantic spatial format, $F_{exp}$, suitable for human-interpretable decoding.
    
4. **Extraction Heads:** $F_{exp}$ branches into the specific heads ($H_{BEV}$, $H_{Obj}$, $H_{Behav}$).
    

**The Mathematical Guarantee:** During backpropagation, the extraction loss $\nabla \mathcal{L}_{exp}$ flows backward until it hits the `stop_gradient` operator, where the derivative is precisely annihilated ($\frac{\partial \mathbf{Z}_{exp}}{\partial \mathbf{Z}_{tactical}} = \mathbf{0}$). The Extraction Heads are forced to decode information _already present_ in the driving latents. If the BEV head fails to render a pedestrian, it definitively proves the Tactical Brain did not perceive it—establishing a mathematically truthful audit rather than a hallucinated reconstruction.

### 2. Multi-Task Loss Weighting Strategy

Because the Extraction Heads share the SEA, their gradients will collide. Static weighting fails because BEV semantic segmentation (dense pixel-wise Cross-Entropy) operates at a massively different scale and variance than sparse Object Bipartite Matching or Trajectory L2 losses. A dense loss would dominate the SEA, starving the Behavioral Head of useful features.

We formulate the loss using **Homoscedastic Task Uncertainty Weighting** (Kendall et al.), which allows the network to dynamically learn optimal loss weights based on the inherent noise of each extraction task.

Let $s_k = \log(\sigma_k^2)$ be a learnable log-variance parameter for each task $k \in \{BEV, Obj, Behav\}$. The joint extraction loss is formulated as:

$$ \mathcal{L}_{Extract} = \sum_{k} \left( \frac{1}{2}\exp(-s_k) \mathcal{L}_k + \frac{1}{2}s_k \right) $$

**Mechanism:** If the BEV reconstruction is highly noisy early in training, the network autonomously increases $s_{BEV}$, which dynamically dampens its gradient contribution ($\exp(-s_{BEV})$). This prevents it from washing out the SEA, ensuring the Behavioral Head converges cleanly. The $+\frac{1}{2}s_k$ term acts as a regularizer to prevent the network from pushing all weights to zero.

### 3. The Behavioral Extraction Head: Counterfactual Query Decoder (CQD)

The most complex requirement is extracting the maneuvers the Tactical Brain **considered but rejected**. Because the primary policy outputs a single deterministic action, the "rejected actions" are hidden in the implicit topological affordances of the latent space.

To extract these, we design the Behavioral Head as a DETR-style Set-Prediction Transformer called the **Counterfactual Query Decoder (CQD)**.

#### A. Neural Architecture

1. **Intention Queries ($Q$):** We initialize $N$ learnable embedding vectors (e.g., $N=8$). Each query represents a prototypical tactical intent (e.g., _Q1: Maintain Lane, Q2: Overtake Left, Q3: Yield, Q4: Hard Brake_).
    
2. **Cross-Attention Decoding:** The $N$ queries act as the Query vectors, while the flattened semantic latents $F_{exp}$ act as the Keys and Values. Each query scours the environment latents to extract spatiotemporal evidence supporting its specific hypothetical maneuver.
    
3. **Tri-Branch Prediction MLP:** Each updated query $q_i$ passes through three parallel MLPs:
    
    - **Trajectory Branch ($\hat{\tau}_i$):** Predicts continuous future waypoints $(x, y, \theta)$ over $T$ seconds.
        
    - **Viability Probability ($P_{valid}^i$):** A Sigmoid output $[0,1]$ estimating whether the latent space views this maneuver as kinematically feasible/valid.
        
    - **Rejection Rationale ($R_{cost}^i$):** A Softmax distribution over $C$ semantic rejection classes (e.g., _0: Safe/Selected, 1: Static Collision Risk, 2: Dynamic Collision Risk, 3: Kinematic Limit, 4: Comfort Penalty_).
        

#### B. Isolating the "Rejected" Counterfactuals (Inference Logic)

During live driving, the primary loop executes action $a^*$. In parallel, the CQD generates 8 fully formed maneuvers.

1. **Identify the Selected Maneuver:** We compute the Fréchet distance between the 8 predicted trajectories and $a^*$. The closest match is tagged as the "Selected Maneuver."
    
2. **Filter the Search Tree:** For the remaining 7 queries, we filter out logically absurd actions by applying a threshold: $P_{valid}^i > 0.15$.
    
3. **Extract Counterfactuals:** The surviving queries are the maneuvers the AI modeled as valid affordances but ultimately rejected.
    
4. **UI Translation:** The system reads the Rationale Branch to generate human text:
    
    > _"Action Executed: Maintain Lane. Considered but Rejected: Overtake Left. Reason: Dynamic Collision Risk (Cross-referencing Object ID 14 in left blind spot)."_
    

#### C. Training the CQD via Expert Distillation

Standard driving datasets only contain the human's _chosen_ action, meaning we lack ground-truth labels for "rejected" maneuvers.

To train the CQD, we use an **Offline Privileged Expert** (e.g., a Monte Carlo Tree Search planner running with simulator access in hindsight). For every training scene, the Expert evaluates all valid topological maneuvers and generates ground-truth alternative trajectories, viability scores, and calculated costs.

We apply the **Hungarian Algorithm (Bipartite Matching)** to map the CQD's $N$ queries to the Offline Expert's search tree, supervising it via L1/Huber loss for trajectories and Cross-Entropy for the Rationale branch.