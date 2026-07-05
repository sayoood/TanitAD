


To resolve the fundamental conflict between the probabilistic smoothness of statistical E2E driving models and the deterministic, absolute boundaries of legal traffic rules, we must bridge continuous generative probability spaces with strict control-theory constraints. Relying on explicit rule-based heuristic overrides (e.g., PID controllers) destroys the End-to-End philosophy, introduces latency, and creates brittle edge-cases. Post-training alignment allows us to structurally embed deterministic boundaries into the model's continuous generation manifold.

Here is the exploration of the three hypotheses, the architectural selection for the 4B Tactical Layer, and the definitive mathematical pipeline for implementation.

### Part 1: Exploration of Algorithmic Hypotheses

#### 1. Trajectory Direct Preference Optimization (T-DPO)

- **Mechanism**: Adapted from LLM alignment techniques, T-DPO works by curating pairs of generated continuous trajectories: a winning trajectory $\mathbf{s}_{win}$ (smooth, strictly stops at the line) and a losing trajectory $\mathbf{s}_{lose}$ (smooth, but slowly rolls the stop line). The policy is fine-tuned to maximize the Bradley-Terry log-probability margin between the two.
    
- **The Strictness Flaw**: DPO optimizes a soft, contrastive statistical objective bounded by a KL-divergence constraint. It effectively shifts probability mass (e.g., reducing the likelihood of running a red light from $5\%$ to $0.05\%$). However, it mathematically preserves the distribution's support, meaning it leaves a **heavy probabilistic tail**. In out-of-distribution (OOD) scenarios, the model will inevitably sample an illegal action. Absolute legal rules require an infinite negative log-likelihood at the boundary, which DPO cannot provide.
    

#### 2. Latent Space Constraint Projection (LSCP)

- **Mechanism**: Absolute traffic rules are codified as differentiable constraints (e.g., Control Barrier Functions or CBFs). An implicit, differentiable convex optimization layer (e.g., OptNet or CvxpyLayers) is inserted into the 4B model’s architecture. If a proposed latent representation $z$ breaches a rule boundary, it is orthogonally projected onto the nearest valid point on the "safe" latent manifold $z_{safe}$ before decoding.
    
- **The Smoothness Flaw**: The latent space of a massive 4B-parameter model is highly entangled and topologically non-isomorphic to physical Cartesian space. A hard geometric projection in latent space induces extreme non-linear discontinuities. When decoded, these projected vectors manifest as jerky, dynamically infeasible trajectories (e.g., infinite jerk, physically impossible steering gradients)—triggering **catastrophic forgetting of the pre-trained kinematic smoothness**. Furthermore, implicit solvers violate the strict sub-50ms real-time latency budget of the Tactical Layer.
    

#### 3. Reward-Modulated Flow Matching (RMFM)

- **Mechanism**: Assumes the Tactical Layer models trajectory generation as a Probability Flow ODE (Continuous Normalizing Flows / Flow Matching), mapping Gaussian noise to driving paths. Traffic rules are formulated offline as highly stiff, differentiable Interior Point Log-Barriers (an energy landscape). The model's continuous vector field is fine-tuned to route the probability flow _around_ these energy barriers, while heavily regularized by the pre-trained smooth vector field.
    
- **The Synergistic Success**: It successfully unifies both requirements. Because generation remains an unbroken ODE integration, the output is mathematically guaranteed to be Lipschitz-continuous (kinematically smooth). Simultaneously, the steep barrier gradients structurally warp the vector field to eliminate probability mass in illegal states, natively achieving strict absolute bounds without runtime projection.
    

### Part 2: Selection for the 4B Architecture's Tactical Layer

**Winner: Reward-Modulated Flow Matching (RMFM)**

**Justification**:

A 4B-parameter Tactical Layer operating in an autonomous vehicle must guarantee dynamic feasibility and adhere to ultra-low latency constraints. T-DPO fails because its probabilistic tails permit unacceptable legal infractions. LSCP fails because it destroys kinematic smoothness and adds hundreds of milliseconds of implicit solver latency.

RMFM is the optimal choice because it compiles the strict logical constraints directly into the neural weights _offline_. By geometrically warping the vector field, the continuous physics of the ODE seamlessly yields right-of-way and respects stop lines. At inference time, the model executes pure neural logic (solving a fast ODE) with zero brittle fallback code and zero rule-checking latency overhead.

### Part 3: The Exact Mathematical Pipeline for RMFM Alignment

To deploy RMFM without exceeding VRAM limits (the OOM problem caused by standard backpropagation through a 4B model's ODE solver), we construct a pipeline combining Exterior Penalty Functions, Flow-Matching Regularization, and the Neural ODE Continuous Adjoint Method.

#### Step 1: Pre-Trained Continuous Prior (Base State)

Let the required physical driving trajectory be $\mathbf{s} \in \mathbb{R}^{T \times D}$ conditioned on multimodal context $\mathbf{c}$.

The pre-trained 4B model defines a smooth vector field $v_\phi(\mathbf{z}_\tau, \tau; \mathbf{c})$ mapping noise $\mathbf{z}_0 \sim \mathcal{N}(0, I)$ to expert human data $\mathbf{s}_1$ over flow time $\tau \in [0, 1]$:

$$ \mathbf{s}_1 = \mathbf{z}_0 + \int_0^1 v_\phi(\mathbf{z}_\tau, \tau; \mathbf{c}) , d\tau $$

#### Step 2: Differentiable Strict Rule Formulation (The Barrier Oracle)

We translate binary traffic laws into differentiable continuous barriers.

Let $h_k(\mathbf{s}_1, \mathbf{c}) \ge 0$ define a strict legal constraint (e.g., $h_{stop} = d_{stop\_line} - v_{ego} \cdot t_{brake} \ge 0$).

We define a highly stiff Penalty Function $J_{rules}(\mathbf{s}_1, \mathbf{c})$:

$$ J_{rules}(\mathbf{s}_1, \mathbf{c}) = \sum_{k} \mu_k \exp\big(-\gamma_k \cdot h_k(\mathbf{s}_1, \mathbf{c})\big) $$

_(As the generated trajectory approaches a legal violation, the penalty explodes exponentially, creating an infinitely steep repulsive gradient)._

#### Step 3: The Alignment Objective

We initialize our aligned model parameters $\theta \leftarrow \phi$. We seek to update $\theta$ to minimize rule violations while explicitly preventing the catastrophic forgetting of smooth kinematics via a Flow-Matching $L_2$ penalty. The objective is to minimize:

$$ \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{c}, \mathbf{z}_0} \Bigg[ J_{rules}(\mathbf{s}_1, \mathbf{c}) + \lambda \int_0^1 \Big| v_\theta(\mathbf{z}_\tau, \tau; \mathbf{c}) - \text{sg}\big[v_\phi(\mathbf{z}_\tau, \tau; \mathbf{c})\big] \Big|^2 d\tau \Bigg] $$

- **Term 1**: Strict geometric constraint enforcement at the boundary.
    
- **Term 2**: The anti-forgetting anchor ($\text{sg}$ is the stop-gradient). It rigorously forces the new vector field to mimic the pre-trained kinematic prior _unless_ absolute deviation is necessary to survive the barrier of Term 1.
    

#### Step 4: Optimization via Continuous Adjoint Sensitivities

We use the Continuous Adjoint Method to compute exact parameter gradients for the 4B model with $\mathcal{O}(1)$ memory.

Let the instantaneous flow penalty be $l(\mathbf{z}_\tau, \theta, \tau) = \lambda \| v_\theta(\mathbf{z}_\tau, \tau; \mathbf{c}) - v_\phi(\mathbf{z}_\tau, \tau; \mathbf{c}) \|^2$.

We define the adjoint state (the sensitivity of the loss to the trajectory flow path):

$$ \mathbf{a}(\tau) = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_\tau} $$

**1. Terminal Condition (The Barrier Force evaluated at $\tau=1$):**

$$ \mathbf{a}(1) = \nabla_{\mathbf{s}_1} J_{rules}(\mathbf{s}_1, \mathbf{c}) $$

**2. Adjoint Backward ODE (Integrated from $\tau=1$ to $0$):**

$$ \frac{d\mathbf{a}(\tau)}{d\tau} = - \left( \frac{\partial v_\theta}{\partial \mathbf{z}_\tau} \right)^\top \mathbf{a}(\tau) - \nabla_{\mathbf{z}_\tau} l(\mathbf{z}_\tau, \theta, \tau) $$

**3. Parameter Gradient Computation:**

Concurrently, we accumulate gradients for the Tactical Layer's weights $\theta$ (or LoRA adapters):

$$ \frac{d\mathcal{L}}{d\theta} = \int_0^1 \left[ \left( \frac{\partial v_\theta}{\partial \theta} \right)^\top \mathbf{a}(\tau) + \nabla_\theta l(\mathbf{z}_\tau, \theta, \tau) \right] d\tau $$

We update $\theta$ using AdamW. This permanently "bends" the vector field to repel rule violations without requiring millions of probabilistic samples.

#### Step 5: Zero-Shot Deployment Inference

At runtime inside the TanitAD inference stack, the computationally heavy Adjoint states, Barrier Functions, and Rule Oracles are entirely stripped away. The system continuously ingests context $\mathbf{c}$, samples noise, and utilizes a fast low-step ODE solver (e.g., DPM-Solver or a 3-step Heun's method) on the aligned vector field:

$$ \mathbf{s}_{drive} = \mathbf{z}_0 + \int_0^1 v_\theta(\mathbf{z}_\tau, \tau; \mathbf{c}) , d\tau $$

**Conclusion:** If the vehicle's context indicates a red light, the aligned neural vector field natively repels the probability flow away from continued acceleration. The ODE seamlessly bends into a smooth deceleration curve, stopping exactly at $v=0$ at the stop-line. This accomplishes the holy grail of E2E AV architectures: hard mathematical rule compliance generated entirely by continuous neural kinematics.