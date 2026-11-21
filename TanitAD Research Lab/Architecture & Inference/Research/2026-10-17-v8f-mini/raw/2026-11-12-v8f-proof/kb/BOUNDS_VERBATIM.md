# Bounds quoted verbatim from banked primaries (text extracted with pypdf; symbols as extracted)

Evidence class: PUBLISHED (primary PDFs banked: MBPO is in the tip Library as 1906.08253; 1810.01257 and 1909.10618 are new in this hold).

## A. MBPO, Janner et al. 2019 (1906.08253), PDF page numbers
Definitions (p. 3-4): MDP (S,A,p,r,gamma,rho0), gamma in (0,1) discount. "generalization error ... epsilon_m = max_t E_{s~pi_D,t}[D_TV(p(s',r|s,a) || p_theta(s',r|s,a))]" (model error on the data-collecting policy pi_D); "maximum total-variation distance, max_s D_TV(pi || pi_D) <= epsilon_pi" (policy shift between iterations). rmax is used in the bound but its definition (max reward magnitude) is not stated in the main text; it is not defined in the passages read (UNSURE: not searched in Appendix B).

**Theorem 4.1 (p. 4).** Let the expected TV-distance between two transition distributions be bounded at each timestep by eps_m and the policy divergence be bounded by eps_pi. Then:
  eta[pi] >= eta_hat[pi] - [ 2 gamma rmax (eps_m + 2 eps_pi) / (1-gamma)^2 + 4 rmax eps_pi / (1-gamma) ]   (eq. 1; bracket = C(eps_m, eps_pi))
(eta = true return, eta_hat = return under the model). Proof: Appendix A, Theorem A.1 (p. 13, same statement).

**Theorem 4.2 (p. 4).** Given returns eta_branch[pi] from the k-branched rollout method:
  eta[pi] >= eta_branch[pi] - 2 rmax [ gamma^(k+1) eps_pi / (1-gamma)^2 + (gamma^k + 2) eps_pi / (1-gamma) + k (eps_m + 2 eps_pi) / (1-gamma) ]   (eq. 2)
Text (p. 4): "taken literally, this lower bound is maximized when k = 0, corresponding to not using the model at all."

**Theorem 4.3 (p. 5).** Under the k-branched rollout method, using model error under the updated policy eps_m' >= max_t E_{s~pi_D,t}[D_TV(p(s'|s,a) || p_hat(s'|s,a))]:
  eta[pi] >= eta_branch[pi] - 2 rmax [ gamma^(k+1) eps_pi / (1-gamma)^2 + gamma^k eps_pi / (1-gamma) + k eps_m' / (1-gamma) ]   (eq. 3)
Text (p. 5): this version "actually motivates model usage" (needs eps_m' as a function of eps_pi, measured empirically).

## B. Nachum et al. 2019 ICLR (1810.01257), "Near-Optimal Representation Learning for Hierarchical RL"
Definitions (p. 3): two-level policy, higher-level pi_hi picks goals g in G, Psi maps (s,g) to a low-level behaviour; "SubOpt(Psi) = sup_{s in S} V^{pi*}(s) - V^{pi*_hier}(s)" (eq. 2), pi* the optimal hierarchical policy agnostic to Psi, pi*_hier the optimal one acting through Psi; Rmax = sup_S |R(s)|; c = steps per high-level action.

**Theorem 1 (p. 3, c = 1, deterministic low level).** If there exists phi: S x A -> G such that sup_{s,a} D_TV( P(s'|s,a) || P(s'|s, Psi(s, phi(s,a))) ) <= eps (eq. 3), then SubOpt(Psi) <= C eps, where C = 2 gamma / (1-gamma)^2 * Rmax.

**Theorem 3 (p. 4, general c).** With eps_k(s_t,pi) = D_TV( P^pi(s_{t+k}|s_t) || P^{Psi(s_t,phi(s_t,pi))}(s_{t+k}|s_t) ), k in [1,c], weights w_k = 1 for k<c, w_k = (1-gamma)^-1 for k=c, w = sum_k gamma^(k-1) w_k: if sup (1/w) sum_{k=1..c} gamma^(k-1) w_k eps_k <= eps (eq. 7), then SubOpt(Psi) <= C eps, with C = 2 gamma / (1 - gamma^c) * Rmax * w (as extracted; the printed layout is "2γ/1−γc Rmax w").

**Claim 4 (p. 5, "the main theoretical contribution of our work").** If sup (1/w) sum_k gamma^(k-1) w_k D_KL( P^pi(s_{t+k}|s_t) || K(s_{t+k}|s_t,pi) ) <= eps^2/8 with K proportional to rho(s_{t+k}) exp(-D(f(s_{t+k}), phi(s_t,pi))) (eq. 8), and the low-level objective is Psi(s_t,g) = argmax_pi sum_k gamma^(k-1) w_k E_{P^pi(s_{t+k}|s_t)}[ -D(f(s_{t+k}),g) + log rho(s_{t+k}) - log P^pi(s_{t+k}|s_t) ] (eq. 9), then SubOpt(Psi) <= C eps.
(Claim 2, p. 4, is the c = 1 version.)

## C. Nachum et al. 2019 (1909.10618), Why Does Hierarchy (Sometimes) Work So Well in RL?
Paraphrase (abstract p. 1, introduction p. 2, overview p. 4): the study is empirical and gives no bound. It attributes most of the measured benefit of hierarchy to improved exploration, not to easier policy learning or to the imposed structure. With exploration controlled, hierarchical training helps only through multi-step rewards, which a non-hierarchical agent can also use. The authors limit the conclusion to the tasks they consider.
