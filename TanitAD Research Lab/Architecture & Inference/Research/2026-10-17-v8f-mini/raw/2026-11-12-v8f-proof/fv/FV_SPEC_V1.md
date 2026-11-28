# v8 proof programme: frozen verification conditions, version 1 (2026-11-13 ~00:10)

Written BEFORE any proof script exists. Recipe after Anthropic's formal-verification post (22 Oct 2026): the statement is fixed first, the solver is the only judge, the statement is never edited to make a proof pass (a changed statement gets a new id and a logged reason), and every condition has a mutated twin that the solver must REFUTE with a counterexample.
Checker: Z3 (real arithmetic, sound). Each VC is proved by showing its negation unsatisfiable.
Trust base, stated: Z3; and the REDUCTION lemmas named per VC (finite sums are linear; induction over the step count). A Lean proof would remove the reduction lemmas from the trust base; it is not run yet.
All variables are reals unless stated.

## Group A: compounding model error, flat against multi-level
* **VC-A1 (step).** L >= 0, e <= b, e' <= L*e + eps  ==>  e' <= L*b + eps.
* **VC-A2 (closed form).** L != 1, b = eps*(P-1)/(L-1)  ==>  L*b + eps = eps*(P*L-1)/(L-1); and P = 1 ==> b = 0. (P stands for L^t.)
  Reduction: induction on t gives e_t <= eps*(L^t - 1)/(L-1) for every rollout with one-step error <= eps and Lipschitz constant L.
* **VC-A3 (dominance criterion, expansive case).** L > 1, Q > 1, R > 1, eps > 0, epsk >= 0:
  epsk < eps*(Q-1)/(L-1)  <==>  epsk*(R-1)/(Q-1) < eps*(R-1)/(L-1). (Q = L^k, R = L^H.)
* **VC-A4 (dominance criterion, non-expansive case L = 1).** m > 0, k > 0, eps > 0, epsk >= 0: epsk < k*eps <==> epsk*m < eps*(m*k).
* **VC-A5 (geometric sum is monotone in its ratio, step).** 0 <= x <= y, 0 <= Sx <= Sy ==> x*Sx + 1 <= y*Sy + 1.

## Group B: mode averaging against a tactical decision
* **VC-B1 (within/selection identity).** S0 > 0: S2 - 2*c*S1 + c^2*S0 = (S2 - S1^2/S0) + S0*(S1/S0 - c)^2.
  Reduction: S0, S1, S2 are the weighted sums of 1, y, y^2 over one group, so the left side is sum w*(y-c)^2 and the right side is sum w*(y-mu)^2 + W*(mu-c)^2.
* **VC-B2 (the group mean is the unique minimiser).** S0 > 0, c != S1/S0 ==> S2 - 2*c*S1 + c^2*S0 > S2 - S1^2/S0.
* **VC-B3 (an averaged plan misses every mode).** 0 < p <= 1/2, D > 0, r >= 0, tol >= 0, p*D > tol + r; da <= r; Da = p*D; Db = (1-p)*D; triangle facts ya >= Da - da and yb >= Db - db with db <= r  ==>  ya > tol and yb > tol.
  (ya, yb: distance from an outcome in mode a, b to the mean plan; Da, Db: distance from the mean plan to the mode centres.)
* **VC-B4 (a decided plan hits its mode).** da <= r, r <= tol ==> da <= tol.

## Group C: rules by construction
* **VC-C1 (stop-line shield keeps its invariant for ANY proposal).** dt > 0, q > 0, v >= 0, B uninterpreted with B(0) = 0, B(x) >= 0 at the instances used, and (v > 0 ==> B(v) = v*dt + B(vb)) where vb = max(v - q, 0). Invariant B(v) <= d. Proposal u arbitrary real. uc = max(u, vb); d' = d - v*dt; v' = uc if B(uc) <= d' else vb.
  ==> B(v') <= d' and d' >= 0 and v' >= vb and v' >= 0.
* **VC-C2 (closed-form braking distance obeys the recursion).** Bcf(v, n) = dt*((n+1)*v - q*n*(n+1)/2). For n >= 1 and n*q <= v: Bcf(v, n) = v*dt + Bcf(v - q, n - 1). For n = 0: Bcf(v, 0) = v*dt.
* **VC-C3 (masked selection, bounded K = 1..6).** The loop "best = first admissible index with the highest score" satisfies: (i) some index admissible ==> best admissible; (ii) every admissible i has score[i] <= score[best]; (iii) if the unmasked first-argmax is admissible, best equals it.
* **VC-C4 (penalty bound).** lam > 0, V >= 0, J + lam*V <= J0, J >= Jinf ==> V <= (J0 - Jinf)/lam.
* **VC-C5 (minimal intervention of the envelope clip).** w = min(v, u): w <= u; and for every z <= u: |v - w| <= |v - z|.

## Group D: imagination
* **VC-D1 (imagined gain above twice the gap is a real gain).** G >= 0, |J1 - I1| <= G, |J0 - I0| <= G, I1 - I0 > 2*G ==> J1 > J0.
* **VC-D2 (selection among candidates).** |J1 - I1| <= G, |J2 - I2| <= G, I1 <= I2 (candidate 1 is chosen as the lower imagined cost) ==> J1 <= J2 + 2*G.

## Group E: data selection
* **VC-E1 (Engel step).** x > 0, y > 0: a^2/x + b^2/y >= (a+b)^2/(x+y).
  Reduction: induction gives sum g_i^2/p_i >= (sum g_i)^2 / sum p_i, so sampling proportional to g_i minimises the second moment of the importance-weighted gradient, and allocation proportional to W_k*sigma_k minimises a stratified variance.
* **VC-E2 (equality case).** x > 0, y > 0, a*y = b*x ==> a^2/x + b^2/y = (a+b)^2/(x+y).
* **VC-E3 (label-shift threshold).** 0 < pb < 1, 0 < pi < 1, 0 < pib < 1; odds = (pb/(1-pb)) * (pi/(1-pi)) / (pib/(1-pib)); p = odds/(1+odds); r = (pib*(1-pi))/((1-pib)*pi); tau = r/(1+r)  ==>  (p >= 1/2 <==> pb >= tau).

## Group F: weight edits
* **VC-F1 (rank-one edit is the least change, 2x2 case).** k = (k1,k2) != 0, any matrix Dm with Dm*k = dl: ||Dm||_F^2 * (k1^2+k2^2) >= dl1^2 + dl2^2.
  Reduction: row-wise Cauchy-Schwarz gives every size; whitening by C^(1/2) gives the covariance-weighted form.
* **VC-F2 (the rank-one edit attains it).** Dm = dl*k^T/(k.k) satisfies Dm*k = dl and ||Dm||_F^2 * (k.k) = dl.dl.
* **VC-F3 (interpolating two models interpolates a linear readout).** For a head linear in its weights, y(alpha) = (1-alpha)*y0 + alpha*y1 when the weights are (1-alpha)*w0 + alpha*w1 (scalar instance: (1-al)*w0*x + al*w1*x = ((1-al)*w0 + al*w1)*x).

## Mutations (each must be REFUTED: the solver returns a counterexample)
M-A1 drop "L >= 0". M-A3 replace "<" by "<=" on one side only. M-B2 drop "c != S1/S0". M-B3 replace "p*D > tol + r" by "p*D > tol". M-C1a no braking fallback (v' = uc always). M-C1b check B(uc) <= d instead of d'. M-C3 drop the admissibility test in the loop. M-C4 drop "V >= 0"... kept as is: drop "J >= Jinf". M-D1 "I1 - I0 > G". M-E1 drop "y > 0". M-E3 tau := 1/2. M-F1 drop "Dm*k = dl".
