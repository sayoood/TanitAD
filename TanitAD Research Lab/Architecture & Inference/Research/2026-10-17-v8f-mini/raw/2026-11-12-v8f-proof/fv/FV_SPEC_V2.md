# v8 proof programme: frozen verification conditions, version 2 (2026-11-13 ~01:05). Adds groups G and H; version 1 is unchanged.
Same recipe and trust base as FV_SPEC_V1.md (sha256 4e746f09...d3fe). Written before its proof script.

## Group G: ensembles and weight averages
* **VC-G1 (two members).** ((a+b)/2 - y)^2 = (a-y)^2/2 + (b-y)^2/2 - (a-b)^2/4.
* **VC-G2 (K members, sufficient statistics S1 = sum a_i, S2 = sum a_i^2).** K > 0: (S1/K - y)^2 = (S2 - 2*y*S1 + K*y^2)/K - (S2 - S1^2/K)/K.
* **VC-G3 (the ensemble is never worse than the member mean).** K > 0, S1^2 <= K*S2 (Cauchy-Schwarz, reduction lemma) ==> (S1/K - y)^2 <= (S2 - 2*y*S1 + K*y^2)/K.

## Group H: what a predictor and a camera can and cannot carry
* **VC-H1 (skill over persistence is the predictable fraction of the change).** D2 > 0 (mean squared change), C = G2 (the predictor g is the conditional mean of the change, so E[change*g] = E[g^2]) ==> D2 - 2*C + G2 = D2 - G2 and 1 - (D2 - G2)/D2 = G2/D2.
* **VC-H2 (AR(1) ceiling).** s2 > 0, -1 < rho < 1 ==> 1 - (s2*(1 - rho^2))/(2*s2*(1 - rho)) = (1 - rho)/2.
* **VC-H3 (mean pooling divides detection SNR^2 by P/k).** P >= 1, 0 < k <= P, s > 0 ==> k*k*s/P <= k*s and (k*k*s/P)/(k*s) = k/P.
* **VC-H4 (ground range per pixel, flat ground, level camera, row y = f*h/D below the horizon).** y > 0, f > 0, h > 0, D = f*h/y ==> f*h/y - f*h/(y+1) = D*D/(f*h + D).
* **VC-H5 (pixel span of a ground mark of length w at range D).** f, h, w, D > 0 ==> f*h/D - f*h/(D+w) = f*h*w/(D*(D+w)).
* **VC-H6 (everything beyond f*h/c lies in the one cell row of height c at the horizon).** 0 < y <= c, f > 0, h > 0, D = f*h/y ==> D >= f*h/c.

## Mutations (each must be REFUTED)
M-G1 last term (a-b)^2/2. M-G3 drop "S1^2 <= K*S2". M-H2 right side (1 - rho). M-H3 drop "k <= P". M-H4 right side D*D/(f*h). M-H6 drop "y <= c".
