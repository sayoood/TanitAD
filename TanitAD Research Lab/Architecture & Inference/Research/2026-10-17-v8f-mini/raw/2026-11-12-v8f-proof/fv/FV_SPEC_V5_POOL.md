# v8 proof programme: frozen verification conditions, version 5 (2026-11-13 13:15). Adds group P (what a mean over cells keeps and loses); versions 1 to 4 are unchanged.
Same recipe and trust base as FV_SPEC_V1.md. Written before its proof script.
Written AFTER the LATVIS read (addendum 10) and the planted-object control f5 of FULLRES (addendum 9): for those two results group P is an explanatory theorem, not a prediction.
Written BEFORE control s2 of SPATIAL stage 0 and control c3 of SPATIAL stage 1 (addendum 12): for those two it is a prediction (s2: the planted object is read from the grid state and not from the mean state; c3: a cell permutation leaves the mean state equal and changes the grid state).

## Setting
A frame is n = 4 cells with one real feature per cell, x1 .. x4 (the statements are per coordinate; the step to n cells and to vector features is linearity and stays a paper step).
The adapter of v8F acts per cell and is affine: a(x) = w*x + b (`v8f_mini.py:694`: LayerNorm and Linear per token; the position term is a constant per cell).
* Mean state: m = (a(x1) + a(x2) + a(x3) + a(x4)) / 4 (the present level-1 state before its projection, `WMAux.z1`).
* Grid state: G = (a(x1), a(x2), a(x3), a(x4)) (addendum 12).
* Block state: B = ((a(x1) + a(x2))/2, (a(x3) + a(x4))/2) (pooling at an intermediate scale).
An object of strength s on a background u1 .. u4: scene A has it in cell 1 (x = (u1 + s, u2, u3, u4)), scene B in cell 2, scene C in cell 3.
A decoder is any function of the state; on two scenes with equal state it returns one value g.

## Group P
* **VC-P1 (pair bound).** For all reals y1, y2, g: (y1 - g)^2 + (y2 - g)^2 >= (y1 - y2)^2 / 2. (Two scenes with equal state and different targets cost every decoder at least half the squared target difference.)
* **VC-P2 (the pair bound is attained).** g = (y1 + y2)/2: (y1 - g)^2 + (y2 - g)^2 = (y1 - y2)^2 / 2.
* **VC-P3 (moving the object leaves the mean state equal).** For all u1 .. u4, s, w, b: m(A) = m(B).
* **VC-P4 (location is unreadable from the mean state: zero skill).** Target y(A) = 1, y(B) = 0, both scenes equally likely, one decoded value g: the risk ((1 - g)^2 + g^2)/2 >= 1/4, the variance of the target; so the skill 1 - risk/variance is at most 0 for every decoder.
* **VC-P5 (the same for four positions).** Object in one of 4 cells with equal probability, target = indicator of cell 1, one decoded value g: the risk ((1 - g)^2 + 3*g^2)/4 >= 3/16, the variance of the target.
* **VC-P6 (location is linearly readable from the grid state, for every background).** w != 0, s != 0: with the read r(G) = G1 - G2, r(G(A)) - r(G(B)) = 2*w*s, which is not 0 and does not depend on u1 .. u4 or b.
* **VC-P7 (the mean state keeps the count).** Scenes with k1 and k2 objects of the same strength s on the same background: m1 - m2 = (k1 - k2)*w*s/4; with w != 0, s != 0, k1 != k2 this is not 0.
* **VC-P8 (object motion is invisible to the mean state and visible to the grid state).** The object moves from cell 1 at time t to cell 2 at time t+1 on a constant background: m(t+1) - m(t) = 0 (a copy predicts the mean state exactly, so nothing is left to imagine), while G1(t+1) - G1(t) = -w*s and G2(t+1) - G2(t) = +w*s.
* **VC-P9 (pooling at an intermediate scale loses location inside a block and keeps it across blocks).** B(A) = B(B) in both coordinates; with w != 0, s != 0: B1(A) - B1(C) = w*s/2, which is not 0.

## Mutations (each must be REFUTED by a counterexample)
* **M-P1 (a stronger pair bound).** (y1 - g)^2 + (y2 - g)^2 >= (y1 - y2)^2.
* **M-P4 (a risk floor above the variance).** ((1 - g)^2 + g^2)/2 >= 26/100.
* **M-P5 (the two-scene floor for four positions).** ((1 - g)^2 + 3*g^2)/4 >= 1/4.
* **M-P6 (readable without a live adapter weight).** s != 0 (w free): r(G(A)) - r(G(B)) != 0.
* **M-P7 (the mean state separates any two counts).** k1 != k2 with strengths s1, s2 free and not 0, w != 0: (k1*s1 - k2*s2)*w/4 != 0.
* **M-P8 (the mean state sees the motion).** w != 0, s != 0: m(t+1) - m(t) != 0.
* **M-P9 (block pooling keeps location inside a block).** w != 0, s != 0: B1(A) - B1(B) != 0.

## Reading for v8F
VC-P3, P4, P8: the present level-1 state cannot carry where an object is, and a moving object changes nothing in it; LATVIS (scene gist kept, no object, the learned rollout not closer than a copy) and the f5 control of FULLRES (AUROC 1.0000 before pooling, 0.5030 after) are measurements of this. VC-P7: it keeps counts and totals (light level, how much of the frame is occupied). VC-P6, P8: a state that keeps the cell layout carries location and gives the predictor a change to predict. VC-P9: the same loss occurs inside each 64x64 pixel cell of the 4x10 grid for objects smaller than a cell.
