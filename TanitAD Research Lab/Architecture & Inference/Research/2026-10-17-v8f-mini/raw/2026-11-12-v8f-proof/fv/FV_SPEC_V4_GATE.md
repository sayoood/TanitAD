# v8 proof programme: frozen verification conditions, version 4 (2026-11-13 ~02:45). Adds group J (commit rules); versions 1 to 3 are unchanged.
Same recipe and trust base as FV_SPEC_V1.md (sha256 4e746f09...d3fe). Written before its proof script. Written AFTER the GATEMIX read (addendum 8): group J is an explanatory theorem for a measured result, not a prediction of it.

## Setting
Two window classes: selection-critical (share p, 0 < p < 1) and bulk (share 1 - p). The per-window error of the hierarchy minus the error of the flat arm has class means dc (critical) and db (bulk). A commit rule lets the hierarchy drive a flagged window and the flat arm drive the rest; the flag fires on a share t of the critical windows and a share f of the bulk windows (0 <= t, f <= 1). Premise A-J (homogeneity): inside each class the flag is independent of the per-window error difference. Then
* DF = composite minus flat = p*t*dc + (1-p)*f*db,
* DH = composite minus hierarchy = -p*(1-t)*dc - (1-p)*(1-f)*db.
The ratio s = (1-p)*db / (p*(-dc)) is the bulk cost per unit of critical gain.

## Group J
* **VC-J1 (accounting identity).** DF - DH = p*dc + (1-p)*db (the plain hierarchy minus flat difference), for every t and f.
* **VC-J2 (beating the flat arm).** dc < 0, db > 0: DF < 0 if and only if t*p*(-dc) > f*(1-p)*db (that is, t > s*f).
* **VC-J3 (beating the hierarchy).** dc < 0, db > 0: DH < 0 if and only if (1-t)*p*(-dc) < (1-f)*(1-p)*db (that is, 1 - t < s*(1 - f)).
* **VC-J4 (the class oracle beats both).** dc < 0, db > 0, t = 1, f = 0: DF < 0 and DH < 0.
* **VC-J5 (a flag without information cannot beat both).** t = f: not (DF < 0 and DH < 0).
* **VC-J6 (beating both needs information).** dc < 0, db > 0, DF < 0, DH < 0: t > f.
* **VC-J7 (no commit rule is needed when one arm wins both classes).** dc < 0, db < 0: DH >= 0 for every t and f (the composite never beats the hierarchy), with equality at t = f = 1.
* **VC-J8 (under A-J an informative flag is never worse than both arms).** dc < 0, db > 0, t >= f: not (DF > 0 and DH > 0).

## Mutations (each must be REFUTED by a counterexample)
* **M-J2 (drop the base rates).** dc < 0, db > 0: DF < 0 if and only if t*(-dc) > f*db.
* **M-J6 (information is sufficient).** dc < 0, db > 0, t > f: DF < 0 and DH < 0.
* **M-J5 (a flag without information ties the better arm).** t = f: DF <= 0.
* **M-J8 (drop the information premise).** dc < 0, db > 0: not (DF > 0 and DH > 0).
