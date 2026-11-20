# PREREG_V8F_PROOF_ADD92, note N1: the clause of control h3 on the sign of the planted gap

Written on 2026-11-20 after the first self-test of the tool on a planted world (dev box, no logged frame read) and before any run of the tool on data.

What the self-test showed: control h3 of the card asks that every planted gap is above 0. On the planted world the absolute level of 4 m brings some planted gaps below 0 where the label gap is short (a label gap of 2 m with a noise value of -1 or lower), so h3 turned red on a healthy tool. The same has to be expected on the logged frames. The clause was meant as a check of the sign of the proportional form, where a planted gap at or below 0 can only come from a wrong scale, and it has no meaning for an absolute error, which is planted without a clip at 0 as in card ADD90.

The reading fixed by this note, with no change of an arm, a level, a name or another control:

* h3, proportional levels: as in the card (the standard deviation of planted over label gap minus 1 is q within 1e-9, its mean is 0 within 1e-9, and every planted gap is above 0).
* h3, absolute levels: the standard deviation of planted minus label gap is s within 1e-9 and its mean is 0 within 1e-9; the number of planted values at or below 0 is recorded per level and reported with no name, and is not judged.
* The label gap itself has to be above 0 on every frame that has one, as the proportional form needs.

No value of the card was known when this note was written.
