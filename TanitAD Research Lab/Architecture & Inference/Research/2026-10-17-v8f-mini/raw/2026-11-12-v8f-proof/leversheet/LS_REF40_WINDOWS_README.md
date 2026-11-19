# LS_REF40_WINDOWS.json : what the keys mean (written 2026-11-19, after the tick-map control of the other session)

The description string inside `LS_REF40_WINDOWS.json` (sha256 166198fb4cbd9da116689cc47380726af053a7dced5e91ca3c1db8448f41c0be) is wrong in two places. The file itself is left as it is, because its hash is cited in the pre-registration of the other session. The meaning, from the dump tool `imaghead/imag_dump.py`:

* `g0` is the CACHE step (bank step) of the read tick, not a raw frame index and not the window start. The header of the tool (line 3) says: window frame 4 = the read tick, `S0 = 4` of the 16-frame window, and window frame f is bank step `g0 - 4 + f` (also line 211).
* The head's present read uses the states of window frames 0 to 4 (`z[:, 0:5]`, line 153), that is cache steps `g0 - 4` to `g0`: a five-frame history that ends at `g0`.
* `gap_t` is the label gap at window frame 4 (`G15[sel, S0]`, line 272), that is at cache step `g0`, raw frame `2 * g0`.
* So the head read and its label are on the same frame. MEASURED by the other session from labels only: `gap_t` equals its label at raw frame `2 * g0` on 533 of the 534 windows (tolerance 1e-3 m).
* `t` is carried over from the window list of the latent reader and is NOT the read tick; on these windows it is 11 cache steps after `g0`. It is not used by the reference read.

Consequence: a head is scored against this reference at cache step `g0` (a history that ends at `g0` for a multi-frame head). The numbers of `LS_REF40.json` are aligned reads.
