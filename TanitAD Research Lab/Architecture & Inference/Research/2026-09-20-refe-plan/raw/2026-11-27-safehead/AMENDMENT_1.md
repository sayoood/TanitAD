# PREREG_SAFEHEAD, Amendment 1: a memory-only fix to `safehead_data.py` (one file, no semantic change)

MM, 2026-11-27 ~17:40Z. This amends `PREREG_SAFEHEAD.md` (sha256 `3e2c69b463cfd5954ad98b0d89f8c07e4ebd940fe76e040846856512200253d3`).

**State when this was frozen.**
- No SH training run has happened.
- No gate token exists.
- No SH number of any kind exists, navtest or held-out.
- The fix was found at the HANDOFF's first real-data step (`safehead_data.py check`), before seal, gate or training.

## What failed
`safehead_data.load_cache` → `take(key)` re-read the whole per-file array **once per row** and returned a VIEW into it. Every row therefore kept its whole file's array alive, an estimated 40+ GB peak on the 2,272-token navtrain cache.
- MEASURED by the launch agent: `check` died with `ArrayMemoryError` allocating 23 MiB while 12 GB of commit was free; it reproduced twice.
- The same pressure coincided with the dev-box navtest feature dump crashing (rc 0xC0000005), which was resumed.
- The synthetic tests could not see it, because their files are tiny.

## The change (this file only)
`take(key)` now:
- reads each array of a file **once**, through a per-file memo;
- returns `np.array(v[i])`, a COPY of the row, instead of a view.

The values are identical: same slice, same dtype conversion downstream. Only the memory lifetime changes. The diff is +9/−2 lines in `take()`, plus the memo initialisation (`run1/fix/safehead_data.memfix.diff`).

| file | sha256 before (card `CODE.sha256`) | sha256 after |
|---|---|---|
| `PK/refe/safehead_data.py` | `002c4fee03677fcd55a0bad059e19cc0db07d65b7e0da5505929e888b0aeddb5` | `0a3ec7985cc9a4ec2688d440a69b7ae760d3c99ffb78eed384de8c497114c23f` |

## Checks (MM, 2026-11-27; CPU, on a scratch copy of tip `5eb0ae09` with only this file replaced)

| check | result |
|---|---|
| `refe/test_safehead.py` (full) | rc 0, **ZZSAFEHEAD_PASS 14** (146 s) |
| `raw/2026-11-27-safehead/test_safehead_read.py` | rc 0, **ZZSHREAD_PASS 6** |
| `safehead_data.py check` on the REAL navtrain cache (2,272 tokens; index adapted to absolute paths by `run1/scripts/adapt_index.py`) | **ZZSH_CHECK_OK, leak []**, 1 s |

## What this does not touch
- No launch-gate profile pins `safehead_data.py`; `launch_gate.py` does not reference it. So no gate fingerprint moves.
- The 56-test regression (`regression/REGRESSION_VERDICT.md`) is not re-opened: the only importers are `safehead_gate.py`, `safehead_train.py`, `test_safehead.py` and the read, and all of their tests pass above.
- The read's CTRL-SHA checks the table, the proposals, the seam, the trained heads and the card, not code files. It is unchanged.
- The bars, arms, steps, loss, selection, controls and outcomes of the card are unchanged.
- `CODE.sha256` stays as frozen. `CODE_AMENDMENT_1.sha256` records the one new hash.
