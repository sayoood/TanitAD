# v8F-mini full corpus: deliverable manifest (2026-10-20)

| artifact | where it lives | sha256 / note |
|---|---|---|
| M1_s0 checkpoint | box 2 `D:\tanitad\runs\v8f_mini_full\M1_s0\ckpt_last.pt`; dev box `C:\Users\Admin\box2_ship\logs\v8f_mini\runs\M1_s0\ckpt_last.pt` | `f6c9c84f2261292a373173a8fe4035d32c606008538d0a3bd07869c9fc1c1015`, 62,703,765 B |
| M1_s1 checkpoint | same layout, `M1_s1` | `6733ccb8ea63325892eb969f5614284b60ca9c4c54fa8e37e866bcaabc5d08eb` |
| M2b_s0 checkpoint | same layout, `M2b_s0` | `601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8` |
| M2b_s1 checkpoint | same layout, `M2b_s1` | `6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6` |
| verdict | `raw/full/eval/verdict.json` (this repo) | `edd2cb44669fb06b4e077b465184dc3f20c5cc3bfe4cf672d070ad8fd83df894` |
| per-window eval outputs (pred, ade6, eid, ...) | `raw/full/eval/eval_*.npz`, `floors_perwindow.npz` (this repo) | every read in `RESULT_FULL.md` re-derives from these + the repo's eval inputs |
| run records | `raw/full/runs/<run>/` (argv, selfcheck, summary, train logs) | the argv differs between seeds only in `--seed` and `--out` |
| box-2 chain + guard logs | `raw/full/box2_logs/` | chain tip `6e5d3768`; boot unchanged since 2026-10-17 18:43:26 |

**Checkpoints are deliberately NOT in git.**
* They are 4 × 62.7 MB, and they sit on two machines (box 2 and the dev box).
* `D:\Projects\TanitAD-artifacts` was NOT used: D: has 19 GB free, below the stagers' 20 GiB floor (`D-SPACE`).
* Durable third copy: the PI's HF account (private) on request. Box 2 holds no HF token.
