# alpha-0 / alpha-1 launch argv pair, and the optional alpha-1s (DRAFT -- not launched)

Derived mechanically by `code/make_argv_pair.py` from `stack/ops/runs.d/refcv8-r101-s0.argv.json` (210 tokens). The JSON files
`argv_v8f-a0-s0.json` / `argv_v8f-a1-s0.json` / `argv_v8f-a1s-s0.json` beside this note are the exact lists; `argv_pair_check.json`
records the diffs, the argv sha256 values and the gate's argv rules under all three profiles.

| | alpha-0 (`v8f-a0-s0`) | **alpha-1 (`v8f-a1-s0`)** | alpha-1s (`v8f-a1s-s0`, OPTIONAL) |
|---|---|---|---|
| gate profile | `refcv8` (its recipe) | **`v8f`** | `v8f_steer` |
| `--residual-prior` | `ha0_ext_pose` | **`p10_pose`** (past poses only) | `p10` (reads the recorded steer) |
| `--p10-prior` | -- | `/home/nvidia/v8f_run/launch_tree/stack/tanitad/configs/v8f_p10/p10_pose_prior.pt` | `.../launch_tree/stack/tanitad/configs/v8f_p10/p10_prior.pt` |
| `--p10-prior-sha256` | -- | `e70a9ae30b4418705442bffdeb62d9810a43b0a6ccce0c177796c72295fe693d` | `af5a29ddadb4c7d98f1db0d8116bddd0634e83a7e687fc601989d3fda984d7b3` |
| `--out` | `/home/nvidia/v8f_run/runs/v8f-a0-s0` | `/home/nvidia/v8f_run/runs/v8f-a1-s0` | `/home/nvidia/v8f_run/runs/v8f-a1s-s0` |
| every other token | the refcv8 launch's | **identical** | identical |
| tokens | 210 (argv sha256 `2a88519851719db5...`) | 214 (`17b9e5199ac290bf...`) | 214 (`5af855f7afb46b9d...`) |
| input set at inference | past poses | past poses (**unchanged**) | past poses **+ the recorded steer** (PI ruling pending) |

MEASURED (`argv_pair_check.json`, `tests/test_launch_gate_v8f.py`): alpha-0 under `refcv8` -> 0 refusals; **alpha-1 under `v8f` -> 0
refusals**; alpha-1s under `v8f_steer` -> 0 refusals. Every other combination is refused by name: alpha-0 under `v8f` -> 5 refusals
(the missing P10 flags and `--residual-prior ['ha0_ext_pose'] is REFUSED`); alpha-1 under `refcv8` -> 1 (`--residual-prior
ha0_ext_pose is REQUIRED`); **alpha-1s under `v8f` -> 3, the first reading "`--residual-prior ['p10'] is REFUSED for a v8f launch
... THIS value: steer at inference: no PI ruling (SPEC_REFCV7 A5)`"**; alpha-1 under `v8f_steer` -> 3 (the sibling is refused with its
own reason). Each profile refuses the other arm's prior by name.

Things in this pair that are **placeholders or decisions, not facts**:

1. **Geometry (D2, Master Mind, open item `V8F-RUNG-GEOMETRY` of both profiles).** `--steps 48852 --batch 16 --eval-batches 8
   --eval-every 500` and the B1 guard are refcv8's FULL-RUN values, carried so the pair is runnable as written. PLAN_V8F section 6
   sizes rung alpha as a TINY rung (batch 8, ~4.5 h per arm, seeds 0 and 1). The Master Mind sets the real values at launch from
   refcv8's measured step time and the gate token is issued then: set the same values in **both** arms and write the six literals into
   `required_values`. Seed 1 is `--seed 1` and `--out .../v8f-aN-s1`; nothing else changes. **The token cannot PASS before this.**
2. **The P10 path (D5, closed).** The artifact ships INSIDE the repo (`stack/tanitad/configs/v8f_p10/`), banked and sha-pinned; the Thor
   checkout (`launch_tree`) references it repo-relative. The **sha256 is the identity** and the gate checks it (a literal in the profile's
   `required_values`); the path is the tree's. The eval loader remaps it by basename to the tree under test.
3. **The continuation tree (D4, Rule Zero) is NOT in either argv.** It lives in `DESIGN.md` sec. 8: if alpha-1 does not clear its bar,
   the next arms are (i) cold-start heads (no `--init-from`), (ii) `--r8-lat-prior-dropout 0`, (iii) both -- each differing from alpha-1
   in that flag alone.
4. **alpha-1s** adds the recorded steer to the input set; it is an arm only if the PI rules the steer admissible
   (`V8F-P10-ACTION-INPUT`, an open item of `v8f_steer` only). It is not needed for rung alpha.

To launch alpha-1: set the geometry (item 1), run `launch_gate.py run --profile v8f` on the launch tree, and only a PASS token starts
the trainer (the supervisor refuses without one); alpha-0 uses `--profile refcv8`. Nothing in this package launches.
