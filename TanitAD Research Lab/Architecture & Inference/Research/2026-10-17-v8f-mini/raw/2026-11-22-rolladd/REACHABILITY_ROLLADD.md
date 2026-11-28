# REACHABILITY_ROLLADD: argv -> appended rows -> loss -> backward, file:line per hop (the build tree, post-edit lines)

| hop | file:line | what |
|---|---|---|
| argparse | `stack/scripts/v8f_mini_train.py:251` | `--dagger-mode {swap,add}`, default `swap` (`DG.DEFAULT_MODE`, `v8f_dagger.py:60-61`) |
| FLAG_CHECKS | `v8f_mini_train.py:106` | `"dagger_mode": "dagger:dagger_mode"` (the parser/FLAG_CHECKS equality test pins it) |
| namespace -> object | `v8f_mini_train.py:639` | `DG.DaggerSwap(..., mode=a.dagger_mode)` in `make_dagger` |
| built read-back | `v8f_dagger.py:718` (`built_levers`), `v8f_mini_train.py:646,656` | `dagger_mode` declared against built; refused at `--dagger-p 0` by the loop that already refuses every other declared-and-idle dagger flag; `v8f_dagger.py:739` (`problems`) refuses `add` at p 0 |
| stage refusals | `v8f_mini_train.py:664-669` | launch / level / lead / ego-perturb(-with-dagger) / recover / t2-bank / lora / wm-only / wm-windows wmonly refused beside `add` |
| selection | `v8f_dagger.py:~776-779` | the same generator, `u`, `sel`, `rows`, `pick`, `swap_rows` call as `swap` |
| the append | `v8f_dagger.py:789` (`append_rows` defined `:676`) | logged B rows + the N swapped rows; `n_appended` counted from the returned batch (`:790`) |
| the loop | `v8f_mini_train.py:3715` | `b = dag.apply(b, step)`; counters `:3736` (`dagger_appended`, `dagger_rows_out`, only under add) |
| loss, backward | `v8f_mini_train.py:3774` (`T["total"].backward()`), loss `v8f_mini.py:1156` | on B + N rows; every term a masked mean (FACTS section 3) |
| proof | `dagger_add_probe` `v8f_mini_train.py:2734`, called `:2842` | non-zero gradient difference add vs logged, p = 0 control exactly 0.0, first B rows == logged, appended == swap rows |
| tests | `stack/tests/test_v8f_rolladd.py` | gold, add, refusals, probe, end-to-end (the loss's own row count), mutants `mutants/run_mutants.py` |
