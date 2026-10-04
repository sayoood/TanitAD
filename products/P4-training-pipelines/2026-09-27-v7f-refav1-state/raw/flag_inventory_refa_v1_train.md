# G-DVB flag inventory — `refa_v1_train.py`

Static AST inventory (no import). **Class = name-pattern PROPOSAL; NO_READ_FOUND = candidate, not proof.** Stack files scanned for forwarded reads: 1126.

**40 flags** · status {'READ': 40} · proposed class {'DATA': 4, 'OPTIM': 9, 'UNCLASSIFIED': 13, 'RUNTIME': 8, 'LOSS': 3, 'MODEL': 3}

## ⛔ NO_READ_FOUND — parsed, never read anywhere in the stack (declared-but-inert CANDIDATES)

| flag | default | proposed class | line |
|---|---|---|---|
| (none) | | | |

## Every flag

| flag | dest | default | class | reads (trainer / stack) | status |
|---|---|---|---|---|---|
| `--allow-unlabelled` | `allow_unlabelled` | `None` | DATA | 1 / 0 | READ |
| `--cache` | `cache` | `None` | DATA | 6 / 58 | READ |
| `--episodes` | `episodes` | `None` | DATA | 2 / 215 | READ |
| `--labels` | `labels` | `None` | DATA | 3 / 50 | READ |
| `--w-aux-head` | `w_aux_head` | `0.0` | LOSS | 3 / 3 | READ |
| `--w-cf` | `w_cf` | `0.0` | LOSS | 2 / 9 | READ |
| `--w-sigreg` | `w_sigreg` | `0.0` | LOSS | 1 / 3 | READ |
| `--proposal-k` | `proposal_k` | `1` | MODEL | 1 / 11 | READ |
| `--skip-nonfinite` | `skip_nonfinite` | `None` | MODEL | 1 / 0 | READ |
| `--target-space` | `target_space` | `frozen` | MODEL | 1 / 7 | READ |
| `--adapter-lr-mult` | `adapter_lr_mult` | `0.1` | OPTIM | 1 / 3 | READ |
| `--bs` | `bs` | `8` | OPTIM | 4 / 27 | READ |
| `--clip` | `clip` | `1.0` | OPTIM | 2 / 125 | READ |
| `--ema-decay` | `ema_decay` | `0.996` | OPTIM | 1 / 18 | READ |
| `--ema-decay-end` | `ema_decay_end` | `0.999` | OPTIM | 1 / 8 | READ |
| `--ema-targets` | `ema_targets` | `None` | OPTIM | 2 / 2 | READ |
| `--lr` | `lr` | `0.0003` | OPTIM | 2 / 78 | READ |
| `--lru` | `lru` | `32` | OPTIM | 1 / 16 | READ |
| `--steps` | `steps` | `30000` | OPTIM | 4 / 163 | READ |
| `--device` | `device` | `<expr> 'cuda' if torch.cuda.is_available` | RUNTIME | 12 / 674 | READ |
| `--log-every` | `log_every` | `50` | RUNTIME | 1 / 34 | READ |
| `--out` | `out` | `<expr> Path('./refa_v1_run')` | RUNTIME | 6 / 701 | READ |
| `--precision` | `precision` | `fp32` | RUNTIME | 3 / 16 | READ |
| `--resume` | `resume` | `None` | RUNTIME | 1 / 5 | READ |
| `--save-every` | `save_every` | `1000` | RUNTIME | 1 / 32 | READ |
| `--seed` | `seed` | `0` | RUNTIME | 2 / 224 | READ |
| `--tf32` | `tf32` | `None` | RUNTIME | 2 / 2 | READ |
| `--bptt-truncate` | `bptt_truncate` | `15` | UNCLASSIFIED | 1 / 13 | READ |
| `--cf-at-step` | `cf_at_step` | `4` | UNCLASSIFIED | 1 / 3 | READ |
| `--cf-negs` | `cf_negs` | `3` | UNCLASSIFIED | 1 / 3 | READ |
| `--detach-aux-targets` | `detach_aux` | `None` | UNCLASSIFIED | 2 / 2 | READ |
| `--no-detach-aux-targets` | `detach_aux` | `None` | UNCLASSIFIED | 2 / 2 | READ |
| `--min-participation` | `min_participation` | `0.0` | UNCLASSIFIED | 1 / 4 | READ |
| `--motion-inject` | `motion_inject` | `None` | UNCLASSIFIED | 1 / 2 | READ |
| `--nav` | `nav` | `None` | UNCLASSIFIED | 3 / 13 | READ |
| `--no-hierarchy` | `no_hierarchy` | `None` | UNCLASSIFIED | 3 / 0 | READ |
| `--smoke` | `smoke` | `None` | UNCLASSIFIED | 4 / 28 | READ |
| `--speed-channel` | `speed_channel` | `None` | UNCLASSIFIED | 1 / 5 | READ |
| `--tmix-groups` | `tmix_groups` | `None` | UNCLASSIFIED | 1 / 6 | READ |
| `--var-floor` | `var_floor` | `0.0` | UNCLASSIFIED | 1 / 2 | READ |
