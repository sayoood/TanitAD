# §12 refusal 1, in both halves — the parsed namespaces AND the built config

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: dev box, CPU**
**Closes:** the last ⚠️ row of `PREREG_REFCV6_V2.ERRATUM-1.md` §E4, under the amendment §E5 makes.

## Why one half was not enough

§12's refusal 1 says arms must differ in one variable *"as parsed namespaces, not as intent"*. That
emphasis is right. ⛔ **It stops one layer too early.**

MEASURED 2026-09-17: `_pin_trainer_cfg` rebuilt `CNNEncoderConfig` from a **hand-written field
list** — 12 fields in the dataclass, 8 in the list — so under `--image-hw` four silently returned to
their defaults:

| asked for | BUILT |
|---|---|
| `--trunk-name resnet34` | **resnet101** |
| `--trunk-fuse last` (the single-frame **control**) | `concat1x1` |
| `--trunk-fuse-plain-init` (the deliberate **regression**) | the identity init |

⇒ the parsed namespaces differ **exactly as declared** and the **built configs do not differ at
all**. The defect lives *between* argv and the model, which is precisely where an argv-only checker
is blind. `--image-hw 256 1024` is the PI's standing instruction, and §1's arm B is `resnet34` — so
A and B would have been **the same model**.

## What this enforces

1. the parsed namespaces differ in exactly the declared lever (plus allow-listed **constitutive**
   and **bookkeeping** keys);
2. ⭐ the **BUILT config** — after every pin and rebuild — differs in exactly the paths the lever
   declares, **and**
   - **no built difference at all** is a VIOLATION (*the two arms are one arm*), and
   - an **undeclared** built difference is a VIOLATION (*a second lever hiding downstream*).

⛔ `expected_from` / `expected_to` are **literals supplied by the caller**, never expressions over
the code under test — an expectation computed from the thing being checked measures determinism,
not correctness. ⛔ `build_parser` and `build_config` are passed in so the checker uses the
**trainer's own** parser and construction, never a copy: a checker with its own copy of the flags is
a check that shares the defect it checks for.

## Mutation-proven

**8 tests.** The one that earns the module reproduces the real defect: a builder that rebuilds a
dataclass from a hand-written list, both arms passing `--image-hw` as every 1024 run does. The test
asserts the old half-check **passes** it (`namespace_diff` still shows `resnet101.a1_in1k →
resnet34`) and that the built configs are **identical** — then requires the refusal to fire. ⭐ A
discriminating control runs the **same arm pair** through the **fixed** builder and requires it to
PASS, so the refusal is about the builder and not about `--image-hw` being present.

**Guard-removal audit: 5 of 5 KILLED, 0 escaped, baseline restored green**
(`raw/audit_one_variable_v2.json`).

## §12 tally

**9 ✅ · 0 ⚠️ · 1 ⛔.** The one remaining is refusal 10 (a capped `goal_pos_weight` quoted without
its cap sentence) — a rule about **prose**, which must either become a check on the panel's own
fields or be demoted to a convention. ⛔ That is the PI's call and is deliberately not taken here.

⚠️ **This module is the checker, not its application.** Wiring it to the real `refc_v3_train`
arms needs the v2 launch commands, which do not exist yet — building a diff for commands nobody has
written would be speculation, not a guard. The checker takes them as arguments precisely so it is
ready when they are.
