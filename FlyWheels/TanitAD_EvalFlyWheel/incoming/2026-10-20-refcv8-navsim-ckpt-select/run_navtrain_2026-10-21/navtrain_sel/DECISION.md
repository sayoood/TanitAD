# D-R8-CKPT-RULE decision -- DEFAULT_EMA_48852

**Choice: ema48852** (2089 tokens, 30 log clusters). Tier: NavSim open-loop PDM scorer (T1-family; NOT closed loop). Choice between two EMA snapshots of ONE training run; says nothing about another training seed. Backend: rolls on Thor CUDA, scoring on the Thor CPU backend (D-R8-CKPT-RULE AMENDMENT 1c).

| seed | 40k - 48,852 PDMS x100 | 95 % log-cluster interval | positive and excludes 0 |
|---|---|---|---|
| 0 | 0.6915 | [-0.22, 1.69] | False |
| 1 | 1.0671 | [-0.17, 2.27] | False |

inference floor |d_s0 - d_s1| = 0.3755 ; margin = min(delta_s0, delta_s1) = 0.6915 ; needed > 0.751 ; exceeds: False

controls read their known values: True

Reasons:
* `A_SEED_IS_NOT_POSITIVE_WITH_AN_INTERVAL_EXCLUDING_ZERO`
* `MARGIN_NOT_ABOVE_2X_INFERENCE_FLOOR`
