# RESULT — WP-F: hand-crafted relational edges carry NO decision-relevant signal

**2026-09-05 · TanitAD_TrainingFlyWheel · Tier T0 · ⚠️ NON-PARITY pilot (675 windows, 15
episodes, episode-disjoint) · Evidence class: MEASURED (ours).**

---

## 1. The numbers — the probe is VALID this time

| arm | acc | |
|---|---|---|
| MAJORITY (always rank-1) | 0.4519 | the baseline |
| **NODE** — scorer signal + candidate's own geometry | **0.4741** | ✅ **beats majority (+0.0222)** — the sanity gate the first run failed |
| **NODE + EDGE** — + candidate×agent relations | **0.4704** | ⛔ **worse than NODE** |
| CONSTANT-EDGE control | 0.4741 | ✅ **exactly NODE** |
| **SHUFFLED-EDGE control** | **0.4815** | ⛔ **BETTER than the real edges** |

**edge gain = −0.0037.** ⇒ ⛔ **These edge features carry nothing. Random edges do as well as
real ones** — the definitive form of a null, because it rules out "the effect is there but small".

## 2. ⭐ What it does and does not say

**Says:** six hand-crafted geometric relations — min clearance, TTC, passing side, closing over the
path, count within 3 m, median clearance — aggregated over agents, add **nothing** to a top-5
re-rank beyond the candidate's own geometry and the scorer's score.

**Does NOT say** the graph is dead. These are *hand-crafted scalars*, not a **learned** edge
representation from message passing over per-agent latents. But ⚠️ **it is real evidence against
the cheap version of v6**, and v6's claim was precisely that relations carry the decision. On the
most natural relations available, they do not.

## 3. ⭐⭐ The finding that connects WP-F to WP-G

**NODE beats MAJORITY by only +0.0222.** The scorer's own signal plus the full geometry of the
candidate barely improves on *"always pick rank-1"* — **the top-5 choice is close to unsolvable
from geometry alone.**

Read that beside the WP-G precondition: **our corpus has no `motorcycle` and no `bicycle` class**
(both collapse into `rider`, 1.36 %, and the footprints are indistinguishable), `occ` is
constant-zero, and there is **no illumination field**.

⇒ **Coherent story: the re-rank needs SEMANTICS, geometry cannot supply them, and the semantic
labels that would supply them do not exist in our data.** That is an argument for the PI's
distillation proposal reached from two independent directions, and it makes **WP-G proper** — can a
strong teacher separate motorcycle from bicycle on our frames? — the highest-value next probe.

## 4. ⚠️ A fourth specification error, mine, and the one I want on the record

The script printed **"CONTROLS FAILED — void"**. **That label is wrong.** The controls behaved
*correctly*: constant-edge equalled NODE exactly, and shuffled-edge told us what we needed to know.
I had put the **effect test** (`edge_gain > shuffled_gain`) inside the **validity gate**, so a clean
negative was reported as an invalid experiment.

⭐ **The rule: a validity gate asks "could this experiment have detected the effect?"; an effect
test asks "did it?". Mixing them makes a real negative unreadable** — and in the other direction it
would let a broken probe pass by producing a large number. Separate them, always.

*(This is the fourth in a series: WP-B's features identical across candidates; WP-B's shuffled
control judged against the wrong chance level; WP-F run 1's arms denied information the baseline
had; and now validity conflated with effect. All four share one root: **the comparison was not
between things that could be compared.** Four probes rebuilt, four measurements that stand.)*
