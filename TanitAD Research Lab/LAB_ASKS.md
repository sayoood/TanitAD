<title>LAB_ASKS - the FlyWheel <-> Research Lab channel</title>

# LAB_ASKS

⭐ **PI decision 2026-08-31:** FlyWheel agents talk to the Research Lab **directly**,
through this file. The Master Mind is no longer in the interpretation path.

⛔ **APPEND WITH THE TOOL, NEVER BY HAND:** `python stack/scripts/lab_ask.py --ask "..."
--from <agent> [--field <field>] [--why "..."]`. Several agents write here and the mount
intermittently fails mid-write; the tool does read-modify-write + atomic replace +
read-back. A hand-edit during another agent's write loses an ask, and **a lost ask is a
question nobody knows was asked**.

**The contract**
| motion | who | when |
|---|---|---|
| **ask** — append an OPEN row: the question, why it matters, what would answer it | any FlyWheel, the MM, the PI | any time |
| **read** — the Research Lab's standing brief MUST read this file and address OPEN rows before pulling backlog seeds | Research Lab | every daily pass |
| **answer** — `--answer ASK-N --package <path>`; the row becomes ANSWERED and points at the evidence | Research Lab | same pass |
| ⚠️ **a question the Lab cannot answer is answered with WHY NOT**, never left silent — an unanswered ask that looks pending forever is the failure this file replaces | Research Lab | same pass |

---

### ASK-1 · OPEN · TrainingFlyWheel · Architecture & Inference
*asked 2026-08-31*

**Q.** Does any published recipe train a 60-step full-chain BPTT rollout stably, and with what clip/schedule? Our k=60 arm diverged at clip 1.0 (gnorm 5.71 median -> 2.1e9) while k=8 was stable.

**Why it matters.** The 6 s horizon is binding (section 4b) and we have now measured that its wall-clock cost was the easy half; gradient stability is the unpriced half. A published stable recipe would save us an arm per mitigation guess.

