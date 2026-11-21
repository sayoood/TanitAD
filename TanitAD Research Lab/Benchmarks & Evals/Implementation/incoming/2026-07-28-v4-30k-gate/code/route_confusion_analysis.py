"""Per-class recall/precision + balanced accuracy from the goal head's route confusion matrix.

WHY: recall alone said the head is blind to turns. PRECISION says it is not — it says the head
knows and will not commit, which is a THRESHOLD problem and a completely different (and much
cheaper) fix than a retrain. Pure arithmetic on the matrix already in
raw_v4fs-30k-produced.json -> goal_agreement_vs_oracle. No GPU, no model, no data reload.

Wilson intervals are printed because one of these cells has n = 8 and would otherwise be quoted
next to a cell with n = 54 as if they were comparable.
"""
import math

# oracle rows x produced cols, verbatim from the artifact
M = [[49, 160, 3, 0, 0],
     [0, 394, 0, 0, 0],
     [5, 111, 5, 0, 0],
     [21, 128, 5, 0, 0],
     [0, 0, 0, 0, 0]]
NAMES = ["left", "straight", "right", "UNKNOWN(masked)", "DROPPED"]
JUDGEABLE = [0, 1, 2]      # cols 3/4 are the masked sentinel and the learned "no route" row


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    tot = sum(sum(r) for r in M)
    print(f"total windows {tot}   oracle row sums {[sum(r) for r in M]}")
    pred_j = [sum(M[r][c] for r in JUDGEABLE) for c in range(5)]
    n_j = sum(sum(M[r]) for r in JUDGEABLE)
    print(f"judgeable n={n_j}   predictions over judgeable rows {pred_j}\n")

    print(f"{'class':10s} {'n':>5s} {'recall':>8s} {'precision':>10s} {'pred n':>7s}   95% Wilson (precision)")
    recalls = []
    for c in JUDGEABLE:
        n, tp, pn = sum(M[c]), M[c][c], pred_j[c]
        rec = tp / n
        recalls.append(rec)
        prec = tp / pn if pn else float("nan")
        lo, hi = wilson(tp, pn)
        warn = "   <-- n too small to quote" if pn < 20 else ""
        print(f"{NAMES[c]:10s} {n:5d} {rec:8.4f} {prec:10.4f} {pn:7d}   [{lo:.3f}, {hi:.3f}]{warn}")

    bal = sum(recalls) / len(recalls)
    acc = sum(M[c][c] for c in JUDGEABLE) / n_j
    maj = sum(M[1]) / n_j
    print(f"\nBALANCED accuracy = {bal:.4f}  vs chance {1/3:.4f}  -> {(bal-1/3)*100:+.1f} pts")
    print(f"plain accuracy    = {acc:.4f}  vs always-straight majority {maj:.4f}  -> {(acc-maj)*100:+.1f} pts")
    print("\nplain accuracy REWARDS the collapse; balanced accuracy does not. Pre-register the latter.")


if __name__ == "__main__":
    main()
