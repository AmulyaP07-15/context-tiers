"""Phase 4: report. Reads results.jsonl, prints pass^1 / pass^k and token
stats per condition, and saves a comparison chart.

pass^k here follows the tau-bench convention: a task counts as passed at
k only if ALL k trials of that task succeeded (reward == 1). It measures
reliability, not one-shot luck.
"""

import json
import sys
from collections import Counter, defaultdict


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pass_hat_k(rows: list[dict], k: int) -> float:
    """Fraction of tasks where all k trials passed."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r["reward"] == 1)
    eligible = {t: trials for t, trials in by_task.items() if len(trials) >= k}
    if not eligible:
        return 0.0
    return sum(all(trials[:k]) for trials in eligible.values()) / len(eligible)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    rows = load(path)
    if not rows:
        print("no results yet")
        return

    conditions = sorted({r["condition"] for r in rows})
    counts = Counter((r["condition"], r["task_id"]) for r in rows)
    # report at the k every task actually reached, so pass^k is not computed
    # over a k that only one lucky task has trials for
    max_k = min(counts.values()) if counts else 1

    print(f"{len(rows)} episodes across {len({r['task_id'] for r in rows})} tasks\n")
    print(f"{'condition':<10} {'pass^1':>7} {'pass^' + str(max_k):>7} {'avg peak ctx':>13} {'avg total ctx':>14} {'avg steps':>10}")
    stats = {}
    for c in conditions:
        sub = [r for r in rows if r["condition"] == c]
        p1 = pass_hat_k(sub, 1)
        pk = pass_hat_k(sub, max_k)
        peak = sum(r["peak_context_tokens"] for r in sub) / len(sub)
        total = sum(r["total_context_tokens"] for r in sub) / len(sub)
        steps = sum(r["steps"] for r in sub) / len(sub)
        stats[c] = (p1, pk, peak, total)
        print(f"{c:<10} {p1:>7.2f} {pk:>7.2f} {peak:>13.0f} {total:>14.0f} {steps:>10.1f}")

    # accuracy by episode length: does naive degrade more on long tasks?
    print("\npass^1 by episode length (steps)")
    buckets = [(0, 10), (11, 20), (21, 99)]
    header = f"{'steps':<10}" + "".join(f"{c:>10}" for c in conditions)
    print(header)
    for lo, hi in buckets:
        line = f"{f'{lo}-{hi}':<10}"
        for c in conditions:
            sub = [r for r in rows if r["condition"] == c and lo <= r["steps"] <= hi]
            line += f"{(sum(r['reward'] == 1 for r in sub) / len(sub)):>10.2f}" if sub else f"{'—':>10}"
        print(line)

    # chart (optional; skipped if matplotlib missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        xs = range(len(conditions))
        ax1.bar([x - 0.2 for x in xs], [stats[c][0] for c in conditions], 0.4, label="pass^1")
        ax1.bar([x + 0.2 for x in xs], [stats[c][1] for c in conditions], 0.4, label=f"pass^{max_k}")
        ax1.set_xticks(list(xs)); ax1.set_xticklabels(conditions)
        ax1.set_ylim(0, 1); ax1.set_title("reliability"); ax1.legend()

        ax2.bar(xs, [stats[c][3] for c in conditions], 0.5, color="tab:orange")
        ax2.set_xticks(list(xs)); ax2.set_xticklabels(conditions)
        ax2.set_title("avg total context tokens per episode")

        fig.tight_layout()
        fig.savefig("results_chart.png", dpi=150)
        print("\nchart saved to results_chart.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping chart)")


if __name__ == "__main__":
    main()
