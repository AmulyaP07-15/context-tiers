"""Read why an episode passed or failed.

The runner now saves a compact trace of what the agent did (tool calls, results,
what it said) plus the reward detail. This prints one episode in readable form.

Usage:
  python inspect_episode.py                 # summarize all, list failures
  python inspect_episode.py 2 managed 0     # show task 2, managed, trial 0
  python inspect_episode.py --failures      # show every failed episode's trace
"""

import json
import sys


def load(path="results.jsonl"):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def show(row):
    print(f"\n=== task {row['task_id']} | {row['condition']} | trial {row['trial']} "
          f"| reward={row['reward']} | steps={row['steps']} ===")
    if row.get("reward_detail"):
        print(f"reward detail: {row['reward_detail']}")
    print("-" * 60)
    for step in row.get("trace", []):
        t = step["t"]
        if t == "call":
            print(f"  AGENT calls {step['name']}({step['args']})")
        elif t == "result":
            print(f"    -> {step['text']}")
        elif t == "say":
            print(f"  AGENT says: {step['text']}")
        elif t == "user":
            print(f"  USER: {step['text']}")


def main():
    args = [a for a in sys.argv[1:]]
    rows = load()

    if not rows:
        print("no episodes found in results.jsonl")
        return

    if not args:
        # summary + which failed
        passed = [r for r in rows if r["reward"] == 1]
        failed = [r for r in rows if r["reward"] != 1]
        print(f"{len(rows)} episodes: {len(passed)} passed, {len(failed)} failed\n")
        print("failures:")
        for r in failed:
            print(f"  task {r['task_id']} {r['condition']} trial {r['trial']} "
                  f"({r['steps']} steps)")
        print("\nrun `python inspect_episode.py <task> <condition> <trial>` to see one,"
              " or `--failures` to see all failed traces")
        return

    if args[0] == "--failures":
        for r in rows:
            if r["reward"] != 1:
                show(r)
        return

    if len(args) == 3:
        task, cond, trial = int(args[0]), args[1], int(args[2])
        for r in rows:
            if r["task_id"] == task and r["condition"] == cond and r["trial"] == trial:
                show(r)
                return
        print("no matching episode")
        return

    print(__doc__)


if __name__ == "__main__":
    main()
