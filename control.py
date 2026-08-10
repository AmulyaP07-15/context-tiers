"""Control run.

Runs tau-bench's OWN unmodified ToolCallingAgent on the same tasks, with the
same models, and no context management at all. This isolates one question:

  are the zero rewards caused by my adapter, or by the model?

If this scores zero too, the harness is fine and the model is the ceiling.
If this scores well, the problem is in adapter.py and the experiment is void.

Usage (same keys as runner.py):

  python control.py --tasks 0-4 --agent-model gpt-4o-mini --agent-provider github \
      --user-model gpt-4.1-mini --user-provider github

Checkpoints to control.jsonl and resumes, same as the main runner.
"""

import argparse
import json
import os
import time

from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent


def parse_task_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="0-4")
    p.add_argument("--env", default="airline")
    p.add_argument("--task-split", default="test")
    p.add_argument("--agent-model", default="gpt-4o-mini")
    p.add_argument("--agent-provider", default="github")
    p.add_argument("--user-model", default="gpt-4.1-mini")
    p.add_argument("--user-provider", default="github")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--out", default="control.jsonl")
    args = p.parse_args()

    tasks = parse_task_range(args.tasks)
    done = load_done(args.out)
    todo = [t for t in tasks if t not in done]
    print(f"control run: {len(tasks)} tasks, {len(done)} already done, {len(todo)} to run")
    print("upstream tau-bench agent, no budget, no context management\n")

    env = get_env(
        env_name=args.env,
        user_strategy="llm",
        user_model=args.user_model,
        user_provider=args.user_provider,
        task_split=args.task_split,
        task_index=0,
    )

    for t in todo:
        print(f"task {t} ...", end=" ", flush=True)
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=args.agent_model,
            provider=args.agent_provider,
            temperature=0.0,
        )
        delay = 10.0
        record = None
        for attempt in range(6):
            try:
                res = agent.solve(env=env, task_index=t, max_num_steps=args.max_steps)
                record = {"task_id": t, "reward": res.reward,
                          "messages": len(res.messages),
                          "agent_model": args.agent_model, "ts": time.time()}
                break
            except Exception as e:
                print(f"\n  retry {attempt + 1}/6: {str(e)[:110]}", end=" ", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)
        if record is None:
            print("gave up")
            continue
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"reward={record['reward']:.0f}  messages={record['messages']}")

    rows = [json.loads(l) for l in open(args.out)] if os.path.exists(args.out) else []
    if rows:
        passed = sum(r["reward"] == 1 for r in rows)
        print(f"\ncontrol result: {passed}/{len(rows)} tasks passed "
              f"({passed / len(rows):.0%}) with the upstream agent and no budget")
        print("\nread this as:")
        print("  0/5   -> the model is the ceiling, my harness is not the problem")
        print("  2/5+  -> the harness is suspect and the comparison is void")


if __name__ == "__main__":
    main()
