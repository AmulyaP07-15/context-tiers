"""Phase 4: benchmark runner.

Runs task x condition x trial episodes on tau-bench, built to survive
free-tier API reality:
  - every finished episode is appended to results.jsonl immediately
  - on restart, finished episodes are skipped (resume for free)
  - exponential backoff on rate limits / transient errors
  - hard per-episode retry cap so one poisoned task can't stall the run

Usage (on your machine, inside the tau-bench repo with our files present):

  export GEMINI_API_KEY=...   # agent model (Google AI Studio free tier)
  export GROQ_API_KEY=...     # user simulator (Groq free tier)

  python runner.py --tasks 0-24 --k 2 --budget 3000 \
      --agent-model gemini-2.0-flash --agent-provider gemini \
      --user-model llama-3.3-70b-versatile --user-provider groq

Then:  python report.py results.jsonl
"""

import argparse
import json
import os
import time
from typing import Optional

from tau_bench.envs import get_env

from adapter import NaiveBudgetAgent, ManagedBudgetAgent

CONDITIONS = {"naive": NaiveBudgetAgent, "managed": ManagedBudgetAgent}


def episode_key(task_id: int, condition: str, trial: int) -> str:
    return f"{task_id}:{condition}:{trial}"


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(episode_key(r["task_id"], r["condition"], r["trial"]))
                except (json.JSONDecodeError, KeyError):
                    continue  # ignore partial/corrupt lines
    return done


_ENV_CACHE = {}


def get_cached_env(args):
    """env.reset() reloads the database and re-primes the user simulator, so a
    single env can serve every episode. Building a new one each time costs an
    extra user-simulator call that is immediately thrown away."""
    key = (args.env, args.task_split, args.user_model, args.user_provider)
    if key not in _ENV_CACHE:
        # task_index=0 is deliberate. Constructing without one sends tau-bench
        # down `random.randint(0, len(tasks))`, whose upper bound is inclusive,
        # so it can index one past the end and raise IndexError. Every episode
        # resets to its real task index before running, so the value here only
        # has to be valid.
        _ENV_CACHE[key] = get_env(
            env_name=args.env,
            user_strategy="llm",
            user_model=args.user_model,
            user_provider=args.user_provider,
            task_split=args.task_split,
            task_index=0,
        )
    return _ENV_CACHE[key]


def run_episode(task_id: int, condition: str, trial: int, args) -> dict:
    env = get_cached_env(args)
    AgentCls = CONDITIONS[condition]
    agent = AgentCls(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=args.agent_model,
        provider=args.agent_provider,
        temperature=0.0,
        budget=args.budget,
        strict_budget=not args.allow_below_floor,
    )
    result = agent.solve(env=env, task_index=task_id, max_num_steps=args.max_steps)

    # a compact trace of what the agent actually DID, so a failed episode can be
    # inspected instead of just recorded as reward=0. Tool calls and their
    # results are the useful signal; long tool payloads are trimmed.
    trace = []
    for m in getattr(result, "messages", []) or []:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                trace.append({"t": "call", "name": fn.get("name"),
                              "args": str(fn.get("arguments"))[:200]})
        elif role == "assistant" and m.get("content"):
            trace.append({"t": "say", "text": str(m.get("content"))[:200]})
        elif role == "tool":
            trace.append({"t": "result", "text": str(m.get("content"))[:200]})
        elif role == "user":
            trace.append({"t": "user", "text": str(m.get("content"))[:200]})

    reward_info = getattr(result, "info", None)
    reward_detail = None
    if reward_info is not None:
        ri = getattr(reward_info, "reward_info", None)
        if ri is not None:
            reward_detail = str(ri)[:500]

    return {
        "task_id": task_id,
        "condition": condition,
        "trial": trial,
        "reward": result.reward,
        "steps": agent.steps,
        "peak_context_tokens": agent.peak_context_tokens,
        "total_context_tokens": agent.total_context_tokens,
        "budget": args.budget,
        "floor_tokens": agent.floor_tokens,
        "conv_budget": agent.conv_budget,
        "agent_model": args.agent_model,
        "trace": trace,
        "reward_detail": reward_detail,
        "ts": time.time(),
    }


FATAL_MARKERS = ("No module named", "BadRequestError", "NotFoundError",
                 "AuthenticationError", "no longer available",
                 "BudgetBelowFloorError", "fixed floor")


def run_with_backoff(task_id: int, condition: str, trial: int, args) -> Optional[dict]:
    delay = 10.0
    for attempt in range(args.max_retries):
        try:
            return run_episode(task_id, condition, trial, args)
        except Exception as e:
            msg = str(e)[:160]
            if any(mark in str(e) for mark in FATAL_MARKERS):
                print(f"  FATAL (not retrying): {msg}")
                raise
            print(f"  retry {attempt + 1}/{args.max_retries} after error: {msg}")
            time.sleep(delay)
            delay = min(delay * 2, 300)  # cap at 5 min
    print(f"  GIVING UP on {episode_key(task_id, condition, trial)} — rerun later, resume will pick it up")
    return None


def parse_task_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="0-24", help="e.g. 0-24 or 3,7,11")
    p.add_argument("--k", type=int, default=2, help="trials per task per condition")
    p.add_argument("--budget", type=int, default=8000,
                   help="TOTAL request budget, tool schemas included")
    p.add_argument("--allow-below-floor", action="store_true",
                   help="run even if the budget cannot hold the domain floor")
    p.add_argument("--env", default="airline")
    p.add_argument("--task-split", default="test")
    p.add_argument("--agent-model", default="gemini-2.0-flash")
    p.add_argument("--agent-provider", default="gemini")
    p.add_argument("--user-model", default="llama-3.3-70b-versatile")
    p.add_argument("--user-provider", default="groq")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--out", default="results.jsonl")
    args = p.parse_args()

    tasks = parse_task_range(args.tasks)
    done = load_done(args.out)
    todo = [
        (t, c, tr)
        for t in tasks
        for c in CONDITIONS
        for tr in range(args.k)
        if episode_key(t, c, tr) not in done
    ]
    total = len(tasks) * len(CONDITIONS) * args.k
    print(f"{total} episodes total, {len(done)} already done, {len(todo)} to run")

    for i, (t, c, tr) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] task {t} | {c} | trial {tr}")
        try:
            record = run_with_backoff(t, c, tr, args)
        except Exception as e:
            if not done and i == 1:
                raise   # nothing has ever succeeded: this is a setup error
            print(f"  SKIPPING {episode_key(t, c, tr)} after fatal error: {str(e)[:120]}")
            continue
        if record is not None:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  reward={record['reward']:.0f} steps={record['steps']} peak_ctx={record['peak_context_tokens']}")

    print("Run complete. Next: python report.py " + args.out)


if __name__ == "__main__":
    main()
