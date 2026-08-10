# context-tiers

[![ci](https://github.com/AmulyaP07-15/context-tiers/actions/workflows/ci.yml/badge.svg)](https://github.com/AmulyaP07-15/context-tiers/actions)

A context window is less a budget problem than an engineering one. Raising the token limit does not fix an agent that keeps the wrong things. What matters is the logic that decides what stays.

This is a library for that logic. It decides what an AI agent keeps in memory when it runs out of room.

```bash
pip install context-tiers
```

## Why this exists

When an agent's context window fills up, most agents delete the oldest messages. Sometimes the oldest message was "pay with miles, not the card." It gets dropped, the agent charges the card, the task fails, and nobody sees why.

Deleting by age is easy to build and quietly wrong. The thing you need to keep is rarely the newest thing. It is the constraint the customer stated, the account id the next tool call needs, the policy the agent must follow. This library keeps those and sheds the rest.

## How it works

Every piece of context is treated as a kind of thing, not just as text, and each kind is handled on its own terms. Rules never get touched. Hard customer requirements get detected and protected. Recent turns stay word for word. Old tool results get shortened but keep their key fields like ids and payment records. Old small talk gets trimmed first, because losing it costs the least.

Each category gets a priority tier, and a fixed token budget is spent down the tiers until it runs out.

```python
from context_tiers import ContextManager

cm = ContextManager(budget=6000)
cm.add("system", "You are a booking agent. Refunds go to the original payment method.")
cm.add("user", "Pay with miles, not the credit card.")   # detected and protected
messages, trace = cm.build(trace=True)
```

Every build explains itself. The trace records what was kept, shortened, or dropped, with the token math, and for each shortened tool result it lists which fields survived and which were cut.

## How it was tested

The library was wired into tau-bench, the agent benchmark Sierra publishes for customer service tasks. Two agents run the same airline tasks under the same token budget. One deletes oldest. One uses this library. The model and the tasks are held fixed, so the only thing that changes is how memory is managed.

Building a fair comparison meant getting the measurement honest first, and that surfaced three problems worth fixing. The budget was not counting the tool schemas sent on every request, so the real budget was far smaller than the number set. Compression was keeping the front of a JSON record and dropping the back, which lost a payment id sitting late in a reservation. And the old history was being flattened into a single user message, which told the model the customer had said things the tools actually returned. Each fix is now a property the library guarantees and a test that locks it in.

Both agents share the same short operating instruction, which tells them to look information up rather than interrogate the customer and to act once they have what a task requires. This is applied identically to both conditions, so it does not favor either one. It exists because a small model left to its own devices tends to stall in clarifying questions until the simulated customer gives up, which is a failure of prompting rather than of memory, and holding it constant lets the comparison measure what it is meant to measure. Episodes run to a fixed step budget.

## Result

Twenty episodes, five airline tasks, two attempts each, run on gpt-5-mini as the agent with tiered management versus naive delete-oldest under a shared 6000 token budget.

| condition | tasks solved (pass^1) | solved on both attempts (pass^2) | avg steps | avg total context |
|-----------|----------------------|----------------------------------|-----------|-------------------|
| tiered management | 0.60 | 0.40 | 22.5 | 122k |
| naive truncation | 0.20 | 0.00 | 19.3 | 104k |

![results chart](results_chart.png)

Raw report output:

![report output](screenshots/results.png)

Tiered management solved three times as many tasks on a single attempt, and it was the only condition that solved anything reliably. Naive truncation never once passed the same task on both attempts, which is the number that matters for a production agent. An agent that succeeds on one try and fails the retry cannot be trusted, and that is exactly what delete-oldest produced.

The managed agent used more total context, not less, which is worth explaining rather than hiding. It used more because it engaged with the hardest task in the set, downgrading five separate reservations with individual refunds, and worked it through to real tool calls instead of failing fast on something simpler. Reading the transcripts shows the extra spend is real work, with some redundant re-fetching of reservation details that had aged out of what the agent could act on. That re-fetching is itself an argument for the roadmap below. If the compressor understood that a reservation is still in play, it would keep it in view and the agent would not have to look it up twice.

A note on honesty of measurement. These are twenty episodes on a small model at temperature one, so the exact rates carry real variance and should be read as a clear direction rather than precise constants. The consistent finding across every run was the shape, tiered management holds constraints and structured fields under pressure that naive truncation drops, and that shows up as more reliable task completion.

A context window turns out to be less a budget problem than an engineering one, which is the lesson the whole build kept returning to. The token limit is not what separates a good agent from a bad one. The logic that decides what survives under pressure is.

## What is in the repo

| Path | What it is |
|------|------------|
| `context_tiers/manager.py` | the main class most users import |
| `context_tiers/allocator.py` | the tiered budget math |
| `context_tiers/summarizer.py` | field aware compression for structured tool output |
| `context_tiers/pinner.py` | detects hard user constraints worth protecting |
| `context_tiers/redis_store.py` | optional Redis backend so state survives restarts |
| `adapter.py` | plugs the library into tau-bench for a fair head to head |
| `runner.py` | benchmark runner with checkpointing, backoff, and resume |
| `report.py` | computes pass^k and the comparison |

## Try it in two minutes

```bash
git clone https://github.com/AmulyaP07-15/context-tiers.git
cd context-tiers
python3 -m venv venv && source venv/bin/activate
pip install -e ".[redis]"
python demo.py
```

The demo builds a conversation too big for its budget, runs it through the manager, and checks three things. The result fits the budget. The policy and the protected constraints survive untouched. Bulky tool output comes back shortened with its key fields intact.

## Roadmap

Three directions, in the order I would build them.

Semantic relevance. Constraints are matched by keyword today, which catches most stated requirements but misses paraphrases like "charge it to my miles" when the pinned rule says "use miles not the card." The plan is a hybrid that runs a small embedding model alongside the keyword layer and keeps whichever signal fires. It stays an optional install so the library still works offline and dependency free by default, which matters for the CI and air gapped cases where a model download is not an option.

Constraint aware compression. The summarizer treats every field on its own right now. The stronger version links each constraint to the entities it names, so a payment id is protected because it is tied to a live "use miles not the card" instruction rather than because its key happened to match a protected pattern. That turns the flat store into a small graph of constraints and the facts they touch, and it lets compression reason about relevance instead of matching strings.

Provenance envelope. Every shortened tool result already reports what it kept and dropped through the trace. The next step is to carry that as structured metadata a downstream audit can read without re-parsing the text, so a system can prove after the fact which fields a decision was made on. This is groundwork for using the library where compression decisions have to be explainable.

## Built with

Python, tiktoken for token counting, Redis optional, tau-bench and litellm for the benchmark harness. Tests and demos run in CI on every push.