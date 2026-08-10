"""Phase 2 demo. Proves three upgrades over Phase 1:
1. Spent tool results get SUMMARIZED into one short item (Phase 1 dropped them to 0)
2. Hard user constraints are auto-detected and pinned
3. State lives in Redis and survives a 'restart'

Runs with fakeredis (no server needed). On your machine, swap in
RedisContextStore.from_url("redis://localhost:6379/0", session=...) with Docker Redis.
Pass an OllamaSummarizer/OllamaPinner for LLM-quality summaries/pins; the
heuristic fallbacks below need no LLM at all.
"""

import fakeredis

from context_tiers.redis_store import RedisContextStore
from context_tiers.assembler import build_context, context_tokens
from context_tiers.allocator import allocate
from context_tiers.pinner import HeuristicPinner
from context_tiers.summarizer import HeuristicSummarizer
from context_tiers.tokens import count_item

CONFIG = {
    "total_budget": 1000,
    "sources": {
        "system_policy": {"tier": 0, "strategy": "verbatim"},
        "pinned":        {"tier": 0, "strategy": "verbatim"},
        "recent_turns":  {"tier": 1, "strategy": "verbatim"},
        "active_tools":  {"tier": 2, "strategy": "verbatim"},
        "old_turns":     {"tier": 4, "strategy": "truncate_oldest"},
        "spent_tools":   {"tier": 3, "strategy": "summarize", "summary_budget": 80},
    },
}


def main() -> None:
    pinner = HeuristicPinner()
    store = RedisContextStore(fakeredis.FakeRedis(decode_responses=True), session="phase2")
    store.clear()

    store.add("system",
              "You are an airline booking agent. Policy: basic economy tickets cannot be "
              "modified, only cancelled and rebooked. Companion bookings must be modified together.",
              "system_policy", 0)

    user_turns = {
        3: "Pay with miles, not the credit card please.",
        6: "My companion Priya must stay on the same booking.",
        9: "What time does the morning flight land?",
    }

    for t in range(1, 16):
        source = "recent_turns" if t >= 12 else "old_turns"
        user_msg = user_turns.get(
            t, f"(turn {t}) User asks about options, baggage, seats and timing for day {t}. " * 2)
        store.add("user", user_msg, source, t)
        # Phase 2 upgrade: every user message runs through the pinner
        for pin in pinner.extract(user_msg):
            store.add("user", f"PINNED: {pin}", "pinned", t)
        store.add("assistant",
                  f"(turn {t}) Agent lays out the options for day {t} with times and fares. " * 2,
                  source, t)

    store.add("tool",
              "FLIGHT SEARCH RESULTS (turn 2): UA482 BOS-ORD 7:05am $214. UA318 9:40am $189. "
              "AA1121 12:15pm $205. B6771 3:30pm $178. UA990 6:55pm $164. Full fare rules, "
              "seat maps and baggage details for all five options in verbose form. " * 4,
              "spent_tools", 2)
    store.add("tool",
              "CURRENT RESERVATION (turn 13): UA482 confirmed for user + companion Priya, "
              "paying with miles, seats 14A/14B held.",
              "active_tools", 13)

    ctx = build_context(store, CONFIG, summarizer=HeuristicSummarizer())
    store_total = store.total_tokens()
    ctx_total = context_tokens(ctx)
    allocation = allocate(store, CONFIG)

    print(f"Store (full state):  {store_total} tokens")
    print(f"Built context:       {ctx_total} tokens (budget {CONFIG['total_budget']})")
    print()
    print("Per-source: needed -> granted")
    for source in CONFIG["sources"]:
        need = sum(count_item(it) for it in store.get_by_source(source))
        note = ""
        if allocation[source] < need:
            note = "   (SUMMARIZED)" if CONFIG["sources"][source]["strategy"] == "summarize" else "   (CUT)"
        print(f"  {source:14s} {need:5d} -> {allocation[source]:5d}{note}")

    texts = [m["content"] for m in ctx]

    # 1. spent tools are summarized, not gone
    assert any("[summary of spent_tools]" in t for t in texts), "spent tools should survive as a summary"
    assert any("UA482" in t and "[summary" in t for t in texts), "summary should keep concrete facts"

    # 2. pins were auto-extracted and survive verbatim
    pinned_items = store.get_by_source("pinned")
    assert len(pinned_items) >= 2, "pinner should have caught the miles + companion constraints"
    for it in pinned_items:
        assert it.content in texts, f"pinned constraint lost: {it.content[:50]}"
    assert not any("morning flight land" in it.content for it in pinned_items), \
        "plain question must NOT be pinned"

    # 3. budget respected, state overflows
    assert store_total > CONFIG["total_budget"] and ctx_total <= CONFIG["total_budget"]

    # 4. Redis persistence: a 'new process' sees the same state
    reopened = RedisContextStore(store.r, session="phase2")
    assert reopened.total_tokens() == store_total, "state must survive restart"

    print(f"\nPins caught: {[it.content for it in pinned_items]}")
    print("\nAll asserts passed: summarize-not-drop, auto-pinning, budget fit, Redis persistence.")


if __name__ == "__main__":
    main()
