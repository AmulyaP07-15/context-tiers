"""Demo: a 15-turn fake flight-booking conversation that overflows a 1000-token
budget, forcing the manager to cut. Proves: policy + pinned always survive."""

from context_tiers.store import ContextStore
from context_tiers.assembler import build_context, context_tokens
from context_tiers.allocator import allocate
from context_tiers.tokens import count_item

CONFIG = {
    "total_budget": 1000,
    "sources": {
        "system_policy": {"tier": 0, "strategy": "verbatim"},
        "pinned":        {"tier": 0, "strategy": "verbatim"},
        "recent_turns":  {"tier": 1, "strategy": "verbatim"},
        "active_tools":  {"tier": 2, "strategy": "verbatim"},
        "old_turns":     {"tier": 3, "strategy": "truncate_oldest"},
        "spent_tools":   {"tier": 4, "strategy": "summarize"},
    },
}


def build_fake_conversation() -> ContextStore:
    store = ContextStore()

    store.add("system",
              "You are an airline booking agent. Policy: basic economy tickets cannot be "
              "modified, only cancelled and rebooked. Refunds go to the original payment "
              "method. Seat changes are free within the same cabin. Companion bookings "
              "must be modified together.",
              "system_policy", 0)

    # user-stated hard constraints -> pinned
    store.add("user", "PINNED: Pay with miles, not the credit card.", "pinned", 3)
    store.add("user", "PINNED: My companion Priya must stay on the same booking.", "pinned", 6)

    for t in range(1, 16):
        source = "recent_turns" if t >= 12 else "old_turns"
        store.add("user",
                  f"(turn {t}) User: I'm looking at my trip to Chicago. Can you check the "
                  f"options for day {t}? I might also want to know about baggage and seats "
                  f"and whether the times work with my meeting schedule that week.",
                  source, t)
        store.add("assistant",
                  f"(turn {t}) Agent: Sure. For day {t} there are several options. The "
                  f"morning departure arrives before noon, and the evening one gets in "
                  f"late. Baggage is one free carry-on; checked bags cost extra on this fare.",
                  source, t)

    # tool results: one spent (user already chose), one active
    store.add("tool",
              "FLIGHT SEARCH RESULTS (turn 2): UA482 BOS-ORD 7:05am $214 ... UA318 "
              "BOS-ORD 9:40am $189 ... AA1121 BOS-ORD 12:15pm $205 ... B6 771 BOS-ORD "
              "3:30pm $178 ... UA990 BOS-ORD 6:55pm $164 ... full fare rules, seat maps, "
              "and baggage details for each of the five options follow in verbose form. " * 3,
              "spent_tools", 2)
    store.add("tool",
              "CURRENT RESERVATION (turn 13): UA482 confirmed for user + companion Priya, "
              "paid pending, seats 14A/14B held.",
              "active_tools", 13)

    return store


def main() -> None:
    store = build_fake_conversation()
    allocation = allocate(store, CONFIG)
    ctx = build_context(store, CONFIG)

    store_total = store.total_tokens()
    ctx_total = context_tokens(ctx)

    print(f"Store (full state):  {store_total} tokens")
    print(f"Built context:       {ctx_total} tokens (budget {CONFIG['total_budget']})")
    print()
    print("Per-source: needed -> granted")
    for source in CONFIG["sources"]:
        need = sum(count_item(it) for it in store.get_by_source(source))
        print(f"  {source:14s} {need:5d} -> {allocation[source]:5d}"
              + ("   (CUT)" if allocation[source] < need else ""))

    # --- proof of correctness ---
    assert store_total > CONFIG["total_budget"], "store should overflow the budget"
    assert ctx_total <= CONFIG["total_budget"], "built context must fit the budget"

    ctx_texts = [m["content"] for m in ctx]
    for it in store.get_by_source("system_policy") + store.get_by_source("pinned"):
        assert it.content in ctx_texts, f"critical item lost: {it.content[:50]}"

    print("\nAll asserts passed: over-budget state, under-budget context, policy+pinned intact.")


if __name__ == "__main__":
    main()
