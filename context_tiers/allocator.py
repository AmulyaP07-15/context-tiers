"""Budget allocator: decides how many tokens each source is ALLOWED this turn.
Tier 0 gets funded first, then tier 1, and so on until the budget runs out."""

from .tokens import count_item
from .store import ContextStore

DEFAULT_CONFIG = {
    "total_budget": 8000,
    "sources": {
        "system_policy": {"tier": 0, "strategy": "verbatim"},
        "pinned":        {"tier": 0, "strategy": "verbatim"},
        "recent_turns":  {"tier": 1, "strategy": "verbatim", "keep_last": 4},
        "active_tools":  {"tier": 2, "strategy": "verbatim"},
        "old_turns":     {"tier": 3, "strategy": "truncate_oldest"},
        "spent_tools":   {"tier": 4, "strategy": "summarize"},
    },
}


def allocate(store: ContextStore, config: dict) -> dict[str, int]:
    budget = config["total_budget"]
    sources_cfg = config["sources"]

    # sources in tier order (tier 0 first)
    ordered = sorted(sources_cfg.keys(), key=lambda s: sources_cfg[s]["tier"])

    allocation: dict[str, int] = {}
    remaining = budget
    for source in ordered:
        need = sum(count_item(it) for it in store.get_by_source(source))
        # Phase 2: a summarize-strategy source never claims more than its
        # summary target — compression frees budget for lower tiers
        if sources_cfg[source].get("strategy") == "summarize" and need > 0:
            need = min(need, sources_cfg[source].get("summary_budget", 120))
        granted = min(need, max(remaining, 0))
        allocation[source] = granted
        remaining -= granted
    return allocation


if __name__ == "__main__":
    store = ContextStore()
    small_config = {**DEFAULT_CONFIG, "total_budget": 100}
    store.add("system", "Policy: " + "rules and more rules. " * 10, "system_policy", 0)
    store.add("user", "Pinned constraint: pay with miles only.", "pinned", 1)
    store.add("user", "Recent question about seats. " * 5, "recent_turns", 5)
    store.add("tool", "Old giant search result. " * 30, "spent_tools", 2)
    print(f"Store needs {store.total_tokens()} tokens, budget is {small_config['total_budget']}")
    print(f"Allocation: {allocate(store, small_config)}")
