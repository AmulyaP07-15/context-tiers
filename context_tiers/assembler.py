"""Assembler: shrinks each source to its allowed budget and builds the
final messages list that would be sent to the model."""

from .tokens import Item, count_item, count_tokens
from .store import ContextStore
from .allocator import allocate

# fixed final ordering: critical stuff at the edges, compressible stuff in the middle
ASSEMBLY_ORDER = ["system_policy", "pinned", "old_turns", "spent_tools", "active_tools", "recent_turns"]


def shrink(items: list[Item], allowed_tokens: int) -> list[Item]:
    """Keep items newest-first until the budget is full; return them in original order."""
    kept: list[Item] = []
    used = 0
    for it in reversed(items):
        cost = count_item(it)
        if used + cost > allowed_tokens:
            break
        kept.append(it)
        used += cost
    kept.reverse()
    return kept


def build_context(store: ContextStore, config: dict, summarizer=None,
                  collect_envelopes=None) -> list[dict]:
    """Phase 2: per-source strategy dispatch.
    - verbatim / truncate_oldest: keep newest items that fit (shrink)
    - summarize: if over budget, replace the source's items with ONE summary item
    """
    allocation = allocate(store, config)

    # any source the caller configured but that is not in the fixed ordering
    # would otherwise be allocated budget and then never emitted, which loses
    # data silently. Place unknown sources by tier, just before recent_turns.
    known = set(ASSEMBLY_ORDER)
    extra = sorted((s for s in config["sources"] if s not in known),
                   key=lambda s: config["sources"][s].get("tier", 99))
    order = ASSEMBLY_ORDER[:-1] + extra + ASSEMBLY_ORDER[-1:] if extra else ASSEMBLY_ORDER

    messages: list[dict] = []
    envelopes: list[dict] = []
    for source in order:
        items = store.get_by_source(source)
        if not items:
            continue
        allowed = allocation.get(source, 0)
        strategy = config["sources"].get(source, {}).get("strategy", "verbatim")
        need = sum(count_item(it) for it in items)

        if strategy == "summarize" and need > allowed and allowed > 0:
            if summarizer is None:
                from .summarizer import HeuristicSummarizer
                summarizer = HeuristicSummarizer()
            joined = "\n".join(it.content or "" for it in items)
            prefix = f"[summary of {source}] "
            target = allowed - 4 - count_tokens(prefix)  # pay for overhead + prefix
            if hasattr(summarizer, "summarize_with_envelope"):
                summary, env = summarizer.summarize_with_envelope(joined, max(target, 1))
                env["source"] = source
                envelopes.append(env)
            else:
                summary = summarizer.summarize(joined, max(target, 1))
            messages.append({"role": "tool", "content": prefix + summary})
        else:
            kept = shrink(items, allowed)
            # A tier-0 verbatim source (policy, pinned constraints) must never
            # vanish. If nothing fit the allocation but the source is critical,
            # emit at least the newest item rather than silently dropping it.
            tier = config["sources"].get(source, {}).get("tier", 99)
            if not kept and tier == 0 and items:
                kept = [items[-1]]
            for it in kept:
                messages.append({"role": it.role, "content": it.content or ""})
    if collect_envelopes is not None:
        collect_envelopes.extend(envelopes)
    return messages


def context_tokens(messages: list[dict]) -> int:
    return sum(count_item(Item(role=m["role"], content=m["content"], source="", turn=0)) for m in messages)


if __name__ == "__main__":
    store = ContextStore()
    store.add("system", "Policy: no changes to basic economy tickets. " * 3, "system_policy", 0)
    for t in range(1, 11):
        store.add("user", f"Turn {t}: user asks something about the booking, seats, or bags. " * 3, "old_turns" if t < 7 else "recent_turns", t)
        store.add("assistant", f"Turn {t}: agent answers with details and options. " * 3, "old_turns" if t < 7 else "recent_turns", t)
    store.add("tool", "Huge flight search dump. " * 40, "spent_tools", 2)

    config = {"total_budget": 300, "sources": {
        "system_policy": {"tier": 0, "strategy": "verbatim"},
        "pinned":        {"tier": 0, "strategy": "verbatim"},
        "recent_turns":  {"tier": 1, "strategy": "verbatim"},
        "active_tools":  {"tier": 2, "strategy": "verbatim"},
        "old_turns":     {"tier": 3, "strategy": "truncate_oldest"},
        "spent_tools":   {"tier": 4, "strategy": "summarize"},
    }}
    ctx = build_context(store, config)
    print(f"Store total:   {store.total_tokens()} tokens")
    print(f"Built context: {context_tokens(ctx)} tokens (budget {config['total_budget']})")
    print(f"Messages in context: {len(ctx)} of {len(store.items)} stored")
