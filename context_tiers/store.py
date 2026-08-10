"""ContextStore: the agent's full state. Everything gets stored here in full.
The context sent to the model is computed FROM this — never equal to it."""

from .tokens import Item, count_item


class ContextStore:
    def __init__(self) -> None:
        self.items: list[Item] = []

    def add(self, role: str, content: str, source: str, turn: int) -> None:
        self.items.append(Item(role=role, content=content, source=source, turn=turn))

    def get_by_source(self, source: str) -> list[Item]:
        return [it for it in self.items if it.source == source]

    def all_sources(self) -> list[str]:
        seen: list[str] = []
        for it in self.items:
            if it.source not in seen:
                seen.append(it.source)
        return seen

    def total_tokens(self) -> int:
        return sum(count_item(it) for it in self.items)

    def reclassify(self, before_turn: int, mapping: dict[str, str]) -> int:
        """Move items older than before_turn into a different source.
        Returns how many items moved."""
        moved = 0
        for it in self.items:
            if it.turn < before_turn and it.source in mapping:
                it.source = mapping[it.source]
                moved += 1
        return moved


if __name__ == "__main__":
    store = ContextStore()
    store.add("system", "Airline policy: basic economy cannot be changed.", "system_policy", 0)
    store.add("user", "I want to change my flight.", "recent_turns", 1)
    store.add("assistant", "Sure, let me look up your reservation.", "recent_turns", 1)
    store.add("tool", "Reservation ABC123: BOS->ORD, basic economy.", "active_tools", 1)
    store.add("user", "Use my miles, not the credit card.", "pinned", 2)
    print(f"Sources: {store.all_sources()}")
    print(f"Total tokens in store: {store.total_tokens()}")
