"""Token counting and the basic message unit (Item)."""

from dataclasses import dataclass


@dataclass
class Item:
    role: str      # "system", "user", "assistant", "tool"
    content: str   # the actual text
    source: str    # which bucket it belongs to, e.g. "system_policy", "pinned"
    turn: int      # conversation turn it was created on


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken; fall back to a chars/4 estimate
    if tiktoken can't load (e.g. no network for its encoding file)."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def count_item(item: Item) -> int:
    """Tokens for one item: its content plus ~4 tokens of per-message overhead.
    content may be None for an assistant tool-call message, so coerce first."""
    return count_tokens(item.content or "") + 4


if __name__ == "__main__":
    a = Item(role="user", content="Book me a flight from Boston to Chicago next Friday.", source="recent_turns", turn=1)
    b = Item(role="system", content="You are a helpful airline booking agent. Follow policy strictly.", source="system_policy", turn=0)
    print(f"Item A tokens: {count_item(a)}")
    print(f"Item B tokens: {count_item(b)}")
