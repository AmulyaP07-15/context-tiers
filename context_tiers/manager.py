"""The facade. One class that wraps store -> pin -> allocate -> assemble,
so using the library is three lines instead of six imports.

    from context_tiers import ContextManager

    cm = ContextManager(budget=3000)
    cm.add("user", "Pay with miles, not the card.")
    messages, trace = cm.build(trace=True)

The trace is the part production people care about. Every build can explain
itself: what was kept, what was cut, what was summarized, and the token math.
An allocator you can't audit is just a different way to lose data silently.
"""

from typing import Any, Optional

from .tokens import count_item, count_tokens
from .store import ContextStore
from .allocator import allocate
from .assembler import build_context, context_tokens
from .pinner import HeuristicPinner
from .summarizer import HeuristicSummarizer

DEFAULT_SOURCES = {
    "system_policy": {"tier": 0, "strategy": "verbatim"},
    "pinned":        {"tier": 0, "strategy": "verbatim"},
    "recent_turns":  {"tier": 1, "strategy": "verbatim"},
    "active_tools":  {"tier": 2, "strategy": "verbatim"},
    "spent_tools":   {"tier": 3, "strategy": "summarize", "summary_budget": 120},
    "old_turns":     {"tier": 4, "strategy": "truncate_oldest"},
}

# where a message lands if the caller doesn't say
ROLE_DEFAULT_SOURCE = {
    "system": "system_policy",
    "user": "recent_turns",
    "assistant": "recent_turns",
    "tool": "active_tools",
}


class ContextManager:
    def __init__(
        self,
        budget: int = 3000,
        sources: Optional[dict] = None,
        store: Optional[ContextStore] = None,
        pinner: Optional[Any] = None,
        summarizer: Optional[Any] = None,
        auto_pin: bool = True,
    ):
        self.config = {"total_budget": budget, "sources": sources or DEFAULT_SOURCES}
        self.store = store if store is not None else ContextStore()
        self.pinner = pinner if pinner is not None else HeuristicPinner()
        self.summarizer = summarizer if summarizer is not None else HeuristicSummarizer()
        self.auto_pin = auto_pin
        self._turn = 0

    # ---- writing state -------------------------------------------------
    def add(self, role: str, content: str, source: Optional[str] = None,
            turn: Optional[int] = None) -> None:
        if turn is None:
            self._turn += 1
            turn = self._turn
        src = source or ROLE_DEFAULT_SOURCE.get(role, "old_turns")
        self.store.add(role, content, src, turn)
        if self.auto_pin and role == "user":
            for pin in self.pinner.extract(content):
                self.store.add("user", f"[pinned constraint] {pin}", "pinned", turn)

    def age(self, before_turn: int) -> int:
        """Demote content older than a turn. recent_turns -> old_turns,
        active_tools -> spent_tools. Call this as the conversation moves on.
        Returns the number of items demoted."""
        demote = {"recent_turns": "old_turns", "active_tools": "spent_tools"}
        return self.store.reclassify(before_turn, demote)

    # ---- reading state -------------------------------------------------
    def build(self, trace: bool = False):
        envelopes: list = []
        messages = build_context(self.store, self.config, summarizer=self.summarizer,
                                 collect_envelopes=envelopes)
        if not trace:
            return messages

        allocation = allocate(self.store, self.config)
        per_source = {}
        for source, cfg in self.config["sources"].items():
            items = self.store.get_by_source(source)
            if not items:
                continue
            need = sum(count_item(it) for it in items)
            granted = allocation.get(source, 0)
            if granted >= need:
                action = "kept"
            elif cfg["strategy"] == "summarize" and granted > 0:
                action = "summarized"
            elif granted > 0:
                action = "truncated"
            else:
                action = "dropped"
            per_source[source] = {
                "items": len(items), "needed_tokens": need,
                "granted_tokens": granted, "action": action,
                "strategy": cfg["strategy"], "tier": cfg["tier"],
            }
        trace_dict = {
            "budget": self.config["total_budget"],
            "state_tokens": self.store.total_tokens(),
            "context_tokens": context_tokens(messages),
            "sources": per_source,
            "summaries": envelopes,
        }
        return messages, trace_dict
