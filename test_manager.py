"""Tests for the ContextManager facade and its decision trace."""

from context_tiers import ContextManager


def test_three_line_usage():
    cm = ContextManager(budget=200)
    cm.add("system", "Policy: refunds go to the original payment method.")
    cm.add("user", "Pay with miles, not the credit card.")
    messages = cm.build()
    assert isinstance(messages, list) and messages, "build must return messages"
    assert any("Policy" in m["content"] for m in messages)
    print("ok: three line usage works")


def test_auto_pin():
    cm = ContextManager(budget=500)
    cm.add("user", "My companion must stay on the same booking.")
    pinned = cm.store.get_by_source("pinned")
    assert pinned, "hard constraint should be auto pinned"
    cm2 = ContextManager(budget=500, auto_pin=False)
    cm2.add("user", "My companion must stay on the same booking.")
    assert not cm2.store.get_by_source("pinned"), "auto_pin=False must disable pinning"
    print("ok: auto pinning on by default, off when disabled")


def test_aging():
    cm = ContextManager(budget=1000)
    cm.add("user", "old question", turn=1)
    cm.add("tool", "old tool result", turn=2)
    cm.add("user", "new question", turn=9)
    cm.age(before_turn=5)
    assert cm.store.get_by_source("old_turns"), "old user turn should demote"
    assert cm.store.get_by_source("spent_tools"), "old tool result should demote"
    assert any(it.content == "new question" for it in cm.store.get_by_source("recent_turns"))
    print("ok: aging demotes recent->old and active->spent")


def test_trace_explains_the_build():
    cm = ContextManager(budget=300)
    cm.add("system", "Policy: basic economy cannot be modified. " * 2)
    cm.add("user", "Pay with miles, not the card.")
    for t in range(1, 9):
        cm.add("user", f"(t{t}) question about options and baggage " * 8, turn=t)
        cm.add("tool", f"(t{t}) big tool dump " * 30, turn=t)
    cm.age(before_turn=7)

    messages, trace = cm.build(trace=True)

    assert trace["context_tokens"] <= trace["budget"], "trace must confirm budget held"
    assert trace["state_tokens"] > trace["budget"], "state should overflow in this test"
    actions = {s: v["action"] for s, v in trace["sources"].items()}
    assert actions.get("system_policy") == "kept"
    assert actions.get("pinned") == "kept"
    assert any(a in ("summarized", "truncated", "dropped") for a in actions.values()), \
        "something must have been shrunk under this much pressure"
    # every source in the trace accounts for its numbers
    for s, v in trace["sources"].items():
        assert v["granted_tokens"] <= v["needed_tokens"] or v["action"] == "kept"
    print(f"ok: trace explains the build -> {actions}")


if __name__ == "__main__":
    test_three_line_usage()
    test_auto_pin()
    test_aging()
    test_trace_explains_the_build()
    print("\nAll facade tests passed.")
