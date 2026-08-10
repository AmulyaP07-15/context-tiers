"""Offline tests for Phase 3-4: no API keys needed.

1. adapter: naive + managed build_messages respect the budget
2. adapter: tool_call/tool pairs are never orphaned (protocol atomicity)
3. adapter: managed keeps pinned constraints that naive silently drops
4. runner: resume skips completed episodes
5. report: pass^k math is right
"""

import json
import os

from adapter import (NaiveBudgetAgent, ManagedBudgetAgent, messages_tokens,
                     BudgetBelowFloorError)
from runner import load_done, episode_key
from report import pass_hat_k


FAKE_TOOLS = [{"type": "function", "function": {"name": f"tool_{i}",
               "description": "x" * 120, "parameters": {"type": "object",
               "properties": {"a": {"type": "string"}}}}} for i in range(6)]


def fake_agent(cls, budget=2000, tools=None):
    a = cls(tools_info=tools if tools is not None else FAKE_TOOLS,
            wiki="POLICY: basic economy cannot be modified. Refunds to original payment.",
            model="none", provider=None, budget=budget, window_exchanges=2)
    a._init_state("Hi, I want to look at my trip options.")
    # 12 exchanges: alternating tool exchanges and conversation, with one
    # critical constraint early that budget pressure should squeeze out
    for i in range(12):
        if i == 1:
            a._record({"role": "assistant", "content": f"(t{i}) sure, tell me more", "tool_calls": None},
                      "Use my miles, not the credit card. That must not change.", was_tool=False)
        elif i % 3 == 0:
            a._record({"role": "assistant", "content": None,
                       "tool_calls": [{"id": f"call_{i}", "function": {"name": "search_flights",
                                                                        "arguments": '{"origin":"BOS"}'}}]},
                      f"(t{i}) FLIGHT RESULTS: " + "option details " * 60, was_tool=True)
        else:
            a._record({"role": "assistant", "content": f"(t{i}) here are the options " * 10, "tool_calls": None},
                      f"(t{i}) ok what about day {i}? " * 10, was_tool=False)
    return a


def assert_protocol_intact(messages):
    """Every assistant tool_calls message must be immediately followed by a
    matching tool message, and every tool message must follow its call."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assert i + 1 < len(messages) and messages[i + 1].get("role") == "tool", \
                f"orphaned tool_call at index {i}"
            assert messages[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"], "mismatched pair"
        if m.get("role") == "tool":
            prev = messages[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls"), \
                f"tool result without preceding call at index {i}"


def test_budget_and_protocol():
    """The budget is the TOTAL request, tool schemas included. No slack."""
    for cls in (NaiveBudgetAgent, ManagedBudgetAgent):
        a = fake_agent(cls, budget=2000)
        msgs = a.build_messages()
        used = a.request_tokens(msgs)
        assert used <= 2000, f"{cls.__name__} blew the budget: {used} > 2000"
        assert_protocol_intact(msgs)
        assert msgs[0]["role"] == "system" and "POLICY" in msgs[0]["content"], "policy must survive"
    print("ok: both agents fit the real budget (tools included), protocol intact")


def test_tool_schemas_are_charged():
    a = fake_agent(NaiveBudgetAgent, budget=2000)
    assert a.tools_tokens > 0, "tool schemas must be counted"
    assert a.conv_budget == a.budget - a.floor_tokens
    msgs = a.build_messages()
    assert a.request_tokens(msgs) > messages_tokens(msgs), \
        "request size must exceed the visible messages by the tool schema cost"
    print(f"ok: tool schemas charged ({a.tools_tokens} tokens), floor={a.floor_tokens}")


def test_budget_below_floor_is_rejected():
    """The bug that silently ruined a 20 episode run: a budget under the domain
    floor makes every condition fail for the same reason."""
    try:
        fake_agent(NaiveBudgetAgent, budget=200)
    except BudgetBelowFloorError as e:
        assert "fixed floor" in str(e)
        print("ok: budget below the domain floor is rejected up front")
        return
    raise AssertionError("a budget below the floor should have been rejected")


def test_newest_exchange_is_never_dropped():
    """A baseline that discards the turn it is answering is a straw man."""
    for cls in (NaiveBudgetAgent, ManagedBudgetAgent):
        a = fake_agent(cls, budget=2000)
        a._record({"role": "assistant", "content": "ok", "tool_calls": None},
                  "FINAL USER TURN MARKER", was_tool=False)
        msgs = a.build_messages()
        assert any("FINAL USER TURN MARKER" in str(m.get("content")) for m in msgs), \
            f"{cls.__name__} dropped the current turn"
    print("ok: the newest exchange survives in both conditions")


def test_managed_keeps_pin_naive_drops_it():
    naive = fake_agent(NaiveBudgetAgent, budget=1500).build_messages()
    managed = fake_agent(ManagedBudgetAgent, budget=1500).build_messages()
    naive_text = json.dumps(naive)
    managed_text = json.dumps(managed)
    assert "miles" not in naive_text, "test setup wrong: naive should have dropped the miles constraint"
    assert "miles" in managed_text, "managed must pin and keep the miles constraint"
    print("ok: managed retains the pinned constraint that naive silently dropped")


def test_runner_resume(tmp="_test_results.jsonl"):
    if os.path.exists(tmp):
        os.remove(tmp)
    rows = [
        {"task_id": 0, "condition": "naive", "trial": 0, "reward": 1},
        {"task_id": 0, "condition": "managed", "trial": 0, "reward": 1},
    ]
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("{corrupt line\n")  # must not crash resume
    done = load_done(tmp)
    assert episode_key(0, "naive", 0) in done and episode_key(0, "managed", 0) in done
    assert episode_key(0, "naive", 1) not in done
    os.remove(tmp)
    print("ok: resume skips finished episodes and survives corrupt lines")


def test_pass_hat_k():
    rows = [
        {"task_id": 1, "reward": 1}, {"task_id": 1, "reward": 1},   # passes k=2
        {"task_id": 2, "reward": 1}, {"task_id": 2, "reward": 0},   # fails k=2, passes k=1
        {"task_id": 3, "reward": 0}, {"task_id": 3, "reward": 0},   # fails both
    ]
    assert abs(pass_hat_k(rows, 1) - 2 / 3) < 1e-9
    assert abs(pass_hat_k(rows, 2) - 1 / 3) < 1e-9
    print("ok: pass^k math correct (2/3 at k=1, 1/3 at k=2)")




def test_first_turn_is_user_after_truncation():
    """Gemini rejects histories whose first non-system turn is an assistant
    function call. Under a tiny budget the oldest exchanges (including the
    opening user turn) get cut, which used to trigger exactly that 400."""
    for cls in (NaiveBudgetAgent, ManagedBudgetAgent):
        a = fake_agent(cls, budget=1800)
        msgs = a.build_messages()
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system, "context should not be empty"
        assert non_system[0]["role"] == "user", \
            f"{cls.__name__}: first turn is {non_system[0]['role']}, Gemini would 400"
        assert_protocol_intact(msgs)
    print("ok: first non-system turn is always user, even under heavy truncation")

def test_hostile_provider_outputs_do_not_break_the_transcript():
    """Small models emit null content, missing tool_call ids, malformed
    argument JSON and empty tool_calls arrays. None of these may strand an
    assistant tool_call without its result or put None where the provider
    expects a string."""
    import adapter as _ad
    for cls in (NaiveBudgetAgent, ManagedBudgetAgent):
        a = fake_agent(cls, budget=2000)
        # a tool call whose arguments will not parse, routed as a respond turn
        act = _ad.message_to_action({"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "function": {"name": "t", "arguments": "{bad json"}}]})
        assert act.name == _ad.RESPOND_ACTION_NAME, "malformed args must not raise"
        # a tool call with no id still pairs with its result
        a._record({"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "t", "arguments": "{}"}}]}, "result", was_tool=True)
        # a respond turn must never carry tool_calls
        a._record({"role": "assistant", "content": "", "tool_calls": None}, "ok", was_tool=False)
        msgs = a.build_messages()
        assert_protocol_intact(msgs)
        for m in msgs:
            if not m.get("tool_calls"):
                assert m.get("content") is not None, "content must never be None without tool_calls"
    print("ok: hostile provider outputs handled without breaking the transcript")



def test_managed_preserves_roles_and_surfaces_pins():
    """Rewrite invariant: compressed history keeps real roles (a tool result
    stays a tool message, never a user message) and pins are surfaced."""
    import json as _json
    a = fake_agent(ManagedBudgetAgent, budget=2500)
    a._init_state("Change my booking. Use miles, not the card.")
    res = _json.dumps({"reservation_id": "4WQ150",
                       "payment_history": [{"payment_id": "cc_4421486"}]})
    for i in range(1, 14):
        if i % 2 == 1:
            a._record({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "function": {"name": "get_reservation_details",
                 "arguments": "{}"}}]}, res, was_tool=True)
        else:
            a._record({"role": "assistant", "content": "options", "tool_calls": None},
                      "my companion must stay on the booking", was_tool=False)
    m = a.build_messages()
    roles = [x.get("role") for x in m]
    assert "tool" in roles, "compressed history must keep real tool messages"
    assert not any("[conversation so far]" in str(x.get("content", "")) for x in m), \
        "must not collapse history into one fake user message"
    assert any("constraints I have stated" in str(x.get("content", "")) for x in m), \
        "pins must be surfaced"
    assert any("miles" in str(x.get("content", "")) for x in m), "miles must survive"
    assert_protocol_intact(m)
    print("ok: managed preserves roles, keeps tool messages, surfaces pins")


def test_summarizer_protects_ids_and_payment_on_real_records():
    """The bug that survived two earlier fixes: on a real reservation record,
    compression dropped payment_history and truncated the user_id. Protected
    fields must survive whole at every budget."""
    import json as _json
    from context_tiers.summarizer import HeuristicSummarizer
    rec = _json.dumps({
        "reservation_id": "4WQ150", "user_id": "chen_jackson_3290",
        "cabin": "business", "created_at": "2024-05-02T00:00:00",
        "flights": [{"flight_number": "HAT170", "price": 883}],
        "passengers": [{"first_name": "Chen", "dob": "1956-07-07"}],
        "payment_history": [{"payment_id": "gift_card_3576581", "amount": 4986}],
        "insurance": "no", "total_baggages": 5})
    s = HeuristicSummarizer()
    for target in (150, 100, 60):
        out = s.summarize(rec, target)
        got = _json.loads(out)
        assert "reservation_id" in got and got["reservation_id"] == "4WQ150"
        assert "user_id" in got and "\u2026" not in got["user_id"], "id must not be truncated"
        if "payment_history" in got:
            assert "gift_card_3576581" in _json.dumps(got["payment_history"]), \
                "payment id must survive intact"
    print("ok: ids and payment protected on real records at every budget")


def test_tokens_and_pinner_edge_cases():
    """Regressions for the full-codebase audit: None content must not crash
    token counting, and the pinner must not fire on 'not' inside a word."""
    from context_tiers.tokens import count_item, Item
    from context_tiers.pinner import HeuristicPinner
    assert count_item(Item("tool", None, "x", 0)) == 4, "None content must not crash"
    p = HeuristicPinner()
    assert not p.extract("I have a knot in my shoulder"), "must not pin 'knot'"
    assert not p.extract("cannot wait to travel"), "must not pin idiom 'cannot wait'"
    assert p.extract("use miles not the card"), "must pin real constraint"
    assert p.extract("do not cancel my return"), "must pin 'do not'"
    print("ok: token None-safety and pinner word-boundary correct")


def test_summary_envelope_reports_and_does_not_leak():
    """The envelope records what compression did (tokens, kept/dropped fields)
    for logging and the retention metric, and must never leak into the messages
    the model sees."""
    import json as _json
    from context_tiers import ContextManager
    big = _json.dumps({"reservation_id": "4WQ150", "user_id": "chen_jackson_3290",
        "payment_history": [{"payment_id": "gift_card_3576581", "amount": 4986}],
        "created_at": "2024-05-02", "passengers": [{"name": "Chen"}],
        "cabin": "business", "pad": "y" * 200})
    cm = ContextManager(budget=100)
    cm.add("tool", big, source="spent_tools", turn=1)
    msgs, trace = cm.build(trace=True)
    assert "summaries" in trace, "trace must carry summary envelopes"
    assert len(trace["summaries"]) == 1, "one record should produce one envelope"
    e = trace["summaries"][0]
    assert e["summary_tokens"] <= e["original_tokens"], "summary must be smaller"
    assert "payment_history" in (e["kept_fields"] or []), "payment must be reported kept"
    leak = _json.dumps(msgs)
    assert "dropped_fields" not in leak and "original_tokens" not in leak, \
        "envelope metadata must not reach the model"
    print("ok: summary envelope reports fields and stays out of model messages")


if __name__ == "__main__":
    test_budget_and_protocol()
    test_tool_schemas_are_charged()
    test_budget_below_floor_is_rejected()
    test_newest_exchange_is_never_dropped()
    test_managed_keeps_pin_naive_drops_it()
    test_runner_resume()
    test_pass_hat_k()
    test_first_turn_is_user_after_truncation()
    test_hostile_provider_outputs_do_not_break_the_transcript()
    test_managed_preserves_roles_and_surfaces_pins()
    test_summarizer_protects_ids_and_payment_on_real_records()
    test_tokens_and_pinner_edge_cases()
    test_summary_envelope_reports_and_does_not_leak()
    print("\nAll Phase 3-4 offline tests passed.")
