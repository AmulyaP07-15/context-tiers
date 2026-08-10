"""Phase 3: tau-bench adapter.

Two agents, both under the SAME token budget, differing only in how they
decide what stays in context:

  NaiveBudgetAgent    drop-oldest-exchange until the context fits
  ManagedBudgetAgent  our tiered allocator (policy/pins protected,
                      spent tool results summarized, old turns truncated)

Design constraints this file is built around (from reading tau-bench source):

1. PROTOCOL ATOMICITY. An assistant message with tool_calls must be
   immediately followed by its matching tool-result message. So we never
   drop half a pair. The unit of history is an EXCHANGE:
     - a (user, assistant-respond) pair, or
     - an (assistant tool_call, tool result) pair.

2. FLATTEN BEFORE MANAGING. Only the recent window keeps raw message dicts
   (protocol-intact). Anything older is flattened to plain text first, and
   the allocator works on that text. Text has no protocol to violate.

3. SAME LOOP AS UPSTREAM. solve() mirrors tau_bench's ToolCallingAgent so
   the only experimental variable is context handling.
"""

import json
from typing import Any, Dict, List, Optional

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from context_tiers.tokens import count_tokens
from context_tiers.store import ContextStore
from context_tiers.assembler import build_context
from context_tiers.pinner import HeuristicPinner
from context_tiers.summarizer import HeuristicSummarizer

MANAGED_CONFIG = {
    "total_budget": 3000,   # overridden by --budget at run time
    "sources": {
        "system_policy": {"tier": 0, "strategy": "verbatim"},
        "pinned":        {"tier": 0, "strategy": "verbatim"},
        "spent_tools":   {"tier": 1, "strategy": "summarize", "summary_budget": 150},
        "old_turns":     {"tier": 2, "strategy": "truncate_oldest"},
    },
}


MALFORMED_TOOL_CALLS = {"count": 0}


def message_to_action(message: Dict[str, Any]) -> Action:
    """Same behavior as tau_bench's ToolCallingAgent, with the failure modes
    a small model actually produces handled instead of raised.

    Upstream calls json.loads on the arguments string directly. A model that
    emits slightly malformed JSON therefore kills the episode, which costs a
    retry cycle and a lost data point. Both conditions get the same handling,
    so the comparison stays fair, and the occurrences are counted so the
    write up can report them honestly."""
    calls = message.get("tool_calls")
    if calls and len(calls) > 0 and calls[0].get("function") is not None:
        fn = calls[0]["function"]
        raw = fn.get("arguments") or "{}"
        try:
            kwargs = json.loads(raw)
            if not isinstance(kwargs, dict):
                raise ValueError("arguments must decode to an object")
        except (json.JSONDecodeError, ValueError, TypeError):
            MALFORMED_TOOL_CALLS["count"] += 1
            return Action(name=RESPOND_ACTION_NAME, kwargs={
                "content": "I hit an internal formatting error. Could you restate that?"})
        return Action(name=fn.get("name", ""), kwargs=kwargs)
    # content can legitimately come back as None; the env expects a string
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content") or ""})


def flatten_exchange(exchange: List[Dict[str, Any]]) -> tuple[str, str]:
    """Turn an exchange (list of raw messages) into (text, source).
    Tool exchanges -> spent_tools. Conversation exchanges -> old_turns."""
    parts: List[str] = []
    is_tool = False
    for m in exchange:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            is_tool = True
            fn = m["tool_calls"][0]["function"]
            parts.append(f"agent called {fn['name']}({fn['arguments']})")
        elif role == "tool":
            is_tool = True
            parts.append(f"result: {m.get('content', '')}")
        else:
            parts.append(f"{role}: {m.get('content', '')}")
    return " | ".join(parts), ("spent_tools" if is_tool else "old_turns")


def messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += 4 + count_tokens(str(m.get("content") or ""))
        for tc in (m.get("tool_calls") or []):
            total += count_tokens(str(tc.get("function", {})))
    return total



def repair_tool_protocol(messages):
    """Every tool message must sit immediately after an assistant message that
    carries tool_calls. Compression can drop or separate an assistant call
    while keeping its tool result, which leaves an orphan tool message that
    strict providers (Azure) reject with a 400. Drop any such orphan, and drop
    an assistant tool_calls message whose tool result did not survive, so no
    half pair is ever sent."""
    out = []
    n = len(messages)
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            prev = out[-1] if out else None
            if not prev or prev.get("role") != "assistant" or not prev.get("tool_calls"):
                continue  # orphan tool result, drop it
            out.append(m)
        elif role == "assistant" and m.get("tool_calls"):
            # keep only if the very next message is its tool result
            nxt = messages[i + 1] if i + 1 < n else None
            if not nxt or nxt.get("role") != "tool":
                # assistant call with no result following: keep the content but
                # strip the dangling tool_calls so it reads as a plain turn
                mm = {k: v for k, v in m.items() if k != "tool_calls"}
                if not mm.get("content"):
                    continue
                mm["content"] = mm.get("content") or ""
                out.append(mm)
            else:
                out.append(m)
        else:
            out.append(m)
    return out


def ensure_starts_with_user(messages):
    """Gemini requires the first non-system turn to be a user turn, and a
    function call turn to follow a user or function-response turn. After
    truncation the history can start with an assistant tool_call, which
    Gemini rejects with a 400. Insert a tiny user stub when that happens."""
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            continue
        if m.get("role") != "user":
            messages.insert(i, {"role": "user", "content": "[earlier conversation truncated to fit the context budget]"})
        break
    return messages


ACTION_GUIDANCE = (
    "\n\nGuidance: look information up with tools instead of asking the "
    "customer for what you can retrieve. Once you have the goal and the required "
    "facts, act rather than gathering more optional details. Still confirm once "
    "before any booking, cancellation, or payment.")

MIN_CONVERSATION_ROOM = 500   # tokens that must remain after the fixed floor


class BudgetBelowFloorError(ValueError):
    """Raised when the requested budget cannot even hold the domain's fixed
    overhead (policy document plus tool schemas) with room to converse.
    Running below the floor makes every condition fail for the same reason,
    which produces a floor effect rather than a comparison."""


class _BudgetAgentBase(Agent):
    """Shared solve() loop; subclasses implement build_messages().

    budget is the TOTAL request size, including the tool schemas that get
    sent on every call. Tool schemas are invisible in the messages list but
    are charged by the provider, so excluding them means the budget governs
    only part of what is actually sent.
    """

    def __init__(self, tools_info, wiki, model, provider,
                 temperature: float = 0.0, budget: int = 8000, window_exchanges: int = 4,
                 strict_budget: bool = True):
        self.tools_info = tools_info
        self.wiki = (wiki or "") + ACTION_GUIDANCE
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.budget = budget
        self.window_exchanges = window_exchanges

        self.tools_tokens = count_tokens(json.dumps(tools_info)) if tools_info else 0
        self.system_tokens = 4 + count_tokens(wiki)
        self.floor_tokens = self.tools_tokens + self.system_tokens
        self.conv_budget = budget - self.floor_tokens
        if strict_budget and self.conv_budget < MIN_CONVERSATION_ROOM:
            raise BudgetBelowFloorError(
                f"budget={budget} leaves {self.conv_budget} tokens for conversation. "
                f"This domain has a fixed floor of {self.floor_tokens} tokens "
                f"(policy {self.system_tokens} + tool schemas {self.tools_tokens}). "
                f"Use a budget of at least {self.floor_tokens + MIN_CONVERSATION_ROOM}."
            )

    def request_tokens(self, messages) -> int:
        """What the provider actually receives: messages plus tool schemas."""
        return messages_tokens(messages) + self.tools_tokens

    def _fit_exchanges(self, exchanges, budget):
        """Pack newest-first. The newest exchange is always kept, even if it
        alone exceeds the budget, because dropping the turn the agent is
        currently answering is not a truncation strategy any real agent uses."""
        kept, used = [], 0
        for i, ex in enumerate(reversed(exchanges)):
            cost = messages_tokens(ex)
            if i > 0 and used + cost > budget:
                break
            kept.append(ex)
            used += cost
        kept.reverse()
        return kept, used

    TRUNC_MARK = " ...[truncated to fit context budget]"

    def _truncate_largest(self, msgs):
        """Last resort. A single tool result can be bigger than the whole
        window, and neither dropping the turn being answered nor blowing the
        cap is acceptable, so the biggest message body gets cut instead.
        The policy is never cut, and tool_calls structures are never touched,
        so the request stays protocol valid."""
        msgs = [dict(m) for m in msgs]
        for _ in range(60):
            over = self.request_tokens(msgs) - self.budget
            if over <= 0:
                return msgs
            idx, longest = None, 0
            for i, m in enumerate(msgs):
                if m.get("role") == "system":
                    continue
                c = m.get("content")
                if isinstance(c, str) and len(c) > longest:
                    idx, longest = i, len(c)
            if idx is None or longest <= len(self.TRUNC_MARK) + 20:
                return msgs
            body = msgs[idx]["content"]
            cut = min(len(body) - 1, max(60, int(over * 4 * 1.2)))
            msgs[idx]["content"] = body[:max(1, len(body) - cut)].rstrip() + self.TRUNC_MARK
        return msgs

    def _finalize(self, system, middle_msgs, recent_raw, protected=None):
        """Assemble, repair turn order, then enforce the cap for real: shed the
        compressible middle first, and if that is not enough, cut the largest
        remaining body rather than exceed the budget.

        protected messages (pinned hard constraints) are placed right after the
        system message and are NEVER shed. They are the whole point of the
        library, so if space is that tight the recent window gives way before a
        stated constraint does."""
        protected = protected or []
        middle = list(middle_msgs)
        while True:
            msgs = ensure_starts_with_user(repair_tool_protocol(
                [system] + protected + middle + recent_raw))
            if self.request_tokens(msgs) <= self.budget:
                return msgs
            if middle:
                middle = middle[1:]
                continue
            # middle exhausted and still over: shed oldest recent exchanges
            # before ever touching the protected constraints
            if recent_raw:
                recent_raw = recent_raw[2:] if len(recent_raw) > 2 else []
                continue
            return self._truncate_largest(msgs)

    # ---- exchange bookkeeping ----------------------------------------
    def _init_state(self, first_obs: str) -> None:
        self.exchanges: List[List[Dict[str, Any]]] = [[{"role": "user", "content": first_obs}]]
        self.turn = 0

    def _record(self, assistant_msg: Dict[str, Any], env_obs: str, was_tool: bool) -> None:
        self.turn += 1
        if was_tool:
            call = assistant_msg["tool_calls"][0]
            # some providers omit the id; the pairing only has to be internally
            # consistent, so synthesize one and write it back so the assistant
            # message and its result still match
            if not call.get("id"):
                call["id"] = f"call_{self.turn}"
            tool_msg = {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call.get("function", {}).get("name", "unknown_tool"),
                "content": env_obs if env_obs is not None else "",
            }
            self.exchanges.append([assistant_msg, tool_msg])
        else:
            self.exchanges.append([assistant_msg, {"role": "user", "content": env_obs}])

    def _window_split(self) -> tuple[list, list]:
        """(old exchanges to manage, recent exchanges kept raw)."""
        w = self.window_exchanges
        return self.exchanges[:-w] if len(self.exchanges) > w else [], self.exchanges[-w:]

    # ---- the loop (mirrors upstream ToolCallingAgent.solve) -----------
    def solve(self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        info = env_reset_res.info.model_dump()
        reward = 0.0
        self._init_state(env_reset_res.observation)
        self.peak_context_tokens = 0
        self.total_context_tokens = 0
        self.steps = 0

        for _ in range(max_num_steps):
            messages = self.build_messages()
            ctx_tokens = self.request_tokens(messages)
            self.peak_context_tokens = max(self.peak_context_tokens, ctx_tokens)
            self.total_context_tokens += ctx_tokens
            self.steps += 1

            kwargs = dict(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
            )
            # gpt-5 models reject temperature != 1. Only pass temperature when
            # the model accepts a custom value; otherwise omit it and let the
            # provider use its default.
            if "gpt-5" not in str(self.model):
                kwargs["temperature"] = self.temperature
            res = completion(**kwargs)
            if not getattr(res, "choices", None):
                raise RuntimeError(
                    f"provider returned no choices for model {self.model}; "
                    "this is a provider side failure, not a context bug")
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params.get("response_cost") or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                self._record(next_message, env_response.observation, was_tool=True)
            else:
                # A respond turn must not carry tool_calls. This matters for the
                # malformed-arguments fallback above: the call was never made,
                # so leaving it in the transcript strands an assistant tool_call
                # with no matching result and the next request is rejected.
                # Record what the environment actually received.
                next_message = {**next_message, "tool_calls": None,
                                "content": action.kwargs.get("content") or ""}
                self._record(next_message, env_response.observation, was_tool=False)

            if env_response.done:
                break

        flat = [m for ex in self.exchanges for m in ex]
        return SolveResult(reward=reward, info=info, messages=flat, total_cost=total_cost)

    def build_messages(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NaiveBudgetAgent(_BudgetAgentBase):
    """Baseline: system + as many newest whole exchanges as fit the budget.
    Oldest exchanges silently fall off. This is what most agents do."""

    def build_messages(self) -> List[Dict[str, Any]]:
        system = {"role": "system", "content": self.wiki}
        kept, _ = self._fit_exchanges(self.exchanges, self.conv_budget)
        return self._finalize(system, [], [m for ex in kept for m in ex])


class ManagedBudgetAgent(_BudgetAgentBase):
    """Compress the old part of the conversation while preserving message
    roles and structure. Old tool results are shortened by a field aware
    summarizer but stay tool messages. Old assistant and user turns stay in
    their own roles. Hard user constraints are pinned and surfaced up front.

    The design principle behind the rewrite: compression must not change what
    kind of thing a message is. A tool result rendered as a user message tells
    the model the customer said something the tools actually returned, which
    is worse than useless. Keep the role, shrink the content.
    """

    # how aggressively an old tool result may be summarized, in tokens
    OLD_TOOL_SUMMARY_BUDGET = 150

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinner = HeuristicPinner()
        self.summarizer = HeuristicSummarizer()

    def _extract_pins(self):
        """Collect hard constraints from every user turn in the whole history,
        recent window included, so a constraint never goes unprotected just
        because it was stated recently. Deduplicated, order preserved."""
        pins, seen = [], set()
        for ex in self.exchanges:
            for m in ex:
                if m.get("role") == "user":
                    for pin in self.pinner.extract(str(m.get("content") or "")):
                        if pin not in seen:
                            seen.add(pin)
                            pins.append(pin)
        return pins

    def _compress_old(self, old, budget):
        """Compress old exchanges into REAL role tagged messages that fit the
        budget. Newest first so the most recent old context survives. Tool
        results are summarized in place and stay tool messages; assistant and
        user turns keep their roles, with long bodies shortened.

        The per tool summary target is adaptive. A tool result is only crushed
        to the floor when space is tight. When the remaining budget has room,
        the newest old tool results keep more of their content, because
        discarding detail the budget did not require discarding makes the
        managed agent needlessly worse informed."""
        out_rev = []
        used = 0
        for ex in reversed(old):
            # how much room is left for this block, capped so one block cannot
            # eat the whole budget but allowed to exceed the 150 floor when free
            room_left = max(budget - used, 0)
            tool_target = max(self.OLD_TOOL_SUMMARY_BUDGET, min(room_left, 400))
            block = []
            for m in ex:
                role = m.get("role")
                if role == "tool":
                    content = self.summarizer.summarize(
                        str(m.get("content") or ""), tool_target)
                    nm = {**m, "content": content}
                elif role == "assistant" and m.get("tool_calls"):
                    # keep the call intact so the following tool message still
                    # has its partner; only trim any prose content
                    nm = dict(m)
                    if nm.get("content"):
                        nm["content"] = self.summarizer.summarize(str(nm["content"]), 60)
                else:
                    body = str(m.get("content") or "")
                    nm = {**m, "content": self.summarizer.summarize(body, 80) if body else body}
                block.append(nm)
            cost = messages_tokens(block)
            if used + cost > budget and out_rev:
                break
            out_rev.append(block)
            used += cost
        out_rev.reverse()
        return [m for block in out_rev for m in block], used

    def build_messages(self) -> List[Dict[str, Any]]:
        system = {"role": "system", "content": self.wiki}

        # Management is a response to pressure. If everything still fits, send
        # it verbatim so the managed agent is never worse informed than naive
        # while there is room to spare.
        all_raw = [m for ex in self.exchanges for m in ex]
        pins = self._extract_pins()
        pin_msgs = ([{"role": "user",
                      "content": "Important constraints I have stated: "
                                 + "; ".join(pins)}] if pins else [])

        if messages_tokens(all_raw) + messages_tokens(pin_msgs) <= self.conv_budget:
            self.turns_managed = getattr(self, "turns_managed", 0)
            return self._finalize(system, [], all_raw, protected=pin_msgs)
        self.turns_managed = getattr(self, "turns_managed", 0) + 1

        old, recent = self._window_split()
        recent, recent_used = self._fit_exchanges(recent, self.conv_budget)
        recent_raw = [m for ex in recent for m in ex]

        # pins are protected: they are funded before the compressible middle
        pin_used = messages_tokens(pin_msgs)
        middle_budget = max(self.conv_budget - recent_used - pin_used, 0)
        middle_msgs, _ = self._compress_old(old, middle_budget)

        # pins are protected and surfaced first, never shed. Then the
        # compressed middle, then the verbatim recent window.
        return self._finalize(system, middle_msgs, recent_raw, protected=pin_msgs)
