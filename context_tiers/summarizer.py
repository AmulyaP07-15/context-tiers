"""Summarizer: shrinks spent tool results into short summaries instead of
dropping them. Two backends:
- OllamaSummarizer: calls your local Ollama (free, runs on your Mac)
- HeuristicSummarizer: no LLM at all; keeps the head of the text (fallback)
"""

import json
import urllib.request

from .tokens import count_tokens


class HeuristicSummarizer:
    """Field aware compression that protects load bearing data.

    tau-bench tool results are JSON records. Not all fields are equal. An id
    or a payment record is load bearing: the task fails if it is missing OR if
    it is altered, so it must survive whole or not at all. A timestamp or a
    verbose free-text note is sheddable. Naive head truncation ignores this and
    drops whatever sits at the back of the string, which is often the payment
    history. This compressor ranks fields and protects the ones that matter.
    """

    SUFFIX = " ...[truncated]"

    # keys whose VALUES must never be altered (ids, money, payment) and that
    # should be kept before anything else. Matched as substrings, lowercased.
    PROTECTED_HINTS = ("id", "payment", "price", "amount", "card", "cabin",
                       "flight_number", "reservation", "status", "insurance")
    # keys shed first when space is tight, before touching anything else
    LOW_VALUE_HINTS = ("created_at", "updated_at", "dob", "_note", "description")

    @staticmethod
    def _cut_to(text: str, tokens: int) -> str:
        tokens = max(int(tokens), 1)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return enc.decode(enc.encode(text)[:tokens])
        except Exception:
            return text[:tokens * 4]

    @staticmethod
    def _find_json(text: str):
        for opener, closer in (("{", "}"), ("[", "]")):
            i = text.find(opener); j = text.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(text[i:j + 1]), (i, j + 1)
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

    def _protected(self, key: str) -> bool:
        k = str(key).lower()
        return any(h in k for h in self.PROTECTED_HINTS)

    def _low_value(self, key: str) -> bool:
        k = str(key).lower()
        return any(h in k for h in self.LOW_VALUE_HINTS)

    def _shrink_value(self, key, v, value_budget_chars: int):
        """Shorten a value while keeping its shape. A protected value is NEVER
        truncated, because a cut id is a wrong id. Lists keep their first two
        elements and note how many were dropped."""
        if isinstance(v, dict):
            return {k: self._shrink_value(k, val, value_budget_chars) for k, val in v.items()}
        if isinstance(v, list):
            if len(v) <= 2:
                return [self._shrink_value(key, x, value_budget_chars) for x in v]
            return [self._shrink_value(key, v[0], value_budget_chars),
                    self._shrink_value(key, v[1], value_budget_chars),
                    f"...(+{len(v) - 2} more)"]
        if isinstance(v, str) and len(v) > value_budget_chars and not self._protected(key):
            return v[:value_budget_chars].rstrip() + "\u2026"
        return v

    def _rank_keys(self, obj: dict) -> list:
        """Protected keys first, ordinary keys next, low value keys last, so
        when space runs out the low value fields are what gets dropped."""
        protected = [k for k in obj if self._protected(k)]
        low = [k for k in obj if self._low_value(k) and k not in protected]
        mid = [k for k in obj if k not in protected and k not in low]
        return protected + mid + low

    def _summarize_json(self, obj, target_tokens: int) -> str:
        if not isinstance(obj, dict):
            for vc in (200, 120, 80, 50, 30, 18, 10):
                out = json.dumps(self._shrink_value("", obj, vc), separators=(",", ":"))
                if count_tokens(out) <= target_tokens:
                    return out
            return self._cut_to(json.dumps(obj, separators=(",", ":")), target_tokens)

        # first try shrinking values while keeping every key
        for vc in (200, 120, 80, 50, 30, 18, 10):
            shrunk = {k: self._shrink_value(k, obj[k], vc) for k in obj}
            out = json.dumps(shrunk, separators=(",", ":"))
            if count_tokens(out) <= target_tokens:
                return out

        # still too big: keep keys by rank until the budget is spent, so the
        # fields that get dropped are the low value ones, never the ids
        kept, out = {}, "{}"
        for k in self._rank_keys(obj):
            trial = {**kept, k: self._shrink_value(k, obj[k], 12)}
            s = json.dumps(trial, separators=(",", ":"))
            if count_tokens(s) > target_tokens and kept:
                continue   # skip this one, a later smaller field may still fit
            if count_tokens(s) <= target_tokens:
                kept, out = trial, s
        return out

    def summarize_with_envelope(self, text: str, target_tokens: int):
        """Return (summary_string, envelope). The string is exactly what
        summarize() returns, so the model sees no change. The envelope is
        metadata for logging and the retention metric, NEVER shown to the
        model. It records the token reduction and, for JSON records, which
        top-level fields survived and which were dropped."""
        summary = self.summarize(text, target_tokens)
        env = {
            "summarized": summary != text,
            "original_tokens": count_tokens(text),
            "summary_tokens": count_tokens(summary),
            "kept_fields": None,
            "dropped_fields": None,
        }
        # field-level accounting only applies to JSON records
        orig = self._find_json(text)
        summ = self._find_json(summary)
        if orig is not None and isinstance(orig[0], dict):
            orig_keys = set(orig[0].keys())
            summ_keys = set(summ[0].keys()) if (summ is not None and isinstance(summ[0], dict)) else set()
            env["kept_fields"] = sorted(orig_keys & summ_keys)
            env["dropped_fields"] = sorted(orig_keys - summ_keys)
        return summary, env

    def summarize(self, text: str, target_tokens: int) -> str:
        target = max(int(target_tokens), 1)
        if count_tokens(text) <= target:
            return text
        parsed = self._find_json(text)
        if parsed is not None:
            obj, (i, j) = parsed
            prefix = text[:i].strip()
            budget = target - (count_tokens(prefix) + 1 if prefix else 0)
            body = self._summarize_json(obj, max(budget, 1))
            out = (prefix + " " + body).strip() if prefix else body
            if count_tokens(out) <= target:
                return out
        suffix_cost = count_tokens(self.SUFFIX)
        if target > suffix_cost + 2:
            out = self._cut_to(text, target - suffix_cost).rstrip() + self.SUFFIX
        else:
            out = self._cut_to(text, target)
        for _ in range(5):
            if count_tokens(out) <= target:
                break
            out = self._cut_to(out, target)
        return out


class OllamaSummarizer:
    """Summarize via a local Ollama model. Falls back to heuristic on any error."""

    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.fallback = HeuristicSummarizer()

    def summarize(self, text: str, target_tokens: int) -> str:
        prompt = (
            f"Summarize the following tool output in at most {target_tokens // 2} words. "
            f"Keep concrete facts (IDs, prices, names, decisions). No preamble.\n\n{text}"
        )
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())["response"].strip()
            # safety: if the model rambled past the budget, heuristic-cut it
            if count_tokens(out) > target_tokens:
                out = self.fallback.summarize(out, target_tokens)
            return out
        except Exception:
            return self.fallback.summarize(text, target_tokens)


if __name__ == "__main__":
    long_text = ("FLIGHT SEARCH RESULTS: UA482 BOS-ORD 7:05am $214. UA318 9:40am $189. "
                 "AA1121 12:15pm $205. B6771 3:30pm $178. UA990 6:55pm $164. "
                 "Fare rules and baggage details for each option follow. " * 5)
    s = HeuristicSummarizer()
    out = s.summarize(long_text, 40)
    print(f"Original: {count_tokens(long_text)} tokens -> Summary: {count_tokens(out)} tokens")
    print(out[:200])
