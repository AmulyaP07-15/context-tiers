"""Pinner: detects hard user constraints ("use miles, NOT the card") and pins
them so they survive every budget cut verbatim.
- OllamaPinner: asks a local LLM to extract constraints (better recall)
- HeuristicPinner: no-LLM fallback using signal words (good enough for demos)
"""

import json
import urllib.request

# whole-word signals of a hard requirement. Matched against tokenized words
# so "not" does not fire inside "knot" or "cannot", which is a real failure
# mode: a spurious pin wastes protected budget on noise.
import re as _re

# Strong constraint words: imperative or exclusionary, rarely used in a
# question. These fire the pin.
SIGNAL_WORDS = {"must", "not", "never", "only", "dont", "mustnt", "wont",
                "avoid", "ensure", "require", "always"}
# Phrases that signal a stated preference or rule rather than a question.
SIGNAL_PHRASES = ("instead of", "make sure", "do not", "don't", "has to",
                  "have to", "same booking", "without", "n't", "be sure",
                  "as long as", "keep my", "keep me on", "stay on", "must stay",
                  "refund to", "same card", "original card", "original payment",
                  "no aisle", "no window", "no middle", "i prefer")

# A question is almost never a hard constraint. If the sentence is
# interrogative, do not pin it even if a signal word appears.
_QUESTION_STARTS = ("what", "when", "where", "which", "who", "how", "can you",
                    "could you", "do i", "is there", "are there", "will i")


class HeuristicPinner:
    def extract(self, user_message: str) -> list[str]:
        pins: list[str] = []
        for sentence in (user_message or "").replace("!", ".").split("."):
            s = sentence.strip()
            if not s:
                continue
            low = s.lower()
            # a question is almost never a hard rule, skip it even if it
            # contains a signal word ("do I need a visa")
            if low.endswith("?") or any(low.startswith(q) for q in _QUESTION_STARTS):
                continue
            words = set(_re.findall(r"[a-z']+", low))
            hit = bool(words & SIGNAL_WORDS) or any(ph in low for ph in SIGNAL_PHRASES)
            if "n't" in low:
                hit = True
            if hit:
                pins.append(s)
        return pins


class OllamaPinner:
    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.fallback = HeuristicPinner()

    def extract(self, user_message: str) -> list[str]:
        prompt = (
            "Extract any HARD constraints from this user message (payment rules, "
            "people who must stay together, things that must NOT happen). "
            "Return one constraint per line, verbatim where possible. "
            "If there are none, return exactly: NONE\n\n"
            f"Message: {user_message}"
        )
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())["response"].strip()
            if out.upper().startswith("NONE"):
                return []
            return [line.strip("-• ").strip() for line in out.splitlines() if line.strip()]
        except Exception:
            return self.fallback.extract(user_message)


if __name__ == "__main__":
    p = HeuristicPinner()
    tests = [
        "Pay with miles, not the credit card please.",
        "My companion Priya must stay on the same booking.",
        "What time does the morning flight land?",
    ]
    for t in tests:
        print(f"{t!r} -> pins: {p.extract(t)}")
