"""RedisContextStore: same interface as ContextStore, but state lives in Redis.
Why: state survives process restarts and can be shared across agent instances —
the production-safety story.

Usage on your machine (Redis via Docker):
    docker run -d -p 6379:6379 redis
    store = RedisContextStore.from_url("redis://localhost:6379/0", session="demo1")
"""

import json

from .tokens import Item, count_item


class RedisContextStore:
    def __init__(self, client, session: str):
        self.r = client
        self.key = f"acm:session:{session}:items"

    @classmethod
    def from_url(cls, url: str, session: str):
        import redis
        return cls(redis.Redis.from_url(url, decode_responses=True), session)

    def add(self, role: str, content: str, source: str, turn: int) -> None:
        self.r.rpush(self.key, json.dumps(
            {"role": role, "content": content, "source": source, "turn": turn}))

    def _load(self) -> list[Item]:
        raw = self.r.lrange(self.key, 0, -1)
        return [Item(**json.loads(x)) for x in raw]

    @property
    def items(self) -> list[Item]:
        return self._load()

    def get_by_source(self, source: str) -> list[Item]:
        return [it for it in self._load() if it.source == source]

    def all_sources(self) -> list[str]:
        seen: list[str] = []
        for it in self._load():
            if it.source not in seen:
                seen.append(it.source)
        return seen

    def total_tokens(self) -> int:
        return sum(count_item(it) for it in self._load())

    def reclassify(self, before_turn: int, mapping: dict[str, str]) -> int:
        """Same contract as ContextStore.reclassify. Items loaded from Redis
        are fresh objects, so mutating them in place changes nothing. The list
        has to be rewritten, which is why this needs its own implementation."""
        items = self._load()
        moved = 0
        for it in items:
            if it.turn < before_turn and it.source in mapping:
                it.source = mapping[it.source]
                moved += 1
        if moved:
            pipe = self.r.pipeline()
            pipe.delete(self.key)
            for it in items:
                pipe.rpush(self.key, json.dumps(
                    {"role": it.role, "content": it.content,
                     "source": it.source, "turn": it.turn}))
            pipe.execute()
        return moved

    def clear(self) -> None:
        self.r.delete(self.key)


if __name__ == "__main__":
    # test without a real Redis server, using fakeredis
    import fakeredis
    store = RedisContextStore(fakeredis.FakeRedis(decode_responses=True), session="test")
    store.clear()
    store.add("system", "Policy text here.", "system_policy", 0)
    store.add("user", "Use miles, not the card.", "pinned", 1)
    store.add("user", "Any window seats left?", "recent_turns", 2)
    print(f"Sources: {store.all_sources()}")
    print(f"Total tokens: {store.total_tokens()}")
    print("Survives 'restart':", RedisContextStore(store.r, session="test").total_tokens(), "tokens")
