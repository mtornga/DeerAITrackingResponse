from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Type, TypeVar

EventT = TypeVar("EventT")


class EventBus:
    def __init__(self) -> None:
        self._queues: Dict[Type[Any], List[asyncio.Queue[Any]]] = defaultdict(list)

    def subscribe(self, event_type: Type[EventT]) -> asyncio.Queue[EventT]:
        queue: asyncio.Queue[EventT] = asyncio.Queue()
        self._queues[event_type].append(queue)
        return queue

    async def publish(self, event: Any) -> None:
        queues = self._queues.get(type(event), [])
        for queue in queues:
            await queue.put(event)


GLOBAL_EVENT_BUS = EventBus()
