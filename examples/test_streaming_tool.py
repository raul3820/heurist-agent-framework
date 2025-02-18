import asyncio
from typing import Union, AsyncGenerator
from random import random
from agents.tool_decorator import tool

@tool(None)
async def roll_dice(n_times: int, agent_context) -> AsyncGenerator[str, None]:
    """This is an asynchronous dice rolling function."""
    i = 0
    yield f"Hello {agent_context}, rolling {n_times} times!"
    while i < int(n_times):
        result = random.randint(1, 6)
        yield str(result)
        await asyncio.sleep(1)
        i += 1


