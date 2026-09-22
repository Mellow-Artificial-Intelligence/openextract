"""Reusable sync/async sessions with a dependency-injected test agent."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from openextract import AsyncExtractor, Extractor


class Contact(BaseModel):
    name: str
    email: str


def test_agent() -> Agent:
    """Build a deterministic agent; production code can inject a configured provider agent."""
    model = TestModel(custom_output_args={"name": "Ada Lovelace", "email": "ada@example.com"})
    return Agent(model, output_type=Contact)


def sync_example() -> Contact:
    with Extractor(Contact, agent=test_agent()) as extractor:
        first = extractor.extract(b"Ada <ada@example.com>", media_type="text/plain")
        batch = extractor.extract_many(
            [b"Ada <ada@example.com>", b"Ada <ada@example.com>"],
            media_type="text/plain",
        )
        assert batch == [first, first]
        return first


async def async_example() -> Contact:
    async with AsyncExtractor(Contact, agent=test_agent()) as extractor:
        first = await extractor.extract(b"Ada <ada@example.com>", media_type="text/plain")
        batch = await extractor.extract_many(
            [b"Ada <ada@example.com>"],
            media_type="text/plain",
        )
        assert batch == [first]
        return first


def main() -> None:
    print(sync_example().model_dump_json(indent=2))
    print(asyncio.run(async_example()).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
