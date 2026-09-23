"""Model-compatible final-context token accounting."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import tiktoken


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TikTokenCounter:
    """Count text tokens using the explicit encoding for the locked answer model."""

    def __init__(self, *, model: str, expected_encoding: str) -> None:
        encoding = tiktoken.encoding_for_model(model)
        if encoding.name != expected_encoding:
            raise ValueError(
                f"tokenizer mapping mismatch: {model} uses {encoding.name}, "
                f"expected {expected_encoding}"
            )
        self.model = model
        self.encoding_name = encoding.name
        self._encoding = encoding

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))
