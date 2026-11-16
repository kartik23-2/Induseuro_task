from __future__ import annotations

from dataclasses import dataclass, field

from shared.tokens import count_tokens

INPUT_PRICE = 3
CACHED_INPUT_PRICE = 0.3
OUTPUT_PRICE = 15
CACHE_MIN_PREFIX = 1024


@dataclass
class LlmCall:
    label: str
    prompt: str
    completion: str
    cached_prefix_tokens: int = 0

    @property
    def input_tokens(self) -> int:
        return count_tokens(self.prompt)

    @property
    def output_tokens(self) -> int:
        return count_tokens(self.completion)

    def points(self) -> int:
        cached = min(self.cached_prefix_tokens, self.input_tokens)
        uncached = self.input_tokens - cached
        return int(
            uncached * INPUT_PRICE
            + cached * CACHED_INPUT_PRICE
            + self.output_tokens * OUTPUT_PRICE
        )


@dataclass
class CostSession:
    calls: list[LlmCall] = field(default_factory=list)

    def record(
        self,
        label: str,
        prompt: str,
        completion: str,
        cached_prefix_tokens: int = 0,
    ) -> LlmCall:
        call = LlmCall(label, prompt, completion, cached_prefix_tokens)
        self.calls.append(call)
        return call

    def total_points(self) -> int:
        return sum(c.points() for c in self.calls)
