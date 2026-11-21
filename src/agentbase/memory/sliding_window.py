from agentbase.memory.base import BaseMemory
from agentbase.model.schema import Message

# Used only for token counting approximation — any GPT-4-family tokenizer
# is a reasonable proxy across OpenAI / Anthropic / Qwen models.
_COUNTING_MODEL_FALLBACK = "gpt-4o"


class SlidingWindowMemory(BaseMemory):
    """Sliding Window Memory: truncates by message count + token count."""

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        counting_model: str = _COUNTING_MODEL_FALLBACK,
    ):
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._counting_model = counting_model
        self._messages: list[Message] = []
        self._token_counts: list[int] = []

    def add(self, message: Message) -> None:
        tokens = self._count_tokens(message)
        self._messages.append(message)
        self._token_counts.append(tokens)
        self._truncate()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._token_counts.clear()

    @property
    def token_count(self) -> int:
        return sum(self._token_counts)

    def _count_tokens(self, message: Message) -> int:
        try:
            import litellm
            msg_dict = message.model_dump(exclude_none=True)
            return litellm.token_counter(model=self._counting_model, messages=[msg_dict])
        except Exception:
            # Fallback: ~4 characters per token is a reasonable approximation
            text = str(message.content or "")
            return max(1, len(text) // 4)

    def _truncate(self) -> None:
        """Truncate oldest messages to stay within limits."""
        if self._max_messages and len(self._messages) > self._max_messages:
            excess = len(self._messages) - self._max_messages
            self._messages = self._messages[excess:]
            self._token_counts = self._token_counts[excess:]

        if self._max_tokens:
            while self._messages and self.token_count > self._max_tokens:
                self._messages.pop(0)
                self._token_counts.pop(0)
