from types import SimpleNamespace
from unittest.mock import patch

from epub_translator.llm.executor import LLMExecutor
from epub_translator.llm.statistics import Statistics
from epub_translator.llm.types import Message, MessageRole


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="translated text"))],
                usage=None,
            )
        ]


class _FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        _FakeOpenAI.last_instance = self


def test_llm_executor_sends_openrouter_provider_in_extra_body():
    provider = {
        "order": ["Azure", "Anthropic"],
        "allow_fallbacks": False,
    }

    with patch("epub_translator.llm.executor.OpenAI", _FakeOpenAI):
        executor = LLMExecutor(
            api_key="test-key",
            url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4-20250514",
            timeout=30.0,
            retry_times=0,
            retry_interval_seconds=0.0,
            create_logger=lambda: None,
            statistics=Statistics(),
            provider=provider,
        )

    result = executor.request(
        messages=[Message(role=MessageRole.USER, message="hello")],
        max_tokens=128,
        temperature=0.2,
        top_p=0.9,
        cache_key=None,
    )

    assert result == "translated text"
    assert _FakeOpenAI.last_instance is not None
    sent_kwargs = _FakeOpenAI.last_instance.chat.completions.last_kwargs
    assert sent_kwargs is not None
    assert sent_kwargs["extra_body"] == {"provider": provider}


def test_llm_executor_passes_custom_headers_to_openai_client():
    headers = {
        "HTTP-Referer": "https://example.com",
        "X-Title": "EPUB Translator",
    }

    with patch("epub_translator.llm.executor.OpenAI", _FakeOpenAI):
        LLMExecutor(
            api_key="test-key",
            url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4-20250514",
            timeout=30.0,
            retry_times=0,
            retry_interval_seconds=0.0,
            create_logger=lambda: None,
            statistics=Statistics(),
            headers=headers,
        )

    assert _FakeOpenAI.last_instance is not None
    assert _FakeOpenAI.last_instance.init_kwargs["default_headers"] == headers
