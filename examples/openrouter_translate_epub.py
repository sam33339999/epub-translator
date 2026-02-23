import os
from pathlib import Path

from epub_translator import SubmitKind, translate
from epub_translator.llm.core import LLM


def main() -> None:
    source_path = Path("book.epub")  # 修改为你的 EPUB 文件路径
    target_path = Path("book.zh-cn.bilingual.epub")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENROUTER_API_KEY")

    llm = LLM(
        key=api_key,
        url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4-20250514",
        token_encoding="o200k_base",
        timeout=360.0,
        provider={
            "order": ["Azure", "Anthropic"],
            "allow_fallbacks": False,
        },
        headers={
            "HTTP-Referer": "https://your-site.example",  # 建议填写你自己的站点
            "X-Title": "EPUB Translator Example",
        },
    )

    translate(
        source_path=source_path,
        target_path=target_path,
        target_language="Chinese",
        submit=SubmitKind.APPEND_BLOCK,
        llm=llm,
        concurrency=2,
    )


if __name__ == "__main__":
    main()
