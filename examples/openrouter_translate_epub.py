import os
from pathlib import Path

from epub_translator import SubmitKind, translate, language
from epub_translator.llm.core import LLM

def main() -> None:
    source_path = Path("source.epub")
    target_path = Path("target.epub")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENROUTER_API_KEY")

    llm = LLM(
        key=api_key,
        url="https://openrouter.ai/api/v1",
        model="z-ai/glm-5",
        token_encoding="o200k_base",
        # timeout=360.0,
        provider={
            "order": ["atlas-cloud/fp8"],
            "allow_fallbacks": False,
        },
        headers={
            # "HTTP-Referer": "https://your-site.example",  # 建议填写你自己的站点
            "X-Title": "EPUB Translator",
        },
        log_dir_path="logs",
        cache_path="./cache",
    )

    translate(
        source_path=source_path,
        target_path=target_path,
        target_language=language.TRADITIONAL_CHINESE,
        submit=SubmitKind.APPEND_BLOCK,
        llm=llm,
        concurrency=4,
    )


if __name__ == "__main__":
    main()
