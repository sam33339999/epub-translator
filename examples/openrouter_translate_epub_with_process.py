import os
from pathlib import Path
import time
from tqdm import tqdm
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

    with tqdm(total=100, desc="翻译中", unit="%") as pbar:
        last_progress = 0.0
        start_time = time.time()

        def on_progress(progress: float):
            nonlocal last_progress
            increment = (progress - last_progress) * 100
            pbar.update(increment)
            last_progress = progress
            pbar.set_postfix({
                'tokens': llm.total_tokens,
                'cost_est': f'${llm.total_tokens * 0.00001:.4f}'
            })

        translate(
            source_path=source_path,
            target_path=target_path,
            target_language=language.TRADITIONAL_CHINESE,
            submit=SubmitKind.APPEND_BLOCK,
            llm=llm,
            on_progress=on_progress,
            concurrency=4,
        )

        elapsed = time.time() - start_time
        print(f"\n翻译完成，耗时 {elapsed:.1f} 秒")
        print(f"使用的 token 总数: {llm.total_tokens:,}")
        print(f"平均 tokens/秒: {llm.total_tokens/elapsed:.1f}")


if __name__ == "__main__":
    main()
