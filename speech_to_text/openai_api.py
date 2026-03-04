from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import os

MAX_TOKENS = 2000
# スライディングコンテクスト保持用の最大文字数（例: 直近300文字を保持）
CONTEXT_MAX_LENGTH = 300


class OpenAIAPI:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.MODEL_NAME = "gpt-4o-mini"
        self.MAX_TOKENS = MAX_TOKENS
        self.chat_model = ChatOpenAI(
            model=self.MODEL_NAME,
            max_tokens=self.MAX_TOKENS,
            temperature=0
        )

        # スライディングコンテクスト保持用
        self.previous_translation_context = ""

    def text_translation(self, new_text: str):
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはリアルタイム字幕翻訳を行っています。"
                    "与えられた英語テキストを自然な日本語に翻訳してください。"
                    "未完の文であっても自然に訳してください。"
                    "以前の翻訳と文体を揃えてください。"
                    "出力は今回の新しい部分の翻訳のみとしてください。"
                    "以前の翻訳を繰り返さないでください。"
                ),
                (
                    "user",
                    "【以前の翻訳】\n{prev_translation}\n\n"
                    "【新しい文字起こし部分】\n{new_text}"
                ),
            ]
        )

        formatted_prompt = chat_prompt.format_messages(
            prev_translation=self.previous_translation_context,
            new_text=new_text
        )

        response = self.chat_model.invoke(formatted_prompt)
        translated = response.content.strip()

        # コンテクスト更新（直近300文字だけ保持）
        combined = self.previous_translation_context + "\n" + translated
        self.previous_translation_context = combined[-CONTEXT_MAX_LENGTH:]

        return translated
