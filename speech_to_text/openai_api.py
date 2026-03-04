from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import os

MAX_TOKENS = 2000
# スライディングコンテクスト保持用の最大文字数（例: 直近300文字を保持）
CONTEXT_MAX_LENGTH = 300

ALPHABET_THRESHOLD = 5  # アルファベットが5文字以上含まれているかどうかの閾値


# 文字列がアルファベットを一定以上含むかどうかを判定する関数
def contains_english(text):
    count = sum(1 for char in text if char.isalpha())
    return count >= ALPHABET_THRESHOLD


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
        self.previous_raw_text_context = ""

    def text_translation(self, new_text: str) -> str:
        if not contains_english(new_text):
            # アルファベットが一定以上含まれていない場合は翻訳しない
            # 新しい文字起こしコンテクストを更新
            # print("=== Skipping translation (not enough English) ===")
            self.previous_raw_text_context += new_text
            self.previous_raw_text_context = self.previous_raw_text_context[-CONTEXT_MAX_LENGTH:]
            return ""
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
                    "【以前の文字起こし】\n{prev_raw_text}\n\n"
                    "【以前の翻訳】\n{prev_translation}\n\n"
                    "【新しい文字起こし部分】\n{new_text}"
                ),
            ]
        )

        formatted_prompt = chat_prompt.format_messages(
            prev_raw_text=self.previous_raw_text_context,
            prev_translation=self.previous_translation_context,
            new_text=new_text
        )
        # print("=== Prompt to OpenAI ===")
        # print(formatted_prompt)

        response = self.chat_model.invoke(formatted_prompt)
        translated = response.content.strip()

        # コンテクスト更新（直近300文字だけ保持）
        combined = self.previous_translation_context + "\n" + translated
        self.previous_translation_context = combined[-CONTEXT_MAX_LENGTH:]

        # 新しい文字起こしコンテクストを更新
        self.previous_raw_text_context += new_text
        self.previous_raw_text_context = self.previous_raw_text_context[-CONTEXT_MAX_LENGTH:]

        return translated
