from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import time

MAX_OUTPUT_TOKENS = 2000  # 出力トークンの最大数（例: 2000トークン)
MAX_INPUT_TOKENS = 2000  # 入力トークンの最大数（例: 2000トークン)

CONTEXT_MAX_LENGTH = 60  # スライディングコンテクスト保持用の最大文字数（例: 直近60文字を保持）

ALPHABET_THRESHOLD = 5  # アルファベットが5文字以上含まれているかどうかの閾値


# 文字列がアルファベットを一定以上含むかどうかを判定する関数
def contains_english(text):
    count = sum(1 for char in text if char.isalpha())
    return count >= ALPHABET_THRESHOLD


class OpenAIAPI:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.MODEL_NAME = "gpt-4o-mini"
        self.chat_model = ChatOpenAI(
            model=self.MODEL_NAME,
            # max_tokens=self.MAX_TOKENS,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            temperature=0
        )

        # スライディングコンテクスト保持用
        self.previous_translation_context = ""
        self.previous_raw_text_context = ""

        # 統計情報
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.start_time = time.time()

    def text_translation(self, new_text: str) -> str:
        if not contains_english(new_text):
            # アルファベットが一定以上含まれていない場合は翻訳しない
            # 新しい文字起こしコンテクストを更新
            # print("=== Skipping translation (not enough English) ===")
            self.previous_raw_text_context += new_text
            self.previous_raw_text_context = self.previous_raw_text_context[-CONTEXT_MAX_LENGTH:]
            return ""
        new_text = new_text[:MAX_INPUT_TOKENS]  # 入力トークンの最大数に制限
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    # "new en:以降のテキストのみを自然な日本語に翻訳せよ"
                    "Translate only the text after 'new en:' into natural Japanese."
                    # "翻訳以外の説明やコメントは不要"
                    "No explanation or comments needed other than the translation."

                ),
                (
                    "user",
                    "old en:\n{prev_raw_text}\n\n"
                    "old ja:\n{prev_translation}\n\n"
                    "new en:\n{new_text}"
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

        try:
            response = self.chat_model.invoke(formatted_prompt)
            # print(response)
            translated = str(response.content).strip()
        except Exception as e:
            print(f"Error occurred while invoking OpenAI API: {e}")
            return ""

        # コンテクスト更新（直近60文字だけ保持）
        combined = self.previous_translation_context + "\n" + translated
        self.previous_translation_context = combined[-CONTEXT_MAX_LENGTH//2:]  # 日本語は英語よりも一文字の情報量が多いことが多いため、保持する文字数を半分にする

        # 新しい文字起こしコンテクストを更新
        self.previous_raw_text_context += new_text
        self.previous_raw_text_context = self.previous_raw_text_context[-CONTEXT_MAX_LENGTH:]

        # 統計情報の更新
        self.total_input_tokens += response.response_metadata["token_usage"]["prompt_tokens"]
        self.total_output_tokens += response.response_metadata["token_usage"]["completion_tokens"]
        elapsed_time = time.time() - self.start_time
        print(f"Total input tokens: {self.total_input_tokens}, Total output tokens: {self.total_output_tokens}, Elapsed time: {elapsed_time:.2f} seconds")

        return translated
