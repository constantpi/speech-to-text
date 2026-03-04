from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import os


class OpenAIAPI:
    def __init__(self):
        load_dotenv(dotenv_path=".env")
        self.MODEL_NAME = "gpt-5-mini"
        self.MAX_TOKENS = 2000
        self.chat_model = ChatOpenAI(model=self.MODEL_NAME, max_tokens=self.MAX_TOKENS)

    def text_translation(self, text: str):
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "このテキストはWhisperで文字起こしされたものです。英語のテキストを日本語に翻訳してください。翻訳結果以外の余計な説明は不要です。翻訳結果のみを出力してください。",
                ),
                ("user", "{text}"),
            ]
        )
        formatted_prompt = chat_prompt.format_messages(text=text)
        response = self.chat_model.invoke(formatted_prompt)
        return response.content.strip()
