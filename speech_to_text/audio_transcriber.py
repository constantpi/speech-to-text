import asyncio
import functools
import eel
import queue
import numpy as np

from typing import NamedTuple, Optional
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

from .utils.audio_utils import create_audio_stream
from .utils.word_merge import word_merge, clean_word_list
from .vad import Vad
from .utils.file_utils import write_audio
from .websoket_server import WebSocketServer
from .openai_api import OpenAIAPI

RATE = 16000
CHUNK = 512


class AudioData:
    def __init__(self, data: np.ndarray, start_time: float, is_last: bool):
        self.audio_data = data
        self.start_time = start_time
        self.is_last = is_last


class AppOptions(NamedTuple):
    audio_device: int
    silence_limit: int = 8
    noise_threshold: int = 5
    recent_audio_duration: int = 32  # 直近の音声データの長さ（サンプル数）大体1秒分
    recent_audio_max_length: int = 5  # 直近の音声データの最大長さ（サンプル数）大体5秒分
    save_result_number: int = 3  # 直近の文字起こしの結果をいくつ保持しておくか
    non_speech_threshold: float = 0.1
    include_non_speech: bool = False
    create_audio_file: bool = True
    use_websocket_server: bool = False
    use_openai_api: bool = False


class AudioTranscriber:
    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        whisper_model: WhisperModel,
        transcribe_settings: dict,
        app_options: AppOptions,
        websocket_server: Optional[WebSocketServer],
        openai_api: Optional[OpenAIAPI],
    ):
        self.event_loop = event_loop
        self.whisper_model: WhisperModel = whisper_model
        self.transcribe_settings = transcribe_settings
        self.app_options = app_options
        self.websocket_server = websocket_server
        self.openai_api = openai_api
        self.vad = Vad(app_options.non_speech_threshold)
        self.silence_counter: int = 0
        self.audio_data_list = []
        self.all_audio_data_list = []
        # self.audio_queue = queue.Queue()
        self.transcribing = False
        self.stream = None
        self._running = asyncio.Event()
        self._recent_transcribe_task = None
        self._translate_task = None
        # 直近の数秒間の音声データ
        self.recent_audio_data: Optional[AudioData] = None
        self.recent_audio_start: int = 0
        self.recent_audio_length: int = 0
        self.transcribe_result_list: list[str] = []  # 文字起こし結果のリスト(3つ程度保持しておく)
        self.texts_to_translate: list[str] = []  # 翻訳待ちのテキストのリスト
        self.word_timestamp_list: list[list[tuple[float, float, str]]] = []  # start, end, textのタプルのリストのリスト

    async def transcribe_recent_audio(self):
        """
        最近保持している音声バッファ（`self.recent_audio_data`）が存在する場合に
        Whisper で文字起こしを行い、結果をクライアントに送信します。

        このメソッドは非同期で呼び出すことを想定しており、内部でスレッドプール
        を使ってモデルの `transcribe` をブロッキング実行します。
        """

        # タイムスタンプ付きで文字起こしをする
        transcribe_settings = self.transcribe_settings.copy()
        transcribe_settings["without_timestamps"] = False
        transcribe_settings["word_timestamps"] = True

        with ThreadPoolExecutor() as executor:
            while self.transcribing:
                if self.recent_audio_data is None:
                    await asyncio.sleep(0.2)
                    continue

                try:
                    audio, start_time, is_last = self.recent_audio_data.audio_data, self.recent_audio_data.start_time, self.recent_audio_data.is_last
                    func = functools.partial(
                        self.whisper_model.transcribe,
                        audio=audio,
                        **transcribe_settings,
                    )
                    self.recent_audio_data = None  # Transcribed recent audio data, reset to None

                    segments, _info = await self.event_loop.run_in_executor(executor, func)
                    word_list = []
                    for segment in segments:
                        # print(f"Segment: start={segment.start:.2f}s, end={segment.end:.2f}s, text='{segment.text}'")
                        if segment.words is not None:
                            for word in segment.words:

                                word_start_time = start_time + word.start
                                word_end_time = start_time + word.end
                                # wordからは空白や-や.などを除去しておく
                                cleaned_word = word.word.strip().strip("-").strip(".")
                                if cleaned_word:  # 空でない場合のみ追加
                                    word_list.append((word_start_time, word_end_time, word.word))
                    if start_time < 0.01:
                        self.word_timestamp_list.clear()  # 音声区間の開始が0秒付近の場合は、前の区間の単語タイムスタンプをクリア
                    self.word_timestamp_list.append(word_list)  # word_timestamp_listに追加

                    merge_result = word_merge(self.word_timestamp_list, start_time)  # word_timestamp_listをマージしてテキストに追加
                    if merge_result.determined_text:
                        print(f"Determined text: '{merge_result.determined_text}', determined_end: {merge_result.determined_end:.2f}s")
                        eel.display_transcription(merge_result.determined_text)
                        self.transcribe_result_list.append(merge_result.determined_text)  # transcribe_result_listに追加
                        # 翻訳待ちのテキストに追加する
                        if self.app_options.use_openai_api:
                            self.texts_to_translate.append(merge_result.determined_text)  # 翻訳待ちのテキストに追加

                        # word_timestamp_listをマージして確定したテキストの終わりまでクリア
                        self.word_timestamp_list = clean_word_list(self.word_timestamp_list, merge_result.determined_end)
                        print("cleaned:", clean_word_list(self.word_timestamp_list, merge_result.determined_end))

                    text = ("\n".join(self.transcribe_result_list) + "\n") if self.transcribe_result_list else ""  # 直近の文字起こし結果を結合
                    merged = merge_result.undetermined_text  # マージして確定できなかったテキスト
                    text += merged
                    eel.display_recent_transcription(text)

                    if is_last:
                        eel.display_transcription(merged)
                        self.transcribe_result_list.append(merged)  # transcribe_result_listに追加
                        self.transcribe_result_list = self.transcribe_result_list[-self.app_options.save_result_number:]
                        # 翻訳待ちのテキストに追加する
                        if self.app_options.use_openai_api:
                            self.texts_to_translate.append(merged)  # 翻訳待ちのテキストに追加
                        self.word_timestamp_list.clear()  # 単語タイムスタンプのリストをクリア

                except Exception as e:
                    eel.on_recive_message(str(e))

    def process_audio(self, audio_data: np.ndarray, frames: int, time, status):
        is_speech = self.vad.is_speech(audio_data)
        if is_speech:
            self.silence_counter = 0
            self.audio_data_list.append(audio_data.flatten())
        else:
            self.silence_counter += 1
            if self.app_options.include_non_speech:
                self.audio_data_list.append(audio_data.flatten())

        # 直近の数秒間の音声データを更新
        # もしstartから現在までの長さが前回の長さ + recent_audio_durationを超えていたら、recent_audio_dataを更新
        if len(self.audio_data_list) - self.recent_audio_start >= self.app_options.recent_audio_duration * (self.recent_audio_length + 1):
            self.recent_audio_length = min(self.recent_audio_length + 1, self.app_options.recent_audio_max_length)
            self.recent_audio_start = len(self.audio_data_list) - self.app_options.recent_audio_duration * self.recent_audio_length
            start_time = self.recent_audio_start * CHUNK / RATE
            self.recent_audio_data = AudioData(np.concatenate(self.audio_data_list[self.recent_audio_start:]), start_time, False)

        if not is_speech and self.silence_counter > self.app_options.silence_limit:
            if len(self.audio_data_list) > self.app_options.noise_threshold:
                # 音声区間の終了とみなす
                # 若干長めになる可能性はあるが誤差の範囲のためOKとする
                start_time = self.recent_audio_start * CHUNK / RATE
                self.recent_audio_data = AudioData(np.concatenate(self.audio_data_list[self.recent_audio_start:]), start_time, True)
            self.silence_counter = 0
            self.recent_audio_length = 0
            self.recent_audio_start = 0

            self.audio_data_list.clear()

    def batch_transcribe_audio(self, audio_data: np.ndarray):
        segment_list = []
        segments, _ = self.whisper_model.transcribe(
            audio=audio_data, **self.transcribe_settings
        )

        for segment in segments:
            word_list = []
            if self.transcribe_settings["word_timestamps"] == True:
                if segment.words is None:
                    continue
                for word in segment.words:
                    word_list.append(
                        {
                            "start": word.start,
                            "end": word.end,
                            "text": word.word,
                        }
                    )
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": word_list,
                }
            )

        eel.transcription_clear()

        # if self.openai_api is not None:
        #     self.text_proofreading(segment_list)
        # else:
        #     eel.on_recive_segments(segment_list)
        eel.on_recive_segments(segment_list)

    async def start_transcription(self):
        try:
            self.transcribing = True
            self.stream = create_audio_stream(
                self.app_options.audio_device, self.process_audio
            )
            self.stream.start()
            self._running.set()
            # start recent-audio transcribe task (runs similarly to transcribe_audio)
            self._recent_transcribe_task = asyncio.run_coroutine_threadsafe(
                self.transcribe_recent_audio(), self.event_loop
            )
            # start translation task (runs similarly to transcribe_audio)
            self._translate_task = asyncio.run_coroutine_threadsafe(
                self.text_translation(), self.event_loop
            )
            eel.on_recive_message("Transcription started.")
            while self._running.is_set():
                await asyncio.sleep(1)
        except Exception as e:
            eel.on_recive_message(str(e))

    async def stop_transcription(self):
        try:
            self.transcribing = False
            if self._recent_transcribe_task is not None:
                self.event_loop.call_soon_threadsafe(self._recent_transcribe_task.cancel)
                self._recent_transcribe_task = None
            if self._translate_task is not None:
                self.event_loop.call_soon_threadsafe(self._translate_task.cancel)
                self._translate_task = None

            if self.app_options.create_audio_file and len(self.all_audio_data_list) > 0:
                audio_data = np.concatenate(self.all_audio_data_list)
                self.all_audio_data_list.clear()
                # write_audio("web", "voice", audio_data)
                self.batch_transcribe_audio(audio_data)

            if self.stream is not None:
                self._running.clear()
                self.stream.stop()
                self.stream.close()
                self.stream = None
                eel.on_recive_message("Transcription stopped.")
            else:
                eel.on_recive_message("No active stream to stop.")
        except Exception as e:
            eel.on_recive_message(str(e))

    async def text_translation(self):
        '''
        self.transcribe_result_listのテキストを翻訳して、翻訳結果をクライアントに送信します。
        翻訳にはOpenAIAPIのtext_translationメソッドを使用します。
        '''
        if self.openai_api is None:
            return
        with ThreadPoolExecutor() as executor:
            while self.transcribing:
                if not self.texts_to_translate:
                    await asyncio.sleep(0.5)
                    continue
                text = "\n".join(self.texts_to_translate)
                self.texts_to_translate.clear()
                try:
                    print(f"Translating text: {text}")
                    translated_text = await self.event_loop.run_in_executor(
                        executor, functools.partial(self.openai_api.text_translation, text)
                    )
                    eel.display_transcription(translated_text)
                    print(f"Translated text: {translated_text}")
                    if self.websocket_server is not None:
                        await self.websocket_server.send_message(translated_text)
                except Exception as e:
                    eel.on_recive_message(str(e))
