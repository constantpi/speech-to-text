# speech-to-text + Real-time Translation

This project is a fork of the original speech-to-text repository by reriiasu:

https://github.com/reriiasu/speech-to-text

The original project provides real-time transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper).
This fork extends it with real-time translation and latency optimization,
designed for live lecture support (e.g., English Zoom lectures → Japanese).



![architecture](docs/architecture.png)

Accepts audio input from a microphone using a [Sounddevice](https://github.com/spatialaudio/python-sounddevice). By using [Silero VAD](https://github.com/snakers4/silero-vad)(Voice Activity Detection), silent parts are detected and recognized as one voice data. This audio data is converted to text using Faster-Whisper.

The HTML-based GUI allows you to check the transcription results and make detailed settings for the faster-whisper.

## New Features in This Fork

- Real-time translation (e.g., English → Japanese)
- Differential translation updates (reduce redundant re-translation)
- Optimized OpenAI API token usage
- Lower latency buffering strategy
- uv-based reproducible environment

## Transcription speed

If the sentences are well separated, the transcription takes less than a second.
![TranscriptionSpeed](docs/transcription_speed.png)

Large-v2 model
Executed with CUDA 11.7 on a NVIDIA GeForce RTX 3060 12GB.

## Installation

This project uses **uv** for environment and dependency management.

### 1. Install uv

If you don't have uv installed:

https://docs.astral.sh/uv/

### 2. Create and sync environment

From the project root:

```
uv sync
```


This will:
- Create a virtual environment
- Install all required dependencies

Dependency versions are locked via `uv.lock` for reproducibility.

---

## Run
```
uv run python -m speech_to_text
```

1. Select "App Settings" and configure the settings.
2. Select "Model Settings" and configure the settings.
3. Select "Transcribe Settings" and configure the settings.
4. Select "VAD Settings" and configure the settings.
5. Start Transcription

If you want to use real-time translation with the OPENAI API, please set the OPENAI_API_KEY in the .env file.
The `.env` file will be loaded automatically at startup.
For example(in .env file):
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Notes

- If you select local_model in "Model size or path", the model with the same name in the local folder will be referenced.

## Demo

![demo](docs/demo.gif)

## Original Project Updates

The following updates were made in the original repository by reriiasu.
### 2023-06-26

- Add generate audio files from input sound.
- Add synchronize audio files with transcription.  
Audio and text highlighting are linked.

### 2023-06-29

- Add transcription from audio files.(only wav format)

### 2023-07-03

- Add Send transcription results from a WebSocket server to a WebSocket client.  
Example of use: Display subtitles in live streaming.

### 2023-07-05

- Add generate SRT files from transcription result.

### 2023-07-08

- Add support for mp3, ogg, and other audio files.  
Depends on Soundfile support.
- Add setting to include non-speech data in buffer.  
While this will increase memory usage, it will improve transcription accuracy.

### 2023-07-09

- Add non-speech threshold setting.

### 2023-07-11

- Add Text proofreading option via OpenAI API.  
Transcription results can be proofread.

### 2023-07-12

- Add feature where audio and word highlighting are synchronized.  
if Word Timestamps is true.

### 2023-10-01

- Support for repetition_penalty and no_repeat_ngram_size in transcribe_settings.
- Updating packages.

### 2023-11-27

- Support "large-v3" model.
- Update faster-whisper requirement to include the latest version "0.10.0".

### 2024-07-23

- Support "Faster Distil-Whisper" model.
- Update faster-whisper requirement to include the latest version "1.0.3".
- Updating packages.
- Add run.bat for Windows.

## Fork Updates

### 2026-03-04
- Added real-time translation feature
- Implemented differential translation updates
- Optimized OpenAI API token usage
- Migrated environment management to uv

## License

This project is licensed under the MIT License.

Original work:
Copyright (c) 2023 reriiasu

Modifications:
Copyright (c) 2026 constantpi

See LICENSE for details.
