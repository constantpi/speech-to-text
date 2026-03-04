"""Type stubs for the `eel` module used in this project.

This file only includes the functions and signatures that the project uses.
Add more declarations here as needed.
"""
from typing import Any, Callable, Dict, List


def expose(func: Callable[..., Any]) -> Callable[..., Any]: ...
def init(*args: Any, **kwargs: Any) -> Any: ...
def start(*args: Any, **kwargs: Any) -> Any: ...
def spawn(*args: Any, **kwargs: Any) -> Any: ...


def display_transcription(text: str) -> Any: ...
def display_recent_transcription(text: str) -> Any: ...
def on_recive_message(msg: str) -> Any: ...
def on_recive_segments(segments: List[Dict[str, Any]]) -> Any: ...
def transcription_clear() -> Any: ...
def transcription_stoppd() -> Any: ...

# Generic send / js helpers often used from Python -> JS


def send(*args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "expose",
    "init",
    "start",
    "spawn",
    "display_transcription",
    "display_recent_transcription",
    "on_recive_message",
    "on_recive_segments",
    "transcription_clear",
    "send",
    "transcription_stoppd"
]
