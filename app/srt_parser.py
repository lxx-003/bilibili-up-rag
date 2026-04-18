from __future__ import annotations

import re
from pathlib import Path

from app.models import SubtitleCue

TIMECODE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def parse_srt(path: Path) -> list[SubtitleCue]:
    raw = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
    if not raw:
        return []
    blocks = re.split(r"\n\s*\n", raw)
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timecode_line = lines[1] if "-->" in lines[1] else lines[0]
        match = TIMECODE_RE.search(timecode_line)
        if not match:
            continue
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        if not text:
            continue
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start=_parse_timestamp(match.group("start")),
                end=_parse_timestamp(match.group("end")),
                text=text,
            )
        )
    return cues


def _parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )
