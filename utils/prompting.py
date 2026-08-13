from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
        text = "\n".join(lines).strip()

    text = _CONTROL_CHARS_RE.sub(" ", text)

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON must be an object.")
        return parsed
    except json.JSONDecodeError:
        pass

    # extract every top-level {...} block and return the first that parses
    candidates: list[str] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            break
        candidates.append(text[start:end + 1])
        pos = end + 1

    if not candidates:
        raise ValueError(f"No JSON object found in model output:\n{raw_text[:400]}")

    last_exc: Exception = ValueError("no candidates parsed")
    for candidate in candidates:
        candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_exc = exc

    raise ValueError(f"JSON parse failed ({last_exc}).\nRaw:\n{raw_text[:400]}") from last_exc
