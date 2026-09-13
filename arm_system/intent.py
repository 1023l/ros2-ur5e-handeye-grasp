"""Natural-language → skill id (keyword first; optional LLM)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


# skill_id → example phrases (CN/EN)
SKILL_CATALOG: Dict[str, List[str]] = {
    "help": ["help", "帮助", "技能", "你能做什么", "skills"],
    "home": ["home", "回零", "回家", "回 home", "go home", "回到初始"],
    "up": ["up", "抬起", "举起", "到 up", "抬起来"],
    "approach": ["approach", "靠近", "接近", "移近", "靠近目标"],
    "pick_place": [
        "pick",
        "place",
        "pick_place",
        "pick and place",
        "抓取",
        "放置",
        "抓放",
        "把物体",
        "搬过去",
        "放到",
        "抓起来",
    ],
    "gripper_open": ["open gripper", "张开", "松开夹爪", "开爪", "松爪"],
    "gripper_close": ["close gripper", "合上", "夹紧", "合爪", "闭爪"],
}


def list_skills_text() -> str:
    lines = ["可用技能:"]
    for sid, phrases in SKILL_CATALOG.items():
        lines.append(f"  - {sid}: 例「{phrases[0]}」/「{phrases[1] if len(phrases)>1 else phrases[0]}」")
    return "\n".join(lines)


def parse_intent_keywords(text: str) -> Tuple[Optional[str], str]:
    """Return (skill_id|None, reason). Longer/more specific phrases win."""
    raw = (text or "").strip()
    if not raw:
        return None, "empty command"
    t = raw.lower()

    # Direct skill id
    if t in SKILL_CATALOG:
        return t, "exact skill id"

    best: Optional[str] = None
    best_len = -1
    for sid, phrases in SKILL_CATALOG.items():
        for p in phrases:
            pl = p.lower()
            if pl in t or pl in raw:
                if len(pl) > best_len:
                    best = sid
                    best_len = len(pl)
    if best:
        return best, f"keyword match → {best}"
    return None, "no keyword match"


def parse_intent_llm(
    text: str,
    *,
    api_base: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Tuple[Optional[str], str]:
    """
    Optional OpenAI-compatible chat API.
    Env or params: API key required. Fail closed to None.
    """
    skills = ", ".join(SKILL_CATALOG.keys())
    system = (
        "You map a user robot command to exactly one skill id. "
        f"Allowed: {skills}. "
        'Reply JSON only: {"skill":"<id or unknown>"}'
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None, f"llm bad reply: {content!r}"
        data = json.loads(m.group(0))
        skill = str(data.get("skill", "unknown")).strip()
        if skill in SKILL_CATALOG:
            return skill, "llm"
        return None, f"llm unknown skill={skill}"
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
        return None, f"llm error: {exc}"


def parse_intent(
    text: str,
    *,
    use_llm: bool = False,
    api_base: str = "https://api.openai.com/v1",
    api_key: str = "",
    model: str = "gpt-4o-mini",
) -> Tuple[Optional[str], str]:
    skill, reason = parse_intent_keywords(text)
    if skill is not None:
        return skill, reason
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if use_llm and key:
        return parse_intent_llm(
            text, api_base=api_base, api_key=key, model=model
        )
    return None, reason
