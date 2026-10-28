"""Tests for the /export and /stats command handlers (issues #13, #14).

Update/context are mocked rather than constructed as real python-telegram-bot
objects - only the attributes the handlers actually touch are stubbed out.
No pytest-asyncio plugin is installed in this project, so each async handler
is driven directly with asyncio.run() instead of an async test function.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from bot.handlers import (
    _convert_markdown_to_telegram,
    export_command,
    stats_command,
)
from utils.cache import RateLimiter
from utils.memory import UserMemory


def _make_update(user_id=1):
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.reply_text = AsyncMock()
    update.message.reply_document = AsyncMock()
    return update


def _make_context(bot_data):
    context = MagicMock()
    context.bot_data = bot_data
    return context


# ---------------------------------------------------------------------------
# /export
# ---------------------------------------------------------------------------


def test_export_sends_document_with_history():
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="hi", response="hello", interaction_type="text")

    update = _make_update(user_id=1)
    context = _make_context({"user_memory": memory})

    asyncio.run(export_command(update, context))

    update.message.reply_document.assert_awaited_once()
    _, kwargs = update.message.reply_document.call_args
    assert "document" in kwargs
    assert "1" in kwargs["caption"]  # 1 interaction mentioned in caption
    update.message.reply_text.assert_not_called()


def test_export_with_no_history_replies_with_message_not_document():
    memory = UserMemory(max_history=3)
    update = _make_update(user_id=2)
    context = _make_context({"user_memory": memory})

    asyncio.run(export_command(update, context))

    update.message.reply_document.assert_not_called()
    update.message.reply_text.assert_awaited_once()


def test_export_without_memory_system_replies_with_error():
    update = _make_update(user_id=3)
    context = _make_context({})

    asyncio.run(export_command(update, context))

    update.message.reply_document.assert_not_called()
    update.message.reply_text.assert_awaited_once()


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------


def test_stats_reports_history_and_rate_limit_status():
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="hi", response="hello")
    limiter = RateLimiter(limit=10, window_seconds=60)
    limiter.allow(1)
    limiter.allow(1)

    update = _make_update(user_id=1)
    context = _make_context({"user_memory": memory, "rate_limiter": limiter})

    asyncio.run(stats_command(update, context))

    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args[0][0]
    assert "1 / 3" in text  # history: 1 of max 3
    assert "8 / 10" in text  # 2 requests used, 8 remaining


def test_stats_handles_missing_systems_gracefully():
    update = _make_update(user_id=1)
    context = _make_context({})

    asyncio.run(stats_command(update, context))

    update.message.reply_text.assert_awaited_once()
    text = update.message.reply_text.call_args[0][0]
    assert "Unavailable" in text


# ---------------------------------------------------------------------------
# _convert_markdown_to_telegram (issue #21)
# ---------------------------------------------------------------------------


def test_convert_markdown_bold_to_telegram():
    # Telegram's legacy Markdown only supports single-asterisk *bold*.
    assert _convert_markdown_to_telegram("This is **bold** text") == "This is *bold* text"


def test_convert_markdown_bold_multiple_occurrences():
    assert (
        _convert_markdown_to_telegram("**one** and **two** and **three**")
        == "*one* and *two* and *three*"
    )


def test_convert_markdown_bold_across_multiple_lines():
    text = "First **bold** line\nSecond **also bold** line"
    assert _convert_markdown_to_telegram(text) == "First *bold* line\nSecond *also bold* line"


def test_convert_markdown_headings_still_converted():
    text = "# Title\n\nSome **bold** body"
    result = _convert_markdown_to_telegram(text)
    assert "📍 *Title*" in result
    assert "Some *bold* body" in result


def test_convert_markdown_leaves_plain_text_unchanged():
    text = "Plain text with no markdown."
    assert _convert_markdown_to_telegram(text) == text
