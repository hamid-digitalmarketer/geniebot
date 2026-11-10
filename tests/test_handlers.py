"""Tests for the /export and /stats command handlers (issues #13, #14).

Update/context are mocked rather than constructed as real python-telegram-bot
objects - only the attributes the handlers actually touch are stubbed out.
No pytest-asyncio plugin is installed in this project, so each async handler
is driven directly with asyncio.run() instead of an async test function.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from bot.handlers import export_command, stats_command, summarize_command
from utils.cache import RateLimiter
from utils.memory import UserMemory


def _make_update(user_id=1):
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.reply_text = AsyncMock()
    update.message.reply_document = AsyncMock()
    return update


def _make_context(bot_data, args=None):
    context = MagicMock()
    context.bot_data = bot_data
    context.args = args if args is not None else []
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
# /summarize (issue #22)
# ---------------------------------------------------------------------------


def test_summarize_without_llm_edits_placeholder_message():
    """When LLM is not configured, /summarize edits the placeholder instead of sending a new message (issue #22)."""
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="What is python?", response="A programming language.", interaction_type="text")

    processing_msg = MagicMock()
    processing_msg.edit_text = AsyncMock()

    update = _make_update(user_id=1)
    update.message.reply_text = AsyncMock(return_value=processing_msg)
    context = _make_context({"user_memory": memory})

    asyncio.run(summarize_command(update, context))

    # reply_text should only be called once for the placeholder
    update.message.reply_text.assert_awaited_once_with("📝 Summarizing the last interaction...")
    # placeholder should be edited with the fallback summary
    processing_msg.edit_text.assert_awaited_once()
    args, kwargs = processing_msg.edit_text.call_args
    assert "📝 *Summary:*" in args[0]
    assert "Latest text: A programming language." in args[0]
    assert kwargs.get("parse_mode") == "Markdown"


def test_summarize_with_llm_edits_placeholder_message():
    """When LLM is configured, /summarize edits the placeholder with generated summary."""
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="What is python?", response="A programming language.", interaction_type="text")

    processing_msg = MagicMock()
    processing_msg.edit_text = AsyncMock()

    update = _make_update(user_id=1)
    update.message.reply_text = AsyncMock(return_value=processing_msg)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Python is a high-level programming language."
    context = _make_context({"user_memory": memory, "llm": mock_llm})

    asyncio.run(summarize_command(update, context))

    update.message.reply_text.assert_awaited_once_with("📝 Summarizing the last interaction...")
    processing_msg.edit_text.assert_awaited_once()
    args, kwargs = processing_msg.edit_text.call_args
    assert "Python is a high-level programming language." in args[0]
    assert kwargs.get("parse_mode") == "Markdown"


def test_summarize_without_history_replies_error():
    """When user has no history, reply directly without placeholder."""
    memory = UserMemory(max_history=3)
    update = _make_update(user_id=1)
    context = _make_context({"user_memory": memory})

    asyncio.run(summarize_command(update, context))

    update.message.reply_text.assert_awaited_once()
    assert "No interactions found to summarize yet." in update.message.reply_text.call_args[0][0]
