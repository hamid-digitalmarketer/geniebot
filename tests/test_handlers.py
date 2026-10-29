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
# /summarize (#22)
# ---------------------------------------------------------------------------


def test_summarize_without_llm_edits_placeholder_message():
    """Regression test for #22: When no LLM is configured, /summarize must
    edit the 'Summarizing...' placeholder message instead of sending a new reply."""
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="What is Python?", response="Python is a versatile programming language.", interaction_type="text")

    update = _make_update(user_id=1)
    processing_msg = MagicMock()
    processing_msg.edit_text = AsyncMock()
    update.message.reply_text.return_value = processing_msg

    context = _make_context({"user_memory": memory})  # No "llm" key
    context.args = []

    asyncio.run(summarize_command(update, context))

    # The initial placeholder message was sent
    update.message.reply_text.assert_awaited_once_with("📝 Summarizing the last interaction...")

    # The placeholder message was edited with the fallback summary
    processing_msg.edit_text.assert_awaited_once()
    final_text = processing_msg.edit_text.call_args[0][0]
    assert "📝 *Summary:*" in final_text
    assert "Latest text: Python is a versatile programming language." in final_text


def test_summarize_with_llm_edits_placeholder_message():
    """Verify that when LLM is configured, /summarize edits the placeholder message with LLM output."""
    memory = UserMemory(max_history=3)
    memory.add_interaction(user_id=1, query="Hello", response="Hi there! How can I help you today?", interaction_type="text")

    llm = MagicMock()
    llm.generate.return_value = "User greeted and bot replied politely."

    update = _make_update(user_id=1)
    processing_msg = MagicMock()
    processing_msg.edit_text = AsyncMock()
    update.message.reply_text.return_value = processing_msg

    context = _make_context({"user_memory": memory, "llm": llm})
    context.args = []

    asyncio.run(summarize_command(update, context))

    update.message.reply_text.assert_awaited_once_with("📝 Summarizing the last interaction...")
    processing_msg.edit_text.assert_awaited_once()
    final_text = processing_msg.edit_text.call_args[0][0]
    assert "📝 *Summary:*" in final_text
    assert "User greeted and bot replied politely." in final_text

