"""Telegram command handlers for GenieBot."""

import asyncio
import io
import json

from telegram import InputFile, Update
from telegram.error import RetryAfter
from telegram.ext import ContextTypes

from utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_TELEGRAM_MSG_LEN = 4000
MAX_QUESTION_LEN = 1000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RETRY_AFTER_ATTEMPTS = 3


async def _with_retry_after(coro_func, *args, **kwargs):
    """Call a Telegram Bot API coroutine, honoring RetryAfter (Telegram's
    flood-control error) by sleeping for the server-specified retry_after
    and retrying, instead of letting it bubble up as a generic failure or
    being silently logged and swallowed.
    """
    for attempt in range(MAX_RETRY_AFTER_ATTEMPTS):
        try:
            return await coro_func(*args, **kwargs)
        except RetryAfter as exc:
            wait_s = exc.retry_after
            logger.warning(
                f"Telegram RetryAfter: sleeping {wait_s}s before retry "
                f"(attempt {attempt + 1}/{MAX_RETRY_AFTER_ATTEMPTS})"
            )
            await asyncio.sleep(wait_s)
    return await coro_func(*args, **kwargs)


async def _safe_reply_text(message, text: str, markdown: bool = True) -> None:
    """Reply with markdown first and fallback to plain text if parsing fails."""
    if markdown:
        try:
            await _with_retry_after(message.reply_text, text, parse_mode="Markdown")
            return
        except Exception:
            # If markdown fails, try without formatting
            pass
    await _with_retry_after(message.reply_text, text)


async def _send_long_response(message, text: str, markdown: bool = True) -> None:
    """Split long responses to stay within Telegram limits."""
    if len(text) <= MAX_TELEGRAM_MSG_LEN:
        await _safe_reply_text(message, text, markdown=markdown)
        return

    for idx in range(0, len(text), MAX_TELEGRAM_MSG_LEN):
        chunk = text[idx : idx + MAX_TELEGRAM_MSG_LEN]
        await _safe_reply_text(message, chunk, markdown=markdown)


def _convert_markdown_to_telegram(text: str) -> str:
    """Convert Markdown to plain text with emoji headers (Telegram Markdown doesn't support #)."""
    lines = text.split('\n')
    converted = []
    
    for line in lines:
        # Convert ### Heading -> 📌 Heading
        if line.startswith('### '):
            heading = line[4:].strip()
            converted.append(f"📌 *{heading}*")
        # Convert ## Heading -> 🔹 Heading
        elif line.startswith('## '):
            heading = line[3:].strip()
            converted.append(f"🔹 *{heading}*")
        # Convert # Heading -> 📍 Heading
        elif line.startswith('# '):
            heading = line[2:].strip()
            converted.append(f"📍 *{heading}*")
        else:
            converted.append(line)
    
    return '\n'.join(converted)


def _format_sources(sources: dict) -> str:
    """Format unique source file names for response display."""
    if not sources:
        return ""
    lines = ["", "", "📚 *Sources:*"]
    for source in sources.keys():
        lines.append(f"• {source}")
    return "\n".join(lines)


async def _check_rate_limit(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    """Return True if the request should proceed, False if it was rate-limited
    (and a reply has already been sent explaining that)."""
    rate_limiter = context.bot_data.get("rate_limiter")
    if not rate_limiter:
        return True
    if rate_limiter.allow(user_id):
        return True
    wait_s = rate_limiter.seconds_until_reset(user_id)
    await _safe_reply_text(
        update.message,
        f"⏳ You're sending requests too fast. Please wait {wait_s}s and try again.",
        markdown=False,
    )
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    logger.info(f"User {user.id} started bot")

    text = (
        "🤖 *GenieBot: RAG & Vision AI Assistant*\n\n"
        "Use these commands:\n"
        "• /ask <query> - Ask questions from local documents\n"
        "• /image - Upload an image for caption + tags\n"
        "• /summarize [chat|image] - Summarize your recent interaction\n"
        "• /export - Download your conversation history\n"
        "• /stats - See your history and rate-limit status\n"
        "• /clear - Clear your saved conversation history\n"
        "• /help - Show usage instructions\n\n"
        "GenieBot keeps your last 3 interactions for better continuity."
    )
    await _safe_reply_text(update.message, text, markdown=True)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    logger.info(f"User {update.effective_user.id} requested help")

    help_text = (
        "📖 *How to use GenieBot*\n\n"
        "*1. /ask <question>*\n"
        "Example: /ask What is the return policy?\n"
        "GenieBot retrieves relevant document chunks and answers with sources.\n\n"
        "*2. /image*\n"
        "Upload an image after this command.\n"
        "GenieBot returns one caption and three tags.\n\n"
        "*3. /summarize [chat|image]*\n"
        "Summarize your latest chat or latest image result.\n\n"
        "*4. /export*\n"
        "Download your saved conversation history as a file.\n\n"
        "*5. /stats*\n"
        "See how much history is saved and your rate-limit status.\n\n"
        "*6. /clear*\n"
        "Clear your saved conversation history.\n\n"
        "*7. /start*\n"
        "Shows the quick command summary."
    )
    await _safe_reply_text(update.message, help_text, markdown=True)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command to reset the user's saved interaction history."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} used /clear command")

    user_memory = context.bot_data.get("user_memory")
    if not user_memory:
        await _safe_reply_text(update.message, "❌ Memory system not initialized", markdown=False)
        return

    user_memory.clear_history(user_id)
    await _safe_reply_text(update.message, "🗑️ Your conversation history has been cleared.", markdown=False)


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /export command to download the user's conversation history."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} used /export command")

    user_memory = context.bot_data.get("user_memory")
    if not user_memory:
        await _safe_reply_text(update.message, "❌ Memory system not initialized", markdown=False)
        return

    history = user_memory.get_history(user_id)
    if not history:
        await _safe_reply_text(update.message, "📝 No interaction history to export yet.", markdown=False)
        return

    payload = json.dumps(history, indent=2, ensure_ascii=False)
    file_buffer = io.BytesIO(payload.encode("utf-8"))
    file_buffer.name = "geniebot_history.json"

    await _with_retry_after(
        update.message.reply_document,
        document=InputFile(file_buffer, filename="geniebot_history.json"),
        caption=f"🗂️ Your last {len(history)} interaction(s).",
    )
    logger.info(f"Exported {len(history)} interactions for user {user_id}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command exposing per-user history and rate-limit status."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} used /stats command")

    user_memory = context.bot_data.get("user_memory")
    rate_limiter = context.bot_data.get("rate_limiter")

    history_line = "Unavailable"
    if user_memory:
        history = user_memory.get_history(user_id)
        history_line = f"{len(history)} / {user_memory.max_history} saved interaction(s)"

    rate_line = "Unavailable"
    if rate_limiter:
        remaining = rate_limiter.remaining(user_id)
        reset_in = rate_limiter.seconds_until_reset(user_id)
        rate_line = f"{remaining} / {rate_limiter.limit} requests left (resets in {reset_in}s)"

    text = (
        "📊 *Your GenieBot status*\n\n"
        f"🧠 History: {history_line}\n"
        f"⏳ Rate limit: {rate_line}"
    )
    await _safe_reply_text(update.message, text, markdown=True)


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summarize command for recent chat or image interactions."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} used /summarize command")

    user_memory = context.bot_data.get("user_memory")
    llm = context.bot_data.get("llm")
    if not user_memory:
        await _safe_reply_text(update.message, "❌ Memory system not initialized", markdown=False)
        return

    history = user_memory.get_history(user_id)
    if not history:
        await _safe_reply_text(update.message, "📝 No interactions found to summarize yet.", markdown=False)
        return

    mode = (context.args[0].strip().lower() if context.args else "").strip()
    if mode and mode not in {"chat", "image"}:
        await _safe_reply_text(update.message, "Usage: /summarize [chat|image]", markdown=False)
        return

    target = None
    if mode == "image":
        for item in reversed(history):
            if item.get("type") == "image":
                target = item
                break
    elif mode == "chat":
        for item in reversed(history):
            if item.get("type") == "text":
                target = item
                break
    else:
        target = history[-1]

    if not target:
        await _safe_reply_text(update.message, f"No {mode} interaction found to summarize.", markdown=False)
        return

    if not await _check_rate_limit(update, context, user_id):
        return

    processing_msg = await _with_retry_after(update.message.reply_text, "📝 Summarizing the last interaction...")

    interaction_type = target.get("type", "text")
    query = target.get("query", "")
    response = target.get("response", "")

    if not llm:
        summary = f"Latest {interaction_type}: {response[:220]}"
    else:
        summary_prompt = (
            "Summarize this user interaction in 1-2 short sentences. "
            "Return only the summary text.\n\n"
            f"Interaction type: {interaction_type}\n"
            f"User input: {query}\n"
            f"Bot response: {response}"
        )

        summary = await asyncio.to_thread(llm.generate, summary_prompt, temperature=0.2)
        if isinstance(summary, str) and summary.strip().lower().startswith("error:"):
            summary = response[:220]

    final_text = f"📝 *Summary:*\n{summary}"
    try:
        await _with_retry_after(processing_msg.edit_text, final_text, parse_mode="Markdown")
    except Exception:
        await _safe_reply_text(update.message, final_text, markdown=False)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command using RAG QA."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} used /ask command")

    if not context.args:
        await _safe_reply_text(
            update.message,
            "❌ Please provide a question.\nUsage: /ask <your question>",
            markdown=False,
        )
        return

    if not await _check_rate_limit(update, context, user_id):
        return

    question = " ".join(context.args)
    if len(question) > MAX_QUESTION_LEN:
        await _safe_reply_text(
            update.message,
            f"❌ Question too long ({len(question)} chars). Max is {MAX_QUESTION_LEN}.",
            markdown=False,
        )
        return
    await _safe_reply_text(
        update.message,
        "🔍 Searching and analyzing... This may take a moment.",
        markdown=False,
    )

    qa_system = context.bot_data.get("qa_system")
    user_memory = context.bot_data.get("user_memory")
    if not qa_system:
        logger.error("QA system not available")
        await _safe_reply_text(update.message, "❌ QA system not initialized", markdown=False)
        return

    try:
        conversation_context = user_memory.get_context(user_id) if user_memory else ""
        result = await asyncio.to_thread(
            qa_system.answer_question,
            question,
            True,
            conversation_context,
        )
        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", {}) if not result.get("error") else {}

        if result.get("error"):
            logger.error(f"QA generation error for user {user_id}: {result.get('details', 'unknown')}")

        if user_memory:
            user_memory.add_interaction(user_id, question, answer, "text")

        # Format response with heading conversion
        formatted_answer = _convert_markdown_to_telegram(answer)
        source_list = _format_sources(sources)

        response = f"🤖 *Answer:*\n{formatted_answer}{source_list}"
        await _send_long_response(update.message, response, markdown=True)
        logger.info(f"Answer provided to user {user_id}")
    except Exception as exc:
        logger.error(f"Error in ask command: {exc}")
        await _safe_reply_text(update.message, "❌ Something went wrong answering that. Please try again later.", markdown=False)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /image command."""
    logger.info(f"User {update.effective_user.id} used /image command")
    await _safe_reply_text(
        update.message,
        "📸 Please upload an image for analysis. I will return a caption and 3 tags.",
        markdown=False,
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photo messages for image captioning + tagging."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} uploaded an image")

    if not update.message.photo:
        return

    if not await _check_rate_limit(update, context, user_id):
        return

    vision_processor = context.bot_data.get("vision_processor")
    user_memory = context.bot_data.get("user_memory")
    if not vision_processor:
        logger.error("Vision processor not available")
        await _safe_reply_text(update.message, "❌ Vision processor not initialized", markdown=False)
        return

    try:
        photo = update.message.photo[-1]
        if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
            await _safe_reply_text(update.message, "❌ Image too large (max 10 MB).", markdown=False)
            return
        photo_file = await _with_retry_after(photo.get_file)
        image_data = await photo_file.download_as_bytearray()

        processing_msg = await _with_retry_after(update.message.reply_text, "🔄 Processing image... Please wait.")
        result = await asyncio.to_thread(vision_processor.process_image, bytes(image_data))

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            logger.error(f"Image processing error: {error}")
            await _with_retry_after(processing_msg.edit_text, "❌ Could not process this image. Please try another.")
            return

        caption = result.get("caption", "No caption generated")
        tags = result.get("tags", [])

        if user_memory:
            user_memory.add_interaction(user_id, "[Image Upload]", caption, "image")

        response = f"📸 *Caption:*\n{caption}\n\n🏷️ *Tags:* {', '.join(tags)}"
        try:
            await _with_retry_after(processing_msg.edit_text, response, parse_mode="Markdown")
        except Exception:
            await _with_retry_after(processing_msg.edit_text, response)

        logger.info(f"Image processed for user {user_id}")
    except Exception as exc:
        logger.error(f"Error handling image: {exc}")
        await _safe_reply_text(update.message, "❌ Error processing image. Please try again later.", markdown=False)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global Telegram error handler.

    Most Bot API calls already go through _with_retry_after, but this is a
    safety net: if a RetryAfter ever reaches here uncaught, honor Telegram's
    flood-control wait instead of just logging it.
    """
    if isinstance(context.error, RetryAfter):
        wait_s = context.error.retry_after
        logger.warning(f"Unhandled Telegram RetryAfter reached error_handler: sleeping {wait_s}s")
        await asyncio.sleep(wait_s)
        return
    await asyncio.sleep(0)
    logger.error(f"Update {update} caused error: {context.error}")
