import os
import io
import re
import time
import html
import asyncio
import logging
from itertools import cycle
from collections import defaultdict

import psutil
from PIL import Image, ImageOps
from pyrogram import Client, filters, idle
from pyrogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, CopyTextButton
)
from google import genai
from google.genai import types

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None


# -------------------- Configuration --------------------

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "").strip()

PRIMARY_MODEL = "gemini-3.1-flash-lite-preview"
FALLBACK_MODEL = "gemini-3-flash-preview"
MAX_IMAGE_BYTES = 80 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
START_TIME = time.monotonic()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("OtakuAura")

app = Client(
    "otakuaura",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workers=8,
)

# RAM fallback data. In RAM mode, this data is lost on process restart.
users = set()
settings = defaultdict(lambda: {
    "style": "danbooru",
    "length": "medium",
    "language": "English",
})
api_keys = []
key_cycle = None
key_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

# Each user's most recent image is kept as a Telegram file_id, not image bytes.
last_images = {}
last_prompts = {}

mongo_client = None
db = None
db_mode = "In-Memory Mode"


# -------------------- Welcome text --------------------

WELCOME_CAPTION = """Hello👋.. {first_name}✨

╔══════════════════════╗
✦ 𝐖𝐄𝐋𝐂𝐎𝐌𝐄 ✦ TO OtakuAura💕
╚══════════════════════╝

♛ 𝐏𝐑𝐎𝐌𝐏𝐓 𝐁𝐎𝐓 ♛

✨ Welcome to the world of creativity.

🎨 Discover premium AI prompts.
🖼️ Explore stunning art styles.
⚡ Create extraordinary AI images.
💎 Unlock your creative potential.

(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)

♛ 𝐏𝐑𝐄𝐌𝐈𝐔𝐌 𝐄𝐗𝐏𝐄𝐑𝐈𝐄𝐍𝐂𝐄
✦ Quality • Creativity • Innovation

(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)

Send me an image or photo, and I’ll turn it into a detailed image-generation prompt.

The prompt is designed for external tools such as Midjourney, Stable Diffusion, and Flux.

How to use 👀: Upload a JPG, PNG, or WebP image 🖼. I’ll analyze it and return a prompt you can copy.

Thank You ❤for using me 🩷💕💕

(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)(⁠◕⁠ᴗ⁠◕⁠✿⁠)"""


# -------------------- Database --------------------

async def init_database():
    global mongo_client, db, db_mode

    if not MONGO_URI or AsyncIOMotorClient is None:
        log.info("Using in-memory storage.")
        return

    try:
        mongo_client = AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
        )
        await asyncio.wait_for(mongo_client.admin.command("ping"), timeout=5)
        db = mongo_client["otakuaura"]
        db_mode = "MongoDB Connected"

        async for doc in db.users.find({}, {"_id": 1}):
            users.add(int(doc["_id"]))

        async for doc in db.settings.find():
            uid = int(doc["_id"])
            settings[uid] = {
                "style": doc.get("style", "danbooru"),
                "length": doc.get("length", "medium"),
                "language": doc.get("language", "English"),
            }

        async for doc in db.api_keys.find({}, {"key": 1}):
            value = doc.get("key")
            if value and value not in api_keys:
                api_keys.append(value)

        log.info("MongoDB connected.")
    except Exception:
        log.exception("MongoDB connection failed; using RAM mode.")
        db_mode = "In-Memory Mode"
        db = None
        if mongo_client:
            mongo_client.close()
            mongo_client = None


async def register_user(uid: int):
    users.add(uid)
    if db is not None:
        try:
            await db.users.update_one(
                {"_id": uid}, {"$setOnInsert": {"_id": uid}}, upsert=True
            )
        except Exception:
            log.exception("Could not save user.")


async def save_user_settings(uid: int):
    if db is not None:
        try:
            await db.settings.update_one(
                {"_id": uid}, {"$set": settings[uid]}, upsert=True
            )
        except Exception:
            log.exception("Could not save settings.")


# -------------------- API key rotation --------------------

def reset_key_rotation():
    global key_cycle
    key_cycle = cycle(api_keys) if api_keys else None


async def next_api_key():
    async with key_lock:
        if not api_keys:
            return None
        # Recreate iterator after key-list mutations.
        if key_cycle is None:
            reset_key_rotation()
        return next(key_cycle)


async def add_api_key(value: str) -> bool:
    if value in api_keys:
        return False
    api_keys.append(value)
    reset_key_rotation()
    if db is not None:
        await db.api_keys.update_one(
            {"key": value}, {"$set": {"key": value}}, upsert=True
        )
    return True


async def remove_api_key(value: str) -> bool:
    if value not in api_keys:
        return False
    api_keys.remove(value)
    reset_key_rotation()
    if db is not None:
        await db.api_keys.delete_one({"key": value})
    return True


def masked_key(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


# -------------------- Image processing --------------------

def compress_image(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((1024, 1024))
        if image.mode not in ("RGB",):
            image = image.convert("RGB")

        # Try progressively lower quality and smaller dimensions.
        for dimension in (1024, 896, 768, 640, 512):
            if max(image.size) > dimension:
                image.thumbnail((dimension, dimension))
            for quality in (82, 72, 62, 52, 42, 32):
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=quality, optimize=True)
                data = output.getvalue()
                output.close()
                if len(data) <= MAX_IMAGE_BYTES:
                    return data
    raise ValueError("Could not compress this image below 80 KB.")


async def get_compressed_image(message) -> bytes:
    media = await app.download_media(message, in_memory=True)
    if media is None:
        raise ValueError("Image download failed.")
    try:
        media.seek(0)
        raw = media.read()
        if not raw:
            raise ValueError("The uploaded image is empty.")
        return await asyncio.to_thread(compress_image, raw)
    finally:
        media.close()


# -------------------- Prompt generation --------------------

def make_instruction(uid: int) -> str:
    pref = settings[uid]
    style = pref["style"]
    length = pref["length"]
    language = pref["language"]

    style_instructions = {
        "danbooru": (
            "Return only comma-separated Danbooru/Stable Diffusion-style tags. "
            "Use concise recognizable tags, not prose."
        ),
        "midjourney": (
            "Return one compact Midjourney-style prompt made of visual phrases "
            "separated by commas. Do not add parameters such as --ar unless visible "
            "or requested."
        ),
        "detailed": (
            "Return detailed comma-separated visual tags covering subject, pose, "
            "expression, clothing, colors, environment, lighting, composition, "
            "camera/view and art style."
        ),
    }
    length_instructions = {
        "short": "Aim for about 10-15 useful tags.",
        "medium": "Aim for about 20-30 useful tags.",
        "long": "Aim for about 35-50 useful tags.",
    }

    return f"""
You are an expert image-to-prompt assistant.
Analyze only what is actually visible. Do not guess character names or hidden details.
Describe the subject, appearance, pose, expression, clothing, setting, lighting,
composition and visual style when relevant.

STYLE: {style_instructions.get(style, style_instructions["danbooru"])}
LENGTH: {length_instructions.get(length, length_instructions["medium"])}
LANGUAGE: Write descriptive tags in {language}.

STRICT OUTPUT:
- Output only comma-separated tags/phrases.
- No heading, markdown fences, explanations, or narrative paragraph.
- Keep tags useful for image-generation tools.
""".strip()


def clean_prompt(value: str) -> str:
    value = re.sub(r"```(?:[a-zA-Z0-9_-]+)?", "", value or "")
    value = value.replace("```", "")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"(,\s*){2,}", ", ", value)
    return value.strip(" ,")


async def call_gemini(image_bytes: bytes, uid: int) -> str:
    if not api_keys:
        raise RuntimeError("No Gemini API key is configured.")

    async with request_semaphore:
        errors = []
        tried = set()

        for _ in range(len(api_keys)):
            key = await next_api_key()
            if not key or key in tried:
                continue
            tried.add(key)
            client = genai.Client(api_key=key)

            contents = [
                make_instruction(uid),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ]

            # Try the requested preview model first, then the fallback model.
            for model in (PRIMARY_MODEL, FALLBACK_MODEL):
                try:
                    config = types.GenerateContentConfig(
                        temperature=0.25,
                        max_output_tokens=700,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                    response = await client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    result = clean_prompt(response.text or "")
                    if result:
                        return result
                    raise RuntimeError("Gemini returned an empty response.")
                except Exception as exc:
                    errors.append(str(exc))
                    log.warning("Gemini model %s failed: %s", model, exc)
                    # Try fallback model for model/config errors. Then rotate key.
                    if model == PRIMARY_MODEL:
                        continue
                    break

        raise RuntimeError(
            "All configured Gemini keys/models failed. "
            "Check API key validity, model availability and quota. "
            + (errors[-1][:250] if errors else "")
        )


# -------------------- Keyboards --------------------

def welcome_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ Prompt Settings", callback_data="settings")],
        [InlineKeyboardButton("🌐 Language", callback_data="language_menu")],
    ])


def prompt_keyboard(prompt: str):
    # Telegram CopyTextButton has a 1-256 character limit.
    # Full prompt is also displayed in a selectable <code> block.
    copy_value = prompt[:256]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            "📋 Copy Prompt",
            copy_text=CopyTextButton(text=copy_value)
        )],
        [
            InlineKeyboardButton("🌐 Translate", callback_data="translate"),
            InlineKeyboardButton("🔄 Regenerate", callback_data="regenerate"),
        ],
        [InlineKeyboardButton("⚙️ Prompt Settings", callback_data="settings")],
    ])


def settings_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎨 Output Style", callback_data="menu_style")],
        [InlineKeyboardButton("📏 Tag Length", callback_data="menu_length")],
        [InlineKeyboardButton("🌐 Language", callback_data="language_menu")],
        [InlineKeyboardButton("✖️ Close", callback_data="close_menu")],
    ])


def language_keyboard(uid: int):
    current = settings[uid]["language"]
    rows = []
    for label, value in [
        ("🇬🇧 English", "English"),
        ("🇮🇳 Hindi", "Hindi"),
        ("🇷🇺 Russian", "Russian"),
    ]:
        rows.append([InlineKeyboardButton(
            ("✅ " if current == value else "") + label,
            callback_data=f"set_language:{value}"
        )])
    rows.append([InlineKeyboardButton("🔙 Settings", callback_data="settings")])
    return InlineKeyboardMarkup(rows)


# -------------------- Start and image handler --------------------

@app.on_message(filters.command("start") & filters.private)
async def start_handler(_, message):
    uid = message.from_user.id
    await register_user(uid)

    first_name = html.escape(message.from_user.first_name or "Creator")
    caption = WELCOME_CAPTION.format(first_name=first_name)
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "welcome.png")

    if os.path.isfile(image_path):
        await message.reply_photo(
            image_path, caption=caption, parse_mode="html",
            reply_markup=welcome_keyboard()
        )
    else:
        await message.reply_text(
            caption + "\n\n⚠️ welcome.png was not found in the project root.",
            parse_mode="html", reply_markup=welcome_keyboard()
        )


@app.on_message(filters.private & (filters.photo | filters.document))
async def image_handler(_, message):
    uid = message.from_user.id
    await register_user(uid)

    if message.document:
        mime = message.document.mime_type or ""
        if not mime.startswith("image/"):
            return await message.reply_text("❌ Please send a JPG, PNG or WebP image.")
        if message.document.file_size and message.document.file_size > MAX_UPLOAD_BYTES:
            return await message.reply_text("❌ Maximum image upload size is 15 MB.")

    status = await message.reply_text("✨ Analyzing your image…")
    try:
        image_bytes = await get_compressed_image(message)
        prompt = await call_gemini(image_bytes, uid)
        del image_bytes

        last_images[uid] = message.id
        last_prompts[uid] = prompt

        await status.edit_text(
            "✨ Here is your generated prompt:\n\n"
            f"<code>{html.escape(prompt)}</code>",
            parse_mode="html",
            reply_markup=prompt_keyboard(prompt),
        )
    except Exception as exc:
        log.exception("Image-to-prompt failed")
        await status.edit_text(
            "❌ I couldn't generate a prompt for this image.\n"
            "Please try another image or try again later."
        )


# -------------------- Settings callbacks --------------------

@app.on_callback_query(filters.regex(r"^settings$"))
async def settings_menu(_, query):
    await query.answer()
    await query.message.edit_text(
        "⚙️ **OtakuAura Prompt Settings**\nChoose what you want to customize:",
        reply_markup=settings_keyboard()
    )


@app.on_callback_query(filters.regex(r"^menu_style$"))
async def style_menu(_, query):
    uid = query.from_user.id
    current = settings[uid]["style"]
    rows = []
    for label, value in [
        ("Danbooru Tags (Default)", "danbooru"),
        ("Midjourney Style", "midjourney"),
        ("Detailed Tags", "detailed"),
    ]:
        rows.append([InlineKeyboardButton(
            ("✅ " if current == value else "") + label,
            callback_data=f"set_style:{value}"
        )])
    rows.append([InlineKeyboardButton("🔙 Settings", callback_data="settings")])
    await query.answer()
    await query.message.edit_text("🎨 Select output style:", reply_markup=InlineKeyboardMarkup(rows))


@app.on_callback_query(filters.regex(r"^menu_length$"))
async def length_menu(_, query):
    uid = query.from_user.id
    current = settings[uid]["length"]
    rows = []
    for label, value in [("Short", "short"), ("Medium", "medium"), ("Long", "long")]:
        rows.append([InlineKeyboardButton(
            ("✅ " if current == value else "") + label,
            callback_data=f"set_length:{value}"
        )])
    rows.append([InlineKeyboardButton("🔙 Settings", callback_data="settings")])
    await query.answer()
    await query.message.edit_text("📏 Select tag length:", reply_markup=InlineKeyboardMarkup(rows))


@app.on_callback_query(filters.regex(r"^language_menu$"))
async def language_menu(_, query):
    await query.answer()
    await query.message.edit_text(
        "🌐 Select your language:",
        reply_markup=language_keyboard(query.from_user.id)
    )


@app.on_callback_query(filters.regex(r"^set_(style|length|language):"))
async def set_preference(_, query):
    uid = query.from_user.id
    kind, value = query.data.split(":", 1)
    allowed = {
        "style": {"danbooru", "midjourney", "detailed"},
        "length": {"short", "medium", "long"},
        "language": {"English", "Hindi", "Russian"},
    }
    if value not in allowed[kind]:
        return await query.answer("Invalid option.", show_alert=True)

    settings[uid][kind] = value
    await save_user_settings(uid)
    await query.answer("✅ Preference saved.")

    if kind == "language":
        await query.message.edit_text("🌐 Select your language:", reply_markup=language_keyboard(uid))
    else:
        await query.message.edit_text(
            "⚙️ **Settings updated**\n\n"
            f"🎨 Style: `{settings[uid]['style']}`\n"
            f"📏 Length: `{settings[uid]['length']}`\n"
            f"🌐 Language: `{settings[uid]['language']}`",
            reply_markup=settings_keyboard()
        )


@app.on_callback_query(filters.regex(r"^close_menu$"))
async def close_menu(_, query):
    await query.answer("Settings closed.")
    try:
        await query.message.delete()
    except Exception:
        pass


# -------------------- Translate and regenerate --------------------

@app.on_callback_query(filters.regex(r"^translate$"))
async def translate_handler(_, query):
    uid = query.from_user.id
    original = last_prompts.get(uid)
    if not original:
        return await query.answer("Generate a prompt first.", show_alert=True)

    languages = ["English", "Hindi", "Russian"]
    current = settings[uid]["language"]
    target = languages[(languages.index(current) + 1) % len(languages)]

    await query.answer("Translating…")
    try:
        key = await next_api_key()
        if not key:
            raise RuntimeError("No API key configured.")
        client = genai.Client(api_key=key)
        response = await client.aio.models.generate_content(
            model=PRIMARY_MODEL,
            contents=(
                f"Translate these image-generation tags into {target}. "
                "Keep comma-separated tags only. No explanation.\n\n" + original
            ),
            config=types.GenerateContentConfig(
                temperature=0.2, max_output_tokens=700,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        translated = clean_prompt(response.text or "")
        if not translated:
            raise RuntimeError("Empty translation.")
        settings[uid]["language"] = target
        last_prompts[uid] = translated
        await save_user_settings(uid)
        await query.message.edit_text(
            f"✨ Here is your generated prompt ({target}):\n\n"
            f"<code>{html.escape(translated)}</code>",
            parse_mode="html", reply_markup=prompt_keyboard(translated)
        )
    except Exception:
        log.exception("Translation failed")
        await query.answer("Translation failed. Please try again.", show_alert=True)


@app.on_callback_query(filters.regex(r"^regenerate$"))
async def regenerate_handler(_, query):
    uid = query.from_user.id
    source_message_id = last_images.get(uid)
    if not source_message_id:
        return await query.answer("Please upload your image again.", show_alert=True)

    await query.answer("🔄 Regenerating…")
    try:
        source = await app.get_messages(query.message.chat.id, source_message_id)
        image_bytes = await get_compressed_image(source)
        prompt = await call_gemini(image_bytes, uid)
        del image_bytes
        last_prompts[uid] = prompt
        await query.message.edit_text(
            "✨ Here is your generated prompt:\n\n"
            f"<code>{html.escape(prompt)}</code>",
            parse_mode="html", reply_markup=prompt_keyboard(prompt)
        )
    except Exception:
        log.exception("Regeneration failed")
        await query.answer("Regeneration failed. Please upload the image again.", show_alert=True)


# -------------------- Owner commands --------------------

def owner_only(message) -> bool:
    return bool(message.from_user and OWNER_ID and message.from_user.id == OWNER_ID)


@app.on_message(filters.command("addkey") & filters.private)
async def addkey_handler(_, message):
    if not owner_only(message):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        return await message.reply_text("Usage: /addkey GEMINI_API_KEY")
    try:
        added = await add_api_key(parts[1].strip())
        await message.reply_text("✅ API key added." if added else "⚠️ This key is already registered.")
    except Exception:
        log.exception("Adding API key failed")
        await message.reply_text("❌ Could not save the API key.")


@app.on_message(filters.command("removekey") & filters.private)
async def removekey_handler(_, message):
    if not owner_only(message):
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        return await message.reply_text("Usage: /removekey GEMINI_API_KEY")
    value = parts[1].strip()
    if value == GEMINI_API_KEY:
        return await message.reply_text("⚠️ The environment primary key cannot be removed here.")
    try:
        removed = await remove_api_key(value)
        await message.reply_text("✅ API key removed." if removed else "❌ Key not found.")
    except Exception:
        log.exception("Removing API key failed")
        await message.reply_text("❌ Could not remove the API key.")


@app.on_message(filters.command("apilist") & filters.private)
async def apilist_handler(_, message):
    if not owner_only(message):
        return
    if not api_keys:
        return await message.reply_text("🔑 No API keys configured.")
    lines = [f"{i}. {masked_key(key)}" for i, key in enumerate(api_keys, 1)]
    await message.reply_text(
        f"🔑 **API Key Manager**\nActive keys: {len(api_keys)}\n\n" + "\n".join(lines)
    )


@app.on_message(filters.command("status") & filters.private)
async def status_handler(_, message):
    if not owner_only(message):
        return
    started = time.perf_counter()
    try:
        await app.get_me()
        ping_ms = (time.perf_counter() - started) * 1000
    except Exception:
        ping_ms = -1

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu = process.cpu_percent(interval=None)
    uptime = int(time.monotonic() - START_TIME)
    hours, rem = divmod(uptime, 3600)
    minutes, seconds = divmod(rem, 60)
    ping_text = f"{ping_ms:.1f} ms" if ping_ms >= 0 else "Unavailable"

    await message.reply_text(
        "📊 **OtakuAura Status**\n\n"
        f"⏱ Uptime: {hours}h {minutes}m {seconds}s\n"
        f"📡 Telegram ping: {ping_text}\n"
        f"💾 RAM: {ram_mb:.1f} MB\n"
        f"⚙️ CPU: {cpu:.1f}%\n"
        f"🗄 Database: {db_mode}\n"
        f"🔑 Active API keys: {len(api_keys)}\n"
        f"👥 Registered users: {len(users)}"
    )


@app.on_message(filters.command(["user", "users"]) & filters.private)
async def users_handler(_, message):
    if not owner_only(message):
        return
    total = len(users)
    if db is not None:
        try:
            total = await db.users.count_documents({})
        except Exception:
            pass
    await message.reply_text(f"👥 Registered users: **{total}**")


@app.on_message(filters.command("broadcast") & filters.private)
async def broadcast_handler(_, message):
    if not owner_only(message):
        return
    if not message.reply_to_message:
        return await message.reply_text("Reply to a message/photo with /broadcast.")

    source = message.reply_to_message
    status = await message.reply_text("📢 Broadcast started…")
    targets = set(users)
    if db is not None:
        try:
            async for doc in db.users.find({}, {"_id": 1}):
                targets.add(int(doc["_id"]))
        except Exception:
            log.exception("Could not fetch MongoDB user list.")

    sent = 0
    failed = 0
    lock = asyncio.Lock()
    limit = asyncio.Semaphore(10)

    async def deliver(uid):
        nonlocal sent, failed
        async with limit:
            try:
                await app.copy_message(
                    chat_id=uid,
                    from_chat_id=source.chat.id,
                    message_id=source.id
                )
                async with lock:
                    sent += 1
            except Exception:
                async with lock:
                    failed += 1

    await asyncio.gather(*(deliver(uid) for uid in targets))
    await status.edit_text(
        "📢 **Broadcast completed**\n\n"
        f"✅ Sent: {sent}\n"
        f"❌ Failed: {failed}\n"
        f"👥 Total: {len(targets)}"
    )


# -------------------- Entrypoint --------------------

async def main():
    if not API_ID or not API_HASH or not BOT_TOKEN:
        raise RuntimeError("Set API_ID, API_HASH and BOT_TOKEN environment variables.")

    if GEMINI_API_KEY and GEMINI_API_KEY not in api_keys:
        api_keys.insert(0, GEMINI_API_KEY)
    reset_key_rotation()

    await init_database()
    await app.start()
    me = await app.get_me()
    log.info("Started as @%s | database=%s | keys=%d", me.username, db_mode, len(api_keys))

    try:
        await idle()
    finally:
        await app.stop()
        if mongo_client is not None:
            mongo_client.close()


if __name__ == "__main__":
    app.run(main())
