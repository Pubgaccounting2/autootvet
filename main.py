import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Aiogram 3.x
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand
)
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# Telethon
from telethon import TelegramClient, events
from telethon.tl.types import User
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction

# G4F
try:
    from g4f.client import Client as G4FClient
    from g4f.Provider import (
        DDG,
        Blackbox,
        PollinationsAI,
        Free2GPT,
        Liaobots,
        Airforce,
        ChatGptEs,
        FreeGpt,
    )
except ImportError as e:
    print(f"Ошибка импорта g4f: {e}")
    print("Установите: pip install -U g4f")
    sys.exit(1)

# ============================================================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================================================

load_dotenv()

# Telegram Bot (aiogram) - панель управления
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")

# Telegram Userbot (Telethon) - автоответчик
API_ID: int = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME: str = os.getenv("SESSION_NAME", "userbot_session")

# Admin ID - кто может управлять ботом (0 = все)
ADMIN_ID: int = int(os.getenv("ADMIN_ID", "0"))

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("aiogram").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("g4f").setLevel(logging.WARNING)

logger = logging.getLogger("AutoResponder")

# ============================================================================
# ПРОВАЙДЕРЫ G4F
# ============================================================================

@dataclass
class ProviderConfig:
    """Конфигурация провайдера."""
    name: str
    provider: type
    models: list[str]
    emoji: str
    description: str


PROVIDERS: dict[str, ProviderConfig] = {
    "ddg": ProviderConfig(
        name="DuckDuckGo",
        provider=DDG,
        models=["gpt-4o-mini", "claude-3-haiku", "llama-3.3-70b", "mixtral-8x7b"],
        emoji="🦆",
        description="Стабильный и быстрый"
    ),
    "blackbox": ProviderConfig(
        name="Blackbox AI",
        provider=Blackbox,
        models=["blackboxai", "gpt-4o", "claude-sonnet-3.5", "gemini-pro", "llama-3.1-70b"],
        emoji="⬛",
        description="Много моделей"
    ),
    "pollinations": ProviderConfig(
        name="Pollinations",
        provider=PollinationsAI,
        models=["openai", "openai-large", "mistral", "llama", "deepseek-r1"],
        emoji="🌸",
        description="Креативные ответы"
    ),
    "free2gpt": ProviderConfig(
        name="Free2GPT",
        provider=Free2GPT,
        models=["llama-3.1-70b"],
        emoji="🆓",
        description="Бесплатный Llama"
    ),
    "liaobots": ProviderConfig(
        name="Liaobots",
        provider=Liaobots,
        models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
        emoji="🤖",
        description="GPT-4o доступен"
    ),
    "airforce": ProviderConfig(
        name="Airforce",
        provider=Airforce,
        models=["llama-3-70b-chat", "mixtral-8x7b", "qwen-72b"],
        emoji="✈️",
        description="Мощные модели"
    ),
    "chatgptes": ProviderConfig(
        name="ChatGptEs",
        provider=ChatGptEs,
        models=["gpt-4o", "gpt-4o-mini"],
        emoji="🇪🇸",
        description="GPT через ES"
    ),
    "freegpt": ProviderConfig(
        name="FreeGpt",
        provider=FreeGpt,
        models=["gemini-pro"],
        emoji="💎",
        description="Gemini Pro"
    ),
}

FALLBACK_ORDER: list[str] = ["ddg", "blackbox", "pollinations", "free2gpt", "liaobots"]

# ============================================================================
# НАСТРОЙКИ (ГЛОБАЛЬНОЕ СОСТОЯНИЕ)
# ============================================================================

@dataclass
class BotSettings:
    """Настройки автоответчика."""
    enabled: bool = True
    current_provider: str = "ddg"
    current_model: str = "gpt-4o-mini"
    only_private: bool = True
    auto_fallback: bool = True
    send_error_msg: bool = False
    max_history: int = 10
    timeout: int = 60
    max_response_len: int = 4000
    system_prompt: str = (
        "Ты — дружелюбный и полезный AI-ассистент. "
        "Отвечай кратко, по существу и на языке собеседника. "
        "Если пишут на русском — отвечай на русском."
    )
    ignore_list: set[int] = field(default_factory=set)
    whitelist: set[int] = field(default_factory=set)
    stats_messages: int = 0
    stats_responses: int = 0
    stats_errors: int = 0


settings = BotSettings()
conversation_history: dict[int, list[dict[str, str]]] = {}

# ============================================================================
# FSM STATES
# ============================================================================

class PromptStates(StatesGroup):
    waiting_for_prompt = State()


class IgnoreStates(StatesGroup):
    waiting_for_id = State()

# ============================================================================
# ИСТОРИЯ СООБЩЕНИЙ
# ============================================================================

def get_history(user_id: int) -> list[dict[str, str]]:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return conversation_history[user_id]


def add_to_history(user_id: int, role: str, content: str) -> None:
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > settings.max_history:
        conversation_history[user_id] = history[-settings.max_history:]


def clear_user_history(user_id: int) -> None:
    conversation_history[user_id] = []


def clear_all_history() -> None:
    conversation_history.clear()

# ============================================================================
# ГЕНЕРАЦИЯ ОТВЕТОВ G4F
# ============================================================================

def clean_response(text: str) -> str:
    """Очищает ответ от рекламы."""
    spam_patterns = [
        "Want best roleplay experience?",
        "https://llmplayground",
        "Try our new",
        "Visit us at",
        "Generated by",
        "Powered by",
    ]
    lines = text.split("\n")
    cleaned = [line for line in lines if not any(spam in line for spam in spam_patterns)]
    return "\n".join(cleaned).strip()


async def generate_response(
    message: str,
    user_id: int,
    provider_key: Optional[str] = None,
    model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Генерирует ответ через g4f."""
    provider_key = provider_key or settings.current_provider
    model = model or settings.current_model

    providers_to_try = [provider_key]
    if settings.auto_fallback:
        providers_to_try.extend([p for p in FALLBACK_ORDER if p != provider_key])

    add_to_history(user_id, "user", message)

    messages = [{"role": "system", "content": settings.system_prompt}]
    messages.extend(get_history(user_id))

    for pkey in providers_to_try:
        if pkey not in PROVIDERS:
            continue

        pconfig = PROVIDERS[pkey]
        use_model = model if model in pconfig.models else pconfig.models[0]

        try:
            logger.info(f"Пробуем {pconfig.name} ({use_model})")

            client = G4FClient(provider=pconfig.provider)

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model=use_model,
                        messages=messages,
                    )
                ),
                timeout=settings.timeout
            )

            if response and response.choices:
                text = response.choices[0].message.content

                if text and len(text.strip()) > 0:
                    text = clean_response(text)

                    if len(text) > settings.max_response_len:
                        text = text[:settings.max_response_len] + "..."

                    add_to_history(user_id, "assistant", text)
                    settings.stats_responses += 1
                    logger.info(f"✅ {pconfig.name}: {len(text)} символов")
                    return text, pconfig.name

            logger.warning(f"Пустой ответ от {pconfig.name}")

        except asyncio.TimeoutError:
            logger.warning(f"⏱ Таймаут: {pconfig.name}")
        except Exception as e:
            logger.warning(f"❌ {pconfig.name}: {type(e).__name__}: {e}")

        if not settings.auto_fallback:
            break

    history = get_history(user_id)
    if history and history[-1]["role"] == "user":
        history.pop()

    settings.stats_errors += 1
    return None, None

# ============================================================================
# КЛАВИАТУРЫ (INLINE KEYBOARDS)
# ============================================================================

def kb_main_menu() -> InlineKeyboardMarkup:
    """Главное меню."""
    status_emoji = "✅" if settings.enabled else "❌"
    status_text = "ВКЛ" if settings.enabled else "ВЫКЛ"

    buttons = [
        [InlineKeyboardButton(
            text=f"🔘 Автоответчик: {status_emoji} {status_text}",
            callback_data="toggle_enabled"
        )],
        [
            InlineKeyboardButton(text="🤖 Провайдер", callback_data="menu_provider"),
            InlineKeyboardButton(text="🧠 Модель", callback_data="menu_model")
        ],
        [
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="menu_settings"),
            InlineKeyboardButton(text="📊 Статистика", callback_data="show_stats")
        ],
        [
            InlineKeyboardButton(text="📝 Промпт", callback_data="menu_prompt"),
            InlineKeyboardButton(text="🚫 Игнор-лист", callback_data="menu_ignore")
        ],
        [
            InlineKeyboardButton(text="🧪 Тест", callback_data="test_provider"),
            InlineKeyboardButton(text="🗑 Очистить историю", callback_data="clear_history")
        ],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="close_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_providers() -> InlineKeyboardMarkup:
    """Выбор провайдера."""
    buttons = []
    row = []

    for key, pconfig in PROVIDERS.items():
        mark = "✓ " if key == settings.current_provider else ""
        btn = InlineKeyboardButton(
            text=f"{mark}{pconfig.emoji} {pconfig.name}",
            callback_data=f"set_provider:{key}"
        )
        row.append(btn)
        if len(row) == 2:
            buttons.append(row)
            row = []

    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_models() -> InlineKeyboardMarkup:
    """Выбор модели."""
    pconfig = PROVIDERS.get(settings.current_provider)
    if not pconfig:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
        ])

    buttons = []
    for model in pconfig.models:
        mark = "✓ " if model == settings.current_model else ""
        display = model[:28] + "..." if len(model) > 31 else model
        buttons.append([InlineKeyboardButton(
            text=f"{mark}{display}",
            callback_data=f"set_model:{model}"
        )])

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_settings() -> InlineKeyboardMarkup:
    """Настройки."""
    private = "✅" if settings.only_private else "❌"
    fallback = "✅" if settings.auto_fallback else "❌"
    error_msg = "✅" if settings.send_error_msg else "❌"

    buttons = [
        [InlineKeyboardButton(text=f"📨 Только ЛС: {private}", callback_data="toggle_private")],
        [InlineKeyboardButton(text=f"🔄 Авто-fallback: {fallback}", callback_data="toggle_fallback")],
        [InlineKeyboardButton(text=f"⚠️ Сообщать об ошибках: {error_msg}", callback_data="toggle_error_msg")],
        [InlineKeyboardButton(text=f"⏱ Таймаут: {settings.timeout}с", callback_data="cycle_timeout")],
        [InlineKeyboardButton(text=f"📚 История: {settings.max_history} сообщений", callback_data="cycle_history")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_prompt() -> InlineKeyboardMarkup:
    """Меню промпта."""
    buttons = [
        [InlineKeyboardButton(text="✏️ Изменить промпт", callback_data="edit_prompt")],
        [InlineKeyboardButton(text="🔄 Сбросить по умолчанию", callback_data="reset_prompt")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_ignore() -> InlineKeyboardMarkup:
    """Меню игнор-листа."""
    count = len(settings.ignore_list)
    buttons = [
        [InlineKeyboardButton(text=f"📋 Список ({count})", callback_data="ignore_list")],
        [InlineKeyboardButton(text="➕ Добавить ID", callback_data="ignore_add")],
        [InlineKeyboardButton(text="➖ Удалить ID", callback_data="ignore_remove")],
        [InlineKeyboardButton(text="🗑 Очистить всё", callback_data="ignore_clear")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ])


def kb_cancel() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❌ Отмена", callback_data="cancel_action")]
    ])


def kb_confirm_clear() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да", callback_data="confirm_clear"),
            InlineKeyboardButton(text="❌ Нет", callback_data="main_menu")
        ]
    ])

# ============================================================================
# ТЕКСТОВЫЕ СООБЩЕНИЯ
# ============================================================================

def get_main_menu_text() -> str:
    pconfig = PROVIDERS.get(settings.current_provider)
    provider_name = f"{pconfig.emoji} {pconfig.name}" if pconfig else "N/A"
    status = "✅ Включён" if settings.enabled else "❌ Выключен"

    return (
        "🎛 <b>Панель управления автоответчиком</b>\n\n"
        f"📍 Статус: {status}\n"
        f"🤖 Провайдер: {provider_name}\n"
        f"🧠 Модель: <code>{settings.current_model}</code>\n\n"
        "Выберите действие:"
    )


def get_stats_text() -> str:
    pconfig = PROVIDERS.get(settings.current_provider)
    provider_name = f"{pconfig.emoji} {pconfig.name}" if pconfig else "N/A"

    return (
        "📊 <b>Статистика</b>\n\n"
        f"📨 Получено сообщений: <b>{settings.stats_messages}</b>\n"
        f"📤 Отправлено ответов: <b>{settings.stats_responses}</b>\n"
        f"❌ Ошибок: <b>{settings.stats_errors}</b>\n\n"
        f"💬 Активных диалогов: <b>{len(conversation_history)}</b>\n"
        f"🚫 В игнор-листе: <b>{len(settings.ignore_list)}</b>\n\n"
        "<b>Текущие настройки:</b>\n"
        f"• Провайдер: {provider_name}\n"
        f"• Модель: <code>{settings.current_model}</code>\n"
        f"• Только ЛС: {'да' if settings.only_private else 'нет'}\n"
        f"• Fallback: {'да' if settings.auto_fallback else 'нет'}\n"
        f"• Таймаут: {settings.timeout}с"
    )


def get_prompt_text() -> str:
    prompt_preview = settings.system_prompt[:200]
    if len(settings.system_prompt) > 200:
        prompt_preview += "..."

    return (
        "📝 <b>Системный промпт</b>\n\n"
        f"<i>{prompt_preview}</i>\n\n"
        f"Длина: {len(settings.system_prompt)} символов"
    )

# ============================================================================
# AIOGRAM ROUTER
# ============================================================================

router = Router()


def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID or ADMIN_ID == 0


@router.message(CommandStart())
async def cmd_start(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔ У вас нет доступа к этому боту.")
        return

    await message.answer(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )


@router.message(Command("menu"))
async def cmd_menu(message: Message):
    if not is_admin(message.from_user.id):
        return

    await message.answer(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not is_admin(message.from_user.id):
        return

    await message.answer(
        get_stats_text(),
        reply_markup=kb_back(),
        parse_mode=ParseMode.HTML
    )


# ============================================================================
# CALLBACK HANDLERS
# ============================================================================

@router.callback_query(F.data == "main_menu")
async def cb_main_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "toggle_enabled")
async def cb_toggle_enabled(callback: CallbackQuery):
    settings.enabled = not settings.enabled
    status = "включён ✅" if settings.enabled else "выключен ❌"

    await callback.message.edit_text(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"Автоответчик {status}")
    logger.info(f"Автоответчик {status}")


@router.callback_query(F.data == "menu_provider")
async def cb_menu_provider(callback: CallbackQuery):
    pconfig = PROVIDERS.get(settings.current_provider)
    text = (
        "🤖 <b>Выбор провайдера</b>\n\n"
        f"Текущий: <b>{pconfig.emoji} {pconfig.name}</b>\n"
        f"<i>{pconfig.description}</i>"
    ) if pconfig else "Провайдер не выбран"

    await callback.message.edit_text(text, reply_markup=kb_providers(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data.startswith("set_provider:"))
async def cb_set_provider(callback: CallbackQuery):
    provider_key = callback.data.split(":")[1]

    if provider_key in PROVIDERS:
        settings.current_provider = provider_key
        pconfig = PROVIDERS[provider_key]

        if settings.current_model not in pconfig.models:
            settings.current_model = pconfig.models[0]

        text = (
            "🤖 <b>Выбор провайдера</b>\n\n"
            f"Текущий: <b>{pconfig.emoji} {pconfig.name}</b>\n"
            f"<i>{pconfig.description}</i>"
        )
        await callback.message.edit_text(text, reply_markup=kb_providers(), parse_mode=ParseMode.HTML)
        await callback.answer(f"✅ {pconfig.name}")
        logger.info(f"Провайдер: {pconfig.name}")


@router.callback_query(F.data == "menu_model")
async def cb_menu_model(callback: CallbackQuery):
    pconfig = PROVIDERS.get(settings.current_provider)
    text = (
        "🧠 <b>Выбор модели</b>\n\n"
        f"Провайдер: <b>{pconfig.emoji} {pconfig.name}</b>\n"
        f"Текущая: <code>{settings.current_model}</code>"
    ) if pconfig else "Провайдер не выбран"

    await callback.message.edit_text(text, reply_markup=kb_models(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data.startswith("set_model:"))
async def cb_set_model(callback: CallbackQuery):
    model = callback.data.split(":", 1)[1]
    settings.current_model = model

    pconfig = PROVIDERS.get(settings.current_provider)
    text = (
        "🧠 <b>Выбор модели</b>\n\n"
        f"Провайдер: <b>{pconfig.emoji} {pconfig.name}</b>\n"
        f"Текущая: <code>{settings.current_model}</code>"
    ) if pconfig else "Провайдер не выбран"

    await callback.message.edit_text(text, reply_markup=kb_models(), parse_mode=ParseMode.HTML)
    await callback.answer(f"✅ {model[:20]}")
    logger.info(f"Модель: {model}")


@router.callback_query(F.data == "menu_settings")
async def cb_menu_settings(callback: CallbackQuery):
    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "toggle_private")
async def cb_toggle_private(callback: CallbackQuery):
    settings.only_private = not settings.only_private
    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"Режим: {'только ЛС' if settings.only_private else 'все чаты'}")


@router.callback_query(F.data == "toggle_fallback")
async def cb_toggle_fallback(callback: CallbackQuery):
    settings.auto_fallback = not settings.auto_fallback
    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"Fallback: {'вкл' if settings.auto_fallback else 'выкл'}")


@router.callback_query(F.data == "toggle_error_msg")
async def cb_toggle_error_msg(callback: CallbackQuery):
    settings.send_error_msg = not settings.send_error_msg
    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"Ошибки: {'вкл' if settings.send_error_msg else 'выкл'}")


@router.callback_query(F.data == "cycle_timeout")
async def cb_cycle_timeout(callback: CallbackQuery):
    timeouts = [30, 45, 60, 90, 120]
    try:
        idx = timeouts.index(settings.timeout)
        settings.timeout = timeouts[(idx + 1) % len(timeouts)]
    except ValueError:
        settings.timeout = 60

    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"Таймаут: {settings.timeout}с")


@router.callback_query(F.data == "cycle_history")
async def cb_cycle_history(callback: CallbackQuery):
    sizes = [5, 10, 15, 20, 30]
    try:
        idx = sizes.index(settings.max_history)
        settings.max_history = sizes[(idx + 1) % len(sizes)]
    except ValueError:
        settings.max_history = 10

    await callback.message.edit_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        reply_markup=kb_settings(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer(f"История: {settings.max_history}")


@router.callback_query(F.data == "show_stats")
async def cb_show_stats(callback: CallbackQuery):
    await callback.message.edit_text(
        get_stats_text(),
        reply_markup=kb_back(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "menu_prompt")
async def cb_menu_prompt(callback: CallbackQuery):
    await callback.message.edit_text(
        get_prompt_text(),
        reply_markup=kb_prompt(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "edit_prompt")
async def cb_edit_prompt(callback: CallbackQuery, state: FSMContext):
    await state.set_state(PromptStates.waiting_for_prompt)
    await callback.message.edit_text(
        "📝 <b>Изменение промпта</b>\n\n"
        "Отправьте новый системный промпт текстовым сообщением.\n\n"
        f"<i>Текущий:</i>\n<code>{settings.system_prompt[:200]}...</code>",
        reply_markup=kb_cancel(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.message(PromptStates.waiting_for_prompt)
async def process_new_prompt(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return

    new_prompt = message.text.strip()
    if len(new_prompt) < 10:
        await message.answer("⚠️ Промпт слишком короткий (минимум 10 символов)")
        return

    settings.system_prompt = new_prompt
    await state.clear()

    await message.answer(
        f"✅ <b>Промпт обновлён!</b>\n\n<i>{new_prompt[:200]}...</i>",
        reply_markup=kb_back(),
        parse_mode=ParseMode.HTML
    )
    logger.info("Системный промпт обновлён")


@router.callback_query(F.data == "reset_prompt")
async def cb_reset_prompt(callback: CallbackQuery):
    settings.system_prompt = (
        "Ты — дружелюбный и полезный AI-ассистент. "
        "Отвечай кратко, по существу и на языке собеседника. "
        "Если пишут на русском — отвечай на русском."
    )
    await callback.message.edit_text(
        get_prompt_text(),
        reply_markup=kb_prompt(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer("✅ Промпт сброшен")


@router.callback_query(F.data == "menu_ignore")
async def cb_menu_ignore(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🚫 <b>Игнор-лист</b>\n\n"
        f"Пользователей: <b>{len(settings.ignore_list)}</b>\n\n"
        "Эти пользователи не получат автоответы.",
        reply_markup=kb_ignore(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "ignore_list")
async def cb_ignore_list(callback: CallbackQuery):
    if settings.ignore_list:
        ids = "\n".join(f"• <code>{uid}</code>" for uid in settings.ignore_list)
        text = f"🚫 <b>Игнор-лист:</b>\n\n{ids}"
    else:
        text = "🚫 <b>Игнор-лист пуст</b>"

    await callback.message.edit_text(text, reply_markup=kb_ignore(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "ignore_add")
async def cb_ignore_add(callback: CallbackQuery, state: FSMContext):
    await state.set_state(IgnoreStates.waiting_for_id)
    await state.update_data(action="add")

    await callback.message.edit_text(
        "➕ <b>Добавить в игнор-лист</b>\n\nОтправьте ID пользователя:",
        reply_markup=kb_cancel(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "ignore_remove")
async def cb_ignore_remove(callback: CallbackQuery, state: FSMContext):
    await state.set_state(IgnoreStates.waiting_for_id)
    await state.update_data(action="remove")

    await callback.message.edit_text(
        "➖ <b>Удалить из игнор-листа</b>\n\nОтправьте ID пользователя:",
        reply_markup=kb_cancel(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.message(IgnoreStates.waiting_for_id)
async def process_ignore_id(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return

    try:
        user_id = int(message.text.strip())
    except ValueError:
        await message.answer("⚠️ Неверный формат ID")
        return

    data = await state.get_data()
    action = data.get("action", "add")

    if action == "add":
        settings.ignore_list.add(user_id)
        text = f"✅ <code>{user_id}</code> добавлен"
    else:
        settings.ignore_list.discard(user_id)
        text = f"✅ <code>{user_id}</code> удалён"

    await state.clear()
    await message.answer(text, reply_markup=kb_ignore(), parse_mode=ParseMode.HTML)


@router.callback_query(F.data == "ignore_clear")
async def cb_ignore_clear(callback: CallbackQuery):
    settings.ignore_list.clear()
    await callback.message.edit_text(
        "🚫 <b>Игнор-лист</b>\n\nПользователей: <b>0</b>\n\nСписок очищен.",
        reply_markup=kb_ignore(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer("✅ Очищено")


@router.callback_query(F.data == "test_provider")
async def cb_test_provider(callback: CallbackQuery):
    await callback.answer("🧪 Тестирую...")
    await callback.message.edit_text(
        "🧪 <b>Тестирование...</b>\n\nПожалуйста, подождите.",
        parse_mode=ParseMode.HTML
    )

    response, provider = await generate_response("Скажи 'работает' одним словом", user_id=0)

    if response:
        text = f"✅ <b>Успех!</b>\n\nПровайдер: <b>{provider}</b>\nОтвет: <i>{response[:200]}</i>"
    else:
        text = "❌ <b>Ошибка</b>\n\nПопробуйте другой провайдер."

    await callback.message.edit_text(text, reply_markup=kb_back(), parse_mode=ParseMode.HTML)


@router.callback_query(F.data == "clear_history")
async def cb_clear_history(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🗑 <b>Очистка истории</b>\n\nДиалогов: <b>{len(conversation_history)}</b>\n\nОчистить?",
        reply_markup=kb_confirm_clear(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "confirm_clear")
async def cb_confirm_clear(callback: CallbackQuery):
    clear_all_history()
    await callback.message.edit_text(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer("✅ История очищена")
    logger.info("История очищена")


@router.callback_query(F.data == "cancel_action")
async def cb_cancel_action(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        get_main_menu_text(),
        reply_markup=kb_main_menu(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer("Отменено")


@router.callback_query(F.data == "close_menu")
async def cb_close_menu(callback: CallbackQuery):
    await callback.message.delete()
    await callback.answer()

# ============================================================================
# TELETHON USERBOT
# ============================================================================

def get_user_name(user: User) -> str:
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or (f"@{user.username}" if user.username else f"ID:{user.id}")


def should_respond(user_id: int, is_private: bool) -> bool:
    if not settings.enabled:
        return False
    if settings.only_private and not is_private:
        return False
    if user_id in settings.ignore_list:
        return False
    if settings.whitelist and user_id not in settings.whitelist:
        return False
    return True


async def run_userbot():
    """Запуск Telethon userbot."""
    if not API_ID or not API_HASH:
        logger.warning("⚠️ Userbot не настроен (API_ID/API_HASH)")
        return

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        sender = await event.get_sender()

        if not isinstance(sender, User) or sender.is_self:
            return

        text = event.raw_text
        if not text or not text.strip():
            return

        user_id = sender.id
        user_name = get_user_name(sender)
        is_private = event.is_private

        settings.stats_messages += 1

        chat_type = "ЛС" if is_private else "Группа"
        logger.info(f"📨 [{chat_type}] {user_name} ({user_id}): {text[:50]}...")

        if not should_respond(user_id, is_private):
            return

        try:
            chat = await event.get_chat()
            await client(SetTypingRequest(peer=chat, action=SendMessageTypingAction()))

            response, provider = await generate_response(text, user_id)

            if response:
                await event.respond(response)
                logger.info(f"📤 [{provider}] → {user_name}")
            elif settings.send_error_msg:
                await event.respond("⚠️ Ошибка. Попробуйте позже.")

        except Exception as e:
            logger.error(f"Ошибка: {e}")

    logger.info("🔐 Подключение Userbot...")
    await client.start()

    me = await client.get_me()
    logger.info(f"✅ Userbot: {get_user_name(me)} (ID: {me.id})")

    await client.run_until_disconnected()

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Главная функция."""
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не задан в .env!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 Telegram AI Auto-Responder v3.0")
    logger.info("=" * 60)

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="start", description="Панель управления"),
        BotCommand(command="menu", description="Открыть меню"),
        BotCommand(command="status", description="Статистика"),
    ])

    logger.info("🤖 Запуск Control Bot...")

    if API_ID and API_HASH:
        logger.info("📱 Запуск Userbot...")
        await asyncio.gather(
            dp.start_polling(bot),
            run_userbot()
        )
    else:
        logger.warning("⚠️ Userbot отключён (нет API_ID/API_HASH)")
        await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Остановлено")
    except Exception as e:
        logger.critical(f"💥 Ошибка: {e}")
        sys.exit(1)
