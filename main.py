#!/usr/bin/env python3
"""
Telegram AI Auto-Responder System v3.1
=======================================
• Control Bot (aiogram 3.24): Панель управления с inline-кнопками
• Userbot (Telethon): Автоответы на ЛС через g4f (актуальный API)

Author: Claude AI Assistant
License: MIT
"""

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

# G4F - актуальный импорт
try:
    from g4f.client import Client as G4FClient
except ImportError as e:
    print(f"Ошибка импорта g4f: {e}")
    print("Установите: pip install -U g4f")
    sys.exit(1)

# ============================================================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================================================

load_dotenv()

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
API_ID: int = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME: str = os.getenv("SESSION_NAME", "userbot_session")
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
# МОДЕЛИ G4F (без указания провайдеров - автовыбор)
# ============================================================================

# Доступные модели для выбора
AVAILABLE_MODELS: dict[str, dict] = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "emoji": "🤖",
        "description": "Быстрый и дешёвый"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "emoji": "🧠",
        "description": "Мощный мультимодальный"
    },
    "gpt-4": {
        "name": "GPT-4",
        "emoji": "💎",
        "description": "Классический GPT-4"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "emoji": "⚡",
        "description": "Быстрый базовый"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "emoji": "🎭",
        "description": "Быстрый Claude"
    },
    "claude-3-sonnet": {
        "name": "Claude 3 Sonnet",
        "emoji": "🎵",
        "description": "Сбалансированный Claude"
    },
    "llama-3.1-70b": {
        "name": "Llama 3.1 70B",
        "emoji": "🦙",
        "description": "Мощная открытая модель"
    },
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B",
        "emoji": "🦙",
        "description": "Лёгкая Llama"
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B",
        "emoji": "🌀",
        "description": "MoE модель"
    },
    "gemini-pro": {
        "name": "Gemini Pro",
        "emoji": "♊",
        "description": "Google Gemini"
    },
    "deepseek-chat": {
        "name": "DeepSeek Chat",
        "emoji": "🔍",
        "description": "DeepSeek V3"
    },
    "qwen-turbo": {
        "name": "Qwen Turbo",
        "emoji": "🐲",
        "description": "Alibaba Qwen"
    },
}

# ============================================================================
# НАСТРОЙКИ (ГЛОБАЛЬНОЕ СОСТОЯНИЕ)
# ============================================================================

@dataclass
class BotSettings:
    """Настройки автоответчика."""
    enabled: bool = True
    current_model: str = "gpt-4o-mini"
    only_private: bool = True
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
# ГЕНЕРАЦИЯ ОТВЕТОВ G4F (АКТУАЛЬНЫЙ API)
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
        "I am an AI",
        "I'm just an AI",
    ]
    lines = text.split("\n")
    cleaned = [line for line in lines if not any(spam in line for spam in spam_patterns)]
    return "\n".join(cleaned).strip()


async def generate_response(message: str, user_id: int) -> tuple[Optional[str], str]:
    """
    Генерирует ответ через g4f.
    
    Returns:
        (ответ, модель) или (None, "") при ошибке
    """
    add_to_history(user_id, "user", message)

    messages = [{"role": "system", "content": settings.system_prompt}]
    messages.extend(get_history(user_id))

    # Список моделей для попытки (текущая + fallback)
    models_to_try = [settings.current_model]
    fallback_models = ["gpt-4o-mini", "gpt-3.5-turbo", "llama-3.1-70b"]
    models_to_try.extend([m for m in fallback_models if m != settings.current_model])

    for model in models_to_try:
        try:
            logger.info(f"Пробуем модель: {model}")

            client = G4FClient()

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda m=model: client.chat.completions.create(
                        model=m,
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
                    
                    model_info = AVAILABLE_MODELS.get(model, {})
                    model_name = model_info.get("name", model)
                    logger.info(f"✅ {model_name}: {len(text)} символов")
                    return text, model_name

            logger.warning(f"Пустой ответ от {model}")

        except asyncio.TimeoutError:
            logger.warning(f"⏱ Таймаут: {model}")
        except Exception as e:
            logger.warning(f"❌ {model}: {type(e).__name__}: {e}")

    # Ошибка - удаляем из истории
    history = get_history(user_id)
    if history and history[-1]["role"] == "user":
        history.pop()

    settings.stats_errors += 1
    return None, ""

# ============================================================================
# КЛАВИАТУРЫ (INLINE KEYBOARDS)
# ============================================================================

def kb_main_menu() -> InlineKeyboardMarkup:
    """Главное меню."""
    status_emoji = "✅" if settings.enabled else "❌"
    status_text = "ВКЛ" if settings.enabled else "ВЫКЛ"
    
    model_info = AVAILABLE_MODELS.get(settings.current_model, {})
    model_emoji = model_info.get("emoji", "🤖")

    buttons = [
        [InlineKeyboardButton(
            text=f"🔘 Автоответчик: {status_emoji} {status_text}",
            callback_data="toggle_enabled"
        )],
        [InlineKeyboardButton(
            text=f"{model_emoji} Модель: {settings.current_model}",
            callback_data="menu_model"
        )],
        [
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="menu_settings"),
            InlineKeyboardButton(text="📊 Статистика", callback_data="show_stats")
        ],
        [
            InlineKeyboardButton(text="📝 Промпт", callback_data="menu_prompt"),
            InlineKeyboardButton(text="🚫 Игнор-лист", callback_data="menu_ignore")
        ],
        [
            InlineKeyboardButton(text="🧪 Тест", callback_data="test_model"),
            InlineKeyboardButton(text="🗑 Очистить", callback_data="clear_history")
        ],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="close_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_models() -> InlineKeyboardMarkup:
    """Выбор модели."""
    buttons = []
    row = []

    for model_id, info in AVAILABLE_MODELS.items():
        mark = "✓ " if model_id == settings.current_model else ""
        btn = InlineKeyboardButton(
            text=f"{mark}{info['emoji']} {info['name']}",
            callback_data=f"set_model:{model_id}"
        )
        row.append(btn)
        if len(row) == 2:
            buttons.append(row)
            row = []

    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_settings() -> InlineKeyboardMarkup:
    """Настройки."""
    private = "✅" if settings.only_private else "❌"
    error_msg = "✅" if settings.send_error_msg else "❌"

    buttons = [
        [InlineKeyboardButton(text=f"📨 Только ЛС: {private}", callback_data="toggle_private")],
        [InlineKeyboardButton(text=f"⚠️ Сообщать об ошибках: {error_msg}", callback_data="toggle_error_msg")],
        [InlineKeyboardButton(text=f"⏱ Таймаут: {settings.timeout}с", callback_data="cycle_timeout")],
        [InlineKeyboardButton(text=f"📚 История: {settings.max_history} сообщений", callback_data="cycle_history")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_prompt() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="✏️ Изменить промпт", callback_data="edit_prompt")],
        [InlineKeyboardButton(text="🔄 Сбросить", callback_data="reset_prompt")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_ignore() -> InlineKeyboardMarkup:
    count = len(settings.ignore_list)
    buttons = [
        [InlineKeyboardButton(text=f"📋 Список ({count})", callback_data="ignore_list")],
        [InlineKeyboardButton(text="➕ Добавить", callback_data="ignore_add")],
        [InlineKeyboardButton(text="➖ Удалить", callback_data="ignore_remove")],
        [InlineKeyboardButton(text="🗑 Очистить", callback_data="ignore_clear")],
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
    model_info = AVAILABLE_MODELS.get(settings.current_model, {})
    model_name = f"{model_info.get('emoji', '🤖')} {model_info.get('name', settings.current_model)}"
    status = "✅ Включён" if settings.enabled else "❌ Выключен"

    return (
        "🎛 <b>Панель управления автоответчиком</b>\n\n"
        f"📍 Статус: {status}\n"
        f"🧠 Модель: {model_name}\n\n"
        "Выберите действие:"
    )


def get_stats_text() -> str:
    model_info = AVAILABLE_MODELS.get(settings.current_model, {})
    model_name = f"{model_info.get('emoji', '🤖')} {model_info.get('name', settings.current_model)}"

    return (
        "📊 <b>Статистика</b>\n\n"
        f"📨 Получено: <b>{settings.stats_messages}</b>\n"
        f"📤 Отправлено: <b>{settings.stats_responses}</b>\n"
        f"❌ Ошибок: <b>{settings.stats_errors}</b>\n\n"
        f"💬 Диалогов: <b>{len(conversation_history)}</b>\n"
        f"🚫 Игнор: <b>{len(settings.ignore_list)}</b>\n\n"
        "<b>Настройки:</b>\n"
        f"• Модель: {model_name}\n"
        f"• Только ЛС: {'да' if settings.only_private else 'нет'}\n"
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
        await message.answer("⛔ Нет доступа.")
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
    await message.answer(get_main_menu_text(), reply_markup=kb_main_menu(), parse_mode=ParseMode.HTML)


@router.message(Command("status"))
async def cmd_status(message: Message):
    if not is_admin(message.from_user.id):
        return
    await message.answer(get_stats_text(), reply_markup=kb_back(), parse_mode=ParseMode.HTML)


# ============================================================================
# CALLBACK HANDLERS
# ============================================================================

@router.callback_query(F.data == "main_menu")
async def cb_main_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(get_main_menu_text(), reply_markup=kb_main_menu(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "toggle_enabled")
async def cb_toggle_enabled(callback: CallbackQuery):
    settings.enabled = not settings.enabled
    status = "включён ✅" if settings.enabled else "выключен ❌"
    await callback.message.edit_text(get_main_menu_text(), reply_markup=kb_main_menu(), parse_mode=ParseMode.HTML)
    await callback.answer(f"Автоответчик {status}")
    logger.info(f"Автоответчик {status}")


@router.callback_query(F.data == "menu_model")
async def cb_menu_model(callback: CallbackQuery):
    model_info = AVAILABLE_MODELS.get(settings.current_model, {})
    text = (
        "🧠 <b>Выбор модели</b>\n\n"
        f"Текущая: <b>{model_info.get('emoji', '🤖')} {model_info.get('name', settings.current_model)}</b>\n"
        f"<i>{model_info.get('description', '')}</i>\n\n"
        "g4f автоматически выберет рабочий провайдер."
    )
    await callback.message.edit_text(text, reply_markup=kb_models(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data.startswith("set_model:"))
async def cb_set_model(callback: CallbackQuery):
    model_id = callback.data.split(":", 1)[1]

    if model_id in AVAILABLE_MODELS:
        settings.current_model = model_id
        model_info = AVAILABLE_MODELS[model_id]

        text = (
            "🧠 <b>Выбор модели</b>\n\n"
            f"Текущая: <b>{model_info['emoji']} {model_info['name']}</b>\n"
            f"<i>{model_info['description']}</i>\n\n"
            "g4f автоматически выберет рабочий провайдер."
        )
        await callback.message.edit_text(text, reply_markup=kb_models(), parse_mode=ParseMode.HTML)
        await callback.answer(f"✅ {model_info['name']}")
        logger.info(f"Модель: {model_id}")


@router.callback_query(F.data == "menu_settings")
async def cb_menu_settings(callback: CallbackQuery):
    await callback.message.edit_text("⚙️ <b>Настройки</b>", reply_markup=kb_settings(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "toggle_private")
async def cb_toggle_private(callback: CallbackQuery):
    settings.only_private = not settings.only_private
    await callback.message.edit_text("⚙️ <b>Настройки</b>", reply_markup=kb_settings(), parse_mode=ParseMode.HTML)
    await callback.answer(f"{'Только ЛС' if settings.only_private else 'Все чаты'}")


@router.callback_query(F.data == "toggle_error_msg")
async def cb_toggle_error_msg(callback: CallbackQuery):
    settings.send_error_msg = not settings.send_error_msg
    await callback.message.edit_text("⚙️ <b>Настройки</b>", reply_markup=kb_settings(), parse_mode=ParseMode.HTML)
    await callback.answer(f"Ошибки: {'вкл' if settings.send_error_msg else 'выкл'}")


@router.callback_query(F.data == "cycle_timeout")
async def cb_cycle_timeout(callback: CallbackQuery):
    timeouts = [30, 45, 60, 90, 120]
    try:
        idx = timeouts.index(settings.timeout)
        settings.timeout = timeouts[(idx + 1) % len(timeouts)]
    except ValueError:
        settings.timeout = 60
    await callback.message.edit_text("⚙️ <b>Настройки</b>", reply_markup=kb_settings(), parse_mode=ParseMode.HTML)
    await callback.answer(f"Таймаут: {settings.timeout}с")


@router.callback_query(F.data == "cycle_history")
async def cb_cycle_history(callback: CallbackQuery):
    sizes = [5, 10, 15, 20, 30]
    try:
        idx = sizes.index(settings.max_history)
        settings.max_history = sizes[(idx + 1) % len(sizes)]
    except ValueError:
        settings.max_history = 10
    await callback.message.edit_text("⚙️ <b>Настройки</b>", reply_markup=kb_settings(), parse_mode=ParseMode.HTML)
    await callback.answer(f"История: {settings.max_history}")


@router.callback_query(F.data == "show_stats")
async def cb_show_stats(callback: CallbackQuery):
    await callback.message.edit_text(get_stats_text(), reply_markup=kb_back(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "menu_prompt")
async def cb_menu_prompt(callback: CallbackQuery):
    await callback.message.edit_text(get_prompt_text(), reply_markup=kb_prompt(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "edit_prompt")
async def cb_edit_prompt(callback: CallbackQuery, state: FSMContext):
    await state.set_state(PromptStates.waiting_for_prompt)
    await callback.message.edit_text(
        "📝 <b>Изменение промпта</b>\n\nОтправьте новый текст:",
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
        await message.answer("⚠️ Минимум 10 символов")
        return

    settings.system_prompt = new_prompt
    await state.clear()
    await message.answer(f"✅ <b>Промпт обновлён!</b>\n\n<i>{new_prompt[:150]}...</i>", reply_markup=kb_back(), parse_mode=ParseMode.HTML)
    logger.info("Промпт обновлён")


@router.callback_query(F.data == "reset_prompt")
async def cb_reset_prompt(callback: CallbackQuery):
    settings.system_prompt = (
        "Ты — дружелюбный и полезный AI-ассистент. "
        "Отвечай кратко, по существу и на языке собеседника. "
        "Если пишут на русском — отвечай на русском."
    )
    await callback.message.edit_text(get_prompt_text(), reply_markup=kb_prompt(), parse_mode=ParseMode.HTML)
    await callback.answer("✅ Сброшено")


@router.callback_query(F.data == "menu_ignore")
async def cb_menu_ignore(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🚫 <b>Игнор-лист</b>\n\nВ списке: <b>{len(settings.ignore_list)}</b>",
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
        text = "🚫 <b>Список пуст</b>"
    await callback.message.edit_text(text, reply_markup=kb_ignore(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "ignore_add")
async def cb_ignore_add(callback: CallbackQuery, state: FSMContext):
    await state.set_state(IgnoreStates.waiting_for_id)
    await state.update_data(action="add")
    await callback.message.edit_text("➕ <b>Добавить</b>\n\nОтправьте ID:", reply_markup=kb_cancel(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.callback_query(F.data == "ignore_remove")
async def cb_ignore_remove(callback: CallbackQuery, state: FSMContext):
    await state.set_state(IgnoreStates.waiting_for_id)
    await state.update_data(action="remove")
    await callback.message.edit_text("➖ <b>Удалить</b>\n\nОтправьте ID:", reply_markup=kb_cancel(), parse_mode=ParseMode.HTML)
    await callback.answer()


@router.message(IgnoreStates.waiting_for_id)
async def process_ignore_id(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return

    try:
        user_id = int(message.text.strip())
    except ValueError:
        await message.answer("⚠️ Неверный ID")
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
    await callback.message.edit_text("🚫 <b>Игнор-лист</b>\n\nСписок очищен.", reply_markup=kb_ignore(), parse_mode=ParseMode.HTML)
    await callback.answer("✅ Очищено")


@router.callback_query(F.data == "test_model")
async def cb_test_model(callback: CallbackQuery):
    await callback.answer("🧪 Тестирую...")
    await callback.message.edit_text("🧪 <b>Тестирование...</b>", parse_mode=ParseMode.HTML)

    response, model_name = await generate_response("Скажи 'работает' одним словом", user_id=0)

    if response:
        text = f"✅ <b>Успех!</b>\n\nМодель: <b>{model_name}</b>\nОтвет: <i>{response[:200]}</i>"
    else:
        text = "❌ <b>Ошибка</b>\n\nПопробуйте другую модель."

    await callback.message.edit_text(text, reply_markup=kb_back(), parse_mode=ParseMode.HTML)


@router.callback_query(F.data == "clear_history")
async def cb_clear_history(callback: CallbackQuery):
    await callback.message.edit_text(
        f"🗑 <b>Очистка</b>\n\nДиалогов: <b>{len(conversation_history)}</b>\n\nОчистить?",
        reply_markup=kb_confirm_clear(),
        parse_mode=ParseMode.HTML
    )
    await callback.answer()


@router.callback_query(F.data == "confirm_clear")
async def cb_confirm_clear(callback: CallbackQuery):
    clear_all_history()
    await callback.message.edit_text(get_main_menu_text(), reply_markup=kb_main_menu(), parse_mode=ParseMode.HTML)
    await callback.answer("✅ Очищено")
    logger.info("История очищена")


@router.callback_query(F.data == "cancel_action")
async def cb_cancel_action(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(get_main_menu_text(), reply_markup=kb_main_menu(), parse_mode=ParseMode.HTML)
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
        logger.warning("⚠️ Userbot не настроен")
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

            response, model_name = await generate_response(text, user_id)

            if response:
                await event.respond(response)
                logger.info(f"📤 [{model_name}] → {user_name}")
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
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не задан!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 Telegram AI Auto-Responder v3.1")
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
        logger.warning("⚠️ Userbot отключён")
        await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Остановлено")
    except Exception as e:
        logger.critical(f"💥 {e}")
        sys.exit(1)
