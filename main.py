import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from telethon import TelegramClient, events, Button
from telethon.tl.types import User
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction

try:
    import g4f
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
        TeachAnything,
    )
except ImportError as e:
    print(f"Ошибка импорта g4f: {e}")
    print("Установите библиотеку командой: pip install -U g4f")
    sys.exit(1)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

load_dotenv()

# Telegram API credentials
API_ID: int = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
SESSION_NAME: str = os.getenv("SESSION_NAME", "userbot_session")

# ============================================================================
# ПРОВАЙДЕРЫ И МОДЕЛИ
# ============================================================================

@dataclass
class ProviderConfig:
    """Конфигурация провайдера."""
    name: str
    provider: type
    models: list[str]
    description: str
    requires_auth: bool = False


# Список рабочих провайдеров (без авторизации) - актуально на 2026
PROVIDERS: dict[str, ProviderConfig] = {
    "ddg": ProviderConfig(
        name="DuckDuckGo",
        provider=DDG,
        models=["gpt-4o-mini", "claude-3-haiku", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        description="🦆 Стабильный, быстрый"
    ),
    "blackbox": ProviderConfig(
        name="Blackbox AI",
        provider=Blackbox,
        models=["blackboxai", "gpt-4o", "claude-sonnet-3.5", "gemini-pro", "llama-3.1-70b"],
        description="⬛ Много моделей"
    ),
    "pollinations": ProviderConfig(
        name="Pollinations",
        provider=PollinationsAI,
        models=["openai", "openai-large", "qwen-coder", "llama", "mistral", "deepseek-r1"],
        description="🌸 Креативный"
    ),
    "free2gpt": ProviderConfig(
        name="Free2GPT",
        provider=Free2GPT,
        models=["llama-3.1-70b"],
        description="🆓 Llama 3.1"
    ),
    "liaobots": ProviderConfig(
        name="Liaobots",
        provider=Liaobots,
        models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
        description="🤖 GPT-4o доступен"
    ),
    "airforce": ProviderConfig(
        name="Airforce",
        provider=Airforce,
        models=["llama-3-70b-chat", "mixtral-8x7b", "qwen-72b"],
        description="✈️ Мощные модели"
    ),
    "chatgptes": ProviderConfig(
        name="ChatGptEs",
        provider=ChatGptEs,
        models=["gpt-4o", "gpt-4o-mini"],
        description="🇪🇸 GPT через ES"
    ),
    "freegpt": ProviderConfig(
        name="FreeGpt",
        provider=FreeGpt,
        models=["gemini-pro"],
        description="💎 Gemini Pro"
    ),
    "teachanything": ProviderConfig(
        name="TeachAnything",
        provider=TeachAnything,
        models=["llama-3.1-70b"],
        description="📚 Обучающий"
    ),
}

# Порядок fallback при ошибках
FALLBACK_ORDER: list[str] = ["ddg", "blackbox", "pollinations", "free2gpt", "liaobots", "airforce"]

# ============================================================================
# НАСТРОЙКИ БОТА (изменяемые в runtime)
# ============================================================================

@dataclass
class BotSettings:
    """Настройки бота."""
    enabled: bool = True
    current_provider: str = "ddg"
    current_model: str = "gpt-4o-mini"
    only_private: bool = True
    auto_fallback: bool = True
    max_history: int = 10
    system_prompt: str = (
        "Ты — дружелюбный и полезный AI-ассистент. "
        "Отвечай кратко, по существу и на языке собеседника. "
        "Если тебе пишут на русском — отвечай на русском."
    )
    ignore_list: set[int] = field(default_factory=set)
    whitelist: set[int] = field(default_factory=set)
    send_error_msg: bool = False
    timeout: int = 60
    max_response_len: int = 4000


# Глобальные настройки
settings = BotSettings()

# История переписок
conversation_history: dict[int, list[dict[str, str]]] = {}

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("g4f").setLevel(logging.WARNING)

logger = logging.getLogger("UserBot")

# ============================================================================
# ИСТОРИЯ СООБЩЕНИЙ
# ============================================================================

def get_history(user_id: int) -> list[dict[str, str]]:
    """Получить историю для пользователя."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return conversation_history[user_id]


def add_to_history(user_id: int, role: str, content: str) -> None:
    """Добавить сообщение в историю."""
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > settings.max_history:
        conversation_history[user_id] = history[-settings.max_history:]


def clear_history(user_id: int) -> None:
    """Очистить историю пользователя."""
    conversation_history[user_id] = []


# ============================================================================
# ГЕНЕРАЦИЯ ОТВЕТОВ
# ============================================================================

async def generate_response(
    message: str,
    user_id: int,
    provider_key: Optional[str] = None,
    model: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Генерирует ответ через g4f.
    
    Returns:
        (ответ, использованный_провайдер) или (None, None) при ошибке
    """
    provider_key = provider_key or settings.current_provider
    model = model or settings.current_model
    
    # Список провайдеров для попытки
    providers_to_try = [provider_key]
    if settings.auto_fallback:
        providers_to_try.extend([p for p in FALLBACK_ORDER if p != provider_key])
    
    add_to_history(user_id, "user", message)
    
    messages = [{"role": "system", "content": settings.system_prompt}]
    messages.extend(get_history(user_id))
    
    last_error = None
    
    for pkey in providers_to_try:
        if pkey not in PROVIDERS:
            continue
            
        pconfig = PROVIDERS[pkey]
        
        # Выбираем модель: если текущая не поддерживается провайдером, берём первую из списка
        use_model = model if model in pconfig.models else pconfig.models[0]
        
        try:
            logger.info(f"Пробуем {pconfig.name} с моделью {use_model}")
            
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
                    # Убираем рекламу если есть
                    text = clean_response(text)
                    
                    if len(text) > settings.max_response_len:
                        text = text[:settings.max_response_len] + "..."
                    
                    add_to_history(user_id, "assistant", text)
                    logger.info(f"✅ Успех через {pconfig.name} ({len(text)} символов)")
                    return text, pconfig.name
            
            logger.warning(f"Пустой ответ от {pconfig.name}")
            
        except asyncio.TimeoutError:
            last_error = f"Таймаут ({settings.timeout}с)"
            logger.warning(f"⏱ Таймаут для {pconfig.name}")
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"❌ Ошибка {pconfig.name}: {type(e).__name__}: {e}")
        
        if not settings.auto_fallback:
            break
    
    logger.error(f"Все провайдеры не сработали. Последняя ошибка: {last_error}")
    # Удаляем последнее сообщение пользователя из истории при ошибке
    history = get_history(user_id)
    if history and history[-1]["role"] == "user":
        history.pop()
    
    return None, None


def clean_response(text: str) -> str:
    """Очищает ответ от рекламы и мусора."""
    # Удаляем типичную рекламу g4f провайдеров
    spam_patterns = [
        "Want best roleplay experience?",
        "https://llmplayground",
        "Try our new",
        "Visit us at",
        "Powered by",
        "Generated by",
    ]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        if not any(spam in line for spam in spam_patterns):
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()


# ============================================================================
# INLINE КНОПКИ
# ============================================================================

def get_main_menu_buttons() -> list[list[Button]]:
    """Главное меню настроек."""
    status = "✅ ВКЛ" if settings.enabled else "❌ ВЫКЛ"
    return [
        [Button.inline(f"🔘 Автоответчик: {status}", b"toggle_enabled")],
        [Button.inline("🤖 Провайдер", b"menu_provider"), 
         Button.inline("🧠 Модель", b"menu_model")],
        [Button.inline("⚙️ Настройки", b"menu_settings"),
         Button.inline("📊 Статус", b"show_status")],
        [Button.inline("🗑 Очистить историю", b"clear_all_history"),
         Button.inline("❌ Закрыть", b"close_menu")]
    ]


def get_provider_buttons() -> list[list[Button]]:
    """Кнопки выбора провайдера."""
    buttons = []
    row = []
    
    for key, pconfig in PROVIDERS.items():
        mark = "✓ " if key == settings.current_provider else ""
        btn = Button.inline(f"{mark}{pconfig.name}", f"set_provider:{key}".encode())
        row.append(btn)
        
        if len(row) == 2:
            buttons.append(row)
            row = []
    
    if row:
        buttons.append(row)
    
    buttons.append([Button.inline("◀️ Назад", b"main_menu")])
    return buttons


def get_model_buttons() -> list[list[Button]]:
    """Кнопки выбора модели для текущего провайдера."""
    pconfig = PROVIDERS.get(settings.current_provider)
    if not pconfig:
        return [[Button.inline("◀️ Назад", b"main_menu")]]
    
    buttons = []
    for model in pconfig.models:
        mark = "✓ " if model == settings.current_model else ""
        # Сокращаем длинные названия моделей
        display_name = model[:25] + "..." if len(model) > 28 else model
        buttons.append([Button.inline(f"{mark}{display_name}", f"set_model:{model}".encode())])
    
    buttons.append([Button.inline("◀️ Назад", b"main_menu")])
    return buttons


def get_settings_buttons() -> list[list[Button]]:
    """Кнопки настроек."""
    private_status = "✅" if settings.only_private else "❌"
    fallback_status = "✅" if settings.auto_fallback else "❌"
    error_msg_status = "✅" if settings.send_error_msg else "❌"
    
    return [
        [Button.inline(f"📨 Только ЛС: {private_status}", b"toggle_private")],
        [Button.inline(f"🔄 Авто-fallback: {fallback_status}", b"toggle_fallback")],
        [Button.inline(f"⚠️ Сообщать об ошибках: {error_msg_status}", b"toggle_error_msg")],
        [Button.inline("📝 Изменить промпт", b"edit_prompt")],
        [Button.inline("⏱ Таймаут: " + str(settings.timeout) + "с", b"cycle_timeout")],
        [Button.inline("◀️ Назад", b"main_menu")]
    ]


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_user_name(user: User) -> str:
    """Получить имя пользователя."""
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or f"@{user.username}" if user.username else f"User#{user.id}"


def should_respond(user_id: int, is_private: bool) -> bool:
    """Проверить, нужно ли отвечать."""
    if not settings.enabled:
        return False
    if settings.only_private and not is_private:
        return False
    if user_id in settings.ignore_list:
        return False
    if settings.whitelist and user_id not in settings.whitelist:
        return False
    return True


def get_status_text() -> str:
    """Текст статуса бота."""
    pconfig = PROVIDERS.get(settings.current_provider)
    provider_name = pconfig.name if pconfig else "Unknown"
    
    return (
        "📊 **Статус бота**\n\n"
        f"• Автоответчик: {'✅ включён' if settings.enabled else '❌ выключен'}\n"
        f"• Провайдер: **{provider_name}**\n"
        f"• Модель: `{settings.current_model}`\n"
        f"• Только ЛС: {'да' if settings.only_private else 'нет'}\n"
        f"• Авто-fallback: {'да' if settings.auto_fallback else 'нет'}\n"
        f"• Таймаут: {settings.timeout}с\n"
        f"• История: {settings.max_history} сообщений\n"
        f"• Активных диалогов: {len(conversation_history)}\n"
        f"• Игнор-лист: {len(settings.ignore_list)}\n"
        f"• Whitelist: {len(settings.whitelist) if settings.whitelist else 'выкл'}"
    )


# ============================================================================
# ОБРАБОТЧИКИ СОБЫТИЙ
# ============================================================================

def setup_handlers(client: TelegramClient) -> None:
    """Настройка всех обработчиков."""
    
    # ========== CALLBACK QUERY (INLINE КНОПКИ) ==========
    @client.on(events.CallbackQuery)
    async def callback_handler(event):
        """Обработчик нажатий на inline-кнопки."""
        data = event.data.decode()
        
        try:
            if data == "toggle_enabled":
                settings.enabled = not settings.enabled
                await event.edit(
                    "⚙️ **Панель управления**\n\nВыберите действие:",
                    buttons=get_main_menu_buttons()
                )
                status = "включён ✅" if settings.enabled else "выключен ❌"
                logger.info(f"Автоответчик {status}")
                
            elif data == "menu_provider":
                pconfig = PROVIDERS.get(settings.current_provider)
                text = (
                    f"🤖 **Выбор провайдера**\n\n"
                    f"Текущий: **{pconfig.name if pconfig else 'N/A'}**\n"
                    f"Описание: {pconfig.description if pconfig else 'N/A'}"
                )
                await event.edit(text, buttons=get_provider_buttons())
                
            elif data == "menu_model":
                pconfig = PROVIDERS.get(settings.current_provider)
                text = (
                    f"🧠 **Выбор модели**\n\n"
                    f"Провайдер: **{pconfig.name if pconfig else 'N/A'}**\n"
                    f"Текущая модель: `{settings.current_model}`"
                )
                await event.edit(text, buttons=get_model_buttons())
                
            elif data == "menu_settings":
                await event.edit(
                    "⚙️ **Настройки**\n\nВыберите параметр:",
                    buttons=get_settings_buttons()
                )
                
            elif data == "show_status":
                await event.edit(get_status_text(), buttons=[
                    [Button.inline("◀️ Назад", b"main_menu")]
                ])
                
            elif data.startswith("set_provider:"):
                provider_key = data.split(":")[1]
                if provider_key in PROVIDERS:
                    settings.current_provider = provider_key
                    pconfig = PROVIDERS[provider_key]
                    # Сбрасываем модель на первую доступную у провайдера
                    if settings.current_model not in pconfig.models:
                        settings.current_model = pconfig.models[0]
                    
                    await event.answer(f"✅ Провайдер: {pconfig.name}", alert=False)
                    
                    text = (
                        f"🤖 **Выбор провайдера**\n\n"
                        f"Текущий: **{pconfig.name}**\n"
                        f"Описание: {pconfig.description}"
                    )
                    await event.edit(text, buttons=get_provider_buttons())
                    logger.info(f"Провайдер изменён на {pconfig.name}")
                    
            elif data.startswith("set_model:"):
                model = data.split(":", 1)[1]
                settings.current_model = model
                await event.answer(f"✅ Модель: {model[:20]}", alert=False)
                
                pconfig = PROVIDERS.get(settings.current_provider)
                text = (
                    f"🧠 **Выбор модели**\n\n"
                    f"Провайдер: **{pconfig.name if pconfig else 'N/A'}**\n"
                    f"Текущая модель: `{settings.current_model}`"
                )
                await event.edit(text, buttons=get_model_buttons())
                logger.info(f"Модель изменена на {model}")
                
            elif data == "toggle_private":
                settings.only_private = not settings.only_private
                await event.edit(
                    "⚙️ **Настройки**\n\nВыберите параметр:",
                    buttons=get_settings_buttons()
                )
                
            elif data == "toggle_fallback":
                settings.auto_fallback = not settings.auto_fallback
                await event.edit(
                    "⚙️ **Настройки**\n\nВыберите параметр:",
                    buttons=get_settings_buttons()
                )
                
            elif data == "toggle_error_msg":
                settings.send_error_msg = not settings.send_error_msg
                await event.edit(
                    "⚙️ **Настройки**\n\nВыберите параметр:",
                    buttons=get_settings_buttons()
                )
                
            elif data == "cycle_timeout":
                # Циклический выбор таймаута
                timeouts = [30, 45, 60, 90, 120]
                try:
                    idx = timeouts.index(settings.timeout)
                    settings.timeout = timeouts[(idx + 1) % len(timeouts)]
                except ValueError:
                    settings.timeout = 60
                await event.edit(
                    "⚙️ **Настройки**\n\nВыберите параметр:",
                    buttons=get_settings_buttons()
                )
                
            elif data == "edit_prompt":
                await event.answer(
                    "📝 Отправьте новый системный промпт командой:\n.ai prompt <текст>",
                    alert=True
                )
                
            elif data == "clear_all_history":
                conversation_history.clear()
                await event.answer("🗑 Вся история очищена!", alert=True)
                logger.info("Вся история очищена")
                
            elif data == "main_menu":
                await event.edit(
                    "⚙️ **Панель управления**\n\nВыберите действие:",
                    buttons=get_main_menu_buttons()
                )
                
            elif data == "close_menu":
                await event.delete()
                
        except Exception as e:
            logger.error(f"Ошибка callback: {e}")
            await event.answer(f"Ошибка: {str(e)[:50]}", alert=True)
    
    # ========== КОМАНДЫ (ИСХОДЯЩИЕ СООБЩЕНИЯ) ==========
    @client.on(events.NewMessage(outgoing=True, pattern=r"\.ai\s*(.*)"))
    async def command_handler(event):
        """Обработчик команд .ai"""
        args = event.pattern_match.group(1).strip().lower().split()
        cmd = args[0] if args else ""
        
        try:
            await event.delete()
        except:
            pass
        
        if cmd in ("", "menu", "help"):
            # Показать меню с кнопками
            await client.send_message(
                "me",
                "⚙️ **Панель управления**\n\nВыберите действие:",
                buttons=get_main_menu_buttons()
            )
            
        elif cmd == "on":
            settings.enabled = True
            await client.send_message("me", "✅ Автоответчик **включён**")
            logger.info("Автоответчик включён")
            
        elif cmd == "off":
            settings.enabled = False
            await client.send_message("me", "❌ Автоответчик **выключен**")
            logger.info("Автоответчик выключен")
            
        elif cmd == "status":
            await client.send_message("me", get_status_text())
            
        elif cmd == "clear":
            chat = await event.get_chat()
            if hasattr(chat, 'id'):
                clear_history(chat.id)
                await client.send_message("me", f"🗑 История с {chat.id} очищена")
                
        elif cmd == "prompt" and len(args) > 1:
            new_prompt = " ".join(event.pattern_match.group(1).split()[1:])
            settings.system_prompt = new_prompt
            await client.send_message(
                "me",
                f"📝 Системный промпт обновлён:\n\n`{new_prompt[:200]}...`"
            )
            logger.info("Системный промпт обновлён")
            
        elif cmd == "ignore":
            subcmd = args[1] if len(args) > 1 else "list"
            
            if subcmd == "add" and len(args) > 2:
                try:
                    uid = int(args[2])
                    settings.ignore_list.add(uid)
                    await client.send_message("me", f"✅ {uid} добавлен в игнор-лист")
                except ValueError:
                    await client.send_message("me", "⚠️ Неверный ID")
                    
            elif subcmd == "remove" and len(args) > 2:
                try:
                    uid = int(args[2])
                    settings.ignore_list.discard(uid)
                    await client.send_message("me", f"✅ {uid} удалён из игнор-листа")
                except ValueError:
                    await client.send_message("me", "⚠️ Неверный ID")
                    
            elif subcmd == "list":
                if settings.ignore_list:
                    ids = ", ".join(str(x) for x in settings.ignore_list)
                    await client.send_message("me", f"📋 Игнор-лист: `{ids}`")
                else:
                    await client.send_message("me", "📋 Игнор-лист пуст")
                    
        elif cmd == "test":
            # Тестовый запрос
            await client.send_message("me", "🧪 Тестирую провайдер...")
            response, provider = await generate_response("Привет! Скажи 'работает' если ты меня слышишь.", 0)
            if response:
                await client.send_message("me", f"✅ **{provider}** работает!\n\n{response[:500]}")
            else:
                await client.send_message("me", "❌ Провайдер не отвечает. Попробуйте другой.")
                
        else:
            # Неизвестная команда - показать справку
            help_text = """
📋 **Команды:**

`.ai` или `.ai menu` — Открыть панель управления
`.ai on/off` — Вкл/выкл автоответчик
`.ai status` — Показать статус
`.ai clear` — Очистить историю текущего чата
`.ai test` — Тестировать провайдер
`.ai prompt <текст>` — Изменить системный промпт
`.ai ignore add/remove/list [id]` — Управление игнор-листом
"""
            await client.send_message("me", help_text)
    
    # ========== ВХОДЯЩИЕ СООБЩЕНИЯ ==========
    @client.on(events.NewMessage(incoming=True))
    async def message_handler(event):
        """Обработчик входящих сообщений."""
        sender = await event.get_sender()
        
        if not isinstance(sender, User) or sender.is_self:
            return
        
        text = event.raw_text
        if not text or not text.strip():
            return
        
        user_id = sender.id
        user_name = get_user_name(sender)
        is_private = event.is_private
        
        chat_type = "ЛС" if is_private else "Группа"
        logger.info(f"📨 [{chat_type}] {user_name} ({user_id}): {text[:80]}...")
        
        if not should_respond(user_id, is_private):
            return
        
        try:
            chat = await event.get_chat()
            
            # Статус "печатает..."
            await client(SetTypingRequest(
                peer=chat,
                action=SendMessageTypingAction()
            ))
            
            # Генерация ответа
            response, provider = await generate_response(text, user_id)
            
            if response:
                await event.respond(response)
                logger.info(f"📤 [{provider}] → {user_name}: {response[:60]}...")
            else:
                logger.warning(f"Не удалось сгенерировать ответ для {user_id}")
                if settings.send_error_msg:
                    await event.respond("⚠️ Извините, произошла ошибка. Попробуйте позже.")
                    
        except Exception as e:
            logger.error(f"Ошибка обработки: {type(e).__name__}: {e}")
            if settings.send_error_msg:
                try:
                    await event.respond("⚠️ Произошла техническая ошибка.")
                except:
                    pass


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    """Главная функция."""
    if not API_ID or not API_HASH:
        logger.error("=" * 60)
        logger.error("ОШИБКА: Не заданы TELEGRAM_API_ID и TELEGRAM_API_HASH!")
        logger.error("")
        logger.error("Создайте файл .env:")
        logger.error("  TELEGRAM_API_ID=12345678")
        logger.error("  TELEGRAM_API_HASH=abcdef1234567890")
        logger.error("")
        logger.error("Получить: https://my.telegram.org/apps")
        logger.error("=" * 60)
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("🚀 Telegram Userbot Auto-Responder v2.0")
    logger.info("=" * 60)
    
    pconfig = PROVIDERS.get(settings.current_provider)
    logger.info(f"📱 Сессия: {SESSION_NAME}")
    logger.info(f"🤖 Провайдер: {pconfig.name if pconfig else 'N/A'}")
    logger.info(f"🧠 Модель: {settings.current_model}")
    logger.info(f"🔄 Авто-fallback: {'да' if settings.auto_fallback else 'нет'}")
    logger.info("=" * 60)
    
    client = TelegramClient(
        SESSION_NAME,
        API_ID,
        API_HASH,
        system_version="4.16.30-vxCUSTOM"
    )
    
    setup_handlers(client)
    
    logger.info("🔐 Подключение к Telegram...")
    await client.start()
    
    me = await client.get_me()
    logger.info(f"✅ Авторизован: {get_user_name(me)} (ID: {me.id})")
    logger.info("")
    logger.info("📋 Отправьте .ai в любой чат для открытия меню")
    logger.info("🎯 Бот готов к работе!")
    logger.info("=" * 60)
    
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Остановлено пользователем")
    except Exception as e:
        logger.critical(f"💥 Критическая ошибка: {e}")
        sys.exit(1)
