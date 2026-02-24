import logging
import httpx
from weather_engine.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"

class TelegramNotifier:
    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)

    def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        if not self.enabled:
            logger.debug("Telegram not configured, skipping notification")
            return False
        try:
            url = TELEGRAM_API.format(token=self.token) + "/sendMessage"
            resp = httpx.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)
            resp.raise_for_status()
            logger.info("Telegram message sent")
            return True
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False
