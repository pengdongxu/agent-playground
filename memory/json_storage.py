import json
import os

from config import Config


class JSONMemory:
    def __init__(self, user_id, max_history=15):
        self.user_id = user_id
        # 直接使用 config 里的路径
        self.file_path = Config.HISTORY_DIR / f"{user_id}.json"
        self.max_history = max_history

    def load(self):
        """加载历史记录"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # 默认返回空，让Agent自己决定初始System prompt
        return []

    def save(self, messages):
        """保存并裁剪历史记录"""
        # 永远保留第 0 条 (System Prompt)
        if len(messages) > self.max_history:
            system_msg = messages[0]
            recent_msgs = messages[-(self.max_history - 1):]
            messages = [system_msg] + recent_msgs

        # 将消息列表中的对象（OpenAI回复对象）序列化
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.preprocess_messages(messages), f, ensure_ascii=False, indent=2)


    def preprocess_messages(self, messages):
        """处理 messages 列表中的 Pydantic 对象，确保能被 JSON 序列化"""
        processed = []
        for msg in messages:
            if hasattr(msg, "model_dump"):  # 针对 OpenAI 最新的响应对象
                processed.append(msg.model_dump())
            elif isinstance(msg, dict):
                processed.append(msg)
            else:
                # 最后的兜底
                processed.append(str(msg))
        return processed


    def clear(self):
        if self.file_path.exists():
            self.file_path.unlink()