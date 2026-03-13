from config import Config

class MemoryReflector:

    def __init__(self, client):
        self.client = client


    def reflect(self, user_id, messages):
        # 仅取最近10条记录进行反思
        recent_context = str(messages[-10:])

        prompt = f"分析此对话并提取用户画像（所在地、偏好等），直接输出摘要，不超50字：{recent_context}"

        try:
            res = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = res.choices[0].message.content
            path = Config.PROFILE_DIR / f"{user_id}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(summary)
        except Exception as e:
            print(f"ERROR: 反思失败: {e}")

    def load_profile(self, user_id):
        path = Config.PROFILE_DIR / f"{user_id}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else "暂无"