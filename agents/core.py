from dotenv import load_dotenv

import os

from openai import OpenAI

from tools.weather import get_weather, get_weather_spec

load_dotenv()  # 加载 .env 文件

class DeepSeekAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.tools_map = {
            "get_weather": get_weather
        }
        self.tools_spec = [get_weather_spec]
        self.message = []

    def chat(self, user_input):
        self.message.append({"role": "user", "content": user_input})

        second_response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.message,
            functions=self.tools_spec
        )
        final_content = second_response.choices[0].message.content
        # self.messages.append({"role": "assistant", "content": final_content})
        return final_content
