from dotenv import load_dotenv
import os
from openai import OpenAI

from memory.reflector import MemoryReflector
from tools.weather import get_weather, get_weather_spec
from memory.json_storage import JSONMemory

load_dotenv()  # 加载 .env 文件

class DeepSeekAgent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory_manager = JSONMemory(user_id)
        self.messages = self.memory_manager.load()
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.reflector = MemoryReflector(self.client)
        self.tools_map = {
            "get_weather": get_weather
        }
        self.tools_spec = [get_weather_spec]
        # self.message = []
        self.messages.append(
            {
                "role": "system",
                "content": (
                    "你是一个实战派助手。当用户询问天气、计算等问题时，"
                    "只要你手里有对应的工具，就必须【立即调用】工具，"
                    "不要向用户确认，不要反问。执行完工具后，再回答用户。"
                )
            }
        )

    def chat(self, user_input):
        # 动态植入长期记忆
        profile = self.reflector.load_profile(self.user_id)

        # 如果profile不为空，则添加到系统提示词中
        if profile:
            profile = "无"


        self.messages[0]["content"] += f"你是一个助手。已知用户信息：{profile}"

        # 注意：system 提示词建议放在 __init__ 中，这里只管 append 用户输入
        self.messages.append({"role": "user", "content": user_input})

        max_steps = 5  # 防止死循环
        step = 0
        while step < max_steps:

            # 第一次请求
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=[get_weather_spec],
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 必须保存模型发出的工具调用指令
            self.messages.append(response_message)

            # 如果不需要工具调用，则直接返回结果
            if not response_message.tool_calls:
                # 如果模型直接回答（没调工具）
                content = response_message.content
                # 此时 response_message 已经 append 过了，不需要额外操作
                self.memory_manager.save(self.messages)
                return content

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                # 注意：API 返回的 arguments 是字符串，需要 json.loads
                import json
                func_args = json.loads(tool_call.function.arguments)

                print(f"DEBUG: 成功捕获工具调用：{func_name}, 参数：{func_args}")

                # 模拟执行结果
                result = get_weather(func_args)

                # tool 消息只需这三个字段
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),  # 转为 JSON 字符串
                }
                self.messages.append(tool_result_message)

            # 第二次请求不需要再传 tools 和 tool_choice
            final_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages
            )

            final_answer = final_response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": final_answer})

            self.memory_manager.save(self.messages)
            step += 1
            return final_answer
