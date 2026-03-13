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
        self.message.append(
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
        # 注意：system 提示词建议放在 __init__ 中，这里只管 append 用户输入
        self.message.append({"role": "user", "content": user_input})

        # 第一次请求
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.message,
            tools=[get_weather_spec],
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 【核心修正 1】：必须保存模型发出的工具调用指令
        self.message.append(response_message)

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                # 注意：API 返回的 arguments 是字符串，需要 json.loads
                import json
                func_args = json.loads(tool_call.function.arguments)

                print(f"DEBUG: 成功捕获工具调用：{func_name}, 参数：{func_args}")

                # 模拟执行结果
                result = get_weather(func_args)

                # 【核心修正 2】：tool 消息只需这三个字段
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),  # 转为 JSON 字符串
                }
                self.message.append(tool_result_message)

            # 【核心修正 3】：第二次请求不需要再传 tools 和 tool_choice
            final_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.message
            )

            final_answer = final_response.choices[0].message.content
            self.message.append({"role": "assistant", "content": final_answer})
            return final_answer

        else:
            # 如果模型直接回答（没调工具）
            content = response_message.content
            # 此时 response_message 已经 append 过了，不需要额外操作
            return content
