import importlib
import pkgutil

from dotenv import load_dotenv
import os
from openai import OpenAI
import tools
import json

from memory.reflector import MemoryReflector
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
        self.messages.append(
            {
                "role": "system",
                "content": (
                    "你是一个基于 ReAct (Reasoning and Acting) 范式的智能助手。\n"
                    "在解答用户问题时，你必须遵循“先思考，后行动”的逻辑。\n\n"
                    "【执行规范】\n"
                    "1. 思考 (Thought)：在你决定调用任何工具之前，你必须在回复的正文内容中，详细写下你的思考过程（例如：分析用户的需求、说明为什么需要这个工具、下一步计划做什么）。\n"
                    "2. 行动 (Action)：思考完成后，再调用对应的工具。\n"
                    "3. 观察 (Observation)：分析工具返回的结果。如果信息不足，继续 思考->行动 的循环。\n"
                    "4. 最终回答：当你收集到足够的信息时，直接输出最终答案，不需要再调用工具。"
                )
            }
        )

        self.tools_map = {}  # 用于：函数名 -> 执行函数 的映射
        self.tools_spec = []  # 用于：发送给 LLM 的说明书列表
        self._load_tools()

    def chat(self, user_input):
        # 1. 安全处理 Profile (建议不修改全局的 messages[0]，而是作为独立的系统提示插入，或者在初始化时处理好)
        profile = self.reflector.load_profile(self.user_id)
        context_msg = f"已知用户信息：{profile}" if profile else "已知用户信息：无"

        # 将用户当前输入和上下文合并
        current_input = f"{context_msg}\n用户输入: {user_input}"
        self.messages.append({"role": "user", "content": current_input})

        max_steps = 5  # 防止死循环
        step = 0
        # 真正的执行循环
        while step < max_steps:
            print(f"\n--- [Agent 思考中... 第 {step + 1} 轮] ---")

            # 向大模型发起请求
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=self.tools_spec if self.tools_spec else None
            )

            response_message = response.choices[0].message

            # 将模型的回复加入记忆
            self.messages.append(response_message)

            # --- 核心分支判断 ---
            # 如果模型没有调用工具（意味着它觉得可以直接回答了）
            if not response_message.tool_calls:
                final_answer = response_message.content
                self.memory_manager.save(self.messages)  # 保存记忆
                return final_answer

            # 如果模型决定调用工具
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"DEBUG: ⚡ 触发工具调用: {func_name}, 参数: {func_args}")

                # 执行工具
                if func_name in self.tools_map:
                    try:
                        result = self.tools_map[func_name](func_args)
                    except Exception as e:
                        result = f"工具执行报错: {str(e)}"
                else:
                    result = "工具未找到"

                # 将工具执行结果构造为 tool 消息格式，并加入历史
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                self.messages.append(tool_result_message)
                print(f"DEBUG: 🛠️ 工具返回结果: {result}")

            # 步数加1，继续下一次 while 循环，模型会读取刚才加入的 tool_result_message 决定下一步
            step += 1

        # 如果超出了最大步数
        error_msg = "抱歉，任务过于复杂，我未能得出最终结论。"
        self.messages.append({"role": "assistant", "content": error_msg})
        self.memory_manager.save(self.messages)
        return error_msg


    def _load_tools(self):
        """扫描 tools 文件夹并动态加载"""
        for _, name, _ in pkgutil.iter_modules(tools.__path__):
            # 动态导入模块
            module = importlib.import_module(f"tools.{name}")
            if hasattr(module, "SPEC") and hasattr(module, "run"):
                spec = module.SPEC
                func_name = spec["function"]["name"]

                self.tools_spec.append(spec)
                self.tools_map[func_name] = module.run
                print(f"DEBUG: 成功注入工具 -> {func_name}")

