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
        # 初始化时只加载历史，不重复添加 System Prompt
        self.history = self.memory_manager.load()
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        self.reflector = MemoryReflector(self.client)

        self.tools_map = {}
        self.tools_spec = []
        self._load_tools()

    def _load_tools(self):
        """动态加载工具"""
        for _, name, _ in pkgutil.iter_modules(tools.__path__):
            module = importlib.import_module(f"tools.{name}")
            if hasattr(module, "SPEC") and hasattr(module, "run"):
                spec = module.SPEC
                func_name = spec["function"]["name"]
                self.tools_spec.append(spec)
                self.tools_map[func_name] = module.run
                print(f"DEBUG: 成功注入工具 -> {func_name}")

    def _prepare_messages(self, user_input: str):
        """构建完整的上下文消息列表"""
        profile = self.reflector.load_profile(self.user_id) or "无"

        system_prompt = (
            "你是一个实战派助手。已知用户信息：{profile}。\n"
            "【核心规则】：\n"
            "1. 当用户询问天气、计算、搜索等问题时，必须【立即调用】工具，不要反问，不要确认。\n"
            "2. 只要手中工具能解决，就持续调用工具直到获得答案。\n"
            "3. 如果工具报错，请尝试分析原因并调整参数重试。"
        ).format(profile=profile)

        # 始终保持第一个是最新的 System Prompt
        messages = [{"role": "system", "content": system_prompt}]
        # 加上历史记录
        messages.extend(self.history)
        # 加上本次输入
        messages.append({"role": "user", "content": user_input})
        return messages

    def _parse_dsml_tags(self, content: str):
        """专门处理 DeepSeek 偶尔出现的 <｜DSML｜> 标签"""
        if not content or "<｜DSML｜invoke" not in content:
            return None

        name_match = re.search(r'name="([^"]+)"', content)
        # 这种解析较简单，主要针对你之前遇到的 search_web 场景
        query_match = re.search(r'>([^<]+)</｜DSML｜parameter>', content)

        if name_match:
            return [{
                "id": f"ds_{os.urandom(2).hex()}",
                "function": {
                    "name": name_match.group(1),
                    "arguments": json.dumps({"query": query_match.group(1).strip() if query_match else ""})
                }
            }]
        return None

    def chat(self, user_input: str) -> str:
        """
        主对话逻辑：标准的 ReAct 循环
        """
        messages = self._prepare_messages(user_input)

        max_steps = 1
        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self.tools_spec if self.tools_spec else None,
                tool_choice="auto"
            )

            msg = response.choices[0].message
            messages.append(msg)  # 存入当前会话上下文

            # 1. 提取工具调用（兼容标准格式和 DSML 标签）
            tool_calls = msg.tool_calls
            if not tool_calls:
                # 尝试解析 DSML 标签
                tool_calls = self._parse_dsml_tags(msg.content)

            # 2. 如果没有工具调用，说明推理结束
            if not tool_calls:
                # 同步到历史记录并持久化
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": msg.content})
                self.memory_manager.save(self.history)
                return msg.content

            # 3. 处理工具执行
            for tool_call in tool_calls:
                # 统一转为对象访问或字典访问（适配 DSML 解析出来的字典）
                if isinstance(tool_call, dict):
                    f_name = tool_call["function"]["name"]
                    f_args = json.loads(tool_call["function"]["arguments"])
                    f_id = tool_call["id"]
                else:
                    f_name = tool_call.function.name
                    f_args = json.loads(tool_call.function.arguments)
                    f_id = tool_call.id

                print(f"DEBUG: 正在执行工具 {f_name}, 参数: {f_args}")

                if f_name in self.tools_map:
                    try:
                        # 路线二优化：此处不解包 **f_args，而是统一传整个字典
                        # 这样在 tools/ 里定义 def run(arguments) 就能万能适配
                        result = self.tools_map[f_name](f_args)
                    except Exception as e:
                        result = f"工具执行报错: {str(e)}，请检查参数并重试。"
                else:
                    result = "错误：未找到该工具。"

                messages.append({
                    "role": "tool",
                    "tool_call_id": f_id,
                    "name": f_name,
                    "content": str(result)
                })

        return "抱歉，我经过多次尝试仍无法完成任务。"

