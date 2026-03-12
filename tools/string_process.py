def get_weather(city: str) -> str:
    """获取指定城市的实时天气情况。参数 city 是城市名称。"""
    return f"{city}今天晴转多云，25度。"

def string_length(s: str) -> int:
    """计算输入字符串的字符个数。参数 s 是目标字符串。"""
    return len(s)

# 把函数放到列表里
my_tools = [get_weather, string_length]


def generate_instruction(tools):
    instruction = "你是一个 AI Agent，你可以使用以下工具：\n"
    for func in tools:
        # 提取函数名和文档字符串
        name = func.__name__
        description = func.__doc__
        instruction += f"- {name}: {description}\n"

        instruction += "\n如果需要使用工具，请回复 JSON：{'action': '工具名', 'action_input': '参数值'}"
    return instruction