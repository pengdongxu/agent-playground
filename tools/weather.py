import os

from openai import OpenAI, base_url


def get_weather(city: str) -> str:
    """获取指定城市的实时天气情况。参数 city 是城市名称。"""
    return f"{city}今天晴转多云，25度。"

get_weather_spec = {
    "name": "get_weather",
    "description": "获取指定城市的实时天气情况",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    }
}