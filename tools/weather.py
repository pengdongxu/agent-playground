import os

import requests
from openai import OpenAI, base_url


def get_weather(city: str) -> str:
    search_url ="https://api.asilu.com/weather/?city=%s"
    weather_data = requests.get(search_url % city).json()
    """获取指定城市的实时天气情况。参数 city 是城市名称。"""
    return weather_data

get_weather_spec = {
    "type": "function",  # 1. 明确类型
    "function": {        # 2. 详细信息全部包装在 function 键里
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
}