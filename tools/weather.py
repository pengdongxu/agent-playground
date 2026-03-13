import requests


def get_weather(city: str) -> str:
    search_url ="https://api.asilu.com/weather/?city=%s"
    weather_data = requests.get(search_url % city).json()
    """获取指定城市的实时天气情况。参数 city 是城市名称。"""

    # 提取基本信息
    clean_data = {
        "city": weather_data["city"],
        "update_time": weather_data["update_time"],
        "forecast": []
    }

    # 遍历weather列表，将数据保存在clean_data中
    for day in weather_data["weather"]:
        day_data = {
            "date": day["date"],
            "weather": day["weather"],
            "temp": day["temp"],
            "wind": day["wind"]
        }
        clean_data["forecast"].append(day_data)

    return clean_data

get_weather_spec = {
    "type": "function",  # 1. 明确类型
    "function": {        # 2. 详细信息全部包装在 function 键里
        "name": "get_weather",
        "description": "获取指定城市的实时天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",           # 需要的字段类型
                    "description": "城市名称，例如：北京市, 上海市, 武汉市"
                }
            },
            "required": ["city"]        # 这里列出的，LLM 才会强制寻找
        }
    }
}