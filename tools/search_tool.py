# tools/search_tool.py
import json
from ddgs import DDGS

# 1. 说明书 (保持不变，但确保 name 叫 search_web 或跟 core 逻辑一致)
SPEC = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "当用户询问实时新闻、百科知识、产品评价或需要联网获取最新信息时使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }
}


# 2. 内部逻辑函数
def _do_search(query: str, max_results: int = 5):
    """
    执行 DuckDuckGo 搜索逻辑
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r["title"],
                    "link": r["href"],
                    "snippet": r["body"]
                })
        return results
    except Exception as e:
        return f"搜索过程中出现错误: {str(e)}"


# 3. 动态加载器统一调用的接口函数
def run(arguments: str):
    """
    动态注入规范：接收字典参数，返回字符串结果
    """
    query = arguments.get("query")
    if not query:
        return "未提供有效的搜索关键词"
    # 执行搜索
    raw_results = _do_search(query)
    print("search results:", raw_results)
    # 数据清洗：如果结果太长，LLM 的 Token 会爆炸
    # 建议只取前 3 条核心信息，并转为 JSON 字符串返回
    if isinstance(raw_results, list):
        return json.dumps(raw_results[:3], ensure_ascii=False)

    return str(raw_results)