import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from agents.core import DeepSeekAgent

app = FastAPI (title="DeepSeek API", description="DeepSeek API", version="0.1.0")

# 模拟一个内存中的Agent仓库，key是user_id
# 这样支持多用户同时在线，每个用户都有自己的记忆文件
agents_pool = {}

class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str

class ChaetResponse(BaseModel):
    user_id: str
    reply: str


def get_or_create_agent(user_id: str):
    if user_id not in agents_pool:
        agents_pool[user_id] = DeepSeekAgent(user_id = user_id)
    return agents_pool[user_id]


@app.post("/chat")
async def chat(request: ChatRequest, bt: BackgroundTasks):
    try:
        agent = get_or_create_agent(request.user_id)
        answer = agent.chat(request.message)

        bt.add_task(agent.reflector.reflect, request.user_id, agent.messages)
        print(f"DEBUG: 回复：{answer}")
        return ChaetResponse(
            user_id=request.user_id,
            reply=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动服务器，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
