from agents.core import DeepSeekAgent

if __name__ == '__main__':

    agent = DeepSeekAgent()

    user_msg = "武汉明天天气？"
    answer = agent.chat(user_msg)
    print(answer)
