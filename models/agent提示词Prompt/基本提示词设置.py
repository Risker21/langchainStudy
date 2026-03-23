from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from my_llm import deepseek_llm

agent = create_agent(
    model=deepseek_llm,
    # 系统提示词(两种方式)
    # 方式一
    # system_prompt="你是CC同学,你可以回答用户的问题",
    system_prompt=SystemMessage(content="你是CC同学,你是一个情感疗愈机器人,你可以帮助用户处理情感问题")
)

# 初始化消息列表，包含系统消息
messages = [
    {"role": "system", "content": "你是一个情感疗愈机器人"}
]

# 持续对话循环
while True:
    # 获取用户输入
    user_input = input("用户: ")
    
    # 检查是否退出
    if user_input == "再见":
        print("CC同学: 再见！祝你有美好的一天！")
        break
    
    # 添加用户消息到列表
    messages.append({"role": "user", "content": user_input})
    
    # 调用agent处理消息
    resp = agent.invoke({"messages": messages})
    
    # 获取并打印回复
    response_content = resp["messages"][-1].content
    print(f"CC同学: {response_content}")
    
    # 将agent的回复添加到消息列表，保持上下文
    messages.append({"role": "assistant", "content": response_content})