from langchain.agents import create_agent
from langchain_core.tools import tool

from my_llm import deepseek_llm


@tool
def get_Airfare(origin: str, destination: str) -> str:
    """获取从origin到destination的机票价格"""
    return f"从{origin}到{destination}的机票价格为100元"


agent = create_agent(
    model=deepseek_llm,
    tools=[get_Airfare],
    system_prompt="你是一个旅游助手，帮助用户查询机票价格。"
)
# 调用agent
response = agent.invoke({"messages":[{"role":"user","content":"北京到上海的机票价格"}]})
print(response)
