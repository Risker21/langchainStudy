from langchain.agents import create_agent
from my_llm import dashscope_llm, zhipu_llm, hunyuan_llm
agent = create_agent(
    # model=dashscope_llm
    # model=zhipu_llm
    model = hunyuan_llm
)


response = agent.invoke({"messages":[{"role":"user","content":"介绍一下你自己"}]})
print(type(response))
print(response)

# print(response["messages"][-1].content)

