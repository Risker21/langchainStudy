from http.client import responses

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from my_llm import deepseek_llm, dashscope_llm

# print(deepseek_llm.invoke("介绍一下你自己"))
# print(dashscope_llm.invoke("介绍一下你自己"))

# 1.单条消息调用模型
# resp = deepseek_llm.invoke("介绍一下你自己")
# print(type(resp))
# print(resp.content)

# 2.字典格式的消息列表
# conversions = [
#     {"role": "system", "content": "你是一个专业的语言模型助手。"},
#     {"role": "user", "content": "介绍一下你自己"},
#     {"role": "assistant", "content": "你好，我是Momo同学，你的智能助手。"},
#     {"role": "user", "content": "你能做什么？"}]
# resp = deepseek_llm.invoke(conversions)
# print(type(resp))
# print(resp.content)


# 3.消息对象格式的消息列表
# conversions = [
#     SystemMessage(content="你是一个专业的语言模型助手。"),
#     HumanMessage(content="介绍一下你自己"),
#     AIMessage(content="你好，我是Momo同学，你的智能助手。"),
#     HumanMessage(content="你能做什么？")
# ]
# resp = deepseek_llm.invoke(conversions)
# print(type(resp))
# print(resp)
# print(resp.content)

# 4.流式输出
# response = deepseek_llm.stream("介绍一下你自己")
# for chunk in response:
#     print(chunk.content, end="", flush=True)

# 5.批量调用模型
# resp = deepseek_llm.batch(
#     ["介绍一下你自己",
#      "大数据工程技术专业的就业前景如何？",
#      "请帮我写一首关于夏天的诗歌"]
# )
# for item in resp:
#     print(item.content)

# 6.批量调用模型，按完成顺序返回结果
resp = deepseek_llm.batch_as_completed(
    ["介绍一下你自己",
     "大数据工程技术专业的就业前景如何？",
     "请帮我写一首关于夏天的诗歌"],
    config={"max_concurrency": 3,}  # 设置最大并发数为3
)

for item in resp:
    # 元组
    print(item)
