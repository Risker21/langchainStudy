import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.tools import tool
from openai.types import ModelDeleted

from my_llm import deepseek_llm


@tool
def query_order_info(order_id: str):
    """根据订单ID查询订单的详细信息，包括状态、商品列表和创建时间。"""
    #     模拟数据库查询结果
    order_database = {
        "OR01": {
            "status": "已支付",
            "items": [{"name": "商品A", "price": 100}, {"name": "商品B", "price": 200}],
            "created_at": "2023-08-01 10:00:00"
        },
        "OR02": {
            "status": "已取消",
            "items": [{"name": "商品C", "price": 300}],
            "created_at": "2023-08-02 15:00:00"
        },
        "OR03": {
            "status": "已发货",
            "items": [{"name": "商品D", "price": 400}],
            "created_at": "2023-08-03 12:00:00"
        },
    }
    order_data = order_database.get(order_id)
    if order_data:
        return json.dumps(order_data, ensure_ascii=False)
    else:
        return f"订单ID为{order_id}的订单不存在"


@tool
#     模拟检索常见问题解答
def search_faq(keyword: str) -> str:
    """根据关键词检索常见问题解答。"""
    #     模拟常见问题数据库
    faq_database = {
        "订单状态": "订单状态查询",
        "订单取消": "订单取消查询",
        "订单发货": "订单发货查询",
        "订单退货": "支持7天无理由退货，商品需完好无损",
        "保修": "电子产品保修，保修期1年",
        "发货": "发货时间：订单支付后24小时内发货"
    }
    # 简单关键字匹配
    for topic, answer in faq_database.items():
        if topic in keyword:
            return answer
    return f"未找到与{keyword}相关的问题"


class AgentContext(TypedDict):
    query_type: str
    uid: str


@dynamic_prompt
def dynamic_support_prompt(request: ModelRequest) -> str:
    """
    根据 query_type 生成不同的系统提示词。
    """
    # print("request:", request)
    query_type = request.runtime.context.get("query_type", "vip")
    base_instruction = "你是一名专业的电商客服助手。请根据工具查询结果，准确、清晰地回答用户问题。"

    if query_type == "vip":
        # 针对复杂或需要升级处理的问题
        return f"""{base_instruction}
    当前角色：高级支持专员
    工作要求：
    1.深度分析：仔细分析用户描述，识别潜在的根本问题。
    2.精准分类：将问题明确归类（如“物流问题”、“产品质量”、“售后申请”）。
    3.方案规划：若工具能解决，提供具体步骤；若需人工，明确告知后续流程。
    请使用更专业、严谨的语言。
    """
    else:
        # 针对常规客服问题
        return f"""{base_instruction}
    当前角色：一线客服助手
    工作要求：
    1.快速响应：优先使用工具获取订单/政策信息。
    2.直接解答：对于明确问题（如退货、物流），直接给出基于知识的答案。
    3.简洁友好：回复要简单明了，避免复杂术语。
    保持友好和高效的沟通风格。
    """
agent =  create_agent(
    model=deepseek_llm,
    tools=[query_order_info, search_faq],
    middleware=[dynamic_support_prompt],
    context_schema= AgentContext,
)

# normal
print("====== normal=======")
resp = agent.invoke({"messages": [{"role": "user", "content": "订单OR01状态查询"}]},
             context={"query_type":"normal"})
print(resp["messages"][-1].content)

# vip
print("====== vip=======")
resp = agent.invoke({"messages": [{"role": "user", "content": "订单OR01状态查询"}]},
             context={"query_type":"vip"})
print(resp["messages"][-1].content)
