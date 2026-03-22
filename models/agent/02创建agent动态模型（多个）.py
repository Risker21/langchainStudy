from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage

from my_llm import deepseek_llm, dashscope_llm, zhipu_llm, hunyuan_llm

deepseek_llm = deepseek_llm

dashscope_llm = dashscope_llm

zhipu_llm = zhipu_llm

hunyuan_llm = hunyuan_llm


@tool
def Book(book: str) -> str:
    """查找书籍工具,返回书籍的基本信息"""
    book_info = {
        "红楼梦": "《红楼梦》是中国古典四大名著之一，作者是曹雪芹。",
        "三国演义": "《三国演义》是中国古典四大名著之一，作者是罗贯中。",
        "水浒传": "《水浒传》是中国古典四大名著之一，作者是施耐庵。",
        "西游记": "《西游记》是中国古典四大名著之一，作者是吴承恩。"
    }
    return book_info.get(book, f"未找到{book}的信息")


@tool
def Book_info(book_Description: str) -> str:
    """书籍信息工具,返回书籍的详细信息"""
    book_details = {
        "红楼梦": "《红楼梦》是中国古典四大名著之一，作者是曹雪芹。小说以贾、史、王、薛四大家族的兴衰为背景，以贾宝玉、林黛玉、薛宝钗的爱情婚姻悲剧为主线，描绘了一批举止见识高于须眉之上的闺阁佳人的人生百态。",
        "三国演义": "《三国演义》是中国古典四大名著之一，作者是罗贯中。小说描写了从东汉末年到西晋初年之间近百年的历史风云，以描写战争为主，诉说了东汉末年的群雄割据混战和魏、蜀、吴三国之间的政治和军事斗争。",
        "水浒传": "《水浒传》是中国古典四大名著之一，作者是施耐庵。小说描写了北宋末年以宋江为首的108位好汉在梁山聚义，以及聚义之后接受招安、四处征战的故事。",
        "西游记": "《西游记》是中国古典四大名著之一，作者是吴承恩。小说描写了孙悟空、猪八戒、沙僧保护唐僧西天取经，历经九九八十一难，最终取得真经的故事。"
    }
    return book_details.get(book_Description, f"未找到{book_Description}的详细信息")


@tool
def Book_role(book_role: str) -> str:
    """书籍角色工具,返回书籍的角色详情"""
    role_info = {
        "红楼梦": "主要角色包括：贾宝玉（贾府公子）、林黛玉（贾母外孙女）、薛宝钗（薛家千金）、王熙凤（贾府管家）等。",
        "三国演义": "主要角色包括：刘备（蜀汉开国皇帝）、关羽（刘备义弟）、张飞（刘备义弟）、诸葛亮（蜀汉丞相）、曹操（魏国奠基者）、孙权（吴国开国皇帝）等。",
        "水浒传": "主要角色包括：宋江（梁山首领）、卢俊义（梁山第二首领）、吴用（梁山军师）、林冲（八十万禁军教头）、鲁智深（花和尚）、武松（行者）等。",
        "西游记": "主要角色包括：唐僧（取经和尚）、孙悟空（齐天大圣）、猪八戒（天蓬元帅）、沙僧（卷帘大将）等。"
    }
    return role_info.get(book_role, f"未找到{book_role}的角色信息")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """动态模型选择中间件"""
    message_count = len(request.state["messages"])
    if message_count < 2:
        model = dashscope_llm
    elif message_count < 3:
        model = deepseek_llm
    elif message_count < 4:
        model = hunyuan_llm
    else:
        model = zhipu_llm
    # 修改request对象的model属性
    return handler(request.override(model=model))


agent = create_agent(
    model=deepseek_llm,
    tools=[Book, Book_info, Book_role],
    middleware=[dynamic_model_selection]
)
# 3. 调用
response = agent.invoke({"messages": [{"role":"user","content":"查询红楼梦的信息"}]})
print(response)
