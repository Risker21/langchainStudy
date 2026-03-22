from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, prompt

from pydantic import BaseModel, Field

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_reasoner_llm = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# 1.定义结构
class Actor(BaseModel):
    name: str = Field(description="人物姓名")
    role: str = Field(description="人物角色")


class Book(BaseModel):
    title: str = Field(description="书籍标题")
    summary: str = Field(description="书籍简介")
    author: str = Field(description="书籍作者")
    # 嵌套对象列表
    cast: list[Actor] = Field(description="书籍主要人物")


# 2.设置提示词
prompt = ChatPromptTemplate.from_template(
    """
    请介绍一下书籍{title},
     请严格按照JSON格式返回结果，要求包含以下字段：
- title: 书籍标题
- summary: 书籍简介
- author: 书籍作者
- cast: 书籍主要人物
    其中cast是一个列表，列表中的每个元素包含以下字段：
- name: 人物姓名
- role: 人物角色
    """
)

# 3.设置链
chain = prompt | deepseek_reasoner_llm | JsonOutputParser(pydantic_object=Book)

# 4.执行链（返回字典）
resource = chain.invoke({"title": "介绍《红楼梦》"})
print(resource)
