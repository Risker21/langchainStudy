from pydantic import BaseModel, Field
from my_llm import deepseek_llm

class Actor(BaseModel):
    name: str = Field(description="人物姓名")
    role: str = Field(description="人物角色")

class Book(BaseModel):
    title: str = Field(description="书籍标题")
    summary: str = Field(description="书籍简介")
    author: str = Field(description="书籍作者")
    # 嵌套对象列表
    cast:list[Actor] = Field(description="书籍主要人物")

model_with_struct_output = deepseek_llm.with_structured_output(Book)
resp = model_with_struct_output.invoke("介绍一下《红楼梦》")
print(type(resp))
print(resp)



