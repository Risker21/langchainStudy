from idlelib.rpc import response_queue
from typing import TypedDict, Annotated

from my_llm import deepseek_llm


class Actor(TypedDict):
    name: Annotated[str,"人物姓名"]
    role: Annotated[str,"人物角色"]


class Book(TypedDict):
    title: Annotated[str,"书籍标题"]
    summary: Annotated[str,"书籍简介"]
    author: Annotated[str,"书籍作者"]
    # 嵌套对象列表
    cast:list[Annotated[Actor,"书籍主要人物"]]

model_with_struct_output = deepseek_llm.with_structured_output(Book)
resp = model_with_struct_output.invoke("介绍一下《红楼梦》")
print(type(resp))
print(resp)
