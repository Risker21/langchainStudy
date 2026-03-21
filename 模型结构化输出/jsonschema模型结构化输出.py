from my_llm import deepseek_llm

json_schema = {
    "title": "BookInfo",
    "description": "书籍信息",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "书籍标题"},
        "summary": {"type": "string", "description": "书籍简介"},
        "author": {"type": "string", "description": "书籍作者"},
        # 嵌套对象列表
        "cast": {
            "type": "array",
            "description": "书籍主要人物",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "人物姓名"},
                    "role": {"type": "string", "description": "人物角色"}
                },
                "required": ["name", "role"]
            }
        }
    },
    "required": ["title", "summary", "author", "cast"],
}

model_with_struct_output = deepseek_llm.with_structured_output(json_schema)
resp = model_with_struct_output.invoke("介绍一下《红楼梦》")
print(type(resp))
print(resp)

import jsonschema

try:
    error = jsonschema.validate(instance=resp, schema=json_schema)
    print("验证通过")
except jsonschema.exceptions.ValidationError as e:
    print("验证失败", e)
