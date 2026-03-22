from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_response = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

response = deepseek_response.invoke("我有5个苹果，吃掉了2个，还剩几个？")
print(response.content_blocks)
# print(response.content)
