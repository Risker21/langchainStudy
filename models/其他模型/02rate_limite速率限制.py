import time

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from env_utils import DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 每10秒允许1次请求
    check_every_n_seconds=0.1,  # 每0.1秒检查一次速率限制
    max_bucket_size=1,  # 应对流量高峰最多允许的请求个数
)
dashscope_llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    rate_limiter=rate_limiter,  # 1次/秒
)
for i in range(3):
    response = dashscope_llm.invoke("介绍一下千问模型的速率限制功能。")
    print(response.content)
    print(time.time())
