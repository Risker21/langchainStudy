"""调用模型配合参数"""
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# 自定义回调函数
class MyCustomCallbackHandler(BaseCallbackHandler):
    # 模型调用开始时触发
    def on_llm_start(self, serialized, prompts, **kwargs):
        # **kwargs 中的参数如下
        print("kwargs:", kwargs)

        # 从 kwargs 中提取配置信息
        run_name = kwargs.get("name")  # 对应 config 中的 'run_name'
        tags = kwargs.get("tags", [])
        metadata = kwargs.get("metadata", {})
        run_id = kwargs.get("run_id")

        print(f"[LLM开始] 运行名称: {run_name}")
        print(f"        标签: {tags}")
        print(f"        元数据: {metadata}")
        print(f"        运行ID: {run_id}")

        # 可以在这里执行更复杂的操作，比如：
        # 1. 将信息记录到日志系统或数据库
        # 2. 根据 metadata 中的 user_id 进行用户行为分析
        # 3. 根据 tags 对运行进行分类和监控
        # 4. 触发自定义事件，比如发送通知到监控系统

    def on_llm_end(self, response, **kwargs):
        # **kwargs 中的参数如下
        print("kwargs:", kwargs)

        # 提取运行结束时的信息
        run_id = kwargs.get("run_id")
        print(f"[LLM结束] 运行ID: {run_id}")
        # 可以在这里记录运行结束的信息，比如消耗的令牌数等
        # 从 response 中提取模型消耗令牌数
        print("response:", response)

        # 消耗的令牌数
        num_tokens = response.llm_output["token_usage"]["total_tokens"]
        print(f"        消耗的令牌数: {num_tokens}")
# 实例化自定义回调函数
custom_handler = MyCustomCallbackHandler()

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    # 指定可调整参数，用户可以在调用时覆盖这些参数的默认值
    configurable_fields=("model", "model_provider", "temperature", "max_tokens")
)
# 配置模型参数
config = {
    "run_name": "joke_generator",  # 运行名称，用于标识调用记录
    "tags": ["tag1", "tag2"],  # 标签，用于分类和组织调用记录
    # 其他自定义字段
    "metadata": {
        "user": "Momo同学",  # 用户信息，用于记录调用者
        "description": "这是一个生成情诗的模型调用",  # 模型描述，用于记录调用的场景
        "version": "1.0"  # 模型版本，用于记录模型的版本号
    },
    "callbacks": [custom_handler],  # 启用自定义回调函数
    "configurable": {
        "model": "deepseek-reasoner",  # 切换到另一个模型
        "temperature": 0.7,  # 温度参数
        "max_tokens": 1024  # 最大输出长度
    }
}
response = deepseek_llm.invoke("写一封情诗", config=config)

print(response)
# print(response.content)
