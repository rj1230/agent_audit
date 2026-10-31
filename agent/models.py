from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="qwen/qwen3.8-27b",
    model_provider="groq",
)
