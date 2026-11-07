from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0,
    max_tokens=512,
)
