import requests
import json
from enum import Enum
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"Using OpenRouter API key: {OPENROUTER_API_KEY}")
class ModelEndpoint(Enum):
    GPT_4_1_MINI = "openai/gpt-4.1-mini"
    CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GROK_3_MINI = "x-ai/grok-3-mini"
    LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
    DEEPSEEK_CHAT_V3 = "deepseek/deepseek-chat-v3-0324"
    MISTRAL_SMALL_3_2_24B = "mistralai/mistral-small-3.2-24b-instruct"

def generate(model_url, message):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": model_url,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message
                        }
                    ]
                }
            ],
        })
    )

    return response.json()["choices"][0]["message"]["content"]

models = [
    ModelEndpoint.GPT_4_1_MINI,
    ModelEndpoint.CLAUDE_3_HAIKU,
    ModelEndpoint.GEMINI_2_5_FLASH,
    ModelEndpoint.GROK_3_MINI,
    ModelEndpoint.LLAMA_4_MAVERICK,
    ModelEndpoint.DEEPSEEK_CHAT_V3,
    ModelEndpoint.MISTRAL_SMALL_3_2_24B
]