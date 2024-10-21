import os
from typing import Optional
import dotenv

from langchain_openai import ChatOpenAI

dotenv.load_dotenv('local.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_MODEL = 'meta-llama/llama-3.2-3b-instruct:free'

TINYBOX_API_KEY = 'dummy'
TINYBOX_TAILSCALE_URL = os.getenv('TINYBOX_TAILSCALE_URL')
TINYBOX_LOCAL_URL = os.getenv('TINYBOX_LOCAL_URL')
TINYBOX_MODEL = 'dummy'

DEFAULT_PROVIDER = 'openrouter'
DEFAULT_MODEL = OPENROUTER_MODEL
DEFAULT_API_KEY = OPENROUTER_API_KEY
DEFAULT_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_TEMPERATURE = 0.0


def get_chat_openai(
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    if provider == 'openrouter':
        model = model if model is not None else OPENROUTER_MODEL
        api_key = api_key if api_key is not None else OPENROUTER_API_KEY
        base_url = base_url if base_url is not None else OPENROUTER_BASE_URL
    elif provider == 'tinybox':
        model = model if model is not None else TINYBOX_MODEL
        api_key = api_key if api_key is not None else TINYBOX_API_KEY
        base_url = base_url if base_url is not None else TINYBOX_TAILSCALE_URL
    else:
        raise ValueError(f'Invalid provider: {provider}')

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
        streaming=True
    )