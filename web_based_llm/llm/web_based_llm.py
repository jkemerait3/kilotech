import os
from pathlib import Path

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load .env from the web app root explicitly.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Fallbacks if the configured model is unavailable or returns empty output.
FALLBACK_MODELS = [
    MODEL_NAME,
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


def _extract_message_text(response) -> str:
    """Extract assistant text robustly across HF response variants."""
    if response is None:
        return ""

    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    # Some providers return content as structured parts.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        content = "\n".join([p for p in parts if p])

    if content is None:
        content = ""

    return str(content).strip()


def _query_single_model(prompt: str, token: str, model_name: str) -> str:
    client = InferenceClient(model=model_name, token=token)
    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=1100,
        temperature=0.35,
    )
    return _extract_message_text(response)

def query_llm(prompt: str) -> str:
    """
    Queries the Hugging Face Inference API.
    Requires HUGGINGFACE_TOKEN (or HF_TOKEN) to be set in environment variables.
    """
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    
    if not token:
        return "Error: HUGGINGFACE_TOKEN (or HF_TOKEN) not found in environment variables."

    last_error = None

    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            text = _query_single_model(prompt, token, model_name)
            if text:
                return text
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        return f"Error during Hugging Face API execution: {last_error}"

    return "Error: LLM returned an empty response. Try changing HF_MODEL_NAME in .env."