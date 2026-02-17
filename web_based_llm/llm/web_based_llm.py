import os
from huggingface_hub import InferenceClient

# Choose a powerful instruction-tuned model
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def query_llm(prompt: str) -> str:
    """
    Queries the Hugging Face Inference API.
    Requires 'HUGGINGFACE_TOKEN' to be set in environment variables.
    """
    token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if not token:
        return "Error: HUGGINGFACE_TOKEN not found in environment variables."

    try:
        # Initialize the client
        client = InferenceClient(model=MODEL_NAME, token=token)
        
        # Request a chat completion
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error during Hugging Face API execution: {e}"