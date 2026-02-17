import subprocess
import platform

# Set Ollama path based on OS
if platform.system() == 'Darwin':  # macOS
    OLLAMA_PATH = '/usr/local/bin/ollama'
elif platform.system() == 'Windows':
    OLLAMA_PATH = r"C:\Users\12294\AppData\Local\Programs\Ollama\ollama.exe"
else:  # Linux
    OLLAMA_PATH = '/usr/local/bin/ollama'

MODEL_NAME = 'phi'

def query_llm(prompt: str) -> str:
    try:
        result = subprocess.run(
            [OLLAMA_PATH, 'run', MODEL_NAME],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10000
        )
        output = result.stdout.decode('utf-8')
        return output.strip()
    except subprocess.TimeoutExpired:
        return "The model took too long to respond."
    except Exception as e:
        return f"Error during model execution: {e}"
