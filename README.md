# How to run locally
1. Install ollama from https://ollama.com/
2. In CLI, run "ollama pull mistral" and "ollama serve" // Alternatively, pull your model of choice and change the value of MODEL_NAME to its name on line 4 of kilotech/llm/local_llm.py
3. Navigate to the project directory and run the project in CLI with "python main.py"

# Note
To change between local and web-based LLM access in Django, simply import the "query_llm" method from the desired file in web_based_llm\advisor\views.py
