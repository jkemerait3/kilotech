import os
import json
import ast
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from llm.local_llm import query_llm
from utils import load_inventory, administer_inventory, generate_output_filename

# ==== RETRIEVAL/CONTEXT CONTROL VARIABLES ====
# --------------------------------------------------------------
# >>> TUNE THESE FOR EACH LLM TESTED <<<
MAX_TOTAL_CHARS_CONTEXT = 1800   # per context injection
RETRIEVAL_TOP_N = 6             # How many top relevant chunks to inject each time
RETRIEVER_MODEL = "all-MiniLM-L6-v2"   # can upgrade for larger LLMs
RETRIEVER_MAX_CHUNKS = 600      # higher for lots of data
# --------------------------------------------------------------


class SemanticRetriever:
    def __init__(self, folders, max_chunks=600, embed_model=RETRIEVER_MODEL):
        self.folders = folders
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.chunk_sources = []
        self.embeddings = None
        self._index_chunks(max_chunks)

    def _index_chunks(self, max_chunks):
        all_chunks = []
        sources = []
        for folder in self.folders:
            # Walk through all subdirectories recursively
            for root, dirs, files in os.walk(folder):
                for filename in sorted(files):
                    if filename.endswith('.jsonl'):
                        path = os.path.join(root, filename)
                        with open(path, 'r', encoding='utf-8') as f:
                            for ix, line in enumerate(f):
                                try:
                                    obj = json.loads(line)
                                    text = obj.get("text", "")
                                    if text and isinstance(text, str):
                                        all_chunks.append(text.strip())
                                        sources.append((filename, ix))
                                    if len(all_chunks) >= max_chunks:
                                        break
                                except Exception:
                                    continue
                    if len(all_chunks) >= max_chunks:
                        break
                if len(all_chunks) >= max_chunks:
                    break
            if len(all_chunks) >= max_chunks:
                break
        self.chunks = all_chunks
        self.chunk_sources = sources
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)

    def retrieve(self, query, top_n=RETRIEVAL_TOP_N, max_total_chars=MAX_TOTAL_CHARS_CONTEXT):
        q_emb = self.model.encode([query])[0]
        sims = np.inner(self.embeddings, q_emb)
        top_ids = np.argsort(sims)[::-1][:top_n]
        results = []
        chars = 0
        for idx in top_ids:
            chunk = self.chunks[idx]
            if chars + len(chunk) > max_total_chars:
                break
            results.append(chunk)
            chars += len(chunk)
        return results

# ==== FILE UTILS ====
def list_json_files(folder_path, exclude=None):
    if not os.path.exists(folder_path):
        return []
    docs = sorted(f for f in os.listdir(folder_path) if f.endswith('.json'))
    if exclude:
        docs = [f for f in docs if f not in exclude]
    return docs

def get_user_query():
    query = input("Welcome to KiloTech. Please enter your question: ")
    return query

# Uses the Free Weather API to get current weather based on a list of coordinates
def get_current_weather(coordinates: list[tuple[float, float]], location_names: list[str]) -> str:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [coord[0] for coord in coordinates],
        "longitude": [coord[1] for coord in coordinates],
        "models": "gfs_seamless",
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m"],
    }
    responses = openmeteo.weather_api(url, params=params)

    weather_summary = ""
    for i, response in enumerate(responses):
        location = location_names[i]
        # Process current data. The order of variables needs to be the same as requested.
        current = response.Current()
        current_temperature_2m = current.Variables(0).Value()
        current_relative_humidity_2m = current.Variables(1).Value()
        current_precipitation = current.Variables(2).Value()
        current_weather_code = current.Variables(3).Value()
        current_wind_speed_10m = current.Variables(4).Value()

        weather_summary += (f"At {location}, the current temperature is {current_temperature_2m}°C, "
                            f"humidity is {current_relative_humidity_2m}%, "
                            f"precipitation is {current_precipitation}mm, "
                            f"weather code is {current_weather_code}, "
                            f"and wind speed is {current_wind_speed_10m}km/h.\n")
    
    return weather_summary.strip()

# ==== MAIN INTERACTIVE FLOW ====
def run_cli():
    # ---- Set up retriever at session start ----
    Hawaiian_Literature_Folder = 'data/hawaiian_chunks'
    retriever = SemanticRetriever(
        [Hawaiian_Literature_Folder],
        max_chunks=RETRIEVER_MAX_CHUNKS
    )

    # ---- original user query ----
    user_query = get_user_query()

    # ---- Retrieve context for summary ----
    summary_context = retriever.retrieve(
        user_query,
        top_n=RETRIEVAL_TOP_N,
        max_total_chars=MAX_TOTAL_CHARS_CONTEXT
    )
    summary_context_text = "\n\n".join(summary_context)

    summary_prompt = (
        "You are an expert in Hawaiian culture and agriculture assisting farmers. "
        "Based on the literature context, current weather, and the user's query, "
        "write a culturally informed, actionable response to their query. "
        "Avoid naming specific sources.\n\n"
        "--- Literature Context ---\n"
        f"{summary_context_text}\n\n"
        "--- Current Weather ---\n"
        f"{get_current_weather([(33.775677, -84.388098)], ["Oahu"])}\n\n"
        "--- User Query ---\n"
        f"{user_query}\n\n"
    )
    answer = query_llm(summary_prompt)
    print(answer)
    
if __name__ == "__main__":
    #run_cli()
    print(get_current_weather([(33.775677, -84.388098)], ["Coda"]))
