import os
import json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# We use the API URL for the model instead of loading it locally
# This is the same model as previously, just hosted remotely.
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

class SemanticRetriever:
    def __init__(self, folders, max_chunks=50):
        self.folders = folders
        self.max_chunks = max_chunks
        self.chunks = []
        self.embeddings = None
        
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN')}"
        }
        
        self._load_and_embed()

    def _query_api(self, texts):
        """
        Sends text to Hugging Face to get embeddings.
        Returns a numpy array of shape (n_texts, 384).
        """
        try:
            response = requests.post(
                API_URL, 
                headers=self.headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            return np.array(response.json())
        except Exception as e:
            print(f"Embedding API Error: {e}")
            return np.array([])

    def _load_and_embed(self):
        all_chunks = []
        
        # 1. Load Text from Files
        for folder in self.folders:
            for root, _, files in os.walk(folder):
                for filename in sorted(files):
                    if filename.endswith('.jsonl'):
                        path = os.path.join(root, filename)
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if len(all_chunks) >= self.max_chunks:
                                        break
                                        
                                    try:
                                        obj = json.loads(line)
                                        text = obj.get("text") or obj.get("body") or ""
                                        if text.strip():
                                            all_chunks.append(text.strip())
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            continue
                            
                    if len(all_chunks) >= self.max_chunks:
                        break
            if len(all_chunks) >= self.max_chunks:
                break

        self.chunks = all_chunks

        # 2. Get Embeddings from API
        if self.chunks:
            # The API is efficient, but let's batch if > 50 to be safe
            self.embeddings = self._query_api(self.chunks)
        else:
            self.embeddings = np.array([])

    def retrieve(self, query, top_n=4):
        if len(self.chunks) == 0 or self.embeddings.size == 0:
            return []

        # 1. Embed the query using the same API
        query_emb = self._query_api([query])
        
        # Safety check if API failed
        if query_emb.size == 0:
            return []
            
        # Flatten query to (384,)
        query_emb = query_emb[0] 

        # 2. Calculate Similarity (Dot Product)
        # Ensure embeddings are a numpy array of floats
        try:
            scores = np.dot(self.embeddings, query_emb)
        except Exception as e:
            print(f"Math Error: {e}")
            return []

        # 3. Sort and Retrieve
        top_k_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for idx in top_k_indices:
            results.append(self.chunks[idx])
            
        return results