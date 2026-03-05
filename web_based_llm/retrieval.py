import os
import json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# API URL
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

class SemanticRetriever:
    def __init__(self, folders, max_chunks=50):
        self.folders = folders
        self.max_chunks = max_chunks
        self.chunks = []
        self.embeddings = None
        
        # Ensure token exists
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if not token:
            print("WARNING: HUGGINGFACE_TOKEN not found in environment variables.")
        
        self.headers = {
            "Authorization": f"Bearer {token}"
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
            
            # 1. Check for HTTP Errors
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return np.array([])
            
            data = response.json()
            
            # 2. Check if API returned a specific error dictionary
            if isinstance(data, dict) and "error" in data:
                print(f"API Error Message: {data['error']}")
                return np.array([])

            # 3. Check if data is actually a list (success)
            if not isinstance(data, list):
                print(f"Unexpected API response format: {type(data)}")
                return np.array([])
                
            return np.array(data)
            
        except Exception as e:
            print(f"Embedding API Connection Error: {e}")
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
            self.embeddings = self._query_api(self.chunks)
        else:
            self.embeddings = np.array([])

    def retrieve(self, query, top_n=4):
        # Safety checks
        if len(self.chunks) == 0:
            return []
        
        if self.embeddings.size == 0:
            # Try to load embeddings one last time if they failed during init
            print("Embeddings were empty, retrying...")
            self.embeddings = self._query_api(self.chunks)
            if self.embeddings.size == 0:
                return []

        # 1. Embed the query
        query_emb = self._query_api([query])
        
        # Safety check if query embedding failed
        if query_emb.size == 0:
            return []
            
        # Flatten query to (384,) if it's (1, 384)
        if query_emb.ndim == 2:
            query_emb = query_emb[0] 

        # 2. Calculate Similarity
        try:
            scores = np.dot(self.embeddings, query_emb)
        except Exception as e:
            print(f"Math Error (Shape Mismatch?): {e}")
            return []

        # 3. Sort and Retrieve
        top_k_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for idx in top_k_indices:
            results.append(self.chunks[idx])
            
        return results