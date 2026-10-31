import logging
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, INDEX_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')

class BioVedaRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.index_path = INDEX_DIR / "bio_veda_plants.index"
        self.meta_path = INDEX_DIR / "plants_metadata.pkl"
        
        # Load the index and metadata
        self._load_assets()
        
        # Load the embedding model
        logging.info("Loading embedding model for query inference...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    def _load_assets(self):
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.df_meta = pd.read_pickle(str(self.meta_path))
            logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"Failed to load index or metadata: {e}")
            raise

    def query(self, user_text: str):
        """
        Embeds the user query and retrieves the Top-K matching plants.
        """
        logging.info(f"\n--- Processing Query: '{user_text}' ---")
        
        # 1. Embed the query text
        query_embedding = self.model.encode([user_text])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # 2. Search FAISS index (returns distances and indices)
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # 3. Format the results
        results = []
        for i in range(self.top_k):
            idx = indices[0][i]
            dist = distances[0][i]  # L2 distance (lower is better/closer)
            plant_data = self.df_meta.iloc[idx]
            
            results.append({
                "rank": i + 1,
                "english_name": plant_data.get('english_name', 'Unknown'),
                "botanical_name": plant_data.get('botanical_name', 'Unknown'),
                "distance_score": round(float(dist), 2),
                "indications": plant_data.get('main_indications', []),
                "preview": plant_data.get('preview', '')
            })
            
        return results

if __name__ == "__main__":
    retriever = BioVedaRetriever(top_k=3)
    
    # Let's test three distinctly different medical queries
    test_queries = [
        "I have a terrible headache and my head is pounding.",
        "I have been experiencing acid reflux and heartburn after meals.",
        "I can't sleep at night and feel very anxious."
    ]
    
    for q in test_queries:
        results = retriever.query(q)
        for res in results:
            print(f"Rank {res['rank']} | Plant: {res['english_name']} ({res['botanical_name']})")
            print(f"   Score: {res['distance_score']}")
            print(f"   Indications: {res['indications']}")
            print(f"   Why: {res['preview'][:100]}...\n")