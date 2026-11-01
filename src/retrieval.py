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
        
        self._load_assets()
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
        #Embeds query and return top results
        logging.info(f"\n--- Processing Query: '{user_text}' ---")
        
        query_embedding = self.model.encode([user_text])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, self.top_k)
        results = []
        for i in range(self.top_k):
            idx = indices[0][i]
            dist = distances[0][i]
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