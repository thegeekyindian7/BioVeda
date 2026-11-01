import os
import logging
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, INDEX_DIR
from src.data_loader import BioVedaDataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BioVedaIndexer:
    def __init__(self):
        self.loader = BioVedaDataLoader()
        self.df = self.loader.load_data()
        logging.info(f"Loading transformer model: {EMBEDDING_MODEL_NAME}. This may take a moment on the first run...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    def _create_hybrid_documents(self) -> pd.DataFrame:
        #Combining Sanskrit indications with English names-> creates a context rich doc for each plant
        documents = []
        for _, row in self.df.iterrows():
            # handle missing lists or text
            eng_name = str(row.get('english_name', ''))
            preview = str(row.get('preview', ''))
            indications = row.get('main_indications', [])
            if isinstance(indications, list):
                indications_str = ", ".join(indications)
            else:
                indications_str = str(indications)
            actions = row.get('prabhav', [])
            if isinstance(actions, list):
                actions_str = ", ".join(actions)
            else:
                actions_str = str(actions)
            # Constructing the hybrid text doc
            doc = f"Name: {eng_name}. Description: {preview}. Indications: {indications_str}. Actions: {actions_str}."
            documents.append(doc)
        self.df['search_document'] = documents
        return self.df

    def build_and_save_index(self):
        # Creating indexing and FAISS index
        self._create_hybrid_documents()
        documents = self.df['search_document'].tolist()
        logging.info(f"Encoding {len(documents)} plant documents into vectors...")
        # Generate vector embeddings
        embeddings = self.model.encode(documents, show_progress_bar=True)
        # float32 for FAISS compatibility
        embeddings = np.array(embeddings).astype('float32')
        # Embedding Dimension
        dim = embeddings.shape[1]
        # FAISS Index 
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss_path = INDEX_DIR / "bio_veda_plants.index"
        faiss.write_index(index, str(faiss_path))
        df_path = INDEX_DIR / "plants_metadata.pkl"
        self.df.to_pickle(str(df_path))        
        logging.info(f"Successfully saved FAISS index with {index.ntotal} vectors to {faiss_path}")
        logging.info(f"Successfully saved plant metadata to {df_path}")

if __name__ == "__main__":
    indexer = BioVedaIndexer()
    indexer.build_and_save_index()