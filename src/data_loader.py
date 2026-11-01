import json
import logging
import pandas as pd
from typing import List, Dict, Tuple
from src.config import AMIDHA_DATA_PATH

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BioVedaDataLoader:
    def __init__(self, file_path: str = AMIDHA_DATA_PATH):
        self.file_path = file_path
        self.raw_data = None
        self.plants_df = None
        
    def load_data(self) -> pd.DataFrame:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            self.plants_df = pd.DataFrame(self.raw_data)
            logging.info(f"Successfully loaded {len(self.plants_df)} plants from database.")
            return self.plants_df
        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.file_path}. Please check the path.")
            raise
            
    def extract_vocabulary(self, target_column: str = 'Rogagnata') -> List[str]:
        #Extracting symptom/disease fo creating embedding search space
        if self.plants_df is None:
            self.load_data()
            
        unique_indications = set()
        if target_column in self.plants_df.columns:
            for items in self.plants_df[target_column].dropna():
                if isinstance(items, str):
                    # Split by comma and strip whitespace
                    cleaned_items = [i.strip().lower() for i in items.split(',')]
                    unique_indications.update(cleaned_items)
                elif isinstance(items, list):
                    cleaned_items = [str(i).strip().lower() for i in items]
                    unique_indications.update(cleaned_items)          
        vocab = sorted(list(unique_indications))
        logging.info(f"Extracted {len(vocab)} unique therapeutic indications.")
        return vocab

if __name__ == "__main__":
    loader = BioVedaDataLoader()
    df = loader.load_data()