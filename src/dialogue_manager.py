import logging
import pandas as pd
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')

class BioVedaDialogueManager:
    def __init__(self, metadata_path: str):
        self.df_meta = pd.read_pickle(metadata_path)        
        self.property_questions = {
            "Ushna": {
                "main_question": "Do you also experience any of the following? (Select all that apply)",
                "options": ["Excessive body heat", "Severe burning sensations", "Excessive sweating", "None of these"],
                "risk_type": "hot_potency"
            },
            "Sheeta": {
                "main_question": "Do you also experience any of the following? (Select all that apply)",
                "options": ["Excessive coldness", "Poor circulation", "Physical sluggishness or heaviness", "None of these"],
                "risk_type": "cold_potency"
            }
        }

    def generate_clarification_question(self, retrieved_results: List[Dict]) -> Dict:
        ushna_count = 0
        sheeta_count = 0
        # Majority Rule on Top 3
        for res in retrieved_results[:3]:
            plant_data = self.df_meta[self.df_meta['botanical_name'] == res['botanical_name']]
            if not plant_data.empty:
                virya = str(plant_data.iloc[0].get('virya', ''))
                if 'Ushna' in virya:
                    ushna_count += 1
                elif 'Sheeta' in virya:
                    sheeta_count += 1
        if ushna_count > sheeta_count and ushna_count >= 2:
            return {
                "main_question": self.property_questions["Ushna"]["main_question"],
                "options": self.property_questions["Ushna"]["options"],
                "flag": "Ushna",
                "reasoning": f"Majority of top candidates ({ushna_count}/3) are heating. Verifying safety."
            }
        elif sheeta_count > ushna_count and sheeta_count >= 2:
            return {
                "main_question": self.property_questions["Sheeta"]["main_question"],
                "options": self.property_questions["Sheeta"]["options"],
                "flag": "Sheeta",
                "reasoning": f"Majority of top candidates ({sheeta_count}/3) are cooling. Verifying safety."
            }
        return {
            "main_question": "Which of the following best describes your condition? (Select all that apply)",
            "options": ["Symptoms have lasted more than a week", "Symptoms are getting worse", "I have taken medication for this already", "None of these"],
            "flag": "chronic",
            "reasoning": "Balanced candidate pool. Standard baseline check."
        }

if __name__ == "__main__":
    from src.config import INDEX_DIR
    from src.retrieval import BioVedaRetriever
    
    # 1. Run the retrieval
    retriever = BioVedaRetriever()
    user_query = "I have been experiencing acid reflux and heartburn after meals."
    results = retriever.query(user_query)
    
    # 2. Pass results to Dialogue Manager
    dialogue_manager = BioVedaDialogueManager(metadata_path=INDEX_DIR / "plants_metadata.pkl")
    clarification = dialogue_manager.generate_clarification_question(results)
    
    print(f"\nUser Query: {user_query}")
    print(f"System Identified Candidates: {[r['english_name'] for r in results]}")
    print(f"\n--- Dialogue Manager Trigger ---")
    print(f"Generated Question: {clarification['question']}")
    print(f"System Reasoning: {clarification['reasoning']}")