import logging
import pandas as pd
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')

class BioVedaRankingEngine:
    def __init__(self, metadata_path: str):
        self.df_meta = pd.read_pickle(metadata_path)
        # Since FAISS uses L2 distance (lower is better), we ADD a penalty to make the score worse.
        self.safety_penalty_weight = 20.0 

    def apply_safety_filters_and_rank(self, retrieved_results: List[Dict], dialogue_flag: str, user_selections: List[str]) -> List[Dict]:
        final_results = []
        has_risk_symptoms = len(user_selections) > 0 and "None of these" not in user_selections
        for res in retrieved_results:
            plant_data = self.df_meta[self.df_meta['botanical_name'] == res['botanical_name']]
            if plant_data.empty: continue
            plant = plant_data.iloc[0]
            virya = str(plant.get('virya', ''))
            base_score = res['distance_score']
            penalty = 0.0
            rationale = "Recommended based on symptom match."

            # Apply Penalty Logic based on the new boolean
            if dialogue_flag == "Ushna" and has_risk_symptoms:
                if 'Ushna' in virya:
                    penalty = self.safety_penalty_weight
                    rationale = f"WARNING: Penalized due to heating properties which may aggravate: {', '.join(user_selections)}."
                elif 'Sheeta' in virya:
                    penalty = -5.0
                    rationale = "Safest match. Cooling properties counteract your reported heat symptoms."
            
            elif dialogue_flag == "Sheeta" and has_risk_symptoms:
                if 'Sheeta' in virya:
                    penalty = self.safety_penalty_weight
                    rationale = f"WARNING: Penalized due to cooling properties which may aggravate: {', '.join(user_selections)}."

            final_score = base_score + penalty
                       
            final_results.append({
                "rank": 0, 
                "english_name": res['english_name'],
                "botanical_name": res['botanical_name'],
                "original_score": base_score,
                "final_adjusted_score": round(final_score, 2),
                "rationale": rationale,
                "virya": virya
            })

        # Sorting adjusted score
        final_results = sorted(final_results, key=lambda x: x['final_adjusted_score'])
        
        for i, res in enumerate(final_results):
            res['rank'] = i + 1
            
        return final_results

if __name__ == "__main__":
    from src.config import INDEX_DIR
    dialogue_flag = "Ushna"
    ranking_engine = BioVedaRankingEngine(metadata_path=INDEX_DIR / "plants_metadata.pkl")