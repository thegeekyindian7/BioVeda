import logging
import pandas as pd
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')

class BioVedaRankingEngine:
    def __init__(self, metadata_path: str):
        self.df_meta = pd.read_pickle(metadata_path)
        # The penalty multiplier applied to the FAISS distance score.
        # Since FAISS uses L2 distance (lower is better), we ADD a penalty to make the score worse.
        self.safety_penalty_weight = 20.0 

    def apply_safety_filters_and_rank(self, retrieved_results: List[Dict], dialogue_flag: str, user_selections: List[str]) -> List[Dict]:
        """Adjusts scores based on granular user selections."""
        final_results = []
        
        # If the user selected anything other than "None of these" or leaving it empty, 
        # it means they have the risk factor.
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
                "rank": 0, # Will update after sorting
                "english_name": res['english_name'],
                "botanical_name": res['botanical_name'],
                "original_score": base_score,
                "final_adjusted_score": round(final_score, 2),
                "rationale": rationale,
                "virya": virya
            })

        # Sort by the new adjusted score (Ascending, because lower L2 distance is better)
        final_results = sorted(final_results, key=lambda x: x['final_adjusted_score'])
        
        # Update ranks
        for i, res in enumerate(final_results):
            res['rank'] = i + 1
            
        return final_results

if __name__ == "__main__":
    from src.config import INDEX_DIR
    
    # Simulating the pipeline from the previous steps
    mock_retrieved_results = [
        {'english_name': 'Fresh Ginger', 'botanical_name': 'Zingiber officinale', 'distance_score': 52.67},
        {'english_name': 'Cardamom', 'botanical_name': 'Elettaria cardamomum', 'distance_score': 53.31},
        # Injecting a cooling herb (Amla) that also helps with digestion to see if it overtakes Ginger
        {'english_name': 'Indian Gooseberry', 'botanical_name': 'Phyllanthus emblica', 'distance_score': 58.10} 
    ]
    
    dialogue_flag = "Ushna"
    
    ranking_engine = BioVedaRankingEngine(metadata_path=INDEX_DIR / "plants_metadata.pkl")
    
    print("\n--- SCENARIO 1: User says 'NO' (Does not have excessive heat) ---")
    results_no = ranking_engine.apply_safety_filters_and_rank(mock_retrieved_results, dialogue_flag, user_answered_yes=False)
    for r in results_no:
        print(f"Rank {r['rank']}: {r['english_name']} | Score: {r['final_adjusted_score']} | {r['rationale']}")

    print("\n--- SCENARIO 2: User says 'YES' (Has severe burning/heat) ---")
    results_yes = ranking_engine.apply_safety_filters_and_rank(mock_retrieved_results, dialogue_flag, user_answered_yes=True)
    for r in results_yes:
        print(f"Rank {r['rank']}: {r['english_name']} | Score: {r['final_adjusted_score']} | {r['rationale']}")