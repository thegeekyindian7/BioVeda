import pandas as pd
import numpy as np
from pathlib import Path
from src.retrieval import BioVedaRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_SET_PATH = BASE_DIR / "data" / "processed" / "golden_test_set.csv"

class BioVedaEvaluator:
    def __init__(self):
        # We retrieve top 5 to calculate metrics up to K=5
        self.retriever = BioVedaRetriever(top_k=5)
        self.test_df = pd.read_csv(TEST_SET_PATH)
        
    def evaluate(self):
        print(f"Starting evaluation on {len(self.test_df)} queries...\n")
        
        reciprocal_ranks = []
        hits_at_3 = []
        
        for index, row in self.test_df.iterrows():
            query = row['query']
            expected_plant = row['expected_botanical_name']
            
            # 1. Retrieve candidates
            results = self.retriever.query(query)
            retrieved_botanical_names = [res['botanical_name'] for res in results]
            
            # 2. Calculate Rank of the expected plant
            try:
                # Adding 1 because list indices start at 0
                rank = retrieved_botanical_names.index(expected_plant) + 1
                rr = 1.0 / rank
                # Check if it appeared in the top 3
                hit_3 = 1 if rank <= 3 else 0
            except ValueError:
                # Plant was not found in the top 5
                rank = "Not Found (in Top 5)"
                rr = 0.0
                hit_3 = 0
                
            reciprocal_ranks.append(rr)
            hits_at_3.append(hit_3)
            
            print(f"Query: '{query}'")
            print(f"Expected: {expected_plant} | Found at Rank: {rank}")
            print(f"Reciprocal Rank: {rr:.2f}\n")
            
        # 3. Calculate Final Aggregated Metrics
        mrr = np.mean(reciprocal_ranks)
        hit_rate_at_3 = np.mean(hits_at_3) * 100
        
        # 3. Calculate Final Aggregated Metrics
        mrr = np.mean(reciprocal_ranks)
        hit_rate_at_3 = np.mean(hits_at_3) * 100
        
        # 4. Format the report
        report_text = f"""----------------------------------------
🏆 FINAL EVALUATION METRICS 🏆
----------------------------------------
Total Queries Evaluated : {len(self.test_df)}
Mean Reciprocal Rank (MRR): {mrr:.4f}  <-- (Closer to 1.0 is better)
Hit Rate @ 3 (HR@3)       : {hit_rate_at_3:.1f}% <-- (% of times the correct plant was in top 3)
----------------------------------------
"""
        print(report_text)
        
        # 5. Save to a text file
        report_dir = BASE_DIR / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation_metrics.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"✅ Metrics successfully saved to: {report_path}")

if __name__ == "__main__":
    evaluator = BioVedaEvaluator()
    evaluator.evaluate()