import pandas as pd
from pathlib import Path

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def create_starter_set():
    """Generates a starter Golden Test Set for IR evaluation."""
    data = [
        {
            "query": "I have a terrible headache and my head is pounding.",
            "expected_botanical_name": "Selinum wallichianum",
            "condition": "Headache/Neurological"
        },
        {
            "query": "I have been experiencing acid reflux and heartburn after meals.",
            "expected_botanical_name": "Zingiber officinale",
            "condition": "Digestion/Acidity"
        },
        {
            "query": "I can't sleep at night and feel very anxious.",
            "expected_botanical_name": "Valeriana wallichii",
            "condition": "Insomnia/Anxiety"
        },
        {
            "query": "I have a bad cough and difficulty breathing.",
            "expected_botanical_name": "Solanum indicum", # Brihati (respiratory)
            "condition": "Respiratory/Cough"
        }
    ]
    
    df = pd.DataFrame(data)
    csv_path = PROCESSED_DIR / "golden_test_set.csv"
    df.to_csv(csv_path, index=False)
    print(f"Golden Test Set created successfully at: {csv_path}")

if __name__ == "__main__":
    create_starter_set()