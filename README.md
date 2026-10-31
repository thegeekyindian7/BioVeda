# 🌿 BioVeda: Intelligent Medicinal Plant Recommendation System

**BioVeda** is an end-to-end Machine Learning and Information Retrieval (IR) system designed to recommend Ayurvedic medicinal plants based on natural language symptom descriptions. 

Developed as an M.Tech academic project, it bridges the semantic gap between colloquial English health queries and traditional Sanskrit/Ayurvedic medical terminology using zero-shot dense vector embeddings and a dynamic, information-theoretic safety dialogue manager.

---

## 🏗️ System Architecture

The system operates on a two-stage pipeline: **Semantic Retrieval** and **Clinical Ambiguity Resolution**.

```mermaid
flowchart TD
    %% Define Node Styles
    classDef user fill:#f9f,stroke:#333,stroke-width:2px;
    classDef model fill:#bbf,stroke:#333,stroke-width:2px;
    classDef db fill:#fbb,stroke:#333,stroke-width:2px;
    classDef engine fill:#bfb,stroke:#333,stroke-width:2px;

    User([User Input: English Symptoms]) ::: user
    UI[Streamlit Frontend] ::: user

    subgraph Phase 1: Semantic Retrieval
        BERT[PubMedBERT \n Sentence Transformer] ::: model
        FAISS[(FAISS Vector Index \n 360 Plants)] ::: db
        Hybrid[Hybrid Text Documents \n English + Sanskrit Indications] ::: db
    end

    subgraph Phase 2: Safety & Ambiguity Resolution
        DM[Dialogue Manager \n Information Gain Heuristic] ::: engine
        Ranker[Ranking Engine \n Safety Penalty System] ::: engine
    end

    User -->|Enters Text| UI
    UI -->|Query String| BERT
    BERT -->|768-D Vector| FAISS
    Hybrid -.->|Pre-computed| FAISS
    FAISS -->|Top-K Candidates| DM
    
    DM -->|Majority Rule Risk Check| UI
    UI -.->|Prompts Multi-Select Options| User
    User -.->|Selects Risk Factors| UI
    
    UI -->|Passes Boolean Flags| Ranker
    Ranker -->|Applies Mathematical Penalties| UI
    UI -->|Final Ranked Recommendation| User

🚀 Core Methodologies
1. Cross-Lingual Semantic Search (Hybrid Text Approach)
Standard NLP models fail to map English queries (e.g., "insomnia") to Ayurvedic terms (e.g., "Anidra"). BioVeda solves this by creating Hybrid Documents for each plant, concatenating its English description, scientific name, and Sanskrit indications. These are encoded into 768-dimensional dense vectors using a clinically fine-tuned Sentence Transformer (pritamdeka/S-PubMedBert-MS-MARCO).

2. Information-Theoretic Dialogue Manager
Instead of blindly recommending the closest mathematical match, BioVeda analyzes the Top-3 retrieved candidates. It applies a Majority Rule Heuristic to detect if the candidate pool heavily skews toward a specific pharmacological potency (e.g., Ushna/Heating). If a risk is detected, it pauses the retrieval to ask the user a targeted, granular safety question via the UI.

3. Safety Penalty Ranking
Based on the user's clarification, the Ranking Engine applies a mathematical penalty to the FAISS L2 distance scores. If a user reports severe body heat, heating herbs are penalized (pushed down the ranks), and cooling (Sheeta) alternatives are promoted, ensuring safe decision support.

📊 Evaluation Metrics
The system is evaluated as an Information Retrieval (IR) task using a Golden Test Set of colloquial queries.

Hit Rate @ 3 (HR@3): Measures how often the correct plant appears in the top 3 recommendations.

Mean Reciprocal Rank (MRR): Evaluates the exact ranking position of the correct plant across all queries (1.0 = perfect Rank 1 retrieval).

💻 Tech Stack
NLP & Embeddings: HuggingFace sentence-transformers, PubMedBERT

Vector Database: FAISS (Facebook AI Similarity Search)

Data Processing: Pandas, NumPy

Frontend: Streamlit

Knowledge Graph: Amidha Ayurveda Herb Database v2.0