import sys
from pathlib import Path

# Explicitly add the root folder to Python's path so it can find 'src'
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import streamlit as st
from src.config import INDEX_DIR
from src.retrieval import BioVedaRetriever
from src.dialogue_manager import BioVedaDialogueManager
from src.ranking_engine import BioVedaRankingEngine

# Initialize ML Modules (cached to prevent reloading on every button click)
@st.cache_resource
def load_modules():
    retriever = BioVedaRetriever(top_k=5)
    dialogue_manager = BioVedaDialogueManager(metadata_path=INDEX_DIR / "plants_metadata.pkl")
    ranking_engine = BioVedaRankingEngine(metadata_path=INDEX_DIR / "plants_metadata.pkl")
    return retriever, dialogue_manager, ranking_engine

retriever, dialogue_manager, ranking_engine = load_modules()

# UI Setup
st.set_page_config(page_title="BioVeda AI", layout="centered")
st.title("🌿 BioVeda")
st.subheader("Intelligent Medicinal Plant Recommendation System")

# Session State Management
if 'stage' not in st.session_state:
    st.session_state.stage = 1
if 'retrieved_candidates' not in st.session_state:
    st.session_state.retrieved_candidates = []
if 'clarification_data' not in st.session_state:
    st.session_state.clarification_data = {}

# STAGE 1: Natural Language Input
if st.session_state.stage == 1:
    st.write("Describe your health problem or symptoms in everyday language.")
    user_query = st.text_area("Symptoms:", placeholder="E.g., I have been experiencing acid reflux and heartburn...")
    
    if st.button("Analyze Symptoms"):
        if user_query.strip():
            with st.spinner("Analyzing semantics and searching knowledge graph..."):
                # 1. Retrieve candidates
                candidates = retriever.query(user_query)
                st.session_state.retrieved_candidates = candidates
                
                # 2. Generate clarification question
                clarification = dialogue_manager.generate_clarification_question(candidates)
                st.session_state.clarification_data = clarification
                
                # Move to next stage
                st.session_state.stage = 2
                st.rerun()
        else:
            st.warning("Please enter your symptoms to continue.")

# STAGE 2: Ambiguity Resolution (Yes/No)
# STAGE 2: Ambiguity Resolution (Multi-Select)
elif st.session_state.stage == 2:
    st.info("System requires clinical clarification before recommending.")
    st.write(f"**{st.session_state.clarification_data['main_question']}**")
    
    # Render the options as a multi-select box
    selected_options = st.multiselect(
        "Select all that apply:",
        options=st.session_state.clarification_data['options']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Clarification"):
            # Save the list of selections to session state
            st.session_state.user_selections = selected_options
            st.session_state.stage = 3
            st.rerun()
    with col2:
        if st.button("Start Over"):
            st.session_state.stage = 1
            st.rerun()

# STAGE 3: Final Recommendation & Ranking
elif st.session_state.stage == 3:
    st.success("Analysis Complete")
    
    # Pass the list of selections to the ranking engine
    final_results = ranking_engine.apply_safety_filters_and_rank(
        st.session_state.retrieved_candidates,
        st.session_state.clarification_data['flag'],
        st.session_state.user_selections
    )
    
    st.write("### Recommended Medicinal Plants")
    for res in final_results[:3]: # Display top 3
        with st.expander(f"Rank {res['rank']}: {res['english_name']} ({res['botanical_name']})"):
            st.write(f"**Safety Rationale:** {res['rationale']}")
            st.write(f"**Match Distance Score:** {res['final_adjusted_score']}")
            
            # Fetch original preview and indications from retrieved candidates
            original_data = next((item for item in st.session_state.retrieved_candidates if item["botanical_name"] == res['botanical_name']), None)
            if original_data:
                st.write(f"**Indications:** {', '.join(original_data['indications'])}")
                st.write(f"**Details:** {original_data['preview']}")

    if st.button("Start New Consultation"):
        st.session_state.stage = 1
        st.rerun()