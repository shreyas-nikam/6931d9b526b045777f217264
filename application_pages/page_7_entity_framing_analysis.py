
import streamlit as st
import pandas as pd
from utils import analyze_entity_framing_bias

def main():
    st.markdown("### Story: Bias Detection: Entity Representation and Framing Analysis")
    st.markdown(
        """
        Beyond overall sentiment, how an LLM references and frames specific entities can introduce subtle biases. 
        Your role as a **Risk Manager** is to ensure the LLM presents information fairly and avoids perpetuating stereotypes. 
        We will extract key entities and analyze their frequency and contextual sentiment in LLM summaries compared to ground truth.
        """
    )

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return
    
    if not all('llm_summary_baseline' in d for d in st.session_state.data.values()):
        st.warning("Please generate baseline summaries in the 'Baseline LLM Summaries' page first.")
        return

    st.markdown("### How this page helps you (Entity Framing Analysis)")
    st.markdown(
        """
        This step allows you to specify key entities (e.g., specific companies, roles like 'CEO') and then run an analysis comparing how frequently 
        the LLM mentions these entities and the average sentiment around them, versus their representation in the original documents and ground truth. 
        As a Risk Manager, this helps you uncover subtle 'representational harms' – for instance, if the LLM consistently downplays certain entities 
        or frames them negatively without factual basis. Such biases could lead to unfair valuations or misjudgment of market players.
        """
    )

    st.subheader("4. Select Target Entities for Analysis")
    # Initialize target_entities_multiselect in session_state if not present
    if 'target_entities_multiselect' not in st.session_state:
        st.session_state.target_entities_multiselect = ["GlobalBank Inc.", "TechInnovate Solutions", "GreenEnergy Corp."]

    target_entities_input = st.multiselect(
        "Target Entities for Analysis (Optional)",
        options=["GlobalBank Inc.", "TechInnovate Solutions", "GreenEnergy Corp.", "CEO", "CFO", "analyst"],
        default=st.session_state.target_entities_multiselect,
        key="target_entities_multiselect"
    )

    if st.button("Analyze Entity Framing Bias", key="analyze_entity_btn"):
        with st.spinner("Analyzing entity framing bias..."):
            st.session_state.entity_framing_analysis_df, fig = analyze_entity_framing_bias(st.session_state.data, target_entities=target_entities_input)
            st.session_state.fig_entity_framing = fig # Store the figure in session state
            st.success("Entity framing bias analysis complete!")
            st.session_state.current_step = 7 # Move to next logical step
            st.rerun()

    if st.session_state.current_step >= 7 and 'entity_framing_analysis_df' in st.session_state:
        st.markdown("### Entity Representation and Framing Analysis Results")
        st.markdown(
            """
            The bar plot illustrates the mention counts of selected entities in LLM summaries compared to the combined ground truth (original document + GT summary). 
            The table provides a detailed breakdown including average contextual sentiment. 
            As a Risk Manager, a significant discrepancy in mention frequency or sentiment for key entities indicates a potential framing bias. 
            For example, if the LLM mentions a competitor far less frequently or with a consistently negative sentiment than warranted by the source, 
            this could influence strategic competitive analysis.
            """
        )
        if 'fig_entity_framing' in st.session_state:
            st.pyplot(st.session_state.fig_entity_framing)
        else:
            st.info("Click 'Analyze Entity Framing Bias' to see the plot.")
        
        st.dataframe(st.session_state.entity_framing_analysis_df)
        st.markdown("The entity representation and framing analysis provides insights into how the LLM handles specific entities. If the LLM consistently omits specific entities, or frames them with a different sentiment compared to the ground truth, it suggests a 'representational harm' bias. This could lead to a skewed perception of market players, impacting strategic decisions.")
