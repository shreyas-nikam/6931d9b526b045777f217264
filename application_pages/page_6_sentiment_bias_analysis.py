
import streamlit as st
import pandas as pd
from utils import analyze_sentiment_bias

def main():
    st.markdown("### Story: Bias Detection: Analyzing Sentiment Skew Across Categories")
    st.markdown(
        """
        Financial markets are highly sensitive to sentiment. Biases in LLM-generated summaries, such as consistently portraying certain industries more positively or negatively, can introduce distortions. 
        As a **Quantitative Analyst** (or Risk Manager overseeing this aspect), you need to detect if the LLM amplifies or introduces sentiment biases.
        """
    )
    st.markdown("We define the sentiment discrepancy for a document $i$ as $SD_i = |S_{LLM,i} - S_{GT,i}|$, where $S_{LLM,i}$ is the LLM's sentiment score and $S_{GT,i}$ is the ground truth sentiment score.")

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return
    
    if not all('llm_summary_baseline' in d for d in st.session_state.data.values()):
        st.warning("Please generate baseline summaries in the 'Baseline LLM Summaries' page first.")
        return

    st.markdown("### How this page helps you (Sentiment Bias Analysis)")
    st.markdown(
        """
        This section allows you to categorize financial documents by attributes like 'industry' or 'company' and then compare the average sentiment of the LLM's summaries against the ground truth sentiment for each category. 
        As a Risk Manager, this helps you identify systemic biases where the LLM consistently assigns a different sentiment to a particular sector or entity compared to human judgment. 
        Detecting such a 'disparate impact' early is crucial for preventing skewed investment recommendations or misinformed compliance decisions.
        """
    )

    st.subheader("3. Select Categorization for Sentiment Analysis")
    # Initialize sentiment_category_selector in session_state if not present
    if 'sentiment_category_selector' not in st.session_state:
        st.session_state.sentiment_category_selector = "industry"

    sentiment_category = st.selectbox(
        "Categorize by:",
        options=["industry", "company"],
        key="sentiment_category_selector",
        index=["industry", "company"].index(st.session_state.sentiment_category_selector)
    )

    if st.button("Analyze Sentiment Bias", key="analyze_sentiment_btn"):
        with st.spinner("Analyzing sentiment bias..."):
            st.session_state.sentiment_bias_by_category_df, fig = analyze_sentiment_bias(st.session_state.data, sentiment_category)
            st.session_state.fig_sentiment_bias = fig # Store the figure in session state
            st.success("Sentiment bias analysis complete!")
            st.session_state.current_step = 6 # Move to next logical step
            st.rerun()

    if st.session_state.current_step >= 6 and 'sentiment_bias_by_category_df' in st.session_state:
        st.markdown("### Sentiment Bias Analysis Results")
        st.markdown(
            """
            The bar chart below visually compares the mean sentiment scores from the LLM summaries against the ground truth for each selected category. 
            The table provides the numerical breakdown. 
            As a Risk Manager, a significant divergence in sentiment between the LLM and ground truth for any category indicates a potential bias that warrants further investigation. 
            Consistent positive or negative skew could lead to misinterpretations of financial health or market outlook within those categories.
            """
        )
        if 'fig_sentiment_bias' in st.session_state:
            st.pyplot(st.session_state.fig_sentiment_bias)
        else:
            st.info("Click 'Analyze Sentiment Bias' to see the plot.")
        
        st.dataframe(st.session_state.sentiment_bias_by_category_df)
        st.markdown("Analyzing sentiment skew helps us identify if the LLM introduces or amplifies biases related to specific industries. Discrepancies between LLM and ground truth sentiment, especially if consistent across a category, signal a 'disparate impact' risk. This suggests the LLM might be subtly influencing perception, which could lead to skewed investment recommendations.")
