
import streamlit as st
import pandas as pd
import numpy as np
from utils import evaluate_confidence_accuracy_correlation
import matplotlib.pyplot as plt # Import for st.pyplot

def main():
    st.markdown("### Story: Quantifying Uncertainty: Aligning LLM Confidence with Accuracy")
    st.markdown(
        """
        As a **Risk Manager**, you want to assess if the LLM's reported confidence is actually indicative of factual accuracy. 
        A model that is confidently wrong is far more dangerous than one that knows its limitations. 
        We will evaluate how well the LLM's self-reported confidence aligns with the empirically observed factual accuracy of its summaries.
        """
    )
    st.markdown("We will calculate the Pearson correlation coefficient $r$ between the LLM's confidence scores ($X$) and the factual accuracy scores ($Y$) for each document:")
    st.markdown(r"$$ r = \frac{\sum_{i=1}^{N} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{N} (X_i - \bar{X})^2 \sum_{i=1}^{N} (Y_i - \bar{Y})^2}} $$")

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return
    
    # Check if fact assessment has been done for all documents
    if not all('fact_assessment' in d for d in st.session_state.data.values()):
        st.warning("Please complete the 'Factual Accuracy Assessment' page first to generate accuracy scores.")
        return

    st.markdown("### How this page helps you (Confidence vs. Accuracy Evaluation)")
    st.markdown(
        """
        This step directly helps you quantify the trustworthiness of the LLM's self-assessment. 
        By clicking the button below, the application correlates the confidence scores assigned by the LLM to its summaries with their actual factual accuracy. 
        As a Risk Manager, a high positive correlation indicates that the LLM generally 'knows what it knows', making its confidence scores a useful signal for identifying high-risk summaries. 
        A low or negative correlation, however, would flag a critical risk, suggesting the LLM's confidence cannot be relied upon, necessitating broader human review regardless of reported certainty.
        """
    )

    if st.button("Evaluate Confidence-Accuracy Correlation", key="eval_corr_btn"):
        with st.spinner("Calculating correlation and generating plot..."):
            st.session_state.confidence_evaluation_df, st.session_state.confidence_correlation, fig = evaluate_confidence_accuracy_correlation(st.session_state.data)
            st.success("Confidence-accuracy correlation evaluated!")
            st.session_state.current_step = 5 # Move to next logical step
            st.rerun()

    if st.session_state.current_step >= 5 and 'confidence_correlation' in st.session_state:
        st.markdown("### Confidence-Accuracy Correlation Results")
        st.markdown(
            """
            The scatter plot visualizes the relationship between the LLM's self-reported confidence and its actual factual accuracy. 
            Each point represents a document, showing how confident the LLM was in its summary versus how accurate that summary actually was. 
            The red dashed line indicates a minimum acceptable accuracy threshold, helping to identify summaries that are both low confidence and low accuracy.
            """
        )
        st.pyplot(st.session_state.fig_confidence_accuracy if 'fig_confidence_accuracy' in st.session_state else plt.figure())
        # Re-generate the plot to display it consistently. The function evaluate_confidence_accuracy_correlation returns the figure, so we need to store it in session state. 
        # Or, just call the function again and display the fig.
        # Given the instruction 