
import streamlit as st
import pandas as pd
import os
from utils import load_financial_data, create_dummy_data

def main():
    st.markdown("### Story: Welcome to the LLM Risk Assessment Workbench, Apex Financial Services Risk Manager")
    st.markdown(
        """
        As a **Risk Manager** at Apex Financial Services, your mission is to rigorously evaluate the trustworthiness and reliability of our new LLM-powered financial summarization tool. 
        The market demands accuracy and fairness, and any emergent risks like hallucinations or biases in AI outputs could have severe financial and reputational consequences. 
        This application is your interactive workbench to systematically quantify these risks and formulate actionable mitigation strategies.

        Your journey will take you through critical steps:
        1.  **Data Ingestion:** Load and review the source financial documents and ground truth.
        2.  **Baseline Performance:** Understand the LLM's default behavior.
        3.  **Hallucination Detection:** Quantify factual inaccuracies and contradictions.
        4.  **Bias Identification:** Analyze sentiment and entity framing.
        5.  **Adversarial Testing:** Proactively uncover vulnerabilities.
        6.  **Prompt Refinement:** Improve LLM reliability.
        7.  **Risk Aggregation & Recommendations:** Synthesize findings and advise on governance.

        Each step provides you with the tools to make informed decisions, ensuring our AI systems are not just efficient, but also robust and responsible.
        """
    )

    st.markdown("### Real-World Problem: Mitigating LLM Hallucinations and Biases in Finance")
    st.markdown(
        """
        Apex Financial Services aims to leverage Large Language Models (LLMs) to process vast amounts of financial news and reports, accelerating information flow for critical investment decisions. However, the inherent risks of LLMs – specifically **hallucinations** (factually incorrect but credible outputs) and **biases** (skewed interpretations or unfair representations) – pose significant threats. These emergent risks could lead to erroneous financial analyses, misinformed compliance strategies, and severe reputational damage. Your problem is to rigorously quantify these risks and formulate actionable mitigation strategies to ensure the LLM's outputs are accurate, fair, and reliable.
        """
    )

    st.markdown("### How this page helps you (Data Setup)")
    st.markdown(
        """
        To begin our assessment, we need to load the financial data that the LLM will summarize and the corresponding human-verified ground truth data against which we will evaluate the LLM's performance. 
        As a Risk Manager, ensuring the integrity and representativeness of this initial dataset is foundational to a robust risk assessment. 
        This page allows you to either upload your own datasets or generate a set of dummy data to quickly proceed with the demonstration.
        """
    )

    # Initialize session state for data if not present
    if 'data' not in st.session_state:
        st.session_state.data = {}
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0 # Start at step 0 (Welcome/Data Setup)
    if 'confidence_correlation' not in st.session_state:
        st.session_state.confidence_correlation = 0.0
    if 'overall_risk_summary' not in st.session_state:
        st.session_state.overall_risk_summary = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []

    st.subheader("1. Data Loading and Initialization")
    doc_uploader = st.file_uploader("Upload Financial Documents (JSON)", type=["json"], key="doc_uploader")
    summary_uploader = st.file_uploader("Upload Ground Truth Summaries (CSV)", type=["csv"], key="summary_uploader")
    facts_uploader = st.file_uploader("Upload Ground Truth Facts (CSV)", type=["csv"], key="facts_uploader")

    if st.button("Load Data & Initialize", key="load_data_btn"):
        if doc_uploader and summary_uploader and facts_uploader:
            with st.spinner("Loading uploaded data..."):
                # Save uploaded files temporarily
                os.makedirs('uploaded_data', exist_ok=True)
                doc_file_path = os.path.join('uploaded_data', doc_uploader.name)
                summary_file_path = os.path.join('uploaded_data', summary_uploader.name)
                facts_file_path = os.path.join('uploaded_data', facts_uploader.name)

                with open(doc_file_path, "wb") as f:
                    f.write(doc_uploader.getvalue())
                with open(summary_file_path, "wb") as f:
                    f.write(summary_uploader.getvalue())
                with open(facts_file_path, "wb") as f:
                    f.write(facts_uploader.getvalue())
                
                st.session_state.data = load_financial_data(doc_file_path, summary_file_path, facts_file_path)
                st.success("Financial data loaded successfully from uploaded files!")
                st.session_state.current_step = 1 # Move to next logical step
                st.rerun()
        else:
            st.error("Please upload all required files to proceed with custom data.")
            
    st.markdown("---")
    st.info("Or, for a quick demonstration without uploads:")
    if st.button("Generate Dummy Data (for demonstration)", key="dummy_data_btn"):
        with st.spinner("Generating dummy data..."):
            create_dummy_data()
            st.session_state.data = load_financial_data('data/financial_documents.json', 'data/ground_truth_summaries.csv', 'data/ground_truth_facts.csv')
            st.success("Dummy data loaded successfully!")
            st.session_state.current_step = 1 # Move to next logical step
            st.rerun()

    if st.session_state.data:
        st.markdown("### Preview of Loaded Financial Documents")
        st.markdown("As a Risk Manager, reviewing this data ensures that the dataset for our LLM evaluation is correctly ingested and structured, reflecting the real-world financial reports we aim to summarize.")
        
        preview_df = pd.DataFrame([
            {'doc_id': k,
             'company': v['company'],
             'industry': v['industry'],
             'sentiment_tag': v['sentiment_tag'],
             'text_excerpt': v['text'][:150] + '...' if len(v['text']) > 150 else v['text'],
             'ground_truth_summary_excerpt': v['ground_truth_summary'][:150] + '...' if len(v['ground_truth_summary']) > 150 else v['ground_truth_summary'],
             'num_ground_truth_facts': len(v['ground_truth_facts'])}
            for k, v in st.session_state.data.items()
        ])
        st.dataframe(preview_df)
        st.markdown("---")
        st.success("Data successfully prepared. You can now move to the next step to generate baseline LLM summaries.")
    else:
        st.info("Upload data or generate dummy data to begin the LLM risk assessment.")


