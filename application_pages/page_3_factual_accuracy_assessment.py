
import streamlit as st
import pandas as pd
import numpy as np
from utils import extract_and_compare_facts


def main():
    st.markdown(
        "### Story: Hallucination Assessment: Factual Accuracy via Key Fact Extraction")
    st.markdown(
        """
        A primary concern for financial summaries is their absolute factual accuracy. Hallucinations in reported financial figures can be catastrophic. 
        As a **Risk Manager**, your task is to ensure that critical facts are correctly extracted and reported by the LLM. 
        We will implement a systematic way to extract and compare these key facts from both the LLM's summary and the pre-defined ground truth.
        """
    )
    st.markdown("We quantify factual accuracy using a ratio:")
    st.markdown(
        r"$$ Factual\_Accuracy = \frac{\text{Number of Correctly Extracted Facts}}{\text{Total Number of Ground Truth Facts}} $$")
    st.markdown(
        """
        A fact is considered 'correctly extracted' if the LLM output contains the specified `fact_key` and its `fact_value` matches the ground truth 
        (allowing for minor formatting variations). Hallucinated facts are those plausible financial figures in the LLM output that are not present in the ground truth.
        """
    )

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return

    st.subheader("2. Configure Fact Comparison")
    # Initialize comparison_threshold_slider in session_state if not present
    if 'comparison_threshold_slider' not in st.session_state:
        st.session_state.comparison_threshold_slider = 0.8

    comparison_threshold = st.slider(
        "Fact Comparison Threshold (Similarity Ratio)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.comparison_threshold_slider,
        step=0.05,
        key="comparison_threshold_slider"
    )

    st.markdown("### How this page helps you (Factual Accuracy Assessment)")
    st.markdown(
        """
        By adjusting the `Fact Comparison Threshold`, you can control how strictly the application matches facts extracted from the LLM summary against the ground truth. 
        A higher threshold (e.g., 0.9) demands almost exact matches, while a lower one (e.g., 0.7) allows for more semantic flexibility. 
        This control helps you simulate different levels of strictness in reviewing financial reporting, directly impacting the calculated factual accuracy score.
        """
    )

    if st.button("Assess Factual Accuracy", key="assess_facts_btn"):
        if not all('llm_summary_baseline' in d for d in st.session_state.data.values()):
            st.warning(
                "Please generate baseline summaries in the 'Baseline LLM Summaries' page first.")
            return

        with st.spinner("Assessing factual accuracy..."):
            for doc_id, doc_data in st.session_state.data.items():
                fact_assessment_results = extract_and_compare_facts(
                    doc_data['llm_summary_baseline'],
                    doc_data['ground_truth_facts'],
                    comparison_threshold
                )
                st.session_state.data[doc_id]['fact_assessment'] = fact_assessment_results
            st.success("Factual accuracy assessment complete!")

    # Show results if fact assessments exist
    if all('fact_assessment' in d for d in st.session_state.data.values()):
        st.markdown("### Factual Accuracy Results")
        st.markdown(
            """
            The following metrics and detailed view allow you to understand the LLM's performance in accurately extracting and reporting key financial facts. 
            As a Risk Manager, a high factual accuracy score is critical. 
            Any score below your acceptable threshold for financial reporting indicates a clear need for human intervention or further prompt engineering to mitigate hallucination risks.
            """
        )

        all_accuracy_scores = [d['fact_assessment']['accuracy_score']
                               for d in st.session_state.data.values() if 'fact_assessment' in d]
        avg_accuracy = np.mean(
            all_accuracy_scores) if all_accuracy_scores else 0
        st.metric("Overall Average Factual Accuracy", f"{avg_accuracy:.2f}")
        st.markdown(
            f"Your current overall factual accuracy score is **{avg_accuracy:.2f}**. As a Risk Manager, this score directly indicates the reliability of the LLM's financial reporting. A score below $Y\%$ necessitates human review to prevent misinformed investment decisions.")

        st.markdown("---")
        st.subheader("Detailed Fact Review per Document")
        if 'fact_doc_selector' not in st.session_state:
            st.session_state.fact_doc_selector = list(
                st.session_state.data.keys())[0]

        selected_doc_id_fact = st.selectbox(
            "Select a document for detailed fact review:",
            options=list(st.session_state.data.keys()),
            key="fact_doc_selector",
            index=list(st.session_state.data.keys()).index(
                st.session_state.fact_doc_selector) if st.session_state.fact_doc_selector in st.session_state.data else 0
        )

        if selected_doc_id_fact and 'fact_assessment' in st.session_state.data[selected_doc_id_fact]:
            doc_data = st.session_state.data[selected_doc_id_fact]
            st.markdown(
                f"**Document: {selected_doc_id_fact} - {doc_data['company']}**")
            st.dataframe(pd.DataFrame(
                doc_data['fact_assessment']['detailed_results']))
            st.markdown(
                f"**Correct Facts:** {doc_data['fact_assessment']['correct_facts']}")
            st.markdown(
                f"**Missing Facts:** {doc_data['fact_assessment']['missing_facts']}")
            st.markdown(
                f"**Hallucinated Financial Values (potential):** {doc_data['fact_assessment']['hallucinated_facts']}")
            st.markdown(
                f"**Factual Accuracy Score for this Document:** {doc_data['fact_assessment']['accuracy_score']:.2f}")

            with st.expander("LLM Baseline Summary for this Document"):
                st.write(doc_data['llm_summary_baseline'])
            with st.expander("Ground Truth Facts for this Document"):
                st.write(doc_data['ground_truth_facts'])
        else:
            st.info("Select a document to view its detailed factual assessment.")

        st.markdown("---")
        if st.button("Next: Go to Contradiction Detection", key="next_page_3", type="primary"):
            st.session_state.current_step = 3
            st.rerun()
