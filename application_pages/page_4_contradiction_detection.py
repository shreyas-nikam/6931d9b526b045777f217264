
import streamlit as st
import pandas as pd
from utils import detect_contradictions


def main():
    st.markdown(
        "### Story: Hallucination Assessment: Detecting Contradictions and Fabrications")
    st.markdown(
        """
        Beyond simply missing or fabricating numbers, LLMs can generate text that subtly contradicts the source document or presents misleading information. 
        As a **Risk Manager**, you need to identify these subtle yet dangerous misrepresentations. 
        This involves a deeper semantic comparison between the LLM summary and the original document and ground truth summary.
        """
    )

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return

    st.markdown("### How this page helps you (Contradiction Detection)")
    st.markdown(
        """
        This step allows you to trigger a specialized mechanism to detect statements in the LLM's summaries that contradict the original source document or the human-verified ground truth. 
        Identifying these contradictions is crucial for uncovering more sophisticated forms of LLM hallucinations that might not be caught by simple fact-checking. 
        This directly supports your role in ensuring the integrity and consistency of AI-generated financial insights.
        """
    )

    if st.button("Detect Contradictions", key="detect_contradictions_btn"):
        if not all('llm_summary_baseline' in d for d in st.session_state.data.values()):
            st.warning(
                "Please generate baseline summaries in the 'Baseline LLM Summaries' page first.")
            return

        with st.spinner("Detecting contradictions..."):
            for doc_id, doc_data in st.session_state.data.items():
                identified_contradictions = detect_contradictions(
                    doc_data['llm_summary_baseline'],
                    doc_data['text'],
                    doc_data['ground_truth_summary']
                )
                st.session_state.data[doc_id]['contradictions'] = identified_contradictions
            st.success("Contradiction detection complete!")

    # Show results if contradictions have been detected
    if all('contradictions' in d for d in st.session_state.data.values()):
        st.markdown("### Contradiction Detection Results")
        st.markdown(
            """
            The results below highlight any identified contradictions within the LLM summaries. 
            As a Risk Manager, these specific instances of contradiction require immediate attention, as they represent fundamental misrepresentations that could lead to erroneous interpretations or compliance issues. 
            Understanding these helps you design better validation checks and refine LLM instructions.
            """
        )

        total_contradictions = sum(len([c for c in d.get('contradictions', [
        ]) if "No specific issues detected" not in c]) for d in st.session_state.data.values())
        st.metric("Total Detected Contradictions (across all documents)",
                  total_contradictions)

        st.markdown("---")
        st.subheader("Detailed Contradiction Review per Document")
        if 'contradiction_doc_selector' not in st.session_state:
            st.session_state.contradiction_doc_selector = list(
                st.session_state.data.keys())[0]

        selected_doc_id_contr = st.selectbox(
            "Select a document for detailed contradiction review:",
            options=list(st.session_state.data.keys()),
            key="contradiction_doc_selector",
            index=list(st.session_state.data.keys()).index(
                st.session_state.contradiction_doc_selector) if st.session_state.contradiction_doc_selector in st.session_state.data else 0
        )

        if selected_doc_id_contr and 'contradictions' in st.session_state.data[selected_doc_id_contr]:
            doc_data = st.session_state.data[selected_doc_id_contr]
            st.markdown(
                f"**Document: {selected_doc_id_contr} - {doc_data['company']}**")
            with st.expander("Original Document Excerpt"):
                st.write(
                    doc_data['text'][:500] + "..." if len(doc_data['text']) > 500 else doc_data['text'])
            st.markdown(
                f"**LLM Summary (Baseline):**\n{doc_data.get('llm_summary_baseline', 'N/A')}")

            actual_contradictions_for_doc = [
                c for c in doc_data['contradictions'] if "No specific issues detected" not in c]

            if actual_contradictions_for_doc:
                st.markdown("**Identified Contradictions:**")
                for i, contradiction in enumerate(actual_contradictions_for_doc):
                    st.markdown(f"- {contradiction}")
            else:
                st.info(
                    "No specific contradictions detected for this document by our simulated mechanism.")
        else:
            st.info(
                "Select a document to view its detailed contradiction assessment.")

        st.markdown("---")
        if st.button("Next: Go to Confidence vs. Accuracy", key="next_page_4", type="primary"):
            st.session_state.current_step = 4
            st.rerun()
