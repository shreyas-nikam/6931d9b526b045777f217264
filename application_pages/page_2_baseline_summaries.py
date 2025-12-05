        # are documents to select from and that all docs have baseline summaries before trying to access keys
        if st.session_state.data and all('llm_summary_baseline' in d for d in st.session_state.data.values()):
            selected_doc_id = st.selectbox("Select a document to review its baseline summary:", options=list(st.session_state.data.keys()), key="baseline_doc_selector")
            if selected_doc_id:
                doc_data = st.session_state.data[selected_doc_id]
                st.subheader(f"Document: {selected_doc_id} - {doc_data['company']}")
                with st.expander("Original Document Text"):
                    st.write(doc_data['text'])
                st.markdown(f"**Ground Truth Summary:**\n{doc_data['ground_truth_summary']}")
                st.markdown(f"**LLM Summary (Baseline):**\n{doc_data.get('llm_summary_baseline', 'N/A')}")
                st.markdown(f"**LLM Confidence Score (Baseline):** {doc_data.get('llm_confidence_baseline', 0.0):.2f}")
        else:
            st.info("Generate baseline summaries to review them here.")