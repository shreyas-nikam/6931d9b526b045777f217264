))", total_contradictions)

        st.markdown("---")
        st.subheader("Detailed Contradiction Review per Document")
        if 'contradiction_doc_selector' not in st.session_state:
            st.session_state.contradiction_doc_selector = list(st.session_state.data.keys())[0]

        selected_doc_id_contr = st.selectbox(
            "Select a document for detailed contradiction review:",
            options=list(st.session_state.data.keys()),
            key="contradiction_doc_selector",
            index=list(st.session_state.data.keys()).index(st.session_state.contradiction_doc_selector) if st.session_state.contradiction_doc_selector in st.session_state.data else 0
        )

        if selected_doc_id_contr and 'contradictions' in st.session_state.data[selected_doc_id_contr]:
            doc_data = st.session_state.data[selected_doc_id_contr]
            st.markdown(f"**Document: {selected_doc_id_contr} - {doc_data['company']}**")
            with st.expander("Original Document Excerpt"):
                st.write(doc_data['text'][:500] + "..." if len(doc_data['text']) > 500 else doc_data['text'])
            st.markdown(f"**LLM Summary (Baseline):**\n{doc_data.get('llm_summary_baseline', 'N/A')}")
            
            actual_contradictions_for_doc = [c for c in doc_data['contradictions'] if "No specific issues detected" not in c]

            if actual_contradictions_for_doc:
                st.error("#### Identified Contradictions:")
                for i, contradiction in enumerate(actual_contradictions_for_doc):
                    st.markdown(f"- {contradiction}")
            else:
                st.info("No specific contradictions detected for this document by our simulated mechanism.")
        else:
            st.info("Select a document to view its detailed contradiction assessment.")
