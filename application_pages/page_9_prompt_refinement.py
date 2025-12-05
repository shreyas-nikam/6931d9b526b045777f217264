This page provides the crucial opportunity to implement **prompt engineering** based on your insights from the preceding risk assessments. 
        You will craft a more precise and restrictive prompt, and then re-evaluate the LLM's performance on a selected document. 
        As a Risk Manager, this demonstrates the iterative process of mitigating AI risks: identifying vulnerabilities, designing corrective interventions (like improved prompts), and then quantifying the resulting improvements in factual accuracy and bias reduction. 
        This hands-on refinement solidifies your understanding of how to guide LLM behavior for trustworthy outcomes.
        """
    )

    st.subheader("6. Refine Prompt and Retest")
    # Initialize retest_doc_selector in session_state if not present
    if 'retest_doc_selector' not in st.session_state:
        st.session_state.retest_doc_selector = 'doc_003' if 'doc_003' in st.session_state.data else (list(st.session_state.data.keys())[0] if st.session_state.data else None)

    retest_doc_id = st.selectbox(
        "Select a document for prompt refinement retest:",
        options=list(st.session_state.data.keys()),
        key="retest_doc_selector",
        index=list(st.session_state.data.keys()).index(st.session_state.retest_doc_selector) if st.session_state.retest_doc_selector in st.session_state.data else 0
    )

    if 'refined_prompt_input' not in st.session_state:
        st.session_state.refined_prompt_input = "Strictly summarize the following financial document, extracting only explicitly stated factual information and avoiding any speculative language or unmentioned details. Ensure the summary is neutral in tone: \n\n{document_text}"

    refined_prompt = st.text_area(
        "Refined Prompt Template",
        value=st.session_state.refined_prompt_input,
        height=150,
        key="refined_prompt_input"
    )

    if st.button("Refine & Retest LLM", key="retest_llm_btn"):
        if retest_doc_id and retest_doc_id in st.session_state.data:
            with st.spinner(f"Refining and retesting LLM for {retest_doc_id}..."):
                doc_data = st.session_state.data[retest_doc_id]
                retest_results = refine_and_retest_prompts(
                    doc_data['text'],
                    refined_prompt,
                    doc_data['ground_truth_summary'],
                    doc_data['ground_truth_facts'],
                    doc_id_for_simulation=retest_doc_id # Pass for specific simulation
                )
                st.session_state.data[retest_doc_id]['refined_prompt_results'] = retest_results
                st.success("Prompt refined and retested successfully!")
                st.session_state.current_step = 9 # Move to next logical step
                st.rerun()
        else:
            st.error("Please select a valid document for retesting.")

    if st.session_state.current_step >= 9 and retest_doc_id and 'refined_prompt_results' in st.session_state.data.get(retest_doc_id, {}):
        st.markdown(f"### Retest Results with Refined Prompt for {retest_doc_id}")
        st.markdown(
            """
            The comparison below showcases the impact of your refined prompt on the LLM's performance for the selected document. 
            As a Risk Manager, observing improvements in factual accuracy and reduced sentiment discrepancy directly validates the effectiveness of prompt engineering 
            as a mitigation strategy against identified LLM risks. This quantitative evidence supports decisions on deploying improved AI interactions.
            """
        )
        retest_results = st.session_state.data[retest_doc_id]['refined_prompt_results']
        doc_data = st.session_state.data[retest_doc_id]

        # Ensure baseline values exist before trying to access them
        baseline_facts = doc_data['fact_assessment']['accuracy_score'] if 'fact_assessment' in doc_data else 0.0
        baseline_llm_summary = doc_data.get('llm_summary_baseline', '')
        baseline_ground_truth_summary = doc_data.get('ground_truth_summary', '')
        
        baseline_sentiment_compound = sid.polarity_scores(baseline_llm_summary)['compound']
        baseline_sentiment_gt_compound = sid.polarity_scores(baseline_ground_truth_summary)['compound']
        baseline_sentiment_discrepancy = abs(baseline_sentiment_compound - baseline_sentiment_gt_compound)

        st.markdown(f"**New LLM Summary (Refined):**\n{retest_results['llm_summary_refined']}\n")
        st.markdown(f"**New LLM Confidence:** {retest_results['llm_confidence_refined']:.2f}")

        comparison_df = pd.DataFrame({
            'Metric': ['Factual Accuracy Score', 'Sentiment Discrepancy'],
            'Baseline Value': [f"{baseline_facts:.2f}", f"{baseline_sentiment_discrepancy:.2f}"],
            'Refined Value': [f"{retest_results['fact_assessment_refined']['accuracy_score']:.2f}", f"{retest_results['sentiment_discrepancy_refined']:.2f}"]
        })
        st.table(comparison_df)

        st.markdown("The iterative process of prompt engineering demonstrably improves the LLM's output quality. By providing clearer instructions, we observe an improvement in factual accuracy and potentially reduced sentiment discrepancy, showcasing the effectiveness of targeted prompt design.")
