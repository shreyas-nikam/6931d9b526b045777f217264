
import streamlit as st
import pandas as pd
import numpy as np
from utils import aggregate_and_visualize_risks, generate_hitl_recommendations, sid

def main():
    st.markdown("### Story: Aggregating and Visualizing Comprehensive Risk Findings")
    st.markdown(
        """
        As a **Risk Manager**, you need a consolidated and clear view of all identified risks across the LLM summarization process. 
        Now, it's time to synthesize these diverse findings into clear, actionable insights and visualizations. 
        This aggregation will form the backbone of our 'LLM Risk Assessment Report'.
        """
    )

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return

    # Pre-checks for necessary data from previous steps
    if not all('fact_assessment' in d for d in st.session_state.data.values()):
        st.warning("Please complete the 'Factual Accuracy Assessment' page first.")
        return
    if 'confidence_correlation' not in st.session_state:
        st.warning("Please complete the 'Confidence vs. Accuracy' page first.")
        return
    if not all('contradictions' in d for d in st.session_state.data.values()):
        st.warning("Please complete the 'Contradiction Detection' page first.")
        return
    # Adversarial test results are optional per document, but we should have run the page
    # if not all('adversarial_test_results' in d for d in st.session_state.data.values()):
    #     st.warning("Please complete the 'Adversarial Testing' page first.")
    #     return

    st.markdown("### How this page helps you (Risk Aggregation)")
    st.markdown(
        """
        This section provides a holistic view of the LLM's risk profile by aggregating all the individual risk metrics you've assessed. 
        By clicking 'Aggregate All Risks', you will see summary statistics and visualizations that paint a clear picture of the LLM's performance, 
        highlighting areas of high risk (e.g., frequent hallucinations) and areas of strength. 
        As a Risk Manager, this consolidated view is indispensable for reporting to stakeholders and prioritizing mitigation strategies effectively.
        """
    )

    if st.button("Aggregate All Risks", key="aggregate_risks_btn"):
        with st.spinner("Aggregating and visualizing risks..."):
            # Recalculate overall_risk_summary to ensure it includes all fields needed for generate_hitl_recommendations
            st.session_state.overall_risk_summary = aggregate_and_visualize_risks(st.session_state.data, st.session_state.confidence_correlation)
            st.success("Risk aggregation and visualization complete!")
            st.session_state.current_step = 10 # Move to next logical step
            st.rerun()

    if st.session_state.current_step >= 10 and 'overall_risk_summary' in st.session_state:
        st.markdown("### Aggregated LLM Risk Assessment Findings")
        st.markdown(
            """
            The visualizations below provide an at-a-glance understanding of the LLM's overall risk profile. 
            The key metrics summarize its factual accuracy, sentiment balance, and vulnerability to various attacks. 
            As a Risk Manager, this holistic view is crucial for focusing mitigation efforts effectively and clearly communicating key findings to senior management and other stakeholders.
            """
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Avg Factual Accuracy", f"{st.session_state.overall_risk_summary.get('avg_factual_accuracy', 0):.2f}")
        col2.metric("Overall Avg LLM Confidence-Accuracy Correlation", f"{st.session_state.overall_risk_summary.get('avg_confidence_accuracy_corr', 0):.2f}")
        col3.metric("Overall Avg Sentiment Discrepancy", f"{st.session_state.overall_risk_summary.get('avg_sentiment_discrepancy', 0):.2f}")

        col4, col5, col6, col7 = st.columns(4)
        col4.metric("Total Hallucinated Facts", st.session_state.overall_risk_summary.get('total_hallucinated_facts', 0))
        col5.metric("Total Missing Facts", st.session_state.overall_risk_summary.get('total_missing_facts', 0))
        col6.metric("Total Detected Contradictions", st.session_state.overall_risk_summary.get('total_contradictions', 0))
        col7.metric("Total Adversarial Failures", st.session_state.overall_risk_summary.get('total_adversarial_failures', 0))

        st.markdown("---")
        st.markdown("### Story: Recommendations for Human-in-the-Loop (HITL) and Governance")
        st.markdown(
            """
            Based on the identified risks—hallucinations, biases, and confidence misalignments—it is imperative to design robust 'Human-in-the-Loop (HITL)' checkpoints and governance structures. 
            As a **Risk Manager**, your recommendations will ensure human oversight and accountability remain paramount, translating quantitative risks into practical operational policies.
            """
        )

        st.markdown("### How this page helps you (Generate Recommendations)")
        st.markdown(
            """
            This section empowers you to configure thresholds for what you deem acceptable levels of risk across key metrics. 
            Based on these thresholds and the aggregated risk findings, the application will generate a set of concrete, actionable recommendations for HITL interventions and broader AI governance policies. 
            As a Risk Manager, this directly supports your responsibility to formulate and implement risk mitigation strategies that are tailored to Apex Financial Services' risk appetite and operational realities.
            """
        )

        st.subheader("7. Configure Recommendation Thresholds")
        # Initialize sliders in session_state if not present
        if 'acc_thresh_slider' not in st.session_state:
            st.session_state.acc_thresh_slider = 95
        if 'sentiment_discrepancy_thresh_slider' not in st.session_state:
            st.session_state.sentiment_discrepancy_thresh_slider = 0.1
        if 'adv_failure_limit_input' not in st.session_state:
            st.session_state.adv_failure_limit_input = 0

        acc_thresh_input = st.slider(
            "Min Acceptable Factual Accuracy (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.acc_thresh_slider,
            step=1,
            key="acc_thresh_slider"
        ) / 100.0 # Convert to a ratio for the function
        
        sentiment_disp_thresh_input = st.slider(
            "Max Acceptable Sentiment Discrepancy (Abs. Score)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.sentiment_discrepancy_thresh_slider,
            step=0.05,
            key="sentiment_discrepancy_thresh_slider"
        )
        adv_fail_limit_input = st.number_input(
            "Max Acceptable Adversarial Failures",
            min_value=0,
            value=st.session_state.adv_failure_limit_input,
            step=1,
            key="adv_failure_limit_input"
        )

        if st.button("Generate HITL & Governance Recommendations", key="gen_rec_btn"):
            with st.spinner("Generating recommendations..."):
                st.session_state.recommendations = generate_hitl_recommendations(
                    st.session_state.overall_risk_summary,
                    accuracy_threshold=acc_thresh_input,
                    sentiment_discrepancy_threshold=sentiment_disp_thresh_input,
                    adversarial_failure_limit=adv_fail_limit_input
                )
                st.success("Recommendations generated!")
                st.session_state.current_step = 11 # Move to final step
                st.rerun()

        if st.session_state.current_step >= 11 and 'recommendations' in st.session_state:
            st.markdown("### Key Recommendations for Human-in-the-Loop & Governance")
            st.markdown(
                """
                These concrete recommendations outline practical steps for integrating human oversight, audit mechanisms, and continuous monitoring into the LLM deployment lifecycle. 
                As a Risk Manager, presenting these findings ensures that while AI enhances efficiency, critical decisions remain informed by human expertise and accountability, upholding our commitment to responsible AI.
                """
            )
            for i, rec in enumerate(st.session_state.recommendations):
                st.markdown(f"{i+1}. {rec}")

