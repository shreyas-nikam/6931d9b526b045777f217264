
import streamlit as st
from utils import generate_risk_assessment_report_summary


def main():
    st.markdown("### Story: Finalizing the LLM Risk Assessment Report")
    st.markdown(
        """
        You have completed your comprehensive LLM risk assessment. The final, crucial step is to compile all these findings, analyses, and actionable recommendations into a coherent 'LLM Risk Assessment Report'. 
        This report is the tangible deliverable for senior management and relevant stakeholders, articulating the LLM's risk posture and the strategies for responsible deployment.
        """
    )

    if 'data' not in st.session_state or not st.session_state.data:
        st.warning("Please load data in the 'Welcome & Setup' page first.")
        return

    if 'recommendations' not in st.session_state or not st.session_state.recommendations:
        st.warning(
            "Please generate recommendations in the 'Aggregate Risks & Recommendations' page first.")
        return

    if 'overall_risk_summary' not in st.session_state or not st.session_state.overall_risk_summary:
        st.warning(
            "Please aggregate risks in the 'Aggregate Risks & Recommendations' page first.")
        return

    st.markdown("### How this page helps you (Final Report Generation)")
    st.markdown(
        """
        This final step compiles all your hard work into a comprehensive, stakeholder-ready report. 
        By clicking the button below, the application synthesizes all the risk metrics, observations, and policy recommendations you've developed throughout the assessment. 
        As a Risk Manager, this report is your definitive statement on the LLM's trustworthiness, enabling Apex Financial Services to make informed decisions about AI adoption, ensuring compliance, and maintaining investor confidence.
        """
    )

    if st.button("Generate Final Risk Assessment Report", key="gen_final_report_btn"):
        with st.spinner("Compiling final report..."):
            st.session_state.final_risk_assessment_report = generate_risk_assessment_report_summary(
                st.session_state.data,
                st.session_state.recommendations,
                st.session_state.overall_risk_summary
            )
            st.success("Final report compiled!")

    # Show final report if it has been generated
    if 'final_risk_assessment_report' in st.session_state:
        st.markdown("### LLM Risk Assessment Report")
        st.markdown(
            """
            The complete LLM Risk Assessment Report provides a clear, data-driven overview of the LLM's performance, its inherent risks, and pragmatic strategies for mitigation. 
            It empowers Apex Financial Services to move forward with AI adoption confidently, ensuring integrity, compliance, and ultimately, responsible innovation.
            """
        )
        st.markdown(st.session_state.final_risk_assessment_report)
