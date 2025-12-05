
import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: LLM Risk Assessment for Financial Summaries")
st.divider()

st.markdown(
    """
    Welcome to the **LLM Risk Assessment Workbench**, designed for **Risk Managers** at Apex Financial Services. 
    In today's fast-paced financial world, leveraging AI for tasks like summarizing financial reports is essential, but it comes with inherent risks.
    This application guides you through a comprehensive, story-driven workflow to evaluate the trustworthiness and reliability of our LLM-powered financial summarization tool.
    
    **Your Role:** You will navigate through crucial steps, from initial data setup to generating a final risk report, addressing real-world challenges such as mitigating LLM hallucinations and biases.
    
    **The Problem:** LLMs can produce factually incorrect (hallucinations) or skewed (biased) outputs, leading to erroneous financial analyses, misinformed compliance strategies, and severe reputational damage.
    
    **The Solution:** This app provides interactive tools to quantify these risks, allowing you to make data-driven decisions and formulate actionable mitigation strategies, ensuring our AI deployments are robust, fair, and compliant.
    """
)

# Define page names for navigation
page_names = [
    "Welcome & Setup",
    "Baseline LLM Summaries",
    "Factual Accuracy Assessment",
    "Contradiction Detection",
    "Confidence vs. Accuracy",
    "Sentiment Bias Analysis",
    "Entity Framing Analysis",
    "Adversarial Testing",
    "Prompt Refinement",
    "Aggregate Risks & Recommendations",
    "Final Risk Assessment Report"
]

# Navigation selectbox
page = st.sidebar.selectbox(label="Navigation", options=page_names, index=st.session_state.get("current_step", 0))

# Update session state based on selected page for consistent flow
# This ensures that if a user directly selects a page, the current_step reflects it.
# However, the internal logic of each page might still have its own checks for prerequisites.
current_page_index = page_names.index(page)
if st.session_state.get("current_step", 0) != current_page_index:
    st.session_state.current_step = current_page_index

# Display selected page content
if page == "Welcome & Setup":
    from application_pages.page_1_welcome_setup import main
    main()
elif page == "Baseline LLM Summaries":
    from application_pages.page_2_baseline_summaries import main
    main()
elif page == "Factual Accuracy Assessment":
    from application_pages.page_3_factual_accuracy_assessment import main
    main()
elif page == "Contradiction Detection":
    from application_pages.page_4_contradiction_detection import main
    main()
elif page == "Confidence vs. Accuracy":
    from application_pages.page_5_confidence_accuracy import main
    main()
elif page == "Sentiment Bias Analysis":
    from application_pages.page_6_sentiment_bias_analysis import main
    main()
elif page == "Entity Framing Analysis":
    from application_pages.page_7_entity_framing_analysis import main
    main()
elif page == "Adversarial Testing":
    from application_pages.page_8_adversarial_testing import main
    main()
elif page == "Prompt Refinement":
    from application_pages.page_9_prompt_refinement import main
    main()
elif page == "Aggregate Risks & Recommendations":
    from application_pages.page_10_aggregate_risks_recommendations import main
    main()
elif page == "Final Risk Assessment Report":
    from application_pages.page_11_final_risk_assessment_report import main
    main()
