# QuLab: LLM Risk Assessment for Financial Summaries

## Project Title and Description

**QuLab: LLM Risk Assessment for Financial Summaries**

This Streamlit application is an interactive workbench designed for **Risk Managers** at Apex Financial Services to rigorously evaluate the trustworthiness and reliability of an LLM-powered financial summarization tool. In the critical domain of finance, the emergent risks of Large Language Models (LLMs) – such as hallucinations (generating factually incorrect but plausible information) and biases (skewed interpretations or unfair representations) – pose significant threats, potentially leading to erroneous analyses, compliance failures, and reputational damage.

This application provides a structured, step-by-step workflow to systematically quantify these risks. It offers a simulated environment to:
*   Assess factual accuracy and detect contradictions.
*   Identify sentiment and entity framing biases.
*   Proactively test for vulnerabilities using adversarial prompts.
*   Demonstrate prompt engineering for risk mitigation.
*   Aggregate findings and generate actionable recommendations for Human-in-the-Loop (HITL) interventions and AI governance.

While the LLM's responses are simulated for this lab project, the framework and methodologies presented are directly applicable to evaluating real-world LLM deployments in sensitive sectors like finance.

## Features

This application guides you through a comprehensive LLM risk assessment process with the following key features:

1.  **Data Ingestion & Setup (Welcome & Setup)**
    *   Option to upload custom financial documents (JSON), ground truth summaries (CSV), and ground truth facts (CSV).
    *   One-click generation of dummy financial data for quick demonstrations.
    *   Preview of loaded documents, companies, industries, and ground truth information.

2.  **Baseline LLM Summarization (Baseline LLM Summaries)**
    *   Simulated LLM generates initial summaries and confidence scores for all loaded financial documents.
    *   Interactive viewer to compare original document text, ground truth summaries, and baseline LLM summaries.

3.  **Factual Accuracy Assessment (Factual Accuracy Assessment)**
    *   Configurable fact comparison threshold for strictness in matching LLM-extracted facts against ground truth.
    *   Quantifies factual accuracy based on correctly extracted facts.
    *   Identifies missing facts and potential hallucinated financial values.
    *   Displays overall average factual accuracy and detailed per-document results.

4.  **Contradiction Detection (Contradiction Detection)**
    *   Simulated mechanism to detect statements in LLM summaries that contradict the original document or ground truth.
    *   Highlights specific instances of direct contradictions for immediate review.

5.  **Confidence vs. Accuracy Evaluation (Confidence vs. Accuracy)**
    *   Calculates Pearson correlation between LLM's self-reported confidence scores and actual factual accuracy scores.
    *   Visualizes this relationship with a scatter plot to assess the trustworthiness of LLM confidence.

6.  **Sentiment Bias Analysis (Sentiment Bias Analysis)**
    *   Analyzes sentiment discrepancy between LLM summaries and ground truth, categorized by industry or company.
    *   Visualizes mean sentiment scores to identify potential systemic biases in interpretation.

7.  **Entity Framing Analysis (Entity Framing Analysis)**
    *   Extracts key entities (ORG, PERSON, GPE) from both LLM summaries and ground truth.
    *   Compares entity mention counts and average contextual sentiment to detect framing biases.
    *   Includes an interactive plot for top entity mention counts.

8.  **Adversarial Testing (Adversarial Testing)**
    *   Allows users to craft and run custom "adversarial prompts" to intentionally provoke biases or hallucinations.
    *   Simulated LLM responses to adversarial inputs reveal vulnerabilities and failure modes.

9.  **Prompt Refinement (Prompt Refinement)**
    *   Enables users to input a refined prompt template.
    *   Retests the LLM with the new prompt on a selected document, comparing performance metrics (accuracy, sentiment discrepancy) against the baseline.

10. **Risk Aggregation & Recommendations (Aggregate Risks & Recommendations)**
    *   Consolidates all identified risk metrics into a holistic overview.
    *   Visualizes key risk incidents (hallucinations, contradictions, adversarial failures) and sentiment discrepancies.
    *   Generates tailored Human-in-the-Loop (HITL) and AI governance recommendations based on configurable thresholds.

11. **Final Risk Assessment Report (Final Risk Assessment Report)**
    *   Compiles all findings, analyses, and recommendations into a comprehensive, markdown-formatted risk assessment report for stakeholders.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-llm-risk-assessment.git
    cd quolab-llm-risk-assessment
    ```
    *(Replace `https://github.com/your-username/quolab-llm-risk-assessment.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
    **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.0.0
    pandas>=1.0.0
    numpy>=1.0.0
    matplotlib>=3.0.0
    seaborn>=0.11.0
    nltk>=3.6.0
    spacy>=3.0.0
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK and spaCy models:**
    The application will attempt to download these automatically, but you can also do it manually for robustness:
    ```bash
    python -c "import nltk; nltk.download('vader_lexicon')"
    python -m spacy download en_core_web_sm
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Navigate the Workbench:**
    *   Use the sidebar navigation to move through the different assessment pages.
    *   Start with the "Welcome & Setup" page to load data (either by uploading your own files or generating dummy data for a quick demonstration).
    *   Follow the workflow sequentially through the pages to conduct a full LLM risk assessment.

3.  **Interact with Features:**
    *   Click buttons to trigger analyses.
    *   Adjust sliders and input fields to configure thresholds and parameters.
    *   Review results presented in tables, metrics, and plots.
    *   Craft adversarial prompts and refined prompts to observe their impact.

## Project Structure

The project is organized into the following main directories and files:

```
quolab-llm-risk-assessment/
├── app.py                            # Main Streamlit application entry point
├── utils.py                          # Utility functions: data loading, LLM simulation, all assessment logic
├── application_pages/                # Directory containing individual Streamlit pages
│   ├── page_1_welcome_setup.py       # Page for data loading and initial setup
│   ├── page_2_baseline_summaries.py  # Page for generating and reviewing baseline LLM summaries
│   ├── page_3_factual_accuracy_assessment.py # Page for factual accuracy and hallucination detection
│   ├── page_4_contradiction_detection.py     # Page for detecting contradictions in summaries
│   ├── page_5_confidence_accuracy.py # Page for evaluating LLM confidence vs. factual accuracy
│   ├── page_6_sentiment_bias_analysis.py     # Page for analyzing sentiment bias
│   ├── page_7_entity_framing_analysis.py     # Page for analyzing entity representation and framing bias
│   ├── page_8_adversarial_testing.py # Page for running adversarial tests
│   ├── page_9_prompt_refinement.py   # Page for prompt engineering and retesting
│   ├── page_10_aggregate_risks_recommendations.py # Page for aggregating risks and generating recommendations
│   └── page_11_final_risk_assessment_report.py    # Page for generating the final report
├── data/                             # Directory for dummy data (created on first run if chosen)
├── uploaded_data/                    # Directory for temporarily storing uploaded user data
└── requirements.txt                  # List of Python dependencies
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib** & **Seaborn**: For data visualization and plotting.
*   **NLTK (Natural Language Toolkit)**: Specifically for VADER Sentiment Intensity Analyzer.
*   **spaCy**: For advanced NLP tasks, including entity extraction (`en_core_web_sm` model).
*   **`difflib`**: Python's built-in module for comparing sequences, used for fact comparison.
*   **`json`**, **`re`**, **`collections`**: Python's built-in modules for JSON handling, regular expressions, and specialized container datatypes.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-name`).
3.  Make your changes and ensure the code adheres to the existing style.
4.  Write clear and concise commit messages.
5.  Push your branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request describing your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You might need to create a `LICENSE` file in your repository if it doesn't exist yet, with the MIT license text.)*

## Contact

For any questions or inquiries, please reach out to:

*   **QuantUniversity (QuLab)**
*   **Website:** [https://www.quantuniversity.com/](https://www.quantuniversity.com/)
*   **Email:** info@quantuniversity.com
